import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import optuna
import pickle
import warnings
warnings.filterwarnings("ignore")

SEEDS = [42, 123, 777]
N_FOLDS = 7
PRIMARY_SEED = 42

train_raw = pd.read_csv("period_1_train_data.csv")
test_raw = pd.read_csv("test_x.csv")
test_ids = test_raw["id"].values

TE_COLS = ["district_cat", "corpus_cat", "developer_cat", "hc_name_cat",
           "class_cat", "stage_cat", "interior_cat", "region_name_cat"]

LOC_DIST_COLS = [
    "location_hotel_w_mean_distance", "location_pop_bank_w_mean_distance",
    "location_office_w_mean_distance", "location_barrier_w_mean_distance",
    "location_amenity_bank_w_mean_distance", "location_fuel_w_mean_distance",
    "location_shop_clothes_w_mean_distance", "location_commercial_w_mean_distance",
    "location_industrial_w_mean_distance", "location_town_w_mean_distance",
    "location_residential_w_mean_distance", "location_amenity_restaurant_w_mean_distance",
    "location_railway_station_w_mean_distance", "location_social_w_mean_distance",
    "location_shop_other_w_mean_distance", "location_school_w_mean_distance",
    "location_marketplace_w_mean_distance", "location_amenity_leisure_w_mean_distance",
    "location_tourism_w_mean_distance", "location_shop_alco_w_mean_distance",
    "location_shop_product_w_mean_distance", "location_railway_w_mean_distance",
    "location_highway_crossing_w_mean_distance",
    "location_pop_shop_w_mean_distance", "location_amenity_pharmacy_w_mean_distance",
    "location_public_transport_stop_position_w_mean_distance",
    "location_public_transport_platform_w_mean_distance",
    "location_water_w_mean_distance", "location_university_w_mean_distance",
    "location_leisure_w_mean_distance",
]


def base_features(df):
    df = df.copy()
    df["rooms_4"] = df["rooms_4"].replace("студия", "0").replace(">=4", "4")
    df["rooms_4"] = pd.to_numeric(df["rooms_4"], errors="coerce")

    loc_cols = [c for c in df.columns if c.startswith("location_")]
    for c in loc_cols:
        if df[c].dtype in [np.float64, np.int64]:
            df[c] = df[c].replace(-999.0, np.nan)

    df["interior_cat"] = df["interior_cat"].replace(0.0, np.nan)

    df["agreement_date"] = pd.to_datetime(df["agreement_date"])
    df["year"] = df["agreement_date"].dt.year
    df["month"] = df["agreement_date"].dt.month
    df["quarter"] = df["agreement_date"].dt.quarter
    df["dayofweek"] = df["agreement_date"].dt.dayofweek
    df["days_since_start"] = (df["agreement_date"] - pd.Timestamp("2012-01-01")).dt.days
    df["week_of_year"] = df["agreement_date"].dt.isocalendar().week.astype(int)
    df.drop(columns=["agreement_date"], inplace=True)

    max_lvl = df["location_max_levels_max"].replace(0, np.nan)
    df["floor_ratio"] = df["floor"] / max_lvl
    df["floor_from_top"] = max_lvl - df["floor"]
    df["is_top_floor"] = (df["floor"] == df["location_max_levels_max"]).astype(int)
    df["is_ground_floor"] = (df["floor"] == 1).astype(int)
    df["log_square"] = np.log1p(df["square"])
    df["square_per_room"] = df["square"] / (df["rooms_4"].replace(0, np.nan))
    df["floor_x_square"] = df["floor"] * df["square"]
    df["floor_ratio_x_square"] = df["floor_ratio"] * df["square"]

    df["log_buildings_cnt"] = np.log1p(df["location_buildings_cnt"])
    df["log_pop_transport"] = np.log1p(df["location_public_transport_stop_position_cnt"])
    df["transport_density"] = (
        df["location_public_transport_stop_position_cnt"] +
        df["location_public_transport_platform_cnt"] +
        df["location_bus_station_cnt"]
    )
    df["commercial_density"] = (
        df["location_commercial_cnt"] +
        df["location_pop_cafe_cnt"] +
        df["location_pop_bank_cnt"]
    )

    loc_cols = [c for c in df.columns if c.startswith("location_")]
    df["all_loc_nan"] = df[loc_cols].isnull().all(axis=1).astype(int)
    df["loc_miss_count"] = df[loc_cols].isnull().sum(axis=1)

    high_miss = [
        "location_town_w_mean_distance", "location_university_w_mean_distance",
        "location_commercial_w_mean_distance", "location_amenity_leisure_w_mean_distance",
        "location_water_w_mean_distance", "location_hotel_w_mean_distance",
        "location_industrial_w_mean_distance", "location_railway_station_w_mean_distance",
        "location_social_w_mean_distance", "location_shop_alco_w_mean_distance",
        "location_tourism_w_mean_distance", "location_pop_bank_w_mean_distance",
    ]
    for c in high_miss:
        if c in df.columns:
            df[f"{c}_miss"] = df[c].isnull().astype(int)

    dist_cols_present = [c for c in LOC_DIST_COLS if c in df.columns]
    df["min_dist_amenity"] = df[dist_cols_present].min(axis=1)
    df["mean_dist_amenity"] = df[dist_cols_present].mean(axis=1)

    # frequency encoding
    for c in TE_COLS:
        if c in df.columns:
            freq = df[c].value_counts(normalize=True)
            df[f"{c}_freq"] = df[c].map(freq).fillna(0)

    return df


def add_te_fold(df_tr, df_val, df_te, y_tr_series, target_col_name="te_target"):
    """LOO Target Encoding for train, regular TE for val/test.
    y_tr_series: pd.Series aligned with df_tr index, contains the target values to encode against.
    """
    df_tr = df_tr.copy()
    df_val = df_val.copy()
    df_te = df_te.copy()
    global_mean = y_tr_series.mean()
    k = 10

    for c in TE_COLS:
        if c not in df_tr.columns:
            continue
        cat_col = df_tr[c].values
        y_vals = y_tr_series.values

        # group stats
        agg = y_tr_series.groupby(df_tr[c]).agg(["mean", "std", "count", "sum"])
        agg["smooth"] = (agg["mean"] * agg["count"] + global_mean * k) / (agg["count"] + k)

        # LOO for train: for each row, subtract its own value from group stats
        cat_sum = df_tr[c].map(agg["sum"])
        cat_count = df_tr[c].map(agg["count"])
        loo_mean = (cat_sum - y_vals) / (cat_count - 1).clip(lower=1)
        loo_smooth = (loo_mean * (cat_count - 1) + global_mean * k) / ((cat_count - 1) + k)
        df_tr[f"{c}_te"] = loo_smooth.fillna(global_mean)
        df_tr[f"{c}_te_std"] = df_tr[c].map(agg["std"]).fillna(0)
        df_tr[f"{c}_te_count"] = df_tr[c].map(agg["count"]).fillna(0)

        # regular TE for val and test
        for dset in [df_val, df_te]:
            dset[f"{c}_te"] = dset[c].map(agg["smooth"]).fillna(global_mean)
            dset[f"{c}_te_std"] = dset[c].map(agg["std"]).fillna(0)
            dset[f"{c}_te_count"] = dset[c].map(agg["count"]).fillna(0)

    return df_tr, df_val, df_te


def prepare_fold_data(train_base, test_base, y_raw, y_log, splits):
    """Prepare fold data with TE for both raw and log targets."""
    fold_data = []
    feat_cols_final = None

    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        df_tr = train_base.iloc[tr_idx].reset_index(drop=True)
        df_val = train_base.iloc[val_idx].reset_index(drop=True)
        df_te = test_base.copy()

        # TE on raw target (for MAPE-objective models)
        y_tr_raw = pd.Series(y_raw[tr_idx])
        df_tr, df_val, df_te = add_te_fold(df_tr, df_val, df_te, y_tr_raw)

        # TE on log target (for regression-objective models) — add as separate features
        y_tr_log = pd.Series(y_log[tr_idx])
        global_mean_log = y_tr_log.mean()
        k = 10
        for c in TE_COLS:
            if c not in df_tr.columns:
                continue
            agg_log = y_tr_log.groupby(df_tr[c]).agg(["mean", "std", "count", "sum"])
            agg_log["smooth"] = (agg_log["mean"] * agg_log["count"] + global_mean_log * k) / (agg_log["count"] + k)

            cat_sum_log = df_tr[c].map(agg_log["sum"])
            cat_count_log = df_tr[c].map(agg_log["count"])
            loo_mean_log = (cat_sum_log - y_tr_log.values) / (cat_count_log - 1).clip(lower=1)
            loo_smooth_log = (loo_mean_log * (cat_count_log - 1) + global_mean_log * k) / ((cat_count_log - 1) + k)
            df_tr[f"{c}_te_log"] = loo_smooth_log.fillna(global_mean_log)

            for dset in [df_val, df_te]:
                dset[f"{c}_te_log"] = dset[c].map(agg_log["smooth"]).fillna(global_mean_log)

        drop_cols = ["price_target"] + (["id"] if "id" in df_tr.columns else [])
        feat_cols = [c for c in df_tr.columns if c not in drop_cols]

        cat_obj = df_tr[feat_cols].select_dtypes(include=["object", "category"]).columns.tolist()
        for c in cat_obj:
            all_vals = pd.concat([df_tr[c], df_val[c], df_te[c]]).astype("category")
            codes = all_vals.cat.codes
            n_tr = len(df_tr)
            n_val = len(df_val)
            df_tr[c] = codes.values[:n_tr]
            df_val[c] = codes.values[n_tr:n_tr + n_val]
            df_te[c] = codes.values[n_tr + n_val:]

        if feat_cols_final is None:
            feat_cols_final = feat_cols

        fold_data.append({
            "X_tr": df_tr[feat_cols].values.astype(np.float32),
            "X_val": df_val[feat_cols].values.astype(np.float32),
            "X_te": df_te[feat_cols].values.astype(np.float32),
            "y_tr_raw": y_raw[tr_idx],
            "y_val_raw": y_raw[val_idx],
            "y_tr_log": y_log[tr_idx],
            "y_val_log": y_log[val_idx],
            "val_idx": val_idx,
            "tr_idx": tr_idx,
        })

    return fold_data, feat_cols_final


def train_lgb_mape_raw(fold_data, seed, n_folds, n_train, n_test):
    """LightGBM with objective=mape on raw target."""
    params = {
        "objective": "mape",
        "metric": "mape",
        "n_estimators": 15000,
        "learning_rate": 0.02,
        "num_leaves": 255,
        "min_child_samples": 30,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }
    oof = np.zeros(n_train)
    test_pred = np.zeros(n_test)

    for fold, fd in enumerate(fold_data):
        model = lgb.LGBMRegressor(**params)
        model.fit(
            fd["X_tr"], fd["y_tr_raw"],
            eval_set=[(fd["X_val"], fd["y_val_raw"])],
            callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(5000)],
        )
        oof[fd["val_idx"]] = model.predict(fd["X_val"])
        test_pred += model.predict(fd["X_te"]) / n_folds
        fold_mape = mean_absolute_percentage_error(fd["y_val_raw"], oof[fd["val_idx"]])
        print(f"  [LGB-MAPE seed={seed}] Fold {fold+1}: MAPE={fold_mape:.4f}, best_iter={model.best_iteration_}")

    oof = np.clip(oof, 1.0, None)
    test_pred = np.clip(test_pred, 1.0, None)
    return oof, test_pred


def train_lgb_log(fold_data, seed, n_folds, n_train, n_test):
    """LightGBM with objective=regression on log target (proven approach)."""
    params = {
        "objective": "regression",
        "metric": "mape",
        "n_estimators": 15000,
        "learning_rate": 0.02,
        "num_leaves": 255,
        "min_child_samples": 20,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "reg_alpha": 0.05,
        "reg_lambda": 0.05,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }
    oof = np.zeros(n_train)
    test_pred = np.zeros(n_test)

    for fold, fd in enumerate(fold_data):
        model = lgb.LGBMRegressor(**params)
        model.fit(
            fd["X_tr"], fd["y_tr_log"],
            eval_set=[(fd["X_val"], fd["y_val_log"])],
            callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(5000)],
        )
        oof[fd["val_idx"]] = np.expm1(model.predict(fd["X_val"]))
        test_pred += np.expm1(model.predict(fd["X_te"])) / n_folds
        fold_mape = mean_absolute_percentage_error(fd["y_val_raw"], oof[fd["val_idx"]])
        print(f"  [LGB-LOG seed={seed}] Fold {fold+1}: MAPE={fold_mape:.4f}, best_iter={model.best_iteration_}")

    oof = np.clip(oof, 1.0, None)
    test_pred = np.clip(test_pred, 1.0, None)
    return oof, test_pred


def train_catboost_mape_raw(fold_data, train_base, test_base, y_raw, y_log, splits, seed, n_folds, n_train, n_test):
    """CatBoost with MAPE on raw target, native categoricals."""
    cat_features_names = ["district_cat", "corpus_cat", "developer_cat",
                          "hc_name_cat", "class_cat", "stage_cat", "rooms_4", "region_name_cat"]

    params = {
        "iterations": 10000,
        "learning_rate": 0.03,
        "depth": 8,
        "l2_leaf_reg": 3,
        "loss_function": "MAPE",
        "eval_metric": "MAPE",
        "random_seed": seed,
        "od_type": "Iter",
        "od_wait": 300,
        "verbose": 2000,
        "thread_count": -1,
    }

    oof = np.zeros(n_train)
    test_pred = np.zeros(n_test)

    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        df_tr = train_base.iloc[tr_idx].reset_index(drop=True)
        df_val = train_base.iloc[val_idx].reset_index(drop=True)
        df_te = test_base.copy()

        # TE on raw target with LOO
        y_tr_raw = pd.Series(y_raw[tr_idx])
        df_tr, df_val, df_te = add_te_fold(df_tr, df_val, df_te, y_tr_raw)

        drop_cols = ["price_target"] + (["id"] if "id" in df_tr.columns else [])
        feat_cols = [c for c in df_tr.columns if c not in drop_cols]

        for c in cat_features_names:
            if c in df_tr.columns:
                df_tr[c] = df_tr[c].fillna(-1).astype(str)
                df_val[c] = df_val[c].fillna(-1).astype(str)
                df_te[c] = df_te[c].fillna(-1).astype(str)

        obj_cols = df_tr[feat_cols].select_dtypes(include=["object"]).columns.tolist()
        for c in obj_cols:
            if c not in cat_features_names:
                all_vals = pd.concat([df_tr[c], df_val[c], df_te[c]]).astype("category")
                codes = all_vals.cat.codes
                n_tr = len(df_tr)
                n_val = len(df_val)
                df_tr[c] = codes.values[:n_tr]
                df_val[c] = codes.values[n_tr:n_tr + n_val]
                df_te[c] = codes.values[n_tr + n_val:]

        cb_cat_idx = [feat_cols.index(c) for c in cat_features_names if c in feat_cols]

        pool_tr = cb.Pool(df_tr[feat_cols], label=y_raw[tr_idx], cat_features=cb_cat_idx)
        pool_val = cb.Pool(df_val[feat_cols], label=y_raw[val_idx], cat_features=cb_cat_idx)
        pool_te = cb.Pool(df_te[feat_cols], cat_features=cb_cat_idx)

        model = cb.CatBoostRegressor(**params)
        model.fit(pool_tr, eval_set=pool_val, use_best_model=True)

        oof[val_idx] = model.predict(pool_val)
        test_pred += model.predict(pool_te) / n_folds
        fold_mape = mean_absolute_percentage_error(y_raw[val_idx], oof[val_idx])
        print(f"  [CB-MAPE seed={seed}] Fold {fold_idx+1}: MAPE={fold_mape:.4f}")

    oof = np.clip(oof, 1.0, None)
    test_pred = np.clip(test_pred, 1.0, None)
    return oof, test_pred


def train_xgb_mape(fold_data, seed, n_folds, n_train, n_test):
    """XGBoost with custom MAPE-like objective on log target."""
    params = {
        "objective": "reg:squarederror",
        "n_estimators": 15000,
        "learning_rate": 0.02,
        "max_depth": 8,
        "min_child_weight": 10,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.05,
        "reg_lambda": 0.1,
        "random_state": seed,
        "n_jobs": -1,
        "tree_method": "hist",
        "verbosity": 0,
        "early_stopping_rounds": 300,
    }

    oof = np.zeros(n_train)
    test_pred = np.zeros(n_test)

    for fold, fd in enumerate(fold_data):
        model = xgb.XGBRegressor(**params)
        model.fit(
            fd["X_tr"], fd["y_tr_log"],
            eval_set=[(fd["X_val"], fd["y_val_log"])],
            verbose=5000,
        )
        oof[fd["val_idx"]] = np.expm1(model.predict(fd["X_val"]))
        test_pred += np.expm1(model.predict(fd["X_te"])) / n_folds
        fold_mape = mean_absolute_percentage_error(fd["y_val_raw"], oof[fd["val_idx"]])
        print(f"  [XGB seed={seed}] Fold {fold+1}: MAPE={fold_mape:.4f}, best_iter={model.best_iteration}")

    oof = np.clip(oof, 1.0, None)
    test_pred = np.clip(test_pred, 1.0, None)
    return oof, test_pred


print("TRAIN V6: Multi-model multi-seed ensemble")

train_base = base_features(train_raw)
test_base = base_features(test_raw)

TARGET = "price_target"
y_raw = train_base[TARGET].values.astype(np.float64)
y_log = np.log1p(y_raw)

n_train = len(train_base)
n_test = len(test_base)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=PRIMARY_SEED)
splits = list(kf.split(train_base))

fold_data, feat_cols = prepare_fold_data(train_base, test_base, y_raw, y_log, splits)
print(f"Features: {len(feat_cols)}")
print(f"Train: {n_train}, Test: {n_test}, Folds: {N_FOLDS}")

# ---- Model 1: LightGBM MAPE on raw target (multi-seed) ----
print("\n" + "=" * 60)
print("MODEL 1: LightGBM MAPE raw target")
print("=" * 60)
oof_lgb_mape_all = []
test_lgb_mape_all = []
for seed in SEEDS:
    oof, test_pred = train_lgb_mape_raw(fold_data, seed, N_FOLDS, n_train, n_test)
    mape = mean_absolute_percentage_error(y_raw, oof)
    print(f"  Seed {seed} OOF MAPE: {mape:.4f}")
    oof_lgb_mape_all.append(oof)
    test_lgb_mape_all.append(test_pred)

oof_lgb_mape = np.mean(oof_lgb_mape_all, axis=0)
test_lgb_mape = np.mean(test_lgb_mape_all, axis=0)
print(f"[LGB-MAPE] Multi-seed OOF MAPE: {mean_absolute_percentage_error(y_raw, oof_lgb_mape):.4f}")

# ---- Model 2: LightGBM regression on log target (multi-seed) ----
print("\n" + "=" * 60)
print("MODEL 2: LightGBM regression log target")
print("=" * 60)
oof_lgb_log_all = []
test_lgb_log_all = []
for seed in SEEDS:
    oof, test_pred = train_lgb_log(fold_data, seed, N_FOLDS, n_train, n_test)
    mape = mean_absolute_percentage_error(y_raw, oof)
    print(f"  Seed {seed} OOF MAPE: {mape:.4f}")
    oof_lgb_log_all.append(oof)
    test_lgb_log_all.append(test_pred)

oof_lgb_log = np.mean(oof_lgb_log_all, axis=0)
test_lgb_log = np.mean(test_lgb_log_all, axis=0)
print(f"[LGB-LOG] Multi-seed OOF MAPE: {mean_absolute_percentage_error(y_raw, oof_lgb_log):.4f}")

# ---- Model 3: CatBoost MAPE on raw target ----
print("\n" + "=" * 60)
print("MODEL 3: CatBoost MAPE raw target")
print("=" * 60)
oof_cb_all = []
test_cb_all = []
for seed in SEEDS:
    oof, test_pred = train_catboost_mape_raw(
        fold_data, train_base, test_base, y_raw, y_log, splits, seed, N_FOLDS, n_train, n_test
    )
    mape = mean_absolute_percentage_error(y_raw, oof)
    print(f"  Seed {seed} OOF MAPE: {mape:.4f}")
    oof_cb_all.append(oof)
    test_cb_all.append(test_pred)

oof_cb = np.mean(oof_cb_all, axis=0)
test_cb = np.mean(test_cb_all, axis=0)
print(f"[CB-MAPE] Multi-seed OOF MAPE: {mean_absolute_percentage_error(y_raw, oof_cb):.4f}")

# ---- Model 4: XGBoost on log target ----
print("\n" + "=" * 60)
print("MODEL 4: XGBoost log target")
print("=" * 60)
oof_xgb_all = []
test_xgb_all = []
for seed in SEEDS:
    oof, test_pred = train_xgb_mape(fold_data, seed, N_FOLDS, n_train, n_test)
    mape = mean_absolute_percentage_error(y_raw, oof)
    print(f"  Seed {seed} OOF MAPE: {mape:.4f}")
    oof_xgb_all.append(oof)
    test_xgb_all.append(test_pred)

oof_xgb = np.mean(oof_xgb_all, axis=0)
test_xgb = np.mean(test_xgb_all, axis=0)
print(f"[XGB] Multi-seed OOF MAPE: {mean_absolute_percentage_error(y_raw, oof_xgb):.4f}")

# ---- Blend optimization via Optuna ----
print("\n" + "=" * 60)
print("BLEND OPTIMIZATION (Optuna)")
print("=" * 60)

all_oof = np.column_stack([oof_lgb_mape, oof_lgb_log, oof_cb, oof_xgb])
all_test = np.column_stack([test_lgb_mape, test_lgb_log, test_cb, test_xgb])
model_names = ["LGB-MAPE", "LGB-LOG", "CB-MAPE", "XGB"]


def optuna_objective(trial):
    w0 = trial.suggest_float("w_lgb_mape", 0.0, 1.0)
    w1 = trial.suggest_float("w_lgb_log", 0.0, 1.0)
    w2 = trial.suggest_float("w_cb", 0.0, 1.0)
    w3 = trial.suggest_float("w_xgb", 0.0, 1.0)
    total = w0 + w1 + w2 + w3
    w0, w1, w2, w3 = w0 / total, w1 / total, w2 / total, w3 / total
    blend = w0 * all_oof[:, 0] + w1 * all_oof[:, 1] + w2 * all_oof[:, 2] + w3 * all_oof[:, 3]
    return mean_absolute_percentage_error(y_raw, blend)


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(optuna_objective, n_trials=2000)

best = study.best_params
total = best["w_lgb_mape"] + best["w_lgb_log"] + best["w_cb"] + best["w_xgb"]
weights = [best["w_lgb_mape"] / total, best["w_lgb_log"] / total,
           best["w_cb"] / total, best["w_xgb"] / total]

print(f"\nOptuna best MAPE: {study.best_value:.6f}")
for name, w in zip(model_names, weights):
    print(f"  {name}: {w:.4f}")

blend_oof = sum(w * all_oof[:, i] for i, w in enumerate(weights))
blend_test = sum(w * all_test[:, i] for i, w in enumerate(weights))

blend_mape = mean_absolute_percentage_error(y_raw, blend_oof)
print(f"\nFinal blend OOF MAPE: {blend_mape:.6f}")

# Also try grid search for comparison
print("\n=== GRID SEARCH BLEND ===")
best_grid_mape = 1.0
best_grid_w = None
step = 0.05
for w0 in np.arange(0.0, 1.01, step):
    for w1 in np.arange(0.0, 1.01 - w0, step):
        for w2 in np.arange(0.0, 1.01 - w0 - w1, step):
            w3 = round(1.0 - w0 - w1 - w2, 10)
            if w3 < -1e-9:
                continue
            blend = w0 * all_oof[:, 0] + w1 * all_oof[:, 1] + w2 * all_oof[:, 2] + w3 * all_oof[:, 3]
            m = mean_absolute_percentage_error(y_raw, blend)
            if m < best_grid_mape:
                best_grid_mape = m
                best_grid_w = (w0, w1, w2, w3)

print(f"Grid best MAPE: {best_grid_mape:.6f}")
for name, w in zip(model_names, best_grid_w):
    print(f"  {name}: {w:.4f}")

# Use the better of optuna vs grid
if best_grid_mape < blend_mape:
    final_weights = best_grid_w
    final_mape = best_grid_mape
    print("\nUsing grid search weights (better)")
else:
    final_weights = weights
    final_mape = blend_mape
    print("\nUsing Optuna weights (better)")

final_oof = sum(w * all_oof[:, i] for i, w in enumerate(final_weights))
final_test = sum(w * all_test[:, i] for i, w in enumerate(final_weights))

# ---- Save submissions ----
print("\n" + "=" * 60)
print("SAVING SUBMISSIONS")
print("=" * 60)

pd.DataFrame({"id": test_ids, "price_target": test_lgb_mape}).to_csv(
    "submissions/lgbm_mape_v6.csv", index=False)
pd.DataFrame({"id": test_ids, "price_target": test_lgb_log}).to_csv(
    "submissions/lgbm_log_v6.csv", index=False)
pd.DataFrame({"id": test_ids, "price_target": test_cb}).to_csv(
    "submissions/cb_mape_v6.csv", index=False)
pd.DataFrame({"id": test_ids, "price_target": test_xgb}).to_csv(
    "submissions/xgb_v6.csv", index=False)
pd.DataFrame({"id": test_ids, "price_target": final_test}).to_csv(
    "submissions/blend_v6.csv", index=False)

# ---- Save results ----
with open("models/v6_results.pkl", "wb") as f:
    pickle.dump({
        "oof_lgb_mape": oof_lgb_mape,
        "oof_lgb_log": oof_lgb_log,
        "oof_cb": oof_cb,
        "oof_xgb": oof_xgb,
        "test_lgb_mape": test_lgb_mape,
        "test_lgb_log": test_lgb_log,
        "test_cb": test_cb,
        "test_xgb": test_xgb,
        "final_test": final_test,
        "final_weights": final_weights,
        "final_mape": final_mape,
        "y_raw": y_raw,
        "feat_cols": feat_cols,
        "model_names": model_names,
    }, f)

# ---- Summary ----
print("\n" + "=" * 60)
print("EXPERIMENT SUMMARY")
print("=" * 60)
results = [
    ("LGB-MAPE raw (multi-seed)", mean_absolute_percentage_error(y_raw, oof_lgb_mape)),
    ("LGB-LOG (multi-seed)", mean_absolute_percentage_error(y_raw, oof_lgb_log)),
    ("CB-MAPE raw (multi-seed)", mean_absolute_percentage_error(y_raw, oof_cb)),
    ("XGB-LOG (multi-seed)", mean_absolute_percentage_error(y_raw, oof_xgb)),
    ("Final blend", final_mape),
]
for name, mape in sorted(results, key=lambda x: x[1]):
    print(f"  {mape:.6f}  {name}")

print(f"\nPrevious best: 0.0199 (Blend LGB80+CB20 v2)")
print(f"New best:      {final_mape:.6f}")
print(f"Improvement:   {0.0199 - final_mape:.6f}")
print(f"\nSaved: submissions/blend_v6.csv")