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

    # frequency encoding for all TE_COLS
    for c in TE_COLS:
        if c in df.columns:
            freq = df[c].value_counts(normalize=True)
            df[f"{c}_freq"] = df[c].map(freq).fillna(0)

    return df


def add_te_fold_loo(df_tr, df_val, df_te, y_tr_log):
    """LOO Target Encoding on log target for train, regular smoothed TE for val/test."""
    df_tr = df_tr.copy()
    df_val = df_val.copy()
    df_te = df_te.copy()
    global_mean = y_tr_log.mean()
    k = 10

    for c in TE_COLS:
        if c not in df_tr.columns:
            continue

        agg = y_tr_log.groupby(df_tr[c]).agg(["mean", "std", "count", "sum"])
        agg["smooth"] = (agg["mean"] * agg["count"] + global_mean * k) / (agg["count"] + k)

        # LOO for train
        cat_sum = df_tr[c].map(agg["sum"])
        cat_count = df_tr[c].map(agg["count"])
        loo_mean = (cat_sum - y_tr_log.values) / (cat_count - 1).clip(lower=1)
        loo_smooth = (loo_mean * (cat_count - 1) + global_mean * k) / ((cat_count - 1) + k)
        df_tr[f"{c}_te"] = loo_smooth.fillna(global_mean)
        df_tr[f"{c}_te_std"] = df_tr[c].map(agg["std"]).fillna(0)
        df_tr[f"{c}_te_count"] = df_tr[c].map(np.log1p(agg["count"])).fillna(0)

        # regular TE for val and test
        for dset in [df_val, df_te]:
            dset[f"{c}_te"] = dset[c].map(agg["smooth"]).fillna(global_mean)
            dset[f"{c}_te_std"] = dset[c].map(agg["std"]).fillna(0)
            dset[f"{c}_te_count"] = dset[c].map(np.log1p(agg["count"])).fillna(0)

    return df_tr, df_val, df_te


def prepare_fold_data(train_base, test_base, y_log, splits):
    fold_data = []
    feat_cols_final = None

    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        df_tr = train_base.iloc[tr_idx].reset_index(drop=True)
        df_val = train_base.iloc[val_idx].reset_index(drop=True)
        df_te = test_base.copy()

        y_tr_log = pd.Series(y_log[tr_idx])
        df_tr, df_val, df_te = add_te_fold_loo(df_tr, df_val, df_te, y_tr_log)

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
            "y_tr_log": y_log[tr_idx],
            "y_val_log": y_log[val_idx],
            "val_idx": val_idx,
            "tr_idx": tr_idx,
        })

    return fold_data, feat_cols_final


def mape_oof(y_true_log, oof_log):
    return mean_absolute_percentage_error(np.expm1(y_true_log), np.expm1(oof_log))


def train_lgb(fold_data, seed, n_folds, n_train, n_test, params_override=None):
    """LightGBM on log target."""
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
    if params_override:
        params.update(params_override)

    oof = np.zeros(n_train)
    test_pred = np.zeros(n_test)

    for fold, fd in enumerate(fold_data):
        model = lgb.LGBMRegressor(**params)
        model.fit(
            fd["X_tr"], fd["y_tr_log"],
            eval_set=[(fd["X_val"], fd["y_val_log"])],
            callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(5000)],
        )
        oof[fd["val_idx"]] = model.predict(fd["X_val"])
        test_pred += model.predict(fd["X_te"]) / n_folds
        fold_m = mape_oof(fd["y_val_log"], oof[fd["val_idx"]])
        print(f"    Fold {fold+1}: MAPE={fold_m:.4f}, best_iter={model.best_iteration_}")

    return oof, test_pred


def train_lgb_dart(fold_data, seed, n_folds, n_train, n_test):
    """LightGBM DART on log target."""
    params = {
        "objective": "regression",
        "metric": "mape",
        "boosting_type": "dart",
        "n_estimators": 3000,
        "learning_rate": 0.05,
        "num_leaves": 255,
        "min_child_samples": 20,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "reg_alpha": 0.05,
        "reg_lambda": 0.05,
        "drop_rate": 0.1,
        "skip_drop": 0.5,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }

    oof = np.zeros(n_train)
    test_pred = np.zeros(n_test)

    for fold, fd in enumerate(fold_data):
        model = lgb.LGBMRegressor(**params)
        # DART doesn't support early stopping well, train fixed iterations
        model.fit(fd["X_tr"], fd["y_tr_log"])
        oof[fd["val_idx"]] = model.predict(fd["X_val"])
        test_pred += model.predict(fd["X_te"]) / n_folds
        fold_m = mape_oof(fd["y_val_log"], oof[fd["val_idx"]])
        print(f"    Fold {fold+1}: MAPE={fold_m:.4f}")

    return oof, test_pred


def train_catboost(fold_data, train_base, test_base, y_log, splits, seed, n_folds, n_train, n_test):
    """CatBoost on log target with native categoricals."""
    cat_features_names = ["district_cat", "corpus_cat", "developer_cat",
                          "hc_name_cat", "class_cat", "stage_cat", "rooms_4"]

    params = {
        "iterations": 10000,
        "learning_rate": 0.03,
        "depth": 8,
        "l2_leaf_reg": 3,
        "loss_function": "RMSE",
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

        y_tr_log = pd.Series(y_log[tr_idx])
        df_tr, df_val, df_te = add_te_fold_loo(df_tr, df_val, df_te, y_tr_log)

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

        pool_tr = cb.Pool(df_tr[feat_cols], label=y_log[tr_idx], cat_features=cb_cat_idx)
        pool_val = cb.Pool(df_val[feat_cols], label=y_log[val_idx], cat_features=cb_cat_idx)
        pool_te = cb.Pool(df_te[feat_cols], cat_features=cb_cat_idx)

        model = cb.CatBoostRegressor(**params)
        model.fit(pool_tr, eval_set=pool_val, use_best_model=True)

        oof[val_idx] = model.predict(pool_val)
        test_pred += model.predict(pool_te) / n_folds
        fold_m = mape_oof(y_log[val_idx], oof[val_idx])
        print(f"    [CB seed={seed}] Fold {fold_idx+1}: MAPE={fold_m:.4f}")

    return oof, test_pred


def train_xgb(fold_data, seed, n_folds, n_train, n_test):
    """XGBoost on log target."""
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
        oof[fd["val_idx"]] = model.predict(fd["X_val"])
        test_pred += model.predict(fd["X_te"]) / n_folds
        fold_m = mape_oof(fd["y_val_log"], oof[fd["val_idx"]])
        print(f"    [XGB seed={seed}] Fold {fold+1}: MAPE={fold_m:.4f}, best_iter={model.best_iteration}")

    return oof, test_pred


# ============================================================
# Main pipeline
# ============================================================


print("TRAIN V7: Multi-model multi-seed ensemble (log target)")


train_base = base_features(train_raw)
test_base = base_features(test_raw)

TARGET = "price_target"
y_log = np.log1p(train_base[TARGET].values)

n_train = len(train_base)
n_test = len(test_base)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=PRIMARY_SEED)
splits = list(kf.split(train_base))

fold_data, feat_cols = prepare_fold_data(train_base, test_base, y_log, splits)
print(f"Features: {len(feat_cols)}")
print(f"Train: {n_train}, Test: {n_test}, Folds: {N_FOLDS}")

print("MODEL 1: LightGBM GBDT (multi-seed)")

oof_lgb_all = []
test_lgb_all = []
for seed in SEEDS:
    print(f"  Seed {seed}:")
    oof, test_pred = train_lgb(fold_data, seed, N_FOLDS, n_train, n_test)
    m = mape_oof(y_log, oof)
    print(f"  Seed {seed} OOF MAPE: {m:.4f}")
    oof_lgb_all.append(oof)
    test_lgb_all.append(test_pred)

oof_lgb = np.mean(oof_lgb_all, axis=0)
test_lgb = np.mean(test_lgb_all, axis=0)
print(f"[LGB] Multi-seed OOF MAPE: {mape_oof(y_log, oof_lgb):.4f}")


print("MODEL 2: LightGBM low-lr (multi-seed)")

oof_lgb2_all = []
test_lgb2_all = []
for seed in SEEDS:
    print(f"  Seed {seed}:")
    oof, test_pred = train_lgb(fold_data, seed, N_FOLDS, n_train, n_test,
                                params_override={"learning_rate": 0.008, "n_estimators": 25000,
                                                  "num_leaves": 200, "min_child_samples": 30,
                                                  "reg_alpha": 0.1, "reg_lambda": 0.1})
    m = mape_oof(y_log, oof)
    print(f"  Seed {seed} OOF MAPE: {m:.4f}")
    oof_lgb2_all.append(oof)
    test_lgb2_all.append(test_pred)

oof_lgb2 = np.mean(oof_lgb2_all, axis=0)
test_lgb2 = np.mean(test_lgb2_all, axis=0)
print(f"[LGB-lowlr] Multi-seed OOF MAPE: {mape_oof(y_log, oof_lgb2):.4f}")

print("MODEL 3: CatBoost RMSE on log target (multi-seed)")

oof_cb_all = []
test_cb_all = []
for seed in SEEDS:
    print(f"  Seed {seed}:")
    oof, test_pred = train_catboost(
        fold_data, train_base, test_base, y_log, splits, seed, N_FOLDS, n_train, n_test
    )
    m = mape_oof(y_log, oof)
    print(f"  Seed {seed} OOF MAPE: {m:.4f}")
    oof_cb_all.append(oof)
    test_cb_all.append(test_pred)

oof_cb = np.mean(oof_cb_all, axis=0)
test_cb = np.mean(test_cb_all, axis=0)
print(f"[CB] Multi-seed OOF MAPE: {mape_oof(y_log, oof_cb):.4f}")

print("MODEL 4: XGBoost (multi-seed)")

oof_xgb_all = []
test_xgb_all = []
for seed in SEEDS:
    print(f"  Seed {seed}:")
    oof, test_pred = train_xgb(fold_data, seed, N_FOLDS, n_train, n_test)
    m = mape_oof(y_log, oof)
    print(f"  Seed {seed} OOF MAPE: {m:.4f}")
    oof_xgb_all.append(oof)
    test_xgb_all.append(test_pred)

oof_xgb = np.mean(oof_xgb_all, axis=0)
test_xgb = np.mean(test_xgb_all, axis=0)
print(f"[XGB] Multi-seed OOF MAPE: {mape_oof(y_log, oof_xgb):.4f}")


print("MODEL 5: LightGBM DART (single seed)")

oof_dart, test_dart = train_lgb_dart(fold_data, PRIMARY_SEED, N_FOLDS, n_train, n_test)
print(f"[DART] OOF MAPE: {mape_oof(y_log, oof_dart):.4f}")

print("BLEND OPTIMIZATION")

all_oof = np.column_stack([oof_lgb, oof_lgb2, oof_cb, oof_xgb, oof_dart])
all_test = np.column_stack([test_lgb, test_lgb2, test_cb, test_xgb, test_dart])
model_names = ["LGB", "LGB-lowlr", "CB", "XGB", "DART"]

y_real = np.expm1(y_log)


def blend_mape(weights, oof_matrix):
    blend = sum(w * oof_matrix[:, i] for i, w in enumerate(weights))
    return mean_absolute_percentage_error(y_real, np.expm1(blend))


def optuna_objective(trial):
    raw_w = [trial.suggest_float(f"w_{i}", 0.0, 1.0) for i in range(len(model_names))]
    total = sum(raw_w)
    weights = [w / total for w in raw_w]
    return blend_mape(weights, all_oof)


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(optuna_objective, n_trials=3000)

best = study.best_params
raw_w = [best[f"w_{i}"] for i in range(len(model_names))]
total = sum(raw_w)
optuna_weights = [w / total for w in raw_w]

print(f"\nOptuna best MAPE: {study.best_value:.6f}")
for name, w in zip(model_names, optuna_weights):
    print(f"  {name}: {w:.4f}")

final_weights = optuna_weights
final_mape = study.best_value

blend_oof_log = sum(w * all_oof[:, i] for i, w in enumerate(final_weights))
blend_test_log = sum(w * all_test[:, i] for i, w in enumerate(final_weights))

print(f"\nFinal blend OOF MAPE: {final_mape:.6f}")

print("SAVING SUBMISSIONS")

pd.DataFrame({"id": test_ids, "price_target": np.expm1(test_lgb)}).to_csv(
    "submissions/lgbm_v7.csv", index=False)
pd.DataFrame({"id": test_ids, "price_target": np.expm1(test_cb)}).to_csv(
    "submissions/cb_v7.csv", index=False)
pd.DataFrame({"id": test_ids, "price_target": np.expm1(test_xgb)}).to_csv(
    "submissions/xgb_v7.csv", index=False)
pd.DataFrame({"id": test_ids, "price_target": np.expm1(blend_test_log)}).to_csv(
    "submissions/blend_v7.csv", index=False)

with open("models/v7_results.pkl", "wb") as f:
    pickle.dump({
        "oof_lgb": oof_lgb, "oof_lgb2": oof_lgb2,
        "oof_cb": oof_cb, "oof_xgb": oof_xgb, "oof_dart": oof_dart,
        "test_lgb": test_lgb, "test_lgb2": test_lgb2,
        "test_cb": test_cb, "test_xgb": test_xgb, "test_dart": test_dart,
        "blend_test_log": blend_test_log,
        "final_weights": final_weights,
        "final_mape": final_mape,
        "y_log": y_log,
        "feat_cols": feat_cols,
        "model_names": model_names,
    }, f)

results = [
    ("LGB GBDT (multi-seed)", mape_oof(y_log, oof_lgb)),
    ("LGB low-lr (multi-seed)", mape_oof(y_log, oof_lgb2)),
    ("CatBoost (multi-seed)", mape_oof(y_log, oof_cb)),
    ("XGBoost (multi-seed)", mape_oof(y_log, oof_xgb)),
    ("LGB DART", mape_oof(y_log, oof_dart)),
    ("Final blend", final_mape),
]
for name, mape_val in sorted(results, key=lambda x: x[1]):
    print(f"  {mape_val:.6f}  {name}")

print(f"\nPrevious best: 0.0199 (Blend LGB80+CB20 v2)")
print(f"New best:      {final_mape:.6f}")
print(f"Improvement:   {0.0199 - final_mape:.6f}")
print(f"\nSaved: submissions/blend_v7.csv")