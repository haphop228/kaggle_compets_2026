import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
import lightgbm as lgb
import catboost as cb
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
           "class_cat", "stage_cat"]

TE_INTERACTION_PAIRS = [
    ("corpus_cat", "district_cat"),
    ("hc_name_cat", "class_cat"),
    ("corpus_cat", "class_cat"),
    ("developer_cat", "district_cat"),
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
    df["square_x_floor_ratio"] = df["square"] * df["floor_ratio"]
    df["rooms_x_floor"] = df["rooms_4"] * df["floor"]

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

    for c in TE_COLS + ["region_name_cat", "interior_cat"]:
        if c in df.columns:
            freq = df[c].value_counts(normalize=True)
            df[f"{c}_freq"] = df[c].map(freq).fillna(0)

    df["region_name_cat"] = df["region_name_cat"].astype("category").cat.codes
    return df


def add_te_fold(df_tr, df_val, df_te, log_y_tr):
    df_tr = df_tr.copy()
    df_val = df_val.copy()
    df_te = df_te.copy()
    global_mean = log_y_tr.mean()
    global_median = log_y_tr.median()
    k = 10

    for c in TE_COLS:
        agg = log_y_tr.groupby(df_tr[c]).agg(["mean", "std", "count", "median"])
        agg["smooth_mean"] = (agg["mean"] * agg["count"] + global_mean * k) / (agg["count"] + k)
        agg["smooth_median"] = (agg["median"] * agg["count"] + global_median * k) / (agg["count"] + k)

        for dset in [df_tr, df_val, df_te]:
            dset[f"{c}_te"] = dset[c].map(agg["smooth_mean"]).fillna(global_mean)
            dset[f"{c}_te_std"] = dset[c].map(agg["std"]).fillna(0)
            dset[f"{c}_te_count"] = dset[c].map(np.log1p(agg["count"])).fillna(0)
            dset[f"{c}_te_median"] = dset[c].map(agg["smooth_median"]).fillna(global_median)

    for c1, c2 in TE_INTERACTION_PAIRS:
        key = f"{c1}_x_{c2}"
        grp_tr = df_tr[[c1, c2]].astype(str).agg("_".join, axis=1)
        grp_val = df_val[[c1, c2]].astype(str).agg("_".join, axis=1)
        grp_te = df_te[[c1, c2]].astype(str).agg("_".join, axis=1)

        agg = log_y_tr.groupby(grp_tr).agg(["mean", "count"])
        agg["smooth"] = (agg["mean"] * agg["count"] + global_mean * k) / (agg["count"] + k)

        df_tr[f"{key}_te"] = grp_tr.map(agg["smooth"]).fillna(global_mean)
        df_val[f"{key}_te"] = grp_val.map(agg["smooth"]).fillna(global_mean)
        df_te[f"{key}_te"] = grp_te.map(agg["smooth"]).fillna(global_mean)

    return df_tr, df_val, df_te


def mape_oof(y_log, oof_log):
    return mean_absolute_percentage_error(np.expm1(y_log), np.expm1(oof_log))


train_base = base_features(train_raw)
test_base = base_features(test_raw)

TARGET = "price_target"
y_log = np.log1p(train_base[TARGET].values)

n_train = len(train_base)
n_test = len(test_base)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=PRIMARY_SEED)
splits = list(kf.split(train_base))

fold_data = []
feat_cols_final = None

for tr_idx, val_idx in splits:
    df_tr = train_base.iloc[tr_idx].reset_index(drop=True)
    df_val = train_base.iloc[val_idx].reset_index(drop=True)
    df_te = test_base.copy()
    log_y_tr = pd.Series(y_log[tr_idx])

    df_tr, df_val, df_te = add_te_fold(df_tr, df_val, df_te, log_y_tr)

    drop_cols = [TARGET] + (["id"] if "id" in df_tr.columns else [])
    feat_cols = [c for c in df_tr.columns if c not in drop_cols]

    cat_obj = df_tr[feat_cols].select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cat_obj:
        df_tr[c] = df_tr[c].astype("category").cat.codes
        df_val[c] = df_val[c].astype("category").cat.codes
        df_te[c] = df_te[c].astype("category").cat.codes

    if feat_cols_final is None:
        feat_cols_final = feat_cols

    fold_data.append({
        "X_tr": df_tr[feat_cols].values.astype(np.float32),
        "X_val": df_val[feat_cols].values.astype(np.float32),
        "X_te": df_te[feat_cols].values.astype(np.float32),
        "y_tr": y_log[tr_idx],
        "y_val": y_log[val_idx],
        "val_idx": val_idx,
        "tr_idx": tr_idx,
    })


def optuna_lgb_objective(trial):
    params = {
        "objective": "regression",
        "metric": "mape",
        "n_estimators": 15000,
        "learning_rate": trial.suggest_float("lr", 0.005, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 63, 511),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 0.9),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 0.9),
        "bagging_freq": 1,
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "max_depth": trial.suggest_int("max_depth", -1, 15),
        "random_state": PRIMARY_SEED,
        "n_jobs": -1,
        "verbose": -1,
    }

    oof = np.zeros(n_train)
    for fd in fold_data:
        model = lgb.LGBMRegressor(**params)
        model.fit(
            fd["X_tr"], fd["y_tr"],
            eval_set=[(fd["X_val"], fd["y_val"])],
            callbacks=[lgb.early_stopping(500, verbose=False)],
        )
        oof[fd["val_idx"]] = model.predict(fd["X_val"])

    return mape_oof(y_log, oof)


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(optuna_lgb_objective, n_trials=200, show_progress_bar=True)

print(f"\nBest trial MAPE: {study.best_value:.6f}")
print(f"Best params: {study.best_params}")

best_lgb_params = {
    "objective": "regression",
    "metric": "mape",
    "n_estimators": 15000,
    "learning_rate": study.best_params["lr"],
    "num_leaves": study.best_params["num_leaves"],
    "min_child_samples": study.best_params["min_child_samples"],
    "feature_fraction": study.best_params["feature_fraction"],
    "bagging_fraction": study.best_params["bagging_fraction"],
    "bagging_freq": 1,
    "reg_alpha": study.best_params["reg_alpha"],
    "reg_lambda": study.best_params["reg_lambda"],
    "max_depth": study.best_params["max_depth"],
    "n_jobs": -1,
    "verbose": -1,
}

oof_lgb_all, test_lgb_all = [], []
for seed in SEEDS:
    params = {**best_lgb_params, "random_state": seed}
    oof = np.zeros(n_train)
    test_pred = np.zeros(n_test)

    for fd in fold_data:
        model = lgb.LGBMRegressor(**params)
        model.fit(
            fd["X_tr"], fd["y_tr"],
            eval_set=[(fd["X_val"], fd["y_val"])],
            callbacks=[lgb.early_stopping(500, verbose=False), lgb.log_evaluation(5000)],
        )
        oof[fd["val_idx"]] = model.predict(fd["X_val"])
        test_pred += model.predict(fd["X_te"]) / N_FOLDS

    m = mape_oof(y_log, oof)
    print(f"  Seed {seed}: OOF MAPE={m:.6f}")
    oof_lgb_all.append(oof)
    test_lgb_all.append(test_pred)

oof_lgb = np.mean(oof_lgb_all, axis=0)
test_lgb = np.mean(test_lgb_all, axis=0)
lgb_mape = mape_oof(y_log, oof_lgb)
print(f"  [LGB-Optuna] Multi-seed OOF MAPE: {lgb_mape:.6f}")

cat_features_cb = ["region_name_cat", "district_cat", "corpus_cat", "developer_cat",
                   "hc_name_cat", "class_cat", "stage_cat", "rooms_4"]

oof_cb_all, test_cb_all = [], []
for seed in SEEDS:
    oof_cb_s = np.zeros(n_train)
    test_cb_s = np.zeros(n_test)

    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        df_tr = train_base.iloc[tr_idx].reset_index(drop=True)
        df_val = train_base.iloc[val_idx].reset_index(drop=True)
        df_te = test_base.copy()
        log_y_tr = pd.Series(y_log[tr_idx])

        df_tr, df_val, df_te = add_te_fold(df_tr, df_val, df_te, log_y_tr)

        drop_cols = [TARGET] + (["id"] if "id" in df_tr.columns else [])
        feat_cols_cb = [c for c in df_tr.columns if c not in drop_cols]

        for c in cat_features_cb:
            if c in df_tr.columns:
                df_tr[c] = df_tr[c].fillna(-1).astype(str)
                df_val[c] = df_val[c].fillna(-1).astype(str)
                df_te[c] = df_te[c].fillna(-1).astype(str)

        obj_cols = df_tr[feat_cols_cb].select_dtypes(include=["object"]).columns.tolist()
        for c in obj_cols:
            if c not in cat_features_cb:
                df_tr[c] = df_tr[c].astype("category").cat.codes
                df_val[c] = df_val[c].astype("category").cat.codes
                df_te[c] = df_te[c].astype("category").cat.codes

        cb_cat_idx = [feat_cols_cb.index(c) for c in cat_features_cb if c in feat_cols_cb]

        pool_tr = cb.Pool(df_tr[feat_cols_cb], label=y_log[tr_idx], cat_features=cb_cat_idx)
        pool_val = cb.Pool(df_val[feat_cols_cb], label=y_log[val_idx], cat_features=cb_cat_idx)
        pool_te = cb.Pool(df_te[feat_cols_cb], cat_features=cb_cat_idx)

        model_cb = cb.CatBoostRegressor(
            iterations=8000, learning_rate=0.03, depth=8, l2_leaf_reg=3,
            loss_function="RMSE", eval_metric="MAPE",
            random_seed=seed, od_type="Iter", od_wait=200,
            verbose=0, thread_count=-1,
        )
        model_cb.fit(pool_tr, eval_set=pool_val, use_best_model=True)

        oof_cb_s[val_idx] = model_cb.predict(pool_val)
        test_cb_s += model_cb.predict(pool_te) / N_FOLDS

    m = mape_oof(y_log, oof_cb_s)
    print(f"  Seed {seed}: OOF MAPE={m:.6f}")
    oof_cb_all.append(oof_cb_s)
    test_cb_all.append(test_cb_s)

oof_cb = np.mean(oof_cb_all, axis=0)
test_cb = np.mean(test_cb_all, axis=0)
cb_mape = mape_oof(y_log, oof_cb)
print(f"  [CB] Multi-seed OOF MAPE: {cb_mape:.6f}")

y_real = np.expm1(y_log)
best_blend_mape = 1.0
best_w_lgb = 0.5

for w in np.arange(0.0, 1.01, 0.01):
    blend = w * oof_lgb + (1 - w) * oof_cb
    m = mean_absolute_percentage_error(y_real, np.expm1(blend))
    if m < best_blend_mape:
        best_blend_mape = m
        best_w_lgb = w

print(f"Best blend: LGB={best_w_lgb:.2f} CB={1-best_w_lgb:.2f}")
print(f"Blend OOF MAPE: {best_blend_mape:.6f}")

blend_test = best_w_lgb * test_lgb + (1 - best_w_lgb) * test_cb

pd.DataFrame({"id": test_ids, "price_target": np.expm1(test_lgb)}).to_csv(
    "submissions/lgbm_v9.csv", index=False)
pd.DataFrame({"id": test_ids, "price_target": np.expm1(test_cb)}).to_csv(
    "submissions/cb_v9.csv", index=False)
pd.DataFrame({"id": test_ids, "price_target": np.expm1(blend_test)}).to_csv(
    "submissions/blend_v9.csv", index=False)

with open("models/v9_results.pkl", "wb") as f:
    pickle.dump({
        "oof_lgb": oof_lgb, "oof_cb": oof_cb,
        "test_lgb": test_lgb, "test_cb": test_cb,
        "blend_test": blend_test,
        "best_w_lgb": best_w_lgb,
        "best_blend_mape": best_blend_mape,
        "lgb_mape": lgb_mape, "cb_mape": cb_mape,
        "best_lgb_params": best_lgb_params,
        "y_log": y_log,
        "feat_cols": feat_cols_final,
    }, f)
