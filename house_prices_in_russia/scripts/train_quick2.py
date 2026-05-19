import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
import lightgbm as lgb
import catboost as cb
import warnings
warnings.filterwarnings("ignore")

train_raw = pd.read_csv("period_1_train_data.csv")
test_raw = pd.read_csv("test_x.csv")
test_ids = test_raw["id"].values

TE_COLS = ["district_cat", "corpus_cat", "developer_cat", "hc_name_cat",
           "class_cat", "stage_cat"]


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
    df["floor_ratio"] = df["floor"] / (df["location_max_levels_max"].replace(0, np.nan))
    df["floor_from_top"] = df["location_max_levels_max"] - df["floor"]
    df["is_top_floor"] = (df["floor"] == df["location_max_levels_max"]).astype(int)
    df["is_ground_floor"] = (df["floor"] == 1).astype(int)
    df["log_square"] = np.log1p(df["square"])
    df["square_per_room"] = df["square"] / (df["rooms_4"].replace(0, np.nan))
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
    df["region_name_cat"] = df["region_name_cat"].astype("category").cat.codes
    return df


def add_te_fold(df_tr, df_val, df_test_fold, log_y_tr):
    df_tr = df_tr.copy()
    df_val = df_val.copy()
    df_test_fold = df_test_fold.copy()
    global_mean = log_y_tr.mean()
    for c in TE_COLS:
        agg = log_y_tr.groupby(df_tr[c]).agg(["mean", "std", "count"])
        k = 10
        agg["smooth_mean"] = (agg["mean"] * agg["count"] + global_mean * k) / (agg["count"] + k)
        df_tr[f"{c}_te"] = df_tr[c].map(agg["smooth_mean"]).fillna(global_mean)
        df_tr[f"{c}_te_std"] = df_tr[c].map(agg["std"]).fillna(0)
        df_val[f"{c}_te"] = df_val[c].map(agg["smooth_mean"]).fillna(global_mean)
        df_val[f"{c}_te_std"] = df_val[c].map(agg["std"]).fillna(0)
        df_test_fold[f"{c}_te"] = df_test_fold[c].map(agg["smooth_mean"]).fillna(global_mean)
        df_test_fold[f"{c}_te_std"] = df_test_fold[c].map(agg["std"]).fillna(0)
    return df_tr, df_val, df_test_fold


train_base = base_features(train_raw)
test_base = base_features(test_raw)

TARGET = "price_target"
y_log = np.log1p(train_base[TARGET].values)
y_real = train_base[TARGET].values

N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
splits = list(kf.split(train_base))
n_train = len(train_base)
n_test = len(test_base)

cat_features_cb = ["region_name_cat", "district_cat", "corpus_cat", "developer_cat",
                   "hc_name_cat", "class_cat", "stage_cat", "rooms_4"]

lgb_params = {
    "objective": "regression",
    "metric": "mape",
    "n_estimators": 10000,
    "learning_rate": 0.02,
    "num_leaves": 255,
    "min_child_samples": 20,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "reg_alpha": 0.05,
    "reg_lambda": 0.05,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

oof_lgb = np.zeros(n_train)
test_lgb = np.zeros(n_test)

for fold, (tr_idx, val_idx) in enumerate(splits):
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
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        df_tr[feat_cols].values, y_log[tr_idx],
        eval_set=[(df_val[feat_cols].values, y_log[val_idx])],
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    oof_lgb[val_idx] = model.predict(df_val[feat_cols].values)
    test_lgb += model.predict(df_te[feat_cols].values) / N_FOLDS
    print(f"  LGB Fold {fold+1}: MAPE={mean_absolute_percentage_error(np.expm1(y_log[val_idx]), np.expm1(oof_lgb[val_idx])):.6f}, iter={model.best_iteration_}")

print(f"LGB OOF MAPE: {mean_absolute_percentage_error(y_real, np.expm1(oof_lgb)):.6f}")

oof_cb = np.zeros(n_train)
test_cb = np.zeros(n_test)

for fold, (tr_idx, val_idx) in enumerate(splits):
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
        iterations=5000, learning_rate=0.05, depth=8, l2_leaf_reg=3,
        loss_function="MAPE", eval_metric="MAPE",
        random_seed=42, od_type="Iter", od_wait=200,
        verbose=0, thread_count=-1,
    )
    model_cb.fit(pool_tr, eval_set=pool_val, use_best_model=True)
    oof_cb[val_idx] = model_cb.predict(pool_val)
    test_cb += model_cb.predict(pool_te) / N_FOLDS
    print(f"  CB Fold {fold+1}: MAPE={mean_absolute_percentage_error(np.expm1(y_log[val_idx]), np.expm1(oof_cb[val_idx])):.6f}")

print(f"CB OOF MAPE: {mean_absolute_percentage_error(y_real, np.expm1(oof_cb)):.6f}")

best_mape = 1.0
best_w = 0.8
for w in np.arange(0.0, 1.01, 0.01):
    blend = w * oof_lgb + (1 - w) * oof_cb
    m = mean_absolute_percentage_error(y_real, np.expm1(blend))
    if m < best_mape:
        best_mape = m
        best_w = w

print(f"Best blend: LGB={best_w:.2f} CB={1-best_w:.2f}, MAPE={best_mape:.6f}")

test_blend = np.expm1(best_w * test_lgb + (1 - best_w) * test_cb)

pd.DataFrame({"id": test_ids, "price_target": np.expm1(test_lgb)}).to_csv(
    "submissions/lgbm_v10_final.csv", index=False)
pd.DataFrame({"id": test_ids, "price_target": test_blend}).to_csv(
    "submissions/blend_v10_final.csv", index=False)
