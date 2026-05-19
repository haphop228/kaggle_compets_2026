import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
import lightgbm as lgb
import pickle
import warnings
warnings.filterwarnings("ignore")

SEED = 42
N_FOLDS = 5

train_raw = pd.read_csv("period_1_train_data.csv")
test_raw = pd.read_csv("test_x.csv")
test_ids = test_raw["id"].values

TE_COLS = ["district_cat", "corpus_cat", "developer_cat", "hc_name_cat",
           "class_cat", "stage_cat"]

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
    df["is_second_floor"] = (df["floor"] == 2).astype(int)
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

    df["region_name_cat"] = df["region_name_cat"].astype("category").cat.codes
    return df


def add_te_fold(df_tr, df_val, df_te, log_y_tr):
    df_tr = df_tr.copy()
    df_val = df_val.copy()
    df_te = df_te.copy()
    global_mean = log_y_tr.mean()
    k = 10
    for c in TE_COLS:
        agg = log_y_tr.groupby(df_tr[c]).agg(["mean", "std", "count"])
        agg["smooth"] = (agg["mean"] * agg["count"] + global_mean * k) / (agg["count"] + k)
        for dset in [df_tr, df_val, df_te]:
            dset[f"{c}_te"] = dset[c].map(agg["smooth"]).fillna(global_mean)
            dset[f"{c}_te_std"] = dset[c].map(agg["std"]).fillna(0)
    return df_tr, df_val, df_te


train_base = base_features(train_raw)
test_base = base_features(test_raw)

TARGET = "price_target"
y_log = np.log1p(train_base[TARGET].values)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
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
    })

lgb_params = {
    "objective": "regression",
    "metric": "mape",
    "n_estimators": 15000,
    "learning_rate": 0.01,
    "num_leaves": 255,
    "min_child_samples": 20,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "reg_alpha": 0.05,
    "reg_lambda": 0.05,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}

oof_lgb = np.zeros(len(train_base))
test_lgb = np.zeros(len(test_base))

for fold, fd in enumerate(fold_data):
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        fd["X_tr"], fd["y_tr"],
        eval_set=[(fd["X_val"], fd["y_val"])],
        callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(2000)],
    )
    oof_lgb[fd["val_idx"]] = model.predict(fd["X_val"])
    test_lgb += model.predict(fd["X_te"]) / N_FOLDS
    fold_mape = mean_absolute_percentage_error(
        np.expm1(fd["y_val"]), np.expm1(oof_lgb[fd["val_idx"]])
    )
    print(f"[LGB] Fold {fold+1}: MAPE={fold_mape:.4f}, best_iter={model.best_iteration_}")

lgb_oof_mape = mean_absolute_percentage_error(np.expm1(y_log), np.expm1(oof_lgb))
print(f"\n[LGB] OOF MAPE: {lgb_oof_mape:.4f}")

pd.DataFrame({"id": test_ids, "price_target": np.expm1(test_lgb)}).to_csv(
    "submissions/lgbm_v3.csv", index=False)

with open("models/v3_results.pkl", "wb") as f:
    pickle.dump({
        "oof_lgb": oof_lgb,
        "test_lgb": test_lgb,
        "lgb_mape": lgb_oof_mape,
        "feat_cols": feat_cols_final,
        "y_log": y_log,
    }, f)
