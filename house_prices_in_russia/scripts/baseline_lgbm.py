import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

SEED = 42
N_FOLDS = 5

train = pd.read_csv("period_1_train_data.csv")
test = pd.read_csv("test_x.csv")

test_ids = test["id"].values


def preprocess(df):
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

    df["all_loc_nan"] = df[loc_cols].isnull().all(axis=1).astype(int)

    high_miss = [
        "location_town_w_mean_distance",
        "location_university_w_mean_distance",
        "location_commercial_w_mean_distance",
        "location_amenity_leisure_w_mean_distance",
        "location_water_w_mean_distance",
        "location_hotel_w_mean_distance",
        "location_industrial_w_mean_distance",
        "location_railway_station_w_mean_distance",
        "location_social_w_mean_distance",
        "location_shop_alco_w_mean_distance",
        "location_tourism_w_mean_distance",
        "location_pop_bank_w_mean_distance",
    ]
    for c in high_miss:
        if c in df.columns:
            df[f"{c}_miss"] = df[c].isnull().astype(int)

    df["region_name_cat"] = df["region_name_cat"].astype("category").cat.codes

    return df


train = preprocess(train)
test = preprocess(test)

TARGET = "price_target"
DROP_COLS = [TARGET, "id"] if "id" in train.columns else [TARGET]
feature_cols = [c for c in train.columns if c not in DROP_COLS]

cat_cols = train[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()
for c in cat_cols:
    train[c] = train[c].astype("category").cat.codes
    test[c] = test[c].astype("category").cat.codes

X = train[feature_cols].values
y = np.log1p(train[TARGET].values)
X_test = test[feature_cols].values

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

lgb_params = {
    "objective": "regression",
    "metric": "mape",
    "n_estimators": 3000,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)],
    )

    val_pred = model.predict(X_val)
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(X_test) / N_FOLDS

    fold_mape = mean_absolute_percentage_error(np.expm1(y_val), np.expm1(val_pred))
    print(f"Fold {fold+1}: MAPE={fold_mape:.4f}, best_iter={model.best_iteration_}")

oof_mape = mean_absolute_percentage_error(np.expm1(y), np.expm1(oof_preds))
print(f"\nOOF MAPE: {oof_mape:.4f}")

sub = pd.DataFrame({"id": test_ids, "price_target": np.expm1(test_preds)})
sub.to_csv("submissions/baseline_lgbm.csv", index=False)
