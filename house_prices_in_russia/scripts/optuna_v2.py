import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
import lightgbm as lgb
import optuna
import pickle
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
N_FOLDS = 5
N_TRIALS = 150

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
    df["floor_x_square"] = df["floor"] * df["square"]

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

    fold_data.append({
        "X_tr": df_tr[feat_cols].values,
        "X_val": df_val[feat_cols].values,
        "X_te": df_te[feat_cols].values,
        "y_tr": y_log[tr_idx],
        "y_val": y_log[val_idx],
        "val_idx": val_idx,
    })

feat_cols_final = feat_cols


def objective(trial):
    params = {
        "objective": "regression",
        "metric": "mape",
        "n_estimators": 10000,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 63, 511),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.3),
        "max_depth": trial.suggest_int("max_depth", 5, 12),
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }

    oof = np.zeros(len(train_base))
    for fd in fold_data:
        model = lgb.LGBMRegressor(**params)
        model.fit(
            fd["X_tr"], fd["y_tr"],
            eval_set=[(fd["X_val"], fd["y_val"])],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)],
        )
        oof[fd["val_idx"]] = model.predict(fd["X_val"])

    return mean_absolute_percentage_error(np.expm1(y_log), np.expm1(oof))


study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print(f"\nBest MAPE: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

best_params = dict(study.best_params)
best_params.update({
    "objective": "regression",
    "metric": "mape",
    "n_estimators": 10000,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
})

oof_preds = np.zeros(len(train_base))
test_preds = np.zeros(len(test_base))
models = []

for fold, fd in enumerate(fold_data):
    model = lgb.LGBMRegressor(**best_params)
    model.fit(
        fd["X_tr"], fd["y_tr"],
        eval_set=[(fd["X_val"], fd["y_val"])],
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(1000)],
    )
    oof_preds[fd["val_idx"]] = model.predict(fd["X_val"])
    test_preds += model.predict(fd["X_te"]) / N_FOLDS
    models.append(model)
    fold_mape = mean_absolute_percentage_error(
        np.expm1(fd["y_val"]), np.expm1(oof_preds[fd["val_idx"]])
    )
    print(f"Fold {fold+1}: MAPE={fold_mape:.4f}, best_iter={model.best_iteration_}")

oof_mape = mean_absolute_percentage_error(np.expm1(y_log), np.expm1(oof_preds))
print(f"\nOOF MAPE: {oof_mape:.4f}")

pd.DataFrame({"id": test_ids, "price_target": np.expm1(test_preds)}).to_csv(
    "submissions/optuna_lgbm_v2.csv", index=False)

with open("models/lgbm_optuna_v2.pkl", "wb") as f:
    pickle.dump({
        "models": models,
        "feature_cols": feat_cols_final,
        "best_params": best_params,
        "oof_mape": oof_mape,
        "oof_preds": oof_preds,
        "test_preds": test_preds,
        "study": study,
    }, f)
