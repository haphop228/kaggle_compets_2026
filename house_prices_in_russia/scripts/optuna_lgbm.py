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
N_TRIALS = 100

train_raw = pd.read_csv("period_1_train_data.csv")
test_raw = pd.read_csv("test_x.csv")
test_ids = test_raw["id"].values


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

    df["all_loc_nan"] = df[loc_cols].isnull().all(axis=1).astype(int)

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


TE_COLS = ["district_cat", "corpus_cat", "developer_cat", "hc_name_cat",
           "class_cat", "stage_cat", "interior_cat"]


def add_target_encoding_oof(df_train, df_test):
    df_train = df_train.copy()
    df_test = df_test.copy()
    log_y = np.log1p(df_train["price_target"])
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for c in TE_COLS:
        df_train[f"{c}_te_mean"] = np.nan
        df_train[f"{c}_te_std"] = np.nan

        for tr_idx, val_idx in kf.split(df_train):
            agg = log_y.iloc[tr_idx].groupby(df_train[c].iloc[tr_idx]).agg(["mean", "std"])
            df_train.loc[df_train.index[val_idx], f"{c}_te_mean"] = \
                df_train[c].iloc[val_idx].map(agg["mean"]).values
            df_train.loc[df_train.index[val_idx], f"{c}_te_std"] = \
                df_train[c].iloc[val_idx].map(agg["std"]).values

        agg_full = log_y.groupby(df_train[c]).agg(["mean", "std"])
        df_test[f"{c}_te_mean"] = df_test[c].map(agg_full["mean"])
        df_test[f"{c}_te_std"] = df_test[c].map(agg_full["std"])

    return df_train, df_test


train_base = base_features(train_raw)
test_base = base_features(test_raw)

train_fe, test_fe = add_target_encoding_oof(train_base, test_base)

TARGET = "price_target"
drop_cols = [TARGET] + (["id"] if "id" in train_fe.columns else [])
feature_cols = [c for c in train_fe.columns if c not in drop_cols]

cat_obj = train_fe[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()
for c in cat_obj:
    train_fe[c] = train_fe[c].astype("category").cat.codes
    test_fe[c] = test_fe[c].astype("category").cat.codes

X = train_fe[feature_cols].values
y = np.log1p(train_fe[TARGET].values)
X_test = test_fe[feature_cols].values

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
splits = list(kf.split(X))


def objective(trial):
    params = {
        "objective": "regression",
        "metric": "mape",
        "n_estimators": 5000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 511),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }

    oof = np.zeros(len(X))
    for tr_idx, val_idx in splits:
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X[tr_idx], y[tr_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        oof[val_idx] = model.predict(X[val_idx])

    return mean_absolute_percentage_error(np.expm1(y), np.expm1(oof))


study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print(f"\nBest MAPE: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

best_params = dict(study.best_params)
best_params.update({
    "objective": "regression",
    "metric": "mape",
    "n_estimators": 5000,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
})

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
models = []

for fold, (tr_idx, val_idx) in enumerate(splits):
    model = lgb.LGBMRegressor(**best_params)
    model.fit(
        X[tr_idx], y[tr_idx],
        eval_set=[(X[val_idx], y[val_idx])],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)],
    )
    oof_preds[val_idx] = model.predict(X[val_idx])
    test_preds += model.predict(X_test) / N_FOLDS
    models.append(model)
    fold_mape = mean_absolute_percentage_error(np.expm1(y[val_idx]), np.expm1(oof_preds[val_idx]))
    print(f"Fold {fold+1}: MAPE={fold_mape:.4f}, best_iter={model.best_iteration_}")

oof_mape = mean_absolute_percentage_error(np.expm1(y), np.expm1(oof_preds))
print(f"\nOOF MAPE: {oof_mape:.4f}")

sub = pd.DataFrame({"id": test_ids, "price_target": np.expm1(test_preds)})
sub.to_csv("submissions/optuna_lgbm.csv", index=False)

with open("models/lgbm_tuned.pkl", "wb") as f:
    pickle.dump({
        "models": models,
        "feature_cols": feature_cols,
        "best_params": best_params,
        "oof_mape": oof_mape,
        "oof_preds": oof_preds,
    }, f)
