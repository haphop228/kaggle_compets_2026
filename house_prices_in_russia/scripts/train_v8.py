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

    max_lvl = df["location_max_levels_max"].replace(0, np.nan)
    df["floor_ratio"] = df["floor"] / max_lvl
    df["floor_from_top"] = max_lvl - df["floor"]
    df["is_top_floor"] = (df["floor"] == df["location_max_levels_max"]).astype(int)
    df["is_ground_floor"] = (df["floor"] == 1).astype(int)
    df["log_square"] = np.log1p(df["square"])
    df["square_per_room"] = df["square"] / (df["rooms_4"].replace(0, np.nan))
    df["floor_x_square"] = df["floor"] * df["square"]
    df["floor_ratio_x_square"] = df["floor_ratio"] * df["square"]

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
    """Smoothed fold-TE: fit on train fold, apply to val and test. Same as v2."""
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
            dset[f"{c}_te_count"] = dset[c].map(np.log1p(agg["count"])).fillna(0)

    return df_tr, df_val, df_te


def mape_oof(y_log, oof_log):
    return mean_absolute_percentage_error(np.expm1(y_log), np.expm1(oof_log))


print("=" * 60)
print("TRAIN V8: Proven approach + multi-seed + 7 folds")
print("=" * 60)

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

print(f"Features: {len(feat_cols_final)}")
print(f"Train: {n_train}, Test: {n_test}, Folds: {N_FOLDS}")


def train_lgb_seed(fold_data, seed, params_override=None):
    params = {
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
            fd["X_tr"], fd["y_tr"],
            eval_set=[(fd["X_val"], fd["y_val"])],
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(5000)],
        )
        oof[fd["val_idx"]] = model.predict(fd["X_val"])
        test_pred += model.predict(fd["X_te"]) / N_FOLDS

    return oof, test_pred


def train_xgb_seed(fold_data, seed):
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
        "early_stopping_rounds": 200,
    }

    oof = np.zeros(n_train)
    test_pred = np.zeros(n_test)

    for fold, fd in enumerate(fold_data):
        model = xgb.XGBRegressor(**params)
        model.fit(
            fd["X_tr"], fd["y_tr"],
            eval_set=[(fd["X_val"], fd["y_val"])],
            verbose=0,
        )
        oof[fd["val_idx"]] = model.predict(fd["X_val"])
        test_pred += model.predict(fd["X_te"]) / N_FOLDS

    return oof, test_pred

print("\n" + "=" * 60)
print("MODEL 1: LightGBM GBDT (multi-seed x3)")
print("=" * 60)
oof_lgb_all, test_lgb_all = [], []
for seed in SEEDS:
    oof, test_pred = train_lgb_seed(fold_data, seed)
    m = mape_oof(y_log, oof)
    print(f"  Seed {seed}: OOF MAPE={m:.6f}")
    oof_lgb_all.append(oof)
    test_lgb_all.append(test_pred)

oof_lgb = np.mean(oof_lgb_all, axis=0)
test_lgb = np.mean(test_lgb_all, axis=0)
print(f"  [LGB] Multi-seed OOF MAPE: {mape_oof(y_log, oof_lgb):.6f}")

oof_lgb2_all, test_lgb2_all = [], []
for seed in SEEDS:
    oof, test_pred = train_lgb_seed(fold_data, seed,
                                     params_override={"learning_rate": 0.008,
                                                       "n_estimators": 20000,
                                                       "num_leaves": 200,
                                                       "min_child_samples": 30,
                                                       "reg_alpha": 0.1,
                                                       "reg_lambda": 0.1})
    m = mape_oof(y_log, oof)
    print(f"  Seed {seed}: OOF MAPE={m:.6f}")
    oof_lgb2_all.append(oof)
    test_lgb2_all.append(test_pred)

oof_lgb2 = np.mean(oof_lgb2_all, axis=0)
test_lgb2 = np.mean(test_lgb2_all, axis=0)
print(f"  [LGB-lowlr] Multi-seed OOF MAPE: {mape_oof(y_log, oof_lgb2):.6f}")

oof_xgb_all, test_xgb_all = [], []
for seed in SEEDS:
    oof, test_pred = train_xgb_seed(fold_data, seed)
    m = mape_oof(y_log, oof)
    print(f"  Seed {seed}: OOF MAPE={m:.6f}")
    oof_xgb_all.append(oof)
    test_xgb_all.append(test_pred)

oof_xgb = np.mean(oof_xgb_all, axis=0)
test_xgb = np.mean(test_xgb_all, axis=0)
print(f"  [XGB] Multi-seed OOF MAPE: {mape_oof(y_log, oof_xgb):.6f}")


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
print(f"  [CB] Multi-seed OOF MAPE: {mape_oof(y_log, oof_cb):.6f}")


all_oof = np.column_stack([oof_lgb, oof_lgb2, oof_xgb, oof_cb])
all_test = np.column_stack([test_lgb, test_lgb2, test_xgb, test_cb])
model_names = ["LGB", "LGB-lowlr", "XGB", "CB"]

y_real = np.expm1(y_log)


def optuna_objective(trial):
    raw_w = [trial.suggest_float(f"w_{i}", 0.0, 1.0) for i in range(len(model_names))]
    total = sum(raw_w)
    weights = [w / total for w in raw_w]
    blend = sum(w * all_oof[:, i] for i, w in enumerate(weights))
    return mean_absolute_percentage_error(y_real, np.expm1(blend))


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(optuna_objective, n_trials=3000)

best = study.best_params
raw_w = [best[f"w_{i}"] for i in range(len(model_names))]
total_w = sum(raw_w)
final_weights = [w / total_w for w in raw_w]

print(f"\nOptuna best MAPE: {study.best_value:.6f}")
for name, w in zip(model_names, final_weights):
    print(f"  {name}: {w:.4f}")

blend_oof = sum(w * all_oof[:, i] for i, w in enumerate(final_weights))
blend_test = sum(w * all_test[:, i] for i, w in enumerate(final_weights))
final_mape = mean_absolute_percentage_error(y_real, np.expm1(blend_oof))

# Also try simple 2-model blends
print("\n=== SIMPLE BLENDS ===")
for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        best_w = 0.5
        best_m = 1.0
        for w in np.arange(0.0, 1.01, 0.05):
            blend = w * all_oof[:, i] + (1 - w) * all_oof[:, j]
            m = mean_absolute_percentage_error(y_real, np.expm1(blend))
            if m < best_m:
                best_m = m
                best_w = w
        print(f"  {model_names[i]}={best_w:.2f} + {model_names[j]}={1-best_w:.2f}: MAPE={best_m:.6f}")

pd.DataFrame({"id": test_ids, "price_target": np.expm1(test_lgb)}).to_csv(
    "submissions/lgbm_v8.csv", index=False)
pd.DataFrame({"id": test_ids, "price_target": np.expm1(test_xgb)}).to_csv(
    "submissions/xgb_v8.csv", index=False)
pd.DataFrame({"id": test_ids, "price_target": np.expm1(test_cb)}).to_csv(
    "submissions/cb_v8.csv", index=False)
pd.DataFrame({"id": test_ids, "price_target": np.expm1(blend_test)}).to_csv(
    "submissions/blend_v8.csv", index=False)

with open("models/v8_results.pkl", "wb") as f:
    pickle.dump({
        "oof_lgb": oof_lgb, "oof_lgb2": oof_lgb2,
        "oof_xgb": oof_xgb, "oof_cb": oof_cb,
        "test_lgb": test_lgb, "test_lgb2": test_lgb2,
        "test_xgb": test_xgb, "test_cb": test_cb,
        "blend_test": blend_test,
        "final_weights": final_weights,
        "final_mape": final_mape,
        "y_log": y_log,
        "feat_cols": feat_cols_final,
        "model_names": model_names,
    }, f)

results = [
    ("LGB GBDT (multi-seed x3)", mape_oof(y_log, oof_lgb)),
    ("LGB low-lr (multi-seed x3)", mape_oof(y_log, oof_lgb2)),
    ("XGBoost (multi-seed x3)", mape_oof(y_log, oof_xgb)),
    ("CatBoost (multi-seed x3)", mape_oof(y_log, oof_cb)),
    ("Final blend", final_mape),
]
for name, m in sorted(results, key=lambda x: x[1]):
    print(f"  {m:.6f}  {name}")

print(f"\nPrevious best: 0.0199 (Blend LGB80+CB20 v2, 5 folds)")
print(f"New best:      {final_mape:.6f}")
print(f"Improvement:   {0.0199 - final_mape:.6f}")
