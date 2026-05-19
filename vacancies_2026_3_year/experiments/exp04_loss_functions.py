"""
Experiment 04: Loss Function Optimization (MAE / MAPE)
exp03 best: Ridge(200k, alpha=0.5) + XGB_sep800, OOF=0.2169

Tests:
  A) LGBM with MAE loss on log(y)
  B) LGBM with MAPE loss on raw y
  C) XGBoost with MAE loss on log(y)
  D) XGBoost with MAPE eval metric (custom objective if needed, or just MAE on raw y)
  E) Blend best tree from here with best Ridge from exp03

Run: .venv/bin/python experiments/exp04_loss_functions.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgblib
import warnings
warnings.filterwarnings("ignore")

from experiments.features import (
    load_data, build_tabular_features, build_separate_text_features, build_fold, mape,
    TARGET, SEED, N_SPLITS, DESC_COL, TITLE_COL, SKILLS_COL, NUM_COLS, LOW_CARD,
)

train, test = load_data()
y_raw = train[TARGET].values.astype(np.float64)
y_log = np.log1p(y_raw).astype(np.float32)
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

X_num_tr, X_num_te, X_low_tr, X_low_te, _ = build_tabular_features(train, test)

for df in [train, test]:
    df[DESC_COL]    = df[DESC_COL].fillna("")
    df[TITLE_COL]   = df[TITLE_COL].fillna("")
    df[SKILLS_COL]  = df[SKILLS_COL].fillna("")

# Use the best SVD configuration from exp03 (sep800)
print("\nBuilding Separate SVD features (title200+desc500+skills100)...")
X_svd_tr, X_svd_te = build_separate_text_features(
    train, test, title_svd=200, desc_svd=500, skills_svd=100,
    title_max=20000, desc_max=50000, skills_max=10000)

results = []

def run_trees(lgb_params, xgb_params, label, use_raw_y=False):
    oof_l = np.zeros(len(train)); pred_l = np.zeros(len(test))
    oof_x = np.zeros(len(train)); pred_x = np.zeros(len(test))
    
    target_tr = y_raw if use_raw_y else y_log
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_svd_tr)):
        # build_fold uses y_log internally for Target Encoding. We keep it that way for TE stability.
        Xtr, Xval, Xte = build_fold(
            train, test, tr_idx, val_idx, y_log,
            X_svd_tr, X_svd_te, X_low_tr, X_low_te, X_num_tr, X_num_te)
        
        if lgb_params is not None:
            ml = lgb.LGBMRegressor(**lgb_params)
            ml.fit(Xtr, target_tr[tr_idx], eval_set=[(Xval, target_tr[val_idx])],
                   callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
            
            preds_val = ml.predict(Xval)
            preds_te = ml.predict(Xte)
            
            if not use_raw_y:
                preds_val = np.expm1(preds_val)
                preds_te = np.expm1(preds_te)
                
            oof_l[val_idx] = preds_val
            pred_l += preds_te / N_SPLITS

        if xgb_params is not None:
            mx = xgblib.XGBRegressor(**xgb_params)
            mx.fit(Xtr, target_tr[tr_idx], eval_set=[(Xval, target_tr[val_idx])], verbose=False)
            
            preds_val = mx.predict(Xval)
            preds_te = mx.predict(Xte)
            
            if not use_raw_y:
                preds_val = np.expm1(preds_val)
                preds_te = np.expm1(preds_te)
                
            oof_x[val_idx] = preds_val
            pred_x += preds_te / N_SPLITS

        if lgb_params and xgb_params:
            fl = mape(y_raw[val_idx], oof_l[val_idx])
            fx = mape(y_raw[val_idx], oof_x[val_idx])
            print(f"    Fold {fold+1}: LGB={fl:.4f} XGB={fx:.4f}")
        elif lgb_params:
            fl = mape(y_raw[val_idx], oof_l[val_idx])
            print(f"    Fold {fold+1}: LGB={fl:.4f}")
        elif xgb_params:
            fx = mape(y_raw[val_idx], oof_x[val_idx])
            print(f"    Fold {fold+1}: XGB={fx:.4f}")

    sl, sx = 1.0, 1.0
    if lgb_params:
        sl = mape(y_raw, oof_l)
        print(f"  {label}: LGB={sl:.4f}")
    if xgb_params:
        sx = mape(y_raw, oof_x)
        print(f"  {label}: XGB={sx:.4f}")
        
    return oof_l, pred_l, sl, oof_x, pred_x, sx

# Base params from exp03
LGB_BASE = {
    "n_estimators": 8000, "learning_rate": 0.02,
    "num_leaves": 255, "min_child_samples": 15,
    "feature_fraction": 0.6, "bagging_fraction": 0.8, "bagging_freq": 5,
    "reg_alpha": 0.05, "reg_lambda": 0.1,
    "random_state": SEED, "n_jobs": -1, "verbose": -1,
}
XGB_BASE = {
    "n_estimators": 8000, "learning_rate": 0.02,
    "max_depth": 7, "min_child_weight": 10,
    "subsample": 0.8, "colsample_bytree": 0.6,
    "reg_alpha": 0.05, "reg_lambda": 0.1,
    "random_state": SEED, "n_jobs": -1, "tree_method": "hist",
    "early_stopping_rounds": 200, "verbosity": 0,
}

# ════════════════════════════════════════════════════════════════════════════════
# A: MAE Loss on log(y)
# ════════════════════════════════════════════════════════════════════════════════
print("\n=== A: MAE Loss on log(y) ===")
lgb_mae = {**LGB_BASE, "objective": "mae", "metric": "mae"}
xgb_mae = {**XGB_BASE, "objective": "reg:absoluteerror", "eval_metric": "mae"}

oof_lgb_A, pred_lgb_A, sl_A, oof_xgb_A, pred_xgb_A, sx_A = run_trees(
    lgb_mae, xgb_mae, "MAE_log_y", use_raw_y=False)
results.append({"config": "A_LGB_MAE_log", "score": sl_A})
results.append({"config": "A_XGB_MAE_log", "score": sx_A})

# ════════════════════════════════════════════════════════════════════════════════
# B: MAPE Loss on raw y
# ════════════════════════════════════════════════════════════════════════════════
print("\n=== B: MAPE Loss on raw y ===")
lgb_mape = {**LGB_BASE, "objective": "mape", "metric": "mape"}
# XGBoost doesn't have a native MAPE objective, only eval_metric. 
# We can use reg:absoluteerror (MAE) on raw y and eval_metric='mape'
xgb_mape = {**XGB_BASE, "objective": "reg:absoluteerror", "eval_metric": "mape"}

oof_lgb_B, pred_lgb_B, sl_B, oof_xgb_B, pred_xgb_B, sx_B = run_trees(
    lgb_mape, xgb_mape, "MAPE_raw_y", use_raw_y=True)
results.append({"config": "B_LGB_MAPE_raw", "score": sl_B})
results.append({"config": "B_XGB_MAE_raw", "score": sx_B})

# ════════════════════════════════════════════════════════════════════════════════
# C: Baseline RMSE on log(y) (for direct comparison in this script)
# ════════════════════════════════════════════════════════════════════════════════
print("\n=== C: Baseline RMSE on log(y) ===")
lgb_rmse = {**LGB_BASE, "objective": "regression", "metric": "rmse"}
xgb_rmse = {**XGB_BASE, "objective": "reg:squarederror", "eval_metric": "rmse"}

oof_lgb_C, pred_lgb_C, sl_C, oof_xgb_C, pred_xgb_C, sx_C = run_trees(
    lgb_rmse, xgb_rmse, "RMSE_log_y", use_raw_y=False)
results.append({"config": "C_LGB_RMSE_log", "score": sl_C})
results.append({"config": "C_XGB_RMSE_log", "score": sx_C})

# ════════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════════
df_res = pd.DataFrame(results)
df_res.to_csv("experiments/results_exp04.csv", index=False)
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(df_res[["config", "score"]].to_string(index=False))

best = df_res.loc[df_res["score"].idxmin()]
print(f"\nBest: {best['config']}  MAPE={best['score']:.4f}")
