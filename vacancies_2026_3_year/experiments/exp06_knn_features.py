"""
Experiment 06: KNN Features
exp03 best: Ridge(200k, alpha=0.5) + XGB_sep800, OOF=0.2169

Tests:
  A) Add KNN features (median salary of top-K similar job titles) to XGBoost
  B) Add KNN features to LightGBM
  C) Blend best Ridge with KNN-enhanced trees

Run: .venv/bin/python experiments/exp06_knn_features.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
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

# ── 1. Build KNN Features ─────────────────────────────────────────────────────
print("\nBuilding KNN Features...")
t0 = time.time()

# We use title + skills for KNN
corpus_knn_tr = train[TITLE_COL] + " " + train[SKILLS_COL]
corpus_knn_te = test[TITLE_COL] + " " + test[SKILLS_COL]
corpus_knn = pd.concat([corpus_knn_tr, corpus_knn_te], ignore_index=True)

tfidf_knn = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True, min_df=2, dtype=np.float32)
X_knn_all = tfidf_knn.fit_transform(corpus_knn)
X_knn_tr = X_knn_all[:len(train)]
X_knn_te = X_knn_all[len(train):]

K_NEIGHBORS = 10
knn_feats_tr = np.zeros((len(train), 4), dtype=np.float32) # median, mean, min_dist, mean_dist
knn_feats_te = np.zeros((len(test), 4), dtype=np.float32)

# OOF KNN for train
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_knn_tr)):
    nn = NearestNeighbors(n_neighbors=K_NEIGHBORS, metric='cosine', n_jobs=-1)
    nn.fit(X_knn_tr[tr_idx])
    
    distances, indices = nn.kneighbors(X_knn_tr[val_idx])
    
    # Get targets of neighbors
    neighbor_targets = y_log[tr_idx][indices] # shape: (len(val_idx), K)
    
    knn_feats_tr[val_idx, 0] = np.median(neighbor_targets, axis=1)
    knn_feats_tr[val_idx, 1] = np.mean(neighbor_targets, axis=1)
    knn_feats_tr[val_idx, 2] = distances[:, 0] # distance to nearest
    knn_feats_tr[val_idx, 3] = np.mean(distances, axis=1)

# KNN for test (using full train)
nn_full = NearestNeighbors(n_neighbors=K_NEIGHBORS, metric='cosine', n_jobs=-1)
nn_full.fit(X_knn_tr)
distances_te, indices_te = nn_full.kneighbors(X_knn_te)
neighbor_targets_te = y_log[indices_te]

knn_feats_te[:, 0] = np.median(neighbor_targets_te, axis=1)
knn_feats_te[:, 1] = np.mean(neighbor_targets_te, axis=1)
knn_feats_te[:, 2] = distances_te[:, 0]
knn_feats_te[:, 3] = np.mean(distances_te, axis=1)

print(f"KNN features built in {time.time() - t0:.1f}s")

# ── 2. Build Base Features ────────────────────────────────────────────────────
print("\nBuilding Separate SVD features (title200+desc500+skills100)...")
X_svd_tr, X_svd_te = build_separate_text_features(
    train, test, title_svd=200, desc_svd=500, skills_svd=100,
    title_max=20000, desc_max=50000, skills_max=10000)

# Append KNN features to SVD features
X_svd_tr_knn = np.hstack([X_svd_tr, knn_feats_tr])
X_svd_te_knn = np.hstack([X_svd_te, knn_feats_te])

results = []

def run_trees(X_tr_svd, X_te_svd, lgb_params, xgb_params, label):
    oof_l = np.zeros(len(train)); pred_l = np.zeros(len(test))
    oof_x = np.zeros(len(train)); pred_x = np.zeros(len(test))
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tr_svd)):
        Xtr, Xval, Xte = build_fold(
            train, test, tr_idx, val_idx, y_log,
            X_tr_svd, X_te_svd, X_low_tr, X_low_te, X_num_tr, X_num_te)
        
        if lgb_params is not None:
            ml = lgb.LGBMRegressor(**lgb_params)
            ml.fit(Xtr, y_log[tr_idx], eval_set=[(Xval, y_log[val_idx])],
                   callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
            oof_l[val_idx] = np.expm1(ml.predict(Xval))
            pred_l += np.expm1(ml.predict(Xte)) / N_SPLITS

        if xgb_params is not None:
            mx = xgblib.XGBRegressor(**xgb_params)
            mx.fit(Xtr, y_log[tr_idx], eval_set=[(Xval, y_log[val_idx])], verbose=False)
            oof_x[val_idx] = np.expm1(mx.predict(Xval))
            pred_x += np.expm1(mx.predict(Xte)) / N_SPLITS

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

LGB_BASE = {
    "objective": "regression", "metric": "rmse",
    "n_estimators": 8000, "learning_rate": 0.02,
    "num_leaves": 255, "min_child_samples": 15,
    "feature_fraction": 0.6, "bagging_fraction": 0.8, "bagging_freq": 5,
    "reg_alpha": 0.05, "reg_lambda": 0.1,
    "random_state": SEED, "n_jobs": -1, "verbose": -1,
}
XGB_BASE = {
    "objective": "reg:squarederror", "eval_metric": "rmse",
    "n_estimators": 8000, "learning_rate": 0.02,
    "max_depth": 7, "min_child_weight": 10,
    "subsample": 0.8, "colsample_bytree": 0.6,
    "reg_alpha": 0.05, "reg_lambda": 0.1,
    "random_state": SEED, "n_jobs": -1, "tree_method": "hist",
    "early_stopping_rounds": 200, "verbosity": 0,
}

# ════════════════════════════════════════════════════════════════════════════════
# A: Trees WITH KNN Features
# ════════════════════════════════════════════════════════════════════════════════
print("\n=== A: Trees WITH KNN Features ===")
oof_lgb_knn, pred_lgb_knn, sl_knn, oof_xgb_knn, pred_xgb_knn, sx_knn = run_trees(
    X_svd_tr_knn, X_svd_te_knn, LGB_BASE, XGB_BASE, "Trees_with_KNN")
results.append({"config": "A_LGB_with_KNN", "score": sl_knn})
results.append({"config": "A_XGB_with_KNN", "score": sx_knn})

# ════════════════════════════════════════════════════════════════════════════════
# B: Trees WITHOUT KNN Features (Baseline for comparison)
# ════════════════════════════════════════════════════════════════════════════════
print("\n=== B: Trees WITHOUT KNN Features ===")
oof_lgb_base, pred_lgb_base, sl_base, oof_xgb_base, pred_xgb_base, sx_base = run_trees(
    X_svd_tr, X_svd_te, LGB_BASE, XGB_BASE, "Trees_baseline")
results.append({"config": "B_LGB_baseline", "score": sl_base})
results.append({"config": "B_XGB_baseline", "score": sx_base})

# ════════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════════
df_res = pd.DataFrame(results)
df_res.to_csv("experiments/results_exp06.csv", index=False)
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(df_res[["config", "score"]].to_string(index=False))

best = df_res.loc[df_res["score"].idxmin()]
print(f"\nBest: {best['config']}  MAPE={best['score']:.4f}")
