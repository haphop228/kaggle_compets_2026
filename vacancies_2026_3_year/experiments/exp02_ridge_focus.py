"""
Experiment 02: Focus on improving Ridge (dominant model at weight=0.5)
Ridge OOF = 0.2434 — this is the bottleneck.

Tests:
  A) Ridge baseline (60k mega, alpha=1)
  B) Ridge with 100k features
  C) Ridge with char n-grams added
  D) Separate Ridge for title (20k) + desc (80k) + skills (10k)
  E) Ridge on desc only (80k) — title may add noise
  F) Ridge alpha grid search (0.1, 0.5, 1, 2, 5)
  G) Ridge + separate SVD (B from exp01) as best tree blend

Run: .venv/bin/python experiments/exp02_ridge_focus.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings("ignore")

from experiments.features import (
    load_data, build_tabular_features, mape,
    TARGET, SEED, N_SPLITS,
    DESC_COL, TITLE_COL, SKILLS_COL,
)

train, test = load_data()
y_raw = train[TARGET].values.astype(np.float64)
y_log = np.log1p(y_raw).astype(np.float32)
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# Tabular (needed for OHE cat features in Ridge)
_, _, _, _, _ = build_tabular_features(train, test)

for df in [train, test]:
    df[DESC_COL]   = df[DESC_COL].fillna("")
    df[TITLE_COL]  = df[TITLE_COL].fillna("")
    df[SKILLS_COL] = df[SKILLS_COL].fillna("")
    df["mega_text"] = df[TITLE_COL] + " " + df[SKILLS_COL] + " " + df[DESC_COL]

n_train = len(train)

# OHE categorical features (shared across all Ridge configs)
ohe = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
cat_cols = ["experience_name", "professional_roles_name", "schedule_name",
            "employment_name", "specializations_profarea_name", "unified_address_region",
            "unified_address_state", "employer_industries"]
Xcat_tr = ohe.fit_transform(train[cat_cols].astype(str))
Xcat_te = ohe.transform(test[cat_cols].astype(str))
print(f"OHE cat features: {Xcat_tr.shape[1]}")

results = []


def run_ridge_sparse(X_tr, X_te, label, alpha=1.0):
    oof = np.zeros(len(train)); pred = np.zeros(len(test))
    for tr_idx, val_idx in kf.split(range(n_train)):
        m = Ridge(alpha=alpha)
        m.fit(X_tr[tr_idx], y_log[tr_idx])
        oof[val_idx] = np.expm1(m.predict(X_tr[val_idx]))
        pred += np.expm1(m.predict(X_te)) / N_SPLITS
    score = mape(y_raw, oof)
    print(f"  Ridge {label} (alpha={alpha}): {score:.4f}")
    return oof, pred, score


# ── A: Baseline (60k mega + OHE) ─────────────────────────────────────────────
print("\n=== A: Baseline 60k mega ===")
corpus = pd.concat([train["mega_text"], test["mega_text"]], ignore_index=True)
tf_A = TfidfVectorizer(max_features=60000, ngram_range=(1,2), sublinear_tf=True, min_df=2, dtype=np.float32)
X_A = tf_A.fit_transform(corpus)
Xr_A_tr = sp.hstack([X_A[:n_train], Xcat_tr])
Xr_A_te = sp.hstack([X_A[n_train:], Xcat_te])
oof_A, pred_A, sc_A = run_ridge_sparse(Xr_A_tr, Xr_A_te, "60k_mega")
results.append({"config": "A_60k_mega", "score": sc_A, "alpha": 1.0})

# ── B: 100k features ─────────────────────────────────────────────────────────
print("\n=== B: 100k mega ===")
tf_B = TfidfVectorizer(max_features=100000, ngram_range=(1,2), sublinear_tf=True, min_df=2, dtype=np.float32)
X_B = tf_B.fit_transform(corpus)
Xr_B_tr = sp.hstack([X_B[:n_train], Xcat_tr])
Xr_B_te = sp.hstack([X_B[n_train:], Xcat_te])
oof_B, pred_B, sc_B = run_ridge_sparse(Xr_B_tr, Xr_B_te, "100k_mega")
results.append({"config": "B_100k_mega", "score": sc_B, "alpha": 1.0})

# ── C: Add char n-grams ───────────────────────────────────────────────────────
print("\n=== C: word(60k) + char(30k) n-grams ===")
tf_char = TfidfVectorizer(max_features=30000, analyzer="char_wb", ngram_range=(3,5),
                          sublinear_tf=True, min_df=5, dtype=np.float32)
X_char = tf_char.fit_transform(corpus)
Xr_C_tr = sp.hstack([X_A[:n_train], X_char[:n_train], Xcat_tr])
Xr_C_te = sp.hstack([X_A[n_train:], X_char[n_train:], Xcat_te])
oof_C, pred_C, sc_C = run_ridge_sparse(Xr_C_tr, Xr_C_te, "60k+char30k")
results.append({"config": "C_word+char", "score": sc_C, "alpha": 1.0})

# ── D: Separate TF-IDF for title, desc, skills ───────────────────────────────
print("\n=== D: Separate TF-IDF (title20k + desc80k + skills10k) ===")
corpus_t = pd.concat([train[TITLE_COL],  test[TITLE_COL]],  ignore_index=True)
corpus_d = pd.concat([train[DESC_COL],   test[DESC_COL]],   ignore_index=True)
corpus_s = pd.concat([train[SKILLS_COL], test[SKILLS_COL]], ignore_index=True)

tf_t = TfidfVectorizer(max_features=20000, ngram_range=(1,2), sublinear_tf=True, min_df=2, dtype=np.float32)
tf_d = TfidfVectorizer(max_features=80000, ngram_range=(1,2), sublinear_tf=True, min_df=2, dtype=np.float32)
tf_s = TfidfVectorizer(max_features=10000, ngram_range=(1,2), sublinear_tf=True, min_df=2, dtype=np.float32)

Xt = tf_t.fit_transform(corpus_t)
Xd = tf_d.fit_transform(corpus_d)
Xs = tf_s.fit_transform(corpus_s)

Xr_D_tr = sp.hstack([Xt[:n_train], Xd[:n_train], Xs[:n_train], Xcat_tr])
Xr_D_te = sp.hstack([Xt[n_train:], Xd[n_train:], Xs[n_train:], Xcat_te])
print(f"  Separate TF-IDF shape: {Xr_D_tr.shape}")
oof_D, pred_D, sc_D = run_ridge_sparse(Xr_D_tr, Xr_D_te, "sep_title20k+desc80k+skills10k")
results.append({"config": "D_sep_tfidf", "score": sc_D, "alpha": 1.0})

# ── E: Alpha grid search on best config so far ───────────────────────────────
print("\n=== E: Alpha grid search on best config ===")
best_so_far = min(results, key=lambda x: x["score"])
print(f"  Best so far: {best_so_far['config']} = {best_so_far['score']:.4f}")

# Use D (separate) as it's likely best
for alpha in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]:
    _, _, sc = run_ridge_sparse(Xr_D_tr, Xr_D_te, f"sep_alpha{alpha}", alpha=alpha)
    results.append({"config": f"E_sep_alpha{alpha}", "score": sc, "alpha": alpha})

# ── F: ElasticNet on best config ─────────────────────────────────────────────
print("\n=== F: ElasticNet (l1_ratio=0.1) ===")
# ElasticNet needs dense — skip for large sparse, use only on SVD
# Instead try Ridge with 150k features
tf_F = TfidfVectorizer(max_features=150000, ngram_range=(1,2), sublinear_tf=True, min_df=2, dtype=np.float32)
X_F = tf_F.fit_transform(corpus)
Xr_F_tr = sp.hstack([X_F[:n_train], Xcat_tr])
Xr_F_te = sp.hstack([X_F[n_train:], Xcat_te])
oof_F, pred_F, sc_F = run_ridge_sparse(Xr_F_tr, Xr_F_te, "150k_mega")
results.append({"config": "F_150k_mega", "score": sc_F, "alpha": 1.0})

# ── G: Best Ridge blend — find best ridge from A-F, blend with sep SVD trees ─
print("\n=== G: Best Ridge blend ===")
from experiments.features import (
    build_separate_text_features, build_fold,
    NUM_COLS, LOW_CARD,
)
import lightgbm as lgb
import xgboost as xgblib

X_num_tr = train[NUM_COLS].values.astype(np.float32)
X_num_te = test[NUM_COLS].values.astype(np.float32)
X_low_tr = train[LOW_CARD].values.astype(np.float32)
X_low_te = test[LOW_CARD].values.astype(np.float32)

X_svd_tr_B, X_svd_te_B = build_separate_text_features(
    train, test, title_svd=100, desc_svd=300, skills_svd=50)

LGB_PARAMS = {
    "objective": "regression", "metric": "rmse",
    "n_estimators": 8000, "learning_rate": 0.02,
    "num_leaves": 255, "min_child_samples": 15,
    "feature_fraction": 0.6, "bagging_fraction": 0.8, "bagging_freq": 5,
    "reg_alpha": 0.05, "reg_lambda": 0.1,
    "random_state": SEED, "n_jobs": -1, "verbose": -1,
}
XGB_PARAMS = {
    "objective": "reg:squarederror", "eval_metric": "rmse",
    "n_estimators": 8000, "learning_rate": 0.02,
    "max_depth": 7, "min_child_weight": 10,
    "subsample": 0.8, "colsample_bytree": 0.6,
    "reg_alpha": 0.05, "reg_lambda": 0.1,
    "random_state": SEED, "n_jobs": -1, "tree_method": "hist",
    "early_stopping_rounds": 200, "verbosity": 0,
}

oof_lgb_G = np.zeros(len(train)); pred_lgb_G = np.zeros(len(test))
oof_xgb_G = np.zeros(len(train)); pred_xgb_G = np.zeros(len(test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_svd_tr_B)):
    Xtr, Xval, Xte = build_fold(
        train, test, tr_idx, val_idx, y_log,
        X_svd_tr_B, X_svd_te_B, X_low_tr, X_low_te, X_num_tr, X_num_te)
    m_lgb = lgb.LGBMRegressor(**LGB_PARAMS)
    m_lgb.fit(Xtr, y_log[tr_idx], eval_set=[(Xval, y_log[val_idx])],
              callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(9999)])
    oof_lgb_G[val_idx] = np.expm1(m_lgb.predict(Xval))
    pred_lgb_G += np.expm1(m_lgb.predict(Xte)) / N_SPLITS

    m_xgb = xgblib.XGBRegressor(**XGB_PARAMS)
    m_xgb.fit(Xtr, y_log[tr_idx], eval_set=[(Xval, y_log[val_idx])], verbose=False)
    oof_xgb_G[val_idx] = np.expm1(m_xgb.predict(Xval))
    pred_xgb_G += np.expm1(m_xgb.predict(Xte)) / N_SPLITS
    print(f"  Fold {fold+1}: LGB={mape(y_raw[val_idx], oof_lgb_G[val_idx]):.4f} "
          f"XGB={mape(y_raw[val_idx], oof_xgb_G[val_idx]):.4f}")

lgb_G = mape(y_raw, oof_lgb_G)
xgb_G = mape(y_raw, oof_xgb_G)
print(f"  LGB sep_svd: {lgb_G:.4f}, XGB sep_svd: {xgb_G:.4f}")

# Find best ridge from A-F
best_ridge_score = 1.0
best_ridge_oof = oof_A; best_ridge_pred = pred_A; best_ridge_name = "A"
for oof_r, pred_r, cfg in [(oof_A, pred_A, "A"), (oof_B, pred_B, "B"),
                             (oof_C, pred_C, "C"), (oof_D, pred_D, "D"),
                             (oof_F, pred_F, "F")]:
    sc = mape(y_raw, oof_r)
    if sc < best_ridge_score:
        best_ridge_score = sc
        best_ridge_oof = oof_r
        best_ridge_pred = pred_r
        best_ridge_name = cfg
print(f"  Best ridge: {best_ridge_name} = {best_ridge_score:.4f}")

# Grid search blend
best_blend_score = 1.0; best_blend_w = (0.2, 0.3, 0.5); best_blend_pred = None
for wl in np.arange(0.1, 0.5, 0.05):
    for wx in np.arange(0.1, 0.5, 0.05):
        wr = round(1.0 - wl - wx, 4)
        if wr < 0.2 or wr > 0.7: continue
        blend = wl*oof_lgb_G + wx*oof_xgb_G + wr*best_ridge_oof
        sc = mape(y_raw, blend)
        if sc < best_blend_score:
            best_blend_score = sc
            best_blend_w = (round(wl,3), round(wx,3), round(wr,3))
            best_blend_pred = wl*pred_lgb_G + wx*pred_xgb_G + wr*best_ridge_pred

print(f"  Best blend G: {best_blend_score:.4f}  "
      f"w=(lgb={best_blend_w[0]}, xgb={best_blend_w[1]}, ridge={best_blend_w[2]})")
results.append({"config": f"G_sep_svd+ridge_{best_ridge_name}",
                "score": best_blend_score,
                "alpha": f"lgb={best_blend_w[0]},xgb={best_blend_w[1]},r={best_blend_w[2]}"})

# ── Summary ───────────────────────────────────────────────────────────────────
df_res = pd.DataFrame(results)
df_res.to_csv("experiments/results_exp02.csv", index=False)
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(df_res[["config", "score"]].to_string(index=False))

best = df_res.loc[df_res["score"].idxmin()]
print(f"\nBest: {best['config']}  MAPE={best['score']:.4f}")

if best_blend_pred is not None:
    sub = pd.DataFrame({"id": test["id"], TARGET: best_blend_pred})
    sub.to_csv("submission_exp02_best.csv", index=False)
    print("Saved submission_exp02_best.csv")
