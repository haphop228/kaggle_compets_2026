"""
Experiment 01: Improvements over v7 (Kaggle 0.233)
Tests:
  A) v7 baseline reproduced (SVD300 mega + Ridge + LGB + XGB)
  B) Separate SVD (title100 + desc300 + skills50) instead of mega SVD300
  C) Add Ridge OOF as stacking feature for LGB/XGB
  D) CatBoost with native categorical support
  E) Larger SVD (500) on mega_text
  F) More TE cols: professional_roles_name, experience_name+role combo

Run: .venv/bin/python experiments/exp01_improvements.py
Results saved to: experiments/results_exp01.csv
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

from experiments.features import (
    load_data, build_text_features, build_separate_text_features,
    build_tabular_features, build_fold, mape,
    TARGET, SEED, N_SPLITS, STE_K, TE_COLS, EXP_MAP, EXP_ORD,
    DESC_COL, TITLE_COL, SKILLS_COL, LOW_CARD, NUM_COLS, smoothed_te,
)

# ── Setup ─────────────────────────────────────────────────────────────────────
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
train, test = load_data()
y_raw = train[TARGET].values.astype(np.float64)
y_log = np.log1p(y_raw).astype(np.float32)
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

results = []

# ── Shared: tabular features ──────────────────────────────────────────────────
X_num_tr, X_num_te, X_low_tr, X_low_te, oe = build_tabular_features(train, test)

# ── Shared: LGB / XGB params ─────────────────────────────────────────────────
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


def run_lgb(X_svd_tr, X_svd_te, label, extra_te=None):
    oof = np.zeros(len(train)); pred = np.zeros(len(test))
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_svd_tr)):
        Xtr, Xval, Xte = build_fold(
            train, test, tr_idx, val_idx, y_log,
            X_svd_tr, X_svd_te, X_low_tr, X_low_te, X_num_tr, X_num_te,
            extra_te_cols=extra_te,
        )
        m = lgb.LGBMRegressor(**LGB_PARAMS)
        m.fit(Xtr, y_log[tr_idx], eval_set=[(Xval, y_log[val_idx])],
              callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(9999)])
        oof[val_idx] = np.expm1(m.predict(Xval))
        pred += np.expm1(m.predict(Xte)) / N_SPLITS
    score = mape(y_raw, oof)
    print(f"  LGB {label}: {score:.4f}")
    return oof, pred, score


def run_xgb(X_svd_tr, X_svd_te, label, extra_te=None):
    oof = np.zeros(len(train)); pred = np.zeros(len(test))
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_svd_tr)):
        Xtr, Xval, Xte = build_fold(
            train, test, tr_idx, val_idx, y_log,
            X_svd_tr, X_svd_te, X_low_tr, X_low_te, X_num_tr, X_num_te,
            extra_te_cols=extra_te,
        )
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(Xtr, y_log[tr_idx], eval_set=[(Xval, y_log[val_idx])], verbose=False)
        oof[val_idx] = np.expm1(m.predict(Xval))
        pred += np.expm1(m.predict(Xte)) / N_SPLITS
    score = mape(y_raw, oof)
    print(f"  XGB {label}: {score:.4f}")
    return oof, pred, score


def run_ridge(X_tfidf_tr, X_tfidf_te, alpha=1.0):
    ohe = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    cat_cols = ["experience_name", "professional_roles_name", "schedule_name",
                "employment_name", "specializations_profarea_name", "unified_address_region"]
    Xcat_tr = ohe.fit_transform(train[cat_cols].astype(str))
    Xcat_te = ohe.transform(test[cat_cols].astype(str))
    Xr_tr = sp.hstack([X_tfidf_tr, Xcat_tr])
    Xr_te = sp.hstack([X_tfidf_te, Xcat_te])

    oof = np.zeros(len(train)); pred = np.zeros(len(test))
    for tr_idx, val_idx in kf.split(Xr_tr):
        m = Ridge(alpha=alpha)
        m.fit(Xr_tr[tr_idx], y_log[tr_idx])
        oof[val_idx] = np.expm1(m.predict(Xr_tr[val_idx]))
        pred += np.expm1(m.predict(Xr_te)) / N_SPLITS
    score = mape(y_raw, oof)
    print(f"  Ridge(alpha={alpha}): {score:.4f}")
    return oof, pred, score


def best_blend(oofs: dict, preds: dict):
    """Grid search over blend weights, return best (weights, mape, pred)."""
    keys = list(oofs.keys())
    best_m, best_w, best_p = 1.0, None, None
    # 3-model blend
    if len(keys) == 3:
        for w0 in np.arange(0.2, 0.8, 0.05):
            for w1 in np.arange(0.1, 0.7, 0.05):
                w2 = round(1.0 - w0 - w1, 4)
                if w2 < 0 or w2 > 0.5: continue
                blend_oof = w0*oofs[keys[0]] + w1*oofs[keys[1]] + w2*oofs[keys[2]]
                m = mape(y_raw, blend_oof)
                if m < best_m:
                    best_m = m
                    best_w = {keys[0]: round(w0,3), keys[1]: round(w1,3), keys[2]: round(w2,3)}
                    best_p = w0*preds[keys[0]] + w1*preds[keys[1]] + w2*preds[keys[2]]
    elif len(keys) == 2:
        for w0 in np.arange(0.1, 1.0, 0.05):
            w1 = round(1.0 - w0, 4)
            blend_oof = w0*oofs[keys[0]] + w1*oofs[keys[1]]
            m = mape(y_raw, blend_oof)
            if m < best_m:
                best_m = m
                best_w = {keys[0]: round(w0,3), keys[1]: round(w1,3)}
                best_p = w0*preds[keys[0]] + w1*preds[keys[1]]
    return best_w, best_m, best_p


# ════════════════════════════════════════════════════════════════════════════════
# CONFIG A: v7 baseline — mega SVD300 + Ridge + LGB + XGB
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("CONFIG A: v7 baseline (mega SVD300 + Ridge + LGB + XGB)")
print("="*60)
t0 = time.time()

X_svd_tr_A, X_svd_te_A, X_tfidf_tr, X_tfidf_te, tfidf, svd = build_text_features(
    train, test, svd_components=300, max_features=60000)

oof_lgb_A, pred_lgb_A, lgb_A = run_lgb(X_svd_tr_A, X_svd_te_A, "mega300")
oof_xgb_A, pred_xgb_A, xgb_A = run_xgb(X_svd_tr_A, X_svd_te_A, "mega300")
oof_rdg_A, pred_rdg_A, rdg_A  = run_ridge(X_tfidf_tr, X_tfidf_te, alpha=1.0)

w_A, blend_A, pred_A = best_blend(
    {"lgb": oof_lgb_A, "xgb": oof_xgb_A, "ridge": oof_rdg_A},
    {"lgb": pred_lgb_A, "xgb": pred_xgb_A, "ridge": pred_rdg_A},
)
print(f"  Blend A: {blend_A:.4f}  weights={w_A}  time={time.time()-t0:.0f}s")
results.append({"config": "A_v7_baseline", "lgb": lgb_A, "xgb": xgb_A, "ridge": rdg_A,
                "blend": blend_A, "weights": str(w_A), "time_s": round(time.time()-t0)})
pd.DataFrame(results).to_csv("experiments/results_exp01.csv", index=False)


# ════════════════════════════════════════════════════════════════════════════════
# CONFIG B: Separate SVD (title100 + desc300 + skills50) instead of mega SVD300
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("CONFIG B: Separate SVD (title100+desc300+skills50)")
print("="*60)
t0 = time.time()

X_svd_tr_B, X_svd_te_B = build_separate_text_features(
    train, test, title_svd=100, desc_svd=300, skills_svd=50)

oof_lgb_B, pred_lgb_B, lgb_B = run_lgb(X_svd_tr_B, X_svd_te_B, "sep450")
oof_xgb_B, pred_xgb_B, xgb_B = run_xgb(X_svd_tr_B, X_svd_te_B, "sep450")

w_B, blend_B, pred_B = best_blend(
    {"lgb": oof_lgb_B, "xgb": oof_xgb_B, "ridge": oof_rdg_A},
    {"lgb": pred_lgb_B, "xgb": pred_xgb_B, "ridge": pred_rdg_A},
)
print(f"  Blend B: {blend_B:.4f}  weights={w_B}  time={time.time()-t0:.0f}s")
results.append({"config": "B_sep_svd", "lgb": lgb_B, "xgb": xgb_B, "ridge": rdg_A,
                "blend": blend_B, "weights": str(w_B), "time_s": round(time.time()-t0)})
pd.DataFrame(results).to_csv("experiments/results_exp01.csv", index=False)


# ════════════════════════════════════════════════════════════════════════════════
# CONFIG C: Stack Ridge OOF as feature for LGB/XGB
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("CONFIG C: Ridge OOF as stacking feature")
print("="*60)
t0 = time.time()

# Add ridge OOF (log-space) as extra column to SVD matrix
ridge_oof_log = np.log1p(np.clip(oof_rdg_A, 1, None)).astype(np.float32).reshape(-1, 1)
ridge_pred_log = np.log1p(np.clip(pred_rdg_A, 1, None)).astype(np.float32).reshape(-1, 1)

X_svd_tr_C = np.hstack([X_svd_tr_A, ridge_oof_log])
X_svd_te_C = np.hstack([X_svd_te_A, ridge_pred_log])

oof_lgb_C, pred_lgb_C, lgb_C = run_lgb(X_svd_tr_C, X_svd_te_C, "mega300+ridge_stack")
oof_xgb_C, pred_xgb_C, xgb_C = run_xgb(X_svd_tr_C, X_svd_te_C, "mega300+ridge_stack")

w_C, blend_C, pred_C = best_blend(
    {"lgb": oof_lgb_C, "xgb": oof_xgb_C, "ridge": oof_rdg_A},
    {"lgb": pred_lgb_C, "xgb": pred_xgb_C, "ridge": pred_rdg_A},
)
print(f"  Blend C: {blend_C:.4f}  weights={w_C}  time={time.time()-t0:.0f}s")
results.append({"config": "C_ridge_stack", "lgb": lgb_C, "xgb": xgb_C, "ridge": rdg_A,
                "blend": blend_C, "weights": str(w_C), "time_s": round(time.time()-t0)})
pd.DataFrame(results).to_csv("experiments/results_exp01.csv", index=False)


# ════════════════════════════════════════════════════════════════════════════════
# CONFIG D: Larger SVD (500) on mega_text
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("CONFIG D: mega SVD500")
print("="*60)
t0 = time.time()

X_svd_tr_D, X_svd_te_D, _, _, _, _ = build_text_features(
    train, test, svd_components=500, max_features=60000)

oof_lgb_D, pred_lgb_D, lgb_D = run_lgb(X_svd_tr_D, X_svd_te_D, "mega500")
oof_xgb_D, pred_xgb_D, xgb_D = run_xgb(X_svd_tr_D, X_svd_te_D, "mega500")

w_D, blend_D, pred_D = best_blend(
    {"lgb": oof_lgb_D, "xgb": oof_xgb_D, "ridge": oof_rdg_A},
    {"lgb": pred_lgb_D, "xgb": pred_xgb_D, "ridge": pred_rdg_A},
)
print(f"  Blend D: {blend_D:.4f}  weights={w_D}  time={time.time()-t0:.0f}s")
results.append({"config": "D_mega500", "lgb": lgb_D, "xgb": xgb_D, "ridge": rdg_A,
                "blend": blend_D, "weights": str(w_D), "time_s": round(time.time()-t0)})
pd.DataFrame(results).to_csv("experiments/results_exp01.csv", index=False)


# ════════════════════════════════════════════════════════════════════════════════
# CONFIG E: Best SVD + extra TE cols (specializations, experience_name)
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("CONFIG E: mega SVD300 + extra TE (specializations_profarea_name)")
print("="*60)
t0 = time.time()

# specializations_profarea_name already in LOW_CARD as ordinal,
# but adding as TE gives smoothed salary signal
for df in [train, test]:
    df["specializations_profarea_name"] = df["specializations_profarea_name"].astype(str)

extra_te = ["specializations_profarea_name"]
oof_lgb_E, pred_lgb_E, lgb_E = run_lgb(X_svd_tr_A, X_svd_te_A, "mega300+extra_te", extra_te=extra_te)
oof_xgb_E, pred_xgb_E, xgb_E = run_xgb(X_svd_tr_A, X_svd_te_A, "mega300+extra_te", extra_te=extra_te)

w_E, blend_E, pred_E = best_blend(
    {"lgb": oof_lgb_E, "xgb": oof_xgb_E, "ridge": oof_rdg_A},
    {"lgb": pred_lgb_E, "xgb": pred_xgb_E, "ridge": pred_rdg_A},
)
print(f"  Blend E: {blend_E:.4f}  weights={w_E}  time={time.time()-t0:.0f}s")
results.append({"config": "E_extra_te", "lgb": lgb_E, "xgb": xgb_E, "ridge": rdg_A,
                "blend": blend_E, "weights": str(w_E), "time_s": round(time.time()-t0)})
pd.DataFrame(results).to_csv("experiments/results_exp01.csv", index=False)


# ════════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════════
df_res = pd.DataFrame(results)
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(df_res[["config", "lgb", "xgb", "ridge", "blend"]].to_string(index=False))

best_cfg = df_res.loc[df_res["blend"].idxmin()]
print(f"\nBest config: {best_cfg['config']}  blend MAPE={best_cfg['blend']:.4f}")

# Save best submission
best_idx = df_res["blend"].idxmin()
best_pred = [pred_A, pred_B, pred_C, pred_D, pred_E][best_idx]
sub = pd.DataFrame({"id": test["id"], TARGET: best_pred})
sub.to_csv("submission_exp01_best.csv", index=False)
print(f"Saved submission_exp01_best.csv")
