"""
Experiment 03: Push further from exp02 best (OOF 0.2193)
exp02 best: Ridge(150k, alpha=1) + sep_svd trees, w=(0.1, 0.2, 0.7)

Tests:
  A) Ridge alpha grid on 150k (alpha=1.5, 2, 3)
  B) Ridge 200k features
  C) Ridge 300k features
  D) Trees with more SVD: sep(title200+desc500+skills100)
  E) Trees: LGB with more leaves (511) and lower lr (0.01)
  F) Best Ridge + best trees blend (fine-grained weight search)
  G) 4-model blend: Ridge150k + Ridge_sep + LGB + XGB

Run: .venv/bin/python experiments/exp03_push_further.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
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
    df["mega_text"] = df[TITLE_COL] + " " + df[SKILLS_COL] + " " + df[DESC_COL]

n_train = len(train)
corpus  = pd.concat([train["mega_text"], test["mega_text"]], ignore_index=True)

# OHE cat features (shared)
ohe = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
cat_cols = ["experience_name", "professional_roles_name", "schedule_name",
            "employment_name", "specializations_profarea_name", "unified_address_region",
            "unified_address_state", "employer_industries"]
Xcat_tr = ohe.fit_transform(train[cat_cols].astype(str))
Xcat_te = ohe.transform(test[cat_cols].astype(str))
print(f"OHE: {Xcat_tr.shape[1]} features")

results = []


def run_ridge(X_tr, X_te, label, alpha=1.0):
    oof = np.zeros(n_train); pred = np.zeros(len(test))
    for tr_idx, val_idx in kf.split(range(n_train)):
        m = Ridge(alpha=alpha)
        m.fit(X_tr[tr_idx], y_log[tr_idx])
        oof[val_idx] = np.expm1(m.predict(X_tr[val_idx]))
        pred += np.expm1(m.predict(X_te)) / N_SPLITS
    sc = mape(y_raw, oof)
    print(f"  Ridge {label} alpha={alpha}: {sc:.4f}")
    return oof, pred, sc


def run_trees(X_svd_tr, X_svd_te, lgb_params, xgb_params, label):
    oof_l = np.zeros(n_train); pred_l = np.zeros(len(test))
    oof_x = np.zeros(n_train); pred_x = np.zeros(len(test))
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_svd_tr)):
        Xtr, Xval, Xte = build_fold(
            train, test, tr_idx, val_idx, y_log,
            X_svd_tr, X_svd_te, X_low_tr, X_low_te, X_num_tr, X_num_te)
        ml = lgb.LGBMRegressor(**lgb_params)
        ml.fit(Xtr, y_log[tr_idx], eval_set=[(Xval, y_log[val_idx])],
               callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(9999)])
        oof_l[val_idx] = np.expm1(ml.predict(Xval))
        pred_l += np.expm1(ml.predict(Xte)) / N_SPLITS

        mx = xgblib.XGBRegressor(**xgb_params)
        mx.fit(Xtr, y_log[tr_idx], eval_set=[(Xval, y_log[val_idx])], verbose=False)
        oof_x[val_idx] = np.expm1(mx.predict(Xval))
        pred_x += np.expm1(mx.predict(Xte)) / N_SPLITS

        fl = mape(y_raw[val_idx], oof_l[val_idx])
        fx = mape(y_raw[val_idx], oof_x[val_idx])
        print(f"    Fold {fold+1}: LGB={fl:.4f} XGB={fx:.4f}")

    sl = mape(y_raw, oof_l); sx = mape(y_raw, oof_x)
    print(f"  {label}: LGB={sl:.4f} XGB={sx:.4f}")
    return oof_l, pred_l, sl, oof_x, pred_x, sx


def blend3(oof1, oof2, oof3, pred1, pred2, pred3, names):
    best_sc = 1.0; best_w = None; best_p = None
    for w1 in np.arange(0.05, 0.6, 0.05):
        for w2 in np.arange(0.05, 0.6, 0.05):
            w3 = round(1.0 - w1 - w2, 4)
            if w3 < 0.1 or w3 > 0.85: continue
            sc = mape(y_raw, w1*oof1 + w2*oof2 + w3*oof3)
            if sc < best_sc:
                best_sc = sc
                best_w = (round(w1,3), round(w2,3), round(w3,3))
                best_p = w1*pred1 + w2*pred2 + w3*pred3
    print(f"  Blend3 {names}: {best_sc:.4f}  w={best_w}")
    return best_w, best_sc, best_p


# ════════════════════════════════════════════════════════════════════════════════
# A: Ridge alpha grid on 150k
# ════════════════════════════════════════════════════════════════════════════════
print("\n=== A: Ridge 150k alpha grid ===")
tf_150 = TfidfVectorizer(max_features=150000, ngram_range=(1,2), sublinear_tf=True,
                          min_df=2, dtype=np.float32)
X_150 = tf_150.fit_transform(corpus)
Xr_150_tr = sp.hstack([X_150[:n_train], Xcat_tr])
Xr_150_te = sp.hstack([X_150[n_train:], Xcat_te])
print(f"  150k TF-IDF shape: {Xr_150_tr.shape}")

best_alpha_A = 1.0; best_sc_A = 1.0
oof_best_A = None; pred_best_A = None
for alpha in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    oof_a, pred_a, sc_a = run_ridge(Xr_150_tr, Xr_150_te, "150k", alpha)
    results.append({"config": f"A_150k_alpha{alpha}", "score": sc_a})
    if sc_a < best_sc_A:
        best_sc_A = sc_a; best_alpha_A = alpha
        oof_best_A = oof_a; pred_best_A = pred_a
print(f"  Best alpha for 150k: {best_alpha_A} → {best_sc_A:.4f}")

# ════════════════════════════════════════════════════════════════════════════════
# B: Ridge 200k
# ════════════════════════════════════════════════════════════════════════════════
print("\n=== B: Ridge 200k ===")
tf_200 = TfidfVectorizer(max_features=200000, ngram_range=(1,2), sublinear_tf=True,
                          min_df=2, dtype=np.float32)
X_200 = tf_200.fit_transform(corpus)
Xr_200_tr = sp.hstack([X_200[:n_train], Xcat_tr])
Xr_200_te = sp.hstack([X_200[n_train:], Xcat_te])
print(f"  200k TF-IDF shape: {Xr_200_tr.shape}")

best_sc_B = 1.0; oof_best_B = None; pred_best_B = None; best_alpha_B = 1.0
for alpha in [0.5, 1.0, 2.0, 3.0]:
    oof_b, pred_b, sc_b = run_ridge(Xr_200_tr, Xr_200_te, "200k", alpha)
    results.append({"config": f"B_200k_alpha{alpha}", "score": sc_b})
    if sc_b < best_sc_B:
        best_sc_B = sc_b; best_alpha_B = alpha
        oof_best_B = oof_b; pred_best_B = pred_b
print(f"  Best alpha for 200k: {best_alpha_B} → {best_sc_B:.4f}")

# ════════════════════════════════════════════════════════════════════════════════
# C: Trees with larger SVD
# ════════════════════════════════════════════════════════════════════════════════
print("\n=== C: Trees with larger sep SVD (title200+desc500+skills100) ===")
X_svd_tr_C, X_svd_te_C = build_separate_text_features(
    train, test, title_svd=200, desc_svd=500, skills_svd=100,
    title_max=20000, desc_max=50000, skills_max=10000)

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

oof_lgb_C, pred_lgb_C, sl_C, oof_xgb_C, pred_xgb_C, sx_C = run_trees(
    X_svd_tr_C, X_svd_te_C, LGB_BASE, XGB_BASE, "sep800")
results.append({"config": "C_lgb_sep800", "score": sl_C})
results.append({"config": "C_xgb_sep800", "score": sx_C})

# ════════════════════════════════════════════════════════════════════════════════
# D: LGB with more leaves (511) and lower lr (0.01)
# ════════════════════════════════════════════════════════════════════════════════
print("\n=== D: LGB 511 leaves, lr=0.01 (sep SVD 450) ===")
X_svd_tr_D, X_svd_te_D = build_separate_text_features(
    train, test, title_svd=100, desc_svd=300, skills_svd=50)

LGB_DEEP = {**LGB_BASE, "num_leaves": 511, "learning_rate": 0.01, "min_child_samples": 20}
XGB_DEEP = {**XGB_BASE, "max_depth": 8, "learning_rate": 0.01}

oof_lgb_D, pred_lgb_D, sl_D, oof_xgb_D, pred_xgb_D, sx_D = run_trees(
    X_svd_tr_D, X_svd_te_D, LGB_DEEP, XGB_DEEP, "deep_sep450")
results.append({"config": "D_lgb_deep", "score": sl_D})
results.append({"config": "D_xgb_deep", "score": sx_D})

# ════════════════════════════════════════════════════════════════════════════════
# E: Best blend — best Ridge + best trees
# ════════════════════════════════════════════════════════════════════════════════
print("\n=== E: Best blend combinations ===")

# Pick best ridge
best_ridge_oof  = oof_best_A if best_sc_A <= best_sc_B else oof_best_B
best_ridge_pred = pred_best_A if best_sc_A <= best_sc_B else pred_best_B
best_ridge_sc   = min(best_sc_A, best_sc_B)
best_ridge_name = f"150k_a{best_alpha_A}" if best_sc_A <= best_sc_B else f"200k_a{best_alpha_B}"
print(f"  Best ridge: {best_ridge_name} = {best_ridge_sc:.4f}")

# Pick best trees
tree_options = [
    ("lgb_sep800", oof_lgb_C, pred_lgb_C, sl_C),
    ("xgb_sep800", oof_xgb_C, pred_xgb_C, sx_C),
    ("lgb_deep",   oof_lgb_D, pred_lgb_D, sl_D),
    ("xgb_deep",   oof_xgb_D, pred_xgb_D, sx_D),
]
tree_options.sort(key=lambda x: x[3])
print(f"  Tree ranking: {[(t[0], f'{t[3]:.4f}') for t in tree_options]}")

# Try top-2 trees + best ridge
t1_name, t1_oof, t1_pred, t1_sc = tree_options[0]
t2_name, t2_oof, t2_pred, t2_sc = tree_options[1]

w_E1, sc_E1, pred_E1 = blend3(
    t1_oof, t2_oof, best_ridge_oof,
    t1_pred, t2_pred, best_ridge_pred,
    f"({t1_name},{t2_name},{best_ridge_name})")
results.append({"config": f"E_blend3_{t1_name}+{t2_name}+ridge", "score": sc_E1})

# Also try single best tree + best ridge
best_blend_2 = 1.0; best_w_2 = None; best_pred_2 = None
for w_r in np.arange(0.4, 0.9, 0.05):
    w_t = round(1.0 - w_r, 4)
    sc = mape(y_raw, w_t*t1_oof + w_r*best_ridge_oof)
    if sc < best_blend_2:
        best_blend_2 = sc
        best_w_2 = (round(w_t,3), round(w_r,3))
        best_pred_2 = w_t*t1_pred + w_r*best_ridge_pred
print(f"  Blend2 ({t1_name}+{best_ridge_name}): {best_blend_2:.4f}  w={best_w_2}")
results.append({"config": f"E_blend2_{t1_name}+ridge", "score": best_blend_2})

# ════════════════════════════════════════════════════════════════════════════════
# F: 4-model blend
# ════════════════════════════════════════════════════════════════════════════════
print("\n=== F: 4-model blend ===")
# Ridge_150k + Ridge_200k + best_lgb + best_xgb
best_4_sc = 1.0; best_4_w = None; best_4_pred = None
for w1 in np.arange(0.05, 0.5, 0.05):   # ridge_150k
    for w2 in np.arange(0.05, 0.5, 0.05):  # ridge_200k
        for w3 in np.arange(0.05, 0.4, 0.05):  # lgb
            w4 = round(1.0 - w1 - w2 - w3, 4)
            if w4 < 0.05 or w4 > 0.4: continue
            blend = w1*oof_best_A + w2*oof_best_B + w3*t1_oof + w4*t2_oof
            sc = mape(y_raw, blend)
            if sc < best_4_sc:
                best_4_sc = sc
                best_4_w = (round(w1,3), round(w2,3), round(w3,3), round(w4,3))
                best_4_pred = w1*pred_best_A + w2*pred_best_B + w3*t1_pred + w4*t2_pred

print(f"  4-model blend: {best_4_sc:.4f}  w={best_4_w}")
results.append({"config": "F_4model_blend", "score": best_4_sc})

# ════════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════════
df_res = pd.DataFrame(results)
df_res.to_csv("experiments/results_exp03.csv", index=False)
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(df_res[["config", "score"]].to_string(index=False))

best = df_res.loc[df_res["score"].idxmin()]
print(f"\nBest: {best['config']}  MAPE={best['score']:.4f}")

# Save best submission
all_preds = {
    "E_blend3": (sc_E1, pred_E1),
    "E_blend2": (best_blend_2, best_pred_2),
    "F_4model": (best_4_sc, best_4_pred),
}
best_name = min(all_preds, key=lambda k: all_preds[k][0])
best_sc_final, best_pred_final = all_preds[best_name]
sub = pd.DataFrame({"id": test["id"], TARGET: best_pred_final})
sub.to_csv("submission_exp03_best.csv", index=False)
print(f"Saved submission_exp03_best.csv  (config={best_name}, MAPE={best_sc_final:.4f})")
