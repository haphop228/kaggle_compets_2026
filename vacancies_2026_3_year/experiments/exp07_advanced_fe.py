"""
Experiment 07: Advanced Feature Engineering (No external data, no huge TF-IDF)
Based on exp03 best: Ridge(200k, alpha=0.5) + XGB_sep800, OOF=0.2169

New Features:
1. RegEx markers from raw_description (senior, lead, intern, manager, etc.)
2. Target Encoding on `name_clean` (with strong smoothing)
3. Hierarchical Target Encoding: `name_clean` + `unified_address_region`

Run: .venv/bin/python experiments/exp07_advanced_fe.py
"""
import sys, os, time, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import xgboost as xgblib
import warnings
warnings.filterwarnings("ignore")

from experiments.features import (
    load_data, build_separate_text_features, mape,
    TARGET, SEED, N_SPLITS, DESC_COL, TITLE_COL, SKILLS_COL,
    EXP_MAP, LOW_CARD, TE_COLS, NUM_COLS, smoothed_te
)

train, test = load_data()
y_raw = train[TARGET].values.astype(np.float64)
y_log = np.log1p(y_raw).astype(np.float32)
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# ── 1. RegEx Feature Extraction ───────────────────────────────────────────────
print("Extracting RegEx features...")
def extract_regex_features(df):
    # Use raw_description for regex matching
    desc = df['raw_description'].fillna('').str.lower()
    
    df['has_senior'] = desc.str.contains(r'\b(senior|сеньор|синьор|ведущий|главный)\b').astype(np.int8)
    df['has_lead'] = desc.str.contains(r'\b(lead|лид|руководитель|директор|начальник)\b').astype(np.int8)
    df['has_junior'] = desc.str.contains(r'\b(junior|джуниор|младший|стажер|помощник|ассистент|без опыта)\b').astype(np.int8)
    df['has_middle'] = desc.str.contains(r'\b(middle|мидл|специалист)\b').astype(np.int8)
    
    # Count bullet points / requirements as a proxy for job complexity
    df['req_count'] = desc.str.count(r'[•\-\*]|\d+\.').astype(np.int32)
    return df

train = extract_regex_features(train)
test = extract_regex_features(test)

# Update NUM_COLS with new features
NEW_NUM_COLS = NUM_COLS + ['has_senior', 'has_lead', 'has_junior', 'has_middle', 'req_count']

# ── 2. Tabular Features (Updated) ─────────────────────────────────────────────
print("Building tabular features...")
for df in [train, test]:
    df["experience_ord"]     = df["experience_name"].map(EXP_MAP).fillna(-1).astype(np.int8)
    df["has_languages"]      = (df["languages_name"].fillna("[]") != "[]").astype(np.int8)
    df["desc_len"]           = df[DESC_COL].str.len().astype(np.int32)
    df["title_len"]          = df[TITLE_COL].str.len().astype(np.int32)
    df["skills_len"]         = df[SKILLS_COL].str.len().astype(np.int32)
    city = df["unified_address_city"].fillna("").str.lower()
    df["is_moscow"]          = city.str.contains("москва").astype(np.int8)
    df["is_spb"]             = city.str.contains("санкт").astype(np.int8)
    df["accept_handicapped"] = df["accept_handicapped"].astype(np.int8)
    df["accept_kids"]        = df["accept_kids"].astype(np.int8)
    
    # Create hierarchical category
    df['name_region'] = df['name_clean'].fillna('') + "_" + df['unified_address_region'].fillna('')
    
    for col in TE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("__NA__").astype(str)

oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
train[LOW_CARD] = oe.fit_transform(train[LOW_CARD].astype(str))
test[LOW_CARD]  = oe.transform(test[LOW_CARD].astype(str))

X_num_tr = train[NEW_NUM_COLS].values.astype(np.float32)
X_num_te = test[NEW_NUM_COLS].values.astype(np.float32)
X_low_tr = train[LOW_CARD].values.astype(np.float32)
X_low_te = test[LOW_CARD].values.astype(np.float32)

# OHE for Ridge
ohe = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
cat_cols_ohe = ["experience_name", "professional_roles_name", "schedule_name",
            "employment_name", "specializations_profarea_name", "unified_address_region",
            "unified_address_state", "employer_industries"]
Xcat_tr = ohe.fit_transform(train[cat_cols_ohe].astype(str))
Xcat_te = ohe.transform(test[cat_cols_ohe].astype(str))

# ── 3. Text Features ──────────────────────────────────────────────────────────
print("Building text features...")
for df in [train, test]:
    df[DESC_COL]    = df[DESC_COL].fillna("")
    df[TITLE_COL]   = df[TITLE_COL].fillna("")
    df[SKILLS_COL]  = df[SKILLS_COL].fillna("")
    df["mega_text"] = df[TITLE_COL] + " " + df[SKILLS_COL] + " " + df[DESC_COL]

n_train = len(train)
corpus  = pd.concat([train["mega_text"], test["mega_text"]], ignore_index=True)

# Ridge TF-IDF (200k)
tf_200 = TfidfVectorizer(max_features=200000, ngram_range=(1,2), sublinear_tf=True, min_df=2, dtype=np.float32)
X_200 = tf_200.fit_transform(corpus)
Xr_200_tr = sp.hstack([X_200[:n_train], Xcat_tr])
Xr_200_te = sp.hstack([X_200[n_train:], Xcat_te])

# XGBoost SVD (sep800)
X_svd_tr, X_svd_te = build_separate_text_features(
    train, test, title_svd=200, desc_svd=500, skills_svd=100,
    title_max=20000, desc_max=50000, skills_max=10000)

# ── 4. Advanced Fold Builder ──────────────────────────────────────────────────
def build_fold_adv(tr_idx, val_idx):
    gm = float(np.mean(y_log[tr_idx]))
    te_tr, te_val, te_te = [], [], []

    # Standard TE
    for col in TE_COLS:
        a, b, c = smoothed_te(train[col].iloc[tr_idx], train[col].iloc[val_idx], test[col], y_log[tr_idx], gm, k=20)
        te_tr.append(a); te_val.append(b); te_te.append(c)

    # Interaction: experience_ord × role TE
    role_tr, role_val, role_te = smoothed_te(
        train["professional_roles_name"].iloc[tr_idx],
        train["professional_roles_name"].iloc[val_idx],
        test["professional_roles_name"], y_log[tr_idx], gm, k=20)
    
    exp_tr  = train["experience_ord"].iloc[tr_idx].values.astype(float)
    exp_val = train["experience_ord"].iloc[val_idx].values.astype(float)
    exp_te  = test["experience_ord"].values.astype(float)
    te_tr.append(exp_tr * role_tr)
    te_val.append(exp_val * role_val)
    te_te.append(exp_te * role_te)
    
    # NEW: TE on name_clean (strong smoothing K=50)
    nc_tr, nc_val, nc_te = smoothed_te(
        train["name_clean"].iloc[tr_idx], train["name_clean"].iloc[val_idx], test["name_clean"], y_log[tr_idx], gm, k=50)
    te_tr.append(nc_tr); te_val.append(nc_val); te_te.append(nc_te)
    
    # NEW: Hierarchical TE (name_clean + region) (strong smoothing K=50)
    nr_tr, nr_val, nr_te = smoothed_te(
        train["name_region"].iloc[tr_idx], train["name_region"].iloc[val_idx], test["name_region"], y_log[tr_idx], gm, k=50)
    te_tr.append(nr_tr); te_val.append(nr_val); te_te.append(nr_te)

    te_tr  = np.column_stack(te_tr)
    te_val = np.column_stack(te_val)
    te_te  = np.column_stack(te_te)

    X_tr  = np.hstack([X_svd_tr[tr_idx],  X_low_tr[tr_idx],  X_num_tr[tr_idx],  te_tr])
    X_val = np.hstack([X_svd_tr[val_idx],  X_low_tr[val_idx], X_num_tr[val_idx], te_val])
    X_te  = np.hstack([X_svd_te,           X_low_te,          X_num_te,          te_te])
    return X_tr, X_val, X_te

# ── 5. Training ───────────────────────────────────────────────────────────────
results = []

# Ridge 200k
print("\nTraining Ridge(200k, alpha=0.5)...")
oof_ridge = np.zeros(n_train); pred_ridge = np.zeros(len(test))
for tr_idx, val_idx in kf.split(range(n_train)):
    m = Ridge(alpha=0.5)
    m.fit(Xr_200_tr[tr_idx], y_log[tr_idx])
    oof_ridge[val_idx] = np.expm1(m.predict(Xr_200_tr[val_idx]))
    pred_ridge += np.expm1(m.predict(Xr_200_te)) / N_SPLITS
sc_ridge = mape(y_raw, oof_ridge)
print(f"  Ridge OOF MAPE: {sc_ridge:.4f}")
results.append({"config": "Ridge_200k_alpha0.5", "score": sc_ridge})

# XGBoost sep800
print("\nTraining XGBoost(sep800) with Advanced FE...")
XGB_PARAMS = {
    "objective": "reg:squarederror", "eval_metric": "rmse",
    "n_estimators": 8000, "learning_rate": 0.02,
    "max_depth": 7, "min_child_weight": 10,
    "subsample": 0.8, "colsample_bytree": 0.6,
    "reg_alpha": 0.05, "reg_lambda": 0.1,
    "random_state": SEED, "n_jobs": -1, "tree_method": "hist",
    "early_stopping_rounds": 200, "verbosity": 0,
}

oof_xgb = np.zeros(n_train); pred_xgb = np.zeros(len(test))
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_svd_tr)):
    Xtr, Xval, Xte = build_fold_adv(tr_idx, val_idx)
    mx = xgblib.XGBRegressor(**XGB_PARAMS)
    mx.fit(Xtr, y_log[tr_idx], eval_set=[(Xval, y_log[val_idx])], verbose=False)
    oof_xgb[val_idx] = np.expm1(mx.predict(Xval))
    pred_xgb += np.expm1(mx.predict(Xte)) / N_SPLITS
    print(f"  Fold {fold+1} XGB MAPE: {mape(y_raw[val_idx], oof_xgb[val_idx]):.4f}")

sc_xgb = mape(y_raw, oof_xgb)
print(f"  XGB OOF MAPE: {sc_xgb:.4f}")
results.append({"config": "XGB_sep800_AdvFE", "score": sc_xgb})

# Blend
print("\nBlending Ridge + XGBoost...")
best_blend_sc = 1.0; best_w = None; best_pred = None
for wr in np.arange(0.1, 0.95, 0.05):
    wx = round(1.0 - wr, 4)
    blend = wr * oof_ridge + wx * oof_xgb
    sc = mape(y_raw, blend)
    if sc < best_blend_sc:
        best_blend_sc = sc
        best_w = (round(wr,3), round(wx,3))
        best_pred = wr * pred_ridge + wx * pred_xgb

print(f"  Best Blend: {best_blend_sc:.4f} w=(ridge={best_w[0]}, xgb={best_w[1]})")
results.append({"config": "Blend_Ridge+XGB_AdvFE", "score": best_blend_sc})

df_res = pd.DataFrame(results)
df_res.to_csv("experiments/results_exp07.csv", index=False)
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(df_res.to_string(index=False))

sub = pd.DataFrame({"id": test["id"], TARGET: best_pred})
sub.to_csv("submission_exp07_best.csv", index=False)
print(f"Saved submission_exp07_best.csv")
