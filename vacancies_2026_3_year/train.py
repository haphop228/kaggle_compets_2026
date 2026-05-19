import sys, os, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import xgboost as xgblib
import warnings
warnings.filterwarnings("ignore")

TARGET     = "salary_mean_net"
SEED       = 42
N_SPLITS   = 5
STE_K      = 20

DESC_COL   = "lemmaized_wo_stopwords_raw_description"
TITLE_COL  = "name_clean"
SKILLS_COL = "key_skills_name"

EXP_ORD = ["Нет опыта", "От 1 года до 3 лет", "От 3 до 6 лет", "Более 6 лет"]
EXP_MAP = {v: i for i, v in enumerate(EXP_ORD)}

LOW_CARD = [
    "schedule_name", "employment_name", "specializations_profarea_name",
    "professional_roles_name", "unified_address_region", "unified_address_state",
    "if_foreign_language", "is_branded_description",
]
TE_COLS = ["employer_id", "employer_name", "unified_address_city", "employer_industries"]
NUM_COLS = [
    "experience_ord", "has_languages", "is_moscow", "is_spb",
    "desc_len", "title_len", "skills_len", "accept_handicapped", "accept_kids",
]

def mape(y_true, y_pred):
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))))

def smoothed_te(col_tr, col_val, col_te, y_tr, gm, k=STE_K):
    stats = pd.DataFrame({"y": y_tr, "cat": col_tr.values}).groupby("cat")["y"].agg(["mean", "count"])
    stats["smooth"] = (stats["count"] * stats["mean"] + k * gm) / (stats["count"] + k)
    te_map = stats["smooth"]
    return (
        col_tr.map(te_map).fillna(gm).values,
        col_val.map(te_map).fillna(gm).values,
        col_te.map(te_map).fillna(gm).values,
    )

train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test_x.csv")

y_raw = train[TARGET].values.astype(np.float64)
y_log = np.log1p(y_raw).astype(np.float32)
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

for df in [train, test]:
    df[DESC_COL]    = df[DESC_COL].fillna("")
    df[TITLE_COL]   = df[TITLE_COL].fillna("")
    df[SKILLS_COL]  = df[SKILLS_COL].fillna("")
    df["mega_text"] = df[TITLE_COL] + " " + df[SKILLS_COL] + " " + df[DESC_COL]
    
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
    
    for col in TE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("__NA__").astype(str)

n_train = len(train)

corpus = pd.concat([train["mega_text"], test["mega_text"]], ignore_index=True)

tf_200 = TfidfVectorizer(max_features=200000, ngram_range=(1,2), sublinear_tf=True, min_df=2, dtype=np.float32)
X_tfidf_200 = tf_200.fit_transform(corpus)

ohe = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
cat_cols_ohe = ["experience_name", "professional_roles_name", "schedule_name",
            "employment_name", "specializations_profarea_name", "unified_address_region",
            "unified_address_state", "employer_industries"]
Xcat_tr = ohe.fit_transform(train[cat_cols_ohe].astype(str))
Xcat_te = ohe.transform(test[cat_cols_ohe].astype(str))

Xr_200_tr = sp.hstack([X_tfidf_200[:n_train], Xcat_tr])
Xr_200_te = sp.hstack([X_tfidf_200[n_train:], Xcat_te])

corpus_t = pd.concat([train[TITLE_COL],  test[TITLE_COL]],  ignore_index=True)
corpus_d = pd.concat([train[DESC_COL],   test[DESC_COL]],   ignore_index=True)
corpus_s = pd.concat([train[SKILLS_COL], test[SKILLS_COL]], ignore_index=True)

def fit_svd(corpus, max_f, n_svd):
    tf = TfidfVectorizer(max_features=max_f, ngram_range=(1, 2), sublinear_tf=True, min_df=2, dtype=np.float32)
    X = tf.fit_transform(corpus)
    sv = TruncatedSVD(n_components=n_svd, random_state=SEED)
    return sv.fit_transform(X).astype(np.float32)

Xt = fit_svd(corpus_t, 20000, 200)
Xd = fit_svd(corpus_d, 50000, 500)
Xs = fit_svd(corpus_s, 10000, 100)

X_sep = np.hstack([Xt, Xd, Xs])
X_svd_tr = X_sep[:n_train]
X_svd_te = X_sep[n_train:]

oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
train[LOW_CARD] = oe.fit_transform(train[LOW_CARD].astype(str))
test[LOW_CARD]  = oe.transform(test[LOW_CARD].astype(str))

X_num_tr = train[NUM_COLS].values.astype(np.float32)
X_num_te = test[NUM_COLS].values.astype(np.float32)
X_low_tr = train[LOW_CARD].values.astype(np.float32)
X_low_te = test[LOW_CARD].values.astype(np.float32)

def build_fold_xgb(tr_idx, val_idx):
    gm = float(np.mean(y_log[tr_idx]))
    te_tr, te_val, te_te = [], [], []

    for col in TE_COLS:
        a, b, c = smoothed_te(train[col].iloc[tr_idx], train[col].iloc[val_idx], test[col], y_log[tr_idx], gm)
        te_tr.append(a); te_val.append(b); te_te.append(c)

    role_tr, role_val, role_te = smoothed_te(
        train["professional_roles_name"].iloc[tr_idx],
        train["professional_roles_name"].iloc[val_idx],
        test["professional_roles_name"], y_log[tr_idx], gm)
    
    exp_tr  = train["experience_ord"].iloc[tr_idx].values.astype(float)
    exp_val = train["experience_ord"].iloc[val_idx].values.astype(float)
    exp_te  = test["experience_ord"].values.astype(float)
    
    te_tr.append(exp_tr * role_tr)
    te_val.append(exp_val * role_val)
    te_te.append(exp_te * role_te)

    te_tr  = np.column_stack(te_tr)
    te_val = np.column_stack(te_val)
    te_te  = np.column_stack(te_te)

    X_tr  = np.hstack([X_svd_tr[tr_idx],  X_low_tr[tr_idx],  X_num_tr[tr_idx],  te_tr])
    X_val = np.hstack([X_svd_tr[val_idx],  X_low_tr[val_idx], X_num_tr[val_idx], te_val])
    X_te  = np.hstack([X_svd_te,           X_low_te,          X_num_te,          te_te])
    return X_tr, X_val, X_te

oof_ridge = np.zeros(n_train); pred_ridge = np.zeros(len(test))
for tr_idx, val_idx in kf.split(range(n_train)):
    m = Ridge(alpha=0.5)
    m.fit(Xr_200_tr[tr_idx], y_log[tr_idx])
    oof_ridge[val_idx] = np.expm1(m.predict(Xr_200_tr[val_idx]))
    pred_ridge += np.expm1(m.predict(Xr_200_te)) / N_SPLITS
print(f"  Ridge OOF MAPE: {mape(y_raw, oof_ridge):.4f}")

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
    Xtr, Xval, Xte = build_fold_xgb(tr_idx, val_idx)
    mx = xgblib.XGBRegressor(**XGB_PARAMS)
    mx.fit(Xtr, y_log[tr_idx], eval_set=[(Xval, y_log[val_idx])], verbose=False)
    oof_xgb[val_idx] = np.expm1(mx.predict(Xval))
    pred_xgb += np.expm1(mx.predict(Xte)) / N_SPLITS
    print(f"  Fold {fold+1} XGB MAPE: {mape(y_raw[val_idx], oof_xgb[val_idx]):.4f}")
print(f"  XGB OOF MAPE: {mape(y_raw, oof_xgb):.4f}")

w_ridge = 0.75
w_xgb = 0.25

oof_blend = w_ridge * oof_ridge + w_xgb * oof_xgb
pred_blend = w_ridge * pred_ridge + w_xgb * pred_xgb

final_mape = mape(y_raw, oof_blend)
print(f"Final Blend OOF MAPE: {final_mape:.4f}")

pred_blend_rounded = np.round(pred_blend / 100) * 100

sub = pd.DataFrame({"id": test["id"], TARGET: pred_blend_rounded})
sub.to_csv("submission_final.csv", index=False)
print(f"\nSaved submission_final.csv")
print(sub[TARGET].describe())
