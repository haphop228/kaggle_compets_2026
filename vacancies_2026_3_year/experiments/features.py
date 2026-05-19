"""
Shared feature engineering for all experiments.
Usage:
    from features import load_data, build_text_features, build_tabular_features, smoothed_te, build_fold
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OrdinalEncoder

# ── Constants ─────────────────────────────────────────────────────────────────
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


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    train = pd.read_csv("data/train.csv")
    test  = pd.read_csv("data/test_x.csv")
    print(f"train: {train.shape}, test: {test.shape}")
    return train, test


# ── Text features ─────────────────────────────────────────────────────────────
def build_text_features(train, test, svd_components=300, max_features=60000):
    """
    Build TF-IDF on mega_text (title+skills+desc) and reduce with SVD.
    Returns (X_svd_tr, X_svd_te, X_tfidf_tr, X_tfidf_te, tfidf, svd).
    """
    n_train = len(train)
    for df in [train, test]:
        df[DESC_COL]    = df[DESC_COL].fillna("")
        df[TITLE_COL]   = df[TITLE_COL].fillna("")
        df[SKILLS_COL]  = df[SKILLS_COL].fillna("")
        df["mega_text"] = df[TITLE_COL] + " " + df[SKILLS_COL] + " " + df[DESC_COL]

    corpus = pd.concat([train["mega_text"], test["mega_text"]], ignore_index=True)

    print(f"Building TF-IDF (max_features={max_features})...")
    tfidf = TfidfVectorizer(
        max_features=max_features, ngram_range=(1, 2),
        sublinear_tf=True, min_df=2, dtype=np.float32,
    )
    X_tfidf = tfidf.fit_transform(corpus)
    print(f"  TF-IDF shape: {X_tfidf.shape}")

    print(f"Building SVD({svd_components})...")
    svd = TruncatedSVD(n_components=svd_components, random_state=SEED)
    X_svd_all = svd.fit_transform(X_tfidf).astype(np.float32)
    print(f"  SVD explained variance: {svd.explained_variance_ratio_.sum():.3f}")

    X_svd_tr = X_svd_all[:n_train]
    X_svd_te = X_svd_all[n_train:]
    X_tfidf_tr = X_tfidf[:n_train]
    X_tfidf_te = X_tfidf[n_train:]
    return X_svd_tr, X_svd_te, X_tfidf_tr, X_tfidf_te, tfidf, svd


def build_separate_text_features(train, test,
                                  title_svd=100, desc_svd=300, skills_svd=50,
                                  title_max=20000, desc_max=50000, skills_max=10000):
    """
    Build separate TF-IDF+SVD for title, description, and skills.
    Returns (X_sep_tr, X_sep_te).
    """
    n_train = len(train)
    for df in [train, test]:
        df[DESC_COL]   = df[DESC_COL].fillna("")
        df[TITLE_COL]  = df[TITLE_COL].fillna("")
        df[SKILLS_COL] = df[SKILLS_COL].fillna("")

    def fit_svd(corpus, max_f, n_svd, name):
        tf = TfidfVectorizer(max_features=max_f, ngram_range=(1, 2),
                             sublinear_tf=True, min_df=2, dtype=np.float32)
        X = tf.fit_transform(corpus)
        sv = TruncatedSVD(n_components=n_svd, random_state=SEED)
        Xr = sv.fit_transform(X).astype(np.float32)
        print(f"  {name}: TF-IDF{X.shape} → SVD({n_svd}), var={sv.explained_variance_ratio_.sum():.3f}")
        return Xr

    corpus_t = pd.concat([train[TITLE_COL],  test[TITLE_COL]],  ignore_index=True)
    corpus_d = pd.concat([train[DESC_COL],   test[DESC_COL]],   ignore_index=True)
    corpus_s = pd.concat([train[SKILLS_COL], test[SKILLS_COL]], ignore_index=True)

    Xt = fit_svd(corpus_t, title_max,  title_svd,  "title")
    Xd = fit_svd(corpus_d, desc_max,   desc_svd,   "desc")
    Xs = fit_svd(corpus_s, skills_max, skills_svd, "skills")

    X_sep = np.hstack([Xt, Xd, Xs])
    return X_sep[:n_train], X_sep[n_train:]


# ── Tabular features ──────────────────────────────────────────────────────────
def build_tabular_features(train, test):
    """
    Build numeric + low-cardinality encoded features.
    Returns (X_num_tr, X_num_te, X_low_tr, X_low_te, oe).
    """
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
        for col in TE_COLS:
            if col in df.columns:
                df[col] = df[col].fillna("__NA__").astype(str)

    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
    train[LOW_CARD] = oe.fit_transform(train[LOW_CARD].astype(str))
    test[LOW_CARD]  = oe.transform(test[LOW_CARD].astype(str))

    X_num_tr = train[NUM_COLS].values.astype(np.float32)
    X_num_te = test[NUM_COLS].values.astype(np.float32)
    X_low_tr = train[LOW_CARD].values.astype(np.float32)
    X_low_te = test[LOW_CARD].values.astype(np.float32)

    print(f"Tabular: num={X_num_tr.shape[1]}, low_card={X_low_tr.shape[1]}")
    return X_num_tr, X_num_te, X_low_tr, X_low_te, oe


# ── Smoothed TE ───────────────────────────────────────────────────────────────
def smoothed_te(col_tr, col_val, col_te, y_tr, gm, k=STE_K):
    """Smoothed target encoding: (n*mean + k*global) / (n+k)."""
    stats = (
        pd.DataFrame({"y": y_tr, "cat": col_tr.values})
        .groupby("cat")["y"]
        .agg(["mean", "count"])
    )
    stats["smooth"] = (stats["count"] * stats["mean"] + k * gm) / (stats["count"] + k)
    te_map = stats["smooth"]
    return (
        col_tr.map(te_map).fillna(gm).values,
        col_val.map(te_map).fillna(gm).values,
        col_te.map(te_map).fillna(gm).values,
    )


def build_fold(train, test, tr_idx, val_idx, y_log,
               X_svd_tr, X_svd_te, X_low_tr, X_low_te, X_num_tr, X_num_te,
               extra_te_cols=None):
    """
    Assemble (X_tr, X_val, X_te) for one CV fold.
    Applies smoothed TE for TE_COLS + professional_roles_name inside fold.
    extra_te_cols: additional column names for smoothed TE.
    """
    gm = float(np.mean(y_log[tr_idx]))
    te_tr, te_val, te_te = [], [], []

    all_te = TE_COLS + (extra_te_cols or [])
    for col in all_te:
        a, b, c = smoothed_te(
            train[col].iloc[tr_idx], train[col].iloc[val_idx],
            test[col], y_log[tr_idx], gm,
        )
        te_tr.append(a); te_val.append(b); te_te.append(c)

    # Interaction: experience_ord × role TE
    role_tr, role_val, role_te = smoothed_te(
        train["professional_roles_name"].iloc[tr_idx],
        train["professional_roles_name"].iloc[val_idx],
        test["professional_roles_name"], y_log[tr_idx], gm,
    )
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


# ── MAPE metric ───────────────────────────────────────────────────────────────
def mape(y_true, y_pred):
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))))
