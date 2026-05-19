import pandas as pd
import numpy as np
import os
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

os.makedirs("models", exist_ok=True)

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

train["body"] = train["body"].fillna("")
test["body"] = test["body"].fillna("")

train["text"] = train["title"].astype(str) + " [SEP] " + train["body"].astype(str)
test["text"] = test["title"].astype(str) + " [SEP] " + test["body"].astype(str)

DEPRESSION_KEYWORDS = [
    "kill myself", "killing myself", "hate myself", "suicidal", "suicide",
    "want to die", "wish i was dead", "end my life", "no reason to live",
    "worthless", "hopeless", "depressed", "depression", "self harm",
    "cutting myself", "overdose",
]

POSITIVE_KEYWORDS = [
    "happy", "excited", "love", "great", "awesome", "amazing", "fun",
    "good", "wonderful", "fantastic", "enjoy", "laugh", "smile",
]

def meta_features(df):
    body = df["body"].astype(str)
    text = df["text"].astype(str)
    title = df["title"].astype(str)
    body_len = body.str.len()
    title_len = title.str.len()
    word_count = text.str.split().str.len().clip(lower=1)
    kw_count = text.str.lower().apply(
        lambda x: sum(1 for kw in DEPRESSION_KEYWORDS if kw in x)
    )
    pos_kw_count = text.str.lower().apply(
        lambda x: sum(1 for kw in POSITIVE_KEYWORDS if kw in x)
    )
    feats = pd.DataFrame({
        "body_len": body_len,
        "word_count": word_count,
        "sent_count": text.str.count(r'[.!?]+'),
        "body_empty": (body.str.strip() == "").astype(int),
        "upper_ratio": text.apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)),
        "question_count": text.str.count(r'\?'),
        "excl_count": text.str.count(r'!'),
        "title_len": title_len,
        "avg_word_len": text.apply(lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0),
        "body_title_ratio": body_len / (title_len + 1),
        "ellipsis_count": text.str.count(r'\.\.\.'),
        "newline_count": text.str.count(r'\n'),
        "unique_word_ratio": text.apply(
            lambda x: len(set(x.lower().split())) / max(len(x.split()), 1)
        ),
        "kw_count": kw_count,
        "kw_density": kw_count / word_count,
        "pos_kw_count": pos_kw_count,
        "kw_net": kw_count - pos_kw_count,
        "log_body_len": np.log1p(body_len),
        "log_word_count": np.log1p(word_count),
    })
    return feats.values.astype(float)

tfidf_word = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=150000,
    sublinear_tf=True,
    min_df=2,
    analyzer="word",
    token_pattern=r'\b\w+\b',
)
tfidf_char = TfidfVectorizer(
    ngram_range=(2, 5),
    max_features=100000,
    sublinear_tf=True,
    min_df=3,
    analyzer="char_wb",
)

tfidf_word.fit(train["text"])
tfidf_char.fit(train["text"])

meta_train_raw = meta_features(train)
meta_test_raw = meta_features(test)

scaler = StandardScaler()
meta_train_scaled = scaler.fit_transform(meta_train_raw)
meta_test_scaled = scaler.transform(meta_test_raw)

X_train = hstack([
    tfidf_word.transform(train["text"]),
    tfidf_char.transform(train["text"]),
    csr_matrix(meta_train_scaled),
]).tocsr()
X_test = hstack([
    tfidf_word.transform(test["text"]),
    tfidf_char.transform(test["text"]),
    csr_matrix(meta_test_scaled),
]).tocsr()

y = train["label"].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for alpha in [1e-4, 5e-5, 1e-5]:
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
        model = SGDClassifier(
            loss="modified_huber",
            alpha=alpha,
            max_iter=1000,
            tol=1e-4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train[tr_idx], y[tr_idx])
        oof_preds[val_idx] = model.predict_proba(X_train[val_idx])[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / 5

    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.2, 0.8, 0.01):
        f1 = f1_score(y, (oof_preds >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"SGD alpha={alpha}: OOF F1={best_f1:.4f} (t={best_t:.2f})")

    if alpha == 5e-5:
        np.save("models/sgd_oof.npy", oof_preds)
        np.save("models/sgd_test.npy", test_preds)
        sub = test[["id"]].copy()
        sub["label"] = (test_preds >= best_t).astype(int)
        sub.to_csv("submissions/sgd_model.csv", index=False)
        print(f"Saved submissions/sgd_model.csv, positive rate: {sub['label'].mean():.3f}")
