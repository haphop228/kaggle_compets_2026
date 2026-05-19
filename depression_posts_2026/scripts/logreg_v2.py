import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import re

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

def meta_features(df):
    body = df["body"].astype(str)
    text = df["text"].astype(str)
    title = df["title"].astype(str)
    body_len = body.str.len()
    title_len = title.str.len()
    feats = pd.DataFrame({
        "body_len": body_len,
        "word_count": text.str.split().str.len(),
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
        "kw_count": text.str.lower().apply(
            lambda x: sum(1 for kw in DEPRESSION_KEYWORDS if kw in x)
        ),
    })
    return csr_matrix(feats.values.astype(float))

X_meta_train = meta_features(train)
X_meta_test = meta_features(test)

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

all_texts = pd.concat([train["text"], test["text"]])
tfidf_word.fit(all_texts)
tfidf_char.fit(all_texts)

X_word_train = tfidf_word.transform(train["text"])
X_char_train = tfidf_char.transform(train["text"])
X_word_test = tfidf_word.transform(test["text"])
X_char_test = tfidf_char.transform(test["text"])

X_train = hstack([X_word_train, X_char_train, X_meta_train]).tocsr()
X_test = hstack([X_word_test, X_char_test, X_meta_test]).tocsr()

y = train["label"].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model = LogisticRegression(
        C=5.0, max_iter=5000, solver="saga",
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_tr, y_tr)

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / 5
    fold_f1 = f1_score(y_val, (oof_preds[val_idx] >= 0.5).astype(int))
    print(f"Fold {fold+1}: F1={fold_f1:.4f}")

oof_f1 = f1_score(y, (oof_preds >= 0.5).astype(int))
print(f"\nOOF F1 (threshold=0.5): {oof_f1:.4f}")

best_t, best_f1 = 0.5, 0.0
for t in np.arange(0.2, 0.8, 0.01):
    f1 = f1_score(y, (oof_preds >= t).astype(int))
    if f1 > best_f1:
        best_f1, best_t = f1, t
print(f"OOF F1 (best threshold={best_t:.2f}): {best_f1:.4f}")

np.save("models/logreg_v2_oof.npy", oof_preds)
np.save("models/logreg_v2_test.npy", test_preds)

sub = test[["id"]].copy()
sub["label"] = (test_preds >= best_t).astype(int)
sub.to_csv("submissions/logreg_v2.csv", index=False)
print(f"Saved submissions/logreg_v2.csv, positive rate: {sub['label'].mean():.3f}")
