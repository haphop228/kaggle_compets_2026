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

def meta_features(df):
    body = df["body"].astype(str)
    text = df["text"].astype(str)
    feats = pd.DataFrame({
        "body_len": body.str.len(),
        "word_count": text.str.split().str.len(),
        "sent_count": text.str.count(r'[.!?]+'),
        "body_empty": (body.str.strip() == "").astype(int),
        "upper_ratio": text.apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)),
        "question_count": text.str.count(r'\?'),
        "excl_count": text.str.count(r'!'),
        "title_len": df["title"].astype(str).str.len(),
        "avg_word_len": text.apply(lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0),
    })
    return csr_matrix(feats.values.astype(float))

X_meta_train = meta_features(train)
X_meta_test = meta_features(test)

tfidf_word = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=100000,
    sublinear_tf=True,
    min_df=2,
    analyzer="word",
    token_pattern=r'\b\w+\b',
)
tfidf_char = TfidfVectorizer(
    ngram_range=(2, 4),
    max_features=50000,
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

X_train = hstack([X_word_train, X_char_train, X_meta_train])
X_test = hstack([X_word_test, X_char_test, X_meta_test])

y = train["label"].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model = LogisticRegression(C=5.0, max_iter=1000, solver="lbfgs", class_weight="balanced")
    model.fit(X_tr, y_tr)

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / 5

threshold = 0.5
oof_f1 = f1_score(y, (oof_preds >= threshold).astype(int))
print(f"OOF F1 (threshold=0.5): {oof_f1:.4f}")

best_t, best_f1 = 0.5, 0.0
for t in np.arange(0.2, 0.8, 0.01):
    f1 = f1_score(y, (oof_preds >= t).astype(int))
    if f1 > best_f1:
        best_f1, best_t = f1, t
print(f"OOF F1 (best threshold={best_t:.2f}): {best_f1:.4f}")

sub = test[["id"]].copy()
sub["label"] = (test_preds >= best_t).astype(int)
sub.to_csv("submissions/baseline_logreg.csv", index=False)
print(f"Saved submissions/baseline_logreg.csv, positive rate: {sub['label'].mean():.3f}")
