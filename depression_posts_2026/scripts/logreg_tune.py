import pandas as pd
import numpy as np
import os
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

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

all_texts = pd.concat([train["text"], test["text"]])

configs = [
    {"word_max": 150000, "char_max": 100000, "word_ngram": (1,2), "char_ngram": (2,5), "C": 5.0},
    {"word_max": 200000, "char_max": 100000, "word_ngram": (1,3), "char_ngram": (2,5), "C": 5.0},
    {"word_max": 150000, "char_max": 100000, "word_ngram": (1,2), "char_ngram": (2,5), "C": 10.0},
    {"word_max": 150000, "char_max": 100000, "word_ngram": (1,2), "char_ngram": (2,5), "C": 2.0},
]

y = train["label"].values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_overall_f1 = 0.0
best_cfg_idx = 0

for cfg_idx, cfg in enumerate(configs):
    tfidf_word = TfidfVectorizer(
        ngram_range=cfg["word_ngram"], max_features=cfg["word_max"],
        sublinear_tf=True, min_df=2, analyzer="word", token_pattern=r'\b\w+\b',
    )
    tfidf_char = TfidfVectorizer(
        ngram_range=cfg["char_ngram"], max_features=cfg["char_max"],
        sublinear_tf=True, min_df=3, analyzer="char_wb",
    )
    tfidf_word.fit(all_texts)
    tfidf_char.fit(all_texts)

    X_train = hstack([
        tfidf_word.transform(train["text"]),
        tfidf_char.transform(train["text"]),
        X_meta_train
    ]).tocsr()
    X_test_mat = hstack([
        tfidf_word.transform(test["text"]),
        tfidf_char.transform(test["text"]),
        X_meta_test
    ]).tocsr()

    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
        model = LogisticRegression(
            C=cfg["C"], max_iter=5000, solver="liblinear",
            class_weight="balanced", random_state=42,
        )
        model.fit(X_train[tr_idx], y[tr_idx])
        oof_preds[val_idx] = model.predict_proba(X_train[val_idx])[:, 1]
        test_preds += model.predict_proba(X_test_mat)[:, 1] / 5

    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.2, 0.8, 0.01):
        f1 = f1_score(y, (oof_preds >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t

    print(f"Config {cfg_idx}: word_ngram={cfg['word_ngram']}, C={cfg['C']}, OOF F1={best_f1:.4f} (t={best_t:.2f})")

    if best_f1 > best_overall_f1:
        best_overall_f1 = best_f1
        best_cfg_idx = cfg_idx
        np.save("models/logreg_best_oof.npy", oof_preds)
        np.save("models/logreg_best_test.npy", test_preds)
        best_threshold = best_t

print(f"\nBest config: {cfg_idx}, F1={best_overall_f1:.4f}")

sub = test[["id"]].copy()
sub["label"] = (np.load("models/logreg_best_test.npy") >= best_threshold).astype(int)
sub.to_csv("submissions/logreg_best.csv", index=False)
print(f"Saved submissions/logreg_best.csv, positive rate: {sub['label'].mean():.3f}")
