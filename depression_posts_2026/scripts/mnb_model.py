import pandas as pd
import numpy as np
import os
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.calibration import CalibratedClassifierCV

os.makedirs("models", exist_ok=True)

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

train["body"] = train["body"].fillna("")
test["body"] = test["body"].fillna("")

train["text"] = train["title"].astype(str) + " [SEP] " + train["body"].astype(str)
test["text"] = test["title"].astype(str) + " [SEP] " + test["body"].astype(str)

cv_word = CountVectorizer(
    ngram_range=(1, 2),
    max_features=150000,
    min_df=2,
    analyzer="word",
    token_pattern=r'\b\w+\b',
)
cv_char = CountVectorizer(
    ngram_range=(2, 5),
    max_features=100000,
    min_df=3,
    analyzer="char_wb",
)

all_texts = pd.concat([train["text"], test["text"]])
cv_word.fit(all_texts)
cv_char.fit(all_texts)

X_train = hstack([
    cv_word.transform(train["text"]),
    cv_char.transform(train["text"]),
]).tocsr()
X_test = hstack([
    cv_word.transform(test["text"]),
    cv_char.transform(test["text"]),
]).tocsr()

y = train["label"].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model_cls, alpha in [
    ("ComplementNB_0.1", ComplementNB, 0.1),
    ("ComplementNB_0.5", ComplementNB, 0.5),
    ("MultinomialNB_0.1", MultinomialNB, 0.1),
]:
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
        model = model_cls(alpha=alpha)
        model.fit(X_train[tr_idx], y[tr_idx])
        oof_preds[val_idx] = model.predict_proba(X_train[val_idx])[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / 5

    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.2, 0.8, 0.01):
        f1 = f1_score(y, (oof_preds >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"{model_name}: OOF F1={best_f1:.4f} (t={best_t:.2f})")

    if model_name == "ComplementNB_0.1":
        np.save("models/cnb_oof.npy", oof_preds)
        np.save("models/cnb_test.npy", test_preds)
