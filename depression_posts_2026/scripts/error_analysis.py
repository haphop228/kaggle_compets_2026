import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix

train = pd.read_csv("data/train.csv")
train["body"] = train["body"].fillna("")
train["text"] = train["title"].astype(str) + " [SEP] " + train["body"].astype(str)

y = train["label"].values

oof_logreg = np.load("models/logreg_v2_oof.npy")

best_t, best_f1 = 0.5, 0.0
for t in np.arange(0.2, 0.8, 0.01):
    f1 = f1_score(y, (oof_logreg >= t).astype(int))
    if f1 > best_f1:
        best_f1, best_t = f1, t

preds = (oof_logreg >= best_t).astype(int)

print(f"Best threshold: {best_t:.2f}, OOF F1: {best_f1:.4f}")
print(f"\nClassification Report:\n{classification_report(y, preds, target_names=['not_dep', 'dep'])}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y, preds)}")

train["pred"] = preds
train["prob"] = oof_logreg
train["correct"] = (train["pred"] == train["label"]).astype(int)

fp = train[(train["pred"] == 1) & (train["label"] == 0)].copy()
fn = train[(train["pred"] == 0) & (train["label"] == 1)].copy()

print(f"\nFalse Positives: {len(fp)}")
print(f"False Negatives: {len(fn)}")

print("\nTop False Positives (predicted dep, actually not):")
fp_sorted = fp.sort_values("prob", ascending=False).head(10)
for _, row in fp_sorted.iterrows():
    print(f"  prob={row['prob']:.3f} | title: {row['title'][:80]}")
    if row["body"]:
        print(f"    body: {str(row['body'])[:100]}")

print("\nTop False Negatives (predicted not dep, actually dep):")
fn_sorted = fn.sort_values("prob", ascending=True).head(10)
for _, row in fn_sorted.iterrows():
    print(f"  prob={row['prob']:.3f} | title: {row['title'][:80]}")
    if row["body"]:
        print(f"    body: {str(row['body'])[:100]}")

print("\nError analysis by text length:")
train["text_len"] = train["text"].str.len()
train["len_bucket"] = pd.cut(train["text_len"], bins=[0, 100, 300, 1000, 5000, 100000], labels=["tiny", "short", "medium", "long", "very_long"])
err_by_len = train.groupby("len_bucket").apply(
    lambda g: pd.Series({
        "count": len(g),
        "error_rate": 1 - g["correct"].mean(),
        "fp_rate": ((g["pred"] == 1) & (g["label"] == 0)).mean(),
        "fn_rate": ((g["pred"] == 0) & (g["label"] == 1)).mean(),
    })
)
print(err_by_len)

print("\nError analysis by body_empty:")
train["body_empty"] = (train["body"].str.strip() == "").astype(int)
err_by_body = train.groupby("body_empty").apply(
    lambda g: pd.Series({
        "count": len(g),
        "error_rate": 1 - g["correct"].mean(),
        "fp_rate": ((g["pred"] == 1) & (g["label"] == 0)).mean(),
        "fn_rate": ((g["pred"] == 0) & (g["label"] == 1)).mean(),
    })
)
print(err_by_body)
