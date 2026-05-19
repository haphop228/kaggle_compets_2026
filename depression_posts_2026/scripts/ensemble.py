import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

test = pd.read_csv("data/test.csv")
train = pd.read_csv("data/train.csv")
y = train["label"].values

oof_logreg = np.load("models/logreg_v2_oof.npy")
test_logreg = np.load("models/logreg_v2_test.npy")
oof_lgbm = np.load("models/lgbm_oof.npy")
test_lgbm = np.load("models/lgbm_test.npy")

for w_lr in [0.3, 0.4, 0.5, 0.6, 0.7]:
    w_lgbm = 1 - w_lr
    oof_blend = w_lr * oof_logreg + w_lgbm * oof_lgbm
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.2, 0.8, 0.01):
        f1 = f1_score(y, (oof_blend >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"w_lr={w_lr:.1f} w_lgbm={w_lgbm:.1f}: OOF F1={best_f1:.4f} (t={best_t:.2f})")

best_blend_f1 = 0.0
best_w = 0.5
for w_lr in np.arange(0.1, 1.0, 0.05):
    w_lgbm = 1 - w_lr
    oof_blend = w_lr * oof_logreg + w_lgbm * oof_lgbm
    for t in np.arange(0.2, 0.8, 0.01):
        f1 = f1_score(y, (oof_blend >= t).astype(int))
        if f1 > best_blend_f1:
            best_blend_f1 = f1
            best_w = w_lr
            best_t_final = t

print(f"\nBest blend: w_lr={best_w:.2f}, w_lgbm={1-best_w:.2f}, OOF F1={best_blend_f1:.4f}, t={best_t_final:.2f}")

test_blend = best_w * test_logreg + (1 - best_w) * test_lgbm
sub = test[["id"]].copy()
sub["label"] = (test_blend >= best_t_final).astype(int)
sub.to_csv("submissions/ensemble_blend.csv", index=False)
print(f"Saved submissions/ensemble_blend.csv, positive rate: {sub['label'].mean():.3f}")
