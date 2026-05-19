import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from itertools import product

test = pd.read_csv("data/test.csv")
train = pd.read_csv("data/train.csv")
y = train["label"].values

models = {
    "logreg_v3":      ("models/logreg_v3_oof.npy",       "models/logreg_v3_test.npy"),
    "logreg_v4":      ("models/logreg_v4_oof.npy",       "models/logreg_v4_test.npy"),
    "logreg_v5":      ("models/logreg_v5_oof.npy",       "models/logreg_v5_test.npy"),
    "logreg_v5_tune": ("models/logreg_v5_tune_oof.npy",  "models/logreg_v5_tune_test.npy"),
    "logreg_v6":      ("models/logreg_v6_oof.npy",       "models/logreg_v6_test.npy"),
    "lgbm":           ("models/lgbm_oof.npy",            "models/lgbm_test.npy"),
    "cnb":            ("models/cnb_oof.npy",             "models/cnb_test.npy"),
    "sgd":            ("models/sgd_oof.npy",             "models/sgd_test.npy"),
}

oof = {}
test_preds = {}
available = []

import os

for name, (oof_path, test_path) in models.items():
    if os.path.exists(oof_path) and os.path.exists(test_path):
        oof[name] = np.load(oof_path)
        test_preds[name] = np.load(test_path)
        available.append(name)
        single_best_t, single_best_f1 = 0.5, 0.0
        for t in np.arange(0.2, 0.8, 0.01):
            f1 = f1_score(y, (oof[name] >= t).astype(int))
            if f1 > single_best_f1:
                single_best_f1, single_best_t = f1, t
        print(f"{name}: OOF F1={single_best_f1:.4f} (t={single_best_t:.2f})")

print(f"\navailable models: {available}")

best_pair_f1 = 0.0
best_pair_config = None
for i, m1 in enumerate(available):
    for m2 in available[i+1:]:
        for w1 in np.arange(0.1, 1.0, 0.1):
            w2 = 1 - w1
            blend = w1 * oof[m1] + w2 * oof[m2]
            for t in np.arange(0.2, 0.8, 0.01):
                f1 = f1_score(y, (blend >= t).astype(int))
                if f1 > best_pair_f1:
                    best_pair_f1 = f1
                    best_pair_config = (m1, w1, m2, w2, t)

m1, w1, m2, w2, t = best_pair_config
print(f"Лучшая пара: {m1}({w1:.1f}) + {m2}({w2:.1f}), F1={best_pair_f1:.4f} (t={t:.2f})")

single_scores = {}
for name in available:
    best_f1 = 0.0
    for t in np.arange(0.2, 0.8, 0.01):
        f1 = f1_score(y, (oof[name] >= t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
    single_scores[name] = best_f1

top3_lr = sorted(
    [n for n in available if n not in ("lgbm", "cnb", "sgd")],
    key=single_scores.get, reverse=True
)[:3]
candidates = top3_lr + ["lgbm"]
if "cnb" in available:
    candidates.append("cnb")
top4 = candidates

best_triple_f1 = 0.0
best_triple_config = None
weight_grid = np.arange(0.1, 0.9, 0.1)

for i, m1 in enumerate(top4):
    for j, m2 in enumerate(top4):
        if j <= i:
            continue
        for m3 in top4[j+1:]:
            for w1 in weight_grid:
                for w2 in weight_grid:
                    w3 = 1.0 - w1 - w2
                    if w3 <= 0 or w3 > 0.9:
                        continue
                    blend = w1 * oof[m1] + w2 * oof[m2] + w3 * oof[m3]
                    for t in np.arange(0.2, 0.8, 0.01):
                        f1 = f1_score(y, (blend >= t).astype(int))
                        if f1 > best_triple_f1:
                            best_triple_f1 = f1
                            best_triple_config = (m1, w1, m2, w2, m3, w3, t)

if best_triple_config:
    m1, w1, m2, w2, m3, w3, t = best_triple_config
    print(f"Лучший тройной: {m1}({w1:.1f}) + {m2}({w2:.1f}) + {m3}({w3:.1f}), F1={best_triple_f1:.4f} (t={t:.2f})")

if best_triple_f1 >= best_pair_f1:
    m1, w1, m2, w2, m3, w3, best_t = best_triple_config
    final_oof = w1 * oof[m1] + w2 * oof[m2] + w3 * oof[m3]
    final_test = w1 * test_preds[m1] + w2 * test_preds[m2] + w3 * test_preds[m3]
    print(f"Используем тройной: {m1}({w1:.1f}) + {m2}({w2:.1f}) + {m3}({w3:.1f}), F1={best_triple_f1:.4f}")
else:
    m1, w1, m2, w2, best_t = best_pair_config
    final_oof = w1 * oof[m1] + w2 * oof[m2]
    final_test = w1 * test_preds[m1] + w2 * test_preds[m2]
    print(f"Используем пару: {m1}({w1:.1f}) + {m2}({w2:.1f}), F1={best_pair_f1:.4f}")

np.save("models/ensemble_v2_oof.npy", final_oof)
np.save("models/ensemble_v2_test.npy", final_test)

sub = test[["id"]].copy()
sub["label"] = (final_test >= best_t).astype(int)
sub.to_csv("submissions/ensemble_v2.csv", index=False)
print(f"Saved submissions/ensemble_v2.csv, positive rate: {sub['label'].mean():.3f}")
