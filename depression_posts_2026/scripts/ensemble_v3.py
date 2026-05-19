import pandas as pd
import numpy as np
import os
from sklearn.metrics import f1_score

test = pd.read_csv("data/test.csv")
train = pd.read_csv("data/train.csv")
y = train["label"].values

models = {
    "logreg_v5":      ("models/logreg_v5_oof.npy",       "models/logreg_v5_test.npy"),
    "logreg_v5_tune": ("models/logreg_v5_tune_oof.npy",  "models/logreg_v5_tune_test.npy"),
    "logreg_v4":      ("models/logreg_v4_oof.npy",       "models/logreg_v4_test.npy"),
    "logreg_v3":      ("models/logreg_v3_oof.npy",       "models/logreg_v3_test.npy"),
    "lgbm":           ("models/lgbm_oof.npy",            "models/lgbm_test.npy"),
    "cnb":            ("models/cnb_oof.npy",             "models/cnb_test.npy"),
}

oof = {}
test_preds = {}
for name, (oof_path, test_path) in models.items():
    oof[name] = np.load(oof_path)
    test_preds[name] = np.load(test_path)

available = list(models.keys())

def best_f1_threshold(oof_blend, y):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.2, 0.8, 0.01):
        f1 = f1_score(y, (oof_blend >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_f1, best_t

print("Single models:")
for name in available:
    f1, t = best_f1_threshold(oof[name], y)
    print(f"{name}: F1={f1:.4f} (t={t:.2f})")

print("\nTriple ensembles:")
best_f1_global = 0.0
best_config = None
weight_grid = np.arange(0.1, 0.9, 0.1)

for i, m1 in enumerate(available):
    for j, m2 in enumerate(available):
        if j <= i:
            continue
        for k, m3 in enumerate(available):
            if k <= j:
                continue
            for w1 in weight_grid:
                for w2 in weight_grid:
                    w3 = round(1.0 - w1 - w2, 2)
                    if w3 <= 0.05 or w3 > 0.85:
                        continue
                    blend = w1 * oof[m1] + w2 * oof[m2] + w3 * oof[m3]
                    f1, t = best_f1_threshold(blend, y)
                    if f1 > best_f1_global:
                        best_f1_global = f1
                        best_config = (m1, w1, m2, w2, m3, w3, t)

m1, w1, m2, w2, m3, w3, t = best_config
print(f"Лучший тройной: {m1}({w1:.1f}) + {m2}({w2:.1f}) + {m3}({w3:.2f}), F1={best_f1_global:.4f} (t={t:.2f})")

print("\nQuad ensembles (lgbm+cnb required):")
lr_models = [n for n in available if n not in ("lgbm", "cnb")]
best_quad_f1 = 0.0
best_quad_config = None

for i, m1 in enumerate(lr_models):
    for m2 in lr_models[i+1:]:
        for w1 in weight_grid:
            for w2 in weight_grid:
                for w_lgbm in weight_grid:
                    w_cnb = round(1.0 - w1 - w2 - w_lgbm, 2)
                    if w_cnb <= 0.05 or w_cnb > 0.5:
                        continue
                    blend = (w1 * oof[m1] + w2 * oof[m2] +
                             w_lgbm * oof["lgbm"] + w_cnb * oof["cnb"])
                    f1, t = best_f1_threshold(blend, y)
                    if f1 > best_quad_f1:
                        best_quad_f1 = f1
                        best_quad_config = (m1, w1, m2, w2, "lgbm", w_lgbm, "cnb", w_cnb, t)

if best_quad_config:
    m1, w1, m2, w2, m3, w3, m4, w4, t = best_quad_config
    print(f"Лучший четвёрной: {m1}({w1:.1f}) + {m2}({w2:.1f}) + {m3}({w3:.1f}) + {m4}({w4:.2f}), F1={best_quad_f1:.4f} (t={t:.2f})")

if best_quad_f1 >= best_f1_global:
    m1, w1, m2, w2, m3, w3, m4, w4, best_t = best_quad_config
    final_oof = w1 * oof[m1] + w2 * oof[m2] + w3 * oof[m3] + w4 * oof[m4]
    final_test = w1 * test_preds[m1] + w2 * test_preds[m2] + w3 * test_preds[m3] + w4 * test_preds[m4]
    print(f"quad: F1={best_quad_f1:.4f}")
else:
    m1, w1, m2, w2, m3, w3, best_t = best_config
    final_oof = w1 * oof[m1] + w2 * oof[m2] + w3 * oof[m3]
    final_test = w1 * test_preds[m1] + w2 * test_preds[m2] + w3 * test_preds[m3]
    print(f"triple: F1={best_f1_global:.4f}")

np.save("models/ensemble_v3_oof.npy", final_oof)
np.save("models/ensemble_v3_test.npy", final_test)

sub = test[["id"]].copy()
sub["label"] = (final_test >= best_t).astype(int)
sub.to_csv("submissions/ensemble_v3.csv", index=False)
print(f"Saved submissions/ensemble_v3.csv, positive rate: {sub['label'].mean():.3f}")
