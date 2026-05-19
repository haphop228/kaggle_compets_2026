import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import Ridge, Lasso
import pickle
import warnings
warnings.filterwarnings("ignore")

SEED = 42

with open("models/v3_results.pkl", "rb") as f:
    v3 = pickle.load(f)

with open("models/v2_results.pkl", "rb") as f:
    v2 = pickle.load(f)

y_log = v3["y_log"]
y = np.expm1(y_log)

oof_lgb_v3 = v3["oof_lgb"]
oof_xgb_v3 = v3["oof_xgb"]
oof_cb_v3 = v3["oof_cb"]

oof_lgb_v2 = v2["oof_lgb"]
oof_cb_v2 = v2["oof_cb"]

test_lgb_v3 = v3["test_lgb"]
test_xgb_v3 = v3["test_xgb"]
test_cb_v3 = v3["test_cb"]

test_lgb_v2 = v2["test_lgb"]
test_cb_v2 = v2["test_cb"]

print("Individual OOF MAPEs:")
print(f"  LGB v2:  {mean_absolute_percentage_error(y, np.expm1(oof_lgb_v2)):.4f}")
print(f"  CB  v2:  {mean_absolute_percentage_error(y, np.expm1(oof_cb_v2)):.4f}")
print(f"  LGB v3:  {mean_absolute_percentage_error(y, np.expm1(oof_lgb_v3)):.4f}")
print(f"  XGB v3:  {mean_absolute_percentage_error(y, np.expm1(oof_xgb_v3)):.4f}")
print(f"  CB  v3:  {mean_absolute_percentage_error(y, np.expm1(oof_cb_v3)):.4f}")

oof_stack = np.column_stack([
    oof_lgb_v3,
    oof_xgb_v3,
    oof_cb_v3,
    oof_lgb_v2,
    oof_cb_v2,
])

test_stack = np.column_stack([
    test_lgb_v3,
    test_xgb_v3,
    test_cb_v3,
    test_lgb_v2,
    test_cb_v2,
])

print(f"\nStack shape: {oof_stack.shape}")

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof_meta = np.zeros(len(y_log))
test_meta = np.zeros(len(test_stack))

for tr_idx, val_idx in kf.split(oof_stack):
    ridge = Ridge(alpha=1.0, positive=True)
    ridge.fit(oof_stack[tr_idx], y_log[tr_idx])
    oof_meta[val_idx] = ridge.predict(oof_stack[val_idx])
    test_meta += ridge.predict(test_stack) / 5

meta_mape = mean_absolute_percentage_error(y, np.expm1(oof_meta))
print(f"\nRidge meta OOF MAPE: {meta_mape:.4f}")
print(f"Ridge coefficients: {ridge.coef_}")

print("\nEXHAUSTIVE BLEND (5 models)")
best_mape = 1.0
best_w = None

for w0 in np.arange(0.0, 1.01, 0.1):
    for w1 in np.arange(0.0, 1.01 - w0, 0.1):
        for w2 in np.arange(0.0, 1.01 - w0 - w1, 0.1):
            for w3 in np.arange(0.0, 1.01 - w0 - w1 - w2, 0.1):
                w4 = round(1.0 - w0 - w1 - w2 - w3, 10)
                if w4 < -1e-9:
                    continue
                blend = w0*oof_lgb_v3 + w1*oof_xgb_v3 + w2*oof_cb_v3 + w3*oof_lgb_v2 + w4*oof_cb_v2
                m = mean_absolute_percentage_error(y, np.expm1(blend))
                if m < best_mape:
                    best_mape = m
                    best_w = (w0, w1, w2, w3, w4)

print(f"Best 5-model blend MAPE: {best_mape:.4f}")
print(f"Weights: lgb_v3={best_w[0]:.2f} xgb_v3={best_w[1]:.2f} cb_v3={best_w[2]:.2f} lgb_v2={best_w[3]:.2f} cb_v2={best_w[4]:.2f}")

test_best_blend = (best_w[0]*test_lgb_v3 + best_w[1]*test_xgb_v3 + best_w[2]*test_cb_v3 +
                   best_w[3]*test_lgb_v2 + best_w[4]*test_cb_v2)

test_raw = pd.read_csv("test_x.csv")
test_ids = test_raw["id"].values

pd.DataFrame({"id": test_ids, "price_target": np.expm1(test_meta)}).to_csv(
    "submissions/ridge_meta.csv", index=False)
pd.DataFrame({"id": test_ids, "price_target": np.expm1(test_best_blend)}).to_csv(
    "submissions/best_blend_5models.csv", index=False)

print(f"\nSaved submissions/ridge_meta.csv (MAPE={meta_mape:.4f})")
print(f"Saved submissions/best_blend_5models.csv (MAPE={best_mape:.4f})")

print("\nEXPERIMENT SUMMARY")
results = [
    ("Baseline LightGBM", 0.0206),
    ("LightGBM v2 (correct TE)", 0.0201),
    ("CatBoost v2", 0.0221),
    ("Blend LGB+CB v2", 0.0199),
    ("LightGBM v3 (more FE)", 0.0203),
    ("XGBoost v3", 0.0210),
    ("CatBoost v3", 0.0220),
    ("Blend LGB+XGB+CB v3", 0.0200),
    ("LightGBM v4 (SHAP pruned)", 0.0208),
    ("LightGBM v5 (interaction TE)", 0.0209),
    ("Ridge meta-learner", meta_mape),
    ("Best 5-model blend", best_mape),
]
for name, mape in sorted(results, key=lambda x: x[1]):
    print(f"  {mape:.4f}  {name}")
