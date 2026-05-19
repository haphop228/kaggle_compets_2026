"""
Experiment 05: CatBoost with Native Text Features
exp03 best: Ridge(200k, alpha=0.5) + XGB_sep800, OOF=0.2169

Tests:
  A) CatBoost with text_features (name_clean, key_skills_name, lemmaized_wo_stopwords_raw_description)
  B) CatBoost with text_features + MAE loss

Run: .venv/bin/python experiments/exp05_catboost_text.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor, Pool
import warnings
warnings.filterwarnings("ignore")

from experiments.features import (
    load_data, mape, TARGET, SEED, N_SPLITS, DESC_COL, TITLE_COL, SKILLS_COL,
)

train, test = load_data()
y_raw = train[TARGET].values.astype(np.float64)
y_log = np.log1p(y_raw).astype(np.float32)
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# Fill NaNs in text columns
for df in [train, test]:
    df[DESC_COL]    = df[DESC_COL].fillna("")
    df[TITLE_COL]   = df[TITLE_COL].fillna("")
    df[SKILLS_COL]  = df[SKILLS_COL].fillna("")

text_cols = [TITLE_COL, SKILLS_COL, DESC_COL]

# Categorical columns
cat_cols = [
    "experience_name", "professional_roles_name", "schedule_name",
    "employment_name", "specializations_profarea_name", "unified_address_region",
    "unified_address_state", "employer_industries", "employer_id", "employer_name",
    "unified_address_city", "if_foreign_language", "is_branded_description"
]

# Fill NaNs in categorical columns and convert to string
for col in cat_cols:
    for df in [train, test]:
        if col in df.columns:
            df[col] = df[col].fillna("__NA__").astype(str)

# Numeric columns
num_cols = ["accept_handicapped", "accept_kids"]
for df in [train, test]:
    df["accept_handicapped"] = df["accept_handicapped"].astype(np.int8)
    df["accept_kids"]        = df["accept_kids"].astype(np.int8)
    df["has_languages"]      = (df["languages_name"].fillna("[]") != "[]").astype(np.int8)

num_cols.append("has_languages")

features = text_cols + cat_cols + num_cols

X_tr = train[features]
X_te = test[features]

results = []

def run_catboost(params, label, use_raw_y=False):
    oof = np.zeros(len(train))
    pred = np.zeros(len(test))
    
    target_tr = y_raw if use_raw_y else y_log
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tr)):
        train_pool = Pool(
            X_tr.iloc[tr_idx], target_tr[tr_idx],
            cat_features=cat_cols, text_features=text_cols
        )
        val_pool = Pool(
            X_tr.iloc[val_idx], target_tr[val_idx],
            cat_features=cat_cols, text_features=text_cols
        )
        test_pool = Pool(
            X_te,
            cat_features=cat_cols, text_features=text_cols
        )
        
        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool, verbose=100, early_stopping_rounds=200)
        
        preds_val = model.predict(val_pool)
        preds_te = model.predict(test_pool)
        
        if not use_raw_y:
            preds_val = np.expm1(preds_val)
            preds_te = np.expm1(preds_te)
            
        oof[val_idx] = preds_val
        pred += preds_te / N_SPLITS
        
        fold_mape = mape(y_raw[val_idx], oof[val_idx])
        print(f"    Fold {fold+1}: MAPE={fold_mape:.4f}")

    score = mape(y_raw, oof)
    print(f"  {label}: MAPE={score:.4f}")
    return oof, pred, score

# ════════════════════════════════════════════════════════════════════════════════
# A: CatBoost RMSE on log(y)
# ════════════════════════════════════════════════════════════════════════════════
print("\n=== A: CatBoost RMSE on log(y) ===")
cb_rmse_params = {
    "iterations": 5000,
    "learning_rate": 0.05,
    "depth": 6,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": SEED,
    "task_type": "CPU",
    "thread_count": -1,
    "text_processing": {
        "dictionaries": [
            {"dictionary_id": "Word", "max_dictionary_size": "50000"}
        ],
        "feature_processing": {
            "default": [
                {"dictionaries_names": ["Word"], "feature_calcers": ["BoW", "NaiveBayes"]}
            ]
        }
    }
}

oof_A, pred_A, sc_A = run_catboost(cb_rmse_params, "CB_RMSE_log", use_raw_y=False)
results.append({"config": "A_CB_RMSE_log", "score": sc_A})

# ════════════════════════════════════════════════════════════════════════════════
# B: CatBoost MAE on log(y)
# ════════════════════════════════════════════════════════════════════════════════
print("\n=== B: CatBoost MAE on log(y) ===")
cb_mae_params = {**cb_rmse_params, "loss_function": "MAE", "eval_metric": "MAE"}

oof_B, pred_B, sc_B = run_catboost(cb_mae_params, "CB_MAE_log", use_raw_y=False)
results.append({"config": "B_CB_MAE_log", "score": sc_B})

# ════════════════════════════════════════════════════════════════════════════════
# C: CatBoost MAPE on raw y
# ════════════════════════════════════════════════════════════════════════════════
print("\n=== C: CatBoost MAPE on raw y ===")
cb_mape_params = {**cb_rmse_params, "loss_function": "MAPE", "eval_metric": "MAPE"}

oof_C, pred_C, sc_C = run_catboost(cb_mape_params, "CB_MAPE_raw", use_raw_y=True)
results.append({"config": "C_CB_MAPE_raw", "score": sc_C})

# ════════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════════
df_res = pd.DataFrame(results)
df_res.to_csv("experiments/results_exp05.csv", index=False)
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(df_res[["config", "score"]].to_string(index=False))

best = df_res.loc[df_res["score"].idxmin()]
print(f"\nBest: {best['config']}  MAPE={best['score']:.4f}")
