import pandas as pd
import numpy as np

train = pd.read_csv("period_1_train_data.csv")
test = pd.read_csv("test_x.csv")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print()

print("DTYPES")
print(train.dtypes.value_counts())
print()

print("TARGET STATS")
tgt = train["price_target"]
print(tgt.describe())
print(f"Skew: {tgt.skew():.3f}")
print(f"Log skew: {np.log(tgt).skew():.3f}")
print(f"Zeros/negatives: {(tgt <= 0).sum()}")
print()

print("-999 COUNTS (train)")
num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
neg999 = {c: (train[c] == -999).sum() for c in num_cols if (train[c] == -999).sum() > 0}
for c, v in sorted(neg999.items(), key=lambda x: -x[1]):
    print(f"  {c}: {v} ({v/len(train)*100:.1f}%)")
print()

print("-999 COUNTS (test)")
num_cols_test = test.select_dtypes(include=[np.number]).columns.tolist()
neg999_test = {c: (test[c] == -999).sum() for c in num_cols_test if (test[c] == -999).sum() > 0}
for c, v in sorted(neg999_test.items(), key=lambda x: -x[1]):
    print(f"  {c}: {v} ({v/len(test)*100:.1f}%)")
print()

print("MISSING VALUES (NaN)")
missing = train.isnull().sum()
missing = missing[missing > 0]
print(missing)
print()

print("rooms_4 VALUE COUNTS")
print(train["rooms_4"].value_counts().head(20))
print()

print("CATEGORICAL COLS")
cat_cols = ["region_name_cat", "district_cat", "class_cat", "stage_cat", "interior_cat", "hc_name_cat"]
for c in cat_cols:
    n_unique = train[c].nunique()
    print(f"  {c}: {n_unique} unique values")
print()

print("AGREEMENT_DATE RANGE")
train["agreement_date"] = pd.to_datetime(train["agreement_date"])
print(f"  Min: {train['agreement_date'].min()}")
print(f"  Max: {train['agreement_date'].max()}")
print(f"  Year distribution:")
print(train["agreement_date"].dt.year.value_counts().sort_index())
print()

print("CORRELATIONS WITH TARGET (top 20)")
train_num = train[num_cols].copy()
for c in num_cols:
    train_num[c] = train_num[c].replace(-999, np.nan)
corr = train_num.corrwith(train["price_target"]).abs().sort_values(ascending=False)
print(corr.head(20))
print()

print("LOCATION COLS COUNT")
loc_cols = [c for c in train.columns if c.startswith("location_")]
print(f"  Total location cols: {len(loc_cols)}")
print()

print("interior_cat VALUE COUNTS")
print(train["interior_cat"].value_counts().head(10))
print()

print("class_cat VALUE COUNTS")
print(train["class_cat"].value_counts().head(10))
print()

print("stage_cat VALUE COUNTS")
print(train["stage_cat"].value_counts().head(10))
