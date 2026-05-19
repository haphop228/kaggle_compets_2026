import pandas as pd
import numpy as np
from collections import Counter
import re

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nLabel distribution:\n{train['label'].value_counts()}")
print(f"\nLabel balance: {train['label'].mean():.3f} (fraction positive)")

print(f"\nMissing values train:\n{train.isnull().sum()}")
print(f"\nMissing values test:\n{test.isnull().sum()}")

train["body_empty"] = train["body"].isna() | (train["body"].astype(str).str.strip() == "")
print(f"\nEmpty body by class:")
print(train.groupby("label")["body_empty"].mean())

train["body"] = train["body"].fillna("")
train["text"] = train["title"].astype(str) + " " + train["body"].astype(str)
train["text"] = train["text"].str.strip()

train["title_len"] = train["title"].astype(str).str.len()
train["body_len"] = train["body"].astype(str).str.len()
train["text_len"] = train["text"].str.len()
train["word_count"] = train["text"].str.split().str.len()
train["sent_count"] = train["text"].str.count(r'[.!?]+')
train["excl_count"] = train["text"].str.count(r'!')
train["upper_ratio"] = train["text"].apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
train["question_count"] = train["text"].str.count(r'\?')

print("\nText length stats by class:")
for col in ["title_len", "body_len", "text_len", "word_count", "sent_count", "excl_count", "upper_ratio", "question_count"]:
    g = train.groupby("label")[col].mean()
    print(f"{col}: label=0 → {g[0]:.2f}, label=1 → {g[1]:.2f}")

def top_ngrams(texts, n=1, top_k=20):
    tokens = []
    for t in texts:
        words = re.findall(r'\b[a-z]{3,}\b', t.lower())
        if n == 1:
            tokens.extend(words)
        else:
            tokens.extend([" ".join(words[i:i+n]) for i in range(len(words)-n+1)])
    return Counter(tokens).most_common(top_k)

STOPWORDS = {"the","and","for","that","this","with","have","are","was","but","not","you","your",
             "just","like","its","from","they","been","has","had","all","can","get","got","one",
             "out","what","when","who","will","would","there","their","about","more","some","than",
             "then","into","also","very","even","know","think","want","feel","really","dont","its",
             "its","ive","im","its","its","its"}

def top_ngrams_filtered(texts, n=1, top_k=20):
    tokens = []
    for t in texts:
        words = re.findall(r'\b[a-z]{3,}\b', t.lower())
        words = [w for w in words if w not in STOPWORDS]
        if n == 1:
            tokens.extend(words)
        else:
            tokens.extend([" ".join(words[i:i+n]) for i in range(len(words)-n+1)])
    return Counter(tokens).most_common(top_k)

dep_texts = train[train["label"] == 1]["text"].tolist()
non_dep_texts = train[train["label"] == 0]["text"].tolist()

print("\nTop unigrams: depressed")
print(top_ngrams_filtered(dep_texts, 1, 20))

print("\nTop unigrams: not depressed")
print(top_ngrams_filtered(non_dep_texts, 1, 20))

print("\nTop bigrams: depressed")
print(top_ngrams_filtered(dep_texts, 2, 20))

print("\nTop bigrams: not depressed")
print(top_ngrams_filtered(non_dep_texts, 2, 20))

print("\nCorrelation of meta-features with label:")
meta_cols = ["title_len", "body_len", "text_len", "word_count", "sent_count",
             "excl_count", "upper_ratio", "question_count", "body_empty"]
for col in meta_cols:
    corr = train[col].astype(float).corr(train["label"].astype(float))
    print(f"{col}: {corr:.4f}")
