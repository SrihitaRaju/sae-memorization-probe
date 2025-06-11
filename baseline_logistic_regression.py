#!/usr/bin/env python3
"""
Baseline logistic regression using only confounding variables to predict memorization.
"""

import json, argparse, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score

# ────────────────────────────────────────────────────────────────────────────
def load_df(path, drop_wiki=True):
    df = pd.read_json(path, lines=True)
    if drop_wiki:
        df = df[df["domain"] != "wiki"].reset_index(drop=True)
    return df

from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_pipeline(num_cols, cat_cols):
    numeric = Pipeline([("scale", StandardScaler())])
    categorical = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    pre = ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols),
    ])
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    return Pipeline([("prep", pre), ("clf", clf)])

# ────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="matched_*.jsonl file")
    ap.add_argument("--drop_wiki", action="store_true", default=False,
                    help="omit tiny wiki slice (recommended)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 1. Load
    df = load_df(args.jsonl, drop_wiki=args.drop_wiki)
    X_num = ["dup_count", "prefix_nll", "rare_rate"]
    X_cat = ["domain"]
    y = (df["label"] == "mem").astype(int).values

    # 2. Pipeline
    pipe = build_pipeline(X_num, X_cat)

    # 3. Five-fold CV (stratified)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_auc = cross_val_score(pipe, df[X_num+X_cat], y,
                             scoring="roc_auc", cv=cv, n_jobs=-1)
    print(f"5-fold CV ROC-AUC  :  {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

    # 4. Hold-out split for final metrics
    X_train, X_test, y_train, y_test = train_test_split(
        df[X_num+X_cat], y, test_size=0.2, stratify=y, random_state=args.seed
    )
    pipe.fit(X_train, y_train)
    prob = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob)
    ap_ = average_precision_score(y_test, prob)
    print(f"Hold-out ROC-AUC   :  {auc:.4f}")
    print(f"Hold-out PR-AUC    :  {ap_:.4f}")

if __name__ == "__main__":
    main()

