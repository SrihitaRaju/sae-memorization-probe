#!/usr/bin/env python
"""
Create a balanced dataset of memorized and non-memorized examples.

Matches samples based on confounding factors (duplication count, prefix NLL, rare token rate)
to ensure fair comparison between memorized and non-memorized sequences.
"""

import re, json, random, argparse, math, sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

# ────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ────────────────────────────────────────────────────────────────────────────

CODE_TRIGGERS = re.compile(
    r"\b(def|class|import|function|return|#include)\b|[{;}<>]|::"
)
WIKI_PATTERN = re.compile(r"==")

def tag_domain(text: str) -> str:
    """Fallback domain heuristic."""
    if CODE_TRIGGERS.search(text):
        return "code"
    if WIKI_PATTERN.search(text):
        return "wiki"
    return "prose"

def effect_size(arr1, arr2):
    """Cohen's d."""
    if len(arr1) < 2 or len(arr2) < 2:
        return 0.0
    v1, v2 = np.var(arr1, ddof=1), np.var(arr2, ddof=1)
    s = math.sqrt(((len(arr1)-1)*v1 + (len(arr2)-1)*v2) / (len(arr1)+len(arr2)-2))
    return (np.mean(arr1) - np.mean(arr2)) / s if s else 0.0

def bin_column(arr, edges):
    """Return 0-3 quartile bin indices for a NumPy array."""
    return np.searchsorted(edges, arr, side="right")

# ────────────────────────────────────────────────────────────────────────────
# Matching core
# ────────────────────────────────────────────────────────────────────────────

def match_domain(df_pos, df_neg, conf_cols):
    """
    Match negatives to positives inside one domain with adaptive relaxation.
    Returns a DataFrame of matched negatives (<= len(df_pos)).
    """
    grp_cols = [f"{c}_bin" for c in conf_cols]
    need = df_pos.groupby(grp_cols).size()

    remaining = df_neg.copy()
    chosen = []

    def sample(level_cols):
        nonlocal remaining, chosen
        for cell, n_pos in need.items():
            mask = (remaining[level_cols] ==
                    pd.Series(cell, index=grp_cols).loc[level_cols]).all(axis=1)
            pool = remaining[mask]
            if pool.empty:
                continue
            take = pool.sample(min(len(pool), n_pos), random_state=42)
            chosen.append(take)
            remaining = remaining.drop(take.index)

    # exact (dup, nll, rare) bins
    sample(grp_cols)
    # relax rare_bin if needed
    if sum(map(len, chosen)) < len(df_pos):
        sample(["dup_count_bin", "prefix_nll_bin"])
    # relax nll_bin last
    if sum(map(len, chosen)) < len(df_pos):
        sample(["dup_count_bin"])

    if not chosen:
        return pd.DataFrame(columns=df_neg.columns)
    return pd.concat(chosen, ignore_index=True)


def build_matched(df, target_each=5000, min_total=8000):
    """Main orchestrator – returns matched DataFrame, stats DataFrame."""
    # --- safety checks ------------------------------------------------------
    required = ["label", "dup_count", "prefix_nll", "rare_rate", "text"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.dropna(subset=["dup_count", "prefix_nll", "rare_rate"]).reset_index(drop=True)

    conf = ["dup_count", "prefix_nll", "rare_rate"]

    # --- global quartile bins ----------------------------------------------
    edges = {c: df[c].quantile([0.25, 0.5, 0.75]).values for c in conf}
    for c in conf:
        df[f"{c}_bin"] = bin_column(df[c].values, edges[c])

    # --- domain tagging -----------------------------------------------------
    if "domain" not in df.columns:
        if "meta" in df.columns:                                   # nested dict column
            df["domain"] = df["meta"].apply(
                lambda m: (m.get("dataset") if isinstance(m, dict) else None)
            )
        else:
            df["domain"] = None
        df["domain"] = df["domain"].fillna(df["text"].map(tag_domain))
        df["domain"] = (
            df["domain"]
            .replace({r".*code.*": "code", r".*wiki.*": "wiki"}, regex=True)
            .where(df["domain"].isin(["code", "wiki", "prose"]), "prose")
        )

    pos_df = df[df.label == "mem"].reset_index(drop=True)
    neg_df = df[df.label == "nonmem"].reset_index(drop=True)

    out_frames, stats = [], []

    for dom in ["code", "wiki", "prose"]:
        dp = pos_df[pos_df.domain == dom]
        dn = neg_df[neg_df.domain == dom]
        if dp.empty or dn.empty:
            continue

        # target counts
        n_pos_tgt = min(len(dp), target_each)
        n_neg_tgt = n_pos_tgt

        neg_match = match_domain(dp, dn, conf)
        neg_match = neg_match.head(n_neg_tgt)

        # enforce 60/40 cap inside domain
        max_pos = int(1.5 * len(neg_match))
        dp_trim = dp.sample(min(len(dp), max_pos), random_state=42)
        n_pos_final = min(len(dp_trim), n_pos_tgt)

        out_frames.append(dp_trim.head(n_pos_final))
        out_frames.append(neg_match)

        stats.append(
            {"domain": dom, "pos": n_pos_final, "neg": len(neg_match)}
        )

    matched = pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()

    # --- fallback: relax worst domain to 70/30 if too small -----------------
    if len(matched) < min_total and not matched.empty:
        dom_stats = pd.DataFrame(stats)
        dom_stats["surplus"] = dom_stats.pos - dom_stats.neg
        worst = dom_stats.sort_values("surplus", ascending=False).iloc[0]["domain"]

        dp_worst = pos_df[pos_df.domain == worst]
        dn_worst = neg_df[neg_df.domain == worst]
        extra_cap = max(len(dp_worst) - len(dn_worst), 0)
        extra = int(0.1667 * len(dp_worst))  # 70/30 allows ~17 % extra
        extra = min(extra, extra_cap)
        if extra:
            take = dp_worst.sample(extra, random_state=42)
            matched = pd.concat([matched, take], ignore_index=True)

    # --- effect-size audit --------------------------------------------------
    audit_rows = []
    for dom in matched.domain.unique():
        mp = matched[(matched.domain == dom) & (matched.label == "mem")]
        mn = matched[(matched.domain == dom) & (matched.label == "nonmem")]
        row = {"domain": dom, "pos": len(mp), "neg": len(mn)}
        for c in conf:
            row[f"d_{c}"] = round(effect_size(mp[c].values, mn[c].values), 3)
            row[f"ks_{c}"] = round(
                ks_2samp(mp[c], mn[c], alternative="two-sided")[1], 4
            )
        audit_rows.append(row)

    audit = pd.DataFrame(audit_rows).sort_values("domain")
    return matched, audit

# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_jsonl", help="enriched mem/nonmem JSONL")
    ap.add_argument("--out_jsonl", default="matched_12k.jsonl")
    ap.add_argument("--stats_csv", default="matching_stats.csv")
    ap.add_argument("--target_each", type=int, default=6000)
    ap.add_argument("--min_total", type=int, default=10000)
    args = ap.parse_args()

    df = pd.read_json(args.in_jsonl, lines=True)
    matched, report = build_matched(df, args.target_each, args.min_total)

    if matched.empty:
        sys.exit("❌  No matched rows – check input data / labels.")
    matched.to_json(args.out_jsonl, orient="records", lines=True, force_ascii=False)
    report.to_csv(args.stats_csv, index=False)

    print(f"✅  wrote {len(matched)} rows → {args.out_jsonl}")
    print(report.to_markdown(index=False))

if __name__ == "__main__":
    main()