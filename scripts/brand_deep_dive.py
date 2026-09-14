"""
Phase 1, extra: Deep-dive on top brand candidates.

Measures reply QUALITY, not just volume:
  - reply length distribution
  - templated/DM/redirect replies vs substantive replies
  - share of usable substantive paired examples

A support history full of "DM us" replies gives retrieval and reply-generation
nothing to ground on, so this materially affects brand selection.
"""
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path("data/raw/twcs.csv")
OUTPUT_DIR = Path("data/analysis")
TOP_N = 6

DM_PATTERNS = re.compile(
    r"\b\d*m\b|\bdm(m)?us\b|\bsend us\b|\bprivate message\b|\bpm us\b|"
    r"\bdirect message\b|\bgo to dms?\b|\bcheck your dm\b|"
    r"\bcontact us via\b|\breach us by\b|"
    r"\bphone or chat\b|\btwitter um nicht|"
    r"\bwrite to support\b|\ballow me to help\b",
    re.IGNORECASE,
)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def main() -> None:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    print(f"Loaded {len(df):,} rows")

    tweet_author = pd.Series(df["author_id"].values, index=df["tweet_id"].values)
    tweet_inbound = pd.Series(df["inbound"].values, index=df["tweet_id"].values)
    tweet_text = pd.Series(df["text"].values, index=df["tweet_id"].values)

    has_parent = df.dropna(subset=["in_response_to_tweet_id"]).copy()
    has_parent["parent_tid"] = has_parent["in_response_to_tweet_id"].astype("int64")
    has_parent["parent_inbound"] = (
        has_parent["parent_tid"].map(tweet_inbound).fillna(False).astype(bool)
    )
    has_parent["parent_text"] = has_parent["parent_tid"].map(tweet_text)

    # candidate brands (recomputed quickly: authors with many replies-to-customers)
    pairs = has_parent[~has_parent["inbound"] & has_parent["parent_inbound"]]
    pair_counts = pairs.groupby("author_id")["tweet_id"].size()
    top_authors = pair_counts.sort_values(ascending=False).head(TOP_N).index.tolist()

    results = {}
    for i, brand in enumerate(top_authors, 1):
        brand_pairs = pairs[pairs["author_id"] == brand].copy()
        brand_pairs = brand_pairs[brand_pairs["parent_text"].notna()]

        lens = brand_pairs["text"].str.len()
        reply_lens = brand_pairs["text"].dropna().str.len()

        # templated reply detection
        tiant_text = brand_pairs["text"].fillna("")
        tmpl_mask = (
            tiant_text.str.len().le(45)
            | tiant_text.str.contains(DM_PATTERNS, regex=True)
            | (tiant_text.str.len().le(90) & tiant_text.str.contains(r"https?://", regex=True))
        )

        substantive = brand_pairs[~tmpl_mask]
        # also require the parent (customer) message to have some content
        substantive = substantive[
            substantive["parent_text"].str.len().ge(5)
        ]

        # non-Latin (multi-language) share as an approximation
        latin_mask = brand_pairs["parent_text"].str.contains(
            r"[A-Za-z]{3,}", regex=True
        )
        non_latin_share = float((~latin_mask).mean())

        results[brand] = {
            "rank_by_volume": i,
            "total_replies": int(len(brand_pairs)),
            "reply_len_mean": round(float(reply_lens.mean()), 1),
            "reply_len_median": float(reply_lens.median()),
            "reply_len_p10": float(reply_lens.quantile(0.10)),
            "reply_len_p90": float(reply_lens.quantile(0.90)),
            "templated_or_redirect_pct": round(float(tmpl_mask.mean()) * 100, 1),
            "substantive_pairs": int(len(substantive)),
            "substantive_pct": round(float(len(substantive) / len(brand_pairs)) * 100, 1),
            "customer_nonlatin_pct": round(non_latin_share * 100, 1),
            "sample_substantive": [
                {
                    "customer": str(r["parent_text"])[:200],
                    "reply": str(r["text"])[:200],
                }
                for _, r in substantive.head(4).iterrows()
            ],
        }
        r = results[brand]
        print(
            f"\n#{r['rank_by_volume']} {brand:<20} replies={r['total_replies']:>7,} "
            f"median_len={r['reply_len_median']:.0f} templated={r['templated_or_redirect_pct']:.0f}% "
            f"substantive={r['substantive_pct']:.0f}% nonlatin={r['customer_nonlatin_pct']:.0f}%"
        )
        for s in r["sample_substantive"]:
            print(f"    C: {s['customer']}")
            print(f"    R: {s['reply'][:160]}")

    with open(OUTPUT_DIR / "brand_deep_dive.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved deep-dive to {OUTPUT_DIR / 'brand_deep_dive.json'}")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()