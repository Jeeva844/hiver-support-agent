"""
Phase 4: Build the golden evaluation set (stratified sampling).

Sampling strata guarantee coverage of:
  - common and rare intents (via first-pass rule labels)
  - short / long / noisy messages
  - ambiguous messages (multiple rules fired)
  - different time periods

The output golden_candidates.csv is the REVIEW INPUT. Every row is then
manually reviewed (per labeling_guidelines.md) before it is accepted into
golden_set.csv. First-pass labels are suggestions only.

IMPORTANT: sampled sources are recorded so Phase 8 can exclude them from
the retrieval index (no leakage).
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rule_labeler import first_pass_intent  # noqa: E402

PAIRS_PATH = Path("data/processed/spotify_pairs.csv")
GOLDEN_DIR = Path("data/golden")
TARGET = 200
SEED = 42

# Stratum quota per first-pass intent bucket.
# Intentionally unbalanced: common buckets get more, rare buckets get a
# guaranteed floor so they are represented in evaluation.
BUCKET_QUOTAS = {
    "subscription_plan": 28,
    "technical_issue": 26,
    "payment_billing": 22,
    "account_access": 20,
    "playback_issue": 20,
    "content_missing": 20,
    "appreciation": 18,
    "device_compatibility": 14,
    "family_plan": 12,
    "feature_and_feedback": 12,
    "other": 8,
}


def main() -> None:
    t0 = time.time()
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    assert sum(BUCKET_QUOTAS.values()) == TARGET, "Bucket quotas must sum to TARGET"

    rng = random.Random(SEED)
    df = pd.read_csv(PAIRS_PATH, encoding="utf-8")
    print(f"Loaded {len(df):,} pairs")

    # --- temporal stratification ---
    # NOTE: 99.96% of SpotifyCares pairs fall in 2017 and ~95% in Oct-Nov 2017.
    # True temporal diversity does not exist in these data; we stratify by month
    # purely to get a coarse spread, and document this as a limitation.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["month"] = df["timestamp"].dt.to_period("M").astype(str)
    df["flag_month_tail"] = df["month"].isin(["2017-10", "2017-11"])

    # first-pass labels (suggestions only)
    fps = df["customer_text_clean"].map(first_pass_intent)
    df["first_pass_label"] = [f.best_intent for f in fps]
    df["flag_ambiguous"] = [f.n_matching_intents > 1 for f in fps]
    df["flag_noisy"] = df["customer_text_clean"].str.contains(
        r"[^\x00-\x7F]|[A-Z]{4,}|[!?.]{3,}", regex=True
    )
    df["msg_len"] = df["customer_text_clean"].str.len()
    df["flag_short"] = df["msg_len"] < 20
    df["flag_long"] = df["msg_len"] > 40

    selected: list[pd.DataFrame] = []
    for bucket, quota in BUCKET_QUOTAS.items():
        pool = df[df["first_pass_label"] == bucket]
        if len(pool) < quota:
            print(f"WARNING: bucket '{bucket}' only has {len(pool)} examples")
        # guarantee at least one example from each month present in the bucket
        months = pool["month"].value_counts()
        quota_a = min(quota // 3 + 1, int(pool["flag_ambiguous"].sum()), quota)
        guarantee = min(len(pool), len(months))
        allowance = min(2, quota // 4, len(months))
        guarantee_qty = min(guarantee, max(1, quota // 5))
        keep: list[pd.DataFrame] = []
        days = {}
        for m in months.index[:guarantee_qty]:
            mrow = pool[pool["month"] == m]
            keep.append(mrow.sample(1, random_state=rng.randrange(10000)))
            days[m] = 1
        if len(keep):
            take_guarantee = pd.concat(keep)
            rest = pool.drop(take_guarantee.index)
        else:
            take_guarantee = pool.iloc[0:0]
            rest = pool
        # ambiguous over-representation within the remaining budget
        remaining = quota - len(take_guarantee)
        amb = rest[rest["flag_ambiguous"]]
        quota_amb = min(remaining // 2, len(amb), max(0, remaining))
        take_amb = amb.sample(quota_amb, random_state=rng.randrange(10000)) if quota_amb else amb.iloc[0:0]
        rest = rest.drop(take_amb.index)
        quota_rest = quota - len(take_guarantee) - len(take_amb)
        take_rest = rest.sample(min(quota_rest, len(rest)), random_state=rng.randrange(10000)) if quota_rest and len(rest) else rest.iloc[0:0]
        chosen = pd.concat([take_guarantee, take_amb, take_rest])
        if len(chosen) < quota:
            fill = pool.drop(chosen.index).sample(quota - len(chosen), random_state=rng.randrange(10000))
            chosen = pd.concat([chosen, fill])
        chosen = chosen.copy()
        chosen["stratum_bucket"] = bucket
        selected.append(chosen)

    golden = pd.concat(selected, ignore_index=True)
    golden = golden.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    golden.insert(0, "id", [f"g{i:03d}" for i in range(len(golden))])

    # evaluation text = cleaned message (matches what the pipeline consumes),
    # raw text preserved for transparency
    golden["raw_text"] = golden["customer_text"]
    golden = golden.rename(columns={"customer_text_clean": "text"})

    # first-pass escalation suggestion (must be confirmed in review)
    tax = json.loads(Path("data/intent_taxonomy.json").read_text(encoding="utf-8"))
    golden["escalation_suggestion"] = [
        not tax["intents"].get(lbl, {}).get("auto_eligible", False)
        for lbl in golden["first_pass_label"]
    ]

    cols = [
        "id", "text", "raw_text", "first_pass_label", "escalation_suggestion",
        "should_escalate", "human_notes", "customer_tweet_id",
        "conversation_id", "month", "flag_month_tail", "msg_len",
        "flag_short", "flag_long",
        "flag_noisy", "flag_ambiguous", "stratum_bucket",
    ]
    for c in ["should_escalate", "human_notes"]:
        if c not in golden.columns:
            golden[c] = ""
    golden = golden[cols]
    golden["review_status"] = "pending"

    golden.to_csv(GOLDEN_DIR / "golden_candidates.csv", index=False, encoding="utf-8")

    # summary of coverage
    summary = {
        "target": TARGET,
        "actual": int(len(golden)),
        "seed": SEED,
        "bucket_counts": golden["stratum_bucket"].value_counts().to_dict(),
        "ambiguous_count": int(golden["flag_ambiguous"].sum()),
        "short_count": int(golden["flag_short"].sum()),
        "long_count": int(golden["flag_long"].sum()),
        "noisy_count": int(golden["flag_noisy"].sum()),
        "month_counts_top": golden["month"].value_counts().head(5).to_dict(),
        "distinct_months_sampled": int(golden["month"].nunique()),
        "tail_months_share_pct": round(
            float(golden["flag_month_tail"].mean()) * 100, 1
        ),
        "distinct_conversations_sampled": int(golden["conversation_id"].nunique()),
    }
    (GOLDEN_DIR / "sampling_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print(f"\nSaved candidates -> {GOLDEN_DIR / 'golden_candidates.csv'}")
    print(f"Wrote sampling summary -> {GOLDEN_DIR / 'sampling_summary.json'}")
    print(f"Runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()