"""Phase 4: finalize the golden set from manual review (data/golden/human_review.json).

Applies reviewer decisions onto golden_candidates.csv, producing
data/golden/golden_set.csv and data/golden/golden_label_stats.json.
Run from repo root:  python scripts/finalize_golden_set.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))
from intents import load_taxonomy, intent_names  # noqa: E402

REVIEW_PATH = Path("data/golden/human_review.json")
CANDIDATES_PATH = Path("data/golden/golden_candidates.csv")
OUTPUT_PATH = Path("data/golden/golden_set.csv")
STATS_PATH = Path("data/golden/golden_label_stats.json")


def main() -> None:
    tax = load_taxonomy()
    valid_intents = set(intent_names(tax))

    review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))["decisions"]
    if len(review) != 200:
        raise SystemExit(f"Expected 200 reviewer decisions, found {len(review)}")

    import pandas as pd

    df = pd.read_csv(CANDIDATES_PATH)
    if len(df) != 200:
        raise SystemExit(f"Expected 200 candidates, found {len(df)}")

    missing = [g for g in df["id"] if g not in review]
    if missing:
        raise SystemExit(f"Missing reviewer decisions for: {missing[:5]} ...")

    bad_intents = {}
    df["intent"] = None
    df["should_escalate"] = None
    df["human_notes"] = None
    for i, gid in enumerate(df["id"]):
        intent, escalate, note = review[gid]
        if intent not in valid_intents:
            bad_intents[gid] = intent
        df.at[i, "intent"] = intent
        df.at[i, "should_escalate"] = escalate
        df.at[i, "human_notes"] = note
    if bad_intents:
        raise SystemExit(f"Intents not in taxonomy: {bad_intents}")

    df["review_status"] = "reviewed"
    df["label_disagrees_with_first_pass"] = df["intent"] != df["first_pass_label"]
    df["first_pass_matches"] = df["intent"] == df["first_pass_label"]

    df.to_csv(OUTPUT_PATH, index=False)

    stats = {
        "n_examples": int(len(df)),
        "n_distinct_tweets": int(df["customer_tweet_id"].nunique()),
        "n_distinct_conversations": int(df["conversation_id"].nunique()),
        "n_distinct_months": int(df["month"].nunique()),
        "month_tail_share": float((df["month"] >= "2017-10").mean()),
        "all_reviewed": bool((df["review_status"] == "reviewed").all()),
        "intent_distribution": df["intent"].value_counts().to_dict(),
        "escale_rate_overall": float(df["should_escalate"].mean()),
        "escalate_rate_by_intent": {
            k: float(g["should_escalate"].mean())
            for k, g in df.groupby("intent")
        },
        "reviewer_vs_first_pass_disagree_rate": float(
            df["label_disagrees_with_first_pass"].mean()
        ),
        "disagreements": [
            {
                "id": r["id"],
                "first_pass": r["first_pass_label"],
                "reviewer": r["intent"],
                "escalate": bool(r["should_escalate"]),
                "note": r["human_notes"],
            }
            for _, r in df[df["label_disagrees_with_first_pass"]].iterrows()
        ],
        "schema_columns": list(df.columns),
    }
    STATS_PATH.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Wrote {len(df)} reviewed rows -> {OUTPUT_PATH}")
    print(f"  distinct tweets/convs/months: "
          f"{stats['n_distinct_tweets']}/{stats['n_distinct_conversations']}/{stats['n_distinct_months']}")
    print(f"  escalate overall: {stats['escale_rate_overall']:.2%}")
    print(f"  reviewer/first-pass disagreement: {stats['reviewer_vs_first_pass_disagree_rate']:.2%}")
    print("Intent distribution:")
    for intent, n in stats["intent_distribution"].items():
        print(f"    {intent:<24} {n:3d}  escalate={stats['escalate_rate_by_intent'][intent]:.2%}")


if __name__ == "__main__":
    main()