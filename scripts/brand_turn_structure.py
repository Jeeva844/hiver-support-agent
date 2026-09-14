"""
Phase 1, extra: Conversation turn-structure for top candidate brands.

Measures how "conversational" each brand's support history is:
  - tweets per conversation
  - share of multi-turn conversations (>=3 tweets)
  - distinct conversations per brand
This affects retrieval value: multi-turn threads give richer grounded examples.
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/raw/twcs.csv")
OUTPUT_DIR = Path("data/analysis")

BRANDS = ["AmazonHelp", "AppleSupport", "Uber_Support", "SpotifyCares", "Delta", "Tesco"]


def main() -> None:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows")

    # conversation reconstruction (same approach as analyze_brands)
    has_parent = df.dropna(subset=["in_response_to_tweet_id"])
    parent = (
        has_parent.set_index("tweet_id")["in_response_to_tweet_id"]
        .astype("int64")
        .to_dict()
    )
    root_memo: dict[int, int] = {}

    def find_root(tid: int) -> int:
        root = root_memo.get(tid)
        if root is not None:
            return root
        path: list[int] = []
        seen: set[int] = set()
        cur = tid
        while cur in parent and cur not in seen:
            cached = root_memo.get(cur)
            if cached is not None:
                cur = cached
                break
            seen.add(cur)
            path.append(cur)
            cur = parent[cur]
        root = cur
        for node in path:
            root_memo[node] = root
        return root

    conv_by_tweet = pd.Series(
        [find_root(int(t)) for t in df["tweet_id"]], index=df["tweet_id"].values
    )
    print(f"Conversations: {conv_by_tweet.nunique():,}")

    df = df.copy()
    df["conv"] = df["tweet_id"].map(conv_by_tweet)
    conv_sizes = df.groupby("conv")["tweet_id"].size()

    results = {}
    for brand in BRANDS:
        brand_tweets = df[df["author_id"] == brand]
        brand_convs = set(brand_tweets["conv"].unique())
        sizes = conv_sizes.reindex(brand_convs).dropna()
        # inbound customer messages inside those conversations
        customer_msgs_in_brand_convs = df[
            df["conv"].isin(brand_convs) & df["inbound"]
        ]
        results[brand] = {
            "conversations": int(len(brand_convs)),
            "tweets_in_those_convs_total": int(sizes.sum()),
            "median_conversation_size": float(sizes.median()),
            "p90_conversation_size": float(sizes.quantile(0.90)),
            "share_convs_with_2_tweets": round(float((sizes == 2).mean()) * 100, 1),
            "share_convs_with_ge3_tweets": round(float((sizes >= 3).mean()) * 100, 1),
            "customer_messages_in_those_convs": int(len(customer_msgs_in_brand_convs)),
        }
        r = results[brand]
        print(
            f"\n{brand:<20} convs={r['conversations']:>7,} "
            f"median_size={r['median_conversation_size']:.0f} "
            f"2-tweet_conv={r['share_convs_with_2_tweets']:.0f}% "
            f">=3-tweet_conv={r['share_convs_with_ge3_tweets']:.0f}% "
            f"customer_msgs={r['customer_messages_in_those_convs']:>7,}"
        )

    with open(OUTPUT_DIR / "brand_turn_structure.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()