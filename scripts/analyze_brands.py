"""
Phase 1: Brand analysis for the twcs.csv dataset.

Vectorized analysis of the full Customer Support on Twitter dataset:

1. Loads the full dataset once.
2. Handles malformed/missing rows.
3. Identifies brand (support) accounts.
4. Computes per-brand support statistics.
5. Reconstructs conversation/thread relationships.
6. Produces a brand selection summary + keyword topics.

Outputs (saved to data/analysis/):
  brand_analysis.json          raw per-brand statistics
  brand_selection_summary.json ranked comparison table
  brand_summary_table.csv      CSV version of the summary table
  brand_keywords.json          top keywords per top brand
  conversation_stats.json      conversation reconstruction stats
  sample_threads/*.txt         sample customer->brand pairs for top brands
"""
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path("data/raw/twcs.csv")
OUTPUT_DIR = Path("data/analysis")

STOPWORDS = {
    "a", "an", "and", "are", "about", "all", "am", "any", "as", "at", "back",
    "be", "been", "but", "by", "can", "could", "did", "do", "for", "from",
    "get", "going", "got", "had", "has", "have", "he", "help", "her", "here",
    "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "just",
    "like", "me", "mine", "more", "my", "no", "not", "now", "of", "on", "one",
    "only", "or", "our", "out", "see", "she", "so", "some", "still", "than",
    "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "those", "to", "up", "us", "very", "was", "we", "were", "what", "when",
    "where", "which", "who", "will", "with", "would", "you", "your",
    "dont", "didnt", "cant", "wont", "im", "ive", "youre", "thats", "it's",
    "u", "ur", "plz", "please", "need", "want",
}


def load_data() -> pd.DataFrame:
    """Load the full dataset once."""
    t0 = time.time()
    if not DATA_PATH.exists():
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        sys.exit(1)
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows in {time.time() - t0:.1f}s")
    return df


def compute_author_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-author tweet counts (total / inbound / outbound)."""
    stats = pd.DataFrame(
        {
            "total_tweets": df.groupby("author_id")["tweet_id"].size(),
            "inbound_tweets": df[df["inbound"]].groupby("author_id")["tweet_id"].size(),
            "outbound_tweets": df[~df["inbound"]].groupby("author_id")["tweet_id"].size(),
        }
    ).fillna(0.0)
    return stats


def compute_reply_relationships(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Map tweet_id -> author and tweet_id -> inbound flag."""
    tweet_author = pd.Series(df["author_id"].values, index=df["tweet_id"].values)
    tweet_inbound = pd.Series(df["inbound"].values, index=df["tweet_id"].values)
    return tweet_author, tweet_inbound


def reconstruct_conversations(df: pd.DataFrame) -> pd.Series:
    """
    Reconstruct conversation IDs from reply chains.

    A conversation is the set of tweets connected through
    in_response_to_tweet_id edges. We assign the ROOT tweet id as the
    conversation id (the tweet that started the thread within the dataset).
    Returns a Series tweet_id -> conversation_id (root tweet id).
    """
    t0 = time.time()
    has_parent = df.dropna(subset=["in_response_to_tweet_id"])
    parent = (
        has_parent.set_index("tweet_id")["in_response_to_tweet_id"]
        .astype("int64")
        .to_dict()
    )
    print(f"  parent links: {len(parent):,}")

    root_memo: dict[int, int] = {}
    n_cycles = 0

    def find_root(tid: int) -> int:
        nonlocal n_cycles
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
            nxt = parent[cur]
            if nxt == cur:  # self-reference guard
                break
            cur = nxt
        else:
            if cur in seen:
                n_cycles += 1  # reply cycle in data — stop at current node
        root = cur
        for node in path:
            root_memo[node] = root
        root_memo[tid] = root
        return root

    conv_by_tweet = pd.Series(
        [find_root(int(t)) for t in df["tweet_id"]],
        index=df["tweet_id"].values,
    )
    if n_cycles:
        print(f"  WARNING: detected {n_cycles} reply cycles, resolved defensively")
    print(f"  conversations reconstructed in {time.time() - t0:.1f}s "
          f"({conv_by_tweet.nunique():,} distinct conversations)")
    return conv_by_tweet


def analyze(df: pd.DataFrame, conv_by_tweet: pd.Series) -> dict:
    tweet_author, tweet_inbound = compute_reply_relationships(df)
    author_stats = compute_author_stats(df)

    has_parent = df.dropna(subset=["in_response_to_tweet_id"]).copy()
    has_parent["parent_tid"] = has_parent["in_response_to_tweet_id"].astype("int64")
    has_parent["parent_inbound"] = (
        has_parent["parent_tid"].map(tweet_inbound).fillna(False).astype(bool)
    )
    has_parent["parent_author"] = has_parent["parent_tid"].map(tweet_author)

    # --- brand replies directly to a customer message (usable pair) ---
    # outbound tweet R, written by author B, whose parent is an inbound
    # customer message C  =>  (customer C -> brand B response R)
    brand_to_customer = has_parent[~has_parent["inbound"] & has_parent["parent_inbound"]]  # type: ignore[operator]
    pair_count_by_author = brand_to_customer.groupby("author_id")["tweet_id"].size()

    # --- customer messages replying to a brand tweet ---
    customer_to_company = has_parent[has_parent["inbound"] & ~has_parent["parent_inbound"]]  # type: ignore[operator]
    mentions_to_brand = customer_to_company.groupby("parent_author")["tweet_id"].size()

    # --- conversation counts per author ---
    df = df.copy()
    df["conv"] = df["tweet_id"].map(conv_by_tweet)
    conv_count_by_author = df.groupby("author_id")["conv"].nunique()
    inbound_pct_by_author = df[df["inbound"]].groupby("author_id")["conv"].nunique()

    # --- brand selection: authors who respond to customers ---
    # A support brand replies to customer messages. Candidate signal:
    #   outbound replies-to-customers >= 20
    #   ratio of replies-to-customers over total outbound tweets is meaningful
    brand_rows = []
    for author in pair_count_by_author.index:
        pairs = int(pair_count_by_author[author])
        if pairs < 20:
            continue
        outb = int(author_stats.loc[author, "outbound_tweets"])
        brand_rows.append(
            {
                "author_id": author,
                "total_tweets": int(author_stats.loc[author, "total_tweets"]),
                "outbound_tweets": outb,
                "inbound_tweets": int(author_stats.loc[author, "inbound_tweets"]),
                "brand_replies_to_customer": pairs,
                "customer_replies_to_brand": int(mentions_to_brand.get(author, 0)),
                "conversations": int(conv_count_by_author.get(author, 0)),
                "conversations_with_customer_msg": int(
                    inbound_pct_by_author.get(author, 0)
                ),
                "responsive_ratio": round(pairs / max(outb, 1), 3),
            }
        )

    brand_df = pd.DataFrame(brand_rows)
    brand_df = brand_df.sort_values("brand_replies_to_customer", ascending=False)
    brand_df = brand_df.reset_index(drop=True)
    return brand_df


def keyword_analysis(df: pd.DataFrame, brand_df: pd.DataFrame, top_n: int = 8) -> dict:
    """Top keywords for the top-n brands from customer messages replying to them."""
    tweet_author, tweet_inbound = compute_reply_relationships(df)
    has_parent = df.dropna(subset=["in_response_to_tweet_id"]).copy()
    has_parent["parent_tid"] = has_parent["in_response_to_tweet_id"].astype("int64")
    has_parent["parent_inbound"] = (
        has_parent["parent_tid"].map(tweet_inbound).fillna(False).astype(bool)
    )
    has_parent["parent_author"] = has_parent["parent_tid"].map(tweet_author)

    customer_msgs = has_parent[has_parent["inbound"] & ~has_parent["parent_inbound"]]
    customer_msgs = customer_msgs[~customer_msgs["parent_author"].isna()]

    top_brands = brand_df.head(top_n)["author_id"].tolist()
    results: dict[str, dict] = {}

    for brand in top_brands:
        msgs = customer_msgs[customer_msgs["parent_author"] == brand]
        texts = msgs["text"].str.lower()
        counter: Counter[str] = Counter()
        # Vectorized word extraction on a bounded sample
        sample = texts.head(8000).tolist()
        for t in sample:
            for w in re.findall(r"[a-z']+", t):
                if w not in STOPWORDS and len(w) > 2:
                    counter[w] += 1
        results[brand] = {
            "customer_msgs": int(len(msgs)),
            "top_keywords": counter.most_common(20),
        }
    return results


def save_samples(df: pd.DataFrame, brand_df: pd.DataFrame, top_n: int = 5) -> None:
    """Save readable sample customer->brand pairs for the top brands."""
    tweet_author, tweet_inbound = compute_reply_relationships(df)
    has_parent = df.dropna(subset=["in_response_to_tweet_id"]).copy()
    has_parent["parent_tid"] = has_parent["in_response_to_tweet_id"].astype("int64")
    has_parent["parent_inbound"] = (
        has_parent["parent_tid"].map(tweet_inbound).fillna(False).astype(bool)
    )
    has_parent["parent_author"] = has_parent["parent_tid"].map(tweet_author)
    tweet_text = pd.Series(df["text"].values, index=df["tweet_id"].values)
    has_parent["parent_text"] = has_parent["parent_tid"].map(tweet_text)

    out_dir = OUTPUT_DIR / "sample_threads"
    out_dir.mkdir(parents=True, exist_ok=True)

    top_brands = brand_df.head(top_n)["author_id"].tolist()
    for brand in top_brands:
        # brand response R replying to customer message C
        pairs = has_parent[
            (~has_parent["inbound"]) & has_parent["parent_inbound"]  # type: ignore[operator]
            & (has_parent["author_id"] == brand)
            & has_parent["parent_text"].notna()
        ]
        lines = [f"SAMPLE THREADS FOR {brand}\n", "=" * 60, ""]
        for _, r in pairs.head(10).iterrows():
            parent_text = str(r["parent_text"]).replace("\n", " ")[:280]
            reply_text = str(r["text"]).replace("\n", " ")[:280]
            lines.append(f"CUSTOMER:  {parent_text}")
            lines.append(f"BRAND:     {reply_text}")
            lines.append("----")
        (out_dir / f"{brand}.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    # --- basic integrity stats ---
    integrity = {
        "total_rows": int(len(df)),
        "unique_authors": int(df["author_id"].nunique()),
        "unique_tweets": int(df["tweet_id"].nunique()),
        "duplicate_tweet_ids": int(df["tweet_id"].duplicated().sum()),
    }
    missing = df.isnull().sum().to_dict()
    integrity["missing_by_column"] = {
        k: int(v) for k, v in missing.items()
    }
    print(f"\nIntegrity stats: {json.dumps(integrity, indent=2)}")

    # --- conversation reconstruction ---
    print("\nReconstructing conversations...")
    conv_by_tweet = reconstruct_conversations(df)
    total_convs = int(conv_by_tweet.nunique())
    single_tweet_convs = int(
        (conv_by_tweet.value_counts() == 1).sum()
    )
    conv_stats = {
        "total_conversations": total_convs,
        "single_tweet_conversations": single_tweet_convs,
        "multi_tweet_conversations": total_convs - single_tweet_convs,
        "median_tweets_per_conversation": int(
            conv_by_tweet.value_counts().median()
        ),
    }
    with open(OUTPUT_DIR / "conversation_stats.json", "w", encoding="utf-8") as f:
        json.dump(conv_stats, f, indent=2)
    print(f"Conversation stats: {conv_stats}")

    # --- brand analysis ---
    print("\nComputing brand statistics...")
    brand_df = analyze(df, conv_by_tweet)
    print(f"\nIdentified {len(brand_df):,} brand candidates.")

    print("\n=== TOP 20 BRANDS BY SUPPORT REPLY VOLUME ===")
    print(f"{'#':<3} {'author_id':<25} {'outbound':>9} {'pairs':>7} "
          f"{'mentions':>9} {'convs':>8} {'resp_ratio':>10}")
    for i, r in brand_df.head(20).iterrows():
        print(f"{i+1:<3} {r['author_id']:<25} {r['outbound_tweets']:>9,} "
              f"{r['brand_replies_to_customer']:>7,} "
              f"{r['customer_replies_to_brand']:>9,} "
              f"{r['conversations']:>8,} {r['responsive_ratio']:>10.2f}")

    brand_df.to_csv(OUTPUT_DIR / "brand_summary_table.csv", index=False)
    brand_df.head(50).to_json(
        OUTPUT_DIR / "brand_analysis.json", orient="records", indent=2
    )

    # --- selection score ---
    # Score volume of usable support + landmark-ness in this dataset.
    brand_df["selection_score"] = 0.0
    for i, r in brand_df.iterrows():
        score = (
            min(r["brand_replies_to_customer"], 50_000) * 0.4
            + min(r["customer_replies_to_brand"], 50_000) * 0.3
            + min(r["conversations"], 10_000) * 0.3
        )
        brand_df.loc[i, "selection_score"] = round(score, 1)
    brand_df = brand_df.sort_values("selection_score", ascending=False).reset_index(
        drop=True
    )

    print("\n=== TOP 10 BRANDS BY SELECTION SCORE ===")
    print(f"{'#':<3} {'author_id':<25} {'pairs':>8} {'mentions':>9} "
          f"{'convs':>8} {'score':>8}")
    for i, r in brand_df.head(10).iterrows():
        print(f"{i+1:<3} {r['author_id']:<25} {r['brand_replies_to_customer']:>8,} "
              f"{r['customer_replies_to_brand']:>9,} {r['conversations']:>8,} "
              f"{r['selection_score']:>8.1f}")

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "integrity": integrity,
        "conversation_stats": conv_stats,
        "top10": brand_df.head(10).to_dict(orient="records"),
    }
    with open(OUTPUT_DIR / "brand_selection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # --- keyword analysis for top 8 ---
    print("\nKeyword analysis for top brands (this may take ~30s)...")
    keywords = keyword_analysis(df, brand_df, top_n=8)
    with open(OUTPUT_DIR / "brand_keywords.json", "w", encoding="utf-8") as f:
        json.dump(keywords, f, indent=2, ensure_ascii=False)
    for brand, info in keywords.items():
        kws = ", ".join(f"{w}({c})" for w, c in info["top_keywords"][:15])
        print(f"  {brand:<25} n={info['customer_msgs']:>8,} | {kws}")

    # --- sample threads ---
    print("\nSaving sample threads...")
    save_samples(df, brand_df, top_n=5)

    print(f"\nTotal runtime: {time.time() - t_start:.1f}s")
    print(f"\nSaved analysis to {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()