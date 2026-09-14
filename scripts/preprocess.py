"""
Phase 2: Build the processed SpotifyCares (customer_text, brand_response) dataset.

Steps:
1. Load twcs.csv, restrict to conversations involving SpotifyCares.
2. Reconstruct conversation ids from reply chains (root tweet id).
3. Extract usable pairs: brand outbound reply R directly replying to a
   customer inbound message C  =>  (customer_text=C.text, brand_response=R.text).
4. Clean (non-destructive): HTML entities, URLs, leading mentions, agent-signature
   initials, stray variation selectors. Originals are preserved alongside.
5. Save data/processed/spotify_pairs.csv + conversation metadata + report.

We deliberately do NOT aggressively clean: typos, slang, emotion are kept.
"""
import html
import json
import re
import time
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/raw/twcs.csv")
OUT_DIR = Path("data/processed")
BRAND = "SpotifyCares"

# Prefix / signature tokens appended by agents (e.g. "... Keep us posted /LS")
TRAILING_SIG = re.compile(r"/\s*[A-Za-z0-9]{1,3}\s*$")
INLINE_SIG = re.compile(r"\s+/\s*[A-Za-z0-9]{1,3}(?=\s|$)")
MENTION = re.compile(r"@[\w]+")
URL = re.compile(r"https?://\S+|www\.\S+")
WS = re.compile(r"\s+")
# Variation selector / invisible chars to drop (keeps emojis)
VARIATION_SELECTORS = re.compile(r"[\ufe00-\ufe0f\u200b-\u200d\ufeff]")


def parse_created_at(series: pd.Series) -> pd.Series:
    """Parse the Ruby-style timestamps used in this dataset."""
    return pd.to_datetime(series, format="%a %b %d %H:%M:%S %z %Y", utc=True)


def clean_tweet(text: str) -> str:
    """Normalise formatting noise. Keeps spelling, slang, emotion, case."""
    if not isinstance(text, str):
        return ""
    t = html.unescape(text)                 # &amp; &gt; &#39; ...
    t = t.replace("\u2026", "...")          # ellipsis char
    t = MENTION.sub(" ", t)                 # remove @handles
    t = URL.sub(" ", t)                     # remove links
    t = VARIATION_SELECTORS.sub("", t)      # invisible selectors
    # strip trailing agent signature initials ("... /LS", "... /CB")
    t = TRAILING_SIG.sub("", t)
    t = INLINE_SIG.sub(" ", t)
    t = WS.sub(" ", t)                      # collapse whitespace
    return t.strip()


def reconstruct_conversations(df: pd.DataFrame) -> pd.Series:
    """Map tweet_id -> conversation_id (root tweet id of the reply chain)."""
    has_parent = df.dropna(subset=["in_response_to_tweet_id"])
    parent = (
        has_parent.set_index("tweet_id")["in_response_to_tweet_id"]
        .astype("int64")
        .to_dict()
    )
    memo: dict[int, int] = {}

    def find_root(tid: int) -> int:
        root = memo.get(tid)
        if root is not None:
            return root
        path: list[int] = []
        seen: set[int] = set()
        cur = tid
        while cur in parent and cur not in seen:
            cached = memo.get(cur)
            if cached is not None:
                cur = cached
                break
            seen.add(cur)
            path.append(cur)
            cur = parent[cur]
        root = cur
        for node in path:
            memo[node] = root
        return root

    return pd.Series(
        [find_root(int(t)) for t in df["tweet_id"]], index=df["tweet_id"].values
    )


def main() -> None:
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows")

    # --- conversation ids for the whole dataset ---
    print("Reconstructing conversations...")
    conv_by_tweet = reconstruct_conversations(df)
    df = df.copy()
    df["conversation_id"] = df["tweet_id"].map(conv_by_tweet)

    # --- keep only conversations that involve SpotifyCares ---
    care_tweets = df[df["author_id"] == BRAND]
    care_convs = set(care_tweets["conversation_id"].unique())
    sub = df[df["conversation_id"].isin(care_convs)].copy()
    print(f"Brand tweets: {len(care_tweets):,} | "
          f"conversations involving brand: {len(care_convs):,} | "
          f"tweets in those conversations: {len(sub):,}")

    # --- usable pairs: brand outbound reply whose parent is a customer msg ---
    tweet_author = pd.Series(df["author_id"].values, index=df["tweet_id"].values)
    tweet_inbound = pd.Series(df["inbound"].values, index=df["tweet_id"].values)
    tweet_text = pd.Series(df["text"].values, index=df["tweet_id"].values)
    tweet_time = pd.Series(parse_created_at(df["created_at"]).values, index=df["tweet_id"].values)

    replies = sub.dropna(subset=["in_response_to_tweet_id"]).copy()
    replies["parent_tid"] = replies["in_response_to_tweet_id"].astype("int64")
    replies["parent_inbound"] = (
        replies["parent_tid"].map(tweet_inbound).fillna(False).astype(bool)
    )
    pairs = replies[replies["author_id"] == BRAND]
    pairs = pairs[pairs["inbound"] == False]           # brand only
    pairs = pairs[pairs["parent_inbound"] == True]     # parent is a customer message
    pairs = pairs[pairs["parent_tid"].isin(tweet_author.index)]

    print(f"Raw brand->customer reply pairs: {len(pairs):,}")

    out = pd.DataFrame(
        {
            "conversation_id": pairs["conversation_id"].values,
            "customer_tweet_id": pairs["parent_tid"].values,
            "response_tweet_id": pairs["tweet_id"].values,
            "customer_text": pairs["parent_tid"].map(tweet_text).values,
            "brand_response": pairs["text"].values,
            "timestamp": pairs["parent_tid"].map(tweet_time).values,
            "brand": BRAND,
        }
    )
    # Re-order timestamp values properly
    out["timestamp"] = out["customer_tweet_id"].map(tweet_time)

    # --- clean copies (originals preserved) ---
    out["customer_text_clean"] = out["customer_text"].map(clean_tweet)
    out["brand_response_clean"] = out["brand_response"].map(clean_tweet)

    n_before = len(out)
    # --- drop unusable rows ---
    bad_customer = out["customer_text"].fillna("").map(lambda s: len(str(s).strip()) == 0)
    bad_clean = out["customer_text_clean"].str.len().lt(3)
    dropped = {
        "empty_or_nan_customer_text": int((out["customer_text"].isna() | (out["customer_text"] == "")).sum()),
        "clean_customer_text_lt_3_chars": int(bad_clean.sum()),
        "empty_brand_response": int(out["brand_response"].fillna("").map(lambda s: len(str(s).strip()) == 0).sum()),
        "duplicate_customer_tweet_id": int(out["customer_tweet_id"].duplicated().sum()),
    }
    out = out[~out["customer_text"].fillna("").map(lambda s: len(str(s).strip()) == 0)]
    out = out[out["customer_text_clean"].str.len().ge(3)]
    out = out[~out["customer_text"].isna()]
    out = out.drop_duplicates(subset=["customer_tweet_id"])

    # --- remediation for brand responses that look like pure redirects ---
    # (kept in dataset, but flagged so retrieval/grounding can be reweighted later)
    dd_pattern = re.compile(
        r"\bsend us\b|\bprivate message\b|\bdm( us)?\b|\bcontact us via\b|"
        r"\breach us by\b|\bphone or chat\b",
        re.IGNORECASE,
    )
    out["is_redirect_reply"] = out["brand_response_clean"].str.contains(
        dd_pattern, regex=True
    )

    print(f"Dropped {n_before - len(out):,} rows -> {len(out):,} usable pairs")
    print(f"Redirect-style brand replies flagged: {int(out['is_redirect_reply'].sum()):,} "
          f"({out['is_redirect_reply'].mean()*100:.1f}%)")

    # --- sort by time ---
    out = out.sort_values("timestamp").reset_index(drop=True)

    out.to_csv(OUT_DIR / "spotify_pairs.csv", index=False, encoding="utf-8")

    # --- conversation metadata ---
    conv_meta = (
        sub.groupby("conversation_id")
        .agg(
            n_tweets=("tweet_id", "size"),
            first_time=("created_at", "min"),
            brand_has_outbound=("inbound", lambda s: (~s.astype(bool)).any()),
            n_customer_msgs=("inbound", lambda s: s.astype(bool).sum()),
        )
        .reset_index()
    )
    conv_meta.to_csv(OUT_DIR / "spotify_conversations.csv", index=False, encoding="utf-8")

    # --- report ---
    timezone_suffixes = sub[sub["author_id"] == BRAND]["text"].str.extract(
        r"(ET|PDT|PST|CST|EST)$"
    )[0]
    report = {
        "brand": BRAND,
        "data_path": str(DATA_PATH),
        "total_rows_full_dataset": int(len(df)),
        "brand_tweets": int(len(care_tweets)),
        "conversations_involving_brand": int(len(care_convs)),
        "tweets_in_those_conversations": int(len(sub)),
        "raw_brand_to_customer_pairs": int(len(pairs)),
        "usable_pairs": int(len(out)),
        "dropped": dropped,
        "redirect_reply_pct": round(float(out["is_redirect_reply"].mean()) * 100, 1),
        "distinct_conversations_with_pairs": int(out["conversation_id"].nunique()),
        "customer_text_len_mean": round(float(out["customer_text_clean"].str.len().mean()), 1),
        "customer_text_len_median": round(float(out["customer_text_clean"].str.len().median()), 1),
        "brand_response_len_mean": round(float(out["brand_response_clean"].str.len().mean()), 1),
        "brand_response_len_median": round(float(out["brand_response_clean"].str.len().median()), 1),
        "trailing_timezone_token_count_in_brand_replies": int(timezone_suffixes.notna().sum()),
        "time_range": [str(out["timestamp"].min()), str(out["timestamp"].max())],
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    with open(OUT_DIR / "preprocessing_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== Preprocessing report ===")
    print(json.dumps(report, indent=2))
    print(f"\nSaved pairs -> data/processed/spotify_pairs.csv")


if __name__ == "__main__":
    main()