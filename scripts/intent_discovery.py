"""
Phase 3, part A: Exploratory topic discovery for SpotifyCares customer messages.

Empirical analysis (no LLM required):
  1. keyword + bigram frequency from cleaned customer messages
  2. TF-IDF -> NMF topic model (k topics) to surface recurring problem groups
  3. top representative customer messages per topic for manual review

Outputs (data/analysis/):
  intent_discovery_keywords.json
  intent_discovery_topics.json
  intent_discovery_topic_examples.txt   (human-readable)
"""
import json
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

PAIRS_PATH = Path("data/processed/spotify_pairs.csv")
OUT_DIR = Path("data/analysis")
N_TOPICS = 14
MAX_DOCS = 40_000
RANDOM_SEED = 42

STOPWORDS_EXTRA = set(
    """
    get got make made want need can could would should have has had didnt dont
    cant won't ur u im ive youre thats etc use using users spotify cares care
    support app just really like one two im going still please help issue issues
    problem problems way thing things stuff something anything nothing every any
    yeah yes no so me my you your they them their with without from into about
    what when where why who how does did do is are be was were been being am
    much many more most very too also just even only then now today yesterday
    thanks thank sorry lol haha hey hi hello we'll weve its it's okay fine work
    works working trying tried try being have having make making best better
    new old last first second big small great good bad worse wrong better
    """.split()
)


def clean_for_stats(text: str) -> str:
    t = re.sub(r"@\w+", " ", text)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"[^a-z '\-]", " ", text.lower())
    return re.sub(r"\s+", " ", t).strip()


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PAIRS_PATH)
    print(f"Loaded {len(df):,} pairs")

    texts = df["customer_text_clean"].dropna().head(MAX_DOCS)
    print(f"Using {len(texts):,} customer messages")

    # --- keyword + bigram frequency ---
    unigrams: Counter[str] = Counter()
    bigrams: Counter[tuple[str, str]] = Counter()
    for t in texts.map(clean_for_stats):
        toks = t.split()
        for w in toks:
            if w not in STOPWORDS_EXTRA and len(w) > 2:
                unigrams[w] += 1
        for i in range(len(toks) - 1):
            a, b = toks[i], toks[i + 1]
            if a not in STOPWORDS_EXTRA and b not in STOPWORDS_EXTRA and len(a) > 2 and len(b) > 2:
                bigrams[(a, b)] += 1

    print("\n--- Top 40 unigrams ---")
    for w, c in unigrams.most_common(40):
        print(f"  {w:<22} {c:>7,}")
    print("\n--- Top 40 bigrams ---")
    for (a, b), c in bigrams.most_common(40):
        print(f"  {a} {b:<20} {c:>7,}")

    # --- TF-IDF + NMF ---
    print(f"\nFitting TF-IDF + NMF ({N_TOPICS} topics)...")
    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.9,
        sublinear_tf=True,
        stop_words="english",
    )
    X = vectorizer.fit_transform(texts)
    print(f"  vocab size: {len(vectorizer.get_feature_names_out()):,}")
    print(f"  matrix shape: {X.shape}")

    nmf = NMF(n_components=N_TOPICS, init="nndsvda", random_state=RANDOM_SEED, max_iter=300)
    W = nmf.fit_transform(X)
    H = nmf.components_
    feat_names = vectorizer.get_feature_names_out()

    topics = []
    for topic_idx, topic in enumerate(H):
        top_idx = topic.argsort()[::-1][:15]
        top_terms = [feat_names[i] for i in top_idx]
        # representative documents for this topic
        doc_scores = W[:, topic_idx]
        top_docs_idx = doc_scores.argsort()[::-1][:8]
        examples = []
        for di in top_docs_idx:
            if doc_scores[di] > 0.15:
                examples.append(str(texts.iloc[di])[:200])
        topics.append(
            {
                "topic_id": topic_idx,
                "top_terms": top_terms,
                "share_explained": float(doc_scores.sum() / W.sum()),
                "examples": examples,
            }
        )
        print(f"\nTopic {topic_idx:>2} | terms: {', '.join(top_terms[:10])}")
        for ex in examples:
            print(f"    - {ex}")

    with open(OUT_DIR / "intent_discovery_keywords.json", "w", encoding="utf-8") as f:
        json.dump(
            {"unigrams": unigrams.most_common(100), "bigrams": bigrams.most_common(100)},
            f, indent=2, ensure_ascii=False,
        )
    with open(OUT_DIR / "intent_discovery_topics.json", "w", encoding="utf-8") as f:
        json.dump(topics, f, indent=2, ensure_ascii=False)

    lines = ["INTENT DISCOVERY TOPICS", "=" * 70]
    for tp in topics:
        lines.append("")
        lines.append(f"Topic {tp['topic_id']}: {', '.join(tp['top_terms'])}")
        for ex in tp["examples"]:
            lines.append(f"  - {ex}")
    (OUT_DIR / "intent_discovery_topic_examples.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print(f"\nSaved keyword + topic outputs to {OUT_DIR}")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()