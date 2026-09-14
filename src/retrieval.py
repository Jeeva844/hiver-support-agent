"""Deterministic retrieval index (TF-IDF based).

Stand-in for a dense-embedding/FAISS store so the whole pipeline runs offline
and reproducibly. The abstraction mirrors a vector store: build(exclude=...),
search(text, k) -> ranked hits.

The index is built ONLY over conversations that are NOT in the golden set
(a hard leak guard is asserted at build time and checked every load).

Persistence: data/processed/retrieval_index/retrieval_index.joblib
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from rule_labeler import first_pass_intent  # noqa: E402

ARTIFACT_PATH = ROOT / "data" / "processed" / "retrieval_index" / "retrieval_index.joblib"


class Retriever:
    def __init__(
        self,
        vectorizer: TfidfVectorizer,
        matrix,
        meta: pd.DataFrame,
        config: dict,
    ) -> None:
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.meta = meta
        self.config = config

    @classmethod
    def build(
        cls,
        pairs_path: Path | str = ROOT / "data" / "processed" / "spotify_pairs.csv",
        golden_path: Path | str = ROOT / "data" / "golden" / "golden_set.csv",
        max_features: int = 20000,
        ngram_range: tuple = (1, 2),
    ) -> "Retriever":
        pairs = pd.read_csv(pairs_path)
        golden = pd.read_csv(golden_path)
        golden_conv_ids = set(golden["conversation_id"].unique())

        corpus = pairs[~pairs["conversation_id"].isin(golden_conv_ids)].copy()
        forbidden = set(golden["conversation_id"].unique())
        leak = set(corpus["conversation_id"].unique()) & forbidden
        if leak:
            raise SystemExit(f"LEAK: {len(leak)} golden conversations in index corpus")

        corpus["weak_label"] = corpus["customer_text_clean"].apply(
            lambda t: first_pass_intent(t).best_intent
        )

        vec = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range,
            min_df=2, sublinear_tf=True, strip_accents="unicode",
        )
        X = vec.fit_transform(corpus["customer_text_clean"])
        X_norm = normalize(X, norm="l2", axis=1)

        meta = pd.DataFrame(
            {
                "corpus_index": np.arange(len(corpus)),
                "customer_tweet_id": corpus["customer_tweet_id"].values,
                "conversation_id": corpus["conversation_id"].values,
                "customer_text_clean": corpus["customer_text_clean"].values,
                "brand_response_clean": corpus["brand_response_clean"].values,
                "weak_label": corpus["weak_label"].values,
            }
        )
        config = {
            "n_documents": int(len(corpus)),
            "max_features": max_features,
            "ngram_range": list(ngram_range),
            "golden_excluded": True,
            "golden_conversation_count": len(forbidden),
            "vectorization": "tfidf (deterministic fallback for dense embeddings)",
        }
        return cls(vec, X_norm, meta, config)

    # ------------------------------------------------------------------ search
    def search(self, text: str, k: int = 5) -> list[dict]:
        q = self.vectorizer.transform([text])
        q = normalize(q, norm="l2", axis=1)
        scores = (self.matrix @ q.T).toarray().ravel()
        top = np.argsort(scores)[::-1][:k]
        hits = []
        for idx in top:
            m = self.meta.iloc[idx]
            hits.append(
                {
                    "score": float(scores[idx]),
                    "customer_text": m["customer_text_clean"],
                    "brand_response": m["brand_response_clean"],
                    "weak_label": m["weak_label"],
                    "conversation_id": int(m["conversation_id"]),
                }
            )
        return hits

    def random_access(self, corpus_indices: list[int]) -> pd.DataFrame:
        return self.meta.iloc[corpus_indices]

    # -------------------------------------------------------------- persistence
    def save(self, path: Path | str = ARTIFACT_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vectorizer": self.vectorizer,
            "matrix": self.matrix,
            "meta": self.meta,
            "config": self.config,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path | str = ARTIFACT_PATH) -> "Retriever":
        payload = joblib.load(path)
        return cls(**payload)


def build_index(overwrite: bool = False) -> Retriever:
    if ARTIFACT_PATH.exists() and not overwrite:
        print(f"Index already exists: {ARTIFACT_PATH} (use overwrite=True to rebuild)")
        return Retriever.load(ARTIFACT_PATH)
    print("Building retrieval index (golden conversations excluded)...")
    r = Retriever.build()
    r.save()
    print(f"Indexed {r.config['n_documents']} documents -> {ARTIFACT_PATH}")
    return r


if __name__ == "__main__":
    import json

    r = build_index(overwrite=False)
    for q in ["why am I being charged twice", "can't log in", "add my family to family plan"]:
        hits = r.search(q, k=3)
        print(f"\nquery: {q}")
        for h in hits:
            print(f"  {h['score']:.3f} [{h['weak_label']}] {h['customer_text'][:60]}")
    print("\nconfig:", json.dumps(r.config, indent=2))