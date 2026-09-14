"""Phase 9-10: retrieval kNN classifier on the golden set.

Predicts the intent of a golden message as the majority weak-label among its
top-k nearest neighbors in the retrieval index (tie -> closest neighbour).
The index excludes every golden conversation, so predictions are leak-free.
Outputs: results/retrieval_knn_results.json
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.metrics import confusion_matrix as sk_cm

sys.path.insert(0, str(Path("src").resolve()))
from retrieval import Retriever  # noqa: E402

GOLDEN_PATH = Path("data/golden/golden_set.csv")
OUTPUT_PATH = Path("results/retrieval_knn_results.json")


def predict(r: Retriever, text: str, k: int = 5) -> tuple[str, float, list[dict]]:
    hits = r.search(text, k=k)
    counts = Counter(h["weak_label"] for h in hits)
    most = counts.most_common()[0]
    ties = [label for label, c in counts.items() if c == most[1]]
    if len(ties) > 1:
        for h in hits:  # tie-break: closest neighbour wins
            if h["weak_label"] in ties:
                return hits[0]["weak_label"], hits[0]["score"], hits
    return most[0], max(h["score"] for h in hits), hits


def main() -> None:
    t0 = time.time()
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    golden = pd.read_csv(GOLDEN_PATH)
    r = Retriever.load()

    preds: list[str] = []
    scores: list[float] = []
    for text in golden["text"]:
        intent, score, _ = predict(r, text)
        preds.append(intent)
        scores.append(score)

    labels = golden["intent"].tolist()
    cm_labels = sorted(set(labels))
    cm = sk_cm(labels, preds, labels=cm_labels)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)

    results = {
        "model": "retrieval_knn_top5",
        "k": 5,
        "n_index_documents": int(r.config["n_documents"]),
        "golden_size": int(len(golden)),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "per_class_f1": {
            k: {
                "precision": float(v["precision"]),
                "recall": float(v["recall"]),
                "f1": float(v["f1-score"]),
                "support": int(v["support"]),
            }
            for k, v in report.items()
            if k in cm_labels
        },
        "confusion_matrix": {l: row.tolist() for l, row in zip(cm_labels, cm)},
        "runtime_seconds": round(time.time() - t0, 1),
    }
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Accuracy: {results['accuracy']:.3f}  "
          f"Macro-F1: {results['macro_f1']:.3f}  Weighted-F1: {results['weighted_f1']:.3f}")
    print("Per-class F1:")
    for intent, c in results["per_class_f1"].items():
        print(f"  {intent:<24}  F1={c['f1']:.3f}  support={c['support']}")


if __name__ == "__main__":
    main()