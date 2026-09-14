"""Phase 8: LLM intent classifier evaluated on the golden set.

Backend is chosen by LLMIntentClassifier (mock unless openai+key present).
Outputs: results/llm_classifier_results.json
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.metrics import confusion_matrix as sk_cm

sys.path.insert(0, str(Path("src").resolve()))
from llm_classifier import LLMIntentClassifier  # noqa: E402

GOLDEN_PATH = Path("data/golden/golden_set.csv")
OUTPUT_PATH = Path("results/llm_classifier_results.json")


def main() -> None:
    t0 = time.time()
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    golden = pd.read_csv(GOLDEN_PATH)

    clf = LLMIntentClassifier()
    print(f"Backend: {clf.backend}")

    preds = [clf.classify(t) for t in golden["text"]]
    pred_intents = [p["intent"] for p in preds]
    labels = golden["intent"].tolist()
    cm_labels = sorted(set(labels))

    cm = sk_cm(labels, pred_intents, labels=cm_labels)
    report = classification_report(labels, pred_intents, output_dict=True, zero_division=0)

    results = {
        "backend": clf.backend,
        "golden_size": int(len(golden)),
        "accuracy": float(accuracy_score(labels, pred_intents)),
        "macro_f1": float(f1_score(labels, pred_intents, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, pred_intents, average="weighted", zero_division=0)),
        "mean_confidence": float(sum(p["confidence"] for p in preds) / len(preds)),
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