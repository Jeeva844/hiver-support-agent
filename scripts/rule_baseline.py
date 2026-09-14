"""Phase 6: rule-based baseline.

The rule classifier (src/rule_labeler.first_pass_intent) is applied directly to
the golden-set messages. This is the same labeler used for sampling, evaluated
honestly on the reviewed golden set.
Outputs: results/rule_results.json
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.metrics import confusion_matrix as sk_cm

sys.path.insert(0, str(Path("src").resolve()))
from rule_labeler import first_pass_intent  # noqa: E402

GOLDEN_PATH = Path("data/golden/golden_set.csv")
OUTPUT_PATH = Path("results/rule_results.json")


def main() -> None:
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    golden = pd.read_csv(GOLDEN_PATH)

    preds = golden["text"].apply(lambda t: first_pass_intent(t).best_intent).tolist()
    labels = golden["intent"].tolist()
    cm_labels = sorted(set(labels))
    cm = sk_cm(labels, preds, labels=cm_labels)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)

    results = {
        "baseline": "rule_keyword_labeler",
        "golden_size": int(len(golden)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
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
    }
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Accuracy: {acc:.3f}  Macro-F1: {macro_f1:.3f}  Weighted-F1: {weighted_f1:.3f}")
    print("Per-class F1:")
    for intent in cm_labels:
        c = report[intent]
        print(f"  {intent:<24}  F1={c['f1-score']:.3f}  support={int(c['support'])}")


if __name__ == "__main__":
    main()