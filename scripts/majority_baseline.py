"""Phase 5: majority-class baseline.

The majority class is the most frequent `first_pass_label` in spotify_pairs.csv
after excluding every conversation that appears in the golden set.

Evaluation is performed on golden_set.csv (200 reviewed examples).
Outputs: results/majority_results.json
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path("src").resolve()))
from rule_labeler import first_pass_intent  # noqa: E402

GOLDEN_PATH = Path("data/golden/golden_set.csv")
PAIRS_PATH = Path("data/processed/spotify_pairs.csv")
OUTPUT_PATH = Path("results/majority_results.json")


def main() -> None:
    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    golden = pd.read_csv(GOLDEN_PATH)
    golden_conv_ids = set(golden["conversation_id"].unique())

    pairs = pd.read_csv(PAIRS_PATH)
    train = pairs[~pairs["conversation_id"].isin(golden_conv_ids)].copy()
    print(f"Golden convs: {len(golden_conv_ids)}, train pairs: {len(train)}")

    train["first_pass_label"] = train["customer_text_clean"].apply(
        lambda t: first_pass_intent(t).best_intent
    )
    majority_intent = train["first_pass_label"].value_counts().idxmax()
    majority_count = int(train["first_pass_label"].value_counts().iloc[0])
    intent_counts = train["first_pass_label"].value_counts().to_dict()
    print(f"Majority class: {majority_intent} ({majority_count} / {len(train)})")

    preds = [majority_intent] * len(golden)
    labels = golden["intent"].tolist()

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
        confusion_matrix,
    )

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    per_class = classification_report(labels, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=sorted(set(labels)))
    cm_labels = sorted(set(labels))

    results = {
        "baseline": "majority_class",
        "train_size": int(len(train)),
        "golden_size": int(len(golden)),
        "majority_intent": majority_intent,
        "majority_count_in_train": majority_count,
        "train_intent_distribution": {k: int(v) for k, v in intent_counts.items()},
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
            for k, v in per_class.items()
            if k in cm_labels
        },
        "confusion_matrix": {l: row.tolist() for l, row in zip(cm_labels, cm)},
    }
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Accuracy: {acc:.3f}  Macro-F1: {macro_f1:.3f}  Weighted-F1: {weighted_f1:.3f}")
    print("Per-class F1:")
    for intent in cm_labels:
        c = per_class[intent]
        print(f"  {intent:<24}  F1={c['f1-score']:.3f}  support={int(c['support'])}")


if __name__ == "__main__":
    main()