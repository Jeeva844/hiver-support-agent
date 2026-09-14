"""Phase 7: TF-IDF + linear classifier baseline.

Trains on spotify_pairs.csv (labels = deterministic first_pass labeler, the
only labels available outside the golden set) with every golden conversation
excluded. Evaluates on the reviewed golden set.

Outputs: results/classifier_results.json
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.metrics import confusion_matrix as sk_cm
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path("src").resolve()))
from rule_labeler import first_pass_intent  # noqa: E402

GOLDEN_PATH = Path("data/golden/golden_set.csv")
PAIRS_PATH = Path("data/processed/spotify_pairs.csv")
OUTPUT_PATH = Path("results/classifier_results.json")


def main() -> None:
    t0 = time.time()
    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    golden = pd.read_csv(GOLDEN_PATH)
    golden_conv_ids = set(golden["conversation_id"].unique())
    pairs = pd.read_csv(PAIRS_PATH)
    train_df = pairs[~pairs["conversation_id"].isin(golden_conv_ids)].copy()
    print(f"Train pairs: {len(train_df)}")

    train_df["y"] = train_df["customer_text_clean"].apply(
        lambda t: first_pass_intent(t).best_intent
    )
    print("Train class distribution (weak labels):", dict(train_df["y"].value_counts()))

    vec = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=20000,
        sublinear_tf=True, strip_accents="unicode",
    )
    X_train = vec.fit_transform(train_df["customer_text_clean"])
    y_train = train_df["y"].values

    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    clf.fit(X_train, y_train)

    X_golden = vec.transform(golden["text"])
    preds = clf.predict(X_golden)
    labels = golden["intent"].tolist()
    cm_labels = sorted(set(labels))
    cm = sk_cm(labels, preds, labels=cm_labels)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)

    also_split = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_df["y"]
    )
    _, val_df = also_split
    X_val = vec.transform(val_df["customer_text_clean"])

    results = {
        "model": "tfidf+logreg (weak labels)",
        "n_train": int(len(train_df)),
        "golden_size": int(len(golden)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "internal_val_accuracy_weak_labels": float(
            accuracy_score(val_df["y"], clf.predict(X_val))
        ),
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

    print(f"Accuracy: {acc:.3f}  Macro-F1: {macro_f1:.3f}  Weighted-F1: {weighted_f1:.3f}")
    print("Per-class F1:")
    for intent in cm_labels:
        c = report[intent]
        print(f"  {intent:<24}  F1={c['f1-score']:.3f}  support={int(c['support'])}")


if __name__ == "__main__":
    main()