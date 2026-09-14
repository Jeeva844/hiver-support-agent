"""Phase 13: unified pipeline evaluated on the golden set.

Metrics:
- intent: accuracy / macro-F1 / weighted-F1 vs human labels
- escalation: accuracy / precision / recall / F1 vs human should_escalate
- reply: for non-escalated (auto) messages that humans also marked auto,
  how often the retrieved reference's intent matches the human intent
  ('intent-consistent reply rate') and mean retrieval similarity.
Outputs: results/pipeline_results.json
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

sys.path.insert(0, str(Path("src").resolve()))
from pipeline import Pipeline  # noqa: E402

GOLDEN_PATH = Path("data/golden/golden_set.csv")
OUTPUT_PATH = Path("results/pipeline_results.json")


def main() -> None:
    t0 = time.time()
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    golden = pd.read_csv(GOLDEN_PATH)
    pipe = Pipeline()

    rows = []
    for text in golden["text"]:
        rows.append(pipe.run(text))
    out = pd.DataFrame(rows)

    labels_intent = golden["intent"].tolist()
    labels_esc = golden["should_escalate"].astype(bool).tolist()
    preds_intent = out["intent"].tolist()
    preds_esc = out["escalate"].astype(bool).tolist()

    intent_acc = accuracy_score(labels_intent, preds_intent)
    intent_macro = f1_score(labels_intent, preds_intent, average="macro", zero_division=0)
    intent_w = f1_score(labels_intent, preds_intent, average="weighted", zero_division=0)
    report = classification_report(labels_intent, preds_intent, output_dict=True, zero_division=0)

    from sklearn.metrics import confusion_matrix as sk_cm
    from sklearn.metrics import precision_recall_fscore_support as prf

    esc_prec, esc_rec, esc_f1, esc_sup = prf(labels_esc, preds_esc, average="binary", pos_label=True, zero_division=0)

    auto_mask = (~out["escalate"].astype(bool)) & (~golden["should_escalate"].astype(bool))
    intent_consistent = float(
        (out.loc[auto_mask, "intent"] == golden.loc[auto_mask, "intent"]).mean()
        if auto_mask.any()
        else float("nan")
    )

    results = {
        "pipeline": "classify -> escalate-or-resolve -> reply",
        "backend": pipe.classifier.backend,
        "golden_size": int(len(golden)),
        "intent_accuracy": float(intent_acc),
        "intent_macro_f1": float(intent_macro),
        "intent_weighted_f1": float(intent_w),
        "intent_report": {
            k: v for k, v in report.items() if isinstance(v, dict) and "f1-score" in v
        },
        "escalation": {
            "accuracy": float(accuracy_score(labels_esc, preds_esc)),
            "precision": float(esc_prec),
            "recall": float(esc_rec),
            "f1": float(esc_f1),
            "human_escalate_rate": float(golden["should_escalate"].mean()),
            "model_escalate_rate": float(out["escalate"].mean()),
        },
        "reply": {
            "auto_auto_pairs": int(auto_mask.sum()),
            "intent_consistent_rate": intent_consistent,
            "mean_similarity_auto": float(out.loc[auto_mask, "retrieval"].apply(
                lambda r: r["similarity"] if r else None
            ).mean()) if auto_mask.any() else None,
            "escalated_no_reply": int((~out["escalate"].astype(bool)).sum() == 0),
        },
        "runtime_seconds": round(time.time() - t0, 1),
    }
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Intent: acc={intent_acc:.3f} macro-F1={intent_macro:.3f} weighted-F1={intent_w:.3f}")
    print(f"Escalation: acc={results['escalation']['accuracy']:.3f} "
          f"P={esc_prec:.3f} R={esc_rec:.3f} F1={esc_f1:.3f}")
    print(f"Reply (auto&human-auto, n={auto_mask.sum()}): "
          f"intent-consistent={intent_consistent:.3f}")


if __name__ == "__main__":
    main()