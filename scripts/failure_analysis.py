"""Phase 15: failure analysis on the golden set.

Reads results/judge_rows.csv (full-pipeline per-example outputs) and reports:
- intent error: most common (human_intent -> pred_intent) confusions with samples
- escalation false positives and false negatives with their stated reasons
- reply failures: auto-served examples whose reply scored 'poor' vs the gold
Outputs: results/failure_analysis.json + results/failure_analysis.md (human-readable)
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROWS_PATH = Path("results/judge_rows.csv")
OUT_JSON = Path("results/failure_analysis.json")
OUT_MD = Path("results/failure_analysis.md")

REPLY_SAMPLE = 8


def main() -> None:
    df = pd.read_csv(ROWS_PATH)

    intent_err = df[~df["intent_correct"]]
    confusions = Counter(
        (r.human_intent, r.pred_intent) for r in intent_err.itertuples()
    )
    esc_fp = df[~df["human_escalate"] & df["pred_escalate"]]
    esc_fn = df[df["human_escalate"] & ~df["pred_escalate"]]

    auto = df[df["reply_verdict"].fillna("") != ""]
    reply_bad = auto[auto["reply_verdict"] == "poor"]

    results = {
        "n_examples": int(len(df)),
        "intent_errors": {
            "count": int(len(intent_err)),
            "rate": float(intent_err["id"].count() / len(df)),
            "top_confusions": [
                {"human": h, "pred": p, "count": c}
                for (h, p), c in confusions.most_common(12)
            ],
            "samples": [
                {
                    "id": r.id,
                    "human_intent": r.human_intent,
                    "pred_intent": r.pred_intent,
                    "confidence": float(r.confidence),
                }
                for r in intent_err.head(12).itertuples()
            ],
        },
        "escalation_false_positives": {
            "count": int(len(esc_fp)),
            "rate": float(len(esc_fp) / len(df)),
            "reasons": Counter(esc_fp["escalation_reason"].tolist()).most_common(),
            "sample_ids": esc_fp["id"].head(10).tolist(),
        },
        "escalation_false_negatives": {
            "count": int(len(esc_fn)),
            "rate": float(len(esc_fn) / len(df)),
            "reasons": Counter(esc_fn["escalation_reason"].tolist()).most_common(),
            "sample_ids": esc_fn["id"].head(10).tolist(),
        },
        "reply_failures": {
            "count": int(len(reply_bad)),
            "rate_among_auto": float(len(reply_bad) / len(auto)) if len(auto) else None,
            "mean_containment": float(reply_bad["containment"].mean()),
            "mean_cosine": float(reply_bad["cosine"].mean()),
            "samples": [
                {
                    "id": r.id,
                    "confidence": float(r.confidence),
                    "containment": float(r.containment),
                    "cosine": float(r.cosine),
                }
                for r in reply_bad.head(REPLY_SAMPLE).itertuples()
            ],
        },
    }
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md = [f"# Failure Analysis", "", f"- examples: {len(df)}", ""]
    md.append("## Intent confusions (human -> predicted)")
    for c in results["intent_errors"]["top_confusions"]:
        md.append(f"- {c['human']} -> {c['pred']}: {c['count']}")
    md.append("")
    md.append(f"## Escalation false positives: {results['escalation_false_positives']['count']}")
    for reason, n in results["escalation_false_positives"]["reasons"]:
        md.append(f"- ({n}) {reason}")
    md.append("")
    md.append(f"## Escalation false negatives: {results['escalation_false_negatives']['count']}")
    for reason, n in results['escalation_false_negatives']['reasons']:
        md.append(f"- ({n}) {reason}")
    md.append("")
    md.append(
        f"## Reply failures (poor vs gold): {results['reply_failures']['count']} "
        f"({results['reply_failures']['rate_among_auto']:.1%} of auto)"
    )
    OUT_MD.write_text("\n".join(md), encoding="utf-8")

    print(f"Intent errors: {results['intent_errors']['count']} "
          f"({results['intent_errors']['rate']:.1%})")
    print(f"Escalation FP={results['escalation_false_positives']['count']}  "
          f"FN={results['escalation_false_negatives']['count']}")
    print(f"Reply poor: {results['reply_failures']['count']} of {len(auto)} auto")
    print("Top confusions:")
    for c in results["intent_errors"]["top_confusions"][:8]:
        print(f"  {c['human']} -> {c['pred']}: {c['count']}")


if __name__ == "__main__":
    main()