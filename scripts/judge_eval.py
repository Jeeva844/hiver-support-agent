"""Phase 14: judge evaluation + judge<->human agreement.

Runs the full pipeline on the golden set, then the Judge, and reports:
- intent agreement (judge/human) = intent accuracy
- escalation agreement = accuracy + Cohen's kappa vs human should_escalate
- reply adequacy distribution for auto-served messages vs the historical gold
  reply (the actual SpotifyCares response to that same conversation)
- safe-to-auto rate
Outputs: results/judge_results.json and results/judge_rows.csv
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path("src").resolve()))
from judge import Judge  # noqa: E402
from pipeline import Pipeline  # noqa: E402
from retrieval import Retriever  # noqa: E402

GOLDEN_PATH = Path("data/golden/golden_set.csv")
PAIRS_PATH = Path("data/processed/spotify_pairs.csv")
OUTPUT_PATH = Path("results/judge_results.json")
ROWS_PATH = Path("results/judge_rows.csv")


def main() -> None:
    t0 = time.time()
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    golden = pd.read_csv(GOLDEN_PATH)
    pairs = pd.read_csv(PAIRS_PATH)

    # historical gold reply per golden conversation (what SpotifyCares actually said)
    gold_reply = (
        pairs.sort_values("timestamp")
        .drop_duplicates("conversation_id", keep="first")
        .set_index("conversation_id")["brand_response_clean"]
        .to_dict()
    )

    pipe = Pipeline()
    judge = Judge(Retriever.load())

    records = []
    for _, row in golden.iterrows():
        result = pipe.run(row["text"])
        v = judge.evaluate(
            result,
            row["text"],
            row["intent"],
            bool(row["should_escalate"]),
            gold_reply.get(int(row["conversation_id"])),
        )
        records.append(
            {
                "id": row["id"],
                "human_intent": row["intent"],
                "pred_intent": result["intent"],
                "human_escalate": bool(row["should_escalate"]),
                "pred_escalate": bool(result["escalate"]),
                "confidence": result["intent_confidence"],
                "intent_correct": v["intent_correct"],
                "escalation_agreement": v["escalation_agreement"],
                "safe_to_auto": v["safe_to_auto"],
                "reply_verdict": v["reply_verdict"] or "",
                "containment": (v["reply_metrics"] or {}).get("containment"),
                "cosine": (v["reply_metrics"] or {}).get("cosine"),
                "reply": result["reply"] or "",
                "escalation_reason": result["escalation_reason"] or "",
            }
        )
    df = pd.DataFrame(records)
    df.to_csv(ROWS_PATH, index=False)

    esc_agree = float(df["escalation_agreement"].mean())
    intent_agree = float(df["intent_correct"].mean())
    kappa = float(cohen_kappa_score(df["human_escalate"], df["pred_escalate"]))
    safe_auto = float(df["safe_to_auto"].mean())
    auto = df[df["reply_verdict"] != ""]
    reply_good_rate = float((auto["reply_verdict"] == "good").mean()) if len(auto) else None

    results = {
        "judge_backend": "deterministic_mock",
        "judge_human_agreement": {
            "intent_agreement": intent_agree,
            "escalation_agreement": esc_agree,
            "escalation_cohen_kappa": kappa,
            "safe_to_auto_rate": safe_auto,
        },
        "reply_adequacy": {
            "auto_examples_with_gold": int(len(auto)),
            "good_rate": reply_good_rate,
            "mean_containment": float(auto["containment"].mean()),
            "mean_cosine": float(auto["cosine"].mean()),
        },
        "human_escalate_rate": float(df["human_escalate"].mean()),
        "model_escalate_rate": float(df["pred_escalate"].mean()),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Judge<->human: intent={intent_agree:.3f} escalation={esc_agree:.3f} "
          f"(kappa={kappa:.3f}) safe_to_auto={safe_auto:.3f}")
    print(f"Reply good rate (auto, n={len(auto)}): {reply_good_rate:.3f} "
          f"containment={results['reply_adequacy']['mean_containment']:.3f} "
          f"cosine={results['reply_adequacy']['mean_cosine']:.3f}")


if __name__ == "__main__":
    main()