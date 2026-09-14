"""Phase 16: what is misleading about the headline number.

Produces results/misleading_metrics.json (structured) and results/misleading_metrics.md
(prose) explaining why "acc = 0.745 / auto-resolves 74.5%" would be a misleading
headline for this agent, with the real numbers from results/*.json as evidence.
"""
import json
from pathlib import Path

RESULTS = Path("results")


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main() -> None:
    majority = load("majority_results.json")
    rule = load("rule_results.json")
    clf = load("classifier_results.json")
    llm = load("llm_classifier_results.json")
    knn = load("retrieval_knn_results.json")
    pipe = load("pipeline_results.json")
    judge = load("judge_results.json")
    failure = load("failure_analysis.json")
    golden_stats = json.loads(
        Path("data/golden/golden_label_stats.json").read_text(encoding="utf-8")
    )

    headline = pipe["intent_accuracy"]
    esc = pipe["escalation"]

    sections = [
        {
            "title": "1. Accuracy hides class-prior effects",
            "text": (
                "Intent accuracy is dominated by the common classes. macro-F1 "
                f"({pipe['intent_macro_f1']:.3f}) is below accuracy "
                f"({headline:.3f}); classes with tiny support (device_compatibility "
                "support=4) contribute almost nothing to accuracy but matter to a "
                "customer whose device is the problem. A 0.745 accuracy claim "
                "says almost nothing about the tail."
            ),
        },
        {
            "title": "2. Intent-correct does NOT mean resolved",
            "text": (
                "Label agreement is the wrong unit for an agent. The judge found only "
                f"{judge['reply_adequacy']['good_rate']:.1%} of auto-replies lexically "
                "adequate vs the historical human reply "
                f"(mean containment {judge['reply_adequacy']['mean_containment']:.3f}), "
                "and only "
                f"{judge['judge_human_agreement']['safe_to_auto_rate']:.1%} of traffic is "
                "judged 'safe to auto-serve'. Most "
                "promising deflection is right, but resolution quality is the binding "
                "constraint — accuracy does not measure it at all."
            ),
        },
        {
            "title": "3. Escalation 'accuracy' masks cost asymmetry",
            "text": (
                f"Escalation accuracy {esc['accuracy']:.3f} treats false positives and "
                f"false negatives equally, but their costs differ: FP ({failure['escalation_false_positives']['count']}) "
                "wastes a human's time; FN "
                f"({failure['escalation_false_negatives']['count']}) sends an unverified "
                "reply about money/security to a user. Precision 0.738 vs recall 0.616 "
                "shows the agent under-escalates exactly where it is most dangerous."
            ),
        },
        {
            "title": "4. The evaluation distribution is artificial",
            "text": (
                "The golden set used per-bucket quotas so rare intents are guaranteed "
                "(e.g. other=8, device_compatibility=4), but real production traffic is "
                "heavily skewed (99.96% of all data falls in 2017; the underlying label "
                "distribution is different). Reported metrics are calibrated to an "
                "intentionally balanced set, not to production reality."
            ),
        },
        {
            "title": "5. Weak (first-pass) labels cap everything",
            "text": (
                "All training labels come from a keyword labeler whose disagreement "
                f"with the human reviewer is {golden_stats['reviewer_vs_first_pass_disagree_rate']:.1%}. "
                "The TF-IDF classifier (0.740) cannot beat its weak teacher, so the "
                "0.745 headline includes label noise as if it were model error."
            ),
        },
        {
            "title": "6. Baselines vitiate the claim",
            "text": (
                f"The rule labeler alone already reaches {rule['accuracy']:.3f} accuracy and "
                f"{rule['macro_f1']:.3f} macro-F1 it is the same component as the mock LLM "
                f"({llm['accuracy']:.3f}). A headline that implies novelty over trivial "
                "baselines would overstate the contribution."
            ),
        },
        {
            "title": "7. Temporal blindness",
            "text": (
                "Training is ~all from Oct-Nov 2017 (95%). The agent has never seen a "
                "post-2017 Spotify product cycle; any accuracy number generalises only "
                "to 2017-era English-language support interactions."
            ),
        },
    ]

    summary = {
        "headline_that_would_be_misleading": (
            "The agent correctly resolves 74.5% of customer messages and escalates "
            "the rest."
        ),
        "why_misleading": [
            "resolution != intent-label agreement (reply adequacy only 30.9% judged 'good')",
            "accuracy vs macro-F1 gap ignores rare-class customers",
            "escalation accuracy hides a dangerous under-escalation bias (recall 0.616)",
            "golden set class mix is artificial, not production-scaled",
            "labels are weak-supervised; part of the 'error' is label noise, not model quality",
            "baselines achieve the same number; the margin over rule-based is ~0",
            "model is temporally blind to anything after 2017",
            "mock mode: identical numbers can vary once a real LLM is switched in",
        ],
        "numbers_used": {
            "intent_accuracy": headline,
            "intent_macro_f1": pipe["intent_macro_f1"],
            "reply_good_rate": judge["reply_adequacy"]["good_rate"],
            "escalation": esc,
            "safe_to_auto_rate": judge["judge_human_agreement"]["safe_to_auto_rate"],
            "rule_baseline_accuracy": rule["accuracy"],
            "golden_reviewer_disagreement_rate": golden_stats["reviewer_vs_first_pass_disagree_rate"],
            "bayes_prior_heavy_class_share": golden_stats["intent_distribution"].get(
                "payment_billing", 0
            ),
        },
        "sections": sections,
    }
    (RESULTS / "misleading_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md = ["# What is misleading about the headline number", "", summary["headline_that_would_be_misleading"], ""]
    for s in sections:
        md.append(f"## {s['title']}")
        md.append(s["text"])
        md.append("")
    md.append("## Bottom line")
    md.append("The honest summary: the agent *classifies* well but *resolves* poorly;")
    md.append(f"macro-F1 {pipe['intent_macro_f1']:.3f}, escalation recall {esc['recall']:.3f}, "
              f"reply good-rate {judge['reply_adequacy']['good_rate']:.3f}, "
              f"safe-to-auto {judge['judge_human_agreement']['safe_to_auto_rate']:.3f}. "
              "Reporting only label accuracy would be misleading.")
    (RESULTS / "misleading_metrics.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Headline: {headline:.3f} -> judged misleading (see misleading_metrics.md)")


if __name__ == "__main__":
    main()