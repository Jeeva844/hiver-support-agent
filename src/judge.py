"""Deterministic judge for pipeline outputs.

Judge contract mirrors an LLM-as-judge but is computed offline:

evaluate(pipeline_result, query, human_intent, human_escalate, gold_reply) -> dict
  - intent_correct          pred intent == human intent
  - escalation_agreement    pred escalate == human escalate
  - safe_to_auto            (not escalate) and intent_correct and confidence>=0.5
  - reply_verdict           for auto cases: 'good'/'poor' vs the historical gold
                            reply using token containment + tfidf cosine.

judge<->human agreement = share of examples where the judge's triple verdict
(intent, escalation, reply-adequate) matches the human labels.

Real-LLM note: this deterministic judge is the mock-bandwidth stand-in that
shares the same verdict schema a GPT judge would emit.
"""
import re
from typing import Any

from retrieval import Retriever

_WS = re.compile(r"\s+")


def _tokens(s: str) -> set[str]:
    return set(_WS.sub(" ", s.lower()).split())


class Judge:
    def __init__(
        self,
        retriever: Retriever,
        containment_threshold: float = 0.35,
        cosine_threshold: float = 0.25,
    ) -> None:
        self.retriever = retriever
        self.containment_threshold = containment_threshold
        self.cosine_threshold = cosine_threshold

    def evaluate(
        self,
        result: dict[str, Any],
        query: str,
        human_intent: str,
        human_escalate: bool,
        gold_reply: str | None,
    ) -> dict[str, Any]:
        pred_intent = result["intent"]
        pred_esc = bool(result["escalate"])
        intent_correct = pred_intent == human_intent
        esc_agreement = pred_esc == human_escalate
        safe_to_auto = (
            (not pred_esc)
            and intent_correct
            and result["intent_confidence"] >= 0.5
        )

        reply_verdict = None
        reply_metrics = None
        if not pred_esc and gold_reply:
            reply = result["reply"] or ""
            gt = _tokens(gold_reply)
            gen = _tokens(reply)
            containment = (
                len(gen & gt) / len(gt) if gt else 0.0
            )
            q = self.retriever.vectorizer.transform([reply, gold_reply])
            from sklearn.preprocessing import normalize

            qn = normalize(q, norm="l2", axis=1)
            cosine = float((qn[0] @ qn[1].T).toarray()[0][0])
            reply_metrics = {"containment": float(containment), "cosine": float(cosine)}
            reply_verdict = (
                "good"
                if containment >= self.containment_threshold
                or cosine >= self.cosine_threshold
                else "poor"
            )

        return {
            "intent_correct": bool(intent_correct),
            "escalation_agreement": bool(esc_agreement),
            "safe_to_auto": bool(safe_to_auto),
            "reply_verdict": reply_verdict,
            "reply_metrics": reply_metrics,
        }