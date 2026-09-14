"""Unified agent pipeline.

Pipeline.run(query) -> {
    intent, intent_confidence, intent_reason,
    escalate, escalation_reason,
    reply, retrieval (provenance of the retrieved reference)
}

Composition: LLMIntentClassifier -> ReplyGenerator (which internally runs the
EscalationDecider and the Retriever). Fully deterministic in mock mode.
"""
from typing import Any

from escalation import decide as escalation_decide
from llm_classifier import LLMIntentClassifier
from reply_generator import ReplyGenerator
from retrieval import Retriever


class Pipeline:
    def __init__(
        self,
        retriever: Retriever | None = None,
        classifier: LLMIntentClassifier | None = None,
        reply_generator: ReplyGenerator | None = None,
    ) -> None:
        self.retriever = retriever or Retriever.load()
        self.classifier = classifier or LLMIntentClassifier()
        self.reply_generator = reply_generator or ReplyGenerator(self.retriever)

    def run(self, query: str) -> dict[str, Any]:
        pred = self.classifier.classify(query)
        esc = escalation_decide(pred["intent"], query, pred["confidence"])
        gen = self.reply_generator.generate(
            query, pred["intent"], pred["confidence"]
        )
        return {
            "query": query,
            "intent": pred["intent"],
            "intent_confidence": pred["confidence"],
            "intent_reason": pred["reason"],
            "escalate": esc["should_escalate"],
            "escalation_reason": esc["reason"],
            "reply": gen["reply"],
            "retrieval": gen["retrieval"],
            "backend": self.classifier.backend,
        }