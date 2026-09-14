"""Reply generation for resolved (non-escalated) messages.

Deterministic mock: retrieve the best historical SpotifyCares reply for a
semantically similar conversation and return it verbatim with provenance.
Real mode (not run here): an LLM rewrites the retrieved reference in the voice
of the new conversation.

Semantics:
- escalate -> reply=None, reason set (the EscalationDecider owns the reason).
- resolve -> reply=retrieved historical response + metadata.
"""
from typing import Any

from escalation import decide as escalation_decide
from retrieval import Retriever


class ReplyGenerator:
    def __init__(
        self,
        retriever: Retriever,
        k: int = 8,
        prefer_intent_match: bool = True,
    ) -> None:
        self.retriever = retriever
        self.k = k
        self.prefer_intent_match = prefer_intent_match

    def generate(
        self,
        query: str,
        predicted_intent: str,
        confidence: float = 0.5,
    ) -> dict[str, Any]:
        esc = escalation_decide(predicted_intent, query, confidence)
        if esc["should_escalate"]:
            return {
                "query": query,
                "escalate": True,
                "escalation_reason": esc["reason"],
                "reply": None,
                "retrieval": None,
            }

        hits = self.retriever.search(query, k=self.k)
        best = self._select(hits, predicted_intent)
        return {
            "query": query,
            "escalate": False,
            "escalation_reason": None,
            "reply": best["brand_response"],
            "retrieval": {
                "similarity": best["score"],
                "source_conversation": best["conversation_id"],
                "source_intent": best["weak_label"],
                "source_customer_text": best["customer_text"],
            },
        }

    def _select(self, hits: list[dict], predicted_intent: str) -> dict:
        if self.prefer_intent_match:
            for h in hits:
                if h["weak_label"] == predicted_intent:
                    return h
        return hits[0]


if __name__ == "__main__":
    from llm_classifier import LLMIntentClassifier
    from retrieval import Retriever

    clf = LLMIntentClassifier()
    gen = ReplyGenerator(Retriever.load())
    for q in [
        "why does my shuffle play songs I don't like",
        "my account got hacked and someone changed my email",
        "how do I get premium for 3 months free",
    ]:
        pred = clf.classify(q)
        out = gen.generate(q, pred["intent"], pred["confidence"])
        print(f"\nQ: {q}")
        print(f"  intent={pred['intent']} conf={pred['confidence']:.2f}")
        print(f"  escalate={out['escalate']}")
        if out["reply"]:
            print(f"  reply: {out['reply'][:120]}")