"""Escalation decision module (deterministic mock; LLM-aware in real mode).

decide(intent, text, confidence) -> {"should_escalate": bool, "reason": str}

Policy mirrors data/intent_taxonomy.json `auto_eligible` + `escalation_note`:
- payment_billing, family_plan, other always escalate (account state / unknown).
- account_access escalates when security/compromise is signalled.
- subscription_plan escalates on account-status mismatch or broken promo flow.
- technical_issue/playback_issue escalate only when standard fixes are
  reported as already exhausted.
- appreciation never escalates.
The bias is deliberately conservative: over-escalation is safe on a public
support channel; under-escalation is not. This is measured honestly later.
"""
import re

ALWAYS_ESCALATE = {"payment_billing", "family_plan", "other"}
NEVER_ESCALATE = {"appreciation"}

SECURITY_PATTERNS = [
    r"\bhack(ed|er|ers)?\b", r"unauthorized", r"someone (else )?(changed|took|used|connected)",
    r"changed my (password|email)", r"unknown (device|activity)", r"unrecognized (device|activity)",
    r"breach", r"take ?over", r"don't have a spotify account", r"i don't have spotify",
    r"double charge", r"charged twice", r"fraud",
]
STATUS_MISMATCH_PATTERNS = [
    r"shows? free", r"switched (me |my )?(to )?free", r"assigned to (the |spotify )?free",
    r"still (free|not active)", r"not activated", r"premium (but|yet)", r"paying.*(not active|free)",
    r"cancelled .* not renewed|was just charged", r"charged .* after (i )?cancell",
]
BROKEN_FLOW_PATTERNS = [
    r"\berror\b", r"won't let", r"wont let", r"keeps (rejecting|saying|giving|bringing)",
    r"not (working|accepted|going through)", r"can't (join|add|activate|verify)", r"cant (join|add|activate|verify)",
    r"doesn't (work|accept)", r"doesnt (work|accept)", r"still (having|facing)",
]
EXHAUSTED_PATTERNS = [
    r"(reinstal|re-instal|uninstal|update(d)?|restar|re-start|cleared? cache|reset)[^.!?]{0,40}(still|nothing|not)",
    r"for weeks", r"for months", r"been (weeks|months)", r"all day", r"a week",
    r"still (broken|happening|persist)", r"i'm (still|really) (frustrated|annoyed)",
]
INFORMATIONAL_HOWTO = [
    r"how (do|can|would) i", r"how to", r"is spotify", r"does spotify",
    r"want to (get|know)", r"can i (get|use|play|listen)", r"when (will|is)",
    r"should i", r"what('s| is) the (difference|price|cost)",
]


def _has(text: str, patterns: list[str]) -> bool:
    low = text.lower()
    return any(re.search(p, low) for p in patterns)


def decide(intent: str, text: str, confidence: float = 0.5) -> dict:
    if intent in NEVER_ESCALATE:
        return {"should_escalate": False, "reason": f"{intent} = no action"}
    if intent in ALWAYS_ESCALATE:
        return {
            "should_escalate": True,
            "reason": f"{intent} requires account verification or human review",
        }

    if intent == "account_access":
        if _has(text, SECURITY_PATTERNS) or _has(text, [r"can'?t (get into|access|log)", r"locked ?out"]):
            if _has(text, SECURITY_PATTERNS):
                return {"should_escalate": True, "reason": "security/compromise signal"}
            return {"should_escalate": False, "reason": "login troubleshooting (self-serve steps)"}
        if _has(text, [r"reset", r"password"]):
            return {"should_escalate": False, "reason": "password guidance is self-serve"}
        return {"should_escalate": True, "reason": "account access, low signal -> verify"}

    if intent == "subscription_plan":
        if _has(text, STATUS_MISMATCH_PATTERNS):
            return {"should_escalate": True, "reason": "account status mismatch -> verify billing state"}
        if _has(text, BROKEN_FLOW_PATTERNS):
            return {"should_escalate": True, "reason": "promo/cancel flow failing on the account"}
        return {"should_escalate": False, "reason": "informational plan/premium FAQ"}

    if intent in ("technical_issue", "playback_issue"):
        if _has(text, EXHAUSTED_PATTERNS):
            return {"should_escalate": True, "reason": "standard fixes already attempted"}
        return {"should_escalate": False, "reason": "standard troubleshooting offered"}

    if intent in ("device_compatibility", "content_missing", "feature_and_feedback"):
        return {"should_escalate": False, "reason": f"{intent} is informational/acknowledge-only"}

    # low-confidence / unknown -> safe default
    if confidence < 0.5:
        return {"should_escalate": True, "reason": "low classifier confidence -> human review"}
    return {"should_escalate": False, "reason": "auto-eligible intent, nothing to verify"}


if __name__ == "__main__":
    for t in [
        ("payment_billing", "I was charged twice for premium, need refund"),
        ("account_access", "my account got hacked and someone changed my email"),
        ("account_access", "forgot my password, need to reset it"),
        ("subscription_plan", "I pay for premium but it shows free"),
        ("subscription_plan", "how do I get premium for $9.99"),
        ("playback_issue", "shuffle won't shuffle, happens for weeks now"),
        ("playback_issue", "shuffle keeps playing the same songs"),
        ("appreciation", "thank you that worked"),
    ]:
        r = decide(t[0], t[1])
        print(f"{t[0]:<20} escalate={r['should_escalate']!s:<5} {r['reason']}")