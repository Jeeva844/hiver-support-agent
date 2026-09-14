from escalation import decide
from llm_classifier import LLMIntentClassifier


def test_mock_classifier_contract():
    clf = LLMIntentClassifier()
    result = clf.classify("my account got hacked, someone changed my email")
    assert set(result.keys()) == {"intent", "confidence", "reason"}
    assert result["intent"] in set(
        __import__("intents").intent_names(__import__("intents").load_taxonomy())
    )
    assert 0 <= result["confidence"] <= 1


def test_escalation_high_risk():
    assert decide("payment_billing", "I was charged twice for premium, refund")["should_escalate"] is True
    assert decide("account_access", "my account got hacked")["should_escalate"] is True
    assert decide("appreciation", "thanks!")["should_escalate"] is False


def test_escalation_low_risk():
    assert decide("subscription_plan", "how do I get premium")["should_escalate"] is False
    assert decide("playback_issue", "shuffle plays the same songs")["should_escalate"] is False
    assert decide("feature_and_feedback", "please add a dark mode")["should_escalate"] is False