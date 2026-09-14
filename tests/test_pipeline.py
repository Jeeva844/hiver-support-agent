from pipeline import Pipeline


def test_pipeline_returns_schema():
    pipe = Pipeline()
    result = pipe.run("I was charged twice for premium")
    for key in (
        "query",
        "intent",
        "intent_confidence",
        "intent_reason",
        "escalate",
        "escalation_reason",
        "reply",
        "retrieval",
        "backend",
    ):
        assert key in result, f"missing {key}"
    assert result["backend"] == "mock"


def test_pipeline_is_deterministic():
    pipe = Pipeline()
    q = "shuffle keeps playing songs I don't like"
    a = pipe.run(q)
    b = pipe.run(q)
    assert a["intent"] == b["intent"]
    assert a["escalate"] == b["escalate"]