import pandas as pd

from retrieval import Retriever


def test_rule_labeler_major_examples():
    from rule_labeler import first_pass_intent

    assert first_pass_intent("I was charged twice, refund please").best_intent == "payment_billing"
    assert first_pass_intent("can't log in, password wrong").best_intent == "account_access"
    assert first_pass_intent("shuffle won't work").best_intent == "playback_issue"
    assert first_pass_intent("thanks!").best_intent == "appreciation"
    assert first_pass_intent("random gibberish xyz123").best_intent == "other"


def test_retrieval_index_excludes_golden_conversations():
    r = Retriever.load()
    golden = pd.read_csv("data/golden/golden_set.csv")
    golden_convs = set(golden["conversation_id"].unique())
    overlap = golden_convs & set(r.meta["conversation_id"].unique())
    assert len(overlap) == 0, f"leak: {len(overlap)} golden conversations indexed"


def test_golden_conversations_not_retrievable():
    r = Retriever.load()
    golden = pd.read_csv("data/golden/golden_set.csv").head(10)
    for conv_id in golden["conversation_id"]:
        hits = r.search(golden.loc[golden["conversation_id"] == conv_id, "text"].iloc[0], k=3)
        assert all(h["conversation_id"] != conv_id for h in hits)