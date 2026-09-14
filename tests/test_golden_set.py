import pandas as pd

from intents import intent_names, load_taxonomy

GOLDEN_PATH = "data/golden/golden_set.csv"


def _df():
    return pd.read_csv(GOLDEN_PATH)


def test_golden_has_200_rows_and_schema():
    df = _df()
    assert len(df) == 200
    for col in ("id", "text", "intent", "should_escalate", "human_notes", "review_status"):
        assert col in df.columns


def test_all_examples_reviewed():
    assert bool((_df()["review_status"] == "reviewed").all())


def test_intents_within_taxonomy():
    df = _df()
    valid = set(intent_names(load_taxonomy()))
    assert set(df["intent"].unique()).issubset(valid)


def test_distinct_conversations():
    df = _df()
    assert df["conversation_id"].nunique() == len(df)


def test_escalate_is_boolean():
    df = _df()
    assert set(df["should_escalate"].unique()).issubset({True, False})