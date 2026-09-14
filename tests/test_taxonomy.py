from intents import load_taxonomy, intent_names, is_auto_eligible


def test_taxonomy_loads_11_intents():
    tax = load_taxonomy()
    assert len(intent_names(tax)) == 11


def test_taxonomy_has_required_fields():
    tax = load_taxonomy()
    for spec in tax["intents"].values():
        for field in ("description", "examples", "negative_examples", "boundaries"):
            assert field in spec, f"missing {field}"


def test_required_intents_present():
    names = set(intent_names(load_taxonomy()))
    for expected in (
        "account_access",
        "payment_billing",
        "subscription_plan",
        "family_plan",
        "playback_issue",
        "technical_issue",
        "content_missing",
        "appreciation",
        "other",
    ):
        assert expected in names


def test_auto_eligibility_flags_are_sane():
    tax = load_taxonomy()
    assert not is_auto_eligible("payment_billing", tax)
    assert not is_auto_eligible("family_plan", tax)
    assert is_auto_eligible("appreciation", tax)