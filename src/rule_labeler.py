"""
Deterministic first-pass intent labeler.

Used ONLY for structuring golden-set sampling and as a weak signal.
It is deliberately simple (counts + priority) and is NOT the final
classifier. Present to make the sampling strata reproducible.

Returns one "best" intent plus the number of rules that fired so the
sampler can flag ambiguous examples.
"""
import re
from dataclasses import dataclass

INTENT_KEYWORDS: dict[str, list[str]] = {
    "account_access": [
        "log in", "login", "logged out", "logged me out", "password", "forgot my password",
        "reset my password", "hacked", "unrecognized activity", "account locked",
        "locked out", "can't get into", "cant get into", "sign in", "signed out",
        "access my account", "accessing my account", "oauth",
    ],
    "payment_billing": [
        "charged", "charge", "refund", "credit card", "billing", "payment",
        "double charged", "charged twice", "took money", "money from", "card declined",
        "fraud", "chargeback", "payment method", "paying for", "paid for",
    ],
    "subscription_plan": [
        "premium", "student discount", "student plan", "free trial", "upgrade",
        "downgrade", "cancel premium", "cancel my premium", "subscription",
        "3 month", "three months", "premium not activated", "shows free",
        "assigned to spotify free", "get premium", "having premium", "i pay for premium",
        "hulu", "free account", "spotify free",
    ],
    "family_plan": [
        "family plan", "family account", "add my family", "family member", "family members",
        "join the family", "premium family", "remove me from", "family group",
    ],
    "playback_issue": [
        "shuffle", "repeat", "won't play", "wont play", "stops playing", "stopped playing",
        "stop playing", "skip", "skipping", "cuts out", "pause", "play next",
        "song ends", "buffering", "plays random", "album covers", "cover to cover",
        "play in order",
    ],
    "technical_issue": [
        "crash", "crashes", "crashing", "won't open", "wont open", "download",
        "downloading", "not downloading", "offline", "web player", "update",
        "reinstall", "error", "freeze", "freezes", "frozen", "notification",
        "stop working", "not working", "worked", "isn't working", "doesn't work" ,
        "dont work", "doesnt work", "not updating", "delete and re",
    ],
    "device_compatibility": [
        "apple watch", "samsung tv", "smart tv", "xbox", "ps4", "alexa", "google home",
        "car", "chromecast", "connect", "desktop", "web player", "sonos", "nest",
        "garmin", "pixel watch", "echo",
    ],
    "content_missing": [
        "album", "song is not", "not available", "unavailable", "this song", "release",
        "released", "where is", "coming to spotify", "not on spotify", "on spotify",
        "when will", "add this song", "add the album", "missing", "removed",
        "not there", "isn't there", "cant find", "can't find", "country",
    ],
    "feature_and_feedback": [
        "presale", "feature", "would be nice", "please add", "please put", "request",
        "feedback", "recommendation", "recommendations", "suggestion", "idea",
        "why can't i", "why cant i", "discover weekly", "daily mix", "please fix",
        "add a", "why is your app", "improve", "bring back",
    ],
    "appreciation": [
        "thank you", "thanks", "thank u", "works now", "working now", "fixed",
        "awesome", "great", "much appreciated", "appreciat", "homies", "dopeee",
        "you guys are", "thx",
    ],
}

STOP_TERMS = [r"https?://\S+", r"@\w+", r"&amp;", r"&gt;", r"&lt;"]


def _normalize(text: str) -> str:
    t = text
    for pat in STOP_TERMS:
        t = re.sub(pat, " ", t)
    return re.sub(r"\s+", " ", t).strip().lower()


@dataclass(frozen=True)
class FirstPass:
    text_original: str
    text_norm: str
    best_intent: str
    matched_intents: list[str]
    n_rule_hits: int
    n_matching_intents: int


def first_pass_intent(text: str) -> FirstPass:
    """Return the deterministic best intent and match info for a message."""
    norm = _normalize(text)
    hits: dict[str, int] = {}
    for intent, patterns in INTENT_KEYWORDS.items():
        score = sum(1 for p in patterns if p in norm)
        if score:
            hits[intent] = score

    matched = sorted(hits, key=lambda k: (-hits[k], k))

    # appreciation dominates only if it is the ONLY match (thanks + problem words)
    best = matched[0] if matched else "other"
    if best == "appreciation" and len(matched) > 1:
        best = matched[1]

    return FirstPass(
        text_original=text,
        text_norm=norm,
        best_intent=best,
        matched_intents=matched,
        n_rule_hits=sum(hits.values()),
        n_matching_intents=len(matched),
    )


if __name__ == "__main__":
    tests = [
        "I can't log in, my password doesn't work",
        "I was charged twice for premium, want a refund",
        "how do I get premium",
        "can I add my family to the family plan",
        "shuffle is not working, it keeps playing in order",
        "the app crashes every time I open it",
        "thanks!",
        "where is the reputation album",
        "why isn't spotify on my smart tv",
        "hello",
    ]
    for t in tests:
        fp = first_pass_intent(t)
        print(f"{fp.best_intent:<22} hits={fp.matched_intents} | {t}")