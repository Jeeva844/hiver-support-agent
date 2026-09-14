"""LLM intent classifier.

Contract: classify(text) -> {"intent": str, "confidence": float, "reason": str}

Two backends, selected automatically:
- "openai": only used when the `openai` package is installed AND OPENAI_API_KEY is
  set (`.env` or environment). Model from OPENAI_MODEL (default: gpt-4o-mini).
  Enforces strict-JSON output with the taxonomy, validates against the taxonomy,
  retries on malformed output, and falls back to the deterministic backend.
- "mock": deterministic, offline, reproducible. Uses the rule labeler plus a
  calibrated confidence heuristic. Exists so the entire pipeline runs without
  any network/API and still yields an auditable result.

The JSON output shape (intent/confidence/reason) is identical in both modes.
"""
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from intents import load_taxonomy
from rule_labeler import first_pass_intent
from intents import TaxonomyError

load_dotenv()

SYSTEM_PROMPT_TEMPLATE = """You are the intent-classification component of a customer-support agent for the Spotify Twitter support account.

Classify the USER MESSAGE only. Return strict JSON with exactly three keys:
{{"intent": <one of the taxonomy intents>, "confidence": <float 0..1>, "reason": <short string>}}

Taxonomy (name -> description, escalate-allowed):
{taxonomy_dump}

Rules:
- Choose the single intent that best captures what the user is ASKING about.
- If the text is content-free or off-topic, use "other".
- Never invent new intents; the intent must be exactly one of the listed names.
- Respond with ONLY the JSON object, no commentary.
"""


class LLMIntentClassifier:
    def __init__(
        self,
        taxonomy_path: Path | str = Path("data/intent_taxonomy.json"),
        model: str | None = None,
        max_retries: int = 2,
    ) -> None:
        self.tax_path = Path(taxonomy_path)
        self.taxonomy = load_taxonomy(self.tax_path)
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.max_retries = max_retries
        self.backend, self._client = self._init_backend()

    def _init_backend(self):
        key = os.getenv("OPENAI_API_KEY")
        try:
            import openai  # noqa: F401

            has_net = True
        except Exception:
            has_net = False
        if key and has_net:
            try:
                from openai import OpenAI

                client = OpenAI(api_key=key)
                return "openai", client
            except Exception:
                return "mock", None
        return "mock", None

    # ------------------------------------------------------------------ public
    def classify(self, text: str) -> dict:
        if self.backend == "openai":
            result = self._classify_openai(text)
            if result is not None and self._valid_result(result):
                return result
            # fallback path is deterministic, so never raises
            return self._classify_mock(text)
        return self._classify_mock(text)

    def classify_many(self, texts: list[str]) -> list[dict]:
        return [self.classify(t) for t in texts]

    # ----------------------------------------------------------------- openai
    def _classify_openai(self, text: str) -> dict | None:
        if self._client is None:
            return None
        sys_msg = SYSTEM_PROMPT_TEMPLATE.format(
            taxonomy_dump=_taxonomy_dump(self.taxonomy)
        )
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": text},
                    ],
                    temperature=0.0,
                )
                raw = resp.choices[0].message.content or ""
            except Exception:
                if attempt == self.max_retries:
                    return None
                continue
            result = _parse_json(raw)
            if result is not None and self._valid_result(result):
                return result
        return None

    # ------------------------------------------------------------------- mock
    def _classify_mock(self, text: str) -> dict:
        """Deterministic fallback sharing the JSON contract with the LLM."""
        fp = first_pass_intent(text)
        intent = fp.best_intent
        if intent != "other" and fp.n_matching_intents == 1:
            confidence = 0.85
            reason = "single rule match"
        elif intent != "other":
            confidence = 0.6
            reason = f"rule matches: {', '.join(fp.matched_intents)}"
        else:
            confidence = 0.3
            reason = "no keyword match; default to other"
        return {"intent": intent, "confidence": float(confidence), "reason": reason}

    # --------------------------------------------------------------- helpers
    def _valid_result(self, result: dict) -> bool:
        intents = set(self.taxonomy["intents"].keys())
        return (
            isinstance(result.get("intent"), str)
            and result["intent"] in intents
            and isinstance(result.get("confidence"), (int, float))
            and 0 <= float(result["confidence"]) <= 1
            and isinstance(result.get("reason"), str)
        )


def _taxonomy_dump(taxonomy: dict) -> str:
    lines = []
    for name, spec in taxonomy["intents"].items():
        lines.append(f"- {name}: {spec.get('description', '')}")
        lines.append(f"  examples: {'; '.join(spec.get('examples', [])[:4])}")
    return "\n".join(lines)


def _parse_json(raw: str) -> dict | None:
    """Try strict JSON parse, then extract the first {...} block."""
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


if __name__ == "__main__":
    try:
        clf = LLMIntentClassifier()
    except TaxonomyError as e:
        raise SystemExit(str(e))
    print(f"Backend: {clf.backend}  model={clf.model if clf.backend == 'openai' else '-'}")
    for t in [
        "I was charged twice for premium and I want a refund",
        "why does my shuffle play songs I don't like",
        "thanks, that worked!",
        "hello",
    ]:
        print(clf.classify(t))