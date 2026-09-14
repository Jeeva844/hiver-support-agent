"""Intent taxonomy loading and validation (Phase 3)."""
import json
from pathlib import Path
from typing import Any

TAXONOMY_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "intent_taxonomy.json"
)


class TaxonomyError(Exception):
    """Raised when the intent taxonomy is missing or malformed."""


def load_taxonomy(path: Path | str = TAXONOMY_PATH) -> dict[str, Any]:
    """Load and validate the intent taxonomy JSON."""
    path = Path(path)
    if not path.exists():
        raise TaxonomyError(f"Intent taxonomy not found at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "intents" not in data:
        raise TaxonomyError("Taxonomy is missing the 'intents' key")
    intents = data["intents"]
    if not isinstance(intents, dict) or not intents:
        raise TaxonomyError("'intents' must be a non-empty object")
    for name, spec in intents.items():
        for field in ("description", "examples", "negative_examples", "boundaries"):
            if field not in spec:
                raise TaxonomyError(f"Intent '{name}' is missing field '{field}'")
    return data


def intent_names(taxonomy: dict[str, Any]) -> list[str]:
    """Return the ordered list of intent names."""
    return list(taxonomy["intents"].keys())


def is_auto_eligible(intent: str, taxonomy: dict[str, Any]) -> bool:
    """Whether the intent's default policy allows automated handling."""
    spec = taxonomy["intents"].get(intent)
    if spec is None:
        return False
    return bool(spec.get("auto_eligible", False))


if __name__ == "__main__":
    tax = load_taxonomy()
    print(f"Loaded taxonomy v{tax['meta']['version']} for {tax['meta']['brand']}")
    print(f"  intents ({len(tax['intents'])}): {', '.join(intent_names(tax))}")
    for name, spec in tax["intents"].items():
        print(f"    - {name:<24} auto_eligible={spec['auto_eligible']}")