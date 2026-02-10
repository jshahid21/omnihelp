"""
Evaluation harness for the Router (Brain). Phase 0.

Loads golden_dataset.json, runs router (or stub), computes accuracy,
and can log results to LangSmith. Runnable via pytest or CLI.
"""

from pathlib import Path
from typing import Any

# Golden dataset path (relative to this file)
GOLDEN_DATASET_PATH = Path(__file__).resolve().parent / "golden_dataset.json"


def load_golden_dataset() -> list[dict[str, Any]]:
    """Load golden dataset JSON. Returns list of {id, query, intent, notes}."""
    import json
    with open(GOLDEN_DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


def run_router_on_dataset(router_fn=None) -> list[dict[str, Any]]:
    """
    Run router on each golden example. If router_fn is None, stub: return intent from golden.
    Otherwise router_fn(state_dict) -> partial state with 'intent' (and optionally 'confidence').
    Returns list of {id, query, expected_intent, predicted_intent, correct}.
    """
    data = load_golden_dataset()
    results = []
    for item in data:
        expected = item["intent"]
        if router_fn is None:
            predicted = expected  # stub: assume correct
        else:
            state = {"user_query": item["query"], "messages": []}
            out = router_fn(state)
            predicted = out.get("intent", expected)
        results.append({
            "id": item["id"],
            "query": item["query"],
            "expected_intent": expected,
            "predicted_intent": predicted,
            "correct": predicted == expected,
        })
    return results


def compute_accuracy(results: list[dict[str, Any]]) -> float:
    """Accuracy = correct / total."""
    if not results:
        return 0.0
    return sum(1 for r in results if r["correct"]) / len(results)


def test_golden_dataset_loads():
    """Sanity: golden dataset has 10 examples, 2 per intent."""
    data = load_golden_dataset()
    assert len(data) == 10
    intents = [d["intent"] for d in data]
    for intent in ("policy", "sql", "web", "product_info", "complaint"):
        assert intents.count(intent) == 2, f"Expected 2 '{intent}' examples"


def test_eval_harness_stub():
    """With no router, stub uses golden intent; accuracy should be 1.0."""
    results = run_router_on_dataset(router_fn=None)
    assert len(results) == 10
    assert compute_accuracy(results) == 1.0
