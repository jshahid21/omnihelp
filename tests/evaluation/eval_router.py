"""
Evaluation harness for the Router (Brain). Phase 0 / Phase 1.

Loads golden_dataset.json, runs each query through the compiled graph,
compares predicted intent to the golden label, and reports accuracy.

Run via pytest:
    pytest tests/evaluation/eval_router.py -v

Run as a script (prints full report):
    python tests/evaluation/eval_router.py
"""

import json
import sys
from pathlib import Path
from typing import Any

GOLDEN_DATASET_PATH = Path(__file__).resolve().parent / "golden_dataset.json"

# Resolve src/ onto the path so imports work when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_golden_dataset() -> list[dict[str, Any]]:
    """Load golden dataset JSON. Returns list of {id, query, intent, notes}."""
    with open(GOLDEN_DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


def compute_accuracy(results: list[dict[str, Any]]) -> float:
    """Accuracy = correct / total."""
    if not results:
        return 0.0
    return sum(1 for r in results if r["correct"]) / len(results)


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation() -> list[dict[str, Any]]:
    """
    Run all golden queries through the compiled graph.

    Returns:
        List of result dicts: id, query, expected_intent, predicted_intent,
        confidence, routing_rationale, correct.
    """
    from graph.graph import graph  # imported here so pytest can stub without importing

    dataset = load_golden_dataset()
    results = []

    for item in dataset:
        state_in = {"user_query": item["query"], "messages": []}
        output = graph.invoke(state_in)

        predicted = output.get("intent", "unknown")
        expected = item["intent"]

        results.append({
            "id": item["id"],
            "query": item["query"],
            "expected_intent": expected,
            "predicted_intent": predicted,
            "confidence": round(output.get("confidence", 0.0), 3),
            "routing_rationale": output.get("routing_rationale", ""),
            "correct": predicted == expected,
        })

    return results


def print_report(results: list[dict[str, Any]]) -> None:
    """Print a formatted accuracy report to stdout."""
    accuracy = compute_accuracy(results)
    passed = sum(1 for r in results if r["correct"])
    total = len(results)

    print("\n" + "=" * 70)
    print("  Omni-Help Router — Golden Dataset Evaluation")
    print("=" * 70)

    for r in results:
        status = "✅" if r["correct"] else "❌"
        print(
            f"{status} [{r['id']}] "
            f"Expected: {r['expected_intent']:<15} "
            f"Got: {r['predicted_intent']:<15} "
            f"Conf: {r['confidence']:.2f}"
        )
        if not r["correct"]:
            print(f"     Query: {r['query']}")
            print(f"     Reasoning: {r['routing_rationale']}")

    print("-" * 70)
    print(f"  Accuracy: {passed}/{total}  ({accuracy * 100:.1f}%)")
    target = "✅ TARGET MET" if accuracy >= 0.95 else "⚠️  Below 95% target"
    print(f"  {target}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# pytest-compatible tests (no live LLM calls — use stubs)
# ---------------------------------------------------------------------------

def test_golden_dataset_loads():
    """Sanity: golden dataset has 10 examples, 2 per intent."""
    data = load_golden_dataset()
    assert len(data) == 10
    intents = [d["intent"] for d in data]
    for intent in ("policy", "sql", "web", "product_info", "complaint"):
        assert intents.count(intent) == 2, f"Expected 2 '{intent}' examples"


def test_eval_harness_stub():
    """Stub accuracy baseline: without LLM calls, harness should report 100%."""
    dataset = load_golden_dataset()
    stub_results = [
        {
            "id": item["id"],
            "query": item["query"],
            "expected_intent": item["intent"],
            "predicted_intent": item["intent"],   # stub always returns golden intent
            "confidence": 0.99,
            "routing_rationale": "stub",
            "correct": True,
        }
        for item in dataset
    ]
    assert compute_accuracy(stub_results) == 1.0


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_evaluation()
    print_report(results)
