"""
Integration tests for Phase 3: the clarification cycle.

Strategy
--------
We mock `agents.router.router_agent` so we control confidence without making
real OpenAI calls. Two scenarios are tested:

1. cycle_resolves  — router returns low confidence on call 1 (→ clarification),
                     then high confidence on call 2 (→ retriever stub).
                     Asserts: clarification_turn_count == 1, final intent correct.

2. cycle_escalates — router always returns low confidence so the graph exhausts
                     MAX_CLARIFICATION_TURNS (2) and escalates to fallback.
                     Asserts: clarification_turn_count == 2, fallback fired.
"""

import sys
import os
from unittest.mock import patch

import pytest

# Allow running directly from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _low_confidence_result() -> dict:
    return {
        "intent": "policy",
        "confidence": 0.4,
        "routing_rationale": "Ambiguous query — confidence below threshold.",
        "missing_info": ["low_confidence"],
    }


def _high_confidence_result(intent: str = "policy") -> dict:
    return {
        "intent": intent,
        "confidence": 0.95,
        "routing_rationale": "Clear policy question after clarification.",
        "missing_info": [],
    }


# ---------------------------------------------------------------------------
# Test 1: cycle resolves after one clarification turn
# ---------------------------------------------------------------------------

def test_clarification_cycle_resolves():
    """
    Scenario: first router call returns low confidence; second returns high.
    Expected path: router → clarification → router → retriever → END.

    Patch target is graph.nodes.router_agent (the name bound at import time in
    nodes.py via `from agents.router import router_agent`).
    """
    call_returns = [_low_confidence_result(), _high_confidence_result("policy")]
    call_index = {"n": 0}

    def mock_router_agent(state):
        result = call_returns[call_index["n"]]
        call_index["n"] += 1
        return result

    with patch("graph.nodes.router_agent", side_effect=mock_router_agent):
        from graph.graph import build_graph
        test_graph = build_graph()
        result = test_graph.invoke({"user_query": "What is your return policy?"})

    assert result["clarification_turn_count"] == 1, (
        f"Expected 1 clarification turn, got {result['clarification_turn_count']}"
    )
    assert result["intent"] == "policy"
    assert result["confidence"] == 0.95
    print(f"\n[test_clarification_cycle_resolves] PASS — turn_count={result['clarification_turn_count']}, intent={result['intent']}")


# ---------------------------------------------------------------------------
# Test 2: cycle exhausts max turns → fallback
# ---------------------------------------------------------------------------

def test_clarification_cycle_escalates_to_fallback():
    """
    Scenario: router always returns low confidence.
    Expected path: router → clarification (×2) → fallback → END.

    Patch target is graph.nodes.router_agent (the name bound at import time in
    nodes.py via `from agents.router import router_agent`).
    """
    def mock_router_agent_always_low(state):
        return _low_confidence_result()

    with patch("graph.nodes.router_agent", side_effect=mock_router_agent_always_low):
        from graph.graph import build_graph, MAX_CLARIFICATION_TURNS
        test_graph = build_graph()

        result = test_graph.invoke({"user_query": "Hm, I'm not sure what I need."})

    assert result["clarification_turn_count"] == MAX_CLARIFICATION_TURNS, (
        f"Expected {MAX_CLARIFICATION_TURNS} turns, got {result['clarification_turn_count']}"
    )
    assert "final_response" in result
    print(
        f"\n[test_clarification_cycle_escalates_to_fallback] PASS — "
        f"turn_count={result['clarification_turn_count']}, escalated to fallback."
    )
