"""
Router Agent (The Brain): intent classification with Structural Output (JSON mode)
and Confidence Gate (route to Clarification when confidence < 0.7).

To be implemented in Phase 1: JSON mode, Pydantic parsing, state write.
"""

from graph.state import AgentState, IntentType  # graph is sibling under src

# Placeholder: actual implementation will call LLM with JSON response format
# and return intent, confidence, routing_rationale, missing_info.


def route(state: AgentState) -> dict:
    """
    Router node logic. Returns partial state update.

    When confidence < 0.7, graph should route to Clarification Node.
    """
    raise NotImplementedError("Router implementation in Phase 1")
