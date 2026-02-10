"""
Omni-Help graph nodes: router, clarification, retriever, sql, web, fallback, synthesis.

Skeleton implementations; full logic in respective phases. Each node returns
Partial[AgentState] for the StateGraph.
"""

from typing import Any

from graph.state import AgentState

# Placeholder node signatures; implementations in Phases 1–7.


def router_node(state: AgentState) -> dict[str, Any]:
    """Router (Brain): intent, confidence, routing_rationale, missing_info. Confidence Gate: <0.7 → Clarification."""
    raise NotImplementedError("Phase 1")


def clarification_node(state: AgentState) -> dict[str, Any]:
    """Clarification: set missing_info and/or return clarifying question; graph may re-route to Router."""
    raise NotImplementedError("Phase 1 / 3")


def retriever_node(state: AgentState) -> dict[str, Any]:
    """Policy RAG: vector search → policy_context."""
    raise NotImplementedError("Phase 4")


def sql_node(state: AgentState) -> dict[str, Any]:
    """Order/SQL: NL→SQL, execute, sql_result or sql_error."""
    raise NotImplementedError("Phase 5")


def web_node(state: AgentState) -> dict[str, Any]:
    """Web search: Tavily → web_context."""
    raise NotImplementedError("Phase 6")


def fallback_node(state: AgentState) -> dict[str, Any]:
    """Fallback: handoff_context for human escalation."""
    raise NotImplementedError("Phase 7")


def synthesis_node(state: AgentState) -> dict[str, Any]:
    """Synthesis: combine pipeline output → final_response."""
    raise NotImplementedError("Phase 3+")
