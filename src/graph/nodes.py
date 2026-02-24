"""
Omni-Help graph nodes.

router_node  — calls the Router Agent (The Brain); live implementation.
Stub nodes   — print their name and pass state through unchanged.
               Will be replaced with real logic in Phases 4-7.
"""

from typing import Any

from graph.state import AgentState
from agents.router import router_agent


# ---------------------------------------------------------------------------
# Phase 1: Router (The Brain)
# ---------------------------------------------------------------------------

def router_node(state: AgentState) -> dict[str, Any]:
    """
    Entry node. Invokes the Router Agent to classify the user query.

    Calls router_agent which returns intent, confidence, routing_rationale,
    and missing_info. The Confidence Gate logic lives in route_decision (graph.py).

    Args:
        state: Current AgentState; must contain 'user_query'.

    Returns:
        Partial state update: intent, confidence, routing_rationale, missing_info.
    """
    return router_agent(state)


# ---------------------------------------------------------------------------
# Stubs — Phases 4-7
# ---------------------------------------------------------------------------

def retriever_node(state: AgentState) -> dict[str, Any]:
    """
    Stub: Policy RAG pipeline (Phase 4).
    Queries vector store and builds policy_context with citations.
    """
    print("[Node] Reached RETRIEVER — Policy RAG (stub)")
    return {"final_response": "⚙️ [Stub] Policy retrieval not yet implemented."}


def sql_node(state: AgentState) -> dict[str, Any]:
    """
    Stub: Order/SQL pipeline (Phase 5).
    Converts NL to SQL, executes against order DB, returns results.
    """
    print("[Node] Reached SQL — Order pipeline (stub)")
    return {"final_response": "⚙️ [Stub] SQL / Order lookup not yet implemented."}


def web_node(state: AgentState) -> dict[str, Any]:
    """
    Stub: Web search pipeline (Phase 6).
    Calls Tavily and synthesises results into web_context.
    """
    print("[Node] Reached WEB — Tavily search (stub)")
    return {"final_response": "⚙️ [Stub] Web search not yet implemented."}


def product_info_node(state: AgentState) -> dict[str, Any]:
    """
    Stub: Product info pipeline (Phase 4/6).
    Routes to retriever or web depending on availability.
    """
    print("[Node] Reached PRODUCT_INFO (stub)")
    return {"final_response": "⚙️ [Stub] Product info not yet implemented."}


def fallback_node(state: AgentState) -> dict[str, Any]:
    """
    Stub: Fallback / human escalation node.
    Handles complaints and low-confidence queries (Phase 7).
    """
    reason = state.get("fallback_reason", "unspecified")
    print(f"[Node] Reached FALLBACK — reason: {reason} (stub)")
    return {
        "final_response": (
            "I'm connecting you with a human agent who can best assist you. "
            "Thank you for your patience."
        ),
        "fallback_reason": reason,
    }


def clarification_node(state: AgentState) -> dict[str, Any]:
    """
    Stub: Clarification node for low-confidence routes.
    Returns a clarifying question; graph cycles back to router (Phase 3).
    """
    print("[Node] Reached CLARIFICATION — low confidence (stub)")
    return {
        "final_response": (
            "I want to make sure I help you correctly. "
            "Could you provide a bit more detail about your question?"
        ),
    }
