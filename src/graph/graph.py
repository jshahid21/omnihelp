"""
Omni-Help LangGraph: cyclic StateGraph.

Topology (Phase 3 — cycle closed)
----------------------------------
START → router
router → (conditional via route_decision):
    complaint                       → fallback → END
    confidence < 0.7                → clarification
    policy                          → retriever → END   (stub)
    sql                             → sql      → END   (stub)
    web                             → web      → END   (stub)
    product_info                    → product_info → END (stub)
    unknown                         → fallback → END

clarification → (conditional via clarification_decision):
    turn_count < MAX_CLARIFICATION_TURNS  → router   ← THE CYCLE
    turn_count >= MAX_CLARIFICATION_TURNS → fallback → END

Confidence Gate
---------------
confidence < CONFIDENCE_THRESHOLD (0.7) → clarification node.

Cycle Guard
-----------
MAX_CLARIFICATION_TURNS = 2. After 2 clarification turns without a
high-confidence re-route, the graph escalates to fallback.
"""

from langgraph.graph import StateGraph, END

from graph.state import AgentState
from graph.nodes import (
    router_node,
    retriever_node,
    sql_node,
    web_node,
    product_info_node,
    fallback_node,
    clarification_node,
)

CONFIDENCE_THRESHOLD = 0.7
MAX_CLARIFICATION_TURNS = 2


# ---------------------------------------------------------------------------
# Conditional edge: router → next node
# ---------------------------------------------------------------------------

def route_decision(state: AgentState) -> str:
    """
    Determine the next node after the router.

    Rules (in priority order):
      1. intent == 'complaint'           → 'fallback'
      2. confidence < threshold          → 'clarification'   (Confidence Gate)
      3. intent == 'policy'              → 'retriever'
      4. intent == 'sql'                 → 'sql'
      5. intent == 'web'                 → 'web'
      6. intent == 'product_info'        → 'product_info'
      7. safety net                      → 'fallback'

    Args:
        state: Current AgentState after router_node has run.

    Returns:
        Name of the next node as a string key.
    """
    intent: str = state.get("intent", "")
    confidence: float = state.get("confidence", 0.0)

    if intent == "complaint":
        return "fallback"

    if confidence < CONFIDENCE_THRESHOLD:
        return "clarification"

    route_map = {
        "policy": "retriever",
        "sql": "sql",
        "web": "web",
        "product_info": "product_info",
    }
    return route_map.get(intent, "fallback")


# ---------------------------------------------------------------------------
# Conditional edge: clarification → router (cycle) or fallback (guard)
# ---------------------------------------------------------------------------

def clarification_decision(state: AgentState) -> str:
    """
    Determine whether to cycle back to the router or escalate to fallback.

    Enforces the MAX_CLARIFICATION_TURNS guard to prevent infinite loops.
    If the turn count has reached the maximum, the graph escalates to fallback
    so the user is handed off to a human agent.

    Args:
        state: Current AgentState after clarification_node has run.

    Returns:
        'router' to re-route, or 'fallback' to escalate.
    """
    turn_count: int = state.get("clarification_turn_count", 0)
    if turn_count >= MAX_CLARIFICATION_TURNS:
        print(f"[Graph] Max clarification turns ({MAX_CLARIFICATION_TURNS}) reached — escalating to fallback.")
        return "fallback"
    return "router"


# ---------------------------------------------------------------------------
# Build and compile the graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Build the Omni-Help StateGraph with the clarification cycle closed.

    Returns:
        Compiled LangGraph ready for .invoke() / .ainvoke() / .stream().
    """
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("router", router_node)
    builder.add_node("retriever", retriever_node)
    builder.add_node("sql", sql_node)
    builder.add_node("web", web_node)
    builder.add_node("product_info", product_info_node)
    builder.add_node("fallback", fallback_node)
    builder.add_node("clarification", clarification_node)

    # Entry point
    builder.set_entry_point("router")

    # Router → pipeline nodes (conditional)
    builder.add_conditional_edges(
        "router",
        route_decision,
        {
            "retriever": "retriever",
            "sql": "sql",
            "web": "web",
            "product_info": "product_info",
            "fallback": "fallback",
            "clarification": "clarification",
        },
    )

    # Clarification → router (cycle) or fallback (guard) — THE CYCLE
    builder.add_conditional_edges(
        "clarification",
        clarification_decision,
        {
            "router": "router",
            "fallback": "fallback",
        },
    )

    # Pipeline stubs and fallback → END
    for node in ("retriever", "sql", "web", "product_info", "fallback"):
        builder.add_edge(node, END)

    return builder.compile()


# Module-level compiled graph — import this in eval and API
graph = build_graph()
