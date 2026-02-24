"""
Omni-Help LangGraph: cyclic StateGraph.

Topology
--------
START → router_node
router_node → (conditional) retriever | sql | web | product_info | clarification | fallback
retriever / sql / web / product_info → END      (stubs for now)
fallback → END
clarification → END                             (cyclic re-entry to router added in Phase 3)

Confidence Gate
---------------
intent == 'complaint'  OR  confidence < 0.7  →  fallback / clarification
otherwise: route by intent.
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


# ---------------------------------------------------------------------------
# Conditional edge: route_decision
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
      7. fallback (safety net)           → 'fallback'

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
# Build and compile the graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Build the Omni-Help StateGraph.

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

    # Conditional edges from router
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

    # All terminal nodes → END (Phase 3 will add cycle: clarification → router)
    for node in ("retriever", "sql", "web", "product_info", "fallback", "clarification"):
        builder.add_edge(node, END)

    return builder.compile()


# Module-level compiled graph — import this in eval and API
graph = build_graph()
