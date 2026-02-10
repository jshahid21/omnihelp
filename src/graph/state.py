"""
Omni-Help graph state: single source of truth for the LangGraph.

State-First Strategy: AgentState defines all keys consumed and produced
by nodes. Conversation history uses Annotated + add_messages reducer.
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

# -----------------------------------------------------------------------------
# Intent & route types (Blueprint: Policy_Question, Order_Status, General_Chat,
# Product_Info, Complaint). Mapped to graph nodes.
# -----------------------------------------------------------------------------
IntentType = Literal[
    "policy",        # Policy_Question → Retriever
    "sql",           # Order_Status → SQL Node
    "web",           # General_Chat → Web Node
    "product_info",  # Product_Info → policy or web
    "complaint",     # Complaint → Fallback / Clarification
]
RouteType = Literal["policy", "sql", "web", "clarification", "fallback"]


class AgentState(TypedDict, total=False):
    """
    Shared state for the Omni-Help LangGraph. All nodes read and write this.

    State-First: intent, confidence, missing_info, and routing_rationale
    are required for the Router and Confidence Gate. messages use the
    add_messages reducer for append semantics.
    """

    # --- Conversation (reducer: append) ---
    messages: Annotated[List[BaseMessage], add_messages]

    # --- Input ---
    user_query: str

    # --- Router (Brain) output — crucial for State-First ---
    intent: IntentType
    confidence: float  # [0, 1]; Confidence Gate: if < 0.7 → Clarification Node
    missing_info: List[str]  # Used by Clarification / self-correction
    routing_rationale: Optional[str]  # FR-005: log 100% of decisions

    # --- Graph route (derived from intent + confidence) ---
    route: RouteType

    # --- Policy (RAG) pipeline ---
    retrieved_docs: List[Any]
    policy_context: Optional[str]

    # --- SQL (Order) pipeline ---
    sql_query: Optional[str]
    sql_result: Optional[List[Dict[str, Any]]]
    sql_error: Optional[str]

    # --- Web pipeline ---
    web_results: Optional[List[Dict[str, Any]]]
    web_context: Optional[str]

    # --- Fallback / Clarification ---
    fallback_reason: Optional[str]
    handoff_context: Optional[Dict[str, Any]]

    # --- Synthesis ---
    final_response: Optional[str]
    error: Optional[str]
