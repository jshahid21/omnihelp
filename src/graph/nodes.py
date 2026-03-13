"""
Omni-Help graph nodes.

router_node    — calls the Router Agent (The Brain); live (Phase 1).
retriever_node — queries ChromaDB and builds policy_context; live (Phase 4).
sql_node       — NL→SQL against the orders SQLite DB; live (Phase 5).
web_node       — Tavily web search for external/general queries; live (Phase 6).
synthesis_node — final LLM call; converts raw context to a conversational response (Phase 7).
Stub nodes     — product_info_node, fallback_node (Phase 7+).
"""

from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from graph.state import AgentState
from agents.router import router_agent
from tools.vector_store import get_policy_retriever, get_product_retriever, format_docs
from tools.sql_db import execute_secure_query, get_schema
from tools.web_search import execute_web_search, format_web_results
from prompts.synthesis_prompts import system_synthesis_prompt

load_dotenv()

# ---------------------------------------------------------------------------
# Module-level LLM clients — singletons, one per distinct temperature/role.
# ---------------------------------------------------------------------------
_sql_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)       # SQL: deterministic
_synthesis_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)  # Synthesis: conversational


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
    Policy RAG pipeline (Phase 4) — live implementation.

    Retrieves the top-k semantically similar policy chunks from ChromaDB
    for the current user query, formats them with source citations, and
    writes the result into policy_context.

    The final_response is a passthrough of the raw context so data flow
    is visible before the Synthesis Node is built in Phase 4.4.

    Args:
        state: Current AgentState; must contain 'user_query'.

    Returns:
        Partial state update: retrieved_docs, policy_context, final_response.
    """
    user_query: str = state.get("user_query", "")
    if not user_query:
        for msg in reversed(state.get("messages", [])):
            if hasattr(msg, "type") and msg.type == "human":
                user_query = msg.content
                break

    print(f"[Node] RETRIEVER — querying ChromaDB for: '{user_query}'")

    retriever = get_policy_retriever(k=5)
    docs = retriever.invoke(user_query)
    policy_context = format_docs(docs)

    print(f"[Node] RETRIEVER — retrieved {len(docs)} chunk(s).")

    return {
        "retrieved_docs": docs,
        "policy_context": policy_context,
        "final_response": f"Here is the raw policy context:\n\n{policy_context}",
    }


def sql_node(state: AgentState) -> dict[str, Any]:
    """
    Order/SQL pipeline (Phase 5) — live NL→SQL implementation.

    Converts the user's natural language query into a SQLite SELECT statement
    using GPT-4o-mini, validates it against the read-only guardrail (FR-015),
    executes it, and writes the result to AgentState.

    Args:
        state: Current AgentState; must contain 'user_query'.

    Returns:
        Partial state update: sql_query, sql_result (or sql_error), final_response.
    """
    user_query: str = state.get("user_query", "")
    if not user_query:
        for msg in reversed(state.get("messages", [])):
            if hasattr(msg, "type") and msg.type == "human":
                user_query = msg.content
                break

    print(f"[Node] SQL — generating query for: '{user_query}'")

    # --- NL → SQL ---
    schema = get_schema()
    system_prompt = (
        "You are an SQLite expert. Given an input question, create a syntactically "
        "correct SQLite query to run. Only return the raw SQL query — no markdown "
        "formatting, no backticks, no explanation.\n\n"
        f"Database schema:\n{schema}"
    )
    response = _sql_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query),
    ])
    sql_query: str = response.content.strip()
    print(f"[Node] SQL — generated: {sql_query}")

    # --- Secure execution (FR-015 guardrail) ---
    try:
        result = execute_secure_query(sql_query)
        print(f"[Node] SQL — result: {result}")
        return {
            "sql_query": sql_query,
            "sql_result": result,
            "sql_error": None,
            "final_response": f"SQL Result:\n{result}",
        }
    except ValueError as guard_err:
        # Read-only guardrail triggered — mutation attempt blocked
        error_msg = str(guard_err)
        print(f"[Node] SQL — GUARDRAIL BLOCKED: {error_msg}")
        return {
            "sql_query": sql_query,
            "sql_result": None,
            "sql_error": error_msg,
            "final_response": f"Security guardrail blocked this query: {error_msg}",
        }
    except Exception as exec_err:
        # Malformed SQL or DB error — surface cleanly without crashing the graph
        error_msg = f"Query execution failed: {exec_err}"
        print(f"[Node] SQL — ERROR: {error_msg}")
        return {
            "sql_query": sql_query,
            "sql_result": None,
            "sql_error": error_msg,
            "final_response": f"I was unable to retrieve that information. {error_msg}",
        }


def web_node(state: AgentState) -> dict[str, Any]:
    """
    Web search pipeline (Phase 6) — live Tavily implementation.

    Calls the Tavily search API for the user query, formats results with
    source citations, and writes them into web_context. Fails gracefully:
    if the API key is missing or the call fails, the error is surfaced in
    final_response without crashing the graph.

    Args:
        state: Current AgentState; must contain 'user_query'.

    Returns:
        Partial state update: web_results, web_context, final_response.
    """
    user_query: str = state.get("user_query", "")
    if not user_query:
        for msg in reversed(state.get("messages", [])):
            if hasattr(msg, "type") and msg.type == "human":
                user_query = msg.content
                break

    print(f"[Node] WEB — searching Tavily for: '{user_query}'")

    results = execute_web_search(user_query, max_results=3)
    web_context = format_web_results(results)

    print(f"[Node] WEB — received {len(results)} result(s).")

    return {
        "web_results": results,
        "web_context": web_context,
        "final_response": f"Web Search Results:\n\n{web_context}",
    }


def synthesis_node(state: AgentState) -> dict[str, Any]:
    """
    Synthesis node (Phase 7) — final LLM call across all pipelines.

    Selects whichever context is populated (policy_context, sql_result,
    web_context, or sql_error), injects it into the synthesis system prompt,
    and generates a natural conversational response via GPT-4o-mini.

    The response is written to final_response AND appended to the conversation
    history as an AIMessage so multi-turn context is preserved.

    Args:
        state: Current AgentState after a pipeline node has run.

    Returns:
        Partial state update: final_response and messages (AIMessage appended).
    """
    from langchain_core.messages import AIMessage

    user_query: str = state.get("user_query", "")

    # --- Pick the richest available context (priority order) ---
    policy_context: str | None = state.get("policy_context")
    product_context: str | None = state.get("product_context")
    sql_result: Any = state.get("sql_result")
    sql_query: str | None = state.get("sql_query")
    sql_error: str | None = state.get("sql_error")
    web_context: str | None = state.get("web_context")

    if sql_error:
        context = f"Database error: {sql_error}"
        source = "sql_error"
    elif sql_result is not None:
        # Include the SQL query so the LLM can read column names from the
        # SELECT clause and map them to the anonymous tuple values in the result.
        context = (
            f"SQL query executed:\n{sql_query}\n\n"
            f"Database result (rows match the SELECT columns above):\n{sql_result}"
        )
        source = "sql"
    elif policy_context:
        context = f"Policy documentation:\n{policy_context}"
        source = "policy"
    elif product_context:
        context = f"Product manual:\n{product_context}"
        source = "product"
    elif web_context:
        context = f"Web search results:\n{web_context}"
        source = "web"
    else:
        context = "No relevant context was found for this query."
        source = "none"

    print(f"[Node] SYNTHESIS — generating response (context source: {source})")

    filled_prompt = system_synthesis_prompt.format(context=context)

    response = _synthesis_llm.invoke([
        SystemMessage(content=filled_prompt),
        HumanMessage(content=user_query),
    ])
    final_response: str = response.content.strip()

    return {
        "final_response": final_response,
        # Append to conversation history — enables multi-turn memory
        "messages": [AIMessage(content=final_response)],
    }


def product_info_node(state: AgentState) -> dict[str, Any]:
    """
    Product Info RAG pipeline — live implementation.

    Retrieves the top-k semantically similar product manual chunks from the
    'products' ChromaDB collection and writes the result into product_context
    for the Synthesis Node to consume.

    Args:
        state: Current AgentState; must contain 'user_query'.

    Returns:
        Partial state update: product_docs, product_context, final_response.
    """
    user_query: str = state.get("user_query", "")
    if not user_query:
        for msg in reversed(state.get("messages", [])):
            if hasattr(msg, "type") and msg.type == "human":
                user_query = msg.content
                break

    print(f"[Node] PRODUCT_INFO — querying product manuals for: '{user_query}'")

    retriever = get_product_retriever(k=3)
    docs = retriever.invoke(user_query)
    product_context = format_docs(docs)

    print(f"[Node] PRODUCT_INFO — retrieved {len(docs)} chunk(s).")

    return {
        "product_docs": docs,
        "product_context": product_context,
        "final_response": f"Here is the raw product context:\n\n{product_context}",
    }


def fallback_node(state: AgentState) -> dict[str, Any]:
    """
    Fallback / human escalation node (Phase 7, FR-028).

    Builds a handoff_context snapshot of the conversation so a human support
    agent can pick up with full context — no re-telling needed by the user.
    Fires for complaints (intent == 'complaint') and after MAX_CLARIFICATION_TURNS
    of low-confidence cycling.

    Args:
        state: Current AgentState; reads user_query, intent, routing_rationale,
               clarification_turn_count, and messages.

    Returns:
        Partial state update: handoff_context, fallback_reason, final_response.
    """
    from langchain_core.messages import AIMessage

    user_query: str = state.get("user_query", "")
    intent: str = state.get("intent", "unknown")
    routing_rationale: str | None = state.get("routing_rationale")
    turn_count: int = state.get("clarification_turn_count", 0)
    messages = state.get("messages", [])

    # Determine why fallback was triggered
    if intent == "complaint":
        reason = "complaint — user expressed dissatisfaction or requested escalation"
    elif turn_count >= 2:
        reason = f"low_confidence — clarification exhausted after {turn_count} turns"
    else:
        reason = state.get("fallback_reason", "unspecified")

    # Build a structured handoff packet for the human agent
    handoff_context: dict[str, Any] = {
        "original_query": user_query,
        "detected_intent": intent,
        "routing_rationale": routing_rationale,
        "escalation_reason": reason,
        "clarification_turns": turn_count,
        "conversation_length": len(messages),
    }

    print(f"[Node] FALLBACK — escalating to human. Reason: {reason}")

    final_response = (
        "I apologize, but I am unable to fully resolve this for you. "
        "I have escalated your ticket to our human support team with all "
        "the context of our conversation. A support specialist will be in "
        "touch shortly. Thank you for your patience."
    )

    return {
        "handoff_context": handoff_context,
        "fallback_reason": reason,
        "final_response": final_response,
        "messages": [AIMessage(content=final_response)],
    }


def clarification_node(state: AgentState) -> dict[str, Any]:
    """
    Clarification node for low-confidence routes (Phase 3).

    Increments clarification_turn_count so the cycle guard in graph.py can
    break the loop after MAX_CLARIFICATION_TURNS. Clears missing_info so the
    router can re-classify cleanly on the next turn.

    Args:
        state: Current AgentState; reads clarification_turn_count and missing_info.

    Returns:
        Partial state update: incremented turn count, cleared missing_info,
        and a clarifying question as final_response.
    """
    turn = state.get("clarification_turn_count", 0) + 1
    print(f"[Node] Reached CLARIFICATION — turn {turn}")
    return {
        "clarification_turn_count": turn,
        "missing_info": [],
        "final_response": (
            "I want to make sure I help you correctly. "
            "Could you provide a bit more detail about your question?"
        ),
    }
