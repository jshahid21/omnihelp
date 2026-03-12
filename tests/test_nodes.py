"""
Unit tests for all live Omni-Help graph nodes.

Strategy
--------
Every external dependency is mocked so the suite runs completely offline
and finishes in under a second. No OpenAI, ChromaDB, SQLite, or Tavily calls
are made. We test:

  - retriever_node  : ChromaDB retriever + format_docs mocked
  - sql_node        : LLM + execute_secure_query mocked; also tests the
                      FR-015 guardrail and DB error paths
  - web_node        : execute_web_search + format_web_results mocked
  - synthesis_node  : LLM mocked; tests all four context-priority branches
  - fallback_node   : pure state logic; no mocks needed

Patch targets use the module where the name is bound (not where it is defined)
to ensure the correct reference is replaced at test time.
"""

import sys
import os
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ===========================================================================
# Helpers
# ===========================================================================

def _base_state(**kwargs) -> dict:
    """Return a minimal AgentState-like dict with sensible defaults."""
    defaults = {
        "user_query": "test query",
        "messages": [],
        "intent": "policy",
        "confidence": 0.95,
        "routing_rationale": "clear policy question",
        "clarification_turn_count": 0,
    }
    defaults.update(kwargs)
    return defaults


# ===========================================================================
# retriever_node
# ===========================================================================

class TestRetrieverNode:
    """Tests for the Policy RAG retriever node."""

    def _make_mock_doc(self, content: str, source: str):
        doc = MagicMock()
        doc.page_content = content
        doc.metadata = {"source": source}
        return doc

    def test_retriever_node_populates_policy_context(self):
        """Should write retrieved chunks as policy_context with citations."""
        mock_doc = self._make_mock_doc(
            "Returns allowed within 30 days.", "data/policies/return_policy.md"
        )
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [mock_doc]

        with patch("graph.nodes.get_policy_retriever", return_value=mock_retriever), \
             patch("graph.nodes.format_docs", return_value="Returns allowed within 30 days.\n[Source: data/policies/return_policy.md]"):
            from graph.nodes import retriever_node
            result = retriever_node(_base_state(user_query="What is the return policy?"))

        assert "policy_context" in result
        assert "Source" in result["policy_context"]
        assert "retrieved_docs" in result
        assert "final_response" in result

    def test_retriever_node_falls_back_to_message_history(self):
        """Should extract query from messages when user_query is empty."""
        from langchain_core.messages import HumanMessage
        mock_doc = self._make_mock_doc("Policy content.", "policy.md")
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [mock_doc]

        state = _base_state(user_query="")
        state["messages"] = [HumanMessage(content="What is the return window?")]

        with patch("graph.nodes.get_policy_retriever", return_value=mock_retriever), \
             patch("graph.nodes.format_docs", return_value="Policy content.\n[Source: policy.md]"):
            from graph.nodes import retriever_node
            result = retriever_node(state)

        mock_retriever.invoke.assert_called_once_with("What is the return window?")
        assert result["policy_context"] is not None


# ===========================================================================
# sql_node
# ===========================================================================

class TestSqlNode:
    """Tests for the NL→SQL order pipeline node."""

    def _mock_llm_response(self, sql: str) -> MagicMock:
        resp = MagicMock()
        resp.content = sql
        return resp

    def test_sql_node_happy_path(self):
        """Should generate SQL, execute it, and return sql_result."""
        with patch("graph.nodes.get_schema", return_value="CREATE TABLE orders ..."), \
             patch("graph.nodes._sql_llm") as mock_llm, \
             patch("graph.nodes.execute_secure_query", return_value="[('ORD-1001', 'Shipped')]"):
            mock_llm.invoke.return_value = self._mock_llm_response(
                "SELECT order_id, status FROM orders WHERE order_id = 'ORD-1001'"
            )
            from graph.nodes import sql_node
            result = sql_node(_base_state(user_query="Where is order ORD-1001?"))

        assert result["sql_result"] == "[('ORD-1001', 'Shipped')]"
        assert result["sql_error"] is None
        assert "ORD-1001" in result["sql_query"]
        assert "SQL Result" in result["final_response"]

    def test_sql_node_fr015_guardrail_blocked(self):
        """FR-015: mutation SQL must be blocked and surface in sql_error."""
        with patch("graph.nodes.get_schema", return_value="CREATE TABLE orders ..."), \
             patch("graph.nodes._sql_llm") as mock_llm, \
             patch("graph.nodes.execute_secure_query",
                   side_effect=ValueError("Forbidden SQL operation detected: 'DROP'")):
            mock_llm.invoke.return_value = self._mock_llm_response("DROP TABLE orders")
            from graph.nodes import sql_node
            result = sql_node(_base_state(user_query="Delete all orders"))

        assert result["sql_result"] is None
        assert "Forbidden" in result["sql_error"]
        assert "guardrail" in result["final_response"].lower()

    def test_sql_node_db_error_handled_gracefully(self):
        """Generic DB errors should surface in sql_error without crashing."""
        with patch("graph.nodes.get_schema", return_value="CREATE TABLE orders ..."), \
             patch("graph.nodes._sql_llm") as mock_llm, \
             patch("graph.nodes.execute_secure_query",
                   side_effect=Exception("no such table: shipments")):
            mock_llm.invoke.return_value = self._mock_llm_response(
                "SELECT * FROM shipments"
            )
            from graph.nodes import sql_node
            result = sql_node(_base_state(user_query="Show shipments"))

        assert result["sql_result"] is None
        assert "no such table" in result["sql_error"]


# ===========================================================================
# web_node
# ===========================================================================

class TestWebNode:
    """Tests for the Tavily web search node."""

    def test_web_node_happy_path(self):
        """Should call search, format results, and write web_context."""
        mock_results = [
            {"title": "AI News", "url": "https://example.com", "content": "AI regulations update."}
        ]
        with patch("graph.nodes.execute_web_search", return_value=mock_results), \
             patch("graph.nodes.format_web_results",
                   return_value="AI regulations update.\n[Source: https://example.com]"):
            from graph.nodes import web_node
            result = web_node(_base_state(user_query="Latest AI news"))

        assert result["web_results"] == mock_results
        assert "Source" in result["web_context"]
        assert "Web Search Results" in result["final_response"]

    def test_web_node_graceful_on_empty_results(self):
        """Empty search results should not crash — returns 'no results' message."""
        with patch("graph.nodes.execute_web_search", return_value=[]), \
             patch("graph.nodes.format_web_results", return_value="No web results found."):
            from graph.nodes import web_node
            result = web_node(_base_state(user_query="Very obscure query"))

        assert result["web_context"] == "No web results found."
        assert result["final_response"] is not None

    def test_web_node_graceful_on_api_error(self):
        """Structured error dict from execute_web_search must not crash the node."""
        error_results = [{"title": "Web Search Error", "url": "", "content": "Search failed: timeout"}]
        with patch("graph.nodes.execute_web_search", return_value=error_results), \
             patch("graph.nodes.format_web_results", return_value="Search failed: timeout\n[Source: unknown]"):
            from graph.nodes import web_node
            result = web_node(_base_state(user_query="Query that times out"))

        assert "final_response" in result


# ===========================================================================
# synthesis_node
# ===========================================================================

class TestSynthesisNode:
    """Tests for the final LLM synthesis node."""

    def _mock_synthesis_response(self, text: str) -> MagicMock:
        resp = MagicMock()
        resp.content = text
        return resp

    def test_synthesis_uses_sql_result(self):
        """sql_result context should produce a grounded response."""
        with patch("graph.nodes._synthesis_llm") as mock_llm:
            mock_llm.invoke.return_value = self._mock_synthesis_response(
                "Your order ORD-1001 is Shipped via UPS."
            )
            from graph.nodes import synthesis_node
            result = synthesis_node(_base_state(
                user_query="Where is ORD-1001?",
                sql_query="SELECT tracking_number FROM shipments WHERE order_id='ORD-1001'",
                sql_result="[('1Z999AA1012345678', 'UPS', 'Shipped')]",
            ))

        assert result["final_response"] == "Your order ORD-1001 is Shipped via UPS."
        # Verify the LLM was called with sql context (not policy or web)
        call_args = mock_llm.invoke.call_args[0][0]
        system_content = call_args[0].content
        assert "SQL query executed" in system_content

    def test_synthesis_uses_policy_context(self):
        """policy_context should be used when sql_result is absent."""
        with patch("graph.nodes._synthesis_llm") as mock_llm:
            mock_llm.invoke.return_value = self._mock_synthesis_response(
                "You can return items within 30 days."
            )
            from graph.nodes import synthesis_node
            result = synthesis_node(_base_state(
                user_query="What is the return policy?",
                policy_context="Returns allowed within 30 days.\n[Source: return_policy.md]",
            ))

        call_args = mock_llm.invoke.call_args[0][0]
        system_content = call_args[0].content
        assert "Policy documentation" in system_content
        assert result["final_response"] == "You can return items within 30 days."

    def test_synthesis_uses_web_context(self):
        """web_context should be used when neither sql nor policy is present."""
        with patch("graph.nodes._synthesis_llm") as mock_llm:
            mock_llm.invoke.return_value = self._mock_synthesis_response(
                "Recent AI regulations include..."
            )
            from graph.nodes import synthesis_node
            result = synthesis_node(_base_state(
                user_query="Latest AI news",
                web_context="EU AI Act update.\n[Source: https://example.com]",
            ))

        call_args = mock_llm.invoke.call_args[0][0]
        system_content = call_args[0].content
        assert "Web search results" in system_content

    def test_synthesis_surfaces_sql_error_politely(self):
        """sql_error should take priority and be injected as context."""
        with patch("graph.nodes._synthesis_llm") as mock_llm:
            mock_llm.invoke.return_value = self._mock_synthesis_response(
                "I was unable to retrieve that information."
            )
            from graph.nodes import synthesis_node
            result = synthesis_node(_base_state(
                user_query="Delete all orders",
                sql_error="Forbidden SQL operation detected: 'DROP'",
            ))

        call_args = mock_llm.invoke.call_args[0][0]
        system_content = call_args[0].content
        assert "Database error" in system_content

    def test_synthesis_appends_ai_message_to_history(self):
        """AIMessage must be returned in 'messages' for conversation history."""
        from langchain_core.messages import AIMessage
        with patch("graph.nodes._synthesis_llm") as mock_llm:
            mock_llm.invoke.return_value = self._mock_synthesis_response("Hello!")
            from graph.nodes import synthesis_node
            result = synthesis_node(_base_state(
                user_query="Hi",
                policy_context="Some policy.",
            ))

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert result["messages"][0].content == "Hello!"


# ===========================================================================
# fallback_node
# ===========================================================================

class TestFallbackNode:
    """Tests for the human escalation fallback node (no mocks needed)."""

    def test_fallback_sets_handoff_context(self):
        """handoff_context must contain original query, intent, and reason."""
        from graph.nodes import fallback_node
        result = fallback_node(_base_state(
            user_query="I want to speak to a manager!",
            intent="complaint",
            routing_rationale="User expressed dissatisfaction.",
        ))

        assert "handoff_context" in result
        ctx = result["handoff_context"]
        assert ctx["original_query"] == "I want to speak to a manager!"
        assert ctx["detected_intent"] == "complaint"
        assert "complaint" in ctx["escalation_reason"]

    def test_fallback_reason_low_confidence(self):
        """Exhausted clarification turns should set escalation reason correctly."""
        from graph.nodes import fallback_node
        result = fallback_node(_base_state(
            intent="policy",
            clarification_turn_count=2,
        ))

        assert "low_confidence" in result["handoff_context"]["escalation_reason"]

    def test_fallback_returns_polite_final_response(self):
        """final_response must be a non-empty escalation message."""
        from graph.nodes import fallback_node
        result = fallback_node(_base_state(intent="complaint"))

        assert "final_response" in result
        assert len(result["final_response"]) > 20
        assert "escalated" in result["final_response"].lower()

    def test_fallback_appends_ai_message(self):
        """AIMessage should be appended to history for multi-turn continuity."""
        from langchain_core.messages import AIMessage
        from graph.nodes import fallback_node
        result = fallback_node(_base_state(intent="complaint"))

        assert "messages" in result
        assert isinstance(result["messages"][0], AIMessage)
