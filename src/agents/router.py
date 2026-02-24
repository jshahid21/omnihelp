"""
Router Agent (The Brain): classifies user queries using Structured Output (JSON mode).

Implements the Confidence Gate: confidence < 0.7 routes to the Clarification Node.
Uses ChatOpenAI (gpt-4o-mini) with .with_structured_output() so the LLM is
forced to return a validated Pydantic object — no manual JSON parsing.
"""

from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from graph.state import AgentState, IntentType
from prompts.router_prompts import standard_router_prompt


# ---------------------------------------------------------------------------
# Structured output schema — the LLM MUST return exactly this shape.
# ---------------------------------------------------------------------------

class RouterOutput(BaseModel):
    """Structured output emitted by the Router LLM call."""

    intent: Literal["policy", "sql", "web", "product_info", "complaint"] = Field(
        description="The classified intent of the user's query."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score in [0, 1]. Below 0.7 triggers Clarification Node.",
    )
    reasoning: str = Field(
        description="One-sentence explanation for the intent classification."
    )


# ---------------------------------------------------------------------------
# LLM client — instantiated once at module load, reused per request.
# ---------------------------------------------------------------------------

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
_structured_llm = _llm.with_structured_output(RouterOutput)


# ---------------------------------------------------------------------------
# Router agent
# ---------------------------------------------------------------------------

def router_agent(state: AgentState) -> dict:
    """
    Classify the user query and return intent, confidence, and routing_rationale.

    Args:
        state: Current AgentState. Must contain 'user_query'.

    Returns:
        Partial state update with 'intent', 'confidence', 'routing_rationale',
        and 'missing_info' (empty list when confidence >= 0.7).

    Raises:
        ValueError: If 'user_query' is absent from state.

    Example:
        >>> result = router_agent({"user_query": "Where is my order #123?"})
        >>> result["intent"]
        'sql'
    """
    user_query: str = state.get("user_query", "")
    if not user_query:
        # Fall back to last human message if user_query not set
        for msg in reversed(state.get("messages", [])):
            if hasattr(msg, "type") and msg.type == "human":
                user_query = msg.content
                break

    if not user_query:
        raise ValueError("AgentState must contain 'user_query' or at least one HumanMessage.")

    messages = [
        SystemMessage(content=standard_router_prompt),
        HumanMessage(content=user_query),
    ]

    output: RouterOutput = _structured_llm.invoke(messages)

    # Confidence Gate: signal missing_info when confidence is low.
    missing_info = [] if output.confidence >= 0.7 else ["low_confidence"]

    return {
        "intent": output.intent,
        "confidence": output.confidence,
        "routing_rationale": output.reasoning,
        "missing_info": missing_info,
    }
