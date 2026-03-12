"""
Pydantic V2 request/response models for the Omni-Help API.

All models use strict Pydantic V2 field validation. The API surface is
intentionally narrow — internal AgentState complexity is never exposed to
callers. Only the fields a client needs are included in each model.
"""

import uuid
from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    Incoming chat request from a client.

    Fields:
        query:           The user's natural language question or message.
        conversation_id: Optional session identifier for multi-turn context.
                         A new UUID is generated if not provided, making
                         every request stateful by default.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's natural language query.",
        examples=["Where is my order ORD-1001?"],
    )
    conversation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Session ID for multi-turn conversation continuity. "
                    "Auto-generated as a UUID if not supplied by the client.",
    )


class ChatResponse(BaseModel):
    """
    Outgoing chat response returned to the client.

    Fields:
        response:        The synthesised natural language answer.
        intent:          The intent the router classified the query as.
        confidence:      The router's confidence score (0.0 – 1.0).
        correlation_id:  Unique ID for this specific request, used to trace
                         logs and LangSmith runs end-to-end.
    """

    response: str = Field(description="The synthesised answer from Omni-Help.")
    intent: str = Field(description="Classified intent (policy, sql, web, complaint, etc.).")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Router confidence score for the intent classification.",
    )
    correlation_id: str = Field(
        description="Unique request ID for log tracing and observability."
    )


class HealthResponse(BaseModel):
    """Minimal health-check response for load balancers and uptime monitors."""

    status: str = Field(default="healthy")
    version: str = Field(default="1.0.0")
