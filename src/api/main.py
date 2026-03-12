"""
Omni-Help FastAPI application.

Exposes two routes:
  GET  /api/v1/health  — liveness probe for load balancers / uptime monitors.
  POST /api/v1/chat    — accepts a natural language query, invokes the
                         LangGraph orchestration engine asynchronously, and
                         returns a synthesised response with routing metadata.

Design decisions:
  - graph is loaded once at startup via the lifespan context manager (not
    per-request) so the compiled StateGraph and all LLM clients are reused.
  - Every request gets a correlation_id (UUID) for end-to-end log tracing.
  - conversation_id is forwarded to LangGraph's thread_id so multi-turn
    context is maintained across requests from the same session.
  - Errors are caught at the route level and surfaced as structured HTTP 500
    responses — the graph never leaks raw Python tracebacks to clients.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
import os
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Ensure src/ is on the path when uvicorn loads the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

load_dotenv()

from api.schema import ChatRequest, ChatResponse, HealthResponse

# ---------------------------------------------------------------------------
# Logging — structured, level from env (default INFO)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("omnihelp.api")

# ---------------------------------------------------------------------------
# Compiled graph — loaded once at startup
# ---------------------------------------------------------------------------

_graph: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: compile and cache the LangGraph StateGraph.
    Shutdown: nothing to tear down for the in-process graph.
    """
    global _graph
    logger.info("Starting Omni-Help API — compiling LangGraph...")
    from graph.graph import build_graph
    _graph = build_graph()
    logger.info("LangGraph compiled and ready.")
    yield
    logger.info("Omni-Help API shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Omni-Help API",
    version="1.0.0",
    description=(
        "Enterprise Adaptive RAG Router. Routes customer queries to the "
        "correct pipeline (Policy RAG, Order SQL, Web Search) and returns "
        "a synthesised natural language response."
    ),
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Security: request body size limit (1 MB) — guards against payload DoS
# ---------------------------------------------------------------------------

MAX_REQUEST_BODY_BYTES = 1 * 1024 * 1024  # 1 MB


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose body exceeds MAX_REQUEST_BODY_BYTES."""

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={"error": "Request body too large. Maximum size is 1 MB."},
            )
        return await call_next(request)


app.add_middleware(RequestSizeLimitMiddleware)

# ---------------------------------------------------------------------------
# CORS — Streamlit (8501) and Next.js (3000) only; tighten further for prod
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",   # Streamlit
        "http://localhost:3000",   # Next.js
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    tags=["Ops"],
)
async def health_check() -> HealthResponse:
    """
    Returns 200 + {"status": "healthy"} when the server is up.
    Used by load balancers, uptime monitors, and Kubernetes readiness probes.
    """
    return HealthResponse()


@app.post(
    "/api/v1/chat",
    response_model=ChatResponse,
    summary="Submit a customer query",
    tags=["Chat"],
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Accept a natural language query and return a synthesised response.

    The request is routed through the full LangGraph pipeline:
      Router → (Policy | SQL | Web | Clarification | Fallback) → Synthesis

    Args:
        request: ChatRequest with query and optional conversation_id.

    Returns:
        ChatResponse with the synthesised answer, intent, confidence, and
        a correlation_id for log tracing.

    Raises:
        HTTPException 500: If the graph invocation fails unexpectedly.
    """
    correlation_id = str(uuid.uuid4())

    logger.info(
        "Request started | correlation_id=%s | conversation_id=%s | query=%r",
        correlation_id,
        request.conversation_id,
        request.query[:120],   # Truncate long queries in logs
    )

    try:
        result = await _graph.ainvoke(
            {"user_query": request.query},
            config={"configurable": {"thread_id": request.conversation_id}},
        )
    except Exception as exc:
        logger.error(
            "Graph invocation failed | correlation_id=%s | error=%s",
            correlation_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "An internal error occurred while processing your request.",
                "correlation_id": correlation_id,
            },
        )

    final_response: str = result.get(
        "final_response",
        "I was unable to generate a response. Please try again.",
    )
    intent: str = result.get("intent", "unknown")
    confidence: float = result.get("confidence", 0.0)

    logger.info(
        "Request complete | correlation_id=%s | intent=%s | confidence=%.2f",
        correlation_id,
        intent,
        confidence,
    )

    return ChatResponse(
        response=final_response,
        intent=intent,
        confidence=confidence,
        correlation_id=correlation_id,
    )
