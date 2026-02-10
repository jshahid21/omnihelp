# Omni-Help Application Structure Plan

**Source:** Omni-Help Project Blueprint (Complete Project Blueprint, Version 1.0)  
**Purpose:** Application structure and directory plan for the Adaptive RAG Router — intelligent customer support with multi-source routing.

---

## 1. Executive Summary (from Blueprint)

Omni-Help is an intelligent customer support platform that uses **adaptive multi-source routing** instead of a single RAG pipeline. A central **Router Agent** classifies user intent and dispatches to the right source: vector DB (policy docs), SQL (orders), or web search (fallback). Target: **>95% routing accuracy**, **70% reduction in resolution time**.

---

## 2. High-Level Application Structure

```
omnihelp/
├── README.md
├── APPLICATION_STRUCTURE_PLAN.md    # This document
├── pyproject.toml                   # or requirements.txt
├── .env.example
├── .gitignore
│
├── src/                             # Main application package
│   ├── __init__.py
│   ├── main.py                      # FastAPI app entry
│   │
│   ├── api/                         # HTTP layer
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── chat.py              # POST /api/v1/chat
│   │   │   ├── conversations.py    # GET/POST conversations, feedback
│   │   │   ├── documents.py        # Document upload/delete (ingestion)
│   │   │   ├── analytics.py        # Routing & satisfaction metrics
│   │   │   ├── health.py            # Health check
│   │   │   └── admin.py             # Config GET/PUT
│   │   ├── dependencies.py          # Auth, DB sessions, etc.
│   │   └── schemas/                 # Pydantic request/response models
│   │       ├── __init__.py
│   │       ├── chat.py
│   │       ├── conversation.py
│   │       └── analytics.py
│   │
│   ├── core/                        # Config, constants, security
│   │   ├── __init__.py
│   │   ├── config.py                # Settings (env, ROUTER_CONFIDENCE_THRESHOLD, etc.)
│   │   ├── constants.py             # Intent labels, escalation keywords
│   │   └── security.py              # PII masking, validation
│   │
│   ├── graph/                       # LangGraph agentic workflow
│   │   ├── __init__.py
│   │   ├── state.py                 # LangGraph state schema (messages, intent, route, etc.)
│   │   ├── nodes/
│   │   │   ├── __init__.py
│   │   │   ├── router.py            # Router Agent (intent classification)
│   │   │   ├── retriever.py         # Policy RAG (vector search + optional rerank)
│   │   │   ├── sql_node.py          # Order management (NL2SQL)
│   │   │   ├── web_search.py        # Fallback web search
│   │   │   └── synthesis.py         # Response synthesis & tone
│   │   ├── edges.py                 # Conditional routing logic (state machine)
│   │   └── graph.py                 # Compose graph from nodes + edges
│   │
│   ├── services/                    # Business logic & integrations
│   │   ├── __init__.py
│   │   ├── conversation_service.py  # Conversation memory, context (e.g. last N turns)
│   │   ├── vector_store.py          # Qdrant client, ingestion, semantic/hybrid search
│   │   ├── sql_service.py           # DB connection, NL2SQL, validation (SELECT only)
│   │   ├── web_search_service.py    # External search API, rate limiting
│   │   ├── llm_service.py           # LLM client (OpenAI/Claude/Ollama), token/cost tracking
│   │   └── feedback_service.py      # Thumbs up/down, analytics
│   │
│   ├── models/                      # Domain & data models (if not only in schemas)
│   │   ├── __init__.py
│   │   ├── intent.py                # Intent enum, confidence
│   │   └── document.py              # Chunk, metadata for vector store
│   │
│   └── db/                          # Persistence (optional separate from services)
│       ├── __init__.py
│       ├── session.py              # SQL session factory
│       └── redis_client.py          # Cache (e.g. CACHE_TTL_SECONDS)
│
├── scripts/                         # One-off and dev scripts
│   ├── ingest_documents.py          # Ingest PDF/DOCX/TXT/HTML into vector DB
│   ├── seed_orders.py               # Seed order DB for dev
│   └── run_eval.py                  # Routing accuracy / golden dataset
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # Pytest fixtures (client, mocks)
│   ├── unit/
│   │   ├── test_router.py
│   │   ├── test_retriever.py
│   │   ├── test_sql_node.py
│   │   └── test_synthesis.py
│   ├── integration/
│   │   ├── test_api_chat.py
│   │   ├── test_documents_api.py
│   │   └── test_analytics_api.py
│   └── e2e/
│       └── test_conversation_flows.py
│
├── frontend/                        # Optional: chat UI (e.g. Next.js/React/Streamlit)
│   └── (Next.js or Streamlit app per blueprint)
│
├── config/                          # Static config files (optional)
│   ├── prompts/
│   │   ├── router_classification.txt
│   │   └── synthesis_system.txt
│   └── query_templates/             # Parameterized SQL templates (order tracking, etc.)
│       └── orders.yaml
│
└── docs/                            # Design and ops docs
    ├── ARCHITECTURE.md
    ├── API.md                       # API endpoint reference (blueprint 11.3)
    └── RUNBOOK.md
```

---

## 3. Component Mapping to Blueprint

| Blueprint Section | Implementation Location |
|-------------------|--------------------------|
| **2.1.1 Router Agent** | `src/graph/nodes/router.py`, `config/prompts/router_classification.txt` |
| **2.1.2 Retriever Node (Policy RAG)** | `src/graph/nodes/retriever.py`, `src/services/vector_store.py` |
| **2.1.3 SQL Node (Order Management)** | `src/graph/nodes/sql_node.py`, `src/services/sql_service.py`, `config/query_templates/` |
| **2.1.4 Web Search Node** | `src/graph/nodes/web_search.py`, `src/services/web_search_service.py` |
| **2.1.5 Response Synthesis & UX** | `src/graph/nodes/synthesis.py`, `src/api/routes/chat.py`, `src/api/schemas/` |
| **4.1 Node Definitions & 4.1.2 Routing Logic** | `src/graph/state.py`, `src/graph/edges.py`, `src/graph/graph.py` |
| **4.2 ReAct / Agentic** | `src/graph/nodes/*` (reasoning + tool calls), `src/graph/graph.py` |
| **11.3 API Endpoints** | `src/api/routes/*` |

---

## 4. Technology Stack (from Blueprint §3)

| Layer | Recommended | Notes |
|-------|-------------|--------|
| **Orchestration** | LangChain / **LangGraph** | State machine, conditional edges |
| **API** | **FastAPI** | Async, OpenAPI, WebSockets for chat |
| **Vector DB** | **Qdrant** | Semantic + optional hybrid (BM25) |
| **SQL DB** | **PostgreSQL** | Orders, conversation metadata |
| **Cache** | **Redis** | Response cache, rate limits, session |
| **LLM** | OpenAI / Azure OpenAI / **Ollama** | Router, NL2SQL, synthesis; fallbacks |
| **Embeddings** | OpenAI / Cohere / local | For policy document indexing |
| **Frontend** | Next.js or Streamlit | Web chat, admin dashboard |

---

## 5. Key Configuration (from Blueprint §11.2)

- `ROUTER_CONFIDENCE_THRESHOLD` (default 0.7) — routing vs clarification/escalation  
- `RETRIEVER_TOP_K`, `RETRIEVER_SIMILARITY_THRESHOLD` (e.g. 0.75)  
- `RERANKER_ENABLED`, `SQL_QUERY_TIMEOUT`, `WEB_SEARCH_MAX_RESULTS`  
- `CONVERSATION_MEMORY_LENGTH` (e.g. 10), `MAX_RESPONSE_TOKENS`  
- `ESCALATION_KEYWORDS`, `CACHE_TTL_SECONDS`  

These should be centralized in `src/core/config.py` and optionally exposed via `GET/PUT /api/v1/admin/config`.

---

## 6. API Surface (from Blueprint §11.3)

| Endpoint | Method | Module |
|----------|--------|--------|
| `/api/v1/chat` | POST | `api/routes/chat.py` |
| `/api/v1/conversations/{id}` | GET | `api/routes/conversations.py` |
| `/api/v1/conversations/{id}/feedback` | POST | `api/routes/conversations.py` |
| `/api/v1/documents` | POST | `api/routes/documents.py` |
| `/api/v1/documents/{id}` | DELETE | `api/routes/documents.py` |
| `/api/v1/analytics/routing` | GET | `api/routes/analytics.py` |
| `/api/v1/analytics/satisfaction` | GET | `api/routes/analytics.py` |
| `/api/v1/health` | GET | `api/routes/health.py` |
| `/api/v1/admin/config` | GET/PUT | `api/routes/admin.py` |

---

## 7. Implementation Order (aligned with Blueprint §6)

1. **Core config & state** — `core/config.py`, `graph/state.py`  
2. **Router node** — Classification prompt, intent enum, confidence thresholds  
3. **Retriever node** — Vector store service, ingestion script, top-k + optional rerank  
4. **SQL node** — Schema-aware NL2SQL, validation (SELECT only), templates  
5. **Web search node** — Fallback, rate limiting, synthesis from multiple sources  
6. **Synthesis node** — Response generation, tone, citations  
7. **Graph assembly** — Nodes + conditional edges, single entry/exit  
8. **API layer** — All routes, schemas, dependencies  
9. **Conversation & feedback** — Memory, handoff, analytics  
10. **Tests & deployment** — Unit, integration, e2e; Docker/CI and monitoring  

---

## 8. Non-Functional Alignment

- **Latency:** Router &lt;200ms, p95 end-to-end &lt;3s (cache, async, indexing).  
- **Security:** Read-only DB user for NL2SQL, PII masking, TLS.  
- **Observability:** Correlation IDs, LangSmith (or similar), routing and satisfaction dashboards.  
- **Scaling:** Stateless API, horizontal scaling; Redis for cache/session; DB connection pooling.  

---

*This plan is derived from the Omni-Help Project Blueprint (Version 1.0) and is intended to guide repository layout and implementation order.*
