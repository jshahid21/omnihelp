# Omni-Help — Application Structure Plan

**Source:** Omni-Help Project Blueprint (Complete Project Blueprint, Version 1.0)  
**Purpose:** Implementation-ready directory layout, modules, and responsibilities for the Adaptive RAG Router platform.

---

## 1. Document Scope (Blueprint Coverage)

This plan is derived from the full blueprint, including:

- **§1 Executive Summary** — Problem, solution, classification-first paradigm  
- **§2 System Requirements** — Router, Retriever, SQL, Web Search, Response Synthesis, NFRs  
- **§3 Technology Stack & Data Sources** — Stack choices, vector DB, SQL schema, web search, API costs  
- **§4 Agentic Architecture** — Nodes, state machine, ReAct, workflows  
- **§5 Implementation Guide** — Env setup, project structure, router prompt, LangGraph state  
- **§6 Project Plan & Timeline** — Phases, sprints, team  
- **§7 Go-to-Market** — Segments, pricing, differentiation, projections  
- **§8 Testing & QA** — Strategy, evaluation metrics  
- **§9 Deployment & Operations** — Architecture, monitoring  
- **§10 Risk Management** — Risks and mitigations  
- **§11 Appendix** — Sample queries, config parameters, API reference  

---

## 2. High-Level Application Structure

```
omnihelp/
├── .github/                    # CI/CD, issue/PR templates
├── config/                     # Environment and feature config
├── docs/                       # Architecture, runbooks, API docs
├── scripts/                    # One-off and automation scripts
├── src/                        # Main application (see §3)
├── tests/                      # Unit, integration, E2E (see §4)
├── deploy/                     # Docker, K8s, Terraform (see §5)
├── frontend/                   # Chat UI, admin dashboard (optional repo or subdir)
├── pyproject.toml              # Python deps, tooling (or requirements.txt)
├── Dockerfile
├── docker-compose.yml
├── README.md
├── APPLICATION_STRUCTURE_PLAN.md
└── omnihelp-project-blueprint.docx
```

---

## 3. Source Layout (`src/`)

Aligned with **§4 Agentic Architecture** (Router + Retriever + SQL + Web Search + Synthesis) and **§5 Implementation Guide**.

### 3.1 Package Layout

```
src/
├── __init__.py
├── main.py                     # FastAPI app entry, lifespan
├── api/                        # HTTP layer (§2.1.5, §11.3)
│   ├── __init__.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── chat.py             # POST /api/v1/chat
│   │   ├── conversations.py   # GET /api/v1/conversations/{id}, feedback
│   │   ├── documents.py       # POST/DELETE documents (ingestion)
│   │   ├── analytics.py       # routing, satisfaction
│   │   ├── health.py          # /api/v1/health
│   │   └── admin.py           # GET/PUT config
│   ├── dependencies.py        # Auth, DB sessions, config inject
│   └── schemas/               # Pydantic request/response models
│       ├── chat.py
│       ├── conversation.py
│       └── analytics.py
├── core/                       # Shared config, logging, errors
│   ├── __init__.py
│   ├── config.py              # Settings from env (§11.2)
│   ├── logging.py
│   └── exceptions.py
├── agents/                     # Router + orchestration (§2.1.1, §4)
│   ├── __init__.py
│   ├── router.py              # Intent classification, confidence, routing
│   ├── prompts/
│   │   ├── router_classification.py   # §5.3.1
│   │   └── synthesis.py
│   └── state.py               # LangGraph state schema (§5.3.2)
├── nodes/                      # Execution nodes (§4.1.1)
│   ├── __init__.py
│   ├── retriever.py           # Policy RAG (§2.1.2): vector search, rerank, citations
│   ├── sql.py                 # Order management (§2.1.3): NL2SQL, validation
│   ├── web_search.py         # Fallback (§2.1.4): live search, synthesis
│   └── synthesis.py           # Response generation, tone, citations
├── graph/                      # LangGraph workflow (§4.1.2)
│   ├── __init__.py
│   ├── builder.py             # Graph construction, conditional edges
│   └── nodes.py               # Thin wrappers calling agents/nodes
├── retrieval/                  # Vector + hybrid search (§2.1.2)
│   ├── __init__.py
│   ├── embeddings.py
│   ├── store.py               # Qdrant (or chosen vector DB)
│   ├── chunking.py            # Doc ingestion, chunk sizes
│   └── reranker.py            # Optional cross-encoder
├── sql/                        # NL2SQL and execution (§2.1.3)
│   ├── __init__.py
│   ├── schema.py              # Introspection, table/column metadata
│   ├── nl2sql.py              # Text → SQL generation + validation
│   └── executor.py            # Read-only execution, timeouts
├── search/                     # Web search (§2.1.4)
│   ├── __init__.py
│   ├── client.py              # External search API, rate limiting
│   └── synthesizer.py         # Multi-source synthesis, attribution
├── memory/                     # Conversation context (§2.1.1 FR-004)
│   ├── __init__.py
│   ├── store.py               # Redis or DB-backed
│   └── formatter.py           # Last N turns for router/LLM
├── ingestion/                  # Document ingestion (§2.1.2 FR-007)
│   ├── __init__.py
│   ├── loaders.py             # PDF, DOCX, TXT, HTML
│   └── pipeline.py           # Chunk → embed → store
└── services/                   # Optional high-level services
    ├── __init__.py
    └── chat_service.py        # End-to-end: route → nodes → response
```

### 3.2 Mapping to Blueprint

| Blueprint Section | Implementation Location |
|-------------------|--------------------------|
| §2.1.1 Router Agent | `agents/router.py`, `agents/prompts/router_classification.py`, `graph/builder.py` |
| §2.1.2 Retriever Node | `nodes/retriever.py`, `retrieval/`, `ingestion/` |
| §2.1.3 SQL Node | `nodes/sql.py`, `sql/` |
| §2.1.4 Web Search Node | `nodes/web_search.py`, `search/` |
| §2.1.5 Response Synthesis & UX | `nodes/synthesis.py`, `api/routes/`, `api/schemas/` |
| §4.1.2 Routing Logic (State Machine) | `graph/builder.py`, `agents/state.py` |
| §5.3.1 Router Classification Prompt | `agents/prompts/router_classification.py` |
| §5.3.2 LangGraph State Schema | `agents/state.py` |
| §11.2 Key Configuration Parameters | `core/config.py` |
| §11.3 API Endpoint Reference | `api/routes/*` |

---

## 4. Configuration (`config/`)

- **§11.2** parameters (e.g. `ROUTER_CONFIDENCE_THRESHOLD`, `RETRIEVER_TOP_K`, `SQL_QUERY_TIMEOUT`, `CACHE_TTL_SECONDS`, `ESCALATION_KEYWORDS`) should be loadable from env and/or `config/` files.
- Prefer a single source of truth in `core/config.py` with overrides per environment (dev/staging/prod).

```
config/
├── default.yaml or .env.example
├── dev.yaml
└── production.yaml
```

---

## 5. Tests (`tests/`)

Per **§8 Testing & Quality Assurance**:

```
tests/
├── conftest.py                # Pytest fixtures, testcontainers
├── unit/
│   ├── agents/
│   ├── nodes/
│   ├── retrieval/
│   ├── sql/
│   └── api/
├── integration/
│   ├── test_graph_e2e.py
│   ├── test_api_chat.py
│   └── test_ingestion.py
├── evaluation/
│   ├── routing_accuracy.py    # §8.2 golden dataset
│   └── response_quality.py   # LLM-as-judge
└── load/
    └── locustfile.py or k6 scripts
```

---

## 6. Deployment (`deploy/`)

Per **§9 Deployment & Operations**:

```
deploy/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml     # API, Qdrant, PostgreSQL, Redis
├── kubernetes/                # Optional
│   ├── api.yaml
│   ├── qdrant.yaml
│   └── ingress.yaml
└── terraform/ or cloud/       # Optional for cloud resources
```

---

## 7. Key Conventions

- **Router-first:** All user messages go through the Router for intent classification before any retrieval or SQL (§1.3, §4).
- **Read-only SQL:** Only `SELECT`; block `DROP/DELETE/UPDATE` (§2.1.3 FR-015).
- **Fallback order:** Internal sources first; web search only when needed (§2.1.4 FR-023).
- **Observability:** Correlation IDs, structured logs, and metrics for routing, latency, and satisfaction (§9.2, §2.2).

---

## 8. Next Steps

1. Initialize repo with the structure above (create dirs and stubs).  
2. Implement `core/config.py` and load §11.2 parameters.  
3. Implement `agents/state.py` and router classification prompt (§5.3).  
4. Build LangGraph in `graph/builder.py` with conditional edges to retriever/sql/web_search/synthesis.  
5. Implement nodes (retriever, sql, web_search, synthesis) and wire to graph.  
6. Expose FastAPI routes per §11.3 and connect to graph.  
7. Add ingestion pipeline and document API.  
8. Add tests (§8) and deploy per §9.

---

*This application structure plan is derived from the Omni-Help Project Blueprint (Version 1.0) and is intended to guide implementation and repository layout.*
