# Omni-Help Implementation Plan

**Document Version:** 1.0  
**Role:** Senior AI Architect & Lead Python Developer  
**System:** Enterprise-grade Adaptive RAG Router — "Omni-Help"

---

## 1. Critical Architectural Constraints (from Blueprint)

The following are **non-negotiable** and must be strictly adhered to:

| Constraint | Requirement |
|------------|-------------|
| **Orchestration** | LangGraph only — stateful, multi-actor agents. No ad-hoc orchestration. |
| **Router** | Dedicated Classifier Node using **GPT-4o-mini** (or designated model) to route between **Policy**, **SQL**, and **Web**. |
| **Backend** | **FastAPI** — async throughout, **Pydantic V2** for all request/response and internal DTOs. |
| **SQL Database** | **SQLite** (local/dev), **PostgreSQL** (prod) — abstracted via **SQLAlchemy** (async). |
| **Vector Store** | **ChromaDB** (local/dev), **Qdrant** (prod). |
| **Search** | **Tavily API** for web search. |
| **Observability** | **LangSmith** for tracing and evaluation. |
| **Code style** | Modular layout: separate `agents`, `nodes`, `tools`, `schema`, `core`. No monolithic files. |
| **Types** | Strict type hints (`typing.List`, `typing.Optional`, etc.) and Pydantic models for all data exchange. |
| **I/O** | Async-first: `async/await` for all DB and API calls. |
| **Secrets** | `python-dotenv`; **no hardcoded API keys**. |
| **Testing** | **pytest** mandatory; every node must be unit-testable in isolation. |

---

## 2. StateGraph Schema (Blueprint Section 5.3.2)

LangGraph state is defined as a **TypedDict** shared by all nodes. Below is the canonical schema for the Omni-Help router and three pipelines.

### 2.1 `AgentState` Definition

```python
# Conceptual schema — to be implemented in src/graph/state.py

from typing import TypedDict, Optional, List, Literal
from langchain_core.messages import BaseMessage

RouteType = Literal["policy", "sql", "web"]

class AgentState(TypedDict, total=False):
    """Shared state for the Omni-Help LangGraph. All nodes read/write this."""

    # --- Input & routing ---
    messages: List[BaseMessage]          # Conversation history (reducer: append)
    user_query: str                      # Current user question (from last message or API)
    route: RouteType                     # Router output: "policy" | "sql" | "web"

    # --- Policy (RAG) pipeline ---
    retrieved_docs: List[Any]             # Chunks from vector store (ChromaDB/Qdrant)
    policy_context: Optional[str]        # Formatted context for policy LLM

    # --- SQL (Order) pipeline ---
    sql_query: Optional[str]             # Generated/adjusted SQL
    sql_result: Optional[List[Dict]]     # Result rows from DB
    sql_error: Optional[str]             # If query failed

    # --- Web pipeline ---
    web_results: Optional[List[Dict]]    # Tavily search results
    web_context: Optional[str]           # Formatted web snippets for LLM

    # --- Synthesis ---
    final_response: Optional[str]         # Answer returned to user
    error: Optional[str]                 # Top-level error message if any
```

### 2.2 State Flow Summary

- **Router node** sets: `route` (and may set `user_query` from `messages`).
- **Retriever node** (Policy): sets `retrieved_docs`, `policy_context`.
- **SQL node**: sets `sql_query`, `sql_result`, or `sql_error`.
- **Web node**: sets `web_results`, `web_context`.
- **Synthesis/response node** (optional): sets `final_response` from pipeline-specific context.

---

## 3. Data Flow: Three Pipelines

### 3.1 Policy (RAG) Pipeline

```
User Query → Router → [route == "policy"] → Retriever Node → Vector DB (ChromaDB/Qdrant)
                → policy_context → LLM (with context) → final_response
```

- **Input:** `messages` / `user_query`
- **Router:** classifies as `policy` (policy docs, procedures, internal knowledge).
- **Retriever node:** queries vector store, formats chunks into `policy_context`.
- **LLM:** generates answer from `policy_context` + query; writes `final_response`.

### 3.2 Order (SQL) Pipeline

```
User Query → Router → [route == "sql"] → SQL Node → SQLAlchemy (SQLite/PostgreSQL)
                → sql_result → LLM (with result) → final_response
```

- **Input:** `messages` / `user_query`
- **Router:** classifies as `sql` (orders, data lookup, transactional queries).
- **SQL node:** generates/validates SQL, executes via SQLAlchemy, sets `sql_result` or `sql_error`.
- **LLM:** formats natural-language answer from `sql_result`; writes `final_response`.

### 3.3 Web Pipeline

```
User Query → Router → [route == "web"] → Web Node → Tavily API
                → web_results → web_context → LLM → final_response
```

- **Input:** `messages` / `user_query`
- **Router:** classifies as `web` (external, real-time, or public info).
- **Web node:** calls Tavily, formats results into `web_context`.
- **LLM:** synthesizes answer from `web_context`; writes `final_response`.

### 3.4 High-Level Graph Topology

```
                    ┌─────────────────┐
                    │   ENTRY (input) │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   ROUTER NODE   │  ← Classifier (GPT-4o-mini)
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │  RETRIEVER  │   │   SQL NODE  │   │  WEB NODE   │
    │    NODE     │   │             │   │             │
    └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
           │                 │                 │
           └─────────────────┼─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  SYNTHESIS /    │  (optional dedicated node or inline)
                    │  RESPONSE       │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   END (output)  │
                    └─────────────────┘
```

Conditional edges from the router direct to exactly one of Retriever, SQL, or Web based on `state["route"]`.

---

## 4. Development Phases → Granular Engineering Tasks (Blueprint Section 6)

Tasks are check-boxable. Order respects dependencies (e.g. state and router before pipelines).

### Phase 1: Foundation & Project Scaffolding

- [ ] **1.1** Create repo structure per Blueprint Section 5.2:  
  `src/agents`, `src/graph`, `src/tools`, `src/api`, `src/utils`, `src/schema`, `src/core`, `tests`, `config`
- [ ] **1.2** Add `requirements.txt` or `pyproject.toml` with deps from Blueprint 5.1.2 (langgraph, langchain, openai, fastapi, sqlalchemy, chromadb, qdrant-client, tavily-python, pydantic>=2, python-dotenv, pytest, langsmith, etc.)
- [ ] **1.3** Create `.env.example` with placeholders: `OPENAI_API_KEY`, `TAVILY_API_KEY`, `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, DB URLs for SQLite/PostgreSQL, vector store config (ChromaDB path / Qdrant URL)
- [ ] **1.4** Add `config/settings.py` (or equivalent) loading env via `python-dotenv`, Pydantic settings, no secrets in code
- [ ] **1.5** Add minimal `src/core/` modules (e.g. logging, env validation) and ensure `src` is a package

### Phase 2: Core State & Router

- [ ] **2.1** Implement `src/graph/state.py`: define `AgentState` TypedDict and `RouteType` as above; add reducer for `messages` if using append semantics
- [ ] **2.2** Implement `src/agents/router.py`: load router prompt template from Blueprint Section 5.3.1; call GPT-4o-mini; return one of `"policy" | "sql" | "web"`; strict type hints and docstrings
- [ ] **2.3** Implement `src/graph/nodes.py`: skeleton nodes with correct signatures `(state: AgentState) -> Partial[AgentState]`:  
  - [ ] **2.3a** `router_node` — calls router agent, sets `state["route"]` (and optionally `user_query`)  
  - [ ] **2.3b** `retriever_node` — stub: set empty `retrieved_docs`/`policy_context`  
  - [ ] **2.3c** `sql_node` — stub: set placeholder `sql_result` or `sql_error`  
  - [ ] **2.3d** `web_node` — stub: set placeholder `web_results`/`web_context`
- [ ] **2.4** Implement `src/graph/graph.py`: build `StateGraph(AgentState)`, add nodes, add conditional edges from router to retriever/sql/web, add edges to synthesis/end; compile graph
- [ ] **2.5** Unit tests for `router_node` (mock LLM) and for state transitions (stub nodes)

### Phase 3: RAG (Policy) Pipeline — First Vertical

- [ ] **3.1** Implement `src/utils/ingest.py`: script to load PDFs/Docs (e.g. PyMuPDF, docx) → chunk (e.g. RecursiveCharacterTextSplitter) → embed (OpenAI or configured embedder) → persist to ChromaDB (and optionally Qdrant for prod path)
- [ ] **3.2** Implement vector store abstraction in `src/tools/` or `src/core/`: ChromaDB (local) / Qdrant (prod) behind a single interface (e.g. `search(query, k=5) -> List[Document]`)
- [ ] **3.3** Implement `retriever_node` in `src/graph/nodes.py`: query vector store with `state["user_query"]`, format chunks into `policy_context`, write to state
- [ ] **3.4** Add policy synthesis: either inside `retriever_node` or a separate node that takes `policy_context` + `user_query` and calls LLM to produce `final_response`
- [ ] **3.5** Add unit tests for ingest script (small sample doc) and for `retriever_node` (mocked vector store)
- [ ] **3.6** Optional: add LangSmith tracing for retriever and policy LLM

### Phase 4: SQL (Order) Pipeline

- [ ] **4.1** Define SQL schema / sample DB (SQLite for dev): tables relevant to “orders” (e.g. orders, customers, products) and create migrations or init script
- [ ] **4.2** Implement `src/core/database.py` (or equivalent): async SQLAlchemy engine/session for SQLite (dev) and PostgreSQL (prod) from config
- [ ] **4.3** Implement `src/tools/sql.py`: safe query execution (parameterized), read-only or controlled write; return rows as list of dicts; set error in state on failure
- [ ] **4.4** Implement `sql_node`: generate SQL from `user_query` (e.g. via LLM with schema context), execute via `src/tools/sql.py`, set `sql_result` or `sql_error`
- [ ] **4.5** Add synthesis step: LLM turns `sql_result` into natural language `final_response`
- [ ] **4.6** Unit tests for SQL execution (mock DB or in-memory SQLite) and for `sql_node` with mocked LLM/tool

### Phase 5: Web Search Pipeline

- [ ] **5.1** Implement `src/tools/web_search.py`: Tavily API client (async), rate limiting and error handling; return normalized list of results (title, snippet, URL)
- [ ] **5.2** Implement `web_node`: call Tavily with `user_query`, format results into `web_context`, write to state
- [ ] **5.3** Add synthesis: LLM generates `final_response` from `web_context` + query
- [ ] **5.4** Unit tests for Tavily client (mocked HTTP) and for `web_node`

### Phase 6: API Layer & Integration

- [ ] **6.1** Implement FastAPI app in `src/api/`: health check, CORS, lifespan (e.g. graph load, DB pool)
- [ ] **6.2** Add Pydantic V2 request/response models for query endpoint (e.g. `QueryRequest`, `QueryResponse` with `query`, `response`, `route`, `sources`)
- [ ] **6.3** Expose single query endpoint (e.g. `POST /query`) that invokes compiled LangGraph with `user_query`, returns `final_response` and optional metadata (route, citations)
- [ ] **6.4** Use async throughout (e.g. `astream` or `ainvoke` on graph); no blocking calls in request path
- [ ] **6.5** Add structured logging and optional LangSmith request IDs for tracing

### Phase 7: Observability, Security & Hardening

- [ ] **7.1** Integrate LangSmith: set `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`; verify traces for router and each pipeline
- [ ] **7.2** Ensure no API keys in code; all via env and `config/settings.py`
- [ ] **7.3** Add input validation and output sanitization (Pydantic, max lengths, safe error messages)
- [ ] **7.4** Document deployment: SQLite vs PostgreSQL, ChromaDB vs Qdrant, env vars

### Phase 8: Testing & Documentation

- [ ] **8.1** pytest: every node testable in isolation with mocks (LLM, DB, Tavily, vector store)
- [ ] **8.2** Integration test: run graph end-to-end with test DB and mock/external APIs as needed
- [ ] **8.3** Google-style docstrings for all public functions and classes
- [ ] **8.4** README: setup, env example, how to run ingest, run API, run tests

---

## 5. Router Prompt Template (Blueprint Section 5.3.1) — Placeholder

The exact router prompt text must be taken from the blueprint document. Use it in `src/agents/router.py`. Conceptual shape:

- **Input:** User message (or `user_query`)
- **Output:** One of `policy`, `sql`, `web`
- **Instructions:** When to choose Policy (internal docs, procedures), SQL (orders, structured data), Web (external/real-time info). Include few-shot examples if specified in the blueprint.

Once the blueprint text is available (e.g. copy-pasted into the repo or a `config/prompts/` file), replace this placeholder with the real template and reference it in code.

---

## 6. File-to-Phase Quick Reference

| Path | Phase |
|------|--------|
| `config/`, `.env.example`, `requirements.txt` or `pyproject.toml` | 1 |
| `src/graph/state.py` | 2 |
| `src/agents/router.py` | 2 |
| `src/graph/nodes.py` | 2, 3, 4, 5 |
| `src/graph/graph.py` | 2 |
| `src/utils/ingest.py` | 3 |
| `src/tools/` (vector, sql, web_search) | 3, 4, 5 |
| `src/api/` | 6 |
| `tests/` | 2–8 |

---

## 7. Next Step

**Step 1 is complete.** This plan is the single source of truth for:

- StateGraph schema (`AgentState`, `RouteType`)
- Data flow for Policy, Order (SQL), and Web
- Phased, check-boxable engineering tasks

Proceed to **Step 2: Project Scaffolding** when ready (folders, dependency file, `.env.example`).
