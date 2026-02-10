# Omni-Help Implementation Plan

**Document Version:** 2.0  
**Strategy:** **State-First** — State is the source of truth; evaluation and scaffolding precede pipeline build-out.  
**Role:** Senior AI Architect & Lead Python Developer  
**System:** Enterprise-grade Adaptive RAG Router — "Omni-Help"

---

## 0. State-First Strategy

Execution order and dependencies are driven by **state first**:

1. **Define and validate state** (and evaluation harness) before building pipelines.
2. **Evaluate the Router (Brain)** against a Golden Dataset from Day 1; route quality gates all downstream work.
3. **Orchestration is cyclic** — the graph supports self-correction loops (e.g. Clarification → back to Router), not a one-shot linear chain.

Dependency order: **Phase 0 (Evaluation & Scaffolding)** → **Phase 1 (The Brain)** → **Phase 2 (State & Nodes)** → **Phase 3 (Orchestration — Cyclic Graph)** → Phases 4+ (Pipelines, API, etc.).

---

## 1. Executive Summary & Context (from Blueprint)

### 1.1 Problem Statement

- **Single-Pipeline Failure:** Traditional RAG uses one retrieval pipeline and fails when queries span multiple data sources.
- **Context Mismatch:** e.g. user asks "Where is my order #123?" but the system searches PDFs instead of the order database.
- **Inefficiency:** Support teams handle ~60% of queries that could be automated with proper routing.
- **Latency:** Average wait times 15+ minutes due to misrouted queries and escalations.

### 1.2 Solution Overview

Omni-Help uses a central **Router Agent** as an intelligent dispatcher: it **classifies query intent first**, then routes to specialized sub-agents (Policy/Retriever, Order/SQL, Web Search). Target: **95%+ routing accuracy**, **70% reduction in resolution time**.

### 1.3 Key Innovation: Classification-First Paradigm

Before retrieving anything, the system answers: *"What type of information does this query need?"* — transforming RAG from blind retrieval into an intelligent reasoning system.

---

## 2. Critical Architectural Constraints

| Constraint | Requirement |
|------------|-------------|
| **Orchestration** | LangGraph only — stateful, multi-actor agents; **Cyclic Graph** for self-correction. |
| **Router** | Dedicated Classifier Node (GPT-4o-mini); **Structural Output (JSON mode)**; **Confidence Gate** → Clarification Node when confidence < 0.7. |
| **Backend** | FastAPI, Pydantic V2, async throughout. |
| **SQL** | SQLite (dev) / PostgreSQL (prod) via SQLAlchemy; aiosqlite for async. |
| **Vector Store** | ChromaDB (dev) / Qdrant (prod). |
| **Search** | Tavily API. |
| **Observability** | LangSmith for tracing and evaluation. |
| **Config** | **pydantic-settings** (strict) for enterprise config. |
| **State** | **State-First:** `AgentState` TypedDict with `intent`, `confidence`, `missing_info`, `routing_rationale`; conversation history via `Annotated` + `add_messages`. |

---

## 3. State Schema (Source of Truth)

State is defined first and drives all nodes. Implement in `src/graph/state.py`.

### 3.1 Intent & Route Types

- **Intents:** Policy_Question, Order_Status, General_Chat, Product_Info, Complaint.
- **Graph routes:** `policy` (Retriever), `sql` (SQL Node), `web` (Web Node), `fallback` (Fallback/Clarification).

### 3.2 Required State Keys (State-First)

- **intent** — `Literal["policy", "sql", "web", "product_info", "complaint"]` (or equivalent).
- **confidence** — `float` in [0, 1]; used by **Confidence Gate**: if < 0.7 → route to **Clarification Node**.
- **missing_info** — `List[str]`; used by Clarification / self-correction to request missing details.
- **routing_rationale** — `str`; explanation for every routing decision (FR-005).
- **messages** — Conversation history via `typing.Annotated` with LangGraph reducer **add_messages** (append semantics).

All other keys (e.g. `user_query`, `route`, `retrieved_docs`, `policy_context`, `sql_result`, `web_context`, `fallback_reason`, `handoff_context`, `final_response`, `error`) remain part of `AgentState` as in the full schema; the above are **crucial** for the State-First and Brain phases.

---

## 4. Development Phases (State-First Order)

### Phase 0: Evaluation & Scaffolding

*Execution order: first. Establishes LangSmith, Golden Dataset, and project structure.*

- [ ] **0.1** Set up **LangSmith**: create project, set `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY` in `.env.example`; document how to run traces and evaluations.
- [ ] **0.2** Create **Golden Dataset**: `tests/evaluation/golden_dataset.json` with **10 example queries** (2 per intent: policy, sql, web, product_info, complaint) as Day 1 baseline for router evaluation.
- [ ] **0.3** Create **evaluation harness**: `tests/evaluation/eval_router.py` — load golden dataset, run router (or stub), compute accuracy / log to LangSmith; runnable via `pytest` or CLI.
- [ ] **0.4** **Project structure** (scaffolding):
  ```text
  omni-help/
  ├── src/
  │   ├── agents/ (__init__.py, router.py)
  │   ├── graph/ (__init__.py, state.py, nodes.py)
  │   ├── tools/ (__init__.py)
  │   ├── utils/ (__init__.py)
  │   ├── config/ (__init__.py, settings.py)
  ├── tests/
  │   ├── evaluation/ (__init__.py, golden_dataset.json, eval_router.py)
  ├── .env.example
  ├── pyproject.toml
  ```
- [ ] **0.5** Add **pyproject.toml** with complete, final dependency list (Core, Config, API, Data, Testing/Ops); no separate requirements.txt for core deps.
- [ ] **0.6** Add **.env.example** with placeholders for OpenAI, Tavily, LangSmith, DB URLs, vector store.
- [ ] **0.7** Implement **src/config/settings.py** using **pydantic-settings** (strict); load from env; no secrets in code.

---

### Phase 1: The Brain (Router Agent)

*Router is the Brain; it must output structured data and enforce the Confidence Gate.*

- [ ] **1.1** Implement **Structural Output (JSON mode)** for the Router: use OpenAI (or designated model) with response format enforcing a single JSON object with fields: `intent`, `confidence`, `routing_rationale`, and optionally `missing_info` (list of strings). Parse and validate with Pydantic; write into `AgentState`.
- [ ] **1.2** Implement **Confidence Gate**: if `confidence < 0.7`, route to **Clarification Node** (not directly to Fallback). Clarification Node may set `missing_info` and/or return a clarifying question, then re-enter the graph (e.g. back to Router or to a dedicated “after-clarification” step). Document the exact edge: Router → Clarification when confidence < 0.7.
- [ ] **1.3** Implement `src/agents/router.py`: load prompt from `src/prompts/` (or config); call GPT-4o-mini with JSON mode; return intent, confidence, routing_rationale, missing_info; strict type hints and docstrings.
- [ ] **1.4** Wire Router into evaluation harness: run Golden Dataset through Router; log predictions and compare to golden labels; report accuracy and log to LangSmith.
- [ ] **1.5** Unit tests for Router (mock LLM); assert JSON parsing, Confidence Gate behaviour (e.g. confidence 0.6 → route to Clarification).

---

### Phase 2: State & Nodes (Skeleton)

- [ ] **2.1** Implement **src/graph/state.py** as **source of truth**: `AgentState` TypedDict with `intent`, `confidence`, `missing_info`, `routing_rationale`; `messages` using `typing.Annotated` and **add_messages** reducer; all other keys as needed for pipelines.
- [ ] **2.2** Implement skeleton nodes in `src/graph/nodes.py`: `router_node`, `clarification_node`, `retriever_node`, `sql_node`, `web_node`, `fallback_node`, `synthesis_node`; each returns `Partial[AgentState]`.
- [ ] **2.3** Implement **src/config/settings.py** (if not done in Phase 0) and ensure config is used by graph/agents.

---

### Phase 3: Orchestration (Cyclic Graph)

*Orchestration is a **Cyclic Graph**, not a linear chain. Self-correction loops are required.*

- [ ] **3.1** Implement **Cyclic Graph** in `src/graph/` (e.g. `graph.py`): build LangGraph `StateGraph(AgentState)` with cycles, e.g.:
  - **Router** → (by intent/confidence) → Retriever | SQL | Web | Clarification | Fallback.
  - **Clarification** → back to **Router** (or to a re-entry node) after collecting missing info, so the graph can re-route with updated state.
  - **SQL node** (or other nodes) may retry or re-route on failure (self-correction loop).
- [ ] **3.2** Define conditional edges from Router: confidence < 0.7 → Clarification; else by intent → Retriever / SQL / Web / Fallback. Define edges from Clarification back into the graph (e.g. to Router).
- [ ] **3.3** Document the cyclic topology in this plan (diagram or list of cycles). Ensure no unbounded loops (e.g. max clarification turns or max retries).
- [ ] **3.4** Compile graph and add integration test that runs a query through Router → Clarification → Router (low confidence then re-route).

---

### Phase 4: RAG (Policy) Pipeline (FR-007 to FR-012)

- [ ] **4.1** Implement `src/utils/ingest.py`: PDF, DOCX, TXT, HTML; chunk 512–1024 tokens; embed (text-embedding-3-small); persist to ChromaDB / Qdrant.
- [ ] **4.2** Vector store abstraction: ChromaDB (dev) / Qdrant (prod); semantic search; optional hybrid BM25, metadata filtering, citations, reranking.
- [ ] **4.3** Implement `retriever_node`: query vector store; format `policy_context` with citations; write to state.
- [ ] **4.4** Policy synthesis: `policy_context` + `user_query` → LLM → `final_response`.
- [ ] **4.5** Unit tests for ingest and retriever_node; optional LangSmith tracing.

---

### Phase 5: SQL (Order) Pipeline (FR-013 to FR-018)

- [ ] **5.1** SQL schema: `orders`, `order_items`, `customers`, `shipments`, `returns`, `products`; SQLite (dev) / PostgreSQL (prod).
- [ ] **5.2** Async SQLAlchemy + `src/tools/sql.py`: read-only enforcement; parameterized execution; return rows as list of dicts.
- [ ] **5.3** Implement sql_node (or sql_agent): NL→SQL, schema introspection, execute, set `sql_result` or `sql_error`.
- [ ] **5.4** Synthesis: LLM → natural language from `sql_result`; unit tests.

---

### Phase 6: Web Search Pipeline (FR-019 to FR-024)

- [ ] **6.1** `src/tools/web_search.py`: Tavily API (async); rate limit; domain allowlist/blocklist; normalized results.
- [ ] **6.2** Implement web_node: combine 3–5 sources into `web_context`; synthesis → `final_response`.
- [ ] **6.3** Unit tests; ensure web < 15% of total queries where possible.

---

### Phase 7: API, Fallback & Integration

- [ ] **7.1** FastAPI app: health check, CORS, lifespan (graph load, DB pool).
- [ ] **7.2** Pydantic V2 request/response models; `POST /query`; multi-turn support; correlation IDs.
- [ ] **7.3** Fallback node: build `handoff_context`; human escalation.
- [ ] **7.4** LangSmith request IDs; structured logging.

---

### Phase 8: Observability, Security & Documentation

- [ ] **8.1** LangSmith integration verified; no secrets in code; input/output validation.
- [ ] **8.2** pytest for all nodes; integration test for cyclic flow (Router → Clarification → Router).
- [ ] **8.3** Google-style docstrings; README with setup, env, run ingest/API/tests/eval.

---

## 5. Router: Structural Output & Confidence Gate (Summary)

- **Structural Output:** Router MUST emit a single JSON object (e.g. via OpenAI JSON mode) with at least: `intent`, `confidence`, `routing_rationale`, and optionally `missing_info`. This is parsed and written into `AgentState`.
- **Confidence Gate:** If `confidence < 0.7`, the graph routes to the **Clarification Node**, not directly to Fallback. Clarification can set `missing_info` and/or ask the user a question; control then returns (e.g. to Router) so the graph can re-route with updated context — forming a **cycle**.

---

## 6. Orchestration: Cyclic Graph (Summary)

- The graph is **cyclic**: at least one cycle is **Router → Clarification → Router** (or re-entry to Router).
- Other cycles (e.g. SQL retry, Retriever retry) may be added for self-correction.
- Bounds must be defined (e.g. max clarification turns, max retries) to avoid infinite loops.

---

## 7. Functional Requirements Traceability (Abbreviated)

- **FR-001–FR-006:** Router (intent, multi-label, confidence < 0.7 → human/clarification, context, rationale, feedback).
- **FR-007–FR-012:** Retriever (ingest, semantic/hybrid, metadata, citations, reranking).
- **FR-013–FR-018:** SQL (NL→SQL, schema, read-only, format, errors, templates).
- **FR-019–FR-024:** Web (latency, allowlist, synthesis, freshness, fallback %, rate limit).
- **FR-025–FR-030:** Response, tone, multi-channel, handoff, feedback, analytics.
- **NFR:** p95 < 3s, router < 200ms, vector < 500ms, SQL < 1s; 1k concurrent; 99.9% uptime; encryption; 100% logging with correlation IDs.

---

## 8. File-to-Phase Quick Reference

| Path | Phase |
|------|--------|
| `pyproject.toml`, `.env.example`, `src/config/settings.py` | 0 |
| `tests/evaluation/golden_dataset.json`, `eval_router.py` | 0 |
| `src/agents/router.py`, `src/prompts/` | 1 |
| `src/graph/state.py`, `nodes.py` | 2 |
| `src/graph/graph.py` (cyclic) | 3 |
| `src/utils/ingest.py`, `src/tools/` (vector, sql, web) | 4, 5, 6 |
| `api/` (FastAPI) | 7 |
| `tests/` (unit + integration) | 0–8 |

---

## 9. Next Step

**State-First Strategy is adopted.** Execution order: **Phase 0 (Evaluation & Scaffolding)** → **Phase 1 (The Brain: Structural Output + Confidence Gate)** → **Phase 2 (State & Nodes)** → **Phase 3 (Cyclic Orchestration)** → Phases 4–8.

Proceed with Phase 0 scaffolding (LangSmith, Golden Dataset, project structure, pyproject.toml, state.py as source of truth, and evaluation harness).
