# Omni-Help Implementation Plan

**Document Version:** 1.1  
**Role:** Senior AI Architect & Lead Python Developer  
**System:** Enterprise-grade Adaptive RAG Router — "Omni-Help"  
**Blueprint:** Complete Project Blueprint v1.0 (missing_context + original)

---

## 0. Executive Summary & Context (from Blueprint)

### 0.1 Problem Statement

- **Single-Pipeline Failure:** Traditional RAG uses one retrieval pipeline and fails when queries span multiple data sources.
- **Context Mismatch:** e.g. user asks "Where is my order #123?" but the system searches PDFs instead of the order database.
- **Inefficiency:** Support teams handle ~60% of queries that could be automated with proper routing.
- **Latency:** Average wait times 15+ minutes due to misrouted queries and escalations.

### 0.2 Solution Overview

Omni-Help uses a central **Router Agent** as an intelligent dispatcher: it **classifies query intent first**, then routes to specialized sub-agents (Policy/Retriever, Order/SQL, Web Search). Target: **95%+ routing accuracy**, **70% reduction in resolution time**.

### 0.3 Key Innovation: Classification-First Paradigm

Before retrieving anything, the system answers: *"What type of information does this query need?"* — transforming RAG from blind retrieval into an intelligent reasoning system.

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
| **Embedding** | **text-embedding-3-small** (1536 dims) for vector store. |
| **LLM (Local option)** | **Llama 3.1 8B (Ollama)** for hybrid/offline deployment. |
| **Frontend** | **Next.js** (prod) / **Streamlit** (POC). |

---

## 1.1 Functional Requirements Traceability (Blueprint Section 2.1)

Implementation must satisfy the following; phases reference these IDs.

### Router Agent (FR-001 to FR-006)

| ID | Requirement | Target |
|----|-------------|--------|
| FR-001 | Intent classification into Policy_Question, Order_Status, General_Chat, Product_Info, Complaint | >95% accuracy on 1k-query test set |
| FR-002 | Multi-label: compound queries (e.g. return policy + order #123) | 90% correct identification |
| FR-003 | Confidence threshold: route to human when &lt;70% | Configurable; enforce in router |
| FR-004 | Context preservation across turns | 5+ turn conversations |
| FR-005 | Routing explanation | Log rationale for 100% of decisions |
| FR-006 | Dynamic route learning from human corrections | Feedback loop; +5% accuracy monthly (medium-term) |

### Retriever Node — Policy RAG (FR-007 to FR-012)

| ID | Requirement | Target |
|----|-------------|--------|
| FR-007 | Document ingestion: PDF, DOCX, TXT, HTML; auto chunking | Docs up to 500 pages; chunk 512–1024 tokens |
| FR-008 | Semantic search (dense embeddings) | Top-k; cosine similarity &gt;0.75 |
| FR-009 | Hybrid search: semantic + BM25 | +15% recall over pure semantic |
| FR-010 | Metadata filtering (doc type, date, category) | 10+ fields; AND/OR logic |
| FR-011 | Citation generation | Source + page for 100% of retrieved info |
| FR-012 | Reranking (cross-encoder) | +20% precision@5 |

### SQL Node — Order Management (FR-013 to FR-018)

| ID | Requirement | Target |
|----|-------------|--------|
| FR-013 | NL to SQL | Correct SQL for 90% of order queries |
| FR-014 | Schema introspection | Dynamic tables/columns, no hardcoding |
| FR-015 | Read-only enforcement | Block 100% of DROP/DELETE/UPDATE; SELECT only |
| FR-016 | Result formatting | Tabular → natural language |
| FR-017 | Error handling | Actionable messages for query failures |
| FR-018 | Query templates | 20+ parameterized templates for common ops |

### Web Search Node (FR-019 to FR-024)

| ID | Requirement | Target |
|----|-------------|--------|
| FR-019 | Live web search | Results within 3s for 95% of searches |
| FR-020 | Source filtering | Allowlist/blocklist domains (admin config) |
| FR-021 | Result synthesis | Combine 3–5 sources into one response |
| FR-022 | Freshness priority | Weight last 7 days 2× for news |
| FR-023 | Fallback cascade | Web &lt;15% of total queries |
| FR-024 | Rate limiting | Cap 1000 searches/hour |

### Response & UX (FR-025 to FR-030)

| ID | Requirement | Target |
|----|-------------|--------|
| FR-025 | Response generation | Conversational; satisfaction &gt;4.2/5 |
| FR-026 | Tone adaptation | By sentiment for 85% of emotional queries |
| FR-027 | Multi-channel | Web, Slack, Discord, Email (4+ channels) |
| FR-028 | Handoff to human | Full history + routing context |
| FR-029 | Feedback (thumbs up/down) | 30%+ response rate |
| FR-030 | Analytics dashboard | Real-time; 5 min updates; 7-day history |

### Non-Functional Requirements (Blueprint Section 2.2)

| Category | Requirement | Target |
|----------|-------------|--------|
| Performance | End-to-end latency (p95) | &lt;3 seconds |
| Performance | Router classification | &lt;200 ms |
| Performance | Vector search (p95) | &lt;500 ms |
| Performance | SQL execution | &lt;1 second |
| Scalability | Concurrent conversations | 1,000 simultaneous |
| Availability | Uptime SLA | 99.9% |
| Security | Encryption | AES-256 (rest), TLS 1.3 (transit) |
| Observability | Logging | 100% of requests with correlation IDs |

### Data Sources (Blueprint Section 3.2)

- **Policy (Vector DB):** Return Policy, Shipping Policy, FAQ, Terms of Service, Product Manuals.
- **Order DB (SQL):** `orders`, `order_items`, `customers`, `shipments`, `returns`, `products`.
- **Web:** Competitor info, industry news, regulatory updates.

---

## 2. StateGraph Schema (Blueprint Section 5.3.2)

LangGraph state is defined as a **TypedDict** shared by all nodes. Router intents (Blueprint §2.1.1) are **Policy_Question**, **Order_Status**, **General_Chat**, **Product_Info**, **Complaint**. Low confidence (&lt;0.7) or Complaint routes to **Fallback** (FR-003).

### 2.1 `AgentState` Definition

```python
# Conceptual schema — to be implemented in src/graph/state.py

from typing import TypedDict, Optional, List, Literal
from langchain_core.messages import BaseMessage

# Blueprint: Policy_Question, Order_Status, General_Chat, Product_Info, Complaint
# Mapped to graph nodes: policy → Retriever; sql → SQL; web → Web; fallback → Fallback
IntentType = Literal["policy", "sql", "web", "fallback"]
RouteType = IntentType  # alias for graph edges

class AgentState(TypedDict, total=False):
    """Shared state for the Omni-Help LangGraph. All nodes read/write this."""

    # --- Input & routing ---
    messages: List[BaseMessage]          # Conversation history (reducer: append)
    user_query: str                      # Current user question (from last message or API)
    route: RouteType                     # Router output: "policy" | "sql" | "web" | "fallback"
    confidence: float                   # Router confidence [0,1]; <0.7 → fallback (FR-003)
    routing_rationale: Optional[str]     # Explanation for routing (FR-005: log 100%)

    # --- Policy (RAG) pipeline ---
    retrieved_docs: List[Any]           # Chunks from vector store (ChromaDB/Qdrant)
    policy_context: Optional[str]        # Formatted context + citations (FR-011)

    # --- SQL (Order) pipeline ---
    sql_query: Optional[str]             # Generated/adjusted SQL
    sql_result: Optional[List[Dict]]     # Result rows from DB
    sql_error: Optional[str]             # If query failed

    # --- Web pipeline ---
    web_results: Optional[List[Dict]]    # Tavily search results
    web_context: Optional[str]           # Formatted web snippets for LLM

    # --- Fallback (Complaint or low confidence) ---
    fallback_reason: Optional[str]       # e.g. "complaint" | "low_confidence"
    handoff_context: Optional[Dict]     # Full history + routing for human (FR-028)

    # --- Synthesis ---
    final_response: Optional[str]        # Answer returned to user
    error: Optional[str]                 # Top-level error message if any
```

### 2.2 State Flow Summary

- **Router node** sets: `route`, `confidence`, `routing_rationale`, `user_query` (from `messages`). Routes to Fallback when intent=Complaint or confidence &lt; 0.7.
- **Retriever node** (Policy): sets `retrieved_docs`, `policy_context` (with citations per FR-011).
- **SQL node**: sets `sql_query`, `sql_result`, or `sql_error`.
- **Web node**: sets `web_results`, `web_context`.
- **Fallback node**: sets `fallback_reason`, `handoff_context`; may set `final_response` or prepare for human handoff.
- **Synthesis node**: sets `final_response` from pipeline-specific context.

---

## 3. Data Flow & Routing Logic (Blueprint Section 4.1.2)

### 3.1 Routing State Machine (Blueprint)

1. **START** → **ROUTER**
2. **ROUTER** → **RETRIEVER** (if Intent = Policy_Question)
3. **ROUTER** → **SQL_AGENT** (if Intent = Order_Status)
4. **ROUTER** → **WEB_SEARCH** (if Intent = General_Chat)
5. **ROUTER** → **FALLBACK** (if Intent = Complaint **or** Confidence &lt; 0.7)
6. **RETRIEVER / SQL / WEB** → **SYNTHESIS** (if successful)
7. **SYNTHESIS** → **END**  
   **FALLBACK** → handoff / response → **END**

Intent mapping: Policy_Question → `policy`; Order_Status → `sql`; General_Chat → `web`; Product_Info → `policy` or `web` (design choice); Complaint or low confidence → `fallback`.

### 3.2 Policy (RAG) Pipeline

```
User Query → Router → [route == "policy"] → Retriever Node → Vector DB (ChromaDB/Qdrant)
                → policy_context (+ citations FR-011) → LLM → final_response
```

- **Router:** classifies as Policy_Question (or Product_Info if mapped to policy).
- **Retriever:** semantic (+ hybrid BM25 per FR-009), reranking (FR-012), metadata filter (FR-010); format with source/page citations (FR-011).
- **LLM:** generates answer from `policy_context`; writes `final_response`.

### 3.3 Order (SQL) Pipeline

```
User Query → Router → [route == "sql"] → SQL Node → SQLAlchemy (SQLite/PostgreSQL)
                → sql_result → LLM (with result) → final_response
```

- **Router:** classifies as Order_Status.
- **SQL node:** NL→SQL (FR-013), schema introspection (FR-014), read-only enforcement (FR-015), execute, set `sql_result` or `sql_error` (FR-017).
- **LLM:** tabular → natural language (FR-016); writes `final_response`.

### 3.4 Web Pipeline

```
User Query → Router → [route == "web"] → Web Node → Tavily API
                → web_results → web_context → LLM → final_response
```

- **Router:** classifies as General_Chat (or Product_Info if mapped to web).
- **Web node:** Tavily (FR-019), domain allowlist/blocklist (FR-020), combine 3–5 sources (FR-021), freshness (FR-022); rate limit (FR-024). Web should be &lt;15% of queries (FR-023).
- **LLM:** synthesizes from `web_context`; writes `final_response`.

### 3.5 Fallback Pipeline

```
User Query → Router → [route == "fallback"] → Fallback Node → handoff_context / response → END
```

- **Router:** Complaint intent **or** confidence &lt; 0.7 (FR-003).
- **Fallback node:** set `fallback_reason`, build `handoff_context` with full history + routing (FR-028); return message or prepare for human handoff.

### 3.6 High-Level Graph Topology

```
                    ┌─────────────────┐
                    │   ENTRY (input) │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   ROUTER NODE   │  ← GPT-4o-mini; outputs route + confidence + rationale
                    └────────┬────────┘
                             │
     ┌───────────────────────┼───────────────────────┬─────────────────┐
     │                       │                       │                 │
     ▼                       ▼                       ▼                 ▼
┌───────────┐         ┌───────────┐         ┌───────────┐       ┌───────────┐
│ RETRIEVER │         │ SQL NODE  │         │ WEB NODE  │       │ FALLBACK  │
│   NODE    │         │           │         │           │       │   NODE    │
└─────┬─────┘         └─────┬─────┘         └─────┬─────┘       └─────┬─────┘
      │                     │                     │                   │
      └─────────────────────┼─────────────────────┘                   │
                            │                                           │
                            ▼                                           │
                    ┌───────────────┐                                    │
                    │  SYNTHESIS   │                                    │
                    │    NODE      │◄───────────────────────────────────┘
                    └───────┬──────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  END (output)│
                    └───────────────┘
```

Conditional edges from the router: exactly one of Retriever, SQL, Web, or Fallback based on `state["route"]`.

### 3.7 Agentic Behaviours (Blueprint Section 4.2)

- **ReAct pattern:** Thought → Action (e.g. `sql_query(...)`) → Observation → Response.
- **Agent communication protocol:** Use standardized message types where applicable: `UserMessage`, `RouterDecision`, `ToolRequest`, `ToolResponse`, `SynthesisRequest`, `AgentResponse`.
- **Self-correction:** e.g. SQL node retries on invalid SQL; tool results feed back into state.

---

## 4. Development Phases → Granular Engineering Tasks (Blueprint Section 6)

Tasks are check-boxable. Order respects dependencies (e.g. state and router before pipelines).

### Phase 1: Foundation & Project Scaffolding

- [ ] **1.1** Create repo structure per Blueprint Section 5.2:  
  `src/agents/` (router.py, retriever.py, sql_agent.py), `src/graph/` (nodes.py, edges.py, state.py), `src/tools/`, `src/prompts/`, `src/utils/`, `data/` (policies, db, vectors), `api/` (FastAPI app), `tests/`, `config/`
- [ ] **1.2** Add `requirements.txt` or `pyproject.toml` with deps from Blueprint 5.1.2 (langgraph, langchain, openai, fastapi, sqlalchemy, chromadb, qdrant-client, tavily-python, pydantic>=2, python-dotenv, pytest, langsmith, etc.)
- [ ] **1.3** Create `.env.example` with placeholders: `OPENAI_API_KEY`, `TAVILY_API_KEY`, `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, DB URLs for SQLite/PostgreSQL, vector store config (ChromaDB path / Qdrant URL)
- [ ] **1.4** Add `config/settings.py` (or equivalent) loading env via `python-dotenv`, Pydantic settings, no secrets in code
- [ ] **1.5** Add minimal `src/core/` modules (e.g. logging, env validation) and ensure `src` is a package

### Phase 2: Core State & Router

- [ ] **2.1** Implement `src/graph/state.py`: define `AgentState` TypedDict and `RouteType` as above; add reducer for `messages` if using append semantics
- [ ] **2.2** Implement `src/agents/router.py`: load router prompt from `src/prompts/`; call GPT-4o-mini; return intent (Policy_Question → policy, Order_Status → sql, General_Chat → web, Product_Info → policy/web, Complaint → fallback) and confidence; set `fallback` when confidence &lt; 0.7 (FR-003); log `routing_rationale` (FR-005); strict type hints and docstrings
- [ ] **2.3** Implement `src/graph/nodes.py`: skeleton nodes with signatures `(state: AgentState) -> Partial[AgentState]`:  
  - [ ] **2.3a** `router_node` — sets `route`, `confidence`, `routing_rationale`, `user_query`  
  - [ ] **2.3b** `retriever_node` — stub: set empty `retrieved_docs`/`policy_context`  
  - [ ] **2.3c** `sql_node` — stub: set placeholder `sql_result` or `sql_error`  
  - [ ] **2.3d** `web_node` — stub: set placeholder `web_results`/`web_context`  
  - [ ] **2.3e** `fallback_node` — stub: set `fallback_reason`, `handoff_context` or `final_response`
- [ ] **2.4** Implement `src/graph/edges.py` (or in graph.py): conditional edge logic from router to retriever/sql/web/fallback based on `state["route"]`
- [ ] **2.5** Implement `src/graph/graph.py`: build `StateGraph(AgentState)`, add all nodes including Fallback, add conditional edges, RETRIEVER/SQL/WEB/FALLBACK → SYNTHESIS → END; compile graph
- [ ] **2.6** Unit tests for `router_node` (mock LLM; assert fallback when confidence &lt; 0.7) and for state transitions (stub nodes)

### Phase 3: RAG (Policy) Pipeline — First Vertical (FR-007 to FR-012)

- [ ] **3.1** Implement `src/utils/ingest.py`: PDF, DOCX, TXT, HTML (FR-007); chunk 512–1024 tokens; embed (text-embedding-3-small); persist to ChromaDB / Qdrant
- [ ] **3.2** Implement vector store abstraction: ChromaDB (dev) / Qdrant (prod); semantic search (FR-008); optional hybrid BM25 (FR-009); metadata filtering (FR-010)
- [ ] **3.3** Implement `retriever_node`: query vector store, format chunks into `policy_context` with **citations and page numbers** (FR-011); optional cross-encoder reranking (FR-012)
- [ ] **3.4** Add policy synthesis: `policy_context` + `user_query` → LLM → `final_response`
- [ ] **3.5** Unit tests for ingest and `retriever_node` (mocked vector store)
- [ ] **3.6** Optional: LangSmith tracing for retriever and policy LLM

### Phase 4: SQL (Order) Pipeline (FR-013 to FR-018)

- [ ] **4.1** Define SQL schema (Blueprint 3.2): `orders`, `order_items`, `customers`, `shipments`, `returns`, `products`; SQLite (dev) / PostgreSQL (prod); migrations or init script
- [ ] **4.2** Implement async SQLAlchemy engine/session from config
- [ ] **4.3** Implement `src/tools/sql.py`: parameterized execution; **read-only enforcement** — block DROP/DELETE/UPDATE (FR-015); return rows as list of dicts; actionable errors (FR-017)
- [ ] **4.4** Implement `sql_node` (or `src/agents/sql_agent.py`): NL→SQL (FR-013), **schema introspection** (FR-014), execute via sql tool, set `sql_result` or `sql_error`; optional query templates (FR-018)
- [ ] **4.5** Synthesis: LLM converts tabular result to natural language (FR-016) → `final_response`
- [ ] **4.6** Unit tests for SQL execution and `sql_node` (mocked LLM/tool)

### Phase 5: Web Search Pipeline (FR-019 to FR-024)

- [ ] **5.1** Implement `src/tools/web_search.py`: Tavily API (async); target &lt;3s for 95% of searches (FR-019); **rate limit 1000/hour** (FR-024); domain allowlist/blocklist (FR-020); return normalized results (title, snippet, URL)
- [ ] **5.2** Implement `web_node`: call Tavily, combine 3–5 sources into `web_context` (FR-021); optional freshness weighting for news (FR-022); ensure web is fallback (&lt;15% of queries, FR-023)
- [ ] **5.3** Synthesis: LLM → `final_response` from `web_context` + query
- [ ] **5.4** Unit tests for Tavily client (mocked HTTP) and `web_node`

### Phase 6: API Layer, Fallback & Integration

- [ ] **6.1** Implement FastAPI app in `api/`: health check, CORS, lifespan (graph load, DB pool)
- [ ] **6.2** Pydantic V2 request/response models: `QueryRequest`, `QueryResponse` (query, response, route, confidence, sources, citations)
- [ ] **6.3** Expose `POST /query`: invoke compiled graph; return `final_response`, route, citations; support multi-turn via session/conversation ID (FR-004)
- [ ] **6.4** Implement **Fallback node** fully: build `handoff_context` with full history + routing (FR-028); return message or handoff payload for human escalation
- [ ] **6.5** Async throughout (`astream`/`ainvoke`); structured logging with **correlation IDs** (NFR); LangSmith request IDs

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

## 5. Router Prompt Template (Blueprint Section 5.3.1)

Implement the router prompt in `src/prompts/` and load it in `src/agents/router.py`.

### 5.1 Intent Categories (FR-001)

| Intent | Description | Graph route |
|--------|-------------|-------------|
| **Policy_Question** | Return policy, shipping, FAQ, terms, manuals, internal procedures | `policy` → Retriever |
| **Order_Status** | Order tracking, order history, shipment status, returns | `sql` → SQL Node |
| **General_Chat** | General knowledge, external/real-time info, news | `web` → Web Node |
| **Product_Info** | Product details (can map to policy docs or web) | `policy` or `web` |
| **Complaint** | Complaints, escalations, negative sentiment | `fallback` → Fallback |

### 5.2 Outputs

- **Intent** (one of the above, or multi-label per FR-002 for compound queries).
- **Confidence** in [0, 1]. If &lt; 0.7 → force route to `fallback` (FR-003).
- **Routing rationale** (for logging, FR-005).

### 5.3 Prompt Instructions (conceptual)

- When to choose Policy: internal docs, procedures, FAQs, terms, product manuals.
- When to choose SQL: orders, order ID, shipment, returns, customer data.
- When to choose Web: external facts, news, competitor info, regulatory updates.
- When to choose Fallback: complaint intent or confidence &lt; 0.7.
- Include few-shot examples for each intent; support multi-turn context (FR-004).

---

## 6. File-to-Phase Quick Reference

| Path | Phase |
|------|--------|
| `config/`, `.env.example`, `requirements.txt` or `pyproject.toml` | 1 |
| `src/graph/state.py`, `edges.py`, `graph.py` | 2 |
| `src/agents/router.py`, `retriever.py`, `sql_agent.py` | 2, 3, 4 |
| `src/graph/nodes.py` (router, retriever, sql, web, **fallback**, synthesis) | 2, 3, 4, 5, 6 |
| `src/prompts/` (router prompt) | 2 |
| `src/utils/ingest.py` | 3 |
| `src/tools/` (vector, sql, web_search) | 3, 4, 5 |
| `data/` (policies, db, vectors) | 1, 3, 4 |
| `api/` (FastAPI app) | 6 |
| `tests/` | 2–8 |

---

## 7. Next Step

This plan now incorporates the **full blueprint context** (Executive Summary, FR/NFR traceability, routing logic with Fallback, ReAct/protocol, data sources, and exact project structure). It is the single source of truth for:

- Executive context (problem, solution, classification-first paradigm)
- Functional & non-functional requirements (FR-001–FR-030, NFR table)
- StateGraph schema (`AgentState`, `RouteType`, confidence, fallback, handoff_context)
- Routing state machine: Router → Retriever | SQL | Web | **Fallback** → Synthesis → End
- Intent categories (Policy_Question, Order_Status, General_Chat, Product_Info, Complaint)
- Phased, check-boxable engineering tasks with FR traceability

Proceed to **Step 2: Project Scaffolding** when ready (folders per Blueprint 5.2, dependency file, `.env.example`).
