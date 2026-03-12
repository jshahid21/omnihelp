# Omni-Help Deep Dive: RAG & SQL Pipelines

**Learning Module 02 — Phases 4–5**  
**Audience:** Intermediate Python developer moving into production AI systems  
**Files covered:** `utils/ingest.py` · `tools/vector_store.py` · `tools/sql_db.py` · `utils/init_db.py` · `graph/nodes.py` (retriever_node, sql_node)

---

## Table of Contents

1. [Mental Model: Two Kinds of Memory](#1-mental-model-two-kinds-of-memory)
2. [The Policy RAG Pipeline (Phase 4)](#2-the-policy-rag-pipeline-phase-4)
   - 2.1 What Are Embeddings?
   - 2.2 Why `text-embedding-3-small`?
   - 2.3 `ingest.py` — Line by Line
   - 2.4 `vector_store.py` — The Retriever Wrapper
   - 2.5 `retriever_node` — How the Graph Queries It
3. [The SQL (Order Management) Pipeline (Phase 5)](#3-the-sql-order-management-pipeline-phase-5)
   - 3.1 The Database Schema
   - 3.2 `sql_db.py` — The Secure Tool Layer
   - 3.3 FR-015: The Read-Only Regex Guardrail
   - 3.4 `sql_node` — NL→SQL in Action
4. [Enterprise Patterns vs. Amateur Code](#4-enterprise-patterns-vs-amateur-code)
5. [Common Misconceptions](#5-common-misconceptions)

---

## 1. Mental Model: Two Kinds of Memory

An AI agent serving real customers needs access to two fundamentally different types of information:

| Memory Type | What it Stores | How You Query It | Omni-Help Implementation |
|---|---|---|---|
| **Semantic / Unstructured** | Policy docs, FAQs, product descriptions — free-form text | "Find me the 3 most similar paragraphs to this question" | ChromaDB + `text-embedding-3-small` |
| **Structured / Relational** | Order records, customer data, inventory — rows and columns | `SELECT * FROM orders WHERE order_id = 'ORD-1001'` | SQLite + LangChain SQLDatabase |

The key insight: **the query shape determines the storage shape**. When a user asks "What is your return policy?", the answer is buried in a paragraph of a document — the right tool is semantic search. When they ask "Where is my order?", the answer is a specific row in a database table — the right tool is SQL.

The Router Agent (Phase 1) is responsible for choosing between these two paths.

```
User Query
    │
    ├── "What is the return policy?"   → intent: policy  → Retriever Node (ChromaDB)
    │
    └── "Where is order ORD-1001?"     → intent: sql     → SQL Node (SQLite)
```

---

## 2. The Policy RAG Pipeline (Phase 4)

RAG stands for **Retrieval-Augmented Generation**. The pattern has three steps:

1. **Retrieve** relevant chunks from a knowledge base
2. **Augment** the LLM's prompt with those chunks as context
3. **Generate** a response grounded in the retrieved context (not the LLM's training data)

This matters enormously for enterprise software. Company policies change. Return windows get updated. New shipping restrictions are added. If you relied on the LLM's pre-trained knowledge, your answers would be stale the moment the document changed. RAG ensures the AI always reads the latest version of the truth.

### 2.1 What Are Embeddings?

Before you can do semantic search, you need to convert text into a format a computer can compare. That format is a **vector** — a list of floating-point numbers (e.g., 1,536 numbers for `text-embedding-3-small`).

The key property: **semantically similar text produces numerically similar vectors.** This means "return policy" and "how do I send an item back?" end up close together in the 1,536-dimensional space, even though they share no words.

```
"What is your return policy?"  → [0.021, -0.148, 0.392, ...]  ─┐
                                                                  ├── cos_similarity ≈ 0.94 (close!)
"How do I send an item back?"  → [0.018, -0.151, 0.389, ...]  ─┘

"Where is my order?"           → [0.872, 0.043, -0.211, ...]     (far away)
```

When a user submits a query, ChromaDB:
1. Embeds the query into a vector (via the same embedding model)
2. Searches for the `k` stored vectors with the highest cosine similarity
3. Returns the corresponding text chunks

### 2.2 Why `text-embedding-3-small`?

OpenAI offers several embedding models. Here is the trade-off table:

| Model | Dimensions | Cost | MTEB Score | Best For |
|---|---|---|---|---|
| `text-embedding-3-small` | 1,536 | 💚 Cheapest | Good | Dev, internal tools, high-volume |
| `text-embedding-3-large` | 3,072 | 🟡 Mid | Best | High-stakes retrieval, multilingual |
| `text-embedding-ada-002` | 1,536 | 🟡 Mid | Moderate | Legacy — avoid for new projects |

For a customer support system with dozens of policy documents, `text-embedding-3-small` provides an excellent quality-to-cost ratio. If you needed multilingual support or were searching across millions of documents, you would graduate to `text-embedding-3-large`.

**Critical constraint:** the same model must be used for both ingestion and querying. If you embed documents with Model A and query with Model B, the vectors live in different spaces and similarity scores are meaningless.

```python
# ingest.py — writes vectors
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# vector_store.py — reads vectors
EMBEDDING_MODEL = "text-embedding-3-small"  # Must match ingest.py exactly
_embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
```

### 2.3 `ingest.py` — Line by Line

`src/utils/ingest.py` is a one-time (or on-update) script that converts raw markdown files into searchable vectors. Here is the complete pipeline it executes:

```
data/policies/*.md
      │
      │  Step 1: LOAD
      │  DirectoryLoader + UnstructuredMarkdownLoader
      │  → list of Document objects (full file each)
      ▼
┌─────────────┐
│  Documents  │  e.g. [Document(page_content="# Return Policy\n...")]
└─────────────┘
      │
      │  Step 2: SPLIT
      │  RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
      │  → smaller Document objects
      ▼
┌────────────────────────────────────────────────────────┐
│  Chunks                                                 │
│  [Document("Items can be returned within 30 days..."), │
│   Document("To initiate a return, visit..."),          │
│   ...]                                                  │
└────────────────────────────────────────────────────────┘
      │
      │  Step 3: EMBED + PERSIST
      │  OpenAIEmbeddings("text-embedding-3-small")
      │  Chroma.from_documents(...)  → persists to data/vectors/
      ▼
┌───────────────────────────┐
│  ChromaDB (data/vectors/) │
│  Vector: [0.021, -0.148, ...]  → "Items can be returned..."
│  Vector: [0.019, -0.142, ...]  → "To initiate a return..."
│  ...                           
└───────────────────────────┘
```

**Why chunking?** LLMs have context windows. You cannot paste an entire 20-page policy document into a prompt. Chunking splits documents into 500-character pieces with 50-character overlap. The overlap prevents a sentence from being split across two chunks and losing meaning at the boundary.

**`add_start_index=True`** is a subtle but important detail — it records the character offset of each chunk within its source document. This enables precise citation: "See our return policy, paragraph starting at character 1,247."

```python
# From ingest.py
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,  # Records byte offset for citation traceability
)
```

**`RecursiveCharacterTextSplitter`** is smarter than a plain character splitter. It tries to split on `\n\n` first (paragraph boundaries), then `\n` (line breaks), then ` ` (spaces), only falling back to raw character splits as a last resort. This means chunks are more likely to be semantically coherent units.

### 2.4 `vector_store.py` — The Retriever Wrapper

`src/tools/vector_store.py` wraps ChromaDB behind a stable interface. This is a critical architectural decision:

```python
# What the rest of the codebase sees:
from tools.vector_store import get_policy_retriever, format_docs

retriever = get_policy_retriever(k=5)
docs = retriever.invoke("What is the return window?")
context = format_docs(docs)
```

The `graph/nodes.py` file never imports `Chroma` or `OpenAIEmbeddings` directly. If you need to migrate from ChromaDB to Qdrant for production, you change exactly one file (`vector_store.py`) and every node continues working unchanged.

**The singleton pattern** prevents re-opening the database on every query:

```python
# ❌ Amateur: opens a new ChromaDB connection per request (expensive)
def get_policy_retriever(k=5):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # New HTTP pool!
    db = Chroma(persist_directory="./data/vectors", ...)
    return db.as_retriever(search_kwargs={"k": k})

# ✅ Enterprise: singleton opened once at module import, reused per request
_embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

def _get_db() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_embeddings,   # Reuses the single instance
        persist_directory=VECTOR_DIR,
    )
```

**`format_docs`** adds citations to every chunk, turning raw document objects into a single string the LLM can read:

```python
def format_docs(docs: List[Document]) -> str:
    sections = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown source")
        sections.append(f"{doc.page_content.strip()}\n[Source: {source}]")
    return "\n\n".join(sections)
```

Output example:
```
Items can be returned within 30 days of purchase...
[Source: data/policies/return_policy.md]

Electronics must be returned in original packaging...
[Source: data/policies/return_policy.md]
```

This citation string is what the Synthesis Node eventually passes to the LLM, allowing it to write responses like "According to our return policy..." instead of hallucinating.

### 2.5 `retriever_node` — How the Graph Queries It

```python
def retriever_node(state: AgentState) -> dict[str, Any]:
    user_query = state.get("user_query", "")
    if not user_query:
        # Fallback: extract from conversation history
        for msg in reversed(state.get("messages", [])):
            if hasattr(msg, "type") and msg.type == "human":
                user_query = msg.content
                break

    retriever = get_policy_retriever(k=5)
    docs = retriever.invoke(user_query)
    policy_context = format_docs(docs)

    return {
        "retrieved_docs": docs,
        "policy_context": policy_context,
        "final_response": f"Here is the raw policy context:\n\n{policy_context}",
    }
```

**Observation — the message history fallback:** If `user_query` is empty (which can happen in multi-turn conversations where a new message was added to `messages` but not re-extracted), the node walks backwards through `messages` to find the last human message. This is defensive coding — the node works correctly regardless of which part of the state was populated.

**Observation — partial state updates:** The node returns only the keys it modified. LangGraph performs a shallow merge of this dict onto the full `AgentState`. Other keys (`intent`, `confidence`, `sql_result`, etc.) are untouched.

---

## 3. The SQL (Order Management) Pipeline (Phase 5)

### 3.1 The Database Schema

The orders database (`data/db/orders.db`) has two tables:

```sql
CREATE TABLE orders (
    order_id       TEXT PRIMARY KEY,
    customer_email TEXT NOT NULL,
    status         TEXT NOT NULL,        -- 'Pending', 'Shipped', 'Delivered', 'Cancelled'
    total_amount   REAL NOT NULL,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE shipments (
    shipment_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id        TEXT NOT NULL REFERENCES orders(order_id),
    tracking_number TEXT NOT NULL,
    carrier         TEXT NOT NULL,       -- 'UPS', 'FedEx', 'USPS', etc.
    eta             TEXT NOT NULL,
    shipped_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

This is initialized by `python src/utils/init_db.py` using Python's built-in `sqlite3` module, which requires no external services for local development.

### 3.2 `sql_db.py` — The Secure Tool Layer

The SQL tool follows the same Facade pattern as `vector_store.py`: the rest of the codebase never imports `sqlite3` or `SQLAlchemy` directly.

```python
# What the rest of the codebase sees:
from tools.sql_db import execute_secure_query, get_schema

schema = get_schema()            # → "CREATE TABLE orders (order_id TEXT, ...)"
result = execute_secure_query(   # → "[('ORD-1001', 'Shipped', ...)]"
    "SELECT * FROM orders WHERE order_id = 'ORD-1001'"
)
```

**LangChain `SQLDatabase`** is used as the connection wrapper rather than raw SQLite because it provides a crucial feature: `get_table_info()` returns a formatted string describing all tables and their columns. This string is injected into the NL→SQL system prompt so the LLM knows the exact schema it is writing SQL for.

```python
_db = SQLDatabase.from_uri("sqlite:///data/db/orders.db")
print(_db.get_table_info())
# Output:
# CREATE TABLE orders (
#   order_id TEXT,
#   customer_email TEXT NOT NULL,
#   ...
# )
```

Without this, the LLM would be generating SQL blindly and produce errors like `no such table: order` (wrong name) or `no such column: order_status` (wrong column name).

### 3.3 FR-015: The Read-Only Regex Guardrail

This is the most important security control in the entire codebase. The requirement is simple: **a customer support AI must never be able to modify or delete data**, even if a malicious user crafts a prompt to trick the LLM into generating a destructive SQL query.

```python
# From sql_db.py
_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|REPLACE|MERGE|CREATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

def validate_read_only_sql(query: str) -> None:
    match = _FORBIDDEN_KEYWORDS.search(query)
    if match:
        raise ValueError(
            f"Forbidden SQL operation detected: '{match.group().upper()}'. "
            "Only SELECT queries are permitted."
        )
```

**Why `\b` word boundaries?** This is a subtle but critical regex decision. Without word boundaries, the pattern would incorrectly match legitimate column names:

```
# Without \b word boundaries:
query = "SELECT * FROM orders WHERE status = 'Dropped'"
_FORBIDDEN_KEYWORDS.search(query)  # Matches "DROP" inside "Dropped"!  ❌

# With \b word boundaries:
query = "SELECT * FROM orders WHERE status = 'Dropped'"
_FORBIDDEN_KEYWORDS.search(query)  # No match — "DROP" is not a standalone word ✅

query = "DROP TABLE orders"
_FORBIDDEN_KEYWORDS.search(query)  # Matches "DROP" — correctly blocked ✅
```

**Why not just check `if query.strip().upper().startswith("SELECT")`?** That approach can be bypassed:

```sql
-- Bypass attempt 1: semicolon injection
SELECT 1; DROP TABLE orders

-- Bypass attempt 2: comment stripping tricks
SELECT * FROM orders; --
DELETE FROM orders
```

The regex approach, while still not a complete sandbox, catches far more mutation patterns because it scans the entire query string, not just the first keyword. A production system would layer this with database-level permissions (the DB user should have `SELECT` privileges only).

**The full attack-defense sequence:**

```
User: "Delete all my orders"
         │
         ▼
  Router classifies: intent=sql
         │
         ▼
  sql_node calls _sql_llm.invoke(...)
         │
         ▼
  LLM generates: "DELETE FROM orders WHERE customer_email = 'user@example.com'"
         │
         ▼
  execute_secure_query("DELETE FROM ...") is called
         │
         ▼
  validate_read_only_sql() scans: found "DELETE" at word boundary
         │
         ▼
  ValueError: "Forbidden SQL operation detected: 'DELETE'"
         │
         ▼
  sql_node catches ValueError, writes to state["sql_error"]
         │
         ▼
  synthesis_node reads sql_error, generates polite error message
         │
         ▼
  User sees: "I'm sorry, I'm only able to look up order information..."
```

**No crash. No data loss. No raw stack trace exposed to the user.**

### 3.4 `sql_node` — NL→SQL in Action

```python
def sql_node(state: AgentState) -> dict[str, Any]:
    user_query = state.get("user_query", "")

    # Step 1: Get schema so the LLM knows the table structure
    schema = get_schema()

    system_prompt = (
        "You are an SQLite expert. Given an input question, create a syntactically "
        "correct SQLite query to run. Only return the raw SQL query — no markdown "
        "formatting, no backticks, no explanation.\n\n"
        f"Database schema:\n{schema}"
    )

    # Step 2: NL → SQL via GPT-4o-mini (temperature=0 for deterministic output)
    response = _sql_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query),
    ])
    sql_query = response.content.strip()

    # Step 3: Validate (FR-015) + execute
    try:
        result = execute_secure_query(sql_query)
        return {
            "sql_query": sql_query,
            "sql_result": result,
            "sql_error": None,
            "final_response": f"SQL Result:\n{result}",
        }
    except ValueError as guard_err:
        return {"sql_query": sql_query, "sql_result": None, "sql_error": str(guard_err), ...}
    except Exception as exec_err:
        return {"sql_query": sql_query, "sql_result": None, "sql_error": str(exec_err), ...}
```

**Why `temperature=0`?** SQL generation is a deterministic task — you want the LLM to always produce the most probable (i.e., most syntactically correct) query, not a creative variation. A `temperature=0` setting makes the LLM's output reproducible and prevents it from "experimenting" with untested SQL constructs.

**Why store both `sql_query` and `sql_result` in state?** The Synthesis Node needs both. SQLite's `SQLDatabase.run()` returns raw tuples like `[('ORD-1001', 'Shipped', 129.99)]` — the LLM cannot know which value corresponds to which column without reading the `SELECT` clause. Storing the query text alongside the result lets the Synthesis Node inject: "The query was `SELECT order_id, status, total_amount FROM orders WHERE order_id = 'ORD-1001'` and returned `[('ORD-1001', 'Shipped', 129.99)]`" — which the LLM can correctly parse.

---

## 4. Enterprise Patterns vs. Amateur Code

### Pattern 1: The Facade / Tool Wrapper

| Amateur | Enterprise |
|---|---|
| `from langchain_chroma import Chroma` directly in `nodes.py` | `from tools.vector_store import get_policy_retriever` |
| `import sqlite3` directly in `nodes.py` | `from tools.sql_db import execute_secure_query, get_schema` |

The Facade pattern decouples the orchestration layer from the infrastructure layer. Nodes express *what* they need; tools express *how* to get it. Migrating from SQLite → PostgreSQL or ChromaDB → Qdrant is a one-file change.

### Pattern 2: Lazy Singleton Initialization

```python
# ❌ Eager singleton — crashes at import if DB is not seeded yet
_db = SQLDatabase.from_uri("sqlite:///data/db/orders.db")

# ✅ Lazy singleton — only connects when first called
_db: SQLDatabase | None = None
def _get_db() -> SQLDatabase:
    global _db
    if _db is None:
        _db = SQLDatabase.from_uri("sqlite:///data/db/orders.db")
    return _db
```

Lazy initialization means importing `sql_db` in tests does not crash if the database file does not exist yet. Tests can mock `_get_db` without fighting a module-level exception.

### Pattern 3: Graceful Error States

```python
# ❌ Crash — graph stops, user sees a 500 error
result = db.run(sql_query)   # raises OperationalError if query is bad

# ✅ Graceful — error is captured in state, Synthesis Node surfaces it politely
try:
    result = execute_secure_query(sql_query)
    return {"sql_result": result, "sql_error": None, ...}
except Exception as e:
    return {"sql_result": None, "sql_error": str(e), ...}
```

---

## 5. Common Misconceptions

### "I can skip ingestion and just give the LLM the full document"

This only works for very small documents (under 4,000 tokens). A real enterprise policy library might contain 50 documents with 10,000 words each. That is ~500,000 tokens per request — prohibitively expensive and exceeds every current context window. Chunking + vector search means each request retrieves only the 5 most relevant paragraphs (~500 tokens).

### "The embedding model generates the answer"

Embedding models are *not* language models. They produce fixed-length vectors and never generate text. The pipeline is:

```
User Query → [Embedding Model] → vector → ChromaDB search → top-5 chunks
                                                                   │
                               User Query + chunks → [LLM] → response
```

The LLM (GPT-4o-mini) generates the final response. The embedding model only indexes and retrieves.

### "The guardrail makes mutation impossible"

The regex guardrail is a necessary but not sufficient control. It is a **defense-in-depth layer** to catch LLM-generated mutation SQL. For complete protection, the database user (in production) should be granted `SELECT` privileges only at the PostgreSQL level. The application layer and the database layer should both enforce read-only access independently.

### "temperature=0 means the SQL is always correct"

`temperature=0` means the LLM always picks the highest-probability token at each step. It does not guarantee correctness — it guarantees *consistency*. The same query will always produce the same SQL, but that SQL might still be wrong for edge cases the LLM has not seen. The error handler in `sql_node` catches execution failures for exactly this reason.
