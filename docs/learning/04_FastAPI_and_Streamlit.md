# Omni-Help Deep Dive: FastAPI Layer & Streamlit UI

**Learning Module 04 — Phase 7 (API) & Phase 8 (Frontend)**  
**Audience:** Intermediate Python developer moving into production AI systems  
**Files covered:** `api/main.py` · `api/schema.py` · `frontend/app.py`

---

## Table of Contents

1. [Mental Model: The Two-Process Architecture](#1-mental-model-the-two-process-architecture)
2. [The FastAPI Application (`api/main.py`)](#2-the-fastapi-application-apimainpy)
   - 2.1 The Lifespan Pattern — Loading the Graph at Startup
   - 2.2 Correlation IDs — End-to-End Request Tracing
   - 2.3 CORS — Why You Need It and How to Tighten It
   - 2.4 The Request Size Limit Middleware
   - 2.5 The `/api/v1/chat` Route — Async All the Way Down
3. [The Pydantic V2 Schemas (`api/schema.py`)](#3-the-pydantic-v2-schemas-apischemapy)
   - 3.1 The Internal/External Boundary
   - 3.2 Field Validation as Documentation
4. [The Streamlit Frontend (`frontend/app.py`)](#4-the-streamlit-frontend-frontendapppy)
   - 4.1 How Streamlit's Execution Model Works
   - 4.2 Session State — Persisting Data Across Rerenders
   - 4.3 The API Bridge — Connecting to FastAPI
   - 4.4 The Routing Metadata Badge
   - 4.5 Multi-Turn Conversation via `conversation_id`
5. [Enterprise Patterns vs. Amateur Code](#5-enterprise-patterns-vs-amateur-code)
6. [Common Misconceptions](#6-common-misconceptions)

---

## 1. Mental Model: The Two-Process Architecture

When you run Omni-Help locally, you start two separate processes:

```
Terminal 1: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload --app-dir src
Terminal 2: streamlit run src/frontend/app.py
```

These are completely independent operating system processes:

```
┌─────────────────────────────┐        HTTP/JSON        ┌──────────────────────────────┐
│   Streamlit Process          │  ──────────────────▶   │   FastAPI Process             │
│   localhost:8501             │  POST /api/v1/chat      │   localhost:8000              │
│                              │                         │                               │
│   Python script that re-     │  ◀──────────────────   │   LangGraph + LLM clients    │
│   runs on every interaction  │  ChatResponse JSON      │   persistent in memory        │
│                              │                         │                               │
│   st.session_state           │                         │   _graph (compiled once)     │
│   (browser-local state)      │                         │   thread state per session   │
└─────────────────────────────┘                         └──────────────────────────────┘
```

Why this split? Because Streamlit re-runs your entire Python script from top to bottom on every user interaction (button click, chat message, etc.). If the LangGraph, ChromaDB connection, and LLM clients lived inside the Streamlit process, they would be reconstructed on every click — an expensive operation. The FastAPI server holds all the heavyweight components loaded once; Streamlit is a thin presentation layer that just makes HTTP requests.

---

## 2. The FastAPI Application (`api/main.py`)

### 2.1 The Lifespan Pattern — Loading the Graph at Startup

FastAPI's `lifespan` context manager is the modern (v0.95+) replacement for `@app.on_event("startup")`. It uses Python's `asynccontextmanager` to define startup and shutdown logic:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Everything before yield runs at startup
    global _graph
    logger.info("Starting Omni-Help API — compiling LangGraph...")
    from graph.graph import build_graph
    _graph = build_graph()   # Compiles the StateGraph, connects all LLM singletons
    logger.info("LangGraph compiled and ready.")

    yield  # ← Server is running here, handling requests

    # Everything after yield runs at shutdown
    logger.info("Omni-Help API shutting down.")


app = FastAPI(..., lifespan=lifespan)
```

**Why compile the graph at startup?** `build_graph()` calls `graph_builder.compile()`, which:
- Validates the graph topology (catches disconnected nodes, missing edges)
- Initializes all module-level singletons (LLM clients, DB connections) via the imports in `nodes.py`
- Returns a compiled `CompiledStateGraph` object that is highly efficient to invoke

If you moved `build_graph()` into the `/chat` route handler, you would recompile the graph on every single request. On a system receiving 100 requests/minute, that is 100 unnecessary compilations per minute.

**The `global _graph` pattern** makes the compiled graph accessible to all route handlers. The `_graph: Any = None` initialization at module level makes it clear to type checkers that `_graph` starts as `None` and is populated at startup.

**The deferred import pattern:**

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    from graph.graph import build_graph    # ← Import inside the function
    _graph = build_graph()
```

Importing `build_graph` inside the lifespan function (rather than at the top of `main.py`) prevents circular import issues. `graph.py` imports from `nodes.py`, which imports from `tools/`, which imports from `config/`. If `main.py` imported all of this at module load time, a failed import deep in the chain would crash the server before it even started — and the error would be harder to trace.

### 2.2 Correlation IDs — End-to-End Request Tracing

Every request generates a UUID:

```python
@app.post("/api/v1/chat", ...)
async def chat(request: ChatRequest) -> ChatResponse:
    correlation_id = str(uuid.uuid4())

    logger.info(
        "Request started | correlation_id=%s | conversation_id=%s | query=%r",
        correlation_id, request.conversation_id, request.query[:120],
    )

    try:
        result = await _graph.ainvoke(...)
    except Exception as exc:
        logger.error("Graph invocation failed | correlation_id=%s | ...", correlation_id, exc)
        raise HTTPException(
            status_code=500,
            detail={"error": "...", "correlation_id": correlation_id}
        )

    return ChatResponse(
        response=final_response,
        correlation_id=correlation_id,  # ← Returned to the client
        ...
    )
```

The `correlation_id` appears in three places:
1. **Start log** — "Request started | correlation_id=abc123..."
2. **Error log** (if the graph fails) — "Graph invocation failed | correlation_id=abc123..."
3. **Response JSON** — returned to the client

This means when a user reports a bug and gives you their `correlation_id` from the UI badge, you can run:

```bash
grep "abc123-def456" /var/log/omnihelp.log
```

And immediately see the complete request lifecycle: what query was submitted, what intent was classified, and exactly where (if anywhere) an error occurred. This is the foundation of production observability.

**LangSmith integration** extends this further: when `LANGCHAIN_TRACING_V2=true`, every `_graph.ainvoke(...)` call automatically creates a LangSmith trace linked to the correlation ID. You can open the LangSmith UI and see the exact LLM prompt, tokens used, and latency for every node in the graph for that specific request.

### 2.3 CORS — Why You Need It and How to Tighten It

CORS (Cross-Origin Resource Sharing) is a browser security mechanism. When JavaScript running at `http://localhost:8501` (Streamlit) tries to call `http://localhost:8000` (FastAPI), the browser first sends a **preflight OPTIONS request** asking: "Does this server allow requests from `localhost:8501`?"

The `CORSMiddleware` configuration tells the browser "yes":

```python
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
```

**Why not `allow_origins=["*"]`?** The wildcard allows any website to call your API from a user's browser. This matters when `allow_credentials=True` is set — combined with a wildcard, any malicious site can make authenticated requests on behalf of your users. Explicitly listing origins is the safe default.

**Production note:** In production, replace `localhost:8501` with your actual Streamlit domain (e.g., `https://support.yourcompany.com`). The `allow_methods=["GET", "POST"]` restriction also prevents preflight-bypass attacks that exploit less-common HTTP verbs.

### 2.4 The Request Size Limit Middleware

```python
MAX_REQUEST_BODY_BYTES = 1 * 1024 * 1024  # 1 MB

class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={"error": "Request body too large. Maximum size is 1 MB."},
            )
        return await call_next(request)

app.add_middleware(RequestSizeLimitMiddleware)
```

This middleware guards against **payload-based DoS attacks** — an attacker sending a 100 MB request body to make your server spend CPU parsing it. Note the important limitation: it relies on the `Content-Length` header. An attacker could omit that header and stream a large body incrementally. A production system would also limit the request stream size using Starlette's `LimitUploadSize` middleware or a reverse proxy (nginx, Caddy) with body size limits.

**Middleware execution order in FastAPI/Starlette:** Middleware is applied in **reverse addition order** for incoming requests. `RequestSizeLimitMiddleware` is added before `CORSMiddleware`, so the execution order is:

```
Incoming request
    │
    ▼
CORSMiddleware (outer)
    │
    ▼
RequestSizeLimitMiddleware (inner)
    │
    ▼
Route handler (/api/v1/chat)
```

The size check happens before the route handler processes the body, which is the correct placement.

### 2.5 The `/api/v1/chat` Route — Async All the Way Down

```python
@app.post("/api/v1/chat", response_model=ChatResponse, ...)
async def chat(request: ChatRequest) -> ChatResponse:
    result = await _graph.ainvoke(
        {"user_query": request.query},
        config={"configurable": {"thread_id": request.conversation_id}},
    )
```

**Why `ainvoke` not `invoke`?** FastAPI runs on an async event loop (via `uvicorn`). If you call the synchronous `invoke()` inside an `async def` handler, you block the event loop — no other requests can be processed while this one is waiting for the LLM to respond. `ainvoke()` yields control back to the event loop during each `await` point (each LLM or DB call), so the server can handle many concurrent requests.

**`thread_id` for multi-turn state:** LangGraph's in-memory `MemorySaver` checkpointer uses `thread_id` as the key to persist state between invocations. When Streamlit sends the same `conversation_id` on every request in a session, the graph "remembers" the previous messages and can answer follow-up questions with full context:

```
Request 1: {"user_query": "Where is ORD-1001?", "thread_id": "session-abc"}
    → Graph state saved under key "session-abc"
    → Response: "Your order is Shipped via UPS"

Request 2: {"user_query": "And when will it arrive?", "thread_id": "session-abc"}
    → Graph loads state from key "session-abc" (messages history included)
    → The LLM knows "it" refers to ORD-1001 from the previous turn
    → Response: "It is expected to arrive by Friday"
```

---

## 3. The Pydantic V2 Schemas (`api/schema.py`)

### 3.1 The Internal/External Boundary

`AgentState` has many fields:

```python
class AgentState(TypedDict):
    user_query: str
    messages: List[BaseMessage]
    intent: str
    confidence: float
    routing_rationale: str
    missing_info: List[str]
    clarification_turn_count: int
    policy_context: Optional[str]
    sql_query: Optional[str]
    sql_result: Optional[Any]
    sql_error: Optional[str]
    web_results: Optional[List[dict]]
    web_context: Optional[str]
    fallback_reason: Optional[str]
    handoff_context: Optional[dict]
    final_response: Optional[str]
    ...
```

The API response exposes only four of these to external clients:

```python
class ChatResponse(BaseModel):
    response: str
    intent: str
    confidence: float
    correlation_id: str
```

This is the **internal/external boundary**. Exposing `sql_query`, `routing_rationale`, or `handoff_context` to clients would:
- Reveal internal implementation details (what database you use, how routing works)
- Create API surface area that breaks if you rename internal fields
- Potentially expose security-relevant information (actual SQL strings)

**The rule:** Pydantic schemas define your public API contract. `AgentState` is your private implementation detail. They evolve independently.

### 3.2 Field Validation as Documentation

```python
class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,       # Prevents empty string submissions
        max_length=2000,    # Prevents extremely long queries
        description="The user's natural language query.",
        examples=["Where is my order ORD-1001?"],
    )
    conversation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Session ID for multi-turn conversation continuity...",
    )
```

`Field(...)` (using `...` as the first argument) marks a field as **required** — FastAPI will return HTTP 422 if `query` is missing or fails validation. The `description` and `examples` are surfaced in the auto-generated OpenAPI docs at `http://localhost:8000/docs`. This means the schema is also the API documentation — there is no separate doc to maintain.

**`default_factory=lambda: str(uuid.uuid4())`** generates a fresh UUID per request. `default_factory` is used instead of `default=str(uuid.uuid4())` because the latter would evaluate `uuid.uuid4()` once at class definition time — giving the same UUID to every request that does not supply a `conversation_id`.

```python
# ❌ Evaluated once at class definition — all requests share the same ID!
conversation_id: str = Field(default=str(uuid.uuid4()))

# ✅ Factory called per request — each request gets a unique ID
conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
```

---

## 4. The Streamlit Frontend (`frontend/app.py`)

### 4.1 How Streamlit's Execution Model Works

Streamlit's execution model is unusual compared to React or Vue. **Your entire Python script reruns from top to bottom on every user interaction.** This is the core mental model you must internalize:

```
User opens browser → script runs top-to-bottom (initial render)
User types message → script runs top-to-bottom (re-render)
User clicks button → script runs top-to-bottom (re-render)
```

This means any variable you define at the top level is recreated on every render:

```python
# This recreates a new UUID on every rerender!
session_id = str(uuid.uuid4())

# This persists across rerenders (stored in browser session)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
```

Without this `if not in` guard, the user gets a different `session_id` on every message — breaking multi-turn conversation continuity with the backend.

### 4.2 Session State — Persisting Data Across Rerenders

`st.session_state` is a dictionary-like object that survives rerenders within a single browser session:

```python
# Initialise once — guard against overwriting on rerenders
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
```

`st.session_state` is scoped to one browser tab. If the user opens a second tab, it gets its own independent session state (and a different `session_id`). If the user refreshes the page, session state is wiped.

The conversation history is stored as a list of dicts:

```python
st.session_state.messages = [
    {"role": "user",      "content": "Where is ORD-1001?"},
    {"role": "assistant", "content": "Your order is Shipped...",
                          "metadata": {"intent": "sql", "confidence": 0.99, "correlation_id": "abc..."}},
    ...
]
```

This is Streamlit's conventional format for chat history. The `metadata` key is custom — standard Streamlit chat does not define it, but we use it to store routing information so it can be displayed in the routing badge.

### 4.3 The API Bridge — Connecting to FastAPI

```python
API_URL = "http://localhost:8000/api/v1/chat"
API_TIMEOUT_SECONDS = 30

if prompt := st.chat_input("Ask Omni-Help..."):
    try:
        api_response = requests.post(
            API_URL,
            json={
                "query": prompt,
                "conversation_id": st.session_state.session_id,
            },
            timeout=API_TIMEOUT_SECONDS,
        )
        api_response.raise_for_status()
        data = api_response.json()
        ai_text = data.get("response", "No response received.")
        intent = data.get("intent", "unknown")
        confidence = data.get("confidence", 0.0)
        correlation_id = data.get("correlation_id", "")

    except requests.exceptions.ConnectionError:
        ai_text = "⚠️ **Cannot connect to the Omni-Help backend.** ..."
        intent, confidence, correlation_id = "unknown", 0.0, ""

    except requests.exceptions.Timeout:
        ai_text = "⚠️ **Request timed out.** ..."
        intent, confidence, correlation_id = "unknown", 0.0, ""

    except Exception as exc:
        ai_text = f"⚠️ **Unexpected error:** {exc}"
        intent, confidence, correlation_id = "unknown", 0.0, ""
```

**Three error cases are handled explicitly:**

1. **`ConnectionError`** — The FastAPI server is not running. Rather than showing a cryptic stack trace, the UI displays the exact command to start the server. This is developer-friendly UX.

2. **`Timeout`** — The graph took longer than 30 seconds. Causes: an LLM call with a very long context, a slow Tavily search. The user is told to retry.

3. **Generic `Exception`** — Any other error (HTTP 500 from FastAPI, JSON parsing failure, etc.) is caught with a generic message that includes the exception text — useful for debugging without crashing the UI.

**`api_response.raise_for_status()`** is important. Without it, a `200 OK` and a `500 Internal Server Error` would both silently fall through to `data = api_response.json()`. Calling `raise_for_status()` converts 4xx/5xx responses into Python exceptions that are caught by the `except Exception` block.

### 4.4 The Routing Metadata Badge

This is the "architect flex" feature — every AI response shows which pipeline handled it:

```python
INTENT_DISPLAY = {
    "policy":       ("📋 Policy", "#4A90D9"),
    "sql":          ("🗄️ Order DB", "#27AE60"),
    "web":          ("🌐 Web Search", "#8E44AD"),
    "complaint":    ("🚨 Escalated", "#E74C3C"),
    ...
}

if intent != "unknown":
    label, color = INTENT_DISPLAY.get(intent, INTENT_DISPLAY["unknown"])
    confidence_pct = confidence * 100
    st.caption(
        f"Routed to: **{label}** &nbsp;·&nbsp; "
        f"Confidence: **{confidence_pct:.1f}%** &nbsp;·&nbsp; "
        f"ID: `{correlation_id[:8]}...`"
    )
```

Output:
```
Routed to: 🗄️ Order DB  ·  Confidence: 99.0%  ·  ID: abc12345...
```

This serves two purposes:
1. **Demo quality:** In a product demo or job interview, this badge immediately shows the interviewer that the system has intelligent routing — not just a single LLM call. It makes the architecture visible.
2. **Debugging:** When a response looks wrong, the badge tells you immediately which pipeline produced it and whether the router was confident. A policy question accidentally routed to the SQL pipeline at 55% confidence is instantly diagnosable.

### 4.5 Multi-Turn Conversation via `conversation_id`

The full multi-turn loop:

```
Browser (Streamlit)                    Server (FastAPI + LangGraph)
─────────────────────────────────────────────────────────────────

1. User types "Where is ORD-1001?"
   session_id = "sess-abc"
                │
                │  POST /api/v1/chat
                │  {"query": "Where is ORD-1001?", "conversation_id": "sess-abc"}
                │
                ▼
                         2. graph.ainvoke({"user_query": "..."}, thread_id="sess-abc")
                            → Routes to SQL Node → Synthesis
                            → State saved to MemorySaver["sess-abc"]
                            → Returns {"final_response": "Your order is Shipped..."}
                │
                │  {"response": "Your order is Shipped via UPS.",
                │   "intent": "sql", "confidence": 0.99, "correlation_id": "corr-xyz"}
                │
                ▼
3. Streamlit appends to st.session_state.messages
   Shows badge: "🗄️ Order DB · 99.0%"

4. User types "When does it arrive?"
   session_id = "sess-abc"  ← same session
                │
                │  POST /api/v1/chat
                │  {"query": "When does it arrive?", "conversation_id": "sess-abc"}
                │
                ▼
                         5. graph.ainvoke({"user_query": "..."}, thread_id="sess-abc")
                            → MemorySaver loads state for "sess-abc"
                            → messages includes "Where is ORD-1001?" + previous AI response
                            → LLM knows "it" = ORD-1001 from context
                            → Returns "It is expected to arrive by Friday, Jan 17"
```

The key: `st.session_state.session_id` is generated once when the browser tab opens and never changes until the user clicks "Clear conversation." Every request sends the same `conversation_id`, so LangGraph's `MemorySaver` correctly threads the conversation.

---

## 5. Enterprise Patterns vs. Amateur Code

### Pattern 1: Lifespan vs. Per-Request Initialization

```python
# ❌ Amateur: graph compiled on every request
@app.post("/api/v1/chat")
async def chat(request: ChatRequest):
    graph = build_graph()   # Expensive! Runs every time
    result = await graph.ainvoke(...)

# ✅ Enterprise: graph compiled once at startup
_graph = None
@asynccontextmanager
async def lifespan(app):
    global _graph
    _graph = build_graph()  # Runs once
    yield

@app.post("/api/v1/chat")
async def chat(request: ChatRequest):
    result = await _graph.ainvoke(...)  # Reuses compiled graph
```

### Pattern 2: Pydantic Schemas as API Contracts

```python
# ❌ Amateur: return the raw state dict
@app.post("/api/v1/chat")
async def chat(request):
    body = await request.json()
    result = graph.invoke({"user_query": body["query"]})
    return result   # Exposes all internal state fields to the client

# ✅ Enterprise: shape the response explicitly
@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    result = await _graph.ainvoke(...)
    return ChatResponse(
        response=result["final_response"],
        intent=result["intent"],
        confidence=result["confidence"],
        correlation_id=correlation_id,
    )
```

### Pattern 3: Explicit Error Handling in the UI

```python
# ❌ Amateur: crash on connection error
ai_text = requests.post(API_URL, json={...}).json()["response"]

# ✅ Enterprise: three explicit failure cases with user-friendly messages
try:
    ...
except requests.exceptions.ConnectionError:
    ai_text = "⚠️ Cannot connect to the backend. Start the server with: uvicorn ..."
except requests.exceptions.Timeout:
    ai_text = "⚠️ Request timed out. Please try again."
except Exception as exc:
    ai_text = f"⚠️ Unexpected error: {exc}"
```

---

## 6. Common Misconceptions

### "I should use `async def` everywhere in Streamlit"

Streamlit is **not async**. It runs in a synchronous context. The `requests` library (synchronous) is the correct choice here, not `httpx` with `await`. If you try to `await` in a Streamlit callback, you will get a `RuntimeError: no running event loop`. Use `requests` in Streamlit; use `httpx` or `aiohttp` in async FastAPI routes if you need async HTTP calls.

### "session_state persists after a page refresh"

It does not. `st.session_state` is stored in the browser's memory, tied to the WebSocket connection between the browser and the Streamlit server. A page refresh destroys the WebSocket and clears session state. For truly persistent sessions (across refreshes or browser tabs), you would need to store the `session_id` in a browser cookie or `localStorage` and pass it explicitly on each request. For a POC, the current behavior (new session on refresh) is acceptable.

### "The FastAPI server handles Streamlit's rendering"

No — FastAPI only handles the `/api/v1/chat` and `/api/v1/health` routes. Streamlit has its own server (on port 8501) that handles rendering, WebSocket communication, and widget state. The two servers are completely independent; FastAPI has no knowledge of the Streamlit UI.

### "I should put my OpenAI key in the FastAPI request headers"

Never. API keys belong in the server's environment (`.env` file loaded at startup), not in client requests. If Streamlit sends the API key in request headers, any user of the app who inspects their browser network traffic can steal it. The server holds the credentials; the client sends only the user's query.

### "The `conversation_id` in the response is the session ID"

No — they are different identifiers with different scopes:

| ID | Scope | Lifetime | Purpose |
|---|---|---|---|
| `conversation_id` | Session | Until user clears chat or refreshes | Groups all turns of a conversation in LangGraph's MemorySaver |
| `correlation_id` | Single request | One HTTP request/response cycle | Links logs, LangSmith traces, and error reports for that specific request |

A single `conversation_id` spans many `correlation_id` values — one per turn.
