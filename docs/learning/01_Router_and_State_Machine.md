# Omni-Help Deep Dive: The Router & State Machine

**Learning Module 01 — Phases 1–3**  
**Audience:** Intermediate Python developer moving into production AI systems  
**Files covered:** `state.py` · `router.py` · `nodes.py` · `graph.py`

---

## Table of Contents

1. [The Conceptual Mental Model](#1-the-conceptual-mental-model)
2. [The File Anatomy — How They Connect](#2-the-file-anatomy--how-they-connect)
3. [Line-by-Line Code Walkthroughs](#3-line-by-line-code-walkthroughs)
   - 3.1 `AgentState` — The Contract
   - 3.2 `RouterOutput` + `router_agent` — Structured Output
   - 3.3 `route_decision` — The Traffic Controller
   - 3.4 `clarification_decision` — The Circuit Breaker
   - 3.5 `build_graph` — Assembling the Machine
4. [Enterprise Best Practices vs. Amateur Code](#4-enterprise-best-practices-vs-amateur-code)
5. [Common Misconceptions](#5-common-misconceptions)

---

## 1. The Conceptual Mental Model

### What is a LangGraph State Machine?

A **State Machine** is a system that exists in exactly one *state* at a time and transitions between states according to defined rules. Traffic lights are a classic example: RED → GREEN → YELLOW → RED. Each state is discrete; the rules (transitions) are explicit.

LangGraph applies this concept to AI agents. Instead of traffic light colors, the "states" are processing *nodes* (Router, Retriever, SQL Agent, etc.). Instead of timers, the *transitions* are decided by functions that read the current data.

The key object is `AgentState` — a **typed dictionary** that travels through every node. Think of it as a clipboard being passed down an assembly line. Each station reads what it needs from the clipboard, writes its results back onto it, and passes it to the next station. The clipboard is never duplicated; it is always the same object, updated in place (via partial merges).

```
                    ┌─────────────────────────────────────────┐
                    │             AgentState (clipboard)       │
                    │  user_query, intent, confidence,         │
                    │  routing_rationale, missing_info, ...    │
                    └─────────────────────────────────────────┘
                                      │
                              passes through
                                      │
              ┌───────────┬───────────┼───────────┬───────────┐
              ▼           ▼           ▼           ▼           ▼
           Router    Retriever      SQL          Web      Fallback
           (Brain)   (Policy)    (Orders)    (Search)   (Human)
```

### Why a Cyclic Graph, Not a Linear Chain?

Most tutorials show a **linear chain**: Input → A → B → C → Output. This is simple but brittle for real-world support systems. Consider:

| Scenario | Linear Chain Behavior | Cyclic Graph Behavior |
|---|---|---|
| Ambiguous query ("help me") | Fails silently or picks wrong pipeline | Routes to Clarification, asks user, retries |
| SQL query after clarification | Cannot loop back to re-classify | Returns to Router with enriched context |
| Repeated low confidence | No escape hatch — infinite loops possible | Circuit breaker: max 2 turns, then fallback |

A cyclic graph encodes **self-correction** as a first-class architectural feature. The system can ask "did I understand correctly?" before committing to expensive downstream operations.

Our cycle is:

```
        ┌──────────────────────────────────┐
        │              CYCLE               │
        ▼                                  │
    [Router] ──confidence < 0.7──▶ [Clarification]
        │                                  │
        │                    turn_count < 2┘
        │                    turn_count >= 2 ──▶ [Fallback]
        │
        ├──policy──▶ [Retriever] ──▶ END
        ├──sql──▶    [SQL]       ──▶ END
        ├──web──▶    [Web]       ──▶ END
        └──complaint──▶ [Fallback] ──▶ END
```

The graph is **not a flowchart you drew on a whiteboard.** It is a formally compiled object that LangGraph validates, executes with checkpointing, and can stream token-by-token. That distinction matters enormously for production.

---

## 2. The File Anatomy — How They Connect

Four files, one direction of dependency:

```
graph/state.py          ← defines AgentState (no imports from our codebase)
       ▲
agents/router.py        ← imports AgentState; owns LLM call + RouterOutput
       ▲
graph/nodes.py          ← imports AgentState + router_agent; wraps them as nodes
       ▲
graph/graph.py          ← imports AgentState + all nodes; builds + compiles the graph
```

This is a **strict one-way dependency chain.** `state.py` knows nothing about the LLM. `router.py` knows nothing about the graph topology. `graph.py` is the only file that knows the full wiring. This is intentional — it is the **Single Responsibility Principle** applied at the file level.

### The Exact Data Flow for a Single Query

Here is what happens, step by step, when a query like *"Where is my order #12345?"* hits the system:

**Step 0 — Invocation**
```python
result = graph.invoke({"user_query": "Where is my order #12345?"})
```
LangGraph initializes `AgentState` with the provided dict. All other keys default to their `total=False` TypedDict defaults (i.e., they are absent until a node writes them).

**Step 1 — Router Node** (`nodes.py` → `router.py`)
`router_node(state)` is called. It delegates immediately to `router_agent(state)`. Inside `router_agent`:
- `state.get("user_query")` extracts the query string.
- The query is wrapped in a `HumanMessage` alongside a `SystemMessage` containing the full classification prompt.
- `_structured_llm.invoke(messages)` sends both to GPT-4o-mini and receives back a validated `RouterOutput` object (not a raw string — Pydantic has already parsed and validated it).
- The function returns a **partial dict**: `{"intent": "sql", "confidence": 0.99, "routing_rationale": "...", "missing_info": []}`.

LangGraph **merges** this partial dict into the existing `AgentState`. Only the keys present in the return dict are updated. Other keys are untouched.

**Step 2 — Conditional Edge** (`graph.py`, `route_decision`)
LangGraph calls `route_decision(state)` with the *updated* state. This function reads `intent="sql"` and `confidence=0.99`. Since confidence ≥ 0.7 and intent is "sql", it returns the string `"sql"`. LangGraph uses this string to look up the next node in the edge map and dispatches to `sql_node`.

**Step 3 — SQL Node** (`nodes.py`)
`sql_node(state)` runs. Currently a stub — it prints a message and returns `{"final_response": "..."}`. This is merged into `AgentState`.

**Step 4 — Static Edge → END**
`sql_node` has a static edge to `END`. LangGraph terminates execution and returns the final `AgentState` to the caller.

**Contrast: the Clarification Path for "Hm, I'm not sure..."**

Step 1: `router_agent` returns `{"intent": "policy", "confidence": 0.4, "missing_info": ["low_confidence"]}`.  
Step 2: `route_decision` reads `confidence=0.4 < 0.7` → returns `"clarification"`.  
Step 3: `clarification_node` increments `clarification_turn_count` to 1, clears `missing_info`, returns a clarifying question.  
Step 4: `clarification_decision` reads `turn_count=1 < 2` → returns `"router"`. **The cycle executes.**  
Step 5: `router_node` runs again with the same `user_query` (in a real system, the user's answer would update this). Router returns `confidence=0.4` again.  
Step 6: `route_decision` → `"clarification"` again. `clarification_node` increments `turn_count` to 2.  
Step 7: `clarification_decision` reads `turn_count=2 >= 2` → returns `"fallback"`. Circuit breaker fires.  
Step 8: `fallback_node` → `END`.

---

## 3. Line-by-Line Code Walkthroughs

### 3.1 `AgentState` — The Contract (`state.py`)

```python
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
```

**Why these specific imports?**
- `TypedDict` — makes `AgentState` a typed dictionary, not a class. LangGraph requires a TypedDict (or Pydantic model) as the state type so it can introspect field names at compile time.
- `Annotated` — Python's mechanism for attaching metadata to a type hint. Here, it is used to attach a *reducer function* to the `messages` field.
- `add_messages` — LangGraph's built-in reducer. Without it, every node that returns `messages` would *replace* the entire list. With it, returned messages are *appended*. This is the difference between a chatbot that remembers history and one that forgets after every turn.
- `BaseMessage` — the abstract base class for all LangChain message types (`HumanMessage`, `AIMessage`, `SystemMessage`). Using the base class keeps the type flexible.

```python
IntentType = Literal[
    "policy", "sql", "web", "product_info", "complaint",
]
```

**Why a `Literal` type alias?**  
This makes `IntentType` a closed enum of strings. If a node writes `intent = "orders"` (a typo), a type checker (mypy, pyright, or the Cursor IDE) flags it *before runtime*. At the enterprise scale, catching errors before deployment is worth more than any runtime check.

```python
class AgentState(TypedDict, total=False):
```

**Why `total=False`?**  
`total=False` means no key is required to be present when the dict is created. This is critical because `AgentState` accumulates values progressively — the SQL result only exists after the SQL node runs. If `total=True` (the default), you would have to provide every key upfront, which defeats the incremental-build design. Each node only writes the keys it owns.

```python
messages: Annotated[List[BaseMessage], add_messages]
```

**The reducer pattern.** This single line encodes a significant design decision: conversation history is *append-only*. When any node returns `{"messages": [new_message]}`, LangGraph calls `add_messages(existing_list, [new_message])` and stores the merged result. The node never needs to read the current list — it just returns what it wants to add.

```python
confidence: float  # [0, 1]; Confidence Gate: if < 0.7 → Clarification Node
```

**Why store confidence in state?**  
The confidence score is written by `router_agent` and read by `route_decision` — two different functions in two different files. The state is the **contract between them**. If you stored confidence as a local variable inside `router_agent`, `route_decision` would never see it. State is the only communication channel between nodes in LangGraph.

```python
clarification_turn_count: int  # incremented by clarification_node; max 2 turns
```

**The cycle guard as a first-class state citizen.** This key exists for one reason: to prevent an infinite loop. In a cyclic graph, you must explicitly encode the termination condition in state. Infinite loops in AI systems have caused real production outages. This one line is the difference between a self-healing system and a runaway loop consuming API credits.

---

### 3.2 `RouterOutput` + `router_agent` — Structured Output (`router.py`)

```python
load_dotenv()
```

**Line 20 — and why it must be here.** The next two lines instantiate `ChatOpenAI`, which reads `OPENAI_API_KEY` from the environment *at import time* (when the module is first loaded). `load_dotenv()` must be called before that instantiation. If it were called inside `router_agent()`, the environment would be loaded on every request — wasteful. If it were called after the `ChatOpenAI` lines — the key would already be missing. Ordering matters.

```python
class RouterOutput(BaseModel):
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
```

**This class is the schema contract with the LLM.** `ge=0.0, le=1.0` are Pydantic validators that enforce `0 ≤ confidence ≤ 1`. If the LLM returns `confidence: 1.5` (a hallucination), Pydantic raises a `ValidationError` — the request fails fast with a clear error rather than silently corrupting downstream logic. The `description` fields are passed to OpenAI as part of the schema, giving the model explicit guidance on what each field means.

```python
_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
_structured_llm = _llm.with_structured_output(RouterOutput)
```

**Module-level singletons — a critical performance pattern.** The `_` prefix signals these are private to the module (Python convention). They are created once when the module is first imported and reused for every request. Creating a new `ChatOpenAI` client on every call would re-initialize the HTTP connection pool on every request. At 1,000 concurrent users, that is a guaranteed bottleneck.

`temperature=0` sets the LLM to its most deterministic mode. For a classifier, randomness is the enemy — you want the same query to always produce the same intent. Creativity is for the synthesis nodes, not the router.

`.with_structured_output(RouterOutput)` is the critical line. It instructs LangChain to:
1. Convert `RouterOutput`'s schema into a JSON schema.
2. Pass that schema to the OpenAI API's function-calling mechanism.
3. Receive the response as structured JSON.
4. Instantiate and validate a `RouterOutput` object automatically.

You never write `json.loads()`. You never write `response["intent"]`. The output is already a typed Python object.

```python
def router_agent(state: AgentState) -> dict:
```

**Why return `dict`, not `AgentState`?**  
Nodes return *partial* state updates. If you returned a full `AgentState`, you would have to populate every field — including ones you know nothing about. Returning a plain dict with only the keys you changed is the LangGraph contract for partial updates.

```python
user_query: str = state.get("user_query", "")
if not user_query:
    for msg in reversed(state.get("messages", [])):
        if hasattr(msg, "type") and msg.type == "human":
            user_query = msg.content
            break
```

**Two-layer defensive extraction.** First, try the explicit `user_query` key (fast path, used by the API and evaluation harness). If absent, fall back to scanning the message history in reverse (most recent message first) for a `HumanMessage`. This makes the router resilient to two different invocation patterns — it works whether the caller sets `user_query` directly or relies on conversation history. Amateur code would crash with a `KeyError` on the first missing key.

```python
if not user_query:
    raise ValueError("AgentState must contain 'user_query' or at least one HumanMessage.")
```

**Fail loudly, fail early.** If both extraction paths fail, raising `ValueError` with a descriptive message is far better than passing an empty string to the LLM and getting a nonsense classification. LangGraph will surface this as a node error with the full traceback. In production, this maps to a 422 Unprocessable Entity response to the API caller.

```python
messages = [
    SystemMessage(content=standard_router_prompt),
    HumanMessage(content=user_query),
]
output: RouterOutput = _structured_llm.invoke(messages)
```

**The LLM call pattern.** The system prompt (`standard_router_prompt`) provides the classification rules. The human message is the raw query. This two-message structure is the OpenAI chat format — system sets the rules, human asks the question. The type annotation `output: RouterOutput` is redundant at runtime (Python ignores it) but essential for IDE autocompletion and type checking. After this line, `output.intent`, `output.confidence`, and `output.reasoning` are fully typed and validated Python attributes.

```python
missing_info = [] if output.confidence >= 0.7 else ["low_confidence"]
```

**The Confidence Gate — one line, two paths.** This ternary encodes the architectural decision from the blueprint: high confidence means the router is certain, so `missing_info` stays empty and the graph routes normally. Low confidence flags the query for clarification by adding `"low_confidence"` to `missing_info`. This is then read by `route_decision` in `graph.py` — two separate files coordinating through state.

```python
return {
    "intent": output.intent,
    "confidence": output.confidence,
    "routing_rationale": output.reasoning,
    "missing_info": missing_info,
}
```

**Field name translation.** Notice `output.reasoning` is stored as `routing_rationale`. The LLM field is named `reasoning` (short, natural for the prompt). The state field is named `routing_rationale` (verbose, self-documenting in logs). This translation layer separates the LLM's vocabulary from the system's internal vocabulary — a subtle but important boundary.

---

### 3.3 `route_decision` — The Traffic Controller (`graph.py`)

```python
def route_decision(state: AgentState) -> str:
```

**This function is not a node.** It is a *conditional edge function* — it is never registered with `add_node`. LangGraph calls it after `router_node` completes. Its sole job is to return a string that names the next node. It has no side effects, writes nothing to state, and is trivially unit-testable.

```python
intent: str = state.get("intent", "")
confidence: float = state.get("confidence", 0.0)
```

**Explicit type annotations on local variables.** `state.get()` returns `Any` because TypedDict's `get()` method cannot be narrowed by type checkers at runtime. The explicit annotations (`intent: str`, `confidence: float`) tell your IDE and type checker what type you expect, enabling autocompletion and catching `if confidence > "high"` style bugs before they run.

```python
if intent == "complaint":
    return "fallback"

if confidence < CONFIDENCE_THRESHOLD:
    return "clarification"
```

**Priority order is the architecture.** These two checks are in a specific order for a specific reason. A complaint with high confidence still goes to fallback — the intent overrides the confidence. The rules are evaluated top-to-bottom, and the first match wins. Changing the order changes the system's behavior. This is why the docstring explicitly labels the rules "in priority order."

Using a named constant `CONFIDENCE_THRESHOLD = 0.7` instead of a bare `0.7` is not stylistic — it means the threshold appears in exactly one place in the codebase. To tune it, you change one line. To log it, you print the constant. To make it configurable, you replace the constant with `settings.CONFIDENCE_THRESHOLD`.

```python
route_map = {
    "policy": "retriever",
    "sql": "sql",
    "web": "web",
    "product_info": "product_info",
}
return route_map.get(intent, "fallback")
```

**A dict dispatch over an if-elif chain.** You could write this as:
```python
# Amateur version
if intent == "policy":
    return "retriever"
elif intent == "sql":
    return "sql"
elif intent == "web":
    return "web"
...
```

The dict version is superior for three reasons:
1. **Adding a new intent** requires one dict entry, not a new `elif` branch in the middle of the function.
2. **The default fallback** is handled by `.get(intent, "fallback")` — if an unknown intent somehow appears, it goes to fallback safely. The `elif` chain would fall through to `None`, which would cause a LangGraph routing error.
3. **Readability** — the mapping is a data structure, not control flow. You can read it at a glance.

---

### 3.4 `clarification_decision` — The Circuit Breaker (`graph.py`)

```python
def clarification_decision(state: AgentState) -> str:
    turn_count: int = state.get("clarification_turn_count", 0)
    if turn_count >= MAX_CLARIFICATION_TURNS:
        print(f"[Graph] Max clarification turns ({MAX_CLARIFICATION_TURNS}) reached — escalating to fallback.")
        return "fallback"
    return "router"
```

**This is the circuit breaker pattern.** In electrical engineering, a circuit breaker stops current from flowing if a fault is detected. Here, it stops the Router → Clarification loop from running forever.

`state.get("clarification_turn_count", 0)` — the default of `0` is critical. On the first invocation, this key doesn't exist in state yet. `.get()` with a default prevents a `KeyError` and treats absence as "zero turns have occurred."

`turn_count >= MAX_CLARIFICATION_TURNS` — using `>=` instead of `==` is defensive. If a bug caused `turn_count` to jump from 0 to 3 in a single step, `==` would miss it and allow a third clarification cycle. `>=` catches any value at or above the limit.

The `print` statement is temporary — in Phase 7, this becomes a structured log event sent to LangSmith with a correlation ID and the full state context. The pattern (detect condition → log → escalate) is preserved; only the transport changes.

---

### 3.5 `build_graph` — Assembling the Machine (`graph.py`)

```python
builder = StateGraph(AgentState)
```

**`StateGraph(AgentState)` is a type-safe contract.** By passing `AgentState` as the type parameter, LangGraph knows the shape of data flowing through the graph. If a node returns a key that doesn't exist in `AgentState`, the framework warns you at runtime. This is the equivalent of a database schema — the graph is schema-enforced.

```python
builder.add_node("router", router_node)
```

**String name + function reference.** The string `"router"` is the name used in all edge definitions and in the execution log. The function `router_node` is what actually runs. Separating them means you can rename a node without changing its implementation, or swap implementations without changing the topology.

```python
builder.add_conditional_edges(
    "router",
    route_decision,
    {
        "retriever": "retriever",
        "sql": "sql",
        ...
    },
)
```

**Three-argument conditional edge.** The third argument (the dict) is the *routing map*. `route_decision` returns a string like `"sql"`. LangGraph looks up `"sql"` in this dict and routes to the node named `"sql"`. The dict makes the valid return values explicit at graph-compile time — if `route_decision` returns `"postgres"` (not in the dict), LangGraph raises an error when `.compile()` is called, not when the first request arrives. Early detection, again.

```python
builder.add_conditional_edges(
    "clarification",
    clarification_decision,
    {"router": "router", "fallback": "fallback"},
)
```

**This line closes the cycle.** Before this line, `clarification` had a static edge to `END`. After this line, `clarification` conditionally returns to `router`. This is what makes the graph cyclic. LangGraph validates that this cycle has a reachable termination path (via `"fallback"` → `END`) during `.compile()`.

```python
for node in ("retriever", "sql", "web", "product_info", "fallback"):
    builder.add_edge(node, END)
```

**Programmatic edge registration.** Adding edges in a loop is cleaner than five identical `add_edge` calls. Notice `clarification` is absent from this list — it has a conditional edge instead of a static one to `END`. This is the exact change made in Phase 3; previously, `clarification` was in this loop.

```python
return builder.compile()
```

**`.compile()` is a validation step, not just a build step.** LangGraph checks: are all nodes reachable? Does every conditional edge's return value appear in its routing map? Are there cycles without exit conditions? If any check fails, you get a `GraphValidationError` — before any user traffic hits the system.

```python
graph = build_graph()
```

**Module-level compilation.** The graph is compiled once when the module is imported. Every invocation of `graph.invoke()` reuses the same compiled graph object. Like the LLM client, this is a singleton by design. Creating a new graph per request would recompile, re-validate, and re-allocate on every call.

---

## 4. Enterprise Best Practices vs. Amateur Code

### Practices in This Codebase That Are Production-Grade

| Code | Why It's Enterprise-Grade |
|---|---|
| `RouterOutput(BaseModel)` with `ge=0.0, le=1.0` | LLM output is validated by Pydantic before it touches state. A hallucinated `confidence: 99` is rejected with a clear error. |
| `_llm = ChatOpenAI(...)` at module level (singleton) | Connection pool created once; not recreated per request. Scales to 1,000 concurrent users without resource exhaustion. |
| `load_dotenv()` before module-level LLM init | Guarantees env vars are loaded before the client that needs them. Prevents the "works locally, crashes on import in production" failure class. |
| `total=False` on `AgentState` | Nodes write only the keys they own. No node is forced to populate fields it doesn't control. This is the **Open/Closed Principle**: new nodes can be added without modifying `AgentState`. |
| `CONFIDENCE_THRESHOLD = 0.7` as a named constant | One source of truth. Tune, log, or configure from one location. Magic numbers are a maintenance time bomb. |
| `route_map.get(intent, "fallback")` | Explicit default. Unknown intents are handled gracefully, not allowed to crash the routing layer. |
| `clarification_turn_count` in state | Cycle guard is encoded in state, not in a mutable global or a local variable. It survives checkpointing, serialization, and graph resume. |
| `raise ValueError("...")` with descriptive message | Fail loudly at the boundary. The error is caught by LangGraph and surfaced to the caller with full context. Silent failures are far more dangerous. |
| Two-layer query extraction in `router_agent` | Defensive design: works whether caller provides `user_query` directly or relies on conversation history. Handles both API and evaluation harness invocation patterns. |
| `builder.compile()` at startup | Graph validation happens at boot time, not request time. Bad topology is caught before any user is affected. |

### What Amateur Code Looks Like in Contrast

```python
# ❌ Amateur: magic numbers, no type safety, crash-prone
def route(state):
    if state["intent"] == "policy":      # KeyError if intent not set
        if state["confidence"] > 0.7:   # magic number, no name
            return "retriever"
    elif state["intent"] == "sql":
        return "sql"
    # ❌ no default — returns None, causes LangGraph routing error

# ❌ Amateur: LLM client created per request, JSON parsed manually
def router_agent(state):
    llm = ChatOpenAI(model="gpt-4o-mini")   # new connection pool every call
    response = llm.invoke(messages)
    data = json.loads(response.content)      # crashes on any non-JSON response
    intent = data["intent"]                  # KeyError if LLM omits the field
    return {"intent": intent}               # no confidence, no rationale logged
```

The difference is not verbosity — it is **failure surface area**. Enterprise code minimizes the number of ways things can go wrong silently.

---

## 5. Common Misconceptions

**"The router is the entire agent."**  
The router is one node. It classifies intent. It does not retrieve documents, query databases, or synthesize responses. Its entire job is to answer: *"What kind of question is this?"* The power comes from what the graph does with that answer.

**"Nodes call each other."**  
Nodes never call each other. They return a partial state dict and stop. LangGraph reads the dict, merges it into `AgentState`, evaluates the appropriate edge function, and calls the next node. The orchestration layer is fully managed by the framework — nodes are stateless functions.

**"The cycle means the graph runs forever."**  
The cycle is bounded. `clarification_turn_count` can only reach `MAX_CLARIFICATION_TURNS` (2) before `clarification_decision` returns `"fallback"` and the graph exits cleanly. Every cycle must have a reachable `END` — LangGraph's `.compile()` enforces this.

**"Structured output just means parsing JSON."**  
It means the LLM is constrained by the API to return a JSON object that conforms to the `RouterOutput` schema — validated server-side by OpenAI before the response is even sent back. Pydantic then validates it again on the Python side. Two layers of validation for the boundary where the most errors originate: the LLM output.

**"The state is passed between files."**  
The state is passed between *node executions* by LangGraph's runtime. Your code never passes it directly. `router_agent` returns a dict; `route_decision` receives the merged state as its argument. The framework is the intermediary. This is why you cannot call `router_agent(state)` directly and expect `route_decision` to see the update — you must invoke through the graph.

---

*Document version: 1.0 — Phase 3 complete. Next: `02_RAG_Policy_Pipeline.md` (Phase 4).*
