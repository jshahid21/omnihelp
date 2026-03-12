# Omni-Help Deep Dive: Web Fallback & Synthesis

**Learning Module 03 — Phases 6–7**  
**Audience:** Intermediate Python developer moving into production AI systems  
**Files covered:** `tools/web_search.py` · `graph/nodes.py` (web_node, synthesis_node, fallback_node) · `prompts/synthesis_prompts.py`

---

## Table of Contents

1. [Mental Model: The Three Failure Modes of AI Systems](#1-mental-model-the-three-failure-modes-of-ai-systems)
2. [The Web Search Pipeline (Phase 6)](#2-the-web-search-pipeline-phase-6)
   - 2.1 Why Tavily, Not Raw Google Search?
   - 2.2 `web_search.py` — The Three-Layer Graceful Failure Pattern
   - 2.3 `web_node` — Connecting to the Graph
3. [The Synthesis Node (Phase 7)](#3-the-synthesis-node-phase-7)
   - 3.1 Why a Separate Synthesis Step?
   - 3.2 Context Priority and the If-Elif Chain
   - 3.3 `synthesis_prompts.py` — Designing the Persona Prompt
   - 3.4 The `temperature=0.3` Decision
   - 3.5 Writing Back to Conversation History
4. [The Fallback Node (Phase 7)](#4-the-fallback-node-phase-7)
   - 4.1 When Does Fallback Trigger?
   - 4.2 Building the Handoff Context
5. [Enterprise Patterns vs. Amateur Code](#5-enterprise-patterns-vs-amateur-code)
6. [Common Misconceptions](#6-common-misconceptions)

---

## 1. Mental Model: The Three Failure Modes of AI Systems

When building production AI pipelines, you must design for three distinct failure modes:

| Failure Mode | Example | Amateur Behavior | Enterprise Behavior |
|---|---|---|---|
| **Configuration error** | `TAVILY_API_KEY` not set | `KeyError` crashes the app | Caught, returns structured error dict |
| **Transient error** | API timeout after 30s | Exception bubbles up, graph crashes | Caught, returns `{"content": "Search failed: timeout"}` |
| **Empty result** | Search returns 0 results | Returns empty list, downstream code crashes | Handled explicitly with `"No usable web content returned."` |

The Omni-Help web pipeline is designed so that **the graph can never crash due to a failed web search**. In the worst case, the user gets a polite "I couldn't find that information" message — not a raw Python traceback.

This principle — **graceful degradation** — is one of the most important distinctions between a proof-of-concept and a production system.

---

## 2. The Web Search Pipeline (Phase 6)

### 2.1 Why Tavily, Not Raw Google Search?

Several options exist for web search in AI pipelines:

| Option | Pros | Cons |
|---|---|---|
| Google Custom Search API | Familiar brand | Complex setup, limited free tier, HTML scraping needed |
| SerpAPI | Rich metadata | Expensive, rate limits |
| Bing Search API | Microsoft ecosystem | Requires Azure account |
| **Tavily** | **AI-native, pre-scraped clean text, built-in relevance scoring** | **Relatively new** |

Tavily was built specifically for LLM pipelines. Its API returns **pre-extracted text content** (not raw HTML you have to parse), **relevance scores**, and **source URLs** in a clean JSON format. For an AI assistant, this is the right tool — you get the substance of a web page without writing a scraper.

```python
# What Tavily returns (already cleaned and structured):
{
    "results": [
        {
            "title": "New AI Regulation Framework 2025",
            "url": "https://example.gov/ai-policy",
            "content": "The new framework requires enterprises to...",
            "score": 0.87
        },
        ...
    ]
}
```

Compare this to using `requests` + `BeautifulSoup` to scrape Google — that approach breaks every time a website updates its HTML structure.

### 2.2 `web_search.py` — The Three-Layer Graceful Failure Pattern

The entire error handling strategy is contained in a single `try/except` block with two distinct catch clauses:

```python
def execute_web_search(query: str, max_results: int = 3) -> list[dict]:
    k = min(max_results, MAX_RESULTS_LIMIT)   # Hard cap: prevents runaway API spend

    try:
        client = _get_client()    # Layer 1: config check (raises EnvironmentError)
        response = client.search( # Layer 2: API call (raises various Exception types)
            query=query,
            max_results=k,
            search_depth="basic",
            include_answer=False,
        )
        raw_results = response.get("results", [])

        # Normalise to guaranteed keys — format_web_results never KeyErrors
        return [
            {
                "title":   r.get("title", "Untitled"),
                "url":     r.get("url", ""),
                "content": r.get("content", ""),
            }
            for r in raw_results
        ]

    except EnvironmentError as env_err:
        # Layer 1 catch: missing API key — returns structured error, not crash
        return [{"title": "Configuration Error", "url": "", "content": str(env_err)}]

    except Exception as api_err:
        # Layer 2 catch: timeout, rate limit, network error — returns structured error
        return [{"title": "Web Search Error", "url": "", "content": f"Search failed: {api_err}"}]
```

**Why two separate `except` clauses?** `EnvironmentError` is a programming error (missing config) — you want to log it differently from a transient `TimeoutError` (infrastructure hiccup). In production, you might alert on config errors immediately but retry on transient errors.

**Why return an error dict instead of `None` or `[]`?** The downstream `format_web_results` function and `web_node` must always have the same shape to work with. An empty list would cause `"No usable web content returned."` — which is fine. But wrapping the error message in the same `{"title": ..., "content": ...}` structure means the Synthesis Node receives a contextual hint about *why* the search failed, allowing it to generate a more helpful message to the user.

**The normalisation step** (`r.get("title", "Untitled")`) is another production-grade detail. Third-party APIs evolve and occasionally omit fields. Using `.get()` with defaults means a missing `"title"` key in Tavily's response never causes a `KeyError` in your production system at 3 AM.

**`MAX_RESULTS_LIMIT = 10`** is a hard cap that prevents a developer from accidentally calling `execute_web_search(query, max_results=10000)` and triggering enormous Tavily API bills. The `min()` function silently corrects the value rather than raising an error.

### 2.3 `web_node` — Connecting to the Graph

```python
def web_node(state: AgentState) -> dict[str, Any]:
    user_query = state.get("user_query", "")

    results = execute_web_search(user_query, max_results=3)
    web_context = format_web_results(results)

    return {
        "web_results": results,        # Raw list of dicts (for debugging/testing)
        "web_context": web_context,    # Formatted string (for Synthesis Node)
        "final_response": f"Web Search Results:\n\n{web_context}",
    }
```

**Why store both `web_results` and `web_context`?** `web_results` is the raw structured data — useful for tests that want to assert on specific URLs or titles without parsing strings. `web_context` is the pre-formatted string the Synthesis Node injects into its prompt. Storing both keeps each layer of the system working with its natural data format.

---

## 3. The Synthesis Node (Phase 7)

### 3.1 Why a Separate Synthesis Step?

A naive implementation would have each pipeline node call the LLM and return a final response directly:

```python
# ❌ Naive approach: each node generates its own response
def retriever_node(state):
    docs = retriever.invoke(state["user_query"])
    response = llm.invoke(f"Answer this using the docs: {docs}\nQuery: {state['user_query']}")
    return {"final_response": response.content}
```

This has three problems:
1. **Code duplication:** Every pipeline node contains identical LLM invocation logic and persona instructions
2. **Inconsistent tone:** Each node might phrase the Omni-Help persona slightly differently
3. **No central control:** Changing the system prompt means editing 4 files

The Synthesis Node solves this by acting as a **single exit point** for all three pipelines:

```
Retriever Node → (policy_context) ──┐
                                     ├──▶ Synthesis Node ──▶ final_response
SQL Node       → (sql_result)    ──┤
                                     │
Web Node       → (web_context)   ──┘
```

One LLM call. One persona. One place to change the prompt.

### 3.2 Context Priority and the If-Elif Chain

The Synthesis Node reads `AgentState` and selects *which* context to inject into the prompt. The priority order is deliberate:

```python
def synthesis_node(state: AgentState) -> dict[str, Any]:
    policy_context = state.get("policy_context")
    sql_result     = state.get("sql_result")
    sql_query      = state.get("sql_query")
    sql_error      = state.get("sql_error")
    web_context    = state.get("web_context")

    if sql_error:
        # Priority 1: Surface errors first — users need to know something failed
        context = f"Database error: {sql_error}"

    elif sql_result is not None:
        # Priority 2: SQL result — include the query so LLM can read column names
        context = (
            f"SQL query executed:\n{sql_query}\n\n"
            f"Database result (rows match the SELECT columns above):\n{sql_result}"
        )

    elif policy_context:
        # Priority 3: Policy chunks from ChromaDB
        context = f"Policy documentation:\n{policy_context}"

    elif web_context:
        # Priority 4: Web search results
        context = f"Web search results:\n{web_context}"

    else:
        # Priority 5: Nothing found
        context = "No relevant context was found for this query."
```

**Why does `sql_error` come before `sql_result`?** Because if `sql_error` is set, `sql_result` will be `None` — but the logic is clearer and more defensive written this way. The node never needs to check both simultaneously.

**The SQL column-name problem:** A critical lesson from building this system. Raw SQLite results look like:
```
[('ORD-1001', 'alice@example.com', 'Shipped', 129.99)]
```

Without column names, the LLM does not know which value is `order_id`, which is `status`, and which is `total_amount`. Injecting the SQL query alongside the result gives the LLM a natural way to read column names:

```
SQL query executed:
SELECT order_id, customer_email, status, total_amount FROM orders WHERE order_id = 'ORD-1001'

Database result (rows match the SELECT columns above):
[('ORD-1001', 'alice@example.com', 'Shipped', 129.99)]
```

Now the LLM can map: `order_id='ORD-1001'`, `status='Shipped'`, `total_amount=129.99` — and write a correct natural language response.

### 3.3 `synthesis_prompts.py` — Designing the Persona Prompt

The synthesis system prompt has six numbered rules, each addressing a specific failure mode:

```
Rule 1: Context-only answers
```
Without this rule, the LLM uses its training data to "fill gaps" when the retrieved context is thin. This leads to hallucination — the LLM might confidently state a return policy that was true when it was trained but has since changed. The rule forces every factual claim to trace back to the provided context.

```
Rule 2: Error handling
```
If the context contains `"Database error: Forbidden SQL operation detected: 'DELETE'"`, the LLM must turn this into a user-friendly message rather than regurgitating the raw error. Without this rule, you would see responses like: "I'm sorry, but there was a `ValueError: Forbidden SQL operation detected: 'DELETE'. Only SELECT queries are permitted.`" — which is confusing and exposes implementation details.

```
Rule 3: No context available
```
Handles the case where all three pipelines return empty results. Instead of the LLM fabricating an answer from training data, it says "I couldn't find that information" and offers to escalate. Honest uncertainty is better than confident hallucination.

```
Rule 4: Conciseness
```
Without this, the LLM might paste the entire policy document back at the user. The instruction to "synthesise into a natural answer" produces a focused 2-3 sentence response.

```
Rule 5: Citations
```
Encourages the LLM to reference sources naturally ("According to our return policy...") rather than either citing nothing or using ugly `[Source: ./data/policies/return_policy.md]` verbatim.

```
Rule 6: Tone
```
Professional, empathetic, solution-oriented. Never dismissive. This is the baseline customer support persona.

### 3.4 The `temperature=0.3` Decision

The three LLMs in this system all use different temperature settings, each justified:

| Node | Temperature | Justification |
|---|---|---|
| `router_agent` | 0.0 (via structured output) | Classification must be deterministic — the same query should always route to the same pipeline |
| `sql_node` | 0.0 | SQL generation is a deterministic task — creative variation produces invalid SQL |
| `synthesis_node` | **0.3** | Responses should be natural and slightly varied — a small amount of "creativity" makes answers feel human rather than robotic |

`temperature=0.3` is a conservative choice: enough variation to avoid sounding canned ("Your order ORD-1001 is Shipped. Your order ORD-1002 is Shipped.") while avoiding hallucination risk (which increases with higher temperatures).

### 3.5 Writing Back to Conversation History

The synthesis node returns an `AIMessage` in `messages` — this is how the graph maintains conversation history across turns:

```python
from langchain_core.messages import AIMessage

return {
    "final_response": final_response,
    "messages": [AIMessage(content=final_response)],
}
```

**Why does this work without overwriting previous messages?** `AgentState` defines `messages` with the `add_messages` reducer:

```python
# From state.py
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
```

`add_messages` is a LangGraph reducer — when a node returns `{"messages": [new_message]}`, LangGraph *appends* the new message to the existing list rather than replacing it. This is what enables multi-turn conversations: every `HumanMessage` and `AIMessage` accumulates in state, giving the LLM full context on follow-up requests.

---

## 4. The Fallback Node (Phase 7)

### 4.1 When Does Fallback Trigger?

Two conditions route to `fallback_node` (defined in `graph.py`'s `route_decision`):

1. **Intent is `complaint`** — Any query classified as a complaint goes directly to a human agent, regardless of confidence. The system never attempts to autonomously resolve complaints.
2. **Clarification exhausted** — After `MAX_CLARIFICATION_TURNS` (2) cycles of the router returning low confidence, the system gives up and escalates.

```
Router → confidence < 0.7 → Clarification → Router
                                                 │
                               confidence < 0.7  │  (turn 2)
                                                 ▼
                                              Fallback
```

### 4.2 Building the Handoff Context

The fallback node does not just end the conversation — it builds a structured `handoff_context` dictionary that a human agent can read:

```python
handoff_context = {
    "original_query": user_query,
    "detected_intent": intent,           # What the AI thought the user wanted
    "routing_rationale": routing_rationale,  # Why the AI classified it this way
    "escalation_reason": reason,         # "complaint" or "low_confidence — exhausted after 2 turns"
    "clarification_turns": turn_count,   # How many times clarification was attempted
    "conversation_length": len(messages),
}
```

This handoff packet means a human support agent picking up the ticket has full context: they know what the user asked, why the AI couldn't resolve it, and how many times it tried. The user does not need to repeat themselves.

In a production system, `handoff_context` would be serialized to a CRM ticket (Zendesk, Salesforce, etc.) via an additional API call in this node.

---

## 5. Enterprise Patterns vs. Amateur Code

### Pattern 1: Structured Error Returns (not raised exceptions)

```python
# ❌ Amateur: exception escapes the tool, crashes the graph
def execute_web_search(query: str) -> list[dict]:
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return client.search(query=query)["results"]

# ✅ Enterprise: errors are data, not control flow
def execute_web_search(query: str) -> list[dict]:
    try:
        ...
        return normalised_results
    except EnvironmentError as e:
        return [{"title": "Configuration Error", "url": "", "content": str(e)}]
    except Exception as e:
        return [{"title": "Web Search Error", "url": "", "content": f"Search failed: {e}"}]
```

### Pattern 2: Single Exit Point for LLM Calls

```python
# ❌ Amateur: every pipeline node contains its own LLM response logic
def retriever_node(state):
    docs = ...
    return {"final_response": llm.invoke(f"Answer: {docs}")}

def sql_node(state):
    result = ...
    return {"final_response": llm.invoke(f"Answer: {result}")}

# ✅ Enterprise: all pipelines write to context fields; one synthesis node generates the response
def retriever_node(state):
    return {"policy_context": format_docs(docs)}  # Just writes context

def synthesis_node(state):
    context = pick_best_context(state)           # Single place for LLM response logic
    return {"final_response": llm.invoke(...)}
```

### Pattern 3: Temperature as a Design Decision

```python
# ❌ Amateur: leave at default (usually 1.0) everywhere
router_llm    = ChatOpenAI(model="gpt-4o-mini")
sql_llm       = ChatOpenAI(model="gpt-4o-mini")
synthesis_llm = ChatOpenAI(model="gpt-4o-mini")

# ✅ Enterprise: each LLM has a temperature calibrated to its task
router_llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)  # deterministic classification
sql_llm       = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)  # deterministic SQL generation
synthesis_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)  # natural but grounded responses
```

---

## 6. Common Misconceptions

### "I should use a higher temperature for synthesis to sound more human"

Higher temperature increases the LLM's tendency to deviate from the provided context. At `temperature=0.7`, the synthesis node might start *inventing* policy details that "sound plausible" but are not in the retrieved chunks. For a customer support system, a false confident answer is worse than an honest "I couldn't find that." Keep synthesis temperature below `0.4`.

### "The fallback node is a failure state"

Fallback is a **designed outcome**, not a failure. In customer support, some queries genuinely require a human — a customer threatening legal action, a complex multi-account billing dispute, a accessibility need the AI cannot address. The system acknowledging its limits and escalating gracefully is the correct behavior. The failure would be the AI hallucinating a resolution for a complaint it cannot handle.

### "I can skip format_web_results and just pass raw Tavily output to the LLM"

Tavily's raw output contains fields like `score`, `raw_content`, `published_date`, and others that the LLM will try to interpret and potentially include in the response. `format_web_results` strips away everything except the text content and source URL, keeping the context focused. It also handles the edge case where `content` is an empty string — those results are silently dropped rather than producing empty citations in the final response.

### "The synthesis node knows which pipeline was used"

It does not — and by design it does not need to. The synthesis node does not check `intent` or `route` from the state. It simply reads whichever of `{sql_error, sql_result, policy_context, web_context}` is populated. This makes it truly pipeline-agnostic: if you add a new "Product Catalog" pipeline tomorrow, you add `product_context` to `AgentState` and add one `elif product_context:` branch in the synthesis node. No other code changes needed.
