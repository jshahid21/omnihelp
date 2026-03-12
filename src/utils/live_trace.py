"""
Live LangSmith tracing script for Omni-Help.

IMPORTANT: load_dotenv() runs before any LangChain import so that
LANGCHAIN_TRACING_V2=true is in os.environ when LangChain tracing
callbacks initialise at module load time.

Usage (from repo root, venv activated):
    python src/utils/live_trace.py
"""

import os
import sys
import asyncio

from dotenv import load_dotenv

load_dotenv()


def _check_environment():
    """
    Verify LangSmith tracing is configured and active.

    Prints clear, actionable instructions for each missing piece.

    Returns:
        True if LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY is set.
    """
    ok = True

    tracing = os.getenv("LANGCHAIN_TRACING_V2", "").strip().lower()
    if tracing != "true":
        print()
        print("=" * 65)
        print("  WARNING: LANGCHAIN_TRACING_V2 is not enabled!")
        print("=" * 65)
        print("  In your .env file change:")
        print("    LANGCHAIN_TRACING_V2=false")
        print("  to:")
        print("    LANGCHAIN_TRACING_V2=true")
        print("  Then re-run: python src/utils/live_trace.py")
        print("=" * 65)
        ok = False

    api_key = os.getenv("LANGCHAIN_API_KEY", "").strip()
    if not api_key:
        print()
        print("=" * 65)
        print("  WARNING: LANGCHAIN_API_KEY is missing!")
        print("=" * 65)
        print("  Get your key at: https://smith.langchain.com/")
        print("  Add to .env:  LANGCHAIN_API_KEY=lsv2_pt_<key>")
        print("=" * 65)
        ok = False

    if ok:
        project = os.getenv("LANGCHAIN_PROJECT", "default")
        print()
        print("LangSmith tracing is ACTIVE")
        print("  Project : " + project)
        print("  Key     : " + api_key[:12] + "...")
        print()

    return ok


def _import_graph():
    """
    Import and compile the graph after env vars are loaded.

    Deferred import ensures LANGCHAIN_TRACING_V2=true is already set
    in os.environ when LangChain tracing hooks init at module load.
    """
    src_path = os.path.join(os.path.dirname(__file__), "..")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from graph.graph import build_graph
    return build_graph()


# Each tuple: (description, query, expected_pipeline)
TEST_CASES = [
    (
        "Multi-intent: policy + order (router must pick one)",
        "I need to know the return policy for electronics, and also, where is my order ORD-1001?",
        "policy or sql",
    ),
    (
        "Pure SQL: order status lookup",
        "What is the current status and estimated delivery date for order ORD-1002?",
        "sql",
    ),
    (
        "Pure Policy: RAG retrieval",
        "What items are excluded from the 30-day return window?",
        "policy",
    ),
    (
        "Web fallback: general knowledge",
        "What are the latest AI regulations in the European Union for 2025?",
        "web",
    ),
    (
        "Complaint: direct fallback, no synthesis",
        "This is completely unacceptable. I have been waiting 3 weeks and no one helps.",
        "fallback",
    ),
]


async def run_trace(graph) -> None:
    """
    Execute all test cases through the compiled graph and print a summary.

    Each ainvoke call creates one run in LangSmith under LANGCHAIN_PROJECT.
    Open https://smith.langchain.com/ after this script finishes to see:
      - The node sequence that executed (router -> sql -> synthesis, etc.)
      - Exact LLM prompts and completions at each node
      - Per-node latency and cumulative token usage
      - Full AgentState snapshot at every step
    """
    total = len(TEST_CASES)
    print("=" * 65)
    print("  Omni-Help Live Trace Runner  --  " + str(total) + " test cases")
    print("=" * 65)

    for idx, (description, query, expected) in enumerate(TEST_CASES, start=1):
        print()
        print("[" + str(idx) + "/" + str(total) + "]  " + description)
        print("  Query    : " + query[:80] + ("..." if len(query) > 80 else ""))
        print("  Expected : " + expected)
        print("  Running  ...")

        try:
            result = await graph.ainvoke(
                {"user_query": query},
                config={
                    "configurable": {"thread_id": "live-trace-" + str(idx)},
                    "run_name": "trace-" + str(idx) + ": " + description[:45],
                },
            )
            intent = result.get("intent", "unknown")
            confidence = result.get("confidence", 0.0)
            response = result.get("final_response", "-- no response --")

            print("  Intent   : " + intent + "  (" + str(round(confidence * 100)) + "% confidence)")
            print("  Response : " + response[:120] + ("..." if len(response) > 120 else ""))

        except Exception as exc:
            print("  ERROR: " + str(exc))

    project = os.getenv("LANGCHAIN_PROJECT", "default")
    print()
    print("=" * 65)
    print()
    print("  Run complete! Open LangSmith to view execution traces:")
    print()
    print("  https://smith.langchain.com/")
    print("  Project: " + project)
    print()
    print("  In the dashboard you will see:")
    print("  - Node execution order for each run")
    print("  - Exact LLM prompts and completions")
    print("  - Latency and token usage per node")
    print("  - Full AgentState at every checkpoint")
    print()
    print("=" * 65)
    print()


if __name__ == "__main__":
    if not _check_environment():
        sys.exit(1)

    print("Importing graph (loads LLM clients and DB connections)...")
    graph = _import_graph()
    print("Graph compiled. Starting trace runs...")

    asyncio.run(run_trace(graph))
