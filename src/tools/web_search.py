"""
Web search tool for the Omni-Help external knowledge pipeline.

Wraps the Tavily API behind a stable interface so the rest of the codebase
never imports Tavily directly. Swapping Tavily for another provider only
requires changes to this file.

Key design decisions:
  - Graceful failure: API errors return structured dicts instead of crashing.
  - Rate-limit awareness: max_results capped at MAX_RESULTS_LIMIT (10).
  - Citations: every formatted chunk includes its source URL.

Usage:
    from tools.web_search import execute_web_search, format_web_results

    results = execute_web_search("Latest AI regulation news", max_results=3)
    context = format_web_results(results)
"""

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_RESULTS_LIMIT = 10   # Hard cap — prevents runaway API spend


# ---------------------------------------------------------------------------
# Tavily client — lazy singleton
# ---------------------------------------------------------------------------

_tavily_client: Any = None


def _get_client():
    """
    Return the Tavily client, initialising it on first call.

    Returns:
        TavilyClient instance.

    Raises:
        EnvironmentError: If TAVILY_API_KEY is not set.
    """
    global _tavily_client
    if _tavily_client is None:
        api_key = os.getenv("TAVILY_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "TAVILY_API_KEY is not set. Add it to your .env file."
            )
        from tavily import TavilyClient
        _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def execute_web_search(query: str, max_results: int = 3) -> list[dict]:
    """
    Search the web using the Tavily API and return structured results.

    Each result dict contains:
        title   (str)  — page title
        url     (str)  — source URL
        content (str)  — extracted text snippet

    Graceful failure: if the API key is missing, the request times out, or
    any other error occurs, an error dict is returned instead of raising —
    so the web_node can write a clean error message to state without crashing
    the graph.

    Args:
        query:       Natural language search query.
        max_results: Number of results to request (capped at MAX_RESULTS_LIMIT).

    Returns:
        List of result dicts, or a single-element list containing an error dict
        on failure.

    Example:
        >>> results = execute_web_search("AI regulations 2025", max_results=3)
        >>> results[0]["url"]
        'https://...'
    """
    k = min(max_results, MAX_RESULTS_LIMIT)

    try:
        client = _get_client()
        response = client.search(
            query=query,
            max_results=k,
            search_depth="basic",
            include_answer=False,
        )
        raw_results: list[dict] = response.get("results", [])

        # Normalise to guaranteed keys so format_web_results never KeyErrors
        return [
            {
                "title":   r.get("title", "Untitled"),
                "url":     r.get("url", ""),
                "content": r.get("content", ""),
            }
            for r in raw_results
        ]

    except EnvironmentError as env_err:
        return [{"title": "Configuration Error", "url": "", "content": str(env_err)}]
    except Exception as api_err:
        return [{"title": "Web Search Error", "url": "", "content": f"Search failed: {api_err}"}]


def format_web_results(results: list[dict]) -> str:
    """
    Format a list of Tavily result dicts into a single context string.

    Each chunk is annotated with its source URL as a citation so the
    synthesis node (Phase 7) can ground every claim.

    Args:
        results: List of dicts from execute_web_search(), each with
                 'title', 'url', and 'content' keys.

    Returns:
        A single string with all results joined by double newlines,
        each annotated with its source URL.

    Example:
        >>> context = format_web_results(results)
        >>> print(context)
        AI regulations update...
        [Source: https://example.com/ai-news]

        New policy framework...
        [Source: https://gov.example.com/policy]
    """
    if not results:
        return "No web results found."

    sections: list[str] = []
    for r in results:
        content = r.get("content", "").strip()
        url = r.get("url", "")
        title = r.get("title", "")

        if not content:
            continue

        citation = f"[Source: {url}]" if url else "[Source: unknown]"
        header = f"**{title}**" if title else ""
        section = f"{header}\n{content}\n{citation}" if header else f"{content}\n{citation}"
        sections.append(section)

    return "\n\n".join(sections) if sections else "No usable web content returned."
