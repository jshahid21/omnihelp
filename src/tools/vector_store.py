"""
Vector store tool for the Policy RAG pipeline.

Wraps ChromaDB behind a stable interface so the rest of the codebase
never imports Chroma directly. Swapping ChromaDB for Qdrant (prod) only
requires changes to this file.

Usage:
    from tools.vector_store import get_policy_retriever, format_docs

    retriever = get_policy_retriever(k=5)
    docs = retriever.invoke("What is the return policy?")
    context = format_docs(docs)
"""

import os
from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration — mirrors ingest.py constants exactly.
# ---------------------------------------------------------------------------

VECTOR_DIR = "./data/vectors"
EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "policy_docs"

# ---------------------------------------------------------------------------
# Singleton embeddings client — one HTTP connection pool, reused per call.
# ---------------------------------------------------------------------------

_embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)


def _get_db() -> Chroma:
    """
    Open the persisted ChromaDB collection.

    Returns:
        Chroma instance pointed at the local vector store.

    Raises:
        FileNotFoundError: If the vector store has not been initialised yet
            (i.e., ingest.py has not been run).
    """
    if not os.path.isdir(VECTOR_DIR):
        raise FileNotFoundError(
            f"Vector store not found at '{VECTOR_DIR}'. "
            "Run the ingestion script first: python src/utils/ingest.py"
        )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_embeddings,
        persist_directory=VECTOR_DIR,
    )


def get_policy_retriever(k: int = 5):
    """
    Return a LangChain retriever for policy documents.

    The retriever accepts a string query and returns the top-k most
    semantically similar document chunks from ChromaDB.

    Args:
        k: Number of chunks to retrieve per query. Blueprint default is 5.

    Returns:
        A LangChain VectorStoreRetriever compatible with .invoke() and
        LangChain LCEL (|) chains.

    Example:
        >>> retriever = get_policy_retriever(k=3)
        >>> docs = retriever.invoke("What is the return window?")
    """
    db = _get_db()
    return db.as_retriever(search_kwargs={"k": k})


def format_docs(docs: List[Document]) -> str:
    """
    Format a list of retrieved Document objects into a single context string.

    Each chunk is appended with its source file as a citation so the
    synthesis node can reference the origin of every claim.

    Args:
        docs: List of LangChain Document objects returned by the retriever.

    Returns:
        A single string with all chunk contents joined by double newlines,
        each annotated with its source path.

    Example:
        >>> context = format_docs(docs)
        >>> print(context)
        Items can be returned within 30 days...
        [Source: data/policies/return_policy.md]

        Standard shipping takes 3-5 business days...
        [Source: data/policies/return_policy.md]
    """
    if not docs:
        return "No relevant policy documents found."

    sections: List[str] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown source")
        sections.append(f"{doc.page_content.strip()}\n[Source: {source}]")

    return "\n\n".join(sections)
