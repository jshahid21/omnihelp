"""
Vector store tool for Omni-Help RAG pipelines.

Wraps ChromaDB behind a stable interface so the rest of the codebase
never imports Chroma directly. Swapping ChromaDB for Qdrant (prod) only
requires changes to this file.

Two collections are maintained in the same persist directory:
  - "policies"  → data/policies/ (return policy, shipping, etc.)
  - "products"  → data/manuals/  (product user guides, troubleshooting)

Usage:
    from tools.vector_store import get_policy_retriever, get_product_retriever, format_docs

    policy_retriever  = get_policy_retriever(k=5)
    product_retriever = get_product_retriever(k=3)
    docs    = policy_retriever.invoke("What is the return policy?")
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
# Configuration — must match the constants in ingest.py exactly.
# ---------------------------------------------------------------------------

VECTOR_DIR = "./data/vectors"
EMBEDDING_MODEL = "text-embedding-3-small"

POLICY_COLLECTION = "policies"
PRODUCT_COLLECTION = "products"

# ---------------------------------------------------------------------------
# Singleton embeddings client — one HTTP connection pool, reused per call.
# ---------------------------------------------------------------------------

_embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)


def _get_collection(collection_name: str) -> Chroma:
    """
    Open a named ChromaDB collection from the shared persist directory.

    Args:
        collection_name: One of POLICY_COLLECTION or PRODUCT_COLLECTION.

    Returns:
        Chroma instance for the requested collection.

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
        collection_name=collection_name,
        embedding_function=_embeddings,
        persist_directory=VECTOR_DIR,
    )


def get_policy_retriever(k: int = 5):
    """
    Return a LangChain retriever for policy documents (return policy, shipping, etc.).

    Args:
        k: Number of chunks to retrieve per query. Default is 5.

    Returns:
        A LangChain VectorStoreRetriever compatible with .invoke() and LCEL chains.

    Example:
        >>> retriever = get_policy_retriever(k=3)
        >>> docs = retriever.invoke("What is the return window?")
    """
    return _get_collection(POLICY_COLLECTION).as_retriever(search_kwargs={"k": k})


def get_product_retriever(k: int = 3):
    """
    Return a LangChain retriever for product manuals and troubleshooting guides.

    Args:
        k: Number of chunks to retrieve per query. Default is 3 (manuals are
           more focused than policy docs, so fewer chunks suffice).

    Returns:
        A LangChain VectorStoreRetriever compatible with .invoke() and LCEL chains.

    Example:
        >>> retriever = get_product_retriever(k=3)
        >>> docs = retriever.invoke("EcoHome thermostat screen blank")
    """
    return _get_collection(PRODUCT_COLLECTION).as_retriever(search_kwargs={"k": k})


def format_docs(docs: List[Document]) -> str:
    """
    Format a list of retrieved Document objects into a single context string.

    Each chunk is appended with its source file as a citation so the
    synthesis node can reference the origin of every claim.

    Args:
        docs: List of LangChain Document objects returned by a retriever.

    Returns:
        A single string with all chunk contents joined by double newlines,
        each annotated with its source path.

    Example:
        >>> context = format_docs(docs)
        >>> print(context)
        Items can be returned within 30 days...
        [Source: data/policies/return_policy.md]
    """
    if not docs:
        return "No relevant documents found."

    sections: List[str] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown source")
        sections.append(f"{doc.page_content.strip()}\n[Source: {source}]")

    return "\n\n".join(sections)
