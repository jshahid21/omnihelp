"""
Document ingestion script for Omni-Help.

Loads markdown files from two source directories, splits them into chunks,
embeds with text-embedding-3-small, and persists to separate ChromaDB
collections under data/vectors/.

Collections:
  - "policies" ← data/policies/  (return policy, shipping, etc.)
  - "products"  ← data/manuals/   (product user guides, troubleshooting)

Run once (or whenever documents are updated):
    python src/utils/ingest.py
"""

import os
import sys

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Ensure project src/ is on the path when run as a script
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# ---------------------------------------------------------------------------
# Configuration — collection names must match vector_store.py exactly
# ---------------------------------------------------------------------------

VECTOR_DIR = "./data/vectors"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

INGESTION_SOURCES = [
    {
        "directory": "./data/policies",
        "collection": "policies",
        "label": "Policy documents",
    },
    {
        "directory": "./data/manuals",
        "collection": "products",
        "label": "Product manuals",
    },
]


def ingest_directory(directory: str, collection: str, label: str) -> int:
    """
    Load, split, embed, and persist markdown files from one directory
    into a named ChromaDB collection.

    Args:
        directory:  Path to the source directory (relative to repo root).
        collection: ChromaDB collection name to write into.
        label:      Human-readable label for log output.

    Returns:
        Number of document chunks successfully ingested.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If no markdown documents are found.
    """
    print(f"\n{'=' * 55}")
    print(f"  Ingesting: {label}")
    print(f"  Source   : {directory}")
    print(f"  Collection: {collection}")
    print(f"{'=' * 55}")

    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"Directory not found: '{directory}'. "
            "Run from the repo root: python src/utils/ingest.py"
        )

    # --- Load ---
    print(f"[Ingest] Loading documents from '{directory}' ...")
    loader = DirectoryLoader(
        directory,
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True,
    )
    docs = loader.load()

    if not docs:
        raise ValueError(f"No markdown documents found in '{directory}'.")

    print(f"[Ingest] Loaded {len(docs)} document(s).")

    # --- Split ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    print(f"[Ingest] Split into {len(chunks)} chunk(s) "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    # --- Embed + Persist ---
    print(f"[Ingest] Embedding with '{EMBEDDING_MODEL}' → collection '{collection}' ...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection,
        persist_directory=VECTOR_DIR,
    )

    count = db._collection.count()
    print(f"[Ingest] Done. {count} chunk(s) in collection '{collection}'.")
    return count


def ingest_all() -> None:
    """
    Ingest all configured source directories into their respective
    ChromaDB collections.
    """
    totals = {}
    for source in INGESTION_SOURCES:
        count = ingest_directory(
            directory=source["directory"],
            collection=source["collection"],
            label=source["label"],
        )
        totals[source["collection"]] = count

    print(f"\n{'=' * 55}")
    print("  Ingestion complete!")
    for collection, count in totals.items():
        print(f"  {collection:12s} → {count} chunk(s) in ChromaDB at '{VECTOR_DIR}'")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    ingest_all()
