"""
Policy document ingestion script for Omni-Help.

Loads all markdown files from data/policies/, splits them into chunks,
embeds them with text-embedding-3-small, and persists to a local ChromaDB
at data/vectors/.

Run once (or whenever policy documents are updated):
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
# Configuration
# ---------------------------------------------------------------------------

# Paths are relative to the repo root — always run from there.
POLICIES_DIR = "./data/policies"
VECTOR_DIR = "./data/vectors"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
COLLECTION_NAME = "policy_docs"


def ingest_policies() -> int:
    """
    Load, split, embed, and persist policy documents to ChromaDB.

    Returns:
        Number of document chunks successfully ingested.

    Raises:
        FileNotFoundError: If POLICIES_DIR does not exist.
        ValueError: If no documents are found in POLICIES_DIR.
    """
    if not os.path.isdir(POLICIES_DIR):
        raise FileNotFoundError(
            f"Policies directory not found: '{POLICIES_DIR}'. "
            "Run from the repo root: python src/utils/ingest.py"
        )

    # --- Load ---
    print(f"[Ingest] Loading documents from '{POLICIES_DIR}' ...")
    loader = DirectoryLoader(
        POLICIES_DIR,
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True,
    )
    docs = loader.load()

    if not docs:
        raise ValueError(f"No markdown documents found in '{POLICIES_DIR}'.")

    print(f"[Ingest] Loaded {len(docs)} document(s).")

    # --- Split ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,  # Adds 'start_index' metadata for citation traceability
    )
    chunks = splitter.split_documents(docs)
    print(f"[Ingest] Split into {len(chunks)} chunk(s) "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    # --- Embed + Persist ---
    print(f"[Ingest] Embedding with '{EMBEDDING_MODEL}' and persisting to '{VECTOR_DIR}' ...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_DIR,
    )

    count = db._collection.count()
    print(f"\n✅ Ingestion complete. {count} chunk(s) stored in ChromaDB at '{VECTOR_DIR}'.")
    return count


if __name__ == "__main__":
    ingest_policies()
