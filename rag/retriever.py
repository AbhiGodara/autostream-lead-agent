"""
rag/retriever.py — AutoStream RAG Retriever
Loads the knowledge base, chunks it by ## headers, embeds chunks using
sentence-transformers, and exposes a retrieve() function using cosine similarity.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

# Constants
_KB_PATH = Path(__file__).parent / "knowledge_base.md"
_MODEL_NAME = "all-MiniLM-L6-v2"
_TOP_K = 2  # number of chunks to return

# Internal state — loaded once at import time

_model: SentenceTransformer | None = None
_chunks: List[str] = []
_embeddings: np.ndarray | None = None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _load_and_chunk(path: Path) -> List[str]:
    """Split the markdown file into chunks by '##' section headers."""
    text = path.read_text(encoding="utf-8")
    # Split on lines that start with '##' (but not '###')
    raw_chunks = re.split(r"\n(?=## )", text)
    chunks: List[str] = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _initialize() -> None:
    """Load model and embed all chunks (runs once)."""
    global _model, _chunks, _embeddings

    if _model is not None:
        return  # already initialized

    _model = SentenceTransformer(_MODEL_NAME)
    _chunks = _load_and_chunk(_KB_PATH)

    if not _chunks:
        raise ValueError(f"No chunks found in knowledge base at {_KB_PATH}")

    # Shape: (num_chunks, embedding_dim)
    _embeddings = _model.encode(_chunks, convert_to_numpy=True)


# Initialize at import time so the first query isn't slow
_initialize()


def retrieve(query: str) -> str:
    """
    Retrieve the top-k most relevant chunks from the knowledge base.

    Args:
        query: The user's question / search string.

    Returns:
        A single string containing the top-k chunks joined by a separator.
    """
    if _model is None or _embeddings is None:
        _initialize()

    query_embedding: np.ndarray = _model.encode([query], convert_to_numpy=True)[0]

    similarities: List[Tuple[int, float]] = [
        (idx, _cosine_similarity(query_embedding, chunk_emb))
        for idx, chunk_emb in enumerate(_embeddings)
    ]

    # Sort descending by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    top_indices = [idx for idx, _ in similarities[:_TOP_K]]
    top_chunks = [_chunks[i] for i in top_indices]

    return "\n\n---\n\n".join(top_chunks)
