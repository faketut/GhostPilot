import logging
import re
from collections.abc import Callable

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except Exception as e:
    HAS_ST = False
    _ST_IMPORT_ERROR = e

logger = logging.getLogger(__name__)

_KB_HEADER = re.compile(r"^\[kb:([^]#]+)#\d+\]")


def kb_relpath_from_chunk(chunk: str) -> str | None:
    """Return the knowledge file path from a chunk header, or None if missing."""
    first = (chunk or "").lstrip().split("\n", 1)[0].strip()
    m = _KB_HEADER.match(first)
    return m.group(1) if m else None


class RAGManager:
    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.model = None
        if HAS_ST:
            logger.info("Loading sentence transformer model for RAG...")
            try:
                self.model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
            except Exception as e:
                # Common on Windows when torch DLLs fail to initialize.
                logger.warning(f"Failed to init sentence-transformers (RAG disabled): {e}")
                self.model = None
        else:
            err = globals().get("_ST_IMPORT_ERROR")
            if err:
                logger.warning(f"sentence-transformers unavailable (RAG disabled): {err}")
            else:
                logger.warning("sentence-transformers unavailable (RAG disabled).")

    def load_documents(self, texts: list[str]):
        """Vectorizes and loads documents (Resume, JD) into memory."""
        self.documents = texts
        logger.info(f"Loaded {len(texts)} documents into RAG memory.")
        if self.model and texts:
            self.embeddings = self.model.encode(self.documents, convert_to_numpy=True)
            # Normalize embeddings for fast cosine similarity
            self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)

    def search(
        self,
        query: str,
        top_k: int = 3,
        *,
        min_score: float = 0.0,
        source_filter: Callable[[str], bool] | None = None,
    ) -> list[str]:
        """
        Return up to top_k chunk texts ranked by cosine similarity to the query.

        - min_score: drop hits with similarity strictly below this (0 = no threshold).
        - source_filter: if set, only chunks whose [kb:rel#n] rel path passes are kept.
        """
        if not self.documents:
            return []

        if self.model and self.embeddings is not None:
            query_emb = self.model.encode([query], convert_to_numpy=True)
            query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)

            similarities = np.dot(self.embeddings, query_emb.T).flatten()
            order = np.argsort(similarities)[::-1]

            out: list[str] = []
            for idx in order:
                if len(out) >= top_k:
                    break
                sim = float(similarities[idx])
                if sim < min_score:
                    continue
                doc = self.documents[int(idx)]
                if source_filter is not None:
                    rel = kb_relpath_from_chunk(doc) or ""
                    if not source_filter(rel):
                        continue
                out.append(doc)

            logger.info(
                "RAG search: %d snippets (min_score=%s, filter=%s)",
                len(out),
                min_score,
                source_filter is not None,
            )
            return out

        logger.info("RAG disabled; returning no context snippets.")
        return []
