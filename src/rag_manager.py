import logging
import re
from collections.abc import Callable

import numpy as np

# Sentence-transformers is large (~80MB + torch). Defer the import until we
# actually have documents to embed (Phase 5.3 bundle slimming).
HAS_ST: bool | None = None  # None = not probed yet
_ST_IMPORT_ERROR: Exception | None = None
SentenceTransformer = None  # type: ignore


def _probe_st():
    global HAS_ST, _ST_IMPORT_ERROR, SentenceTransformer
    if HAS_ST is not None:
        return
    try:
        from sentence_transformers import SentenceTransformer as _ST
        SentenceTransformer = _ST
        HAS_ST = True
    except Exception as e:
        HAS_ST = False
        _ST_IMPORT_ERROR = e

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except Exception:
    HAS_BM25 = False

logger = logging.getLogger(__name__)

_KB_HEADER = re.compile(r"^\[kb:([^]#]+)#\d+\]")


def kb_relpath_from_chunk(chunk: str) -> str | None:
    """Return the knowledge file path from a chunk header, or None if missing."""
    first = (chunk or "").lstrip().split("\n", 1)[0].strip()
    m = _KB_HEADER.match(first)
    return m.group(1) if m else None


class RAGManager:
    # Simple tokenizer for BM25: lower + word chars / CJK chars.
    _TOK_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)

    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.model = None
        self._bm25 = None
        self._bm25_tokens: list[list[str]] = []
        # Model is loaded lazily on first load_documents() call with non-empty
        # texts (Phase 5.3). Keeps cold-start fast when no knowledge base.

    def _ensure_model(self):
        if self.model is not None:
            return
        _probe_st()
        if not HAS_ST:
            if _ST_IMPORT_ERROR is not None:
                logger.warning(f"sentence-transformers unavailable (RAG dense disabled): {_ST_IMPORT_ERROR}")
            return
        logger.info("Loading sentence transformer model for RAG...")
        try:
            self.model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        except Exception as e:
            logger.warning(f"Failed to init sentence-transformers (RAG dense disabled): {e}")
            self.model = None

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        return [t.lower() for t in cls._TOK_RE.findall(text or "")]

    def load_documents(self, texts: list[str]):
        """Vectorizes and loads documents (Resume, JD) into memory."""
        self.documents = texts
        logger.info(f"Loaded {len(texts)} documents into RAG memory.")
        if texts:
            self._ensure_model()
        if self.model and texts:
            self.embeddings = self.model.encode(self.documents, convert_to_numpy=True)
            # Normalize embeddings for fast cosine similarity
            self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        # BM25 index (independent of dense — works even without sentence_transformers).
        if HAS_BM25 and texts:
            self._bm25_tokens = [self._tokenize(t) for t in texts]
            try:
                self._bm25 = BM25Okapi(self._bm25_tokens)
            except Exception as e:
                logger.warning(f"BM25 init failed: {e}")
                self._bm25 = None
        else:
            self._bm25 = None
            self._bm25_tokens = []

    def _rrf(self, *rankings: list[int], k: int = 60) -> list[int]:
        """Reciprocal rank fusion: combine ranked id lists into a single ranking."""
        scores: dict[int, float] = {}
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.keys(), key=lambda i: scores[i], reverse=True)

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

        # Dense ranking
        dense_order: list[int] = []
        dense_sims: np.ndarray | None = None
        if self.model and self.embeddings is not None:
            query_emb = self.model.encode([query], convert_to_numpy=True)
            query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
            dense_sims = np.dot(self.embeddings, query_emb.T).flatten()
            dense_order = list(np.argsort(dense_sims)[::-1][: top_k * 5])

        # BM25 ranking
        bm25_order: list[int] = []
        if self._bm25 is not None:
            try:
                bm25_scores = self._bm25.get_scores(self._tokenize(query))
                bm25_order = list(np.argsort(bm25_scores)[::-1][: top_k * 5])
            except Exception as e:
                logger.warning(f"BM25 query failed: {e}")

        # Combine
        if dense_order and bm25_order:
            fused = self._rrf(dense_order, bm25_order)
            mode = "hybrid(dense+bm25)"
        elif dense_order:
            fused = dense_order
            mode = "dense"
        elif bm25_order:
            fused = bm25_order
            mode = "bm25"
        else:
            logger.info("RAG disabled; returning no context snippets.")
            return []

        out: list[str] = []
        for idx in fused:
            if len(out) >= top_k:
                break
            # min_score only applies when we have dense similarities.
            if dense_sims is not None:
                if float(dense_sims[int(idx)]) < min_score and idx not in bm25_order[:top_k]:
                    continue
            doc = self.documents[int(idx)]
            if source_filter is not None:
                rel = kb_relpath_from_chunk(doc) or ""
                if not source_filter(rel):
                    continue
            out.append(doc)

        logger.info(
            "RAG search [%s]: %d snippets (min_score=%s, filter=%s)",
            mode, len(out), min_score, source_filter is not None,
        )
        return out
