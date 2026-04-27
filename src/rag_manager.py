import logging
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except Exception as e:
    HAS_ST = False
    _ST_IMPORT_ERROR = e

logger = logging.getLogger(__name__)

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
        
    def search(self, query: str, top_k: int = 3) -> list[str]:
        """Returns the top_k most similar document fragments."""
        if not self.documents:
            return []
            
        if self.model and self.embeddings is not None:
            query_emb = self.model.encode([query], convert_to_numpy=True)
            # Normalize query
            query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
            
            # Cosine similarity via dot product (since both are normalized)
            similarities = np.dot(self.embeddings, query_emb.T).flatten()
            
            # Get top_k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            logger.info(f"RAG search matched indices: {top_indices}")
            return [self.documents[i] for i in top_indices]
        
        # RAG disabled → return no context (safe default, avoids hallucinating a random doc)
        logger.info("RAG disabled; returning no context snippets.")
        return []
