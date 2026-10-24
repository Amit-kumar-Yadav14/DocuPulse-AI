"""
Hybrid RAG Retriever — BM25 + FAISS / ChromaDB with Reciprocal Rank Fusion
---------------------------------------------------------------------------
Combines sparse lexical retrieval (BM25Okapi) with dense vector retrieval
(FAISS IndexFlatIP + ChromaDB) and merges candidate lists using Reciprocal
Rank Fusion (RRF) to capture both exact keyword matches and semantic meaning.

Math — RRF score for document *d*:
    RRF(d) = Σ_{r ∈ rankers}  1 / (k + rank_r(d))
where k = 60 (default smoothing constant).
"""

import numpy as np
import faiss
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

from app.config import EMBEDDING_MODEL, CHROMA_COLLECTION_NAME, RRF_K


class HybridRAGRetriever:
    """Dual-index retriever fusing BM25 sparse scores with FAISS dense scores."""

    def __init__(self, embedding_model: str = EMBEDDING_MODEL):
        self.embedder = SentenceTransformer(embedding_model)
        self.dimension: int = self.embedder.get_sentence_embedding_dimension()

        # 1. FAISS Index — in-memory inner-product search (cosine on normalized vecs)
        self.faiss_index = faiss.IndexFlatIP(self.dimension)

        # 2. ChromaDB — persistent metadata + vector storage
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME
        )

        # 3. BM25 lexical corpus state
        self.corpus: List[str] = []
        self.bm25: BM25Okapi | None = None

    # ── Indexing ──────────────────────────────────────────────────

    def index_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]] | None = None,
    ) -> None:
        """Index chunks simultaneously into BM25, FAISS, and ChromaDB."""
        offset = len(self.corpus)
        self.corpus.extend(documents)

        # Rebuild BM25 on the full corpus (fast — pure Python tokenization)
        tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Encode & add to FAISS
        embeddings = self.embedder.encode(
            documents, normalize_embeddings=True, show_progress_bar=False
        )
        self.faiss_index.add(np.array(embeddings, dtype=np.float32))

        # Persist in ChromaDB with metadata
        ids = [f"chunk_{offset + i}" for i in range(len(documents))]
        self.collection.add(
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas or [{"source": "internal_kb"} for _ in documents],
            ids=ids,
        )

    # ── Hybrid Retrieval (BM25 + Dense + RRF) ────────────────────

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 20,
        rrf_k: int = RRF_K,
    ) -> List[Dict[str, Any]]:
        """
        Two-stage retrieval:
          Stage-1a  BM25 lexical top-k
          Stage-1b  FAISS dense top-k
          Merge     Reciprocal Rank Fusion → unified candidate list
        """
        if not self.corpus or self.bm25 is None:
            return []

        # ── Step 1: BM25 Lexical Retrieval ────────────────────────
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k]

        # ── Step 2: FAISS Dense Vector Retrieval ──────────────────
        query_vec = self.embedder.encode([query], normalize_embeddings=True)
        dense_scores, dense_top_indices = self.faiss_index.search(
            np.array(query_vec, dtype=np.float32), top_k
        )
        dense_top_indices = dense_top_indices[0]

        # ── Step 3: Reciprocal Rank Fusion (RRF) ─────────────────
        rrf_scores: Dict[int, float] = {}

        for rank, idx in enumerate(bm25_top_indices):
            if idx != -1:
                rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0.0) + 1.0 / (
                    rrf_k + rank + 1
                )

        for rank, idx in enumerate(dense_top_indices):
            if idx != -1:
                rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0.0) + 1.0 / (
                    rrf_k + rank + 1
                )

        sorted_indices = sorted(
            rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
        )[:top_k]

        return [
            {
                "chunk_id": idx,
                "text": self.corpus[idx],
                "rrf_score": round(rrf_scores[idx], 5),
            }
            for idx in sorted_indices
        ]
