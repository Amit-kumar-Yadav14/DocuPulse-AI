"""
DocuPulse-AI Configuration
--------------------------
Centralized hyperparameters and environment variable management
for the Agentic Hybrid-RAG & MCP Knowledge Engine.
"""

import os


# ─── Embedding Model ──────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# ─── Cross-Encoder Re-Ranker Model ───────────────────────────────
RERANKER_MODEL: str = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# ─── Retrieval Hyperparameters ────────────────────────────────────
BM25_TOP_K: int = int(os.getenv("BM25_TOP_K", "20"))
DENSE_TOP_K: int = int(os.getenv("DENSE_TOP_K", "20"))
RRF_K: int = int(os.getenv("RRF_K", "60"))
RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", "5"))
HALLUCINATION_THRESHOLD: float = float(os.getenv("HALLUCINATION_THRESHOLD", "0.35"))
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "2"))

# ─── Chunking Defaults ───────────────────────────────────────────
CHUNK_SIZE_TOKENS: int = int(os.getenv("CHUNK_SIZE_TOKENS", "512"))
CHUNK_OVERLAP_TOKENS: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "64"))

# ─── MongoDB ──────────────────────────────────────────────────────
MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "docupulse_db")
MONGO_AUDIT_COLLECTION: str = os.getenv("MONGO_AUDIT_COLLECTION", "inference_audit_logs")

# ─── ChromaDB ─────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "docupulse_kb")

# ─── Server ───────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
API_WORKERS: int = int(os.getenv("API_WORKERS", "2"))
