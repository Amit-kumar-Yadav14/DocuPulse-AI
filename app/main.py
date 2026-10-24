"""
DocuPulse-AI — FastAPI Inference Microservice + MongoDB Audit Logging
----------------------------------------------------------------------
Production entry point for the Agentic Hybrid-RAG & MCP Knowledge Engine.

Endpoints:
    POST /v1/ingest   —  Index new document chunks
    POST /v1/query    —  Run the full agentic pipeline (Retrieve → Rerank → Grade → Generate)
    GET  /health      —  Liveness probe for Kubernetes
"""

import time
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.retrieval.hybrid_retriever import HybridRAGRetriever
from app.retrieval.reranker import PyTorchCrossEncoderReranker
from app.agent.graph import DocuPulseAgent
from app.db.mongo_logger import MongoAuditLogger

# ─── Core ML Components ──────────────────────────────────────────
retriever = HybridRAGRetriever()
reranker = PyTorchCrossEncoderReranker()
agent = DocuPulseAgent(retriever=retriever, reranker=reranker)

# ─── Async MongoDB Audit Logger ──────────────────────────────────
audit_logger = MongoAuditLogger()


# ─── Lifespan (startup / shutdown) ───────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Seed a starter corpus on boot so the engine is immediately queryable."""
    sample_docs = [
        "DocuPulse-AI combines BM25 lexical search with dense FAISS vector "
        "embeddings using Reciprocal Rank Fusion.",
        "PyTorch Cross-Encoder re-ranking evaluates query-document pairs jointly "
        "with self-attention, reducing LLM hallucinations by 38%.",
        "Kubernetes HorizontalPodAutoscaler scales FastAPI inference pods across "
        "AWS EC2 and Azure ML endpoints to maintain 1.4s p95 latency.",
        "Model Context Protocol (MCP) standardizes agent tool-calling between "
        "LangGraph state machines and enterprise data sources.",
    ]
    retriever.index_documents(sample_docs)
    yield


app = FastAPI(
    title="DocuPulse-AI: Agentic Hybrid-RAG & MCP Engine",
    version="1.0.0",
    description="Enterprise low-latency Agentic Hybrid-RAG with MCP tool verification",
    lifespan=lifespan,
)


# ─── Request / Response Schemas ──────────────────────────────────
class QueryRequest(BaseModel):
    query: str


class IngestRequest(BaseModel):
    documents: list[str]


# ─── Endpoints ────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """Kubernetes liveness / readiness probe."""
    return {
        "status": "HEALTHY",
        "indexed_chunks": len(retriever.corpus),
        "engine": "DocuPulse-AI v1.0.0",
    }


@app.post("/v1/ingest")
async def ingest_docs(payload: IngestRequest):
    """Index new document chunks into BM25 + FAISS + ChromaDB."""
    retriever.index_documents(payload.documents)
    return {"status": "indexed", "total_chunks": len(retriever.corpus)}


@app.post("/v1/query")
async def query_knowledge_engine(payload: QueryRequest):
    """Run the full agentic pipeline and return grounded synthesis."""
    start_time = time.perf_counter()
    try:
        initial_state = {
            "query": payload.query,
            "rewritten_query": "",
            "candidates": [],
            "reranked_chunks": [],
            "mcp_verification": {},
            "hallucination_passed": False,
            "retries": 0,
            "final_answer": "",
        }
        result = agent.workflow.invoke(initial_state)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Non-blocking MongoDB audit log
        await audit_logger.log_inference(
            query=payload.query,
            latency_ms=latency_ms,
            retries=result.get("retries", 0),
            top_chunk_score=(
                result["reranked_chunks"][0]["rerank_score"]
                if result["reranked_chunks"]
                else 0.0
            ),
            mcp_verified=result.get("mcp_verification", {}).get("verified", False),
        )

        return {
            "answer": result["final_answer"],
            "latency_ms": latency_ms,
            "reranked_sources": result["reranked_chunks"],
            "mcp_audit": result["mcp_verification"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
