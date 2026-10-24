"""
DocuPulse-AI — Ingestion & Latency Benchmark
---------------------------------------------
Indexes 10,000+ synthetic enterprise chunks into the Hybrid-RAG pipeline
and runs end-to-end latency benchmarks to validate the ≤ 1.4s p95 target.

Usage:
    python -m scripts.ingest_benchmark
"""

import time
import numpy as np
from app.retrieval.hybrid_retriever import HybridRAGRetriever
from app.retrieval.reranker import PyTorchCrossEncoderReranker
from app.agent.graph import DocuPulseAgent


def generate_synthetic_corpus(num_chunks: int = 10500) -> list[str]:
    """Generate 10,000+ enterprise technical chunks for benchmarking."""
    base_topics = [
        "Kubernetes HorizontalPodAutoscaler scales FastAPI microservices "
        "based on CPU and custom latency metrics.",
        "BM25 lexical search captures exact error codes like ERR_CUDA_OOM_504 "
        "while dense vectors capture semantics.",
        "PyTorch Cross-Encoder models compute full self-attention over query "
        "and document pairs to reduce hallucinations.",
        "Model Context Protocol (MCP) standardizes tool invocation across "
        "LangGraph autonomous agent workflows.",
        "ChromaDB and FAISS IndexFlatIP allow sub-20ms inner-product "
        "similarity search across embeddings.",
    ]
    return [
        f"[Doc-{i}] {base_topics[i % len(base_topics)]} "
        f"Partition-ID: {i // 100} Checksum-{i * 17}"
        for i in range(num_chunks)
    ]


def run_p95_benchmark(num_queries: int = 50):
    print("🚀 Initializing DocuPulse-AI Benchmark Suite...")
    retriever = HybridRAGRetriever()
    reranker = PyTorchCrossEncoderReranker()
    agent = DocuPulseAgent(retriever=retriever, reranker=reranker)

    corpus = generate_synthetic_corpus(10200)
    print(f"📦 Indexing {len(corpus):,} chunks into BM25 + FAISS + ChromaDB...")
    t0 = time.perf_counter()
    retriever.index_documents(corpus)
    print(f"✅ Indexed {len(corpus):,} chunks in {time.perf_counter() - t0:.2f}s\n")

    sample_queries = [
        "How does PyTorch Cross-Encoder reduce hallucinations in RAG?",
        "What happens during ERR_CUDA_OOM_504 in Kubernetes pods?",
        "Explain Model Context Protocol MCP tool calling in LangGraph",
        "How fast is FAISS IndexFlatIP vector search with BM25?",
    ]

    latencies_ms: list[float] = []
    print(f"⏱️  Running {num_queries} end-to-end Agentic Hybrid-RAG queries...")
    for i in range(num_queries):
        q = sample_queries[i % len(sample_queries)]
        start = time.perf_counter()
        agent.workflow.invoke(
            {
                "query": q,
                "rewritten_query": "",
                "candidates": [],
                "reranked_chunks": [],
                "mcp_verification": {},
                "hallucination_passed": False,
                "retries": 0,
                "final_answer": "",
            }
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)

    p50 = np.percentile(latencies_ms, 50)
    p95 = np.percentile(latencies_ms, 95)
    p99 = np.percentile(latencies_ms, 99)

    print("\n📊 === DOCUPULSE-AI LATENCY REPORT (10,200 CHUNKS) ===")
    print(f"  • Median (p50) Latency : {p50:.2f} ms ({p50 / 1000:.2f} s)")
    print(f"  • p95 Latency          : {p95:.2f} ms ({p95 / 1000:.2f} s)  ← Target ≤ 1.40s")
    print(f"  • p99 Latency          : {p99:.2f} ms ({p99 / 1000:.2f} s)")
    print("=" * 55)


if __name__ == "__main__":
    run_p95_benchmark()
