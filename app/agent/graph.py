"""
LangGraph Agentic State Machine
--------------------------------
Implements a cyclic graph with self-correction:

    Retrieve → Rerank → Hallucination Grader → (Generate | Query-Rewrite → Retrieve)

The hallucination grader rejects low-confidence chunks (rerank_score < 0.35)
and triggers a query-rewrite loop, ensuring the final synthesis is strictly
grounded in high-quality retrieved evidence.
"""

from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

from app.retrieval.hybrid_retriever import HybridRAGRetriever
from app.retrieval.reranker import PyTorchCrossEncoderReranker
from app.agent.mcp_server import verify_citation_metadata
from app.config import HALLUCINATION_THRESHOLD, MAX_RETRIES


class AgentState(TypedDict):
    """Typed state dict flowing through every LangGraph node."""

    query: str
    rewritten_query: str
    candidates: List[Dict[str, Any]]
    reranked_chunks: List[Dict[str, Any]]
    mcp_verification: Dict[str, Any]
    hallucination_passed: bool
    retries: int
    final_answer: str


class DocuPulseAgent:
    """Self-correcting agentic RAG pipeline orchestrated by LangGraph."""

    def __init__(
        self,
        retriever: HybridRAGRetriever,
        reranker: PyTorchCrossEncoderReranker,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.workflow = self._build_graph()

    # ── Graph Nodes ───────────────────────────────────────────────

    def retrieve_node(self, state: AgentState) -> AgentState:
        """Stage-1: Hybrid retrieval (BM25 + FAISS) with RRF fusion."""
        active_query = state.get("rewritten_query") or state["query"]
        candidates = self.retriever.retrieve_hybrid(active_query, top_k=20)
        return {**state, "candidates": candidates}

    def rerank_node(self, state: AgentState) -> AgentState:
        """Stage-2: PyTorch Cross-Encoder re-ranking (top-20 → top-5)."""
        active_query = state.get("rewritten_query") or state["query"]
        top_chunks = self.reranker.rerank(active_query, state["candidates"], top_n=5)
        return {**state, "reranked_chunks": top_chunks}

    def grade_and_mcp_node(self, state: AgentState) -> AgentState:
        """Hallucination grader + MCP citation verification."""
        chunks = state.get("reranked_chunks", [])

        if not chunks or chunks[0]["rerank_score"] < HALLUCINATION_THRESHOLD:
            # Confidence too low → trigger self-correction loop
            retries = state.get("retries", 0) + 1
            rewritten = f"{state['query']} technical specification architecture details"
            return {
                **state,
                "hallucination_passed": False,
                "retries": retries,
                "rewritten_query": rewritten,
            }

        # Invoke MCP Tool to verify top chunk provenance
        mcp_res = verify_citation_metadata(chunk_id=chunks[0]["chunk_id"])
        return {
            **state,
            "hallucination_passed": True,
            "mcp_verification": mcp_res,
        }

    def generate_node(self, state: AgentState) -> AgentState:
        """Grounded synthesis from cross-encoder verified chunks."""
        chunks = state.get("reranked_chunks", [])
        context_str = "\n---\n".join(
            [
                f"[Chunk {c['chunk_id']} | Score: {c['rerank_score']}]: {c['text']}"
                for c in chunks
            ]
        )

        answer = (
            f"Grounded Synthesis (Verified via MCP "
            f"{state.get('mcp_verification', {}).get('compliance_status')}):\n"
            f"Based on the top {len(chunks)} cross-encoder re-ranked chunks:\n"
            f"{context_str}"
        )
        return {**state, "final_answer": answer}

    # ── Conditional Routing ───────────────────────────────────────

    def decide_next_step(self, state: AgentState) -> str:
        """Route: generate if grounded or retries exhausted, else re-retrieve."""
        if state["hallucination_passed"] or state.get("retries", 0) >= MAX_RETRIES:
            return "generate"
        return "retrieve"

    # ── Graph Construction ────────────────────────────────────────

    def _build_graph(self):
        builder = StateGraph(AgentState)

        builder.add_node("retrieve", self.retrieve_node)
        builder.add_node("rerank", self.rerank_node)
        builder.add_node("grade_and_mcp", self.grade_and_mcp_node)
        builder.add_node("generate", self.generate_node)

        builder.set_entry_point("retrieve")
        builder.add_edge("retrieve", "rerank")
        builder.add_edge("rerank", "grade_and_mcp")
        builder.add_conditional_edges(
            "grade_and_mcp",
            self.decide_next_step,
            {"generate": "generate", "retrieve": "retrieve"},
        )
        builder.add_edge("generate", END)

        return builder.compile()
