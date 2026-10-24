"""
Async MongoDB Telemetry & Audit Logger
---------------------------------------
Non-blocking audit logging via motor (async MongoDB driver) to track
inference latency, token usage, retrieval hit-rates, and MCP verification
results for every query hitting the DocuPulse-AI engine.
"""

import time
from typing import Any, Dict, Optional
from motor.motor_asyncio import AsyncIOMotorClient

from app.config import MONGO_URI, MONGO_DB_NAME, MONGO_AUDIT_COLLECTION


class MongoAuditLogger:
    """Async MongoDB logger for inference telemetry and audit trails."""

    def __init__(
        self,
        uri: str = MONGO_URI,
        db_name: str = MONGO_DB_NAME,
        collection_name: str = MONGO_AUDIT_COLLECTION,
    ):
        self.client = AsyncIOMotorClient(uri)
        self.collection = self.client[db_name][collection_name]

    async def log_inference(
        self,
        query: str,
        latency_ms: float,
        retries: int,
        top_chunk_score: float,
        mcp_verified: bool,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert a single audit record (fire-and-forget safe)."""
        entry = {
            "query": query,
            "latency_ms": round(latency_ms, 2),
            "retries": retries,
            "top_chunk_score": round(top_chunk_score, 4),
            "mcp_verified": mcp_verified,
            "timestamp": time.time(),
        }
        if extra:
            entry.update(extra)

        try:
            await self.collection.insert_one(entry)
        except Exception:
            # Graceful degradation — never crash the inference path
            pass

    async def get_recent_logs(self, limit: int = 50) -> list:
        """Retrieve recent audit entries (for dashboard / health checks)."""
        cursor = self.collection.find().sort("timestamp", -1).limit(limit)
        return await cursor.to_list(length=limit)
