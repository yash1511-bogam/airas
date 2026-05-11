"""Redis for hot cache + stream-based async processing.

Hot cache: top-50 most-matched antigens cached to skip Qdrant entirely.
Streams: completed traces queued for background antigen extraction.
"""
import os
import json
import logging
import redis.asyncio as redis

logger = logging.getLogger(__name__)

_pool: redis.Redis | None = None

TRACE_STREAM = "airas:traces"
ANTIGEN_CACHE_PREFIX = "airas:antigen:"
HOT_CACHE_KEY = "airas:hot_antigens"


async def get_redis() -> redis.Redis:
    global _pool
    if _pool is None:
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        _pool = redis.from_url(url, decode_responses=True)
    return _pool


async def publish_trace(trace_data: dict):
    """Push a completed trace to the processing stream."""
    r = await get_redis()
    await r.xadd(TRACE_STREAM, {"data": json.dumps(trace_data)}, maxlen=10000)


async def read_traces(consumer_group: str = "workers", consumer: str = "w1", count: int = 10):
    """Read traces from the stream (consumer group pattern)."""
    r = await get_redis()
    try:
        await r.xgroup_create(TRACE_STREAM, consumer_group, id="0", mkstream=True)
    except redis.ResponseError:
        pass  # group already exists
    messages = await r.xreadgroup(consumer_group, consumer, {TRACE_STREAM: ">"}, count=count, block=5000)
    results = []
    for stream, entries in messages:
        for msg_id, fields in entries:
            results.append((msg_id, json.loads(fields["data"])))
            await r.xack(TRACE_STREAM, consumer_group, msg_id)
    return results


async def cache_intervention(antigen_id: str, intervention_json: str, ttl: int = 300):
    """Cache an intervention result for fast repeated lookups."""
    r = await get_redis()
    await r.setex(f"{ANTIGEN_CACHE_PREFIX}{antigen_id}", ttl, intervention_json)


async def get_cached_intervention(antigen_id: str) -> str | None:
    """Get cached intervention if available."""
    r = await get_redis()
    return await r.get(f"{ANTIGEN_CACHE_PREFIX}{antigen_id}")


async def increment_check_count():
    """Track total immunity checks for stats."""
    r = await get_redis()
    await r.incr("airas:stats:total_checks")


async def get_check_count() -> int:
    r = await get_redis()
    val = await r.get("airas:stats:total_checks")
    return int(val) if val else 0
