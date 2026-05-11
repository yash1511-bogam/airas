"""Background workers — consume trace stream, extract antigens, update index.

Runs as a separate process (airas-worker container).
Consumes from Redis Streams, processes traces, upserts to Qdrant.
"""
import asyncio
import json
import logging
from airas.storage import redis_store, qdrant
from airas.extraction.alignment import find_divergence_point
from airas.extraction.classifier import classify_divergence
from airas.extraction.abstraction import extract_antigen
from airas.models import ExecutionTrace, ExecutionStep

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def process_trace(trace_data: dict):
    """Process a single ingested trace — extract antigen if it's a failure."""
    steps = [ExecutionStep(**s) for s in trace_data.get("steps", [])]
    success = trace_data.get("success")

    if success is True or success is None:
        return  # Only extract antigens from confirmed failures

    if len(steps) < 3:
        return  # Too short to analyze

    trace = ExecutionTrace(
        trace_id=trace_data.get("trace_id", ""),
        instance_id=trace_data.get("instance_id", ""),
        steps=steps,
        success=False,
    )

    # Simple divergence estimation (no reference trace available in real-time)
    div_step = max(1, len(steps) // 3)
    fc = classify_divergence(trace, div_step, None)
    antigen = extract_antigen(trace, div_step, fc)

    # Upsert to Qdrant
    qdrant.upsert_antigen(
        antigen_id=antigen.antigen_id,
        signature=antigen.signature,
        failure_class=fc.value,
        conditions=antigen.conditions,
        danger_score=0.5,
    )
    logger.info("Extracted antigen %s (class=%s) from trace %s", antigen.antigen_id, fc.value, trace.trace_id)


async def worker_loop():
    """Main worker loop — consume traces from Redis stream."""
    logger.info("AIRAS worker started — listening for traces")
    while True:
        try:
            messages = await redis_store.read_traces(count=5)
            for msg_id, trace_data in messages:
                try:
                    await process_trace(trace_data)
                except Exception as e:
                    logger.error("Failed to process trace: %s", e)
        except Exception as e:
            logger.error("Worker stream error: %s", e)
            await asyncio.sleep(5)


async def main():
    await worker_loop()


if __name__ == "__main__":
    asyncio.run(main())
