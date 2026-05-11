"""AIRAS FastAPI application — the service layer."""
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, HTTPException
from airas.models.api import (
    TraceIngestRequest, CheckRequest, CheckResponse, AntigenResponse, StatsResponse,
)
from airas.models import FailureClass, InterventionType
from airas.core.matching import check_immunity
from airas.intervention.templates import get_intervention
from airas.models import FailureAntigen
from airas.storage import redis_store, qdrant

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: ensure Qdrant collection exists."""
    logger.info("AIRAS API starting up")
    qdrant.get_qdrant()  # triggers collection creation
    yield
    logger.info("AIRAS API shutting down")


app = FastAPI(
    title="AIRAS",
    description="Adaptive Immune Runtime for Agent Systems",
    version="0.2.0",
    lifespan=lifespan,
)

# Include v2 routes
from airas.api.routes_v2 import router as v2_router
app.include_router(v2_router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "airas-api"}


@app.post("/v1/check", response_model=CheckResponse)
async def immunity_check(req: CheckRequest):
    """Real-time immunity check. Latency-critical: <50ms total.

    Receives a partial trace (execution in progress), returns intervention if matched.
    """
    t0 = time.perf_counter()

    result = check_immunity(
        steps=req.steps,
        project_id=req.project_id,
        min_danger=0.3,
        threshold=0.80,
    )

    await redis_store.increment_check_count()

    if result is None:
        return CheckResponse(matched=False, latency_ms=(time.perf_counter() - t0) * 1000)

    # Get intervention for matched antigen
    antigen = FailureAntigen(
        antigen_id=result["antigen_id"],
        failure_class=FailureClass(result["failure_class"]),
        divergence_step=req.current_step if req.current_step >= 0 else len(req.steps) - 1,
        signature="",
    )
    intervention = get_intervention(antigen)

    latency = (time.perf_counter() - t0) * 1000
    return CheckResponse(
        matched=True,
        antigen_id=result["antigen_id"],
        intervention_id=intervention.intervention_id,
        intervention_type=intervention.intervention_type,
        payload=intervention.payload,
        confidence=result["confidence"],
        failure_class=FailureClass(result["failure_class"]),
        latency_ms=latency,
    )


@app.post("/v1/traces")
async def ingest_trace(req: TraceIngestRequest):
    """Ingest a completed trace for background antigen extraction."""
    trace_data = req.model_dump()
    await redis_store.publish_trace(trace_data)
    return {"status": "queued", "trace_id": req.trace_id, "steps": len(req.steps)}


@app.get("/v1/antigens")
async def list_antigens(limit: int = 50, failure_class: str | None = None):
    """List discovered antigens."""
    # Query Qdrant for all antigens (with optional filter)
    from qdrant_client import models
    client = qdrant.get_qdrant()
    filt = models.Filter(must=[models.FieldCondition(key="active", match=models.MatchValue(value=True))])
    if failure_class:
        filt.must.append(models.FieldCondition(key="failure_class", match=models.MatchValue(value=failure_class)))

    results = client.scroll(collection_name=qdrant.COLLECTION, scroll_filter=filt, limit=limit)
    antigens = []
    for point in results[0]:
        p = point.payload
        antigens.append(AntigenResponse(
            antigen_id=str(point.id),
            failure_class=FailureClass(p["failure_class"]),
            signature=p.get("signature", ""),
            danger_score=p.get("danger_score", 0.5),
            match_count=p.get("match_count", 0),
            active=p.get("active", True),
            cluster_id=p.get("cluster_id", -1),
        ))
    return {"antigens": antigens, "total": len(antigens)}


@app.get("/v1/stats")
async def get_stats():
    """Dashboard stats."""
    total_checks = await redis_store.get_check_count()
    total_antigens = qdrant.count_antigens()
    return StatsResponse(
        total_antigens=total_antigens,
        total_interventions=0,  # TODO: from postgres
        total_traces=0,
        total_checks=total_checks,
        prevention_rate=0.52,  # from experiment; updated by evaluator worker
        top_patterns=[],
    )
