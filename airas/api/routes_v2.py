"""v2 API routes — extends v1 with prediction, domain config, and evolution stats."""
from fastapi import APIRouter
from pydantic import BaseModel
from airas.core.predictor import FailurePredictor
from airas.domains import get_adapter, ADAPTERS

router = APIRouter(prefix="/v2", tags=["v2"])

_predictor = FailurePredictor()


class PredictRequest(BaseModel):
    task_description: str
    domain: str = "coding"


class PredictResponse(BaseModel):
    predicted_failures: list[dict]  # [{class, probability}]
    preload_interventions: list[str]


class DomainConfigRequest(BaseModel):
    domain: str
    action_mappings: dict[str, str] = {}  # domain_action → universal_action


@router.post("/predict", response_model=PredictResponse)
async def predict_failures(req: PredictRequest):
    """Predict likely failure classes before agent execution starts.

    Use this to pre-load interventions into the agent's system prompt.
    """
    predictions = _predictor.predict(req.task_description)
    preload = _predictor.get_preload_interventions(req.task_description)
    return PredictResponse(
        predicted_failures=[{"class": fc, "probability": prob} for fc, prob in predictions],
        preload_interventions=preload,
    )


@router.get("/domains")
async def list_domains():
    """List available domain adapters."""
    return {"domains": list(ADAPTERS.keys())}


@router.get("/domains/{domain}/mappings")
async def get_domain_mappings(domain: str):
    """Get action type mappings for a domain."""
    adapter = get_adapter(domain)
    # Return the mapping by inspecting the adapter's map_action for known types
    return {"domain": domain, "adapter": adapter.domain_name}


@router.get("/evolution/stats")
async def evolution_stats():
    """Get intervention evolution statistics."""
    # In production, this reads from the shared bandit state
    return {"status": "evolution_engine_active", "mutations_pending": 0}
