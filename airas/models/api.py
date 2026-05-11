"""API request/response schemas."""
from pydantic import BaseModel, Field
from airas.models import FailureClass, InterventionType, ExecutionStep


class TraceIngestRequest(BaseModel):
    trace_id: str = ""
    project_id: str = ""
    instance_id: str = ""
    steps: list[ExecutionStep]
    success: bool | None = None  # None = still in progress


class CheckRequest(BaseModel):
    """Real-time immunity check — send partial trace, get intervention."""
    project_id: str = ""
    trace_id: str = ""
    steps: list[ExecutionStep]  # partial trace so far
    current_step: int = -1  # which step is about to execute


class CheckResponse(BaseModel):
    matched: bool = False
    antigen_id: str | None = None
    intervention_id: str | None = None
    intervention_type: InterventionType | None = None
    payload: str | None = None  # the instruction to inject
    confidence: float = 0.0
    failure_class: FailureClass | None = None
    latency_ms: float = 0.0


class AntigenResponse(BaseModel):
    antigen_id: str
    failure_class: FailureClass
    signature: str
    danger_score: float
    match_count: int
    active: bool
    cluster_id: int


class StatsResponse(BaseModel):
    total_antigens: int
    total_interventions: int
    total_traces: int
    total_checks: int
    prevention_rate: float
    top_patterns: list[dict]
