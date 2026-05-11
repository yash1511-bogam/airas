"""Production domain models — extends experiment models with persistence fields."""
from __future__ import annotations
from datetime import datetime
from uuid import uuid4
from pydantic import BaseModel, Field
from airas.models import FailureClass, InterventionType, ExecutionStep


def new_id() -> str:
    return uuid4().hex[:12]


class Project(BaseModel):
    project_id: str = Field(default_factory=new_id)
    name: str
    api_key_hash: str = ""
    tolerance: ToleranceConfig = Field(default_factory=lambda: ToleranceConfig())
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ToleranceConfig(BaseModel):
    min_danger_score: float = 0.3
    max_interventions_per_trace: int = 3
    cooldown_steps: int = 5
    auto_disable_threshold: float = 0.15
    acceptable_failure_rates: dict[str, float] = Field(default_factory=dict)


class StoredAntigen(BaseModel):
    antigen_id: str = Field(default_factory=new_id)
    project_id: str = ""
    failure_class: FailureClass
    divergence_step: int
    signature: str
    conditions: dict = Field(default_factory=dict)
    cluster_id: int = -1
    danger_score: float = 0.5
    match_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    active: bool = True


class StoredIntervention(BaseModel):
    intervention_id: str = Field(default_factory=new_id)
    antigen_id: str
    intervention_type: InterventionType
    applies_at_step: int
    payload: str
    failure_class: FailureClass
    efficacy_alpha: float = 1.0  # Thompson Sampling: successes + 1
    efficacy_beta: float = 1.0   # Thompson Sampling: failures + 1
    active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def efficacy(self) -> float:
        return self.efficacy_alpha / (self.efficacy_alpha + self.efficacy_beta)


class TraceRecord(BaseModel):
    trace_id: str = Field(default_factory=new_id)
    project_id: str
    instance_id: str = ""
    steps_json_compressed: bytes = b""  # zstd compressed
    success: bool | None = None  # None = in progress
    num_steps: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ImmunityEvent(BaseModel):
    """Audit trail entry for every immunity check decision."""
    event_id: str = Field(default_factory=new_id)
    project_id: str
    trace_id: str
    antigen_id: str | None = None
    intervention_id: str | None = None
    matched: bool = False
    intervened: bool = False
    confidence: float = 0.0
    latency_ms: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
