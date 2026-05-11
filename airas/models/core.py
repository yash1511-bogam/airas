"""Core data models for AIRAS killer experiment.

Design: Pydantic models representing the full lifecycle from raw trace → antigen → intervention → result.
All models are serializable to JSON for experiment reproducibility.
"""
from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np


class FailureClass(str, Enum):
    WRONG_TOOL = "wrong_tool"
    WRONG_PARAMS = "wrong_params"
    WRONG_INTERPRETATION = "wrong_interpretation"
    PREMATURE_TERMINATION = "premature_termination"
    INFINITE_LOOP = "infinite_loop"
    RETRIEVAL_FAILURE = "retrieval_failure"
    PLANNING_FAILURE = "planning_failure"
    UNKNOWN = "unknown"


class InterventionType(str, Enum):
    GATE = "gate"
    CONSTRAINT = "constraint"
    REWRITE = "rewrite"
    FALLBACK = "fallback"
    CHECKPOINT = "checkpoint"


class ExecutionStep(BaseModel):
    """Single step in an agent execution trace."""
    index: int
    action: str  # the command/tool call executed
    action_type: str  # normalized: edit, search, run_test, open, navigate, submit
    observation: str  # tool output (truncated to 500 chars for embedding)
    thought: str = ""
    state: dict = Field(default_factory=dict)  # open_file, working_dir, etc.


class ExecutionTrace(BaseModel):
    """Normalized agent execution trace from any source."""
    trace_id: str
    instance_id: str  # SWE-bench issue ID
    model_name: str = ""
    steps: list[ExecutionStep]
    success: bool
    exit_status: str = ""
    patch: str = ""  # generated diff
    num_steps: int = 0

    def model_post_init(self, __context):
        if not self.num_steps:
            self.num_steps = len(self.steps)


class FailureAntigen(BaseModel):
    """Abstract failure pattern extracted from divergence analysis."""
    antigen_id: str
    failure_class: FailureClass
    divergence_step: int  # step index where failure diverges from success
    signature: str  # human-readable abstract pattern description
    conditions: dict = Field(default_factory=dict)  # triggering conditions
    source_trace_ids: list[str] = Field(default_factory=list)
    cluster_id: int = -1
    # embedding stored separately in Qdrant, not in model


class Intervention(BaseModel):
    """A concrete intervention to prevent a failure pattern."""
    intervention_id: str
    antigen_id: str
    intervention_type: InterventionType
    applies_at_step: int  # relative to divergence point (-1 = one step before)
    payload: str  # the instruction/gate/constraint to inject
    failure_class: FailureClass
    efficacy: float = 0.5  # Thompson Sampling prior


class MatchResult(BaseModel):
    """Result of matching a partial trace against the antigen index."""
    antigen_id: str
    confidence: float
    failure_class: FailureClass
    intervention: Optional[Intervention] = None
    latency_ms: float = 0.0


class ExperimentResult(BaseModel):
    """Full experiment outcome metrics."""
    total_test_failures: int
    matched: int  # how many test failures matched a known antigen
    prevented: int  # how many were judged as prevented by intervention
    prevention_rate: float  # prevented / total_test_failures
    coverage_rate: float  # matched / total_test_failures
    match_prevention_rate: float  # prevented / matched
    false_positives: int  # interventions triggered on successful traces
    false_positive_rate: float
    mean_latency_ms: float
    num_patterns: int  # distinct antigen clusters found
    confidence_interval_95: tuple[float, float] = (0.0, 0.0)
    per_class_results: dict = Field(default_factory=dict)
