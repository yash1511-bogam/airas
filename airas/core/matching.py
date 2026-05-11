"""Production matching engine — Qdrant-backed, two-phase, <5ms target.

Phase 1: Qdrant HNSW search with payload filter (failure_class + active)
Phase 2: Condition predicate check on top candidates (step ratio, behavioral signals)

The RuntimeFeatureExtractor computes match features from a PARTIAL trace
without knowing the outcome — this is the key difference from the experiment.
"""
import time
import logging
from airas.models import ExecutionStep, FailureClass
from airas.extraction.classifier import classify_divergence, detect_infinite_loop
from airas.extraction.abstraction import extract_antigen
from airas.models import ExecutionTrace
from airas.storage.qdrant import search_antigens

logger = logging.getLogger(__name__)


class RuntimeFeatureExtractor:
    """Extract matching features from a partial trace at runtime.

    Cannot use trace.success (unknown). Uses behavioral signals only:
    - Error patterns in recent observations
    - Loop detection (repeated actions)
    - Step count anomalies
    - Action sequence pattern
    """

    def __init__(self, steps: list[ExecutionStep]):
        self.steps = steps
        self.n = len(steps)

    def has_recent_errors(self, window: int = 3) -> bool:
        """Check if recent steps contain error signals."""
        recent = self.steps[-window:] if self.n >= window else self.steps
        return any(
            "error" in s.observation.lower() or "Error" in s.observation or "Traceback" in s.observation
            for s in recent
        )

    def has_loop(self, window: int = 4) -> bool:
        """Check if recent actions are repetitive."""
        if self.n < window:
            return False
        recent_actions = [s.action[:80] for s in self.steps[-window:]]
        return len(set(recent_actions)) <= window // 2

    def estimated_divergence_step(self) -> int:
        """Estimate where divergence likely occurred based on behavioral signals."""
        # Look for first error signal
        for i, s in enumerate(self.steps):
            if "error" in s.observation.lower() or "Error" in s.observation:
                return max(0, i - 1)
        # Look for first repeated action
        seen = set()
        for i, s in enumerate(self.steps):
            key = s.action[:80]
            if key in seen:
                return max(0, i - 1)
            seen.add(key)
        # Default: 1/3 through
        return max(1, self.n // 3)

    def danger_score(self) -> float:
        """Compute danger score from behavioral signals. 0-1 scale."""
        score = 0.0
        if self.has_recent_errors():
            score += 0.4
        if self.has_loop():
            score += 0.3
        if self.n > 30:  # unusually long trace
            score += 0.2
        # Repeated edits to same file
        edit_targets = [s.action[:50] for s in self.steps if s.action_type == "edit"]
        if edit_targets and len(set(edit_targets)) < len(edit_targets) * 0.5:
            score += 0.1
        return min(1.0, score)


def check_immunity(
    steps: list[ExecutionStep],
    project_id: str = "",
    min_danger: float = 0.3,
    threshold: float = 0.80,
) -> dict | None:
    """Real-time immunity check against the antigen index.

    Returns intervention dict if match found, None otherwise.
    Target latency: <5ms for Qdrant search + <3ms for predicate check.
    """
    t0 = time.perf_counter()
    extractor = RuntimeFeatureExtractor(steps)

    # Quick bail: if no danger signals, don't even search
    danger = extractor.danger_score()
    if danger < min_danger:
        return None

    # Build a pseudo-trace for the extraction pipeline
    div_step = extractor.estimated_divergence_step()
    trace = ExecutionTrace(
        trace_id="runtime",
        instance_id="",
        steps=steps,
        success=False,  # assume failure for matching purposes
        num_steps=len(steps),
    )
    fc = classify_divergence(trace, div_step, None)
    antigen = extract_antigen(trace, div_step, fc)

    # Phase 1: Qdrant search with failure_class filter
    results = search_antigens(
        signature=antigen.signature,
        failure_class=fc.value,
        limit=3,
        score_threshold=threshold,
    )

    if not results:
        return None

    # Phase 2: Condition predicate check
    for r in results:
        payload = r["payload"]
        cond = payload.get("conditions", {})
        # Check step ratio alignment
        cand_ratio = cond.get("ratio_through_trace", 0.5)
        test_ratio = div_step / max(len(steps), 1)
        if abs(cand_ratio - test_ratio) > 0.3:
            continue

        latency = (time.perf_counter() - t0) * 1000
        return {
            "antigen_id": r["id"],
            "confidence": r["score"],
            "failure_class": fc.value,
            "danger_score": danger,
            "latency_ms": latency,
        }

    return None
