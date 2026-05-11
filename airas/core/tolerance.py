"""Tolerance layer — prevents autoimmune over-correction.

Controls when AIRAS should NOT intervene, even if a pattern matches:
- Danger score below threshold (noise)
- Too many interventions already applied to this trace
- Cooldown period after last intervention
- Intervention efficacy too low (auto-disabled)
"""
from airas.models.domain import ToleranceConfig


class ToleranceGate:
    """Decides whether to allow or suppress an intervention."""

    def __init__(self, config: ToleranceConfig | None = None):
        self.config = config or ToleranceConfig()
        self._trace_intervention_counts: dict[str, int] = {}
        self._trace_last_intervention_step: dict[str, int] = {}

    def should_intervene(
        self,
        trace_id: str,
        current_step: int,
        danger_score: float,
        intervention_efficacy: float,
        failure_class: str,
    ) -> tuple[bool, str]:
        """Returns (allow, reason). If allow=False, suppress the intervention."""
        # Check 1: Danger score threshold
        if danger_score < self.config.min_danger_score:
            return False, f"danger_score {danger_score:.2f} below threshold {self.config.min_danger_score}"

        # Check 2: Max interventions per trace
        count = self._trace_intervention_counts.get(trace_id, 0)
        if count >= self.config.max_interventions_per_trace:
            return False, f"max interventions ({self.config.max_interventions_per_trace}) reached for trace"

        # Check 3: Cooldown
        last_step = self._trace_last_intervention_step.get(trace_id, -999)
        if current_step - last_step < self.config.cooldown_steps:
            return False, f"cooldown: {self.config.cooldown_steps - (current_step - last_step)} steps remaining"

        # Check 4: Efficacy threshold
        if intervention_efficacy < self.config.auto_disable_threshold:
            return False, f"intervention efficacy {intervention_efficacy:.2f} below auto-disable threshold"

        # Check 5: Acceptable failure rate for this class
        acceptable = self.config.acceptable_failure_rates.get(failure_class, 1.0)
        if acceptable <= 0:
            return False, f"failure class '{failure_class}' marked as tolerable"

        return True, "all checks passed"

    def record_intervention(self, trace_id: str, step: int):
        """Record that an intervention was applied."""
        self._trace_intervention_counts[trace_id] = self._trace_intervention_counts.get(trace_id, 0) + 1
        self._trace_last_intervention_step[trace_id] = step

    def reset_trace(self, trace_id: str):
        """Clear state for a completed trace."""
        self._trace_intervention_counts.pop(trace_id, None)
        self._trace_last_intervention_step.pop(trace_id, None)
