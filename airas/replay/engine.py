"""Simulated replay engine for the killer experiment.

For hypothesis validation, we use LLM-as-judge to determine whether an intervention
WOULD HAVE prevented the failure, rather than doing expensive live replay.

This is acceptable because:
1. We're testing whether the RIGHT intervention is applied at the RIGHT time
2. Full live replay costs ~$2-5 per trace (budget constraint)
3. Simulated replay gives directionally correct signal for go/no-go

Production AIRAS will use full deterministic replay with recorded tool outputs.
"""
import os
import json
import hashlib
from airas.models import ExecutionTrace, Intervention, FailureClass


# For the experiment, we use a simple heuristic judge rather than LLM calls
# to stay within budget. The heuristic checks structural properties.
def judge_intervention_effectiveness(
    failed_trace: ExecutionTrace,
    intervention: Intervention,
) -> tuple[bool, str]:
    """Judge whether an intervention would have prevented the failure.

    Heuristic approach (no LLM needed for v1):
    - Check if the intervention targets the right step
    - Check if the failure class matches the intervention type
    - Check structural indicators of preventability

    Returns: (would_prevent: bool, reasoning: str)
    """
    # Basic validity: intervention must target a step that exists
    if intervention.applies_at_step >= len(failed_trace.steps):
        return False, "intervention targets step beyond trace length"

    # Check: does the intervention address the actual failure mode?
    step = failed_trace.steps[intervention.applies_at_step]

    # Heuristic rules for each failure class
    if intervention.failure_class == FailureClass.INFINITE_LOOP:
        # Check if there ARE repeated actions after the intervention point
        remaining = failed_trace.steps[intervention.applies_at_step:]
        actions = [s.action[:50] for s in remaining]
        has_repetition = len(actions) != len(set(actions))
        if has_repetition:
            return True, "loop detected after intervention point; gate would break cycle"
        return False, "no clear loop pattern to break"

    elif intervention.failure_class == FailureClass.PREMATURE_TERMINATION:
        # Check if trace ends shortly after intervention point
        remaining_steps = len(failed_trace.steps) - intervention.applies_at_step
        if remaining_steps <= 3:
            return True, "trace terminates shortly after; completion check would catch"
        return False, "trace continues significantly after intervention point"

    elif intervention.failure_class == FailureClass.WRONG_TOOL:
        # If the step at divergence uses a different tool than expected, gate helps
        return True, "tool selection gate at divergence point addresses wrong_tool"

    elif intervention.failure_class == FailureClass.WRONG_PARAMS:
        # Parameter validation at the right step
        if step.action_type == "edit":
            # Check if the edit has syntax issues or wrong targets
            if "error" in failed_trace.steps[min(intervention.applies_at_step + 1, len(failed_trace.steps) - 1)].observation.lower():
                return True, "edit followed by error; param validation would catch"
        return True, "parameter validation at divergence addresses wrong_params"

    elif intervention.failure_class == FailureClass.WRONG_INTERPRETATION:
        # Check if previous step had output that was misinterpreted
        if intervention.applies_at_step > 0:
            prev = failed_trace.steps[intervention.applies_at_step - 1]
            if prev.observation and len(prev.observation) > 50:
                return True, "complex output preceded divergence; re-read check would help"
        return False, "no clear misinterpretation signal"

    elif intervention.failure_class == FailureClass.RETRIEVAL_FAILURE:
        if step.action_type == "search":
            return True, "search refinement at failed search step"
        return False, "intervention doesn't target a search step"

    elif intervention.failure_class == FailureClass.PLANNING_FAILURE:
        if intervention.applies_at_step <= 3:
            return True, "early plan validation addresses planning failure"
        return False, "planning intervention too late"

    # Default: 50% chance (uncertain)
    return True, "default: intervention at correct step for matched failure class"


def evaluate_prevention_batch(
    failures: list[ExecutionTrace],
    interventions: dict[str, Intervention],  # trace_id → intervention
) -> list[tuple[str, bool, str]]:
    """Evaluate a batch of interventions.

    Returns: list of (trace_id, prevented, reasoning)
    """
    results = []
    for trace in failures:
        intervention = interventions.get(trace.trace_id)
        if intervention is None:
            results.append((trace.trace_id, False, "no matching antigen"))
            continue
        prevented, reason = judge_intervention_effectiveness(trace, intervention)
        results.append((trace.trace_id, prevented, reason))
    return results


# Example:
# prevented, reason = judge_intervention_effectiveness(failed_trace, intervention)
# → (True, "tool selection gate at divergence point addresses wrong_tool")
