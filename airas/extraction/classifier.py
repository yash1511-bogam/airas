"""Rule-based divergence classifier.

Classifies the divergence point into one of 7 failure archetypes.
Entirely deterministic — no LLM calls. Fast enough for real-time use.
"""
from airas.models import ExecutionTrace, ExecutionStep, FailureClass


def detect_infinite_loop(trace: ExecutionTrace, window: int = 3) -> bool:
    """Check if the trace contains repeated identical actions."""
    if len(trace.steps) < window + 1:
        return False
    for i in range(len(trace.steps) - window):
        actions = [trace.steps[i + j].action_type for j in range(window + 1)]
        if len(set(actions)) == 1 and actions[0] == "edit":
            # Same action type repeated — check if content is similar
            contents = [trace.steps[i + j].action[:100] for j in range(window + 1)]
            if len(set(contents)) <= 2:
                return True
    return False


def classify_divergence(
    failed: ExecutionTrace,
    divergence_step: int,
    reference: ExecutionTrace | None = None,
) -> FailureClass:
    """Classify the failure at the divergence point.

    Rules (applied in priority order):
    1. INFINITE_LOOP: repeated actions detected near divergence
    2. PREMATURE_TERMINATION: failed trace is significantly shorter than reference
    3. WRONG_TOOL: different action_type at divergence vs reference
    4. WRONG_PARAMS: same action_type but different action content
    5. WRONG_INTERPRETATION: divergence follows a search/navigate (misread output)
    6. RETRIEVAL_FAILURE: divergence at or after a search step
    7. PLANNING_FAILURE: divergence is early (first 3 steps)
    """
    # Rule 1: Infinite loop
    if detect_infinite_loop(failed):
        return FailureClass.INFINITE_LOOP

    # Rule 2: Premature termination
    if reference and len(failed.steps) < len(reference.steps) * 0.5:
        return FailureClass.PREMATURE_TERMINATION
    if failed.exit_status in ("early_exit", "autosubmit"):
        return FailureClass.PREMATURE_TERMINATION

    # Need the divergence step to exist
    if divergence_step >= len(failed.steps):
        return FailureClass.UNKNOWN

    failed_step = failed.steps[divergence_step]

    # Rule 3: Wrong tool (different action type vs reference)
    if reference and divergence_step < len(reference.steps):
        ref_step = reference.steps[divergence_step]
        if failed_step.action_type != ref_step.action_type:
            return FailureClass.WRONG_TOOL

    # Rule 5: Wrong interpretation (previous step was search/navigate)
    if divergence_step > 0:
        prev_type = failed.steps[divergence_step - 1].action_type
        if prev_type in ("search", "navigate"):
            return FailureClass.WRONG_INTERPRETATION

    # Rule 6: Retrieval failure
    if failed_step.action_type == "search":
        return FailureClass.RETRIEVAL_FAILURE

    # Rule 7: Planning failure (early divergence)
    if divergence_step <= 2:
        return FailureClass.PLANNING_FAILURE

    return FailureClass.WRONG_PARAMS  # default fallback


# Example:
# fc = classify_divergence(failed_trace, div_step=7, reference=success_trace)
# → FailureClass.WRONG_TOOL
