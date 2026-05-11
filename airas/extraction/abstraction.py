"""Pattern abstraction: convert specific divergence into generalizable antigen.

Key operation: strip instance-specific details, keep structural pattern.
The signature string must be embeddable and matchable across different issues.
"""
import hashlib
from airas.models import ExecutionTrace, ExecutionStep, FailureClass, FailureAntigen


def abstract_step(step: ExecutionStep) -> str:
    """Abstract a single step into a type-level description."""
    parts = [step.action_type]
    # Add structural info without instance-specific content
    if step.action_type == "edit":
        if "def " in step.action or "class " in step.action:
            parts.append("function_or_class")
        elif "import" in step.action:
            parts.append("import_statement")
        else:
            parts.append("code_modification")
    elif step.action_type == "search":
        if "test" in step.action.lower():
            parts.append("test_related")
        else:
            parts.append("code_search")
    elif step.action_type == "run_test":
        if "FAILED" in step.observation or "Error" in step.observation:
            parts.append("test_failed")
        elif "passed" in step.observation.lower():
            parts.append("test_passed")
    return ":".join(parts)


def build_context_window(trace: ExecutionTrace, div_step: int, window: int = 2) -> str:
    """Build a context string from steps around the divergence point."""
    start = max(0, div_step - window)
    end = min(len(trace.steps), div_step + window + 1)
    parts = []
    for i in range(start, end):
        marker = ">>>" if i == div_step else "   "
        parts.append(f"{marker} step[{i}]: {abstract_step(trace.steps[i])}")
    return "\n".join(parts)


def extract_conditions(trace: ExecutionTrace, div_step: int) -> dict:
    """Extract triggering conditions at the divergence point."""
    conditions = {}
    step = trace.steps[div_step]
    conditions["action_type_at_divergence"] = step.action_type
    conditions["trace_length_at_divergence"] = div_step
    conditions["total_steps"] = len(trace.steps)
    conditions["ratio_through_trace"] = round(div_step / max(len(trace.steps), 1), 2)

    # Check what happened in the step before
    if div_step > 0:
        prev = trace.steps[div_step - 1]
        conditions["prev_action_type"] = prev.action_type
        conditions["prev_had_error"] = "error" in prev.observation.lower() or "Error" in prev.observation

    return conditions


def extract_antigen(
    failed: ExecutionTrace,
    divergence_step: int,
    failure_class: FailureClass,
) -> FailureAntigen:
    """Extract a FailureAntigen from a classified divergence.

    The signature is a human-readable abstract pattern that can be embedded
    for similarity matching against future partial traces.
    """
    context = build_context_window(failed, divergence_step)
    conditions = extract_conditions(failed, divergence_step)

    # Build signature: failure_class + context pattern + conditions summary
    sig_parts = [
        f"failure:{failure_class.value}",
        f"at_step_ratio:{conditions['ratio_through_trace']}",
        f"prev:{conditions.get('prev_action_type', 'none')}",
        f"action:{conditions['action_type_at_divergence']}",
    ]
    if conditions.get("prev_had_error"):
        sig_parts.append("after_error:true")

    signature = " | ".join(sig_parts)
    antigen_id = hashlib.md5(f"{signature}_{failed.trace_id}".encode()).hexdigest()[:10]

    return FailureAntigen(
        antigen_id=antigen_id,
        failure_class=failure_class,
        divergence_step=divergence_step,
        signature=signature,
        conditions=conditions,
        source_trace_ids=[failed.trace_id],
    )


# Example:
# antigen = extract_antigen(failed_trace, div_step=7, failure_class=FailureClass.WRONG_TOOL)
# antigen.signature → "failure:wrong_tool | at_step_ratio:0.35 | prev:search | action:edit"
