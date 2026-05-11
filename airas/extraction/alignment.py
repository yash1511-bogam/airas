"""Structural alignment of failed vs successful traces.

Uses edit distance on action_type sequences to find the first divergence point.
Needleman-Wunsch is overkill for this — simple LCS-based alignment suffices.
"""
from airas.models import ExecutionTrace


def action_sequence(trace: ExecutionTrace) -> list[str]:
    """Extract the action_type sequence from a trace."""
    return [s.action_type for s in trace.steps]


def find_divergence_point(
    failed: ExecutionTrace,
    successful: ExecutionTrace,
) -> int:
    """Find the first step where the failed trace diverges from the successful one.

    Compares action_type sequences. Returns the step index in the failed trace
    where behavior first differs from the successful trace.
    """
    f_seq = action_sequence(failed)
    s_seq = action_sequence(successful)
    min_len = min(len(f_seq), len(s_seq))

    for i in range(min_len):
        if f_seq[i] != s_seq[i]:
            return i

    # If one is shorter, divergence is at the end of the shorter one
    if len(f_seq) < len(s_seq):
        return len(f_seq) - 1  # premature termination
    if len(f_seq) > len(s_seq):
        return len(s_seq)  # extra steps (possible loop)

    return max(0, min_len - 1)


def find_best_divergence(
    failed: ExecutionTrace,
    successes: list[ExecutionTrace],
) -> tuple[int, ExecutionTrace | None]:
    """Find divergence against the best-matching successful trace for the same issue.

    Picks the successful trace with the longest common prefix (most similar path).
    Returns (divergence_step, best_matching_success_trace).
    """
    same_issue = [s for s in successes if s.instance_id == failed.instance_id]
    if not same_issue:
        # Fallback: compare against all successes, pick longest common prefix
        same_issue = successes[:10]

    best_div = 0
    best_trace = None
    for s in same_issue:
        div = find_divergence_point(failed, s)
        if div > best_div:
            best_div = div
            best_trace = s

    return best_div, best_trace


# Example:
# div_step, ref = find_best_divergence(failed_trace, success_traces)
# → div_step=7, meaning steps 0-6 match, step 7 is where failure begins
