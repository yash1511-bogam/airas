"""SWE-bench trajectory loader.

Loads from HuggingFace `nebius/SWE-agent-trajectories` (80K trajectories, CC-BY-4.0).
Format per row: instance_id, model_name, target (bool), trajectory (list of steps),
exit_status, generated_patch, eval_logs.

Each trajectory step has: action, observation, thought, response, state.
"""
import hashlib
import re
from datasets import load_dataset
from airas.models import ExecutionTrace, ExecutionStep


# Map raw SWE-agent actions to normalized action types
ACTION_TYPE_MAP = {
    "edit": "edit",
    "create": "edit",
    "open": "navigate",
    "goto": "navigate",
    "scroll_down": "navigate",
    "scroll_up": "navigate",
    "find_file": "search",
    "search_file": "search",
    "search_dir": "search",
    "find": "search",
    "grep": "search",
    "ls": "navigate",
    "cd": "navigate",
    "cat": "navigate",
    "python": "run_test",
    "pytest": "run_test",
    "pip": "setup",
    "submit": "submit",
}


def classify_action(action: str) -> str:
    """Classify a raw action string into a normalized action type."""
    action_lower = action.strip().split("\n")[0].lower()
    for prefix, atype in ACTION_TYPE_MAP.items():
        if action_lower.startswith(prefix):
            return atype
    if "test" in action_lower or "python" in action_lower:
        return "run_test"
    if "edit" in action_lower:
        return "edit"
    return "other"


def normalize_step(raw_step: dict, index: int) -> ExecutionStep:
    """Convert a raw SWE-agent trajectory step to ExecutionStep."""
    action = raw_step.get("action", "")
    observation = raw_step.get("observation", "")
    thought = raw_step.get("thought", "") or raw_step.get("response", "")
    state = {}
    if raw_step.get("state"):
        try:
            import json
            state = json.loads(raw_step["state"]) if isinstance(raw_step["state"], str) else raw_step["state"]
        except (json.JSONDecodeError, TypeError):
            state = {}

    return ExecutionStep(
        index=index,
        action=action[:2000],  # cap for memory
        action_type=classify_action(action),
        observation=observation[:500],
        thought=thought[:500],
        state=state,
    )


def load_swebench_trajectories(
    split: str = "train",
    max_rows: int = 500,
    model_filter: str | None = None,
) -> list[ExecutionTrace]:
    """Load and normalize SWE-bench trajectories from HuggingFace.

    Returns list of ExecutionTrace objects with success/failure labels.
    """
    ds = load_dataset("nebius/SWE-agent-trajectories", split=split, streaming=True)
    traces = []
    for i, row in enumerate(ds):
        if i >= max_rows:
            break
        if model_filter and row.get("model_name") != model_filter:
            continue
        raw_traj = row.get("trajectory", [])
        if not raw_traj or len(raw_traj) < 2:
            continue

        steps = [normalize_step(s, idx) for idx, s in enumerate(raw_traj)]
        trace_id = hashlib.md5(f"{row['instance_id']}_{row.get('model_name','')}_{i}".encode()).hexdigest()[:12]

        traces.append(ExecutionTrace(
            trace_id=trace_id,
            instance_id=row["instance_id"],
            model_name=row.get("model_name", ""),
            steps=steps,
            success=bool(row.get("target", False)),
            exit_status=row.get("exit_status", "") or "",
            patch=(row.get("generated_patch") or "")[:5000],
        ))
    return traces


def split_train_test(
    traces: list[ExecutionTrace],
    train_fail: int = 100,
    train_pass: int = 100,
    test_fail: int = 50,
) -> tuple[list[ExecutionTrace], list[ExecutionTrace], list[ExecutionTrace]]:
    """Split traces into train (fail+pass) and test (fail) sets.

    Returns: (train_failures, train_successes, test_failures)
    """
    failures = [t for t in traces if not t.success]
    successes = [t for t in traces if t.success]

    train_f = failures[:train_fail]
    test_f = failures[train_fail:train_fail + test_fail]
    train_s = successes[:train_pass]

    return train_f, train_s, test_f


# --- Example usage ---
# traces = load_swebench_trajectories(max_rows=400)
# train_f, train_s, test_f = split_train_test(traces)
# print(f"Train: {len(train_f)} failures, {len(train_s)} successes. Test: {len(test_f)} failures")
