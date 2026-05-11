"""Intervention template bank.

Hardcoded templates for each failure archetype. Each template specifies:
- WHERE to inject (relative to divergence point)
- WHAT to inject (the instruction/gate)
- HOW to verify success

These are prompt-level interventions — they modify the agent's instructions
at a specific step to prevent the failure pattern from manifesting.
"""
from airas.models import FailureClass, Intervention, InterventionType, FailureAntigen
import hashlib


TEMPLATES: dict[FailureClass, list[dict]] = {
    FailureClass.WRONG_TOOL: [
        {
            "type": InterventionType.GATE,
            "offset": 0,  # inject AT the divergence step
            "payload": (
                "STOP. Before executing your next action, verify: "
                "Is the tool you're about to use the correct one for this specific sub-goal? "
                "Consider what information you need and which tool provides it. "
                "If you're about to edit a file, confirm you've identified the correct location first."
            ),
        },
    ],
    FailureClass.WRONG_PARAMS: [
        {
            "type": InterventionType.CONSTRAINT,
            "offset": 0,
            "payload": (
                "VALIDATION CHECK: Before executing, verify your parameters: "
                "1) Is the file path correct? 2) Are line numbers accurate based on what you last saw? "
                "3) Does your edit preserve correct indentation and syntax? "
                "Re-read the file content above to confirm."
            ),
        },
    ],
    FailureClass.WRONG_INTERPRETATION: [
        {
            "type": InterventionType.CHECKPOINT,
            "offset": 0,
            "payload": (
                "INTERPRETATION CHECK: Re-read the output from your last action carefully. "
                "What does it actually tell you? Does your planned next step logically follow? "
                "State explicitly what you learned before proceeding."
            ),
        },
    ],
    FailureClass.PREMATURE_TERMINATION: [
        {
            "type": InterventionType.GATE,
            "offset": -1,  # inject one step BEFORE the premature submit
            "payload": (
                "COMPLETION CHECK: Before submitting, verify ALL of the following: "
                "1) Have you reproduced the bug and confirmed your fix resolves it? "
                "2) Have you run the relevant tests? "
                "3) Does your patch handle edge cases mentioned in the issue? "
                "Do NOT submit until all checks pass."
            ),
        },
    ],
    FailureClass.INFINITE_LOOP: [
        {
            "type": InterventionType.FALLBACK,
            "offset": 0,
            "payload": (
                "LOOP DETECTED: You have repeated similar actions multiple times without progress. "
                "STOP and try a fundamentally different approach: "
                "1) Re-read the error message carefully "
                "2) Search for similar patterns in the codebase "
                "3) Consider if your mental model of the code is wrong "
                "Do NOT repeat the same edit again."
            ),
        },
    ],
    FailureClass.RETRIEVAL_FAILURE: [
        {
            "type": InterventionType.REWRITE,
            "offset": 0,
            "payload": (
                "SEARCH REFINEMENT: Your search did not find what you need. "
                "Try: 1) Different keywords (use class/function names from the error) "
                "2) Broader directory scope 3) grep for specific strings from the traceback "
                "4) Look at imports to find the right module."
            ),
        },
    ],
    FailureClass.PLANNING_FAILURE: [
        {
            "type": InterventionType.REWRITE,
            "offset": 0,
            "payload": (
                "PLAN VALIDATION: Before proceeding, state your plan explicitly: "
                "1) What is the root cause of this issue? "
                "2) What specific code change will fix it? "
                "3) How will you verify the fix works? "
                "If you cannot answer all three, gather more information first."
            ),
        },
    ],
    FailureClass.UNKNOWN: [
        {
            "type": InterventionType.CHECKPOINT,
            "offset": 0,
            "payload": (
                "CHECKPOINT: Pause and assess. Is your current approach making progress? "
                "If not, reconsider your strategy."
            ),
        },
    ],
}


def get_intervention(antigen: FailureAntigen) -> Intervention:
    """Select and parameterize an intervention for a matched antigen."""
    templates = TEMPLATES.get(antigen.failure_class, TEMPLATES[FailureClass.UNKNOWN])
    template = templates[0]  # For v1, always pick first. Thompson Sampling in v2.

    applies_at = antigen.divergence_step + template["offset"]
    iid = hashlib.md5(f"{antigen.antigen_id}_{template['type'].value}".encode()).hexdigest()[:10]

    return Intervention(
        intervention_id=iid,
        antigen_id=antigen.antigen_id,
        intervention_type=template["type"],
        applies_at_step=max(0, applies_at),
        payload=template["payload"],
        failure_class=antigen.failure_class,
    )


# Example:
# intervention = get_intervention(antigen)
# → Intervention(type=GATE, applies_at_step=7, payload="STOP. Before executing...")
