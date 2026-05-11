"""Hybrid replay judge: heuristic for easy classes, LLM for hard classes.

Routes to LLM judge only when heuristic prevention <30% for the failure class.
LLM judge uses DeepSeek V4-Flash (~$0.00027 per judgment).
"""
import os
import logging
import httpx
from airas.models import ExecutionTrace, FailureClass
from airas.intervention.templates import Intervention
from airas.replay.engine import judge_intervention_effectiveness

logger = logging.getLogger(__name__)

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

# Classes where heuristic works well — skip LLM
HEURISTIC_CLASSES = {FailureClass.WRONG_TOOL, FailureClass.INFINITE_LOOP, FailureClass.WRONG_PARAMS}

JUDGE_PROMPT = """You are an expert evaluator of AI agent behavior. Assess whether a specific intervention would have changed an agent's execution to produce success.

TASK DESCRIPTION:
{task_description}

FAILURE CLASS: {failure_class}
FAILURE AT STEP: {divergence_step} of {total_steps}

TRACE CONTEXT (steps around failure):
{trace_context}

PROPOSED INTERVENTION (injected at step {injection_step}):
"{intervention_payload}"

If this intervention had been shown to the agent at step {injection_step}, would the agent have succeeded?

Consider:
1. Does the intervention address the ACTUAL reason for failure?
2. Would the agent have the information needed to act on it?
3. Is there evidence the agent COULD have succeeded?
4. Are there OTHER failure points downstream?

Answer ONLY:
VERDICT: YES or NO
CONFIDENCE: HIGH, MEDIUM, or LOW
REASONING: (2-3 sentences)"""


def _build_trace_context(trace: ExecutionTrace, step: int, window: int = 3) -> str:
    start = max(0, step - window)
    end = min(len(trace.steps), step + window + 1)
    lines = []
    for i in range(start, end):
        s = trace.steps[i]
        marker = ">>>" if i == step else "   "
        lines.append(f"{marker} [{i}] {s.action_type}: {s.action[:100]}")
        if s.observation:
            lines.append(f"       obs: {s.observation[:80]}")
    return "\n".join(lines)


async def llm_judge(
    trace: ExecutionTrace,
    intervention: Intervention,
    task_description: str = "",
) -> tuple[bool, str, str]:
    """Call DeepSeek V4-Flash to judge intervention effectiveness.

    Returns: (would_prevent, confidence, reasoning)
    """
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        logger.warning("No DEEPSEEK_API_KEY set, falling back to heuristic")
        prevented, reason = judge_intervention_effectiveness(trace, intervention)
        return prevented, "LOW", reason

    context = _build_trace_context(trace, intervention.applies_at_step)
    prompt = JUDGE_PROMPT.format(
        task_description=task_description or trace.instance_id,
        failure_class=intervention.failure_class.value,
        divergence_step=intervention.applies_at_step,
        total_steps=len(trace.steps),
        trace_context=context,
        injection_step=intervention.applies_at_step,
        intervention_payload=intervention.payload,
    )

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(DEEPSEEK_URL, json={
                "model": DEEPSEEK_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 200,
            }, headers={"Authorization": f"Bearer {api_key}"})
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]

        verdict = "YES" in content.split("VERDICT:")[-1].split("\n")[0].upper() if "VERDICT:" in content else False
        confidence = "MEDIUM"
        if "CONFIDENCE:" in content:
            conf_line = content.split("CONFIDENCE:")[-1].split("\n")[0].strip()
            if "HIGH" in conf_line:
                confidence = "HIGH"
            elif "LOW" in conf_line:
                confidence = "LOW"
        reasoning = content.split("REASONING:")[-1].strip()[:200] if "REASONING:" in content else content[-200:]
        return verdict, confidence, reasoning

    except Exception as e:
        logger.error("LLM judge call failed: %s", e)
        prevented, reason = judge_intervention_effectiveness(trace, intervention)
        return prevented, "LOW", reason


async def hybrid_judge(
    trace: ExecutionTrace,
    intervention: Intervention,
    task_description: str = "",
) -> tuple[bool, str]:
    """Route to heuristic or LLM judge based on failure class."""
    if intervention.failure_class in HEURISTIC_CLASSES:
        prevented, reason = judge_intervention_effectiveness(trace, intervention)
        return prevented, reason

    # Hard classes → LLM judge
    verdict, confidence, reasoning = await llm_judge(trace, intervention, task_description)
    return verdict, f"[LLM/{confidence}] {reasoning}"
