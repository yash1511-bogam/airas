"""Intervention evolution engine — generates improved variants via LLM mutation.

Analyzes WHY current interventions fail on specific cases, then generates
targeted mutations. Uses DeepSeek V4-Flash for generation (~$0.003/mutation).
"""
import os
import logging
import httpx

logger = logging.getLogger(__name__)

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

MUTATION_PROMPT = """You generate improved intervention prompts for AI agent failure prevention.

CURRENT INTERVENTION (for failure class: {failure_class}):
"{current_payload}"

This intervention prevents failures {efficacy:.0%} of the time.

Here are 3 cases where it FAILED (agent still made the mistake despite seeing it):

CASE 1: {case_1}
CASE 2: {case_2}
CASE 3: {case_3}

What do these failure cases have in common that the current intervention doesn't address?

Generate 3 VARIANT interventions that:
1. Keep what works in the original
2. Add specific guidance for the failure pattern you identified
3. Are concise (under 100 words each)

Format exactly:
VARIANT_1: [text]
VARIANT_2: [text]
VARIANT_3: [text]"""


async def generate_mutations(
    current_payload: str,
    failure_class: str,
    efficacy: float,
    failed_cases: list[str],
) -> list[str]:
    """Generate 3 mutation variants of an intervention.

    Returns list of variant payloads. Falls back to empty list on failure.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        logger.warning("No DEEPSEEK_API_KEY, cannot generate mutations")
        return []

    cases = (failed_cases + ["(no additional case)"] * 3)[:3]
    prompt = MUTATION_PROMPT.format(
        failure_class=failure_class,
        current_payload=current_payload,
        efficacy=efficacy,
        case_1=cases[0][:300],
        case_2=cases[1][:300],
        case_3=cases[2][:300],
    )

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(DEEPSEEK_URL, json={
                "model": DEEPSEEK_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 500,
            }, headers={"Authorization": f"Bearer {api_key}"})
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]

        variants = []
        for i in range(1, 4):
            marker = f"VARIANT_{i}:"
            if marker in content:
                text = content.split(marker)[1].split("VARIANT_")[0].strip()
                text = text.split("\n")[0].strip() if "\n" in text else text.strip()
                if len(text) > 20:
                    variants.append(text)
        return variants

    except Exception as e:
        logger.error("Mutation generation failed: %s", e)
        return []


class EvolutionEngine:
    """Manages the intervention evolution lifecycle."""

    def __init__(self, min_trials: int = 20, promote_threshold: float = 0.65, retire_threshold: float = 0.20):
        self.min_trials = min_trials
        self.promote_threshold = promote_threshold
        self.retire_threshold = retire_threshold
        self._failure_log: dict[str, list[str]] = {}  # intervention_id → [case summaries]

    def log_failure(self, intervention_id: str, case_summary: str):
        """Record a case where an intervention failed."""
        if intervention_id not in self._failure_log:
            self._failure_log[intervention_id] = []
        self._failure_log[intervention_id].append(case_summary)
        # Keep last 20
        self._failure_log[intervention_id] = self._failure_log[intervention_id][-20:]

    def should_mutate(self, intervention_id: str, efficacy: float, trials: int) -> bool:
        """Determine if an intervention needs mutation."""
        if trials < self.min_trials:
            return False
        if efficacy < self.promote_threshold:
            return len(self._failure_log.get(intervention_id, [])) >= 3
        return False

    def should_retire(self, efficacy: float, trials: int) -> bool:
        """Determine if an intervention should be retired."""
        return trials >= self.min_trials and efficacy < self.retire_threshold

    def get_failure_cases(self, intervention_id: str, n: int = 3) -> list[str]:
        """Get recent failure cases for mutation analysis."""
        return self._failure_log.get(intervention_id, [])[-n:]
