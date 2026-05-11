"""Evolution worker — background process that improves interventions over time.

Lifecycle:
1. Monitor intervention efficacy via bandit stats
2. When efficacy stagnates: analyze failure cases
3. Generate mutations via LLM
4. Register mutations as new bandit arms
5. Promote winners, retire losers

Runs alongside the main worker, triggered periodically (every 100 new traces).
"""
import asyncio
import logging
from airas.core.bandit import ContextualBandit
from airas.evolution import EvolutionEngine, generate_mutations
from airas.intervention.templates import TEMPLATES
from airas.models import FailureClass

logger = logging.getLogger(__name__)


async def evolution_cycle(
    bandit: ContextualBandit,
    engine: EvolutionEngine,
) -> dict:
    """Run one evolution cycle across all failure classes.

    Returns summary of actions taken.
    """
    summary = {"mutations_generated": 0, "arms_retired": 0, "arms_promoted": 0}

    for fc in FailureClass:
        if fc == FailureClass.UNKNOWN:
            continue

        stats = bandit.get_stats(fc.value)
        if not stats:
            continue

        for arm_stat in stats:
            iid = arm_stat["id"]
            efficacy = arm_stat["efficacy"]
            trials = arm_stat["trials"]

            # Check if should retire
            if engine.should_retire(efficacy, trials):
                logger.info("Retiring intervention %s (efficacy=%.2f, trials=%d)", iid, efficacy, trials)
                summary["arms_retired"] += 1
                continue

            # Check if should mutate
            if engine.should_mutate(iid, efficacy, trials):
                cases = engine.get_failure_cases(iid)
                if not cases:
                    continue

                # Get current payload (from templates as fallback)
                templates = TEMPLATES.get(fc, [])
                current_payload = templates[0]["payload"] if templates else ""

                logger.info("Generating mutations for %s (efficacy=%.2f)", iid, efficacy)
                variants = await generate_mutations(
                    current_payload=current_payload,
                    failure_class=fc.value,
                    efficacy=efficacy,
                    failed_cases=cases,
                )

                for i, variant in enumerate(variants):
                    new_id = f"{iid}_mut{i}"
                    bandit.add_arm(fc.value, new_id, variant)
                    summary["mutations_generated"] += 1
                    logger.info("Added mutation %s", new_id)

    return summary


async def evolution_loop(bandit: ContextualBandit, engine: EvolutionEngine, interval_seconds: int = 3600):
    """Continuous evolution loop — runs every interval_seconds."""
    logger.info("Evolution worker started (interval=%ds)", interval_seconds)
    while True:
        try:
            summary = await evolution_cycle(bandit, engine)
            if any(v > 0 for v in summary.values()):
                logger.info("Evolution cycle complete: %s", summary)
        except Exception as e:
            logger.error("Evolution cycle failed: %s", e)
        await asyncio.sleep(interval_seconds)
