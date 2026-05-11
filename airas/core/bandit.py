"""Contextual Thompson Sampling bandit for intervention selection.

Selects the best intervention variant based on execution context features.
Each arm (intervention variant) has per-context-bucket Beta(α, β) parameters.
"""
import random
import math
from dataclasses import dataclass, field


@dataclass
class Arm:
    intervention_id: str
    payload: str
    alpha: float = 1.0  # successes + prior
    beta: float = 1.0   # failures + prior

    @property
    def efficacy(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def sample(self) -> float:
        return random.betavariate(self.alpha, self.beta)

    def update(self, reward: bool):
        if reward:
            self.alpha += 1.0
        else:
            self.beta += 1.0


def _context_bucket(step_ratio: float, prev_action: str, has_error: bool, trace_len: int) -> str:
    """Discretize context into a bucket key."""
    ratio_bin = "early" if step_ratio < 0.33 else ("mid" if step_ratio < 0.66 else "late")
    len_bin = "short" if trace_len < 10 else ("med" if trace_len < 30 else "long")
    err = "err" if has_error else "ok"
    return f"{ratio_bin}:{prev_action}:{err}:{len_bin}"


class ContextualBandit:
    """Per-failure-class contextual bandit for intervention selection."""

    def __init__(self):
        # {failure_class: {context_bucket: [Arm, ...]}}
        self._arms: dict[str, dict[str, list[Arm]]] = {}

    def add_arm(self, failure_class: str, intervention_id: str, payload: str):
        """Register a new intervention variant."""
        if failure_class not in self._arms:
            self._arms[failure_class] = {}
        for bucket_arms in self._arms[failure_class].values():
            if not any(a.intervention_id == intervention_id for a in bucket_arms):
                bucket_arms.append(Arm(intervention_id=intervention_id, payload=payload))

    def select(
        self,
        failure_class: str,
        step_ratio: float,
        prev_action: str,
        has_error: bool,
        trace_len: int,
    ) -> Arm | None:
        """Select best arm via Thompson Sampling for the given context."""
        if failure_class not in self._arms:
            return None

        bucket = _context_bucket(step_ratio, prev_action, has_error, trace_len)
        arms = self._arms[failure_class].get(bucket)

        if not arms:
            # Fall back to global arms (no context-specific data yet)
            all_arms = []
            for b_arms in self._arms[failure_class].values():
                all_arms.extend(b_arms)
            if not all_arms:
                return None
            # Deduplicate
            seen = set()
            arms = []
            for a in all_arms:
                if a.intervention_id not in seen:
                    arms.append(a)
                    seen.add(a.intervention_id)

        # Thompson Sampling: sample from each arm's Beta distribution, pick highest
        best = max(arms, key=lambda a: a.sample())
        return best

    def update(
        self,
        failure_class: str,
        intervention_id: str,
        reward: bool,
        step_ratio: float,
        prev_action: str,
        has_error: bool,
        trace_len: int,
    ):
        """Update arm with observed reward. Auto-creates arm if not exists."""
        if failure_class not in self._arms:
            self._arms[failure_class] = {}

        bucket = _context_bucket(step_ratio, prev_action, has_error, trace_len)
        if bucket not in self._arms[failure_class]:
            self._arms[failure_class][bucket] = []

        # Find or create arm in this bucket
        arms = self._arms[failure_class][bucket]
        arm = next((a for a in arms if a.intervention_id == intervention_id), None)
        if arm is None:
            # Check other buckets first
            for b_arms in self._arms[failure_class].values():
                source = next((a for a in b_arms if a.intervention_id == intervention_id), None)
                if source:
                    arm = Arm(intervention_id=intervention_id, payload=source.payload)
                    arms.append(arm)
                    break
            if arm is None:
                # Brand new arm
                arm = Arm(intervention_id=intervention_id, payload="")
                arms.append(arm)
        arm.update(reward)

    def get_stats(self, failure_class: str) -> list[dict]:
        """Get efficacy stats for all arms in a failure class."""
        if failure_class not in self._arms:
            return []
        # Aggregate across buckets
        totals: dict[str, Arm] = {}
        for bucket_arms in self._arms[failure_class].values():
            for arm in bucket_arms:
                if arm.intervention_id not in totals:
                    totals[arm.intervention_id] = Arm(intervention_id=arm.intervention_id, payload=arm.payload)
                totals[arm.intervention_id].alpha += arm.alpha - 1
                totals[arm.intervention_id].beta += arm.beta - 1
        return [{"id": a.intervention_id, "efficacy": a.efficacy, "trials": a.alpha + a.beta - 2}
                for a in totals.values()]
