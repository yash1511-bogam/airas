"""Predictive immunity — predict likely failure classes from task description alone.

Inspired by SafetyDrift (arXiv:2603.27148): predict failures BEFORE they happen.
Uses a simple keyword/pattern classifier trained on historical (task → failure_class) pairs.
Pre-loads relevant interventions before the agent even starts.
"""
import re
from collections import Counter
from airas.models import FailureClass


# Keyword signals that predict specific failure classes
PREDICTIVE_SIGNALS: dict[str, list[tuple[str, float]]] = {
    FailureClass.WRONG_PARAMS.value: [
        (r"edit.*file|modify.*code|change.*line|fix.*bug", 0.4),
        (r"indentation|syntax|import", 0.3),
    ],
    FailureClass.PREMATURE_TERMINATION.value: [
        (r"multiple.*files|several.*changes|all.*tests", 0.4),
        (r"comprehensive|thorough|complete", 0.3),
    ],
    FailureClass.RETRIEVAL_FAILURE.value: [
        (r"find.*where|locate.*function|search.*for", 0.4),
        (r"unfamiliar.*codebase|large.*project", 0.3),
    ],
    FailureClass.PLANNING_FAILURE.value: [
        (r"refactor|redesign|architect|migrate", 0.5),
        (r"complex.*interaction|multiple.*components", 0.3),
    ],
    FailureClass.INFINITE_LOOP.value: [
        (r"flaky|intermittent|sometimes.*fails", 0.3),
        (r"race.*condition|timing", 0.2),
    ],
}


class FailurePredictor:
    """Predicts likely failure classes from task description.

    Uses keyword matching for v1. In v2, train a proper classifier
    on (task_description, observed_failure_class) pairs from production data.
    """

    def __init__(self):
        self._history: list[tuple[str, str]] = []  # (task_keywords, failure_class)

    def predict(self, task_description: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Predict top-K likely failure classes with confidence scores.

        Returns: [(failure_class, probability), ...]
        """
        scores: dict[str, float] = {}
        text = task_description.lower()

        for fc, patterns in PREDICTIVE_SIGNALS.items():
            score = 0.0
            for pattern, weight in patterns:
                if re.search(pattern, text):
                    score += weight
            if score > 0:
                scores[fc] = min(score, 1.0)

        # Sort by score, return top-K
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:top_k]

    def learn(self, task_description: str, actual_failure_class: str):
        """Record an observation for future prediction improvement."""
        keywords = " ".join(re.findall(r'\b\w{4,}\b', task_description.lower())[:20])
        self._history.append((keywords, actual_failure_class))

    def get_preload_interventions(self, task_description: str) -> list[str]:
        """Get failure classes that should have interventions pre-loaded."""
        predictions = self.predict(task_description)
        return [fc for fc, prob in predictions if prob >= 0.3]
