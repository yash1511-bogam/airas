"""Seed Qdrant with the 7 validated patterns from the killer experiment.

Run: python -m airas.scripts.seed_antigens
Requires: Qdrant running (docker compose up qdrant)
"""
from airas.storage.qdrant import upsert_antigen, count_antigens

# The 7 patterns discovered in the experiment, manually curated
SEED_ANTIGENS = [
    {
        "antigen_id": "seed_001",
        "signature": "failure:wrong_params | at_step_ratio:0.35 | prev:search | action:edit | after_error:false",
        "failure_class": "wrong_params",
        "conditions": {"ratio_through_trace": 0.35, "prev_action_type": "search", "action_type_at_divergence": "edit"},
        "danger_score": 0.6,
    },
    {
        "antigen_id": "seed_002",
        "signature": "failure:wrong_params | at_step_ratio:0.5 | prev:navigate | action:edit | after_error:true",
        "failure_class": "wrong_params",
        "conditions": {"ratio_through_trace": 0.5, "prev_action_type": "navigate", "prev_had_error": True, "action_type_at_divergence": "edit"},
        "danger_score": 0.7,
    },
    {
        "antigen_id": "seed_003",
        "signature": "failure:wrong_params | at_step_ratio:0.25 | prev:navigate | action:edit | after_error:false",
        "failure_class": "wrong_params",
        "conditions": {"ratio_through_trace": 0.25, "prev_action_type": "navigate", "action_type_at_divergence": "edit"},
        "danger_score": 0.5,
    },
    {
        "antigen_id": "seed_004",
        "signature": "failure:wrong_params | at_step_ratio:0.4 | prev:run_test | action:edit | after_error:true",
        "failure_class": "wrong_params",
        "conditions": {"ratio_through_trace": 0.4, "prev_action_type": "run_test", "prev_had_error": True, "action_type_at_divergence": "edit"},
        "danger_score": 0.7,
    },
    {
        "antigen_id": "seed_005",
        "signature": "failure:premature_termination | at_step_ratio:0.6 | prev:run_test | action:submit",
        "failure_class": "premature_termination",
        "conditions": {"ratio_through_trace": 0.6, "prev_action_type": "run_test", "action_type_at_divergence": "submit"},
        "danger_score": 0.8,
    },
    {
        "antigen_id": "seed_006",
        "signature": "failure:premature_termination | at_step_ratio:0.45 | prev:edit | action:submit",
        "failure_class": "premature_termination",
        "conditions": {"ratio_through_trace": 0.45, "prev_action_type": "edit", "action_type_at_divergence": "submit"},
        "danger_score": 0.8,
    },
    {
        "antigen_id": "seed_007",
        "signature": "failure:premature_termination | at_step_ratio:0.3 | prev:navigate | action:submit",
        "failure_class": "premature_termination",
        "conditions": {"ratio_through_trace": 0.3, "prev_action_type": "navigate", "action_type_at_divergence": "submit"},
        "danger_score": 0.9,
    },
]


def seed():
    for a in SEED_ANTIGENS:
        upsert_antigen(**a)
    print(f"Seeded {len(SEED_ANTIGENS)} antigens. Total in Qdrant: {count_antigens()}")


if __name__ == "__main__":
    seed()
