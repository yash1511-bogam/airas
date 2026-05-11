"""Domain-specific intervention templates for support, research, and data pipeline agents."""
from airas.models import FailureClass

DOMAIN_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "support": {
        FailureClass.PREMATURE_TERMINATION.value: [
            "Before closing this ticket: 1) Has the customer explicitly confirmed resolution? 2) Have you addressed ALL issues mentioned? 3) Is there a follow-up action needed?",
        ],
        FailureClass.WRONG_PARAMS.value: [
            "Verify: Is this the correct customer account? Does the order ID match? Is the policy you're citing applicable to this customer's tier/region?",
        ],
        FailureClass.WRONG_INTERPRETATION.value: [
            "Re-read the customer's last message. What are they ACTUALLY asking for? Are you answering their question or a different one?",
        ],
        FailureClass.RETRIEVAL_FAILURE.value: [
            "Your KB search didn't find a match. Try: 1) Different keywords from the customer's message 2) Check product-specific documentation 3) Search by error code if provided.",
        ],
        FailureClass.PLANNING_FAILURE.value: [
            "Before responding: What is the customer's core issue? What information do you need? What's the resolution path? If unsure, ask a clarifying question instead of guessing.",
        ],
    },
    "research": {
        FailureClass.PREMATURE_TERMINATION.value: [
            "Before finalizing: 1) Have you consulted at least 3 independent sources? 2) Have you addressed counterarguments? 3) Are all claims backed by citations?",
        ],
        FailureClass.RETRIEVAL_FAILURE.value: [
            "Your search returned irrelevant results. Try: 1) More specific technical terms 2) Author names from known papers 3) Different databases 4) Broader then narrower queries.",
        ],
        FailureClass.WRONG_INTERPRETATION.value: [
            "Re-read the source material. Does it actually support your claim? Quote the specific passage. If it's ambiguous, note the uncertainty.",
        ],
        FailureClass.PLANNING_FAILURE.value: [
            "State your research question explicitly. What would CHANGE your conclusion? Search for that disconfirming evidence before synthesizing.",
        ],
        FailureClass.INFINITE_LOOP.value: [
            "You've searched similar terms 3+ times. STOP. Try a fundamentally different approach: different source type, different angle, or acknowledge the gap.",
        ],
    },
    "data_pipeline": {
        FailureClass.WRONG_PARAMS.value: [
            "Before executing this transform: 1) Run DESCRIBE on the source table 2) Verify column types match expectations 3) Check for NULLs in join keys.",
        ],
        FailureClass.PREMATURE_TERMINATION.value: [
            "Before marking complete: 1) Row count sanity check (before vs after) 2) NULL percentage in output 3) Value range validation on key columns.",
        ],
        FailureClass.PLANNING_FAILURE.value: [
            "Before building the pipeline: What's the expected output schema? What are the data quality requirements? What happens if a source is unavailable?",
        ],
        FailureClass.INFINITE_LOOP.value: [
            "API call failed 3 times. STOP retrying. Log the error, skip this record, and continue. Alert on >5% skip rate.",
        ],
        FailureClass.WRONG_INTERPRETATION.value: [
            "The query returned unexpected results. Before proceeding: check row count, inspect 5 sample rows, verify the JOIN didn't create duplicates.",
        ],
    },
}


def get_domain_template(domain: str, failure_class: str) -> str | None:
    """Get domain-specific intervention template. Returns None if not available."""
    domain_templates = DOMAIN_TEMPLATES.get(domain, {})
    templates = domain_templates.get(failure_class, [])
    return templates[0] if templates else None
