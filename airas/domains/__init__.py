"""Universal domain adapter framework for cross-domain antigen transfer.

Maps domain-specific tool calls to 6 universal action types:
SEARCH, EDIT, VERIFY, RESPOND, PLAN, DONE

This enables patterns learned in one domain (e.g., coding) to immunize
agents in another domain (e.g., customer support).
"""
from abc import ABC, abstractmethod
from airas.models import ExecutionStep


class UniversalAction:
    SEARCH = "SEARCH"
    EDIT = "EDIT"
    VERIFY = "VERIFY"
    RESPOND = "RESPOND"
    PLAN = "PLAN"
    DONE = "DONE"
    OTHER = "OTHER"


class DomainAdapter(ABC):
    """Base class for domain-specific adapters."""

    @property
    @abstractmethod
    def domain_name(self) -> str: ...

    @abstractmethod
    def map_action(self, action_type: str, action: str) -> str:
        """Map a domain-specific action to a universal action type."""
        ...

    def normalize_step(self, step: ExecutionStep) -> ExecutionStep:
        """Return step with action_type mapped to universal type."""
        universal = self.map_action(step.action_type, step.action)
        return ExecutionStep(
            index=step.index,
            action=step.action,
            action_type=universal,
            observation=step.observation,
            thought=step.thought,
            state=step.state,
        )


class CodingAdapter(DomainAdapter):
    """SWE-bench / coding agent adapter."""
    domain_name = "coding"

    def map_action(self, action_type: str, action: str) -> str:
        return {
            "edit": UniversalAction.EDIT,
            "search": UniversalAction.SEARCH,
            "navigate": UniversalAction.SEARCH,
            "run_test": UniversalAction.VERIFY,
            "submit": UniversalAction.DONE,
            "setup": UniversalAction.PLAN,
        }.get(action_type, UniversalAction.OTHER)


class SupportAdapter(DomainAdapter):
    """Customer support agent adapter."""
    domain_name = "support"

    def map_action(self, action_type: str, action: str) -> str:
        mapping = {
            "search_kb": UniversalAction.SEARCH,
            "lookup_customer": UniversalAction.SEARCH,
            "lookup_order": UniversalAction.SEARCH,
            "draft_reply": UniversalAction.RESPOND,
            "send_reply": UniversalAction.RESPOND,
            "escalate": UniversalAction.DONE,
            "close_ticket": UniversalAction.DONE,
            "update_ticket": UniversalAction.EDIT,
            "check_policy": UniversalAction.VERIFY,
            "summarize": UniversalAction.PLAN,
        }
        return mapping.get(action_type, UniversalAction.OTHER)


class ResearchAdapter(DomainAdapter):
    """Research/analysis agent adapter."""
    domain_name = "research"

    def map_action(self, action_type: str, action: str) -> str:
        mapping = {
            "web_search": UniversalAction.SEARCH,
            "read_paper": UniversalAction.SEARCH,
            "read_url": UniversalAction.SEARCH,
            "take_notes": UniversalAction.EDIT,
            "synthesize": UniversalAction.RESPOND,
            "write_report": UniversalAction.RESPOND,
            "verify_claim": UniversalAction.VERIFY,
            "outline": UniversalAction.PLAN,
            "submit_report": UniversalAction.DONE,
        }
        return mapping.get(action_type, UniversalAction.OTHER)


class DataPipelineAdapter(DomainAdapter):
    """Data pipeline / ETL agent adapter."""
    domain_name = "data_pipeline"

    def map_action(self, action_type: str, action: str) -> str:
        mapping = {
            "query_db": UniversalAction.SEARCH,
            "read_schema": UniversalAction.SEARCH,
            "transform": UniversalAction.EDIT,
            "write_table": UniversalAction.EDIT,
            "validate": UniversalAction.VERIFY,
            "run_test": UniversalAction.VERIFY,
            "report": UniversalAction.RESPOND,
            "plan_pipeline": UniversalAction.PLAN,
            "complete": UniversalAction.DONE,
        }
        return mapping.get(action_type, UniversalAction.OTHER)


ADAPTERS: dict[str, DomainAdapter] = {
    "coding": CodingAdapter(),
    "support": SupportAdapter(),
    "research": ResearchAdapter(),
    "data_pipeline": DataPipelineAdapter(),
}


def get_adapter(domain: str) -> DomainAdapter:
    return ADAPTERS.get(domain, CodingAdapter())
