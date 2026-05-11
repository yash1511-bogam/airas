"""AIRAS SDK — developer-facing client for agent integration.

Usage:
    from airas.sdk import AIRASClient
    client = AIRASClient(api_key="ak_...", base_url="http://localhost:8100")

    # Check before each step
    result = await client.check(steps_so_far)
    if result.matched:
        # Inject result.payload into agent prompt

    # Report completed trace
    await client.ingest(trace_id, steps, success=True)

Graceful degradation: if AIRAS server is unreachable, all methods return
safe defaults (no intervention) and log warnings. Agent never blocks on AIRAS.
"""
import logging
import httpx
from airas.models import ExecutionStep
from airas.models.api import CheckResponse

logger = logging.getLogger(__name__)


class AIRASClient:
    """Async HTTP client for AIRAS API."""

    def __init__(self, base_url: str = "http://localhost:8100", api_key: str = "", timeout: float = 0.5):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout  # 500ms max — never block the agent
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )

    async def check(self, steps: list[ExecutionStep], trace_id: str = "", project_id: str = "") -> CheckResponse:
        """Real-time immunity check. Returns intervention if pattern matched.

        NEVER raises — returns empty CheckResponse on any failure.
        """
        try:
            resp = await self._client.post("/v1/check", json={
                "project_id": project_id,
                "trace_id": trace_id,
                "steps": [s.model_dump() for s in steps],
                "current_step": len(steps) - 1,
            })
            resp.raise_for_status()
            return CheckResponse(**resp.json())
        except Exception as e:
            logger.debug("AIRAS check failed (graceful degradation): %s", e)
            return CheckResponse(matched=False)

    async def ingest(self, trace_id: str, steps: list[ExecutionStep], success: bool | None = None, project_id: str = ""):
        """Report a completed trace for background learning."""
        try:
            await self._client.post("/v1/traces", json={
                "trace_id": trace_id,
                "project_id": project_id,
                "steps": [s.model_dump() for s in steps],
                "success": success,
            })
        except Exception as e:
            logger.debug("AIRAS ingest failed (non-blocking): %s", e)

    async def close(self):
        await self._client.aclose()
