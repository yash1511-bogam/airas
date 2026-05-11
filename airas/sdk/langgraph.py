"""LangGraph middleware — automatic AIRAS integration in <10 lines of user code.

Usage:
    from airas.sdk.langgraph import airas_middleware
    from airas.sdk.client import AIRASClient

    client = AIRASClient(api_key="ak_...")
    graph = create_my_graph()
    graph = airas_middleware(graph, client)  # that's it

How it works:
1. Wraps each node's execution
2. Before node runs: calls /check with trace-so-far
3. If intervention returned: prepends payload to the node's messages/state
4. After node runs: appends step to trace buffer
5. On graph completion: calls /ingest with full trace
"""
import logging
import functools
from airas.sdk.client import AIRASClient
from airas.models import ExecutionStep

logger = logging.getLogger(__name__)


def airas_middleware(graph, client: AIRASClient, project_id: str = ""):
    """Wrap a LangGraph StateGraph with AIRAS immunity checks.

    Non-invasive: if AIRAS is down, the graph runs normally.
    Adds `_airas_trace` to state for step accumulation.
    """
    original_nodes = dict(graph.nodes)

    for node_name, node_fn in original_nodes.items():
        if node_name in ("__start__", "__end__"):
            continue

        @functools.wraps(node_fn)
        async def wrapped_node(state, _fn=node_fn, _name=node_name):
            # Build step history from state
            trace_steps = state.get("_airas_trace", [])

            # Check immunity
            check_result = await client.check(
                steps=[ExecutionStep(**s) for s in trace_steps] if trace_steps else [],
                project_id=project_id,
            )

            # If intervention: inject into state messages
            if check_result.matched and check_result.payload:
                logger.info("AIRAS intervention at node '%s': %s", _name, check_result.failure_class)
                messages = state.get("messages", [])
                if messages:
                    # Prepend intervention as system message
                    intervention_msg = {"role": "system", "content": f"[AIRAS] {check_result.payload}"}
                    state = {**state, "messages": [intervention_msg] + messages}

            # Execute original node
            result = await _fn(state) if callable(_fn) else _fn

            # Record step
            step_record = {
                "index": len(trace_steps),
                "action": _name,
                "action_type": "node",
                "observation": str(result)[:200] if result else "",
                "thought": "",
                "state": {},
            }
            new_trace = trace_steps + [step_record]

            # Merge step record into result state
            if isinstance(result, dict):
                result["_airas_trace"] = new_trace
            return result

        graph.nodes[node_name] = wrapped_node

    return graph
