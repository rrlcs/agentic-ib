"""Consume agent tasks from Kafka and run the dynamic pipeline planned by the router.

Worker semantics
----------------
- Router runs first (always) and writes ``metadata.pipeline`` and ``metadata.pipeline_idx``.
- After each agent returns, the worker advances ``pipeline_idx`` to the next step.
- A pipeline step can be either a string (single agent) or a list of strings
  (agents run **in parallel**; their context updates are merged back). Parallel
  groups are typically used for independent steps like research + financials.
- Validator can return a feedback dict ``{decision: re_research|re_synthesis|accept, ...}``
  which moves ``pipeline_idx`` back to that agent (subject to ``retry_budget``).
"""

from __future__ import annotations

import contextvars
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from mcp.parser import parse_mcp_message_safe
from mcp.schema import MCPMessage
from messaging.consumer import create_consumer
from messaging.producer import send
from messaging.topics import AGENT_RESULTS, AGENT_TASKS
from observability.logger import get_logger, log_event
from observability.tracer import record_task_completed, record_task_started
from tools import llm_client
from workers.dispatcher import dispatch

log = get_logger(__name__)

_TERMINAL_STATUSES = {"success", "failed"}
_MAX_LOOP = 64


def _bootstrap_servers() -> str:
    return os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


def _peek_next_step(mcp: MCPMessage) -> Any:
    pipeline = list(mcp.metadata.get("pipeline") or [])
    idx = int(mcp.metadata.get("pipeline_idx", -1)) + 1
    if idx >= len(pipeline):
        return None
    return pipeline[idx]


def _advance(mcp: MCPMessage) -> Any:
    """Move pipeline_idx forward by 1 and return the new step (str | list | None)."""
    next_step = _peek_next_step(mcp)
    mcp.metadata["pipeline_idx"] = int(mcp.metadata.get("pipeline_idx", -1)) + 1
    if isinstance(next_step, str):
        mcp.agent = next_step
    return next_step


def _find_step_idx(pipeline: list, agent_name: str) -> int:
    for i, step in enumerate(pipeline):
        if isinstance(step, list) and agent_name in step:
            return i
        if step == agent_name:
            return i
    return -1


def _apply_feedback(mcp: MCPMessage, feedback: dict) -> bool:
    decision = (feedback.get("decision") or "").lower()
    if decision not in {"re_research", "re_synthesis"}:
        return False

    budget = mcp.metadata.setdefault("retry_budget", {"research": 1, "synthesis": 2})
    bucket = "research" if decision == "re_research" else "synthesis"
    if int(budget.get(bucket, 0)) <= 0:
        log_event(log, "feedback_budget_exhausted", task_id=mcp.task_id, decision=decision)
        return False

    target_agent = "research_agent" if decision == "re_research" else "synthesis_agent"
    pipeline = list(mcp.metadata.get("pipeline") or [])
    target_idx = _find_step_idx(pipeline, target_agent)
    if target_idx < 0:
        log_event(log, "feedback_target_missing", task_id=mcp.task_id, target=target_agent)
        return False

    budget[bucket] = int(budget[bucket]) - 1
    # Step the index back so we *re-run* target on next loop iteration.
    mcp.metadata["pipeline_idx"] = target_idx - 1
    mcp.agent = target_agent
    mcp.metadata["last_feedback"] = feedback
    log_event(
        log,
        "feedback_loop_invoked",
        task_id=mcp.task_id,
        decision=decision,
        target_agent=target_agent,
        budget_left=budget[bucket],
    )
    return True


def _terminal_from_pipeline_end(mcp: MCPMessage) -> dict:
    return {
        "task_id": mcp.task_id,
        "status": "success",
        "intent": mcp.context.get("intent"),
        "answer": mcp.context.get("answer"),
        "result": mcp.context.get("final_report"),
        "validation": mcp.context.get("validator_feedback"),
        "action": mcp.context.get("action_result"),
        "memo_source": mcp.context.get("memo_source"),
        "context_keys": sorted(mcp.context.keys()),
    }


def _set_llm_context(mcp: MCPMessage, agent_name: str | None = None) -> None:
    llm_client.current_task_id.set(mcp.task_id)
    llm_client.current_agent.set(agent_name or mcp.agent)
    llm_client.current_model.set(mcp.context.get("model"))


def _run_agent_in_thread(mcp: MCPMessage, agent_name: str) -> Any:
    """Run a single agent in a worker thread with proper LLM contextvars."""
    cp = mcp.model_copy(deep=True)
    cp.agent = agent_name
    parent_ctx = contextvars.copy_context()

    def runner():
        llm_client.current_task_id.set(mcp.task_id)
        llm_client.current_agent.set(agent_name)
        llm_client.current_model.set(mcp.context.get("model"))
        return dispatch(cp)

    return parent_ctx.run(runner)


def _run_parallel_group(mcp: MCPMessage, agents: list[str]) -> None:
    log_event(log, "parallel_started", task_id=mcp.task_id, agents=agents)
    summary: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(2, len(agents))) as executor:
        futures = {
            executor.submit(_run_agent_in_thread, mcp, agent): agent
            for agent in agents
        }
        for fut in as_completed(futures):
            agent = futures[fut]
            try:
                result = fut.result()
                if isinstance(result, MCPMessage):
                    for key, value in result.context.items():
                        if key not in {"payload"}:
                            mcp.context[key] = value
                    summary[agent] = "ok"
                elif isinstance(result, dict) and result.get("status") == "failed":
                    summary[agent] = f"failed:{result.get('reason')}"
                else:
                    summary[agent] = "noop"
            except Exception as exc:  # pragma: no cover - safety net
                log.exception("parallel_agent_error agent=%s", agent)
                summary[agent] = f"error:{exc}"
    log_event(log, "parallel_completed", task_id=mcp.task_id, agents=summary)


def _advance_or_handle_parallel(mcp: MCPMessage) -> tuple[bool, dict | None]:
    """Advance to next step. Loops past parallel groups. Returns (continue, terminal)."""
    while True:
        step = _advance(mcp)
        if step is None:
            return False, _terminal_from_pipeline_end(mcp)
        if isinstance(step, list):
            _run_parallel_group(mcp, step)
            continue
        log_event(log, "agent_handoff", task_id=mcp.task_id, next_agent=step)
        return True, None


def _process_envelope(mcp: MCPMessage) -> dict:
    log_event(log, "mcp_envelope_received", task_id=mcp.task_id, agent=mcp.agent)
    record_task_started(mcp.task_id)

    for _ in range(_MAX_LOOP):
        _set_llm_context(mcp)
        try:
            result = dispatch(mcp)
        except Exception as exc:
            log.exception("dispatch_error task_id=%s agent=%s", mcp.task_id, mcp.agent)
            terminal = {"task_id": mcp.task_id, "status": "failed", "reason": str(exc)}
            record_task_completed(mcp.task_id, success=False)
            return terminal

        if isinstance(result, MCPMessage):
            mcp = result
            advanced, terminal = _advance_or_handle_parallel(mcp)
            if not advanced:
                record_task_completed(mcp.task_id, success=True)
                return terminal  # type: ignore[return-value]
            continue

        if isinstance(result, dict):
            decision = (result.get("decision") or "").lower()
            if decision in {"re_research", "re_synthesis"}:
                if _apply_feedback(mcp, result):
                    continue
                mcp.context["validator_feedback"] = result
                advanced, terminal = _advance_or_handle_parallel(mcp)
                if not advanced:
                    record_task_completed(mcp.task_id, success=True)
                    return terminal  # type: ignore[return-value]
                continue

            if decision == "accept":
                mcp.context["validator_feedback"] = result
                advanced, terminal = _advance_or_handle_parallel(mcp)
                if not advanced:
                    record_task_completed(mcp.task_id, success=True)
                    return terminal  # type: ignore[return-value]
                continue

            if result.get("status") in _TERMINAL_STATUSES:
                record_task_completed(mcp.task_id, success=result.get("status") == "success")
                return result

        record_task_completed(mcp.task_id, success=False)
        return {"task_id": mcp.task_id, "status": "failed", "reason": "unexpected_dispatch_payload"}

    record_task_completed(mcp.task_id, success=False)
    return {"task_id": mcp.task_id, "status": "failed", "reason": "max_iterations"}


def main() -> None:
    brokers = _bootstrap_servers()
    log_event(log, "worker_starting", bootstrap=brokers, topic=AGENT_TASKS)
    consumer = create_consumer(AGENT_TASKS)
    for msg in consumer:
        raw = msg.value if isinstance(msg.value, dict) else {}
        mcp = parse_mcp_message_safe(raw)
        if not mcp:
            log.warning("non-MCP or invalid payload dropped keys=%s", list(raw.keys()))
            continue
        terminal = _process_envelope(mcp)
        send(AGENT_RESULTS, terminal)
        log_event(log, "pipeline_complete", task_id=terminal.get("task_id"), status=terminal.get("status"))


if __name__ == "__main__":
    main()
