"""Tool-calling agent loop (a.k.a. ReAct runtime).

This is what makes each agent *actually* agentic: instead of one LLM call per
step, the agent receives a set of tools (web_search, vector_search, etc.) and
loops:

    user prompt --> LLM
                ↳ tool_call(s) --> handler(s) --> observation messages
    repeat up to ``max_iterations`` until the LLM returns a final message.

The loop publishes per-iteration trace events and bumps the per-task metrics
(``tool_calls``, ``agent_iterations``, token counts via the LLM client) so the
UI can show a real "what the agent thought / called" timeline.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from observability.logger import get_logger, log_event
from observability.tracer import increment_task_metric
from tools import llm_client

log = get_logger(__name__)

DEFAULT_MAX_ITERATIONS = 6


def run_agent_loop(
    *,
    agent_name: str,
    system_prompt: str,
    user_prompt: str,
    tools: list[dict[str, Any]] | None = None,
    tool_handlers: dict[str, Callable[..., Any]] | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run an agent. Returns ``{answer, tool_calls, iterations, status}``.

    If no tools are supplied, this collapses to a single LLM call so simple
    agents (like ``answer_agent``) can opt out of the loop entirely.
    """
    task_id = llm_client.current_task_id.get()
    model = llm_client.current_model.get() or llm_client._DEFAULT_MODEL
    tools = tools or []
    handlers = tool_handlers or {}

    log_event(
        log,
        "agent_loop_started",
        task_id=task_id,
        agent=agent_name,
        model=model,
        n_tools=len(tools),
        tool_names=[t["function"]["name"] for t in tools],
    )

    if not tools:
        text = llm_client.generate_response(system_prompt, user_prompt, model=model)
        log_event(log, "agent_loop_completed", task_id=task_id, agent=agent_name, iterations=1, n_tool_calls=0)
        return {"answer": text, "tool_calls": [], "iterations": 1, "status": "ok"}

    client = llm_client._get_client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    tool_calls_log: list[dict[str, Any]] = []
    final_text = ""
    iteration = 0
    status = "ok"

    for iteration in range(1, max_iterations + 1):
        if task_id:
            increment_task_metric(task_id, "agent_iterations", 1)
            increment_task_metric(task_id, "llm_calls", 1)
        log_event(log, "agent_iteration", task_id=task_id, agent=agent_name, iteration=iteration)

        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "tools": tools,
                "temperature": 0.3,
            }
            if response_format and iteration == max_iterations:
                kwargs["response_format"] = response_format
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:
            log_event(log, "agent_iteration_failed", task_id=task_id, agent=agent_name, error=str(exc))
            status = "llm_error"
            break

        llm_client._track_usage(task_id, response)
        msg = response.choices[0].message
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}",
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        if not msg.tool_calls:
            final_text = msg.content or ""
            log_event(
                log,
                "agent_iteration_final",
                task_id=task_id,
                agent=agent_name,
                iteration=iteration,
                final_chars=len(final_text),
            )
            break

        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
                if not isinstance(args, dict):
                    args = {}
            except json.JSONDecodeError:
                args = {}

            log_event(
                log,
                "agent_tool_call",
                task_id=task_id,
                agent=agent_name,
                tool=tool_name,
                args_preview=json.dumps(args, default=str)[:200],
            )
            if task_id:
                increment_task_metric(task_id, "tool_calls", 1)

            handler = handlers.get(tool_name)
            if handler is None:
                tool_result: Any = {"error": f"unknown_tool:{tool_name}"}
            else:
                try:
                    tool_result = handler(**args)
                except TypeError:
                    try:
                        tool_result = handler(args)
                    except Exception as exc:
                        log.exception("tool_call_failed tool=%s", tool_name)
                        tool_result = {"error": str(exc)}
                except Exception as exc:
                    log.exception("tool_call_failed tool=%s", tool_name)
                    tool_result = {"error": str(exc)}

            try:
                tool_result_str = json.dumps(tool_result, default=str)
            except (TypeError, ValueError):
                tool_result_str = str(tool_result)
            tool_result_str = tool_result_str[:8000]

            log_event(
                log,
                "agent_tool_result",
                task_id=task_id,
                agent=agent_name,
                tool=tool_name,
                result_chars=len(tool_result_str),
            )
            tool_calls_log.append(
                {
                    "name": tool_name,
                    "args": args,
                    "result_preview": tool_result_str[:300],
                }
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result_str,
                }
            )
    else:
        # exhausted iterations
        status = "max_iterations"
        log_event(log, "agent_loop_max_iterations", task_id=task_id, agent=agent_name)

    log_event(
        log,
        "agent_loop_completed",
        task_id=task_id,
        agent=agent_name,
        iterations=iteration,
        n_tool_calls=len(tool_calls_log),
        final_chars=len(final_text),
        status=status,
    )
    return {
        "answer": final_text,
        "tool_calls": tool_calls_log,
        "iterations": iteration,
        "status": status,
    }
