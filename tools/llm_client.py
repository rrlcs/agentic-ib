"""OpenAI client wrapper with streaming, per-task tracing and model selection.

The agent code calls :func:`generate_response` (non-streaming) or
:func:`generate_response_stream` (streaming, publishes ``llm_token`` trace events).

Model selection / task-id / agent-name are picked up from contextvars set by the
worker, so callers don't need to thread them everywhere.
"""

from __future__ import annotations

import contextvars
import os
from typing import Iterator

from openai import OpenAI

from observability.logger import get_logger, log_event
from observability.tracer import increment_task_metric, publish_trace

_client: OpenAI | None = None
log = get_logger(__name__)

_DEFAULT_MODEL = os.getenv("LLM_DEFAULT_MODEL", "gpt-4o-mini")

current_task_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_task_id", default=None)
current_agent: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_agent", default=None)
current_model: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_model", default=None)


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def _resolve_model(model: str | None) -> str:
    return model or current_model.get() or _DEFAULT_MODEL


def generate_response(system_prompt: str, user_prompt: str, *, model: str | None = None) -> str:
    """Run a synchronous (non-streaming) chat completion."""
    task_id = current_task_id.get()
    agent = current_agent.get()
    chosen_model = _resolve_model(model)

    log_event(log, "llm_call_started", task_id=task_id, agent=agent, model=chosen_model, mode="sync")
    if task_id:
        increment_task_metric(task_id, "llm_calls", 1)
    response = _get_client().chat.completions.create(
        model=chosen_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    content = response.choices[0].message.content or ""
    _track_usage(task_id, response)
    log_event(
        log,
        "llm_call_completed",
        task_id=task_id,
        agent=agent,
        model=chosen_model,
        mode="sync",
        response_chars=len(content),
    )
    return content


def generate_response_stream(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
) -> str:
    """Run a streaming chat completion.

    Each delta is published as a ``llm_token`` trace event tagged with the current
    agent so the UI can render the reply token-by-token. Returns the full
    accumulated text once the stream completes.
    """
    task_id = current_task_id.get()
    agent = current_agent.get()
    chosen_model = _resolve_model(model)

    log_event(log, "llm_call_started", task_id=task_id, agent=agent, model=chosen_model, mode="stream")
    if task_id:
        increment_task_metric(task_id, "llm_calls", 1)
    stream = _get_client().chat.completions.create(
        model=chosen_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        stream=True,
        stream_options={"include_usage": True},
    )

    pieces: list[str] = []
    last_chunk = None
    for chunk in _safe_iter(stream):
        last_chunk = chunk
        try:
            delta = chunk.choices[0].delta.content
        except (IndexError, AttributeError):
            delta = None
        if not delta:
            continue
        pieces.append(delta)
        if task_id:
            publish_trace(task_id, "llm_token", agent=agent, delta=delta, model=chosen_model)
    _track_usage(task_id, last_chunk)

    full = "".join(pieces)
    log_event(
        log,
        "llm_call_completed",
        task_id=task_id,
        agent=agent,
        model=chosen_model,
        mode="stream",
        response_chars=len(full),
    )
    return full


def _safe_iter(stream) -> Iterator:
    """Wrap stream iteration so a network blip doesn't crash the agent."""
    try:
        yield from stream
    except Exception as exc:  # pragma: no cover - depends on transient network
        log.warning("llm_stream_aborted err=%s", exc)


def _track_usage(task_id: str | None, response) -> None:
    """Pull token usage from a Chat Completions response and record it per task."""
    if not task_id or response is None:
        return
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    if prompt_tokens:
        increment_task_metric(task_id, "tokens_in", int(prompt_tokens))
    if completion_tokens:
        increment_task_metric(task_id, "tokens_out", int(completion_tokens))
