"""Per-task trace publisher backed by Redis Streams.

Each task gets its own stream key: ``trace:{task_id}``. UI consumes via SSE
endpoint, which tails the stream from the last seen ID.

The tracer also records aggregate metrics used by the ``/metrics`` API:
- ``metrics:agentic`` hash with task counters and event counters
- ``metrics:task_durations`` list with last 200 task latencies (ms)
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any

import redis

from observability.logger import get_logger

log = get_logger(__name__)

_STREAM_PREFIX = "trace:"
_TASK_META_PREFIX = "task:"
_TASK_METRICS_PREFIX = "taskmetrics:"
_GLOBAL_METRICS_KEY = "metrics:agentic"
_DURATIONS_KEY = "metrics:task_durations"
_DEFAULT_MAXLEN = 1000
_DURATIONS_KEEP = 200

_client: redis.Redis | None = None


def _get_client() -> redis.Redis:
    global _client
    if _client is None:
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        _client = redis.Redis.from_url(url, decode_responses=True)
    return _client


def stream_key(task_id: str) -> str:
    return f"{_STREAM_PREFIX}{task_id}"


def task_meta_key(task_id: str) -> str:
    return f"{_TASK_META_PREFIX}{task_id}:meta"


def task_metrics_key(task_id: str) -> str:
    return f"{_TASK_METRICS_PREFIX}{task_id}"


def publish_trace(task_id: str, event: str, **fields: Any) -> None:
    """Publish a structured trace event for ``task_id``."""
    if not task_id:
        return
    payload = {
        "ts": time.time(),
        "event": event,
        "task_id": task_id,
        **fields,
    }
    try:
        client = _get_client()
        client.xadd(
            stream_key(task_id),
            {"data": json.dumps(payload, default=str)},
            maxlen=_DEFAULT_MAXLEN,
            approximate=True,
        )
        client.expire(stream_key(task_id), 60 * 60 * 6)
        client.hincrby(_GLOBAL_METRICS_KEY, f"event:{event}", 1)
    except Exception as exc:  # pragma: no cover - tracing must never crash callers
        log.warning("tracer_publish_failed event=%s err=%s", event, exc)


def record_task_started(task_id: str) -> None:
    """Stamp the task start time so we can measure latency."""
    if not task_id:
        return
    try:
        client = _get_client()
        client.hset(task_meta_key(task_id), mapping={"start_ts": time.time()})
        client.expire(task_meta_key(task_id), 60 * 60)
        client.hincrby(_GLOBAL_METRICS_KEY, "tasks:started", 1)
        # Initialise the per-task metrics hash so /metrics/{task_id} returns zeros early.
        client.hset(
            task_metrics_key(task_id),
            mapping={
                "llm_calls": 0,
                "tool_calls": 0,
                "agent_iterations": 0,
                "feedback_loops": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "started_at": time.time(),
            },
        )
        client.expire(task_metrics_key(task_id), 60 * 60 * 24)
    except Exception as exc:  # pragma: no cover
        log.warning("tracer_task_start_failed err=%s", exc)


def record_task_completed(task_id: str, *, success: bool) -> float | None:
    """Record duration and success/fail counters. Returns latency in ms."""
    if not task_id:
        return None
    try:
        client = _get_client()
        raw = client.hget(task_meta_key(task_id), "start_ts")
        duration_ms: float | None = None
        if raw:
            duration_ms = (time.time() - float(raw)) * 1000.0
            client.lpush(_DURATIONS_KEY, duration_ms)
            client.ltrim(_DURATIONS_KEY, 0, _DURATIONS_KEEP - 1)
            client.hset(task_metrics_key(task_id), "latency_ms", duration_ms)
        client.hset(
            task_metrics_key(task_id),
            mapping={
                "completed_at": time.time(),
                "status": "success" if success else "failed",
            },
        )
        client.hincrby(_GLOBAL_METRICS_KEY, "tasks:completed", 1)
        client.hincrby(_GLOBAL_METRICS_KEY, "tasks:succeeded" if success else "tasks:failed", 1)
        client.delete(task_meta_key(task_id))
        return duration_ms
    except Exception as exc:  # pragma: no cover
        log.warning("tracer_task_end_failed err=%s", exc)
        return None


def increment_task_metric(task_id: str, field: str, by: int = 1) -> None:
    """Bump a per-task counter."""
    if not task_id or not field:
        return
    try:
        client = _get_client()
        client.hincrby(task_metrics_key(task_id), field, int(by))
        client.expire(task_metrics_key(task_id), 60 * 60 * 24)
    except Exception as exc:  # pragma: no cover
        log.warning("tracer_metric_inc_failed err=%s", exc)


def get_task_metrics(task_id: str) -> dict[str, Any]:
    """Return per-task metrics in a UI-friendly shape."""
    if not task_id:
        return {}
    try:
        client = _get_client()
        raw = client.hgetall(task_metrics_key(task_id)) or {}
    except Exception:  # pragma: no cover
        return {}
    if not raw:
        return {}

    started = _maybe_float(raw.get("started_at"))
    completed = _maybe_float(raw.get("completed_at"))
    latency_ms = _maybe_float(raw.get("latency_ms"))
    if latency_ms is None and started:
        latency_ms = ((completed or time.time()) - started) * 1000.0

    return {
        "task_id": task_id,
        "status": raw.get("status") or ("running" if not completed else "unknown"),
        "llm_calls": int(raw.get("llm_calls", 0) or 0),
        "tool_calls": int(raw.get("tool_calls", 0) or 0),
        "agent_iterations": int(raw.get("agent_iterations", 0) or 0),
        "feedback_loops": int(raw.get("feedback_loops", 0) or 0),
        "tokens_in": int(raw.get("tokens_in", 0) or 0),
        "tokens_out": int(raw.get("tokens_out", 0) or 0),
        "latency_ms": round(latency_ms, 1) if latency_ms is not None else None,
        "started_at": started,
        "completed_at": completed,
    }


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def read_trace(
    task_id: str,
    *,
    last_id: str = "0-0",
    block_ms: int = 1000,
    count: int = 100,
) -> tuple[str, list[dict[str, Any]]]:
    """Block-read trace events for ``task_id`` from ``last_id``.

    Returns the new last_id and parsed events.
    """
    client = _get_client()
    response = client.xread({stream_key(task_id): last_id}, count=count, block=block_ms)
    events: list[dict[str, Any]] = []
    new_last_id = last_id
    if not response:
        return new_last_id, events
    _key, entries = response[0]
    for entry_id, fields in entries:
        new_last_id = entry_id
        raw = fields.get("data")
        if not raw:
            continue
        try:
            events.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return new_last_id, events


def get_metrics() -> dict[str, Any]:
    """Return user-facing aggregate metrics."""
    try:
        client = _get_client()
        raw = client.hgetall(_GLOBAL_METRICS_KEY) or {}
        durations_raw = client.lrange(_DURATIONS_KEY, 0, _DURATIONS_KEEP - 1) or []
    except Exception:  # pragma: no cover
        return {
            "tasks_total": 0,
            "tasks_succeeded": 0,
            "tasks_failed": 0,
            "llm_calls_total": 0,
            "latency_ms_avg": None,
            "latency_ms_p95": None,
            "samples": 0,
        }

    counters = {key: int(value) for key, value in raw.items()}
    durations = sorted(float(x) for x in durations_raw)
    avg = sum(durations) / len(durations) if durations else None
    if durations:
        rank = max(0, math.ceil(0.95 * len(durations)) - 1)
        p95 = durations[rank]
    else:
        p95 = None

    return {
        "tasks_total": counters.get("tasks:completed", 0),
        "tasks_succeeded": counters.get("tasks:succeeded", 0),
        "tasks_failed": counters.get("tasks:failed", 0),
        "tasks_in_flight": max(
            0,
            counters.get("tasks:started", 0) - counters.get("tasks:completed", 0),
        ),
        "llm_calls_total": counters.get("event:llm_call_completed", 0),
        "feedback_loops": counters.get("event:feedback_loop_invoked", 0),
        "latency_ms_avg": round(avg, 1) if avg is not None else None,
        "latency_ms_p95": round(p95, 1) if p95 is not None else None,
        "samples": len(durations),
    }
