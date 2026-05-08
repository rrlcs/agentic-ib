"""FastAPI application entry point."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from fastapi import Body, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from messaging.producer import send_task
from messaging.results_reader import get_latest_result, get_result_by_task_id
from observability.logger import get_logger, log_event
from observability.tracer import get_metrics, get_task_metrics, read_trace

app = FastAPI(title="Agentic Platform", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
log = get_logger(__name__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run-agent")
def run_agent(task_data: dict[str, Any] = Body(default_factory=dict)) -> dict[str, str]:
    job_id = send_task(task_name="run_agent", task_data=task_data)
    log_event(log, "api_enqueued_task", job_id=job_id, task_id=job_id, requested_agent=task_data.get("agent", "router"))
    return {"job_id": job_id}


@app.post("/chat")
def chat(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, str]:
    """Chat-based entrypoint.

    ``payload`` may contain: ``message`` (required), ``company``, ``symbol``,
    and ``model`` (e.g. ``gpt-4o-mini``, ``gpt-4o``). The ``model`` is forwarded
    to the router and used for all LLM calls in the resulting pipeline.
    """
    message = (payload.get("message") or "").strip()
    if not message:
        return {"status": "error", "reason": "empty_message"}
    task_data: dict[str, Any] = {
        "agent": "router",
        "question": message,
    }
    for optional in ("company", "symbol", "model"):
        if payload.get(optional):
            task_data[optional] = payload[optional]
    job_id = send_task(task_name="chat_message", task_data=task_data)
    log_event(
        log,
        "api_chat_enqueued",
        task_id=job_id,
        message_preview=message[:80],
        model=task_data.get("model", "default"),
    )
    return {"job_id": job_id, "task_id": job_id}


@app.get("/job/latest")
def latest_result() -> dict[str, Any]:
    result = get_latest_result()
    if result is None:
        log_event(log, "api_result_latest_not_found")
        return {"status": "not_found"}
    log_event(log, "api_result_latest_found", task_id=result.get("task_id"), status=result.get("status"))
    return result


@app.get("/job/{task_id}")
def job_result(task_id: str) -> dict[str, Any]:
    result = get_result_by_task_id(task_id)
    if result is None:
        log_event(log, "api_result_not_found", task_id=task_id)
        return {"task_id": task_id, "status": "not_found"}
    log_event(log, "api_result_found", task_id=task_id, status=result.get("status"))
    return result


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    """Aggregate, all-time metrics."""
    return get_metrics()


@app.get("/metrics/{task_id}")
def metrics_for_task(task_id: str) -> dict[str, Any]:
    """Per-task metrics: counters captured for this single interaction."""
    data = get_task_metrics(task_id)
    if not data:
        return {"task_id": task_id, "status": "not_found"}
    return data


@app.get("/stream/{task_id}")
async def stream_traces(task_id: str, request: Request) -> EventSourceResponse:
    """SSE stream of trace events for a task. Closes when terminal event is seen."""

    async def event_gen():
        last_id = "0-0"
        terminal_seen = False
        while not terminal_seen:
            if await request.is_disconnected():
                break
            new_last, events = await asyncio.to_thread(
                read_trace,
                task_id,
                last_id=last_id,
                block_ms=1000,
                count=200,
            )
            if events:
                last_id = new_last
                for ev in events:
                    yield {
                        "event": ev.get("event", "trace"),
                        "data": json.dumps(ev, default=str),
                    }
                    if ev.get("event") == "pipeline_complete":
                        terminal_seen = True
                        break
            else:
                yield {"event": "ping", "data": "keep-alive"}
            await asyncio.sleep(0.02)

    return EventSourceResponse(event_gen())
