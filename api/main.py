"""FastAPI application entry point."""

from typing import Any

from fastapi import Body, FastAPI

from messaging.producer import send_task
from messaging.results_reader import get_latest_result, get_result_by_task_id
from observability.logger import get_logger, log_event

app = FastAPI(title="Agentic Platform", version="0.1.0")
log = get_logger(__name__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run-agent")
def run_agent(task_data: dict[str, Any] = Body(default_factory=dict)) -> dict[str, str]:
    job_id = send_task(task_name="run_agent", task_data=task_data)
    log_event(log, "api_enqueued_task", job_id=job_id, requested_agent=task_data.get("agent", "router"))
    return {"job_id": job_id}


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


