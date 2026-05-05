"""FastAPI application entry point."""

from typing import Any

from fastapi import Body, FastAPI

from messaging.producer import send_task

app = FastAPI(title="Agentic Platform", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run-agent")
def run_agent(task_data: dict[str, Any] = Body(default_factory=dict)) -> dict[str, str]:
    job_id = send_task(task_name="run_agent", task_data=task_data)
    return {"job_id": job_id}


