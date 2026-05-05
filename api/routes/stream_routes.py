"""Placeholder — implement module."""

from fastapi.responses import StreamingResponse
from messaging.consumer import create_consumer

def stream(job_id):

    def generator():
        consumer = create_consumer("agent_results")

        for msg in consumer:
            if msg.value["task_id"] == job_id:
                yield f"data: {msg.value}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")