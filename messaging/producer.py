"""Kafka producer for dispatching agent tasks."""

from __future__ import annotations

import json
import os
import uuid

from kafka import KafkaProducer

_TOPIC = "agent_tasks"

_producer: KafkaProducer | None = None


def _bootstrap_servers() -> str:
    return os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


def _get_producer() -> KafkaProducer:
    global _producer
    if _producer is None:
        _producer = KafkaProducer(
            bootstrap_servers=_bootstrap_servers(),
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
    return _producer


def send_task(task_name: str, task_data: dict) -> str:
    job_id = str(uuid.uuid4())
    message = {"job_id": job_id, "task_name": task_name, "payload": task_data}
    prod = _get_producer()
    prod.send(_TOPIC, message)
    prod.flush(timeout=10)
    return job_id
