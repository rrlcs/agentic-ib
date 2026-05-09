"""Kafka API producer for dispatching agent tasks to Redpanda."""

from __future__ import annotations

import json
import os
import uuid

from kafka import KafkaProducer
from mcp.builder import build_mcp_message
from mcp.parser import message_to_kafka_dict
from observability.logger import get_logger, log_event

_TOPIC = "agent_tasks"

_producer: KafkaProducer | None = None
log = get_logger(__name__)


def _bootstrap_servers() -> str:
    return os.environ.get("REDPANDA_BOOTSTRAP_SERVERS") or os.environ.get(
        "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
    )


def _get_producer() -> KafkaProducer:
    global _producer
    if _producer is None:
        _producer = KafkaProducer(
            bootstrap_servers=_bootstrap_servers(),
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        log_event(log, "kafka_producer_initialized", bootstrap=_bootstrap_servers())
    return _producer


def send_task(task_name: str, task_data: dict) -> str:
    job_id = str(uuid.uuid4())
    td = dict(task_data)
    agent = td.pop("agent", "router")
    envelope = build_mcp_message(
        job_id,
        agent=agent,
        payload={"task_name": task_name, "payload": td},
    )
    producer = _get_producer()
    producer.send(_TOPIC, message_to_kafka_dict(envelope))
    producer.flush(timeout=10)
    log_event(log, "kafka_task_sent", topic=_TOPIC, job_id=job_id, agent=agent, task_name=task_name)
    return job_id

def send(topic, message):
    producer = _get_producer()
    producer.send(topic, message)
    producer.flush(timeout=10)
    log_event(log, "kafka_message_sent", topic=topic, status=message.get("status") if isinstance(message, dict) else "unknown")