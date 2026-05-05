"""Read completed agent results from Kafka."""

from __future__ import annotations

import json
import os
from typing import Any

from kafka import KafkaConsumer

from messaging.topics import AGENT_RESULTS


def _bootstrap_servers() -> str:
    return os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


def get_result_by_task_id(task_id: str, *, timeout_ms: int = 2000) -> dict[str, Any] | None:
    """Scan results topic and return latest record for task_id."""
    consumer = KafkaConsumer(
        AGENT_RESULTS,
        bootstrap_servers=_bootstrap_servers(),
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        group_id=None,
        enable_auto_commit=False,
        consumer_timeout_ms=timeout_ms,
    )
    latest: dict[str, Any] | None = None
    try:
        for msg in consumer:
            value = msg.value if isinstance(msg.value, dict) else {}
            if value.get("task_id") == task_id:
                latest = value
    finally:
        consumer.close()
    return latest


def get_latest_result(*, timeout_ms: int = 2000) -> dict[str, Any] | None:
    """Return the most recent message available from results topic."""
    consumer = KafkaConsumer(
        AGENT_RESULTS,
        bootstrap_servers=_bootstrap_servers(),
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        group_id=None,
        enable_auto_commit=False,
        consumer_timeout_ms=timeout_ms,
    )
    latest: dict[str, Any] | None = None
    try:
        for msg in consumer:
            value = msg.value if isinstance(msg.value, dict) else {}
            if value:
                latest = value
    finally:
        consumer.close()
    return latest
