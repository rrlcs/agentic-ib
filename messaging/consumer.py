"""Kafka API consumers (connected to Redpanda)."""

from __future__ import annotations

import json
import os

from kafka import KafkaConsumer


def _bootstrap_servers() -> str:
    return os.environ.get("REDPANDA_BOOTSTRAP_SERVERS") or os.environ.get(
        "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
    )


def create_consumer(topic: str, *, group_id: str = "agent-workers") -> KafkaConsumer:
    return KafkaConsumer(
        topic,
        bootstrap_servers=_bootstrap_servers(),
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        group_id=group_id,
    )
