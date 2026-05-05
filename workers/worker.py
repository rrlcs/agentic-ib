"""Consume agent tasks from Kafka (run as a standalone process)."""

from __future__ import annotations

import json
import logging
import os
import sys

from kafka import KafkaConsumer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def _bootstrap_servers() -> str:
    return os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


def main() -> None:
    brokers = _bootstrap_servers()
    log.info("worker starting; kafka bootstrap=%s topic=agent_tasks", brokers)
    consumer = KafkaConsumer(
        "agent_tasks",
        bootstrap_servers=brokers,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        group_id="agent-workers",
    )
    for msg in consumer:
        log.info("task received: %s", msg.value)


if __name__ == "__main__":
    main()
