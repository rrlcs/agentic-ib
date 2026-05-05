"""Consume agent tasks from Kafka (run as a standalone process)."""

from __future__ import annotations

import json
import os

from kafka import KafkaConsumer
from mcp.schema import MCPMessage
from mcp.parser import parse_mcp_message_safe
from messaging.consumer import create_consumer
from messaging.topics import AGENT_TASKS, AGENT_RESULTS
from messaging.producer import send
from observability.logger import get_logger, log_event
from workers.dispatcher import dispatch

log = get_logger(__name__)


def _bootstrap_servers() -> str:
    return os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


def main() -> None:
    brokers = _bootstrap_servers()
    log_event(log, "worker_starting", bootstrap=brokers, topic=AGENT_TASKS)
    consumer = create_consumer(AGENT_TASKS)
    for msg in consumer:
        raw = msg.value if isinstance(msg.value, dict) else {}
        mcp = parse_mcp_message_safe(raw)
        if mcp:
            log_event(log, "mcp_envelope_received", task_id=mcp.task_id, agent=mcp.agent, step=mcp.metadata.get("step"))
            while True:
                try:
                    result = dispatch(mcp)
                except Exception as exc:
                    log.exception("dispatch_error task_id=%s agent=%s", mcp.task_id, mcp.agent)
                    send(AGENT_RESULTS, {"task_id": mcp.task_id, "status": "failed", "reason": str(exc)})
                    break
                if isinstance(result, MCPMessage):
                    mcp = result
                    log_event(log, "agent_handoff", task_id=mcp.task_id, next_agent=mcp.agent, step=mcp.metadata.get("step"))
                    continue
                if result.get("status"):
                    send(AGENT_RESULTS, result)
                    log_event(log, "pipeline_complete", task_id=result.get("task_id"), status=result.get("status"))
                    break
                log.warning("unexpected dispatch payload type=%s", type(result).__name__)
                break
        else:
            log.warning("non-MCP or invalid payload dropped keys=%s", list(raw.keys()))


if __name__ == "__main__":
    main()
