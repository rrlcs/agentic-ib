"""Parse and normalize MCP envelopes from Kafka-compatible broker / HTTP payloads."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from mcp.schema import MCPMessage


def parse_mcp_message(raw: dict[str, Any]) -> MCPMessage:
    """Validate dict-shaped MCP envelope (post JSON decode)."""
    return MCPMessage.model_validate(raw)


def parse_mcp_message_safe(raw: dict[str, Any]) -> MCPMessage | None:
    """Like parse_mcp_message but returns None on validation failure."""
    try:
        return MCPMessage.model_validate(raw)
    except ValidationError:
        return None


def message_to_broker_dict(message: MCPMessage) -> dict[str, Any]:
    """Serialize for broker JSON producers/consumers."""
    return message.model_dump(mode="json")


def message_to_kafka_dict(message: MCPMessage) -> dict[str, Any]:
    """Backward-compatible alias for older imports."""
    return message_to_broker_dict(message)
