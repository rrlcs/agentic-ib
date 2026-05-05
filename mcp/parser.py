"""Parse and normalize MCP envelopes from Kafka / HTTP payloads."""

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


def message_to_kafka_dict(message: MCPMessage) -> dict[str, Any]:
    """Serialize for kafka-python JSON producer."""
    return message.model_dump(mode="json")
