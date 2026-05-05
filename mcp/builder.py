"""Build MCP envelopes for outbound tasks."""

from __future__ import annotations

from typing import Any

from mcp.schema import MCPMessage


def build_mcp_message(
    job_id: str,
    *,
    agent: str = "router",
    payload: dict[str, Any],
) -> MCPMessage:
    """Wrap API/job inputs as MCP envelope context."""
    return MCPMessage(
        task_id=job_id,
        agent=agent,
        context=payload,
        instructions="route task",
        metadata={"step": "start"},
    )
