"""MCP-style task envelope (internal contract for Kafka messages)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MCPMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    agent: str
    context: dict[str, Any] = Field(default_factory=dict)
    instructions: str
    metadata: dict[str, Any] = Field(default_factory=dict)