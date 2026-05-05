"""MCP schema, builder, parser, and Kafka JSON shape."""

import json

import pytest
from pydantic import ValidationError

from mcp.builder import build_mcp_message
from mcp.parser import message_to_kafka_dict, parse_mcp_message, parse_mcp_message_safe
from mcp.schema import MCPMessage


def test_build_serialize_parse_round_trip() -> None:
    job_id = "00000000-0000-0000-0000-000000000001"
    inner = {"task_name": "run_agent", "payload": {"q": 1}}
    m = build_mcp_message(job_id, payload=inner)
    wire = message_to_kafka_dict(m)
    blob = json.dumps(wire)
    loaded = json.loads(blob)
    back = parse_mcp_message(loaded)
    assert back == m
    assert back.context["task_name"] == "run_agent"
    assert back.context["payload"] == {"q": 1}


def test_parse_rejects_missing_fields() -> None:
    with pytest.raises(ValidationError):
        parse_mcp_message({"task_id": "x"})


def test_parse_safe_none_on_bad_input() -> None:
    assert parse_mcp_message_safe({}) is None
    assert parse_mcp_message_safe({"task_id": "a", "nope": True}) is None


def test_mcp_message_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        MCPMessage(
            task_id="t",
            agent="a",
            context={},
            instructions="i",
            metadata={},
            extra_field="bad",
        )
