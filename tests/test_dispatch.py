"""Dispatcher routing — only verifies handler selection (not pipeline)."""

from __future__ import annotations

from mcp.builder import build_mcp_message
from mcp.schema import MCPMessage
from workers.dispatcher import dispatch


def test_dispatch_runs_research_agent(monkeypatch) -> None:
    monkeypatch.setattr(
        "agents.research_agent.run_agent_loop",
        lambda **_: {
            "answer": "research text for Acme",
            "tool_calls": [],
            "iterations": 1,
            "status": "ok",
        },
    )
    mcp = build_mcp_message(
        "job-1",
        agent="research_agent",
        payload={"task_name": "run_agent", "payload": {"company": "Acme"}},
    )
    mcp.context["company"] = "Acme"
    out = dispatch(mcp)
    assert isinstance(out, MCPMessage)
    assert out.metadata["step"] == "research_done"
    assert "research" in out.context
    assert out.context["research"] == "research text for Acme"


def test_dispatch_unknown_agent() -> None:
    mcp = build_mcp_message(
        "job-2",
        agent="totally_made_up",
        payload={"task_name": "run_agent", "payload": {}},
    )
    out = dispatch(mcp)
    assert isinstance(out, dict)
    assert out["status"] == "failed"
    assert "unknown_agent" in out["reason"]
