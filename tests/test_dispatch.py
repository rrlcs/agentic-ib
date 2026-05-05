"""Dispatcher routing."""

from mcp.builder import build_mcp_message
from mcp.schema import MCPMessage
from workers.dispatcher import dispatch


def test_dispatch_runs_research_agent(monkeypatch) -> None:
    monkeypatch.setattr("agents.research_agent.generate_response", lambda *_: "research text for Acme")
    m = build_mcp_message(
        "job-1",
        agent="research_agent",
        payload={"task_name": "run_agent", "payload": {"company": "Acme"}},
    )
    out = dispatch(m)
    assert isinstance(out, MCPMessage)
    assert out.agent == "financial_agent"
    assert out.metadata["step"] == "research_done"
    assert "Acme" in out.context["research"]


def test_dispatch_router_passthrough() -> None:
    m = build_mcp_message(
        "job-2",
        agent="router",
        payload={"task_name": "run_agent", "payload": {}},
    )
    out = dispatch(m)
    assert isinstance(out, MCPMessage)
    assert out.agent == "research_agent"
    assert out.task_id == "job-2"
