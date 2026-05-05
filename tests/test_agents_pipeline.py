from mcp.builder import build_mcp_message
from mcp.schema import MCPMessage
from workers.dispatcher import dispatch


def test_full_agent_pipeline_with_mocks(monkeypatch) -> None:
    monkeypatch.setattr("agents.research_agent.generate_response", lambda *_: "research text")
    monkeypatch.setattr("agents.financial_agent.generate_response", lambda *_: "financial analysis text")
    monkeypatch.setattr(
        "agents.financial_agent.get_financials",
        lambda _symbol: {
            "name": "Acme Inc",
            "market_cap": "12345",
            "pe_ratio": "20.1",
            "revenue_ttm": "999",
            "profit_margin": "0.12",
        },
    )
    monkeypatch.setattr("agents.risk_agent.generate_response", lambda *_: "top risks")
    monkeypatch.setattr(
        "agents.synthesis_agent.generate_response",
        lambda *_: "Recommendation: Buy\nMarket cap: 12345\nDetailed report body " + ("x" * 140),
    )
    monkeypatch.setattr(
        "agents.validator_agent.generate_response",
        lambda *_: '{"valid": true, "issues": "", "confidence": 0.9}',
    )

    current: MCPMessage | dict = build_mcp_message(
        "job-full-1",
        agent="router",
        payload={"task_name": "investment_analysis", "payload": {"company": "Acme", "symbol": "AAPL"}},
    )

    for _ in range(8):
        if isinstance(current, dict) and current.get("status"):
            break
        assert isinstance(current, MCPMessage)
        current = dispatch(current)

    assert isinstance(current, dict)
    assert current["status"] == "success"
    assert current["task_id"] == "job-full-1"
    assert current["grounded"] is True
    assert "Recommendation" in current["result"]
