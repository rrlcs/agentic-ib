"""Pipeline integration test using mocked LLM/Tool calls + the real worker loop."""

from __future__ import annotations

import json
from typing import Any

from mcp.builder import build_mcp_message
from workers.worker import _process_envelope


_VALIDATOR_OK = {
    "valid": True,
    "grounded": True,
    "issues": "",
    "confidence": 0.9,
    "missing_information": [],
    "decision": "accept",
}

_VALIDATOR_RE_SYNTH_FIRST = {
    "valid": False,
    "grounded": False,
    "issues": "missing recommendation details",
    "confidence": 0.3,
    "missing_information": ["valuation"],
    "decision": "re_synthesis",
}


_AGENT_RESPONSES: dict[str, Any] = {}


def _fake_run_agent_loop(*, agent_name, system_prompt, user_prompt, tools=None, tool_handlers=None, **_):
    response = _AGENT_RESPONSES.get(agent_name)
    if callable(response):
        answer = response(agent_name=agent_name, prompt=user_prompt, tool_handlers=tool_handlers or {})
    else:
        answer = response or ""
    return {
        "answer": answer if isinstance(answer, str) else json.dumps(answer),
        "tool_calls": [],
        "iterations": 1,
        "status": "ok",
    }


def _patch_runtime(monkeypatch, *, validator_responses, recommendation_pipeline=True):
    counts = {"validator": list(validator_responses)}

    def validator_answer(**_):
        return json.dumps(counts["validator"].pop(0))

    _AGENT_RESPONSES.clear()
    _AGENT_RESPONSES["research_agent"] = "research-text"
    _AGENT_RESPONSES["financial_agent"] = "fin-analysis"
    _AGENT_RESPONSES["risk_agent"] = "risk-text"
    _AGENT_RESPONSES["synthesis_agent"] = "## Summary\n\n## Key Strengths\n\n## Risks\n\n## Recommendation\n\n**Buy** — strong fundamentals.\n" + ("Body " * 60)
    _AGENT_RESPONSES["validator_agent"] = validator_answer
    _AGENT_RESPONSES["action_agent"] = json.dumps(
        {
            "action": "buy",
            "quantity": 1,
            "rationale": "memo says buy",
            "order_id": "test-order",
            "broker_status": "filled",
            "filled_qty": 1,
            "filled_avg_price": 100.0,
            "executed": True,
            "memo_source": "context",
        }
    )
    _AGENT_RESPONSES["answer_agent"] = "Here's the chat-friendly summary about Acme."

    monkeypatch.setattr("agents.research_agent.run_agent_loop", _fake_run_agent_loop)
    monkeypatch.setattr("agents.financial_agent.run_agent_loop", _fake_run_agent_loop)
    monkeypatch.setattr("agents.risk_agent.run_agent_loop", _fake_run_agent_loop)
    monkeypatch.setattr("agents.synthesis_agent.run_agent_loop", _fake_run_agent_loop)
    monkeypatch.setattr("agents.validator_agent.run_agent_loop", _fake_run_agent_loop)
    monkeypatch.setattr("agents.action_agent.run_agent_loop", _fake_run_agent_loop)
    # Router still uses a non-runtime LLM call.
    monkeypatch.setattr(
        "agents.router.generate_response",
        lambda *_, **__: json.dumps(
            {
                "intent": "recommendation" if recommendation_pipeline else "qna",
                "company": "Acme",
                "symbol": "ACME",
                "question": "Should I invest in Acme?",
            }
        ),
    )
    # Answer agent still streams.
    monkeypatch.setattr(
        "agents.answer_agent.generate_response_stream",
        lambda *_, **__: _AGENT_RESPONSES["answer_agent"],
    )
    # Memo store / retrieve are no-ops in tests.
    monkeypatch.setattr("agents.synthesis_agent.store_memo", lambda **_: True)
    monkeypatch.setattr("agents.action_agent.retrieve_memo", lambda *_, **__: None)
    return counts


def test_recommendation_pipeline_completes(monkeypatch) -> None:
    _patch_runtime(monkeypatch, validator_responses=[_VALIDATOR_OK])

    mcp = build_mcp_message(
        "job-rec-1",
        agent="router",
        payload={"task_name": "chat_message", "payload": {"question": "Should I invest in Acme?"}},
    )
    terminal = _process_envelope(mcp)

    assert terminal["status"] == "success"
    ctx_keys = terminal.get("context_keys") or []
    for k in ("research", "financial_analysis", "risk", "final_report", "action_result", "answer"):
        assert k in ctx_keys, f"expected {k} in context_keys"
    assert terminal.get("answer", "").strip()
    action = terminal.get("action") or {}
    assert action.get("executed") is True
    assert action.get("order_id") == "test-order"


def test_validator_feedback_loop_triggers_re_synthesis(monkeypatch) -> None:
    counts = _patch_runtime(
        monkeypatch,
        validator_responses=[_VALIDATOR_RE_SYNTH_FIRST, _VALIDATOR_OK],
    )

    mcp = build_mcp_message(
        "job-rec-2",
        agent="router",
        payload={"task_name": "chat_message", "payload": {"question": "Buy Acme?"}},
    )
    terminal = _process_envelope(mcp)

    assert terminal["status"] == "success"
    assert counts["validator"] == []
    budget = mcp.metadata["retry_budget"]
    assert budget["synthesis"] == 1


def test_qna_pipeline_uses_answer_agent(monkeypatch) -> None:
    _patch_runtime(monkeypatch, validator_responses=[], recommendation_pipeline=False)
    _AGENT_RESPONSES["research_agent"] = "Acme makes widgets."

    monkeypatch.setattr(
        "agents.answer_agent.generate_response_stream",
        lambda *_, **__: "Acme is a widget company in industrial supply.",
    )

    mcp = build_mcp_message(
        "job-qna-1",
        agent="router",
        payload={"task_name": "chat_message", "payload": {"question": "What does Acme do?"}},
    )
    terminal = _process_envelope(mcp)
    assert terminal["status"] == "success"
    assert "Acme" in (terminal.get("answer") or "")
    assert terminal.get("intent") == "qna"


def test_action_only_uses_pinecone_memo(monkeypatch) -> None:
    """When intent=action_only, the action agent should fetch a stored memo."""
    monkeypatch.setattr(
        "agents.router.generate_response",
        lambda *_, **__: json.dumps(
            {
                "intent": "action_only",
                "company": "Acme",
                "symbol": "ACME",
                "question": "Place a paper trade for ACME",
            }
        ),
    )
    _AGENT_RESPONSES.clear()
    _AGENT_RESPONSES["action_agent"] = json.dumps(
        {
            "action": "buy",
            "quantity": 2,
            "rationale": "memo says buy",
            "order_id": "memo-order",
            "broker_status": "filled",
            "filled_qty": 2,
            "filled_avg_price": 50.0,
            "executed": True,
            "memo_source": "pinecone",
        }
    )
    monkeypatch.setattr("agents.action_agent.run_agent_loop", _fake_run_agent_loop)
    monkeypatch.setattr(
        "agents.action_agent.retrieve_memo",
        lambda *_, **__: {"memo": "Recommendation: Buy ACME", "score": 0.91, "id": "prev-task"},
    )
    monkeypatch.setattr(
        "agents.answer_agent.generate_response_stream",
        lambda *_, **__: "Trade placed for ACME based on the prior memo.",
    )

    mcp = build_mcp_message(
        "job-act-1",
        agent="router",
        payload={"task_name": "chat_message", "payload": {"question": "Place a paper trade for ACME"}},
    )
    terminal = _process_envelope(mcp)
    assert terminal["status"] == "success"
    assert terminal.get("memo_source") == "pinecone"
    action = terminal.get("action") or {}
    assert action.get("executed") is True
    assert action.get("order_id") == "memo-order"
