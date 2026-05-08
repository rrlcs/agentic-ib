"""Tests for the tool-calling agent runtime."""

from __future__ import annotations

import json
from types import SimpleNamespace

from agent_runtime.loop import run_agent_loop


def _make_message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _make_response(message, usage=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=usage or SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


class _FakeOpenAI:
    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        if not self._scripted:
            raise RuntimeError("ran out of scripted responses")
        return self._scripted.pop(0)


def test_runtime_runs_single_tool_loop(monkeypatch):
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="echo", arguments=json.dumps({"text": "hi"})),
    )
    scripted = [
        _make_response(_make_message(content=None, tool_calls=[tool_call])),
        _make_response(_make_message(content="final answer based on tool")),
    ]

    monkeypatch.setattr("tools.llm_client._get_client", lambda: _FakeOpenAI(scripted))

    handlers = {"echo": lambda text: {"echoed": text}}
    tools = [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "echoes",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        }
    ]

    out = run_agent_loop(
        agent_name="research_agent",
        system_prompt="sys",
        user_prompt="say hi",
        tools=tools,
        tool_handlers=handlers,
        max_iterations=4,
    )

    assert out["status"] == "ok"
    assert out["iterations"] == 2
    assert out["answer"] == "final answer based on tool"
    assert out["tool_calls"][0]["name"] == "echo"
    assert json.loads(out["tool_calls"][0]["result_preview"]) == {"echoed": "hi"}


def test_runtime_fallback_no_tools(monkeypatch):
    monkeypatch.setattr("tools.llm_client.generate_response", lambda *_, **__: "plain answer")
    out = run_agent_loop(
        agent_name="answer_agent",
        system_prompt="sys",
        user_prompt="say hi",
        tools=[],
    )
    assert out["status"] == "ok"
    assert out["iterations"] == 1
    assert out["answer"] == "plain answer"
    assert out["tool_calls"] == []
