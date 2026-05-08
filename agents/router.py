"""Router agent — classifies user intent and emits a dynamic agent pipeline.

The pipeline is a list whose entries are either an agent name (str) or a list
of agent names that should run in parallel. The worker honours both shapes.

Every pipeline now ends with ``answer_agent`` so the user always gets a clean,
chat-style reply (not raw research / memo dumps).
"""

from __future__ import annotations

import json
from typing import Any

from mcp.schema import MCPMessage
from observability.logger import get_logger, log_event
from tools.llm_client import generate_response

log = get_logger(__name__)

PipelineStep = str | list[str]
Pipeline = list[PipelineStep]

_DEFAULT_PIPELINES: dict[str, Pipeline] = {
    "recommendation": [
        ["research_agent", "financial_agent"],
        "risk_agent",
        "synthesis_agent",
        "validator_agent",
        "action_agent",
        "answer_agent",
    ],
    "research": ["research_agent", "answer_agent"],
    "financials": ["financial_agent", "answer_agent"],
    "risks": [
        ["research_agent", "financial_agent"],
        "risk_agent",
        "answer_agent",
    ],
    "qna": ["research_agent", "answer_agent"],
    "action_only": ["action_agent", "answer_agent"],
}

_RETRY_BUDGET_DEFAULT = {"research": 1, "synthesis": 2}

_SYSTEM_PROMPT = """You are an intent classifier for a financial agent platform.

Read a user message and produce a JSON plan describing:
- intent: one of [recommendation, research, financials, risks, qna, action_only]
- company: company name (string or null)
- symbol: ticker symbol if known (string or null)
- question: the user's underlying question (string)

Always respond with ONLY a JSON object, no prose."""


def _llm_plan(question: str) -> dict[str, Any]:
    raw = generate_response(_SYSTEM_PROMPT, question)
    try:
        cleaned = (raw or "").strip().strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _normalise_pipeline(value: Any, intent: str) -> Pipeline:
    """Sanity check whatever the LLM produced; fall back to defaults."""
    fallback = _DEFAULT_PIPELINES.get(intent) or _DEFAULT_PIPELINES["qna"]
    if not isinstance(value, list):
        return list(fallback)
    cleaned: Pipeline = []
    for step in value:
        if isinstance(step, str) and step:
            cleaned.append(step)
        elif isinstance(step, list) and all(isinstance(x, str) and x for x in step):
            cleaned.append(list(step))
    return cleaned or list(fallback)


def run(mcp_msg: MCPMessage) -> MCPMessage:
    payload = mcp_msg.context.get("payload") or {}
    question = payload.get("question") or payload.get("company") or "investment recommendation"
    requested_model = payload.get("model") or mcp_msg.context.get("model")

    plan = _llm_plan(question)
    intent = (plan.get("intent") or "qna").lower()
    pipeline = _normalise_pipeline(_DEFAULT_PIPELINES.get(intent, _DEFAULT_PIPELINES["qna"]), intent)

    company = plan.get("company") or payload.get("company")
    symbol = plan.get("symbol") or payload.get("symbol")

    if company and not payload.get("company"):
        payload["company"] = company
    if symbol and not payload.get("symbol"):
        payload["symbol"] = symbol
    payload["question"] = question
    mcp_msg.context["payload"] = payload
    mcp_msg.context["company"] = company
    mcp_msg.context["symbol"] = symbol
    mcp_msg.context["question"] = question
    mcp_msg.context["intent"] = intent
    if requested_model:
        mcp_msg.context["model"] = requested_model

    mcp_msg.metadata["pipeline"] = pipeline
    mcp_msg.metadata["pipeline_idx"] = -1  # worker advances to 0 next
    mcp_msg.metadata.setdefault("retry_budget", dict(_RETRY_BUDGET_DEFAULT))

    log_event(
        log,
        "router_planned",
        task_id=mcp_msg.task_id,
        intent=intent,
        company=company,
        symbol=symbol,
        pipeline=pipeline,
        model=requested_model or "default",
    )
    return mcp_msg
