"""Validator agent — LLM-grounded critique with optional RAG / web evidence.

The agent runtime drives a tool-calling loop where the validator can pull
prior memos via vector_search, optionally hit the web, and then return a
structured JSON verdict. The decision dict drives the worker's feedback loop
(re_research / re_synthesis) just like before.
"""

from __future__ import annotations

import json
from typing import Any

from agent_runtime import handlers_for, run_agent_loop, tools_for
from mcp.schema import MCPMessage
from observability.logger import get_logger, log_event
from observability.tracer import increment_task_metric

log = get_logger(__name__)


_SYSTEM_PROMPT = """You are a strict financial auditor reviewing a memo.

You have tools:
- vector_search: pull prior memos to compare claims with our own historical
  baseline on this name/sector.
- web_search: verify any non-trivial claim against current sources.

Process:
1. Inspect the memo against the supplied research notes and structured
   financials.
2. Use the tools to ground at least one questionable claim if any exists.
3. When you have enough evidence, output ONLY this JSON object:

{
  "grounded": true|false,
  "valid": true|false,
  "issues": "<short plain text>",
  "confidence": 0.0-1.0,
  "missing_information": ["..."],
  "decision": "accept" | "re_research" | "re_synthesis"
}

Decision rules:
- "re_research" if evidence is *insufficient* (need more facts/financials)
- "re_synthesis" if evidence is sufficient but the memo is inconsistent /
  unsupported / missing required sections
- "accept" otherwise

Do not include any text outside the JSON object.
"""


def run(mcp_msg: MCPMessage) -> dict[str, Any]:
    log_event(log, "validator_started", task_id=mcp_msg.task_id)
    ctx = mcp_msg.context
    report = (ctx.get("final_report") or "").strip()
    research = ctx.get("research") or ""
    financials = ctx.get("financials") or {}

    # Cheap structural pre-checks (avoid wasting an LLM call when obvious).
    if len(report) < 100:
        return _emit_decision(
            mcp_msg,
            decision="re_synthesis",
            valid=False,
            grounded=False,
            issues="report_too_short",
            confidence=0.0,
        )
    if "Recommendation" not in report:
        return _emit_decision(
            mcp_msg,
            decision="re_synthesis",
            valid=False,
            grounded=False,
            issues="missing_recommendation_section",
            confidence=0.1,
        )

    user_prompt = f"""
Evidence
========
Research notes:
{research or "<empty>"}

Structured financials (truncated):
{json.dumps(_truncate(financials), indent=2)[:4000]}

Memo under review:
{report}

Audit it now and return ONLY the JSON object."""

    result = run_agent_loop(
        agent_name="validator_agent",
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tools=tools_for("validator_agent"),
        tool_handlers=handlers_for("validator_agent"),
        max_iterations=4,
        response_format={"type": "json_object"},
    )

    parsed = _safe_json(result["answer"])
    decision = (parsed.get("decision") or "accept").lower()
    if decision not in {"accept", "re_research", "re_synthesis"}:
        decision = "accept"

    feedback = _emit_decision(
        mcp_msg,
        decision=decision,
        valid=bool(parsed.get("valid")),
        grounded=bool(parsed.get("grounded")),
        issues=str(parsed.get("issues", "")),
        confidence=float(parsed.get("confidence", 0.0) or 0.0),
        missing_information=parsed.get("missing_information") or [],
    )
    feedback["agent_trace"] = {
        "iterations": result["iterations"],
        "tool_calls": result["tool_calls"],
        "status": result["status"],
    }
    return feedback


def _emit_decision(
    mcp_msg: MCPMessage,
    *,
    decision: str,
    valid: bool,
    grounded: bool,
    issues: str,
    confidence: float,
    missing_information: list | None = None,
) -> dict[str, Any]:
    accept = decision == "accept"
    payload: dict[str, Any] = {
        "task_id": mcp_msg.task_id,
        "decision": decision,
        "valid": valid,
        "grounded": grounded,
        "issues": issues,
        "confidence": confidence,
        "missing_information": list(missing_information or []),
    }
    if accept:
        payload["status"] = "success"
        payload["result"] = mcp_msg.context.get("final_report")
        payload["intent"] = mcp_msg.context.get("intent")
    if decision in {"re_research", "re_synthesis"}:
        increment_task_metric(mcp_msg.task_id, "feedback_loops", 1)

    log_event(
        log,
        "validator_decided",
        task_id=mcp_msg.task_id,
        decision=decision,
        valid=valid,
        grounded=grounded,
        confidence=confidence,
    )
    mcp_msg.context["validator_feedback"] = payload
    return payload


def _safe_json(text: str) -> dict[str, Any]:
    if not text:
        return {}
    cleaned = text.strip().strip("`")
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


def _truncate(financials: Any) -> Any:
    """Drop bulky lists from Alpha Vantage payloads so the prompt stays small."""
    if not isinstance(financials, dict):
        return financials
    out: dict[str, Any] = {}
    for key, value in financials.items():
        if isinstance(value, dict) and isinstance(value.get("annualReports"), list):
            cloned = dict(value)
            cloned["annualReports"] = cloned["annualReports"][:1]
            cloned.pop("quarterlyReports", None)
            out[key] = cloned
        else:
            out[key] = value
    return out
