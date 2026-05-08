"""Answer agent — composes the user-facing chat reply (streamed token-by-token)."""

from __future__ import annotations

import json

from mcp.schema import MCPMessage
from observability.logger import get_logger, log_event
from tools.llm_client import generate_response_stream

log = get_logger(__name__)


_SYSTEM_PROMPT = (
    "You are a professional financial assistant chatting with the user. "
    "Reply in a conversational tone (no headings, no JSON). Keep it tight: "
    "2–6 short paragraphs or a short bullet list. Use only facts from the "
    "supplied context — if something is missing, say so plainly. End with a "
    "single sentence stating your recommendation if one is implied."
)


def run(mcp_msg: MCPMessage) -> MCPMessage:
    ctx = mcp_msg.context
    question = ctx.get("question") or "Provide a concise summary."
    log_event(log, "answer_started", task_id=mcp_msg.task_id, intent=ctx.get("intent"))

    action_section = ""
    action_result = ctx.get("action_result")
    if isinstance(action_result, dict):
        action_section = (
            "\nTrade execution result (paper):\n"
            + json.dumps(_compact_action(action_result), indent=2)
        )

    validator_section = ""
    validator = ctx.get("validator_feedback")
    if isinstance(validator, dict):
        validator_section = (
            f"\nValidator feedback: decision={validator.get('decision')} "
            f"confidence={validator.get('confidence')} "
            f"issues={validator.get('issues')}"
        )

    user_prompt = f"""
User asked: {question}

Company: {ctx.get("company")}
Symbol: {ctx.get("symbol")}
Intent: {ctx.get("intent")}

Research notes:
{ctx.get("research") or "<no research collected>"}

Financial analysis:
{ctx.get("financial_analysis") or "<no financial analysis collected>"}

Risks:
{ctx.get("risk") or "<no risk analysis collected>"}

Synthesised memo:
{ctx.get("final_report") or "<no memo>"}
{validator_section}
{action_section}
"""

    answer = generate_response_stream(_SYSTEM_PROMPT, user_prompt)
    ctx["answer"] = answer
    mcp_msg.metadata["step"] = "answer_done"
    log_event(log, "answer_completed", task_id=mcp_msg.task_id, length=len(answer or ""))
    return mcp_msg


def _compact_action(action: dict) -> dict:
    decision = action.get("decision") or {}
    broker = action.get("broker_response") or {}
    return {
        "executed": action.get("executed"),
        "action": decision.get("action"),
        "quantity": decision.get("quantity"),
        "rationale": decision.get("rationale"),
        "broker_status": broker.get("status"),
        "order_id": broker.get("id"),
    }
