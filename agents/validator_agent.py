"""Validator agent — consumes MCP envelope, returns result dict for Kafka."""

from __future__ import annotations

from mcp.schema import MCPMessage
from observability.logger import get_logger, log_event
from tools.llm_client import generate_response

log = get_logger(__name__)


def run(mcp_msg: MCPMessage) -> MCPMessage:
    context = mcp_msg.context
    log_event(log, "validator_started", task_id=mcp_msg.task_id)

    report = context.get("final_report", "")
    research = context.get("research", "")
    financials = context.get("financials", {})

    # -------------------------
    # 1. Rule-based checks
    # -------------------------
    if not report or len(report) < 100:
        log_event(log, "validator_failed", task_id=mcp_msg.task_id, reason="report_too_short")
        return fail("Report too short")

    if "Recommendation" not in report:
        log_event(log, "validator_failed", task_id=mcp_msg.task_id, reason="missing_recommendation")
        return fail("Missing recommendation")

    # -------------------------
    # 2. LLM validation
    # -------------------------
    system_prompt = "You are a strict financial auditor."
    user_prompt = f"""
    Evaluate this investment report:

    {report}

    Check:
    - Is it logically consistent?
    - Are conclusions supported?
    - Any obvious hallucinations?

    Answer in JSON:
    {{
      "valid": true/false,
      "issues": "...",
      "confidence": 0-1
    }}
    """

    llm_eval = generate_response(system_prompt, user_prompt)


    # -------------------------
    # 3. Grounding check (simple)
    # -------------------------
    if str(financials.get("market_cap")) not in report:
        grounding_flag = False
    else:
        grounding_flag = True

    # -------------------------
    # Final decision
    # -------------------------
    result = {
        "task_id": mcp_msg.task_id,
        "status": "success",
        "result": report,
        "validation": llm_eval,
        "grounded": grounding_flag
    }
    log_event(log, "validator_completed", task_id=mcp_msg.task_id, grounded=grounding_flag)
    return result


def fail(reason):
    return {
        "status": "failed",
        "reason": reason
    }