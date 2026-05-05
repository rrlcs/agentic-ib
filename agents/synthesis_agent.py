"""Synthesis agent — consumes MCP envelope, returns result dict for Kafka."""

from __future__ import annotations
from tools.llm_client import generate_response
from mcp.schema import MCPMessage
from observability.logger import get_logger, log_event

log = get_logger(__name__)


def run(mcp_msg: MCPMessage) -> MCPMessage:
    log_event(log, "synthesis_started", task_id=mcp_msg.task_id)
    context = mcp_msg.context

    system_prompt = "You are an investment banker."
    user_prompt = f"""
    Create a structured investment memo:

    Company: {context.get("company")}
    Research: {context.get("research")}
    Financial Analysis: {context.get("financial_analysis")}
    Risks: {context.get("risk")}

    Include:
    - Summary
    - Key Strengths
    - Risks
    - Recommendation
    """

    report = generate_response(system_prompt, user_prompt)

    mcp_msg.context["final_report"] = report
    mcp_msg.metadata["step"] = "completed"
    mcp_msg.agent = "validator_agent"
    log_event(log, "synthesis_completed", task_id=mcp_msg.task_id, next_agent=mcp_msg.agent)

    return mcp_msg