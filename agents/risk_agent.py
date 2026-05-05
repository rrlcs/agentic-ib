"""Risk agent — consumes MCP envelope, returns result dict for Kafka."""
from __future__ import annotations
from tools.llm_client import generate_response
from mcp.schema import MCPMessage
from observability.logger import get_logger, log_event

log = get_logger(__name__)


def run(mcp_msg: MCPMessage) -> MCPMessage:
    log_event(log, "risk_started", task_id=mcp_msg.task_id)
    research = mcp_msg.context.get("research")
    financials = mcp_msg.context.get("financial_analysis")

    system_prompt = "You are a risk analyst."
    user_prompt = f"""
    Based on:
    Research: {research}
    Financials: {financials}

    Identify top risks in investing in this company.
    """

    risk = generate_response(system_prompt, user_prompt)

    mcp_msg.context["risk"] = risk
    mcp_msg.metadata["step"] = "risk_done"
    mcp_msg.agent = "synthesis_agent"
    log_event(log, "risk_completed", task_id=mcp_msg.task_id, next_agent=mcp_msg.agent)

    return mcp_msg