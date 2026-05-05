"""Research agent — consumes MCP envelope, returns result dict for Kafka."""

from __future__ import annotations

from mcp.schema import MCPMessage
from observability.logger import get_logger, log_event
from tools.llm_client import generate_response

log = get_logger(__name__)


def run(mcp_msg: MCPMessage) -> MCPMessage:
    payload = mcp_msg.context.get("payload") or {}
    company = payload.get("company", "the company")
    log_event(log, "research_started", task_id=mcp_msg.task_id, company=company)

    system_prompt = "You are a financial research analyst."
    user_prompt = f"""
    Give a concise business overview of {company}.
    Include business model, market, and competitors.
    """

    research = generate_response(system_prompt, user_prompt)
    # log research output
    log_event(log, "research_completed", task_id=mcp_msg.task_id, research=research)

    mcp_msg.context["research"] = research
    mcp_msg.metadata["step"] = "research_done"
    mcp_msg.agent = "financial_agent"
    log_event(log, "research_completed", task_id=mcp_msg.task_id, next_agent=mcp_msg.agent)

    return mcp_msg
