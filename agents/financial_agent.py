"""Financial agent — consumes MCP envelope, returns result dict for Kafka."""

from __future__ import annotations

from mcp.schema import MCPMessage
from observability.logger import get_logger, log_event
from tools.llm_client import generate_response
from tools.financial_api import get_financials

log = get_logger(__name__)


def run(mcp_msg: MCPMessage) -> MCPMessage:
    payload = mcp_msg.context.get("payload") or {}
    symbol = payload.get("symbol", "AAPL")
    log_event(log, "financial_started", task_id=mcp_msg.task_id, symbol=symbol)
    financial_data = get_financials(symbol)

    system_prompt = "You are a financial analyst."
    user_prompt = f"""
    Analyze these financials:
    {financial_data}

    Provide key insights in simple terms.
    """

    analysis = generate_response(system_prompt, user_prompt)

    mcp_msg.context["financials"] = financial_data
    mcp_msg.context["financial_analysis"] = analysis

    mcp_msg.metadata["step"] = "financial_done"
    mcp_msg.agent = "risk_agent"
    log_event(log, "financial_completed", task_id=mcp_msg.task_id, next_agent=mcp_msg.agent)

    return mcp_msg