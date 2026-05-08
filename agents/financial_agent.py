"""Financial agent — autonomous fundamentals analysis with tools.

Captures the raw ``get_financials`` result so other agents (synthesis,
validator) can ground claims against the structured data, not just the prose.
"""

from __future__ import annotations

from agent_runtime import handlers_for, run_agent_loop, tools_for
from mcp.schema import MCPMessage
from observability.logger import get_logger, log_event
from tools.financial_api import get_financials

log = get_logger(__name__)


_SYSTEM_PROMPT = """You are a fundamentals-driven financial analyst.

Loop:
1. Call get_financials(symbol) once to pull overview / income / balance sheet
   / cash flow / earnings. The result is large — only quote the numbers you
   actually use.
2. Optionally call web_search for the latest earnings commentary, sector
   trends, or analyst takes that contextualise the numbers.
3. Optionally call vector_search to check our prior write-ups on this name.
4. Produce a tight, numbers-forward analysis covering:
   - revenue trend, profitability, margins
   - cash flow / balance sheet health
   - valuation context (PE, market cap)
   - 1-3 quantitative red or green flags

Never invent figures. If a number is missing in the tool result, say so.
"""


def run(mcp_msg: MCPMessage) -> MCPMessage:
    symbol = mcp_msg.context.get("symbol") or "AAPL"
    log_event(log, "financial_started", task_id=mcp_msg.task_id, symbol=symbol)

    captured: dict = {}

    def get_financials_capture(symbol: str):
        data = get_financials(symbol)
        if isinstance(data, dict):
            captured.update(data)
        return data

    handlers = handlers_for("financial_agent")
    handlers["get_financials"] = get_financials_capture

    user_prompt = (
        f"Symbol: {symbol}\n"
        f"Company: {mcp_msg.context.get('company') or '<unknown>'}\n"
        f"User question: {mcp_msg.context.get('question') or 'Analyse fundamentals'}\n"
    )

    result = run_agent_loop(
        agent_name="financial_agent",
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tools=tools_for("financial_agent"),
        tool_handlers=handlers,
        max_iterations=4,
    )

    mcp_msg.context["financials"] = captured
    mcp_msg.context["financial_analysis"] = result["answer"]
    mcp_msg.context.setdefault("agent_traces", {})["financial_agent"] = {
        "iterations": result["iterations"],
        "tool_calls": result["tool_calls"],
        "status": result["status"],
    }
    mcp_msg.metadata["step"] = "financial_done"
    log_event(
        log,
        "financial_completed",
        task_id=mcp_msg.task_id,
        has_data=bool(captured),
        iterations=result["iterations"],
        n_tool_calls=len(result["tool_calls"]),
    )
    return mcp_msg
