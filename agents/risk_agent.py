"""Risk agent — autonomous risk identification with tools."""

from __future__ import annotations

from agent_runtime import handlers_for, run_agent_loop, tools_for
from mcp.schema import MCPMessage
from observability.logger import get_logger, log_event

log = get_logger(__name__)


_SYSTEM_PROMPT = """You are a buy-side risk analyst.

You will receive prior research and financial analysis. Your job is to
identify the *top* 3-5 investment risks. Use the tools to:

- web_search for recent regulatory issues, lawsuits, supply-chain disruptions,
  or industry headwinds.
- vector_search for prior risk language we used on this name or sector.

Then produce a numbered list of the top risks. For each one give:
- one-line description
- why it matters (cite evidence: snippet or financial figure)
- severity tag (low / medium / high)
"""


def run(mcp_msg: MCPMessage) -> MCPMessage:
    log_event(log, "risk_started", task_id=mcp_msg.task_id)

    user_prompt = (
        f"Company: {mcp_msg.context.get('company')}\n"
        f"Symbol: {mcp_msg.context.get('symbol')}\n\n"
        f"Research notes:\n{mcp_msg.context.get('research') or '<none>'}\n\n"
        f"Financial analysis:\n{mcp_msg.context.get('financial_analysis') or '<none>'}\n\n"
        "Identify the top investment risks."
    )

    result = run_agent_loop(
        agent_name="risk_agent",
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tools=tools_for("risk_agent"),
        tool_handlers=handlers_for("risk_agent"),
        max_iterations=4,
    )

    mcp_msg.context["risk"] = result["answer"]
    mcp_msg.context.setdefault("agent_traces", {})["risk_agent"] = {
        "iterations": result["iterations"],
        "tool_calls": result["tool_calls"],
        "status": result["status"],
    }
    mcp_msg.metadata["step"] = "risk_done"
    log_event(
        log,
        "risk_completed",
        task_id=mcp_msg.task_id,
        length=len(result["answer"] or ""),
        iterations=result["iterations"],
        n_tool_calls=len(result["tool_calls"]),
    )
    return mcp_msg
