"""Research agent — autonomous web/RAG research using the tool-calling runtime."""

from __future__ import annotations

from agent_runtime import handlers_for, run_agent_loop, tools_for
from mcp.schema import MCPMessage
from observability.logger import get_logger, log_event

log = get_logger(__name__)


_SYSTEM_PROMPT = """You are a financial research analyst with web access and a
memo archive. You are given a company and a user question.

Loop:
1. Call web_search at least once for the company's recent business / news /
   competitive landscape. Issue follow-up searches if a result raises a new
   question (e.g. a recent acquisition, regulatory action).
2. Optionally call vector_search to see if we have prior internal memos on
   the same company; reuse anything relevant.
3. When you have enough evidence, return a CONCISE business overview with:
   - business model & primary segments
   - market & competitors
   - 2-4 recent material developments (with citation indexes [1], [2]...)
   - data gaps (one short list, if any)

Never invent figures. If a fact isn't supported by a tool result, say so.
"""


def run(mcp_msg: MCPMessage) -> MCPMessage:
    company = mcp_msg.context.get("company") or "the company"
    question = mcp_msg.context.get("question") or "Provide a research overview"
    log_event(log, "research_started", task_id=mcp_msg.task_id, company=company)

    user_prompt = (
        f"Company: {company}\n"
        f"Symbol: {mcp_msg.context.get('symbol') or '<unknown>'}\n"
        f"User question: {question}\n\n"
        "Research this company. Use the tools available."
    )

    result = run_agent_loop(
        agent_name="research_agent",
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tools=tools_for("research_agent"),
        tool_handlers=handlers_for("research_agent"),
        max_iterations=5,
    )

    mcp_msg.context["research"] = result["answer"]
    mcp_msg.context.setdefault("agent_traces", {})["research_agent"] = {
        "iterations": result["iterations"],
        "tool_calls": result["tool_calls"],
        "status": result["status"],
    }
    mcp_msg.metadata["step"] = "research_done"
    log_event(
        log,
        "research_completed",
        task_id=mcp_msg.task_id,
        length=len(result["answer"] or ""),
        iterations=result["iterations"],
        n_tool_calls=len(result["tool_calls"]),
    )
    return mcp_msg
