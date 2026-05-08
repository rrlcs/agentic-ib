"""Synthesis agent — composes the investment memo, optionally grounded in past memos.

Uses the agent runtime so it can call vector_search to find prior memos for
the same symbol or sector, then writes the final structured memo. The memo is
persisted to Pinecone for future RAG retrieval (action_only follow-ups,
synthesis grounding on subsequent runs).
"""

from __future__ import annotations

from agent_runtime import handlers_for, run_agent_loop, tools_for
from mcp.schema import MCPMessage
from memory.vector_db import store_memo
from observability.logger import get_logger, log_event

log = get_logger(__name__)


_SYSTEM_PROMPT = """You are an investment banker writing a clear, decisive memo.

You may call vector_search to surface prior memos on the same company /
sector for benchmarking. Use them only for grounding; do NOT reuse text
verbatim — write a fresh memo.

Your memo MUST contain these sections, in this order, using exactly these
markdown headings:

## Summary
## Key Strengths
## Risks
## Recommendation

The Recommendation section must end with one of: **Buy**, **Hold**, or
**Sell** (in bold), followed by a one-sentence rationale.
"""


def run(mcp_msg: MCPMessage) -> MCPMessage:
    log_event(log, "synthesis_started", task_id=mcp_msg.task_id)
    ctx = mcp_msg.context

    user_prompt = f"""
Company: {ctx.get("company")}
Symbol: {ctx.get("symbol")}

Research:
{ctx.get("research") or "<no research collected>"}

Financial Analysis:
{ctx.get("financial_analysis") or "<no financial analysis>"}

Risks:
{ctx.get("risk") or "<no risk analysis>"}

Write the memo now."""

    result = run_agent_loop(
        agent_name="synthesis_agent",
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tools=tools_for("synthesis_agent"),
        tool_handlers=handlers_for("synthesis_agent"),
        max_iterations=3,
    )
    report = result["answer"]
    ctx["final_report"] = report
    ctx.setdefault("agent_traces", {})["synthesis_agent"] = {
        "iterations": result["iterations"],
        "tool_calls": result["tool_calls"],
        "status": result["status"],
    }
    mcp_msg.metadata["step"] = "synthesis_done"

    persisted = store_memo(
        task_id=mcp_msg.task_id,
        symbol=ctx.get("symbol"),
        company=ctx.get("company"),
        memo=report,
        intent=ctx.get("intent"),
    )
    ctx["memo_persisted"] = persisted
    log_event(
        log,
        "synthesis_completed",
        task_id=mcp_msg.task_id,
        length=len(report or ""),
        memo_persisted=persisted,
        iterations=result["iterations"],
        n_tool_calls=len(result["tool_calls"]),
    )
    return mcp_msg
