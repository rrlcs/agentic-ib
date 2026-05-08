"""Route MCP envelopes to agent implementations."""

from __future__ import annotations

from agents.action_agent import run as action_run
from agents.answer_agent import run as answer_run
from agents.financial_agent import run as financial_run
from agents.research_agent import run as research_run
from agents.risk_agent import run as risk_run
from agents.router import run as router_run
from agents.synthesis_agent import run as synthesis_run
from agents.validator_agent import run as validator_run
from mcp.schema import MCPMessage
from observability.logger import get_logger, log_event

log = get_logger(__name__)

_AGENTS = {
    "router": router_run,
    "research_agent": research_run,
    "financial_agent": financial_run,
    "risk_agent": risk_run,
    "synthesis_agent": synthesis_run,
    "validator_agent": validator_run,
    "answer_agent": answer_run,
    "action_agent": action_run,
}


def dispatch(mcp_msg: MCPMessage) -> MCPMessage | dict:
    agent = mcp_msg.agent
    log_event(log, "dispatch_invoked", task_id=mcp_msg.task_id, agent=agent)
    handler = _AGENTS.get(agent)
    if handler is None:
        log_event(log, "dispatch_unknown_agent", task_id=mcp_msg.task_id, agent=agent)
        return {"task_id": mcp_msg.task_id, "status": "failed", "reason": f"unknown_agent:{agent}"}
    return handler(mcp_msg)
