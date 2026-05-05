"""Route MCP envelopes to agent implementations."""

from __future__ import annotations

from agents.research_agent import run as research_run
from agents.router import run as router_run
from mcp.schema import MCPMessage
from agents.financial_agent import run as financial_run
from agents.risk_agent import run as risk_run
from agents.synthesis_agent import run as synthesis_run
from agents.validator_agent import run as validator_run
from observability.logger import get_logger, log_event

log = get_logger(__name__)

def dispatch(mcp_msg: MCPMessage) -> MCPMessage | dict:
    agent = mcp_msg.agent
    log_event(log, "dispatch_invoked", task_id=mcp_msg.task_id, agent=agent)
    if agent == "router":
        return router_run(mcp_msg)
    elif agent == "research_agent":
        return research_run(mcp_msg)
    elif agent == "financial_agent":
        return financial_run(mcp_msg)
    elif agent == "risk_agent":
        return risk_run(mcp_msg)
    elif agent == "synthesis_agent":
        return synthesis_run(mcp_msg)
    elif agent == "validator_agent":
        return validator_run(mcp_msg)
    
    return mcp_msg
