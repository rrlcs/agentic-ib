from mcp.schema import MCPMessage
from observability.logger import get_logger, log_event

log = get_logger(__name__)


def run(mcp_msg: MCPMessage) -> MCPMessage:
    task = mcp_msg.context.get("task_name")

    if task == "investment_analysis":
        next_agent = "research_agent"
    else:
        next_agent = "research_agent"

    mcp_msg.agent = next_agent
    log_event(log, "router_selected_next_agent", task_id=mcp_msg.task_id, task=task, next_agent=next_agent)
    return mcp_msg