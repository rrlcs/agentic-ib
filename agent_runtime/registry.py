"""Per-agent tool registry.

Defines OpenAI-compatible tool schemas and the matching Python handlers, so
each agent can be wired to a small, role-appropriate toolset.
"""

from __future__ import annotations

from typing import Any, Callable

from memory.vector_db import retrieve_memo, vector_search
from tools.financial_api import get_financials
from tools.paper_trader import get_order_status, place_paper_order, wait_for_fill
from tools.web_search import web_search


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

WEB_SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current public information about a company, "
            "product, sector, news event, or topic. Returns a list of "
            "{title, url, snippet} results you can quote and cite."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (English)."},
                "max_results": {
                    "type": "integer",
                    "description": "How many results to return (1-10).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}

VECTOR_SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "vector_search",
        "description": (
            "Semantic search over our archive of past investment memos "
            "(Pinecone). Use this to find prior analysis on a ticker / "
            "company / theme to ground new claims or compare recommendations. "
            "Returns up to top_k {symbol, company, memo_excerpt, score}."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 3},
            },
            "required": ["query"],
        },
    },
}

GET_FINANCIALS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_financials",
        "description": (
            "Fetch fundamental financial data for a ticker from Alpha Vantage "
            "(overview, income statement, balance sheet, cash flow, earnings)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker, e.g. NVDA."},
            },
            "required": ["symbol"],
        },
    },
}

PLACE_ORDER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "place_paper_order",
        "description": (
            "Submit a market order on Alpaca paper trading. Returns the "
            "broker response including order_id."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "qty": {"type": "integer", "description": "1-5"},
                "side": {"type": "string", "enum": ["buy", "sell"]},
            },
            "required": ["symbol", "qty", "side"],
        },
    },
}

WAIT_FOR_FILL_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "wait_for_fill",
        "description": (
            "Poll Alpaca for an order until it reaches a terminal state "
            "(filled, canceled, expired, etc.) or the timeout elapses. "
            "Use this immediately after place_paper_order to verify the trade."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "timeout": {"type": "number", "default": 15.0},
            },
            "required": ["order_id"],
        },
    },
}

GET_ORDER_STATUS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_order_status",
        "description": "Fetch the current status of an Alpaca paper order by order_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
            },
            "required": ["order_id"],
        },
    },
}

RETRIEVE_MEMO_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "retrieve_memo",
        "description": (
            "Retrieve the most relevant prior investment memo for a symbol "
            "from the Pinecone memo store. Use when you need the latest "
            "recommendation we wrote for a given ticker."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
            },
            "required": ["symbol"],
        },
    },
}


_AGENT_TOOLS: dict[str, list[dict[str, Any]]] = {
    "research_agent": [WEB_SEARCH_TOOL, VECTOR_SEARCH_TOOL],
    "financial_agent": [GET_FINANCIALS_TOOL, WEB_SEARCH_TOOL, VECTOR_SEARCH_TOOL],
    "risk_agent": [WEB_SEARCH_TOOL, VECTOR_SEARCH_TOOL],
    "synthesis_agent": [VECTOR_SEARCH_TOOL],
    "validator_agent": [VECTOR_SEARCH_TOOL, WEB_SEARCH_TOOL],
    "action_agent": [
        RETRIEVE_MEMO_TOOL,
        PLACE_ORDER_TOOL,
        WAIT_FOR_FILL_TOOL,
        GET_ORDER_STATUS_TOOL,
    ],
}


_HANDLERS: dict[str, Callable[..., Any]] = {
    "web_search": web_search,
    "vector_search": vector_search,
    "get_financials": get_financials,
    "retrieve_memo": lambda symbol=None, query=None: retrieve_memo(symbol, query),
    "place_paper_order": lambda symbol, qty, side: place_paper_order(
        symbol=symbol, qty=qty, side=side
    ),
    "wait_for_fill": lambda order_id, timeout=15.0: wait_for_fill(order_id, timeout=timeout),
    "get_order_status": lambda order_id: get_order_status(order_id),
}


def tools_for(agent_name: str) -> list[dict[str, Any]]:
    return list(_AGENT_TOOLS.get(agent_name, []))


def handlers_for(agent_name: str) -> dict[str, Callable[..., Any]]:
    """Default handler set; agents can wrap individual handlers to capture state."""
    names = {tool["function"]["name"] for tool in _AGENT_TOOLS.get(agent_name, [])}
    return {name: _HANDLERS[name] for name in names if name in _HANDLERS}
