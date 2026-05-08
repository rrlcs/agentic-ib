"""Action agent — autonomous paper-trade execution with verification.

This is the most "agentic" stage of the pipeline. It reads (or retrieves
via RAG) the memo, decides on an action, places the order, then *verifies*
that the order actually filled by polling Alpaca via wait_for_fill. The
final structured ``action_result`` written to context tells the user
whether the trade was actually executed (and at what price).
"""

from __future__ import annotations

import os
import re
from typing import Any

from agent_runtime import handlers_for, run_agent_loop, tools_for
from mcp.schema import MCPMessage
from memory.vector_db import retrieve_memo
from observability.logger import get_logger, log_event
from tools.paper_trader import get_order_status, place_paper_order, wait_for_fill

log = get_logger(__name__)


_SYSTEM_PROMPT = """You are a paper-trading desk operator. You may place at
most ONE order per task, with quantity 0-5.

Tools:
- retrieve_memo(symbol): if you don't have a memo in your prompt, fetch the
  most recent one for this symbol from the memo archive.
- place_paper_order(symbol, qty, side): submit a market order on Alpaca
  paper trading. Returns the order_id and initial broker state.
- wait_for_fill(order_id, timeout): poll until the order is filled,
  cancelled, or the timeout elapses. ALWAYS call this immediately after
  place_paper_order so you can VERIFY the trade actually went through.
- get_order_status(order_id): fetch the latest order state.

Process:
1. If you do not have a memo, retrieve_memo(symbol).
2. Decide buy / sell / hold and a quantity (1-3 unless very high conviction;
   never above 5; 0 for hold).
3. If buy or sell: place_paper_order, then wait_for_fill on the returned
   order_id (timeout 15-20s). If the broker reports `skipped` (no
   credentials), do NOT retry — that's expected.
4. Output ONLY this JSON object as your final message (no extra prose):

{
  "action": "buy" | "sell" | "hold",
  "quantity": 0-5,
  "rationale": "one sentence",
  "order_id": "<alpaca order id or null>",
  "broker_status": "<status returned by wait_for_fill or place_paper_order>",
  "filled_qty": <number or null>,
  "filled_avg_price": <number or null>,
  "executed": true|false,
  "memo_source": "context" | "pinecone" | "missing"
}

`executed` must be true ONLY if the broker reported `filled` or
`partially_filled`. Anything else (skipped, dry_run, timeout, error,
canceled, expired, rejected) is `false`.
"""


def run(mcp_msg: MCPMessage) -> MCPMessage:
    ctx = mcp_msg.context
    symbol = ctx.get("symbol")
    log_event(log, "action_started", task_id=mcp_msg.task_id, symbol=symbol)

    # If we already have a memo, prefer that. Otherwise fetch one once
    # before letting the agent loop run, so it can reason from it.
    in_memo = (ctx.get("final_report") or "").strip()
    memo_source = "context" if in_memo else "missing"
    if not in_memo:
        cached = retrieve_memo(symbol)
        if cached and cached.get("memo"):
            in_memo = cached["memo"]
            ctx["final_report"] = in_memo
            ctx["memo_source"] = "pinecone"
            ctx["memo_score"] = cached.get("score")
            memo_source = "pinecone"
    else:
        ctx["memo_source"] = "context"
    log_event(
        log,
        "action_memo_resolved",
        task_id=mcp_msg.task_id,
        source=memo_source,
        chars=len(in_memo),
    )

    handlers = _build_handlers(memo_source)

    user_prompt = f"""
Symbol: {symbol or "<unknown>"}
User intent: {ctx.get("question") or "investment recommendation"}

Investment memo:
{in_memo or "<no memo available — call retrieve_memo if a symbol is set>"}
"""
    result = run_agent_loop(
        agent_name="action_agent",
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tools=tools_for("action_agent"),
        tool_handlers=handlers,
        max_iterations=5,
        response_format={"type": "json_object"},
    )

    decision = _parse_decision(result["answer"])
    captured = handlers["_captured"]
    decision.setdefault("symbol", symbol)
    explicit_side, explicit_qty = _extract_explicit_order(ctx.get("question"))
    if explicit_side:
        decision["action"] = explicit_side
        decision["quantity"] = explicit_qty
        decision["rationale"] = f"user explicitly requested {explicit_side} {explicit_qty} share(s)"
        log_event(
            log,
            "action_explicit_order_override",
            task_id=mcp_msg.task_id,
            side=explicit_side,
            qty=explicit_qty,
        )
    decision.setdefault("memo_source", memo_source)
    # Reconcile from captured tool outputs in case the LLM misreports.
    if captured.get("order_id") and not decision.get("order_id"):
        decision["order_id"] = captured["order_id"]
    if captured.get("broker_status") and not decision.get("broker_status"):
        decision["broker_status"] = captured["broker_status"]
    if captured.get("filled_qty") is not None:
        decision.setdefault("filled_qty", captured["filled_qty"])
    if captured.get("filled_avg_price") is not None:
        decision.setdefault("filled_avg_price", captured["filled_avg_price"])

    # Deterministic broker verification: if we have an order id and the order
    # can still move, poll once more so "executed" is broker-truthful.
    _enforce_order_verification(decision)
    _ensure_execution_if_needed(decision)

    decision["dashboard_url"] = os.getenv(
        "ALPACA_DASHBOARD_URL", "https://app.alpaca.markets/paper/dashboard/overview"
    )

    ctx["action_result"] = decision
    ctx.setdefault("agent_traces", {})["action_agent"] = {
        "iterations": result["iterations"],
        "tool_calls": result["tool_calls"],
        "status": result["status"],
    }
    mcp_msg.metadata["step"] = "action_done"
    log_event(
        log,
        "action_completed",
        task_id=mcp_msg.task_id,
        action=decision.get("action"),
        quantity=decision.get("quantity"),
        executed=decision.get("executed"),
        order_id=decision.get("order_id"),
        broker_status=decision.get("broker_status"),
        memo_source=memo_source,
    )
    return mcp_msg


def _build_handlers(memo_source: str) -> dict:
    """Wrap broker tools so we can capture order metadata for reconciliation."""
    captured: dict[str, Any] = {"memo_source": memo_source}

    def _place(symbol, qty, side):
        result = place_paper_order(symbol=symbol, qty=int(qty), side=side)
        if isinstance(result, dict):
            captured["order_id"] = result.get("order_id")
            captured["broker_status"] = result.get("status") or result.get("broker_status")
            captured["filled_qty"] = result.get("filled_qty")
            captured["filled_avg_price"] = result.get("filled_avg_price")
        return result

    def _wait(order_id, timeout=15.0):
        oid = _normalize_order_id(order_id) or captured.get("order_id")
        result = wait_for_fill(oid, timeout=float(timeout)) if oid else {"status": "error", "reason": "missing_order_id"}
        if isinstance(result, dict):
            captured["broker_status"] = result.get("status") or captured.get("broker_status")
            if result.get("order_id"):
                captured["order_id"] = result["order_id"]
            if result.get("filled_qty") is not None:
                captured["filled_qty"] = result["filled_qty"]
            if result.get("filled_avg_price") is not None:
                captured["filled_avg_price"] = result["filled_avg_price"]
        return result

    def _retrieve(symbol=None, query=None):
        return retrieve_memo(symbol, query)

    base = handlers_for("action_agent")
    base["place_paper_order"] = _place
    base["wait_for_fill"] = _wait
    base["get_order_status"] = lambda order_id: get_order_status(order_id)
    base["retrieve_memo"] = _retrieve
    base["_captured"] = captured  # type: ignore[assignment]
    return base


def _parse_decision(text: str) -> dict[str, Any]:
    import json

    if not text:
        return {"action": "hold", "quantity": 0, "executed": False}
    cleaned = text.strip().strip("`")
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError("not_dict")
    except (json.JSONDecodeError, ValueError):
        return {"action": "hold", "quantity": 0, "executed": False, "rationale": "decision_parse_failed"}

    parsed["action"] = (parsed.get("action") or "hold").lower()
    try:
        parsed["quantity"] = max(0, min(5, int(parsed.get("quantity") or 0)))
    except (TypeError, ValueError):
        parsed["quantity"] = 0
    parsed["order_id"] = _normalize_order_id(parsed.get("order_id"))
    parsed.setdefault("executed", False)
    return parsed


def _normalize_order_id(order_id: Any) -> str | None:
    if order_id is None:
        return None
    text = str(order_id).strip()
    if not text or text.lower() in {"null", "none", "nil"}:
        return None
    return text


def _enforce_order_verification(decision: dict[str, Any]) -> None:
    order_id = _normalize_order_id(decision.get("order_id"))
    if order_id:
        decision["order_id"] = order_id
    status = str(decision.get("broker_status") or decision.get("status") or "").lower()
    action = str(decision.get("action") or "").lower()

    # For a real buy/sell, if order is still active, poll once more.
    maybe_active = {"submitted", "new", "accepted", "accepted_for_bidding", "partially_filled", "pending_new"}
    if order_id and action in {"buy", "sell"} and status in maybe_active:
        latest = wait_for_fill(order_id, timeout=20.0)
        if isinstance(latest, dict):
            decision["broker_status"] = latest.get("status") or decision.get("broker_status")
            decision["filled_qty"] = latest.get("filled_qty", decision.get("filled_qty"))
            decision["filled_avg_price"] = latest.get("filled_avg_price", decision.get("filled_avg_price"))

    final_status = str(decision.get("broker_status") or "").lower()
    decision["executed"] = final_status in {"filled", "partially_filled"}


def _ensure_execution_if_needed(decision: dict[str, Any]) -> None:
    """If action says buy/sell but no order was placed, place + verify now."""
    action = str(decision.get("action") or "").lower()
    qty = int(decision.get("quantity") or 0)
    if action not in {"buy", "sell"} or qty <= 0:
        return
    order_id = _normalize_order_id(decision.get("order_id"))
    if order_id:
        return  # already submitted

    symbol = decision.get("symbol")
    if not symbol:
        return

    submit = place_paper_order(symbol=symbol, qty=qty, side=action)
    decision["broker_status"] = submit.get("status") or submit.get("broker_status")
    decision["order_id"] = _normalize_order_id(submit.get("order_id"))
    decision["filled_qty"] = submit.get("filled_qty")
    decision["filled_avg_price"] = submit.get("filled_avg_price")
    if decision["order_id"]:
        latest = wait_for_fill(decision["order_id"], timeout=20.0)
        decision["broker_status"] = latest.get("status") or decision["broker_status"]
        decision["filled_qty"] = latest.get("filled_qty", decision["filled_qty"])
        decision["filled_avg_price"] = latest.get("filled_avg_price", decision["filled_avg_price"])
    final_status = str(decision.get("broker_status") or "").lower()
    decision["executed"] = final_status in {"filled", "partially_filled"}


def _extract_explicit_order(question: Any) -> tuple[str | None, int]:
    text = str(question or "").lower()
    side: str | None = None
    if any(tok in text for tok in [" buy ", "buy ", "purchase ", " long "]):
        side = "buy"
    elif any(tok in text for tok in [" sell ", "short ", "exit "]):
        side = "sell"
    if not side:
        return None, 0
    qty_match = re.search(r"\b(\d{1,2})\s*(share|shares)\b", text)
    if qty_match:
        qty = max(1, min(5, int(qty_match.group(1))))
    else:
        qty = 1
    return side, qty
