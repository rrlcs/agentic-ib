"""Alpaca paper-trading wrapper with order verification.

Falls back to a structured dry-run / skipped response if Alpaca credentials are
missing so the pipeline always returns a usable result.

Public surface:
- ``place_paper_order(...)``  : submits a market order, returns initial state.
- ``get_order_status(order_id)``: fetches the current state of an order.
- ``wait_for_fill(order_id, timeout)``: poll until fill / final state / timeout.
"""

from __future__ import annotations

import os
import time
from typing import Any

from observability.logger import get_logger, log_event

log = get_logger(__name__)


# -- public helpers ----------------------------------------------------------


def place_paper_order(*, symbol: str, qty: int, side: str, dry_run: bool = False) -> dict[str, Any]:
    side = (side or "").lower()
    if side not in {"buy", "sell"}:
        return {"status": "skipped", "reason": "invalid_side", "side": side}

    if dry_run:
        return {
            "status": "dry_run",
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "dashboard_url": _dashboard_url(),
        }

    api_key, secret_key = _credentials()
    if not api_key or not secret_key:
        log_event(log, "paper_trade_no_credentials", symbol=symbol)
        return {
            "status": "skipped",
            "reason": "no_alpaca_credentials",
            "dashboard_url": _dashboard_url(),
        }

    client = _trading_client(api_key, secret_key)
    if isinstance(client, dict):
        return client  # error response

    try:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        submitted = client.submit_order(request)
    except Exception as exc:
        log_event(log, "paper_trade_failed", symbol=symbol, error=str(exc))
        return {"status": "error", "reason": str(exc)}

    snapshot = _order_snapshot(submitted)
    log_event(
        log,
        "paper_trade_submitted",
        symbol=symbol,
        qty=qty,
        side=side,
        order_id=snapshot.get("order_id"),
        status=snapshot.get("status"),
    )
    snapshot.update(
        {
            "status": "submitted",
            "broker_status": snapshot.get("status"),
            "dashboard_url": _dashboard_url(),
        }
    )
    return snapshot


def get_order_status(order_id: str) -> dict[str, Any]:
    """Look up the current state of a submitted paper order."""
    if not order_id:
        return {"status": "error", "reason": "missing_order_id"}

    api_key, secret_key = _credentials()
    if not api_key or not secret_key:
        return {"status": "skipped", "reason": "no_alpaca_credentials"}

    client = _trading_client(api_key, secret_key)
    if isinstance(client, dict):
        return client

    try:
        order = client.get_order_by_id(order_id)
    except Exception as exc:
        log_event(log, "order_status_failed", order_id=order_id, error=str(exc))
        return {"status": "error", "reason": str(exc), "order_id": order_id}

    snapshot = _order_snapshot(order)
    log_event(
        log,
        "order_status_fetched",
        order_id=order_id,
        broker_status=snapshot.get("status"),
        filled_qty=snapshot.get("filled_qty"),
    )
    return snapshot


def wait_for_fill(order_id: str, timeout: float = 20.0, poll_interval: float = 1.0) -> dict[str, Any]:
    """Poll the order until it reaches a terminal state or ``timeout`` elapses.

    Terminal states are: filled, partially_filled, canceled, expired, rejected,
    done_for_day.
    """
    deadline = time.monotonic() + max(0.0, float(timeout))
    last: dict[str, Any] = {}
    terminal = {
        "filled",
        "canceled",
        "expired",
        "rejected",
        "done_for_day",
        "stopped",
        "suspended",
    }
    while True:
        last = get_order_status(order_id)
        broker_status = (last.get("status") or "").lower()
        if broker_status in terminal:
            last["resolved"] = True
            return last
        if time.monotonic() >= deadline:
            last["resolved"] = False
            last["reason"] = "timeout"
            return last
        time.sleep(max(0.2, float(poll_interval)))


# -- private helpers ---------------------------------------------------------


def _credentials() -> tuple[str | None, str | None]:
    return os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")


def _dashboard_url() -> str:
    return os.getenv("ALPACA_DASHBOARD_URL", "https://app.alpaca.markets/paper/dashboard/overview")


def _trading_client(api_key: str, secret_key: str):
    try:
        from alpaca.trading.client import TradingClient  # type: ignore

        return TradingClient(api_key, secret_key, paper=True)
    except ImportError as exc:
        return {"status": "error", "reason": f"alpaca_sdk_missing: {exc}"}
    except Exception as exc:  # pragma: no cover
        return {"status": "error", "reason": str(exc)}


def _order_snapshot(order: Any) -> dict[str, Any]:
    """Normalize an alpaca-py order object to a JSON-friendly dict."""

    def get(name: str, default: Any = None) -> Any:
        if isinstance(order, dict):
            return order.get(name, default)
        return getattr(order, name, default)

    raw_status = get("status")
    status_value = (
        raw_status.value if hasattr(raw_status, "value") else str(raw_status or "")
    ).lower()

    return {
        "order_id": str(get("id") or ""),
        "client_order_id": str(get("client_order_id") or ""),
        "symbol": get("symbol"),
        "qty": _to_float(get("qty")),
        "side": str(get("side") or "").lower().replace("orderside.", ""),
        "status": status_value,
        "filled_qty": _to_float(get("filled_qty")) or 0.0,
        "filled_avg_price": _to_float(get("filled_avg_price")),
        "submitted_at": _to_str(get("submitted_at")),
        "filled_at": _to_str(get("filled_at")),
    }


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_str(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)
