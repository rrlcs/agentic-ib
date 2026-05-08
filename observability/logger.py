"""Shared logging helpers for the platform."""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logging.getLogger("kafka").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


def log_event(log: logging.Logger, message: str, **fields: object) -> None:
    """Emit key-value structured fields in message text and forward to tracer."""
    if fields:
        rendered = " ".join(f"{key}={value}" for key, value in fields.items())
        log.info("%s %s", message, rendered)
    else:
        log.info(message)

    task_id = fields.get("task_id") if fields else None
    if task_id:
        try:
            from observability.tracer import publish_trace

            payload = {key: value for key, value in fields.items() if key != "task_id"}
            payload["component"] = log.name
            publish_trace(str(task_id), message, **payload)
        except Exception:  # pragma: no cover - tracing is best-effort
            pass
