"""Web search tool for agents.

Uses DuckDuckGo (no API key needed). Returns a list of
``{title, url, snippet}`` dictionaries the LLM can cite. If anything fails
(network, rate-limit), we degrade to an empty list rather than crashing the
agent loop.
"""

from __future__ import annotations

from typing import Any

from observability.logger import get_logger, log_event

log = get_logger(__name__)


def web_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Run a DuckDuckGo text search."""
    if not query or not query.strip():
        return []
    max_results = max(1, min(10, int(max_results or 5)))
    log_event(log, "web_search_started", query=query[:100], max_results=max_results)
    DDGS = _load_ddgs()
    if DDGS is None:
        log_event(log, "web_search_unavailable", reason="ddgs not installed")
        return []

    raw: list[dict[str, Any]] = []
    backends = ("duckduckgo", "bing", "auto")
    for backend in backends:
        try:
            with DDGS() as ddgs:
                kwargs: dict[str, Any] = {"max_results": max_results}
                if backend != "auto":
                    kwargs["backend"] = backend
                raw = list(ddgs.text(query, **kwargs))
            if raw:
                break
        except TypeError:
            try:
                with DDGS() as ddgs:
                    raw = list(ddgs.text(query, max_results=max_results))
                if raw:
                    break
            except Exception as exc:  # pragma: no cover - transient
                log_event(log, "web_search_failed", backend=backend, error=str(exc)[:160])
        except Exception as exc:  # pragma: no cover - transient
            log_event(log, "web_search_failed", backend=backend, error=str(exc)[:160])

    results: list[dict[str, Any]] = []
    for r in raw:
        results.append(
            {
                "title": (r.get("title") or "").strip(),
                "url": (r.get("href") or r.get("link") or "").strip(),
                "snippet": (r.get("body") or r.get("snippet") or "").strip(),
            }
        )
    log_event(log, "web_search_completed", query=query[:100], n_results=len(results))
    return results


def _load_ddgs():
    """Import DDGS from whichever package the user has installed.

    The DuckDuckGo client was renamed from ``duckduckgo_search`` to ``ddgs``.
    We try the new package first to silence the deprecation warning.
    """
    try:
        from ddgs import DDGS  # type: ignore

        return DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # type: ignore

            return DDGS
        except ImportError:
            return None
