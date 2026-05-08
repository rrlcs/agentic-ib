"""Pinecone-backed long term memory for synthesised investment memos.

The synthesis agent calls :func:`store_memo` after generating a report.
The action agent calls :func:`retrieve_memo` when it doesn't have a report in
its current MCP context (e.g. ``intent=action_only``) and needs the most recent
memo for a given symbol.

If ``PINECONE_API_KEY`` is not set we degrade to a no-op so local development
still works without an external dependency.
"""

from __future__ import annotations

import os
import time
from typing import Any

from openai import OpenAI

from observability.logger import get_logger, log_event

log = get_logger(__name__)

_INDEX = None
_PINECONE_INIT_FAILED = False
_EMBED_CLIENT: OpenAI | None = None

_INDEX_NAME = os.getenv("PINECONE_INDEX", "agentic-memos")
_REGION = os.getenv("PINECONE_REGION", "us-east-1")
_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
_EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
_EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))
_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "memos")


def _get_index():
    """Lazily build a Pinecone index handle. Returns ``None`` if disabled."""
    global _INDEX, _PINECONE_INIT_FAILED
    if _INDEX is not None:
        return _INDEX
    if _PINECONE_INIT_FAILED:
        return None

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        return None

    try:
        from pinecone import Pinecone, ServerlessSpec  # type: ignore
    except ImportError:
        log_event(log, "pinecone_import_failed")
        _PINECONE_INIT_FAILED = True
        return None

    try:
        pc = Pinecone(api_key=api_key)
        existing = {idx["name"] for idx in pc.list_indexes()}
        if _INDEX_NAME not in existing:
            pc.create_index(
                name=_INDEX_NAME,
                dimension=_EMBED_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud=_CLOUD, region=_REGION),
            )
            log_event(log, "pinecone_index_created", index=_INDEX_NAME)
        _INDEX = pc.Index(_INDEX_NAME)
        log_event(log, "pinecone_ready", index=_INDEX_NAME)
        return _INDEX
    except Exception as exc:
        log_event(log, "pinecone_init_failed", error=str(exc))
        _PINECONE_INIT_FAILED = True
        return None


def _get_embed_client() -> OpenAI:
    global _EMBED_CLIENT
    if _EMBED_CLIENT is None:
        _EMBED_CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _EMBED_CLIENT


def _embed(text: str) -> list[float] | None:
    if not text:
        return None
    try:
        client = _get_embed_client()
        resp = client.embeddings.create(model=_EMBED_MODEL, input=text[:8000])
        return list(resp.data[0].embedding)
    except Exception as exc:
        log_event(log, "embedding_failed", error=str(exc))
        return None


def store_memo(
    *,
    task_id: str,
    symbol: str | None,
    company: str | None,
    memo: str,
    intent: str | None = None,
) -> bool:
    """Persist a memo to Pinecone. Returns ``True`` on success."""
    if not memo:
        return False
    idx = _get_index()
    if idx is None:
        return False

    text_to_embed = f"{(company or '').strip()} {(symbol or '').strip()}\n\n{memo}"
    vector = _embed(text_to_embed)
    if vector is None:
        return False

    metadata: dict[str, Any] = {
        "symbol": (symbol or "").upper(),
        "company": company or "",
        "intent": intent or "",
        "memo": memo[:30000],
        "ts": time.time(),
    }
    try:
        idx.upsert(
            vectors=[{"id": task_id, "values": vector, "metadata": metadata}],
            namespace=_NAMESPACE,
        )
        log_event(log, "memo_stored", task_id=task_id, symbol=metadata["symbol"], chars=len(memo))
        return True
    except Exception as exc:
        log_event(log, "memo_store_failed", task_id=task_id, error=str(exc))
        return False


def vector_search(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """RAG search across stored memos. Returns up to ``top_k`` matches.

    Each match has: ``score``, ``symbol``, ``company``, ``intent``,
    ``memo_excerpt`` (first 1500 chars of the memo), and ``ts``.
    Empty list if Pinecone is not configured.
    """
    idx = _get_index()
    if idx is None or not (query or "").strip():
        return []

    vector = _embed(query)
    if vector is None:
        return []

    try:
        res = idx.query(
            vector=vector,
            top_k=max(1, min(10, int(top_k or 3))),
            include_metadata=True,
            namespace=_NAMESPACE,
        )
    except Exception as exc:
        log_event(log, "vector_search_failed", error=str(exc))
        return []

    out: list[dict[str, Any]] = []
    for match in _extract_matches(res):
        meta = _attr(match, "metadata") or {}
        if not isinstance(meta, dict):
            continue
        out.append(
            {
                "id": _attr(match, "id"),
                "score": _attr(match, "score"),
                "symbol": meta.get("symbol"),
                "company": meta.get("company"),
                "intent": meta.get("intent"),
                "memo_excerpt": (meta.get("memo") or "")[:1500],
                "ts": meta.get("ts"),
            }
        )
    log_event(log, "vector_search_completed", query=query[:100], n_results=len(out))
    return out


def retrieve_memo(symbol: str | None, query: str | None = None) -> dict[str, Any] | None:
    """Fetch the most relevant memo for a symbol (or generic query)."""
    idx = _get_index()
    if idx is None:
        return None

    seed = f"investment memo for {symbol}" if symbol else (query or "investment memo")
    vector = _embed(seed)
    if vector is None:
        return None

    try:
        kwargs: dict[str, Any] = {
            "vector": vector,
            "top_k": 1,
            "include_metadata": True,
            "namespace": _NAMESPACE,
        }
        if symbol:
            kwargs["filter"] = {"symbol": {"$eq": symbol.upper()}}
        res = idx.query(**kwargs)
    except Exception as exc:
        log_event(log, "memo_retrieve_failed", error=str(exc))
        return None

    matches = _extract_matches(res)
    if not matches:
        return None
    match = matches[0]
    metadata = _attr(match, "metadata") or {}
    return {
        "id": _attr(match, "id"),
        "score": _attr(match, "score"),
        **(metadata if isinstance(metadata, dict) else {}),
    }


def _extract_matches(res: Any) -> list[Any]:
    if isinstance(res, dict):
        return list(res.get("matches") or [])
    return list(getattr(res, "matches", []) or [])


def _attr(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)
