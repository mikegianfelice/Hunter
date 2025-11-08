from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .db import get_connection, get_meta, set_meta

_DELISTED_JSON_PATH = Path("data/delisted_tokens.json")
_LOCK = threading.Lock()
_DEFAULT_STATE = {
    "delisted_tokens": [],
    "failure_counts": {},
}


def _init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS delisted_state (
                key TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                updated_at REAL DEFAULT (strftime('%s','now'))
            )
            """
        )
    _migrate_from_json()


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8") or "{}")
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _sanitize_state(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(state, dict):
        state = {}
    sanitized: Dict[str, Any] = dict(state)
    tokens_raw = sanitized.get("delisted_tokens", []) or []
    if not isinstance(tokens_raw, list):
        tokens_raw = []
    sanitized_tokens: List[str] = []
    seen = set()
    for token in tokens_raw:
        if not token:
            continue
        norm = str(token).lower()
        if norm in seen:
            continue
        sanitized_tokens.append(norm)
        seen.add(norm)
    sanitized["delisted_tokens"] = sanitized_tokens

    failure_counts = sanitized.get("failure_counts", {}) or {}
    if not isinstance(failure_counts, dict):
        failure_counts = {}
    sanitized_counts: Dict[str, int] = {}
    for token, count in failure_counts.items():
        if not token:
            continue
        try:
            sanitized_counts[str(token).lower()] = int(count)
        except (TypeError, ValueError):
            continue
    sanitized["failure_counts"] = sanitized_counts

    last_added = sanitized.get("last_added")
    if isinstance(last_added, dict):
        last_added.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
        sanitized["last_added"] = last_added

    sanitized.setdefault("cleaned_at", None)
    sanitized.setdefault("removed_count", 0)
    sanitized.setdefault("remaining_count", len(sanitized_tokens))
    sanitized.setdefault("reactivated_tokens", [])

    return sanitized


def _migrate_from_json() -> None:
    if not _DELISTED_JSON_PATH.exists():
        return
    payload = _sanitize_state(_read_json(_DELISTED_JSON_PATH))
    if not payload:
        return
    with _LOCK:
        with get_connection() as conn:
            count = conn.execute("SELECT COUNT(1) FROM delisted_state").fetchone()[0]
            if count > 0:
                return
            conn.execute(
                "REPLACE INTO delisted_state (key, data) VALUES ('state', ?)",
                (json.dumps(payload),),
            )
            set_meta("delisted_json_mtime", str(_DELISTED_JSON_PATH.stat().st_mtime), conn)


def _write_json_snapshot(state: Dict[str, Any]) -> None:
    _DELISTED_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    _DELISTED_JSON_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    set_meta("delisted_json_mtime", str(_DELISTED_JSON_PATH.stat().st_mtime))


def _refresh_from_json_if_needed() -> None:
    if not _DELISTED_JSON_PATH.exists():
        return
    try:
        json_mtime = _DELISTED_JSON_PATH.stat().st_mtime
    except OSError:
        return
    recorded = get_meta("delisted_json_mtime")
    if recorded is not None and float(recorded) >= json_mtime:
        return
    payload = _sanitize_state(_read_json(_DELISTED_JSON_PATH))
    with _LOCK:
        with get_connection() as conn:
            conn.execute(
                "REPLACE INTO delisted_state (key, data, updated_at) VALUES ('state', ?, strftime('%s','now'))",
                (json.dumps(payload),),
            )
            set_meta("delisted_json_mtime", str(json_mtime), conn)


def load_delisted_state() -> Dict[str, Any]:
    _init_db()
    _refresh_from_json_if_needed()
    with get_connection() as conn:
        row = conn.execute("SELECT data FROM delisted_state WHERE key = 'state'").fetchone()
    if row and row[0]:
        try:
            return _sanitize_state(json.loads(row[0]))
        except Exception:
            pass
    if _DELISTED_JSON_PATH.exists():
        return _sanitize_state(_read_json(_DELISTED_JSON_PATH))
    return dict(_DEFAULT_STATE)


def save_delisted_state(state: Dict[str, Any]) -> None:
    _init_db()
    sanitized = _sanitize_state(state)
    with _LOCK:
        with get_connection() as conn:
            conn.execute(
                "REPLACE INTO delisted_state (key, data, updated_at) VALUES ('state', ?, strftime('%s','now'))",
                (json.dumps(sanitized),),
            )
        _write_json_snapshot(sanitized)


def add_delisted_token(token: str, *, symbol: Optional[str] = None, reason: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
    token_norm = (token or "").lower()
    if not token_norm:
        return False
    state = load_delisted_state()
    tokens: List[str] = state.get("delisted_tokens", [])
    if token_norm in tokens:
        return False
    tokens.append(token_norm)
    state["delisted_tokens"] = tokens
    last_added = {
        "address": token,
        "symbol": symbol or "UNKNOWN",
        "reason": reason or "",
        "metadata": metadata or {},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    state["last_added"] = last_added
    state["remaining_count"] = len(tokens)
    save_delisted_state(state)
    return True
