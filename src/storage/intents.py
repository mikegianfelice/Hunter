from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .db import get_connection, get_meta, set_meta

_INTENTS_JSON_PATH = Path("data/trade_intents.json")
_LOCK = threading.Lock()


def _init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_intents (
                intent_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at REAL,
                updated_at REAL
            )
            """
        )
    _migrate_from_json()


def _migrate_from_json() -> None:
    if not _INTENTS_JSON_PATH.exists():
        return
    try:
        payload = json.loads(_INTENTS_JSON_PATH.read_text(encoding="utf-8") or "{}")
        if not isinstance(payload, dict):
            return
    except Exception:
        return

    with get_connection() as conn:
        count = conn.execute("SELECT COUNT(1) FROM trade_intents").fetchone()[0]
        if count > 0:
            return
        for intent_id, data in payload.items():
            try:
                created_at = float(data.get("created_at", time.time()))
                updated_at = float(data.get("updated_at", created_at))
                conn.execute(
                    "REPLACE INTO trade_intents (intent_id, data, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (intent_id, json.dumps(data), created_at, updated_at),
                )
            except Exception:
                continue
        set_meta("trade_intents_json_mtime", str(_INTENTS_JSON_PATH.stat().st_mtime), conn)


def _fetch_all(conn) -> Dict[str, Dict[str, Any]]:
    rows = conn.execute("SELECT intent_id, data FROM trade_intents").fetchall()
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        try:
            result[row["intent_id"]] = json.loads(row["data"]) or {}
        except Exception:
            continue
    return result


def _write_json_snapshot(data: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
    if data is None:
        with get_connection() as conn:
            data = _fetch_all(conn)
    _INTENTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    _INTENTS_JSON_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    set_meta("trade_intents_json_mtime", str(_INTENTS_JSON_PATH.stat().st_mtime))


def _refresh_from_json_if_needed() -> None:
    if not _INTENTS_JSON_PATH.exists():
        return
    try:
        json_mtime = _INTENTS_JSON_PATH.stat().st_mtime
    except OSError:
        return
    recorded = get_meta("trade_intents_json_mtime")
    if recorded is not None and float(recorded) >= json_mtime:
        return
    try:
        payload = json.loads(_INTENTS_JSON_PATH.read_text(encoding="utf-8") or "{}")
        if not isinstance(payload, dict):
            return
    except Exception:
        return
    with _LOCK:
        with get_connection() as conn:
            conn.execute("DELETE FROM trade_intents")
            for intent_id, data in payload.items():
                try:
                    created_at = float(data.get("created_at", time.time()))
                    updated_at = float(data.get("updated_at", created_at))
                    conn.execute(
                        "REPLACE INTO trade_intents (intent_id, data, created_at, updated_at) VALUES (?, ?, ?, ?)",
                        (intent_id, json.dumps(data), created_at, updated_at),
                    )
                except Exception:
                    continue
            set_meta("trade_intents_json_mtime", str(json_mtime), conn)


def load_all_trade_intents() -> Dict[str, Dict[str, Any]]:
    _init_db()
    _refresh_from_json_if_needed()
    with get_connection() as conn:
        return _fetch_all(conn)


def get_trade_intent(intent_id: str) -> Optional[Dict[str, Any]]:
    _init_db()
    _refresh_from_json_if_needed()
    with get_connection() as conn:
        row = conn.execute("SELECT data FROM trade_intents WHERE intent_id = ?", (intent_id,)).fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0]) or {}
    except Exception:
        return None


def upsert_trade_intent(intent_id: str, data: Dict[str, Any]) -> None:
    _init_db()
    payload = data or {}
    created_at = float(payload.get("created_at", time.time()))
    updated_at = float(payload.get("updated_at", created_at))
    with _LOCK:
        with get_connection() as conn:
            conn.execute(
                "REPLACE INTO trade_intents (intent_id, data, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (intent_id, json.dumps(payload), created_at, updated_at),
            )
            snapshot = _fetch_all(conn)
        _write_json_snapshot(snapshot)


def delete_trade_intent(intent_id: str) -> None:
    _init_db()
    with _LOCK:
        with get_connection() as conn:
            conn.execute("DELETE FROM trade_intents WHERE intent_id = ?", (intent_id,))
            snapshot = _fetch_all(conn)
        _write_json_snapshot(snapshot)


def prune_trade_intents(older_than_ts: float) -> None:
    _init_db()
    with _LOCK:
        with get_connection() as conn:
            conn.execute(
                "DELETE FROM trade_intents WHERE COALESCE(updated_at, created_at, ?) < ?",
                (time.time(), older_than_ts),
            )
            snapshot = _fetch_all(conn)
        _write_json_snapshot(snapshot)
