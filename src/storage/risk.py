from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict

from .db import get_connection, get_meta, set_meta

RISK_STATE_JSON_PATH = Path("data/risk_state.json")
BALANCE_CACHE_JSON_PATH = Path("data/balance_cache.json")
_LOCK = threading.Lock()


def _init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS risk_state (
                key TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                updated_at REAL DEFAULT (strftime('%s','now'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS balance_cache (
                key TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                updated_at REAL DEFAULT (strftime('%s','now'))
            )
            """
        )
    _migrate_from_json()


def _migrate_from_json() -> None:
    with _LOCK:
        with get_connection() as conn:
            risk_row = conn.execute("SELECT COUNT(1) FROM risk_state").fetchone()[0]
            balance_row = conn.execute("SELECT COUNT(1) FROM balance_cache").fetchone()[0]

        if risk_row == 0 and RISK_STATE_JSON_PATH.exists():
            try:
                payload = json.loads(RISK_STATE_JSON_PATH.read_text(encoding="utf-8") or "{}")
                with get_connection() as conn:
                    conn.execute(
                        "REPLACE INTO risk_state (key, data) VALUES ('default', ?)",
                        (json.dumps(payload),),
                    )
                    set_meta("risk_state_json_mtime", str(RISK_STATE_JSON_PATH.stat().st_mtime), conn)
            except Exception:
                pass

        if balance_row == 0 and BALANCE_CACHE_JSON_PATH.exists():
            try:
                payload = json.loads(BALANCE_CACHE_JSON_PATH.read_text(encoding="utf-8") or "{}")
                with get_connection() as conn:
                    conn.execute(
                        "REPLACE INTO balance_cache (key, data) VALUES ('cache', ?)",
                        (json.dumps(payload),),
                    )
                    set_meta("balance_cache_json_mtime", str(BALANCE_CACHE_JSON_PATH.stat().st_mtime), conn)
            except Exception:
                pass


def _write_risk_json(data: Dict[str, Any]) -> None:
    RISK_STATE_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    RISK_STATE_JSON_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    set_meta("risk_state_json_mtime", str(RISK_STATE_JSON_PATH.stat().st_mtime))


def _write_balance_json(data: Dict[str, Any]) -> None:
    BALANCE_CACHE_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    BALANCE_CACHE_JSON_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    set_meta("balance_cache_json_mtime", str(BALANCE_CACHE_JSON_PATH.stat().st_mtime))


def _refresh_from_json_if_needed() -> None:
    with _LOCK:
        need_risk_refresh = False
        need_balance_refresh = False

        if RISK_STATE_JSON_PATH.exists():
            try:
                json_mtime = RISK_STATE_JSON_PATH.stat().st_mtime
                recorded = get_meta("risk_state_json_mtime")
                if recorded is None or float(recorded) < json_mtime:
                    need_risk_refresh = True
            except Exception:
                pass

        if BALANCE_CACHE_JSON_PATH.exists():
            try:
                json_mtime = BALANCE_CACHE_JSON_PATH.stat().st_mtime
                recorded = get_meta("balance_cache_json_mtime")
                if recorded is None or float(recorded) < json_mtime:
                    need_balance_refresh = True
            except Exception:
                pass

        if need_risk_refresh:
            try:
                payload = json.loads(RISK_STATE_JSON_PATH.read_text(encoding="utf-8") or "{}")
            except Exception:
                payload = {}
            with get_connection() as conn:
                conn.execute("DELETE FROM risk_state")
                conn.execute(
                    "REPLACE INTO risk_state (key, data) VALUES ('default', ?)",
                    (json.dumps(payload),),
                )
                set_meta("risk_state_json_mtime", str(RISK_STATE_JSON_PATH.stat().st_mtime), conn)

        if need_balance_refresh:
            try:
                payload = json.loads(BALANCE_CACHE_JSON_PATH.read_text(encoding="utf-8") or "{}")
            except Exception:
                payload = {}
            with get_connection() as conn:
                conn.execute("DELETE FROM balance_cache")
                conn.execute(
                    "REPLACE INTO balance_cache (key, data) VALUES ('cache', ?)",
                    (json.dumps(payload),),
                )
                set_meta("balance_cache_json_mtime", str(BALANCE_CACHE_JSON_PATH.stat().st_mtime), conn)


def load_risk_state() -> Dict[str, Any]:
    _init_db()
    _refresh_from_json_if_needed()
    with get_connection() as conn:
        row = conn.execute("SELECT data FROM risk_state WHERE key = 'default'").fetchone()
    if row and row[0]:
        try:
            return json.loads(row[0]) or {}
        except Exception:
            return {}
    if RISK_STATE_JSON_PATH.exists():
        try:
            return json.loads(RISK_STATE_JSON_PATH.read_text(encoding="utf-8") or "{}") or {}
        except Exception:
            return {}
    return {}


def save_risk_state(state: Dict[str, Any]) -> None:
    _init_db()
    payload = state or {}
    with _LOCK:
        with get_connection() as conn:
            conn.execute(
                "REPLACE INTO risk_state (key, data, updated_at) VALUES ('default', ?, strftime('%s','now'))",
                (json.dumps(payload),),
            )
        _write_risk_json(payload)


def load_balance_cache() -> Dict[str, Any]:
    _init_db()
    _refresh_from_json_if_needed()
    with get_connection() as conn:
        row = conn.execute("SELECT data FROM balance_cache WHERE key = 'cache'").fetchone()
    if row and row[0]:
        try:
            return json.loads(row[0]) or {}
        except Exception:
            return {}
    if BALANCE_CACHE_JSON_PATH.exists():
        try:
            return json.loads(BALANCE_CACHE_JSON_PATH.read_text(encoding="utf-8") or "{}") or {}
        except Exception:
            return {}
    return {}


def save_balance_cache(cache: Dict[str, Any]) -> None:
    _init_db()
    payload = cache or {}
    with _LOCK:
        with get_connection() as conn:
            conn.execute(
                "REPLACE INTO balance_cache (key, data, updated_at) VALUES ('cache', ?, strftime('%s','now'))",
                (json.dumps(payload),),
            )
        _write_balance_json(payload)
