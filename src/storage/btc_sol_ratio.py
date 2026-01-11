from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Dict, Any

from .db import get_connection, get_meta, set_meta

_BTC_SOL_RATIO_JSON = Path("data/btc_sol_ratio_cache.json")
_LOCK = threading.Lock()


def _init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS btc_sol_ratio_cache (
                key TEXT PRIMARY KEY,
                ratio REAL,
                timestamp REAL,
                payload TEXT
            )
            """
        )
    _migrate_from_json()


def _read_json() -> Dict[str, Any]:
    try:
        payload = json.loads(_BTC_SOL_RATIO_JSON.read_text(encoding="utf-8") or "{}")
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_json(data: Dict[str, Any]) -> None:
    _BTC_SOL_RATIO_JSON.parent.mkdir(parents=True, exist_ok=True)
    _BTC_SOL_RATIO_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")
    set_meta("btc_sol_ratio_cache_mtime", str(_BTC_SOL_RATIO_JSON.stat().st_mtime))


def _migrate_from_json() -> None:
    if not _BTC_SOL_RATIO_JSON.exists():
        return
    payload = _read_json()
    if not payload:
        return
    with _LOCK:
        with get_connection() as conn:
            row = conn.execute("SELECT COUNT(1) FROM btc_sol_ratio_cache").fetchone()[0]
            if row > 0:
                return
            ratio = float(payload.get("ratio", 0) or 0.0)
            timestamp = float(payload.get("timestamp", time.time()))
            conn.execute(
                "REPLACE INTO btc_sol_ratio_cache (key, ratio, timestamp, payload) VALUES ('btc_sol', ?, ?, ?)",
                (ratio, timestamp, json.dumps(payload)),
            )
            set_meta("btc_sol_ratio_cache_mtime", str(_BTC_SOL_RATIO_JSON.stat().st_mtime), conn)


def _refresh_from_json_if_needed() -> None:
    if not _BTC_SOL_RATIO_JSON.exists():
        return
    try:
        json_mtime = _BTC_SOL_RATIO_JSON.stat().st_mtime
    except OSError:
        return
    recorded = get_meta("btc_sol_ratio_cache_mtime")
    if recorded is not None and float(recorded) >= json_mtime:
        return
    payload = _read_json()
    with _LOCK:
        with get_connection() as conn:
            ratio = float(payload.get("ratio", 0) or 0.0)
            timestamp = float(payload.get("timestamp", time.time()))
            conn.execute(
                "REPLACE INTO btc_sol_ratio_cache (key, ratio, timestamp, payload) VALUES ('btc_sol', ?, ?, ?)",
                (ratio, timestamp, json.dumps(payload)),
            )
            set_meta("btc_sol_ratio_cache_mtime", str(json_mtime), conn)


def load_btc_sol_ratio_cache() -> Dict[str, Any]:
    """Load BTC/SOL ratio from cache (DB or JSON)"""
    _init_db()
    _refresh_from_json_if_needed()
    with get_connection() as conn:
        row = conn.execute("SELECT ratio, timestamp, payload FROM btc_sol_ratio_cache WHERE key = 'btc_sol'").fetchone()
    if row:
        payload = {
            "ratio": row["ratio"],
            "timestamp": row["timestamp"],
        }
        try:
            payload.update(json.loads(row["payload"]) or {})
        except Exception:
            pass
        return payload
    if _BTC_SOL_RATIO_JSON.exists():
        return _read_json()
    return {}


def save_btc_sol_ratio_cache(ratio: float, timestamp: float | None = None) -> None:
    """Save BTC/SOL ratio to cache (DB and JSON)"""
    _init_db()
    ts = timestamp if timestamp is not None else time.time()
    payload = {"ratio": float(ratio or 0.0), "timestamp": float(ts)}
    with _LOCK:
        with get_connection() as conn:
            conn.execute(
                "REPLACE INTO btc_sol_ratio_cache (key, ratio, timestamp, payload) VALUES ('btc_sol', ?, ?, ?)",
                (payload["ratio"], payload["timestamp"], json.dumps(payload)),
            )
        _write_json(payload)
