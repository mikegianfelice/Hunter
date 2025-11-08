from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Dict, Any

from .db import get_connection, get_meta, set_meta

_SOL_PRICE_JSON = Path("data/sol_price_cache.json")
_LOCK = threading.Lock()


def _init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sol_price_cache (
                key TEXT PRIMARY KEY,
                price REAL,
                timestamp REAL,
                payload TEXT
            )
            """
        )
    _migrate_from_json()


def _read_json() -> Dict[str, Any]:
    try:
        payload = json.loads(_SOL_PRICE_JSON.read_text(encoding="utf-8") or "{}")
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_json(data: Dict[str, Any]) -> None:
    _SOL_PRICE_JSON.parent.mkdir(parents=True, exist_ok=True)
    _SOL_PRICE_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")
    set_meta("sol_price_cache_mtime", str(_SOL_PRICE_JSON.stat().st_mtime))


def _migrate_from_json() -> None:
    if not _SOL_PRICE_JSON.exists():
        return
    payload = _read_json()
    if not payload:
        return
    with _LOCK:
        with get_connection() as conn:
            row = conn.execute("SELECT COUNT(1) FROM sol_price_cache").fetchone()[0]
            if row > 0:
                return
            price = float(payload.get("price", 0) or 0.0)
            timestamp = float(payload.get("timestamp", time.time()))
            conn.execute(
                "REPLACE INTO sol_price_cache (key, price, timestamp, payload) VALUES ('sol', ?, ?, ?)",
                (price, timestamp, json.dumps(payload)),
            )
            set_meta("sol_price_cache_mtime", str(_SOL_PRICE_JSON.stat().st_mtime), conn)


def _refresh_from_json_if_needed() -> None:
    if not _SOL_PRICE_JSON.exists():
        return
    try:
        json_mtime = _SOL_PRICE_JSON.stat().st_mtime
    except OSError:
        return
    recorded = get_meta("sol_price_cache_mtime")
    if recorded is not None and float(recorded) >= json_mtime:
        return
    payload = _read_json()
    with _LOCK:
        with get_connection() as conn:
            price = float(payload.get("price", 0) or 0.0)
            timestamp = float(payload.get("timestamp", time.time()))
            conn.execute(
                "REPLACE INTO sol_price_cache (key, price, timestamp, payload) VALUES ('sol', ?, ?, ?)",
                (price, timestamp, json.dumps(payload)),
            )
            set_meta("sol_price_cache_mtime", str(json_mtime), conn)


def load_sol_price_cache() -> Dict[str, Any]:
    _init_db()
    _refresh_from_json_if_needed()
    with get_connection() as conn:
        row = conn.execute("SELECT price, timestamp, payload FROM sol_price_cache WHERE key = 'sol'").fetchone()
    if row:
        payload = {
            "price": row["price"],
            "timestamp": row["timestamp"],
        }
        try:
            payload.update(json.loads(row["payload"]) or {})
        except Exception:
            pass
        return payload
    if _SOL_PRICE_JSON.exists():
        return _read_json()
    return {}


def save_sol_price_cache(price: float, timestamp: float | None = None) -> None:
    _init_db()
    ts = timestamp if timestamp is not None else time.time()
    payload = {"price": float(price or 0.0), "timestamp": float(ts)}
    with _LOCK:
        with get_connection() as conn:
            conn.execute(
                "REPLACE INTO sol_price_cache (key, price, timestamp, payload) VALUES ('sol', ?, ?, ?)",
                (payload["price"], payload["timestamp"], json.dumps(payload)),
            )
        _write_json(payload)
