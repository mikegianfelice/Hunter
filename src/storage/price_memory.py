from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict

from .db import get_connection, get_meta, set_meta

_PRICE_MEMORY_JSON = Path("data/price_memory.json")
_LOCK = threading.Lock()


def _init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS price_memory (
                address TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                updated_at REAL DEFAULT (strftime('%s','now'))
            )
            """
        )
    _migrate_from_json()


def _read_json() -> Dict[str, Any]:
    try:
        payload = json.loads(_PRICE_MEMORY_JSON.read_text(encoding="utf-8") or "{}")
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_json(data: Dict[str, Any]) -> None:
    _PRICE_MEMORY_JSON.parent.mkdir(parents=True, exist_ok=True)
    _PRICE_MEMORY_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")
    set_meta("price_memory_json_mtime", str(_PRICE_MEMORY_JSON.stat().st_mtime))


def _migrate_from_json() -> None:
    if not _PRICE_MEMORY_JSON.exists():
        return
    payload = _read_json()
    if not payload:
        return
    with _LOCK:
        with get_connection() as conn:
            count = conn.execute("SELECT COUNT(1) FROM price_memory").fetchone()[0]
            if count > 0:
                return
            rows = [
                (
                    str(address).lower(),
                    json.dumps(entry if isinstance(entry, dict) else {}),
                )
                for address, entry in payload.items()
            ]
            if rows:
                conn.executemany(
                    "INSERT INTO price_memory (address, data) VALUES (?, ?)",
                    rows,
                )
            set_meta("price_memory_json_mtime", str(_PRICE_MEMORY_JSON.stat().st_mtime), conn)


def _refresh_from_json_if_needed() -> None:
    if not _PRICE_MEMORY_JSON.exists():
        return
    try:
        json_mtime = _PRICE_MEMORY_JSON.stat().st_mtime
    except OSError:
        return
    recorded = get_meta("price_memory_json_mtime")
    if recorded is not None and float(recorded) >= json_mtime:
        return
    payload = _read_json()
    with _LOCK:
        with get_connection() as conn:
            conn.execute("DELETE FROM price_memory")
            rows = [
                (
                    str(address).lower(),
                    json.dumps(entry if isinstance(entry, dict) else {}),
                )
                for address, entry in payload.items()
            ]
            if rows:
                conn.executemany(
                    "INSERT INTO price_memory (address, data, updated_at) VALUES (?, ?, strftime('%s','now'))",
                    rows,
                )
            set_meta("price_memory_json_mtime", str(json_mtime), conn)


def load_price_memory() -> Dict[str, Dict[str, Any]]:
    _init_db()
    _refresh_from_json_if_needed()
    with get_connection() as conn:
        rows = conn.execute("SELECT address, data FROM price_memory").fetchall()
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        try:
            result[row["address"]] = json.loads(row["data"]) or {}
        except Exception:
            result[row["address"]] = {}
    if not result and _PRICE_MEMORY_JSON.exists():
        return _read_json()
    return result


def save_price_memory(mem: Dict[str, Dict[str, Any]]) -> None:
    _init_db()
    sanitized: Dict[str, Dict[str, Any]] = {}
    for address, entry in (mem or {}).items():
        if not address:
            continue
        payload = dict(entry or {})
        sanitized[address.lower()] = payload
    with _LOCK:
        with get_connection() as conn:
            conn.execute("DELETE FROM price_memory")
            conn.executemany(
                "INSERT INTO price_memory (address, data, updated_at) VALUES (?, ?, strftime('%s','now'))",
                ((addr, json.dumps(data)) for addr, data in sanitized.items()),
            )
        _write_json(sanitized)
