from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Set

from .db import get_connection, get_meta, set_meta

_BLACKLIST_JSON = Path("data/blacklist.json")
_FAILURES_JSON = Path("data/blacklist_failures.json")
_REASONS_JSON = Path("data/blacklist_reasons.json")
_LOCK = threading.Lock()


def _init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS blacklist_entries (
                address TEXT PRIMARY KEY,
                created_at REAL DEFAULT (strftime('%s','now'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS blacklist_failures (
                address TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                updated_at REAL DEFAULT (strftime('%s','now'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS blacklist_reasons (
                address TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                updated_at REAL DEFAULT (strftime('%s','now'))
            )
            """
        )
    _migrate_from_json()


def _read_json(path: Path, default: Any) -> Any:
    try:
        payload = json.loads(path.read_text(encoding="utf-8") or "")
        return payload if payload is not None else default
    except Exception:
        return default


def _write_json(path: Path, payload: Any, meta_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    set_meta(meta_key, str(path.stat().st_mtime))


def _migrate_from_json() -> None:
    with _LOCK:
        with get_connection() as conn:
            count = conn.execute("SELECT COUNT(1) FROM blacklist_entries").fetchone()[0]
            if count == 0 and _BLACKLIST_JSON.exists():
                entries = _read_json(_BLACKLIST_JSON, [])
                records = [(str(addr).lower(),) for addr in entries if addr]
                if records:
                    conn.executemany("INSERT INTO blacklist_entries (address) VALUES (?)", records)
                set_meta("blacklist_json_mtime", str(_BLACKLIST_JSON.stat().st_mtime), conn)

            count_failures = conn.execute("SELECT COUNT(1) FROM blacklist_failures").fetchone()[0]
            if count_failures == 0 and _FAILURES_JSON.exists():
                failures = _read_json(_FAILURES_JSON, {})
                rows = [
                    (
                        str(addr).lower(),
                        json.dumps(value if isinstance(value, dict) else {}),
                    )
                    for addr, value in failures.items()
                ]
                if rows:
                    conn.executemany(
                        "INSERT INTO blacklist_failures (address, data) VALUES (?, ?)",
                        rows,
                    )
                set_meta("blacklist_failures_mtime", str(_FAILURES_JSON.stat().st_mtime), conn)

            count_reasons = conn.execute("SELECT COUNT(1) FROM blacklist_reasons").fetchone()[0]
            if count_reasons == 0 and _REASONS_JSON.exists():
                reasons = _read_json(_REASONS_JSON, {})
                rows = [
                    (
                        str(addr).lower(),
                        json.dumps(value if isinstance(value, dict) else {}),
                    )
                    for addr, value in reasons.items()
                ]
                if rows:
                    conn.executemany(
                        "INSERT INTO blacklist_reasons (address, data) VALUES (?, ?)",
                        rows,
                    )
                set_meta("blacklist_reasons_mtime", str(_REASONS_JSON.stat().st_mtime), conn)


def _refresh_from_json_if_needed(path: Path, meta_key: str, table: str, serializer) -> None:
    if not path.exists():
        return
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return
    recorded = get_meta(meta_key)
    if recorded is not None and float(recorded) >= mtime:
        return
    payload = serializer(_read_json(path, serializer()))
    with _LOCK:
        with get_connection() as conn:
            conn.execute(f"DELETE FROM {table}")
            if isinstance(payload, list):
                conn.executemany(
                    f"INSERT INTO {table} (address) VALUES (?)",
                    ((addr,) for addr in payload),
                )
            elif isinstance(payload, dict):
                conn.executemany(
                    f"INSERT INTO {table} (address, data) VALUES (?, ?)",
                    ((addr, json.dumps(data)) for addr, data in payload.items()),
                )
            set_meta(meta_key, str(mtime), conn)


def load_blacklist() -> Set[str]:
    _init_db()
    _refresh_from_json_if_needed(_BLACKLIST_JSON, "blacklist_json_mtime", "blacklist_entries", list)
    with get_connection() as conn:
        rows = conn.execute("SELECT address FROM blacklist_entries").fetchall()
    return {row["address"] for row in rows}


def save_blacklist(entries: Set[str]) -> None:
    _init_db()
    sanitized = sorted({(entry or "").lower() for entry in entries if entry})
    with _LOCK:
        with get_connection() as conn:
            conn.execute("DELETE FROM blacklist_entries")
            conn.executemany(
                "INSERT INTO blacklist_entries (address, created_at) VALUES (?, strftime('%s','now'))",
                ((address,) for address in sanitized),
            )
        _write_json(_BLACKLIST_JSON, sanitized, "blacklist_json_mtime")


def load_failures() -> Dict[str, Dict[str, Any]]:
    _init_db()
    _refresh_from_json_if_needed(_FAILURES_JSON, "blacklist_failures_mtime", "blacklist_failures", dict)
    with get_connection() as conn:
        rows = conn.execute("SELECT address, data FROM blacklist_failures").fetchall()
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        try:
            result[row["address"]] = json.loads(row["data"]) or {}
        except Exception:
            result[row["address"]] = {}
    return result


def save_failures(failures: Dict[str, Dict[str, Any]]) -> None:
    _init_db()
    sanitized: Dict[str, Dict[str, Any]] = {}
    for address, payload in (failures or {}).items():
        if not address:
            continue
        record = dict(payload or {})
        record.setdefault("count", 0)
        record.setdefault("last_failure", 0)
        record.setdefault("failure_types", [])
        sanitized[address.lower()] = record
    with _LOCK:
        with get_connection() as conn:
            conn.execute("DELETE FROM blacklist_failures")
            conn.executemany(
                "INSERT INTO blacklist_failures (address, data, updated_at) VALUES (?, ?, strftime('%s','now'))",
                ((addr, json.dumps(data)) for addr, data in sanitized.items()),
            )
        _write_json(_FAILURES_JSON, sanitized, "blacklist_failures_mtime")


def load_reasons() -> Dict[str, Dict[str, Any]]:
    _init_db()
    _refresh_from_json_if_needed(_REASONS_JSON, "blacklist_reasons_mtime", "blacklist_reasons", dict)
    with get_connection() as conn:
        rows = conn.execute("SELECT address, data FROM blacklist_reasons").fetchall()
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        try:
            result[row["address"]] = json.loads(row["data"]) or {}
        except Exception:
            result[row["address"]] = {}
    return result


def save_reasons(reasons: Dict[str, Dict[str, Any]]) -> None:
    _init_db()
    sanitized: Dict[str, Dict[str, Any]] = {}
    for address, payload in (reasons or {}).items():
        if not address:
            continue
        record = dict(payload or {})
        record.setdefault("timestamp", time.time())
        sanitized[address.lower()] = record
    with _LOCK:
        with get_connection() as conn:
            conn.execute("DELETE FROM blacklist_reasons")
            conn.executemany(
                "INSERT INTO blacklist_reasons (address, data, updated_at) VALUES (?, ?, strftime('%s','now'))",
                ((addr, json.dumps(data)) for addr, data in sanitized.items()),
            )
        _write_json(_REASONS_JSON, sanitized, "blacklist_reasons_mtime")
