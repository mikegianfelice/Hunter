from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict

from .db import get_connection, get_meta, set_meta

_COOLDOWN_JSON_PATH = Path("data/cooldown.json")
_LEGACY_JSON_PATH = Path("cooldown_log.json")
_LOCK = threading.Lock()


def _init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cooldown_log (
                token TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                updated_at REAL DEFAULT (strftime('%s','now'))
            )
            """
        )
    _migrate_from_json()


def _load_json_file(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8") or "{}")
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _migrate_from_json() -> None:
    if _COOLDOWN_JSON_PATH.exists():
        source_path = _COOLDOWN_JSON_PATH
    elif _LEGACY_JSON_PATH.exists():
        source_path = _LEGACY_JSON_PATH
    else:
        return

    payload = _load_json_file(source_path)
    if not payload:
        return

    with _LOCK:
        with get_connection() as conn:
            count = conn.execute("SELECT COUNT(1) FROM cooldown_log").fetchone()[0]
            if count > 0:
                return
            data_to_write = []
            for token, record in payload.items():
                if not isinstance(record, dict):
                    record = {"last_failure": float(record), "failure_count": 1}
                record.setdefault("last_failure", time.time())
                record.setdefault("failure_count", 1)
                data_to_write.append((token.lower(), json.dumps(record)))
            if data_to_write:
                conn.executemany(
                    "INSERT INTO cooldown_log (token, data) VALUES (?, ?)",
                    data_to_write,
                )
            mtime = str(source_path.stat().st_mtime)
            set_meta("cooldown_json_mtime", mtime, conn)

    if source_path is _LEGACY_JSON_PATH and not _COOLDOWN_JSON_PATH.exists():
        _COOLDOWN_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        _COOLDOWN_JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_json_snapshot(data: Dict[str, Any]) -> None:
    _COOLDOWN_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    _COOLDOWN_JSON_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    set_meta("cooldown_json_mtime", str(_COOLDOWN_JSON_PATH.stat().st_mtime))


def _refresh_from_json_if_needed() -> None:
    if not _COOLDOWN_JSON_PATH.exists():
        return
    try:
        json_mtime = _COOLDOWN_JSON_PATH.stat().st_mtime
    except OSError:
        return
    recorded = get_meta("cooldown_json_mtime")
    if recorded is not None and float(recorded) >= json_mtime:
        return
    payload = _load_json_file(_COOLDOWN_JSON_PATH)
    with _LOCK:
        with get_connection() as conn:
            conn.execute("DELETE FROM cooldown_log")
            data_to_write = [
                (token.lower(), json.dumps(record if isinstance(record, dict) else {"last_failure": float(record), "failure_count": 1}))
                for token, record in payload.items()
            ]
            if data_to_write:
                conn.executemany(
                    "INSERT INTO cooldown_log (token, data) VALUES (?, ?)",
                    data_to_write,
                )
            set_meta("cooldown_json_mtime", str(json_mtime), conn)


def load_cooldown_log() -> Dict[str, Dict[str, Any]]:
    _init_db()
    _refresh_from_json_if_needed()
    with get_connection() as conn:
        rows = conn.execute("SELECT token, data FROM cooldown_log").fetchall()
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        try:
            payload = json.loads(row["data"]) or {}
        except Exception:
            payload = {}
        result[row["token"]] = payload
    if not result and _COOLDOWN_JSON_PATH.exists():
        return _load_json_file(_COOLDOWN_JSON_PATH)
    return result


def save_cooldown_log(log: Dict[str, Dict[str, Any]]) -> None:
    _init_db()
    sanitized: Dict[str, Dict[str, Any]] = {}
    for token, record in (log or {}).items():
        if isinstance(record, dict):
            payload = dict(record)
        else:
            payload = {"last_failure": float(record), "failure_count": 1}
        payload.setdefault("last_failure", time.time())
        payload.setdefault("failure_count", 1)
        sanitized[token.lower()] = payload

    with _LOCK:
        with get_connection() as conn:
            conn.execute("DELETE FROM cooldown_log")
            conn.executemany(
                "INSERT INTO cooldown_log (token, data, updated_at) VALUES (?, ?, strftime('%s','now'))",
                ((token, json.dumps(payload),) for token, payload in sanitized.items()),
            )
        _write_json_snapshot(sanitized)
