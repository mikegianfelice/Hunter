from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict

from .db import get_connection, get_meta, set_meta

POSITIONS_JSON_PATH = Path("data/open_positions.json")
_LOCK = threading.Lock()


def _init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS positions (
                position_key TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                updated_at REAL DEFAULT (strftime('%s','now'))
            )
            """
        )
    _migrate_from_json()


def _migrate_from_json() -> None:
    if not POSITIONS_JSON_PATH.exists():
        return
    try:
        with POSITIONS_JSON_PATH.open("r", encoding="utf-8") as fh:
            raw = fh.read().strip() or "{}"
            positions = json.loads(raw)
    except Exception:
        return

    if not isinstance(positions, dict) or not positions:
        return

    with get_connection() as conn:
        count = conn.execute("SELECT COUNT(1) FROM positions").fetchone()[0]
        if count > 0:
            return
        for position_key, payload in positions.items():
            try:
                conn.execute(
                    "REPLACE INTO positions (position_key, data) VALUES (?, ?)",
                    (position_key, json.dumps(payload)),
                )
            except Exception:
                continue
        set_meta("positions_json_mtime", str(POSITIONS_JSON_PATH.stat().st_mtime), conn)


def _write_json_snapshot() -> None:
    positions = load_positions()
    POSITIONS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with POSITIONS_JSON_PATH.open("w", encoding="utf-8") as fh:
        json.dump(positions, fh, indent=2)
    set_meta("positions_json_mtime", str(POSITIONS_JSON_PATH.stat().st_mtime))


def load_positions() -> Dict[str, Dict[str, Any]]:
    _init_db()
    json_mtime = None
    if POSITIONS_JSON_PATH.exists():
        try:
            json_mtime = POSITIONS_JSON_PATH.stat().st_mtime
        except OSError:
            json_mtime = None

    if json_mtime is not None:
        recorded = get_meta("positions_json_mtime")
        if recorded is None or float(recorded) < json_mtime:
            try:
                with POSITIONS_JSON_PATH.open("r", encoding="utf-8") as fh:
                    raw = fh.read().strip() or "{}"
                    positions_from_json = json.loads(raw)
            except Exception:
                positions_from_json = {}
            if isinstance(positions_from_json, dict) and positions_from_json:
                with _LOCK:
                    with get_connection() as conn:
                        conn.execute("DELETE FROM positions")
                        conn.executemany(
                            "INSERT INTO positions (position_key, data, updated_at) VALUES (?, ?, strftime('%s','now'))",
                            ((k, json.dumps(v)) for k, v in positions_from_json.items()),
                        )
                        set_meta("positions_json_mtime", str(json_mtime), conn)

    with get_connection() as conn:
        rows = conn.execute("SELECT position_key, data FROM positions").fetchall()
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        try:
            payload = json.loads(row["data"]) if row["data"] else {}
            result[row["position_key"]] = payload
        except Exception:
            continue
    if not result and POSITIONS_JSON_PATH.exists():
        try:
            with POSITIONS_JSON_PATH.open("r", encoding="utf-8") as fh:
                return json.load(fh) or {}
        except Exception:
            return {}
    return result


def upsert_position(position_key: str, position_data: Dict[str, Any]) -> None:
    _init_db()
    with _LOCK:
        serialized = json.dumps(position_data)
        with get_connection() as conn:
            conn.execute(
                "REPLACE INTO positions (position_key, data, updated_at) VALUES (?, ?, strftime('%s','now'))",
                (position_key, serialized),
            )
        _write_json_snapshot()


def remove_position(position_key: str) -> None:
    _init_db()
    with _LOCK:
        with get_connection() as conn:
            conn.execute("DELETE FROM positions WHERE position_key = ?", (position_key,))
        _write_json_snapshot()


def replace_positions(positions: Dict[str, Dict[str, Any]]) -> None:
    _init_db()
    with _LOCK:
        with get_connection() as conn:
            conn.execute("DELETE FROM positions")
            conn.executemany(
                "INSERT INTO positions (position_key, data, updated_at) VALUES (?, ?, strftime('%s','now'))",
                ((k, json.dumps(v)) for k, v in positions.items()),
            )
        _write_json_snapshot()


def export_positions_json() -> None:
    _write_json_snapshot()
