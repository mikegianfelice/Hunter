from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional

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


def _write_json_snapshot(positions: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
    """Write positions to JSON. If positions is provided, write it directly (avoids load_positions
    re-import logic that can overwrite DB with stale JSON). Otherwise load from DB."""
    if positions is not None:
        to_write = positions
    else:
        _init_db()
        to_write = _load_positions_from_db()
    POSITIONS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with POSITIONS_JSON_PATH.open("w", encoding="utf-8") as fh:
        json.dump(to_write, fh, indent=2)
    set_meta("positions_json_mtime", str(POSITIONS_JSON_PATH.stat().st_mtime))


def _load_positions_from_db() -> Dict[str, Dict[str, Any]]:
    """Load positions from database only (no JSON re-import)."""
    with get_connection() as conn:
        rows = conn.execute("SELECT position_key, data FROM positions").fetchall()
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        try:
            payload = json.loads(row["data"]) if row["data"] else {}
            result[row["position_key"]] = payload
        except Exception:
            continue
    return result


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


def remove_position(position_key: str, token_address: Optional[str] = None) -> bool:
    """
    Remove position by key. If key match fails and token_address provided,
    search by address and remove matching position.
    Returns True if position was removed, False otherwise.
    """
    _init_db()
    with _LOCK:
        with get_connection() as conn:
            # Try exact key match first
            cursor = conn.execute("DELETE FROM positions WHERE position_key = ?", (position_key,))
            deleted = cursor.rowcount > 0
            
            # If no match and token_address provided, search by address
            if not deleted and token_address:
                positions = load_positions()
                token_address_lower = token_address.lower()
                
                # Search for position by address
                for key, pos_data in positions.items():
                    if isinstance(pos_data, dict):
                        pos_addr = (pos_data.get("address") or key).lower()
                    else:
                        pos_addr = key.lower()
                    
                    if pos_addr == token_address_lower:
                        # Found matching position, remove it
                        conn.execute("DELETE FROM positions WHERE position_key = ?", (key,))
                        deleted = True
                        print(f"⚠️ Removed position by address lookup: {key} (original key: {position_key})")
                        break
            
            if deleted:
                _write_json_snapshot()
            elif token_address:
                print(f"⚠️ Failed to remove position: key '{position_key}' not found, address '{token_address}' also not found")
            
            return deleted


def replace_positions(positions: Dict[str, Dict[str, Any]]) -> None:
    _init_db()
    with _LOCK:
        with get_connection() as conn:
            conn.execute("DELETE FROM positions")
            conn.executemany(
                "INSERT INTO positions (position_key, data, updated_at) VALUES (?, ?, strftime('%s','now'))",
                ((k, json.dumps(v)) for k, v in positions.items()),
            )
        _write_json_snapshot(positions)  # Write provided dict directly - avoids load_positions re-import


def export_positions_json() -> None:
    _write_json_snapshot()
