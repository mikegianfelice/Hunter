from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .db import get_connection, get_meta, set_meta

_PERFORMANCE_JSON_PATH = Path("data/performance_data.json")
_LOCK = threading.Lock()


def set_json_path(path: str | Path) -> None:
    global _PERFORMANCE_JSON_PATH
    _PERFORMANCE_JSON_PATH = Path(path)


def _init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS performance_trades (
                trade_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                entry_time TEXT,
                status TEXT,
                updated_at REAL DEFAULT (strftime('%s','now'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS performance_daily_stats (
                day TEXT PRIMARY KEY,
                data TEXT NOT NULL
            )
            """
        )
    _migrate_from_json()


def _migrate_from_json() -> None:
    if not _PERFORMANCE_JSON_PATH.exists():
        return
    try:
        with _PERFORMANCE_JSON_PATH.open("r", encoding="utf-8") as fh:
            payload = json.load(fh) or {}
    except Exception:
        return

    trades = payload.get("trades", []) or []
    daily_stats = payload.get("daily_stats", {}) or {}

    with get_connection() as conn:
        trade_count = conn.execute("SELECT COUNT(1) FROM performance_trades").fetchone()[0]
        if trade_count > 0:
            return

        for trade in trades:
            trade_id = trade.get("id") or _generate_trade_id(trade)
            conn.execute(
                "REPLACE INTO performance_trades (trade_id, data, entry_time, status) VALUES (?, ?, ?, ?)",
                (
                    trade_id,
                    json.dumps(trade),
                    trade.get("entry_time"),
                    trade.get("status"),
                ),
            )
        for day, stats in daily_stats.items():
            conn.execute(
                "REPLACE INTO performance_daily_stats (day, data) VALUES (?, ?)",
                (day, json.dumps(stats)),
            )
        set_meta("performance_json_mtime", str(_PERFORMANCE_JSON_PATH.stat().st_mtime), conn)


def _generate_trade_id(trade: Dict[str, Any]) -> str:
    symbol = trade.get("symbol", "UNKNOWN")
    entry_time = trade.get("entry_time") or datetime.now().isoformat()
    safe_time = entry_time.replace(":", "").replace("-", "").replace("T", "_")
    return f"{symbol}_{safe_time}"


def _write_json_snapshot(data: Dict[str, Any]) -> None:
    snapshot = dict(data)
    snapshot.setdefault("last_updated", datetime.now().isoformat())
    _PERFORMANCE_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _PERFORMANCE_JSON_PATH.open("w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, indent=2)
    set_meta("performance_json_mtime", str(_PERFORMANCE_JSON_PATH.stat().st_mtime))


def load_performance_data() -> Dict[str, Any]:
    _init_db()

    json_mtime = None
    if _PERFORMANCE_JSON_PATH.exists():
        try:
            json_mtime = _PERFORMANCE_JSON_PATH.stat().st_mtime
        except OSError:
            json_mtime = None

    if json_mtime is not None:
        recorded = get_meta("performance_json_mtime")
        if recorded is None or float(recorded) < json_mtime:
            try:
                with _PERFORMANCE_JSON_PATH.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh) or {}
            except Exception:
                payload = {}
            trades = payload.get("trades", []) or []
            daily_stats = payload.get("daily_stats", {}) or {}
            with _LOCK:
                with get_connection() as conn:
                    conn.execute("DELETE FROM performance_trades")
                    conn.execute("DELETE FROM performance_daily_stats")
                    conn.executemany(
                        "INSERT INTO performance_trades (trade_id, data, entry_time, status, updated_at) VALUES (?, ?, ?, ?, strftime('%s','now'))",
                        (
                            (
                                trade.get("id") or _generate_trade_id(trade),
                                json.dumps(trade),
                                trade.get("entry_time"),
                                trade.get("status"),
                            )
                            for trade in trades
                        ),
                    )
                    conn.executemany(
                        "INSERT INTO performance_daily_stats (day, data) VALUES (?, ?)",
                        ((day, json.dumps(stats)) for day, stats in daily_stats.items()),
                    )
                    set_meta("performance_json_mtime", str(json_mtime), conn)

    with get_connection() as conn:
        trade_rows = conn.execute("SELECT data FROM performance_trades").fetchall()
        stats_rows = conn.execute("SELECT day, data FROM performance_daily_stats").fetchall()

    trades: List[Dict[str, Any]] = []
    for row in trade_rows:
        try:
            trades.append(json.loads(row["data"]))
        except Exception:
            continue

    daily_stats: Dict[str, Any] = {}
    for row in stats_rows:
        try:
            daily_stats[row["day"]] = json.loads(row["data"])
        except Exception:
            continue

    if not trades and _PERFORMANCE_JSON_PATH.exists():
        try:
            with _PERFORMANCE_JSON_PATH.open("r", encoding="utf-8") as fh:
                payload = json.load(fh) or {}
                trades = payload.get("trades", []) or []
                daily_stats = payload.get("daily_stats", {}) or {}
        except Exception:
            pass

    return {
        "trades": trades,
        "daily_stats": daily_stats,
    }


def replace_performance_data(data: Dict[str, Any]) -> None:
    _init_db()
    trades = data.get("trades", []) or []
    daily_stats = data.get("daily_stats", {}) or {}

    with _LOCK:
        with get_connection() as conn:
            conn.execute("DELETE FROM performance_trades")
            conn.execute("DELETE FROM performance_daily_stats")
            conn.executemany(
                "INSERT INTO performance_trades (trade_id, data, entry_time, status, updated_at) VALUES (?, ?, ?, ?, strftime('%s','now'))",
                (
                    (
                        trade.get("id") or _generate_trade_id(trade),
                        json.dumps(trade),
                        trade.get("entry_time"),
                        trade.get("status"),
                    )
                    for trade in trades
                ),
            )
            conn.executemany(
                "INSERT INTO performance_daily_stats (day, data) VALUES (?, ?)",
                ((day, json.dumps(stats)) for day, stats in daily_stats.items()),
            )
        _write_json_snapshot(data)
