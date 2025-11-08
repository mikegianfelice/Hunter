from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

DB_PATH = Path("data/hunter_state.db")


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    return conn


def get_meta(key: str, conn: Optional[sqlite3.Connection] = None) -> Optional[str]:
    if conn is not None:
        row = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    with get_connection() as scoped:
        row = scoped.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None


def set_meta(key: str, value: str, conn: Optional[sqlite3.Connection] = None) -> None:
    if conn is not None:
        conn.execute("REPLACE INTO metadata (key, value) VALUES (?, ?)", (key, value))
        return

    with get_connection() as scoped:
        scoped.execute("REPLACE INTO metadata (key, value) VALUES (?, ?)", (key, value))
