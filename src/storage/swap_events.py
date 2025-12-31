from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from .db import get_connection, get_meta, set_meta

_LOCK = threading.Lock()


def _init_db() -> None:
    """Initialize swap_events table with indexes"""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS swap_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_address TEXT NOT NULL,
                pool_address TEXT,
                tx_signature TEXT UNIQUE NOT NULL,
                block_time REAL NOT NULL,
                price_usd REAL NOT NULL,
                volume_usd REAL,
                amount_in REAL,
                amount_out REAL,
                base_mint TEXT,
                quote_mint TEXT,
                dex_program TEXT,
                signer_wallet TEXT,
                created_at REAL DEFAULT (strftime('%s','now'))
            )
            """
        )
        # Add signer_wallet column if it doesn't exist (migration)
        try:
            conn.execute("ALTER TABLE swap_events ADD COLUMN signer_wallet TEXT")
        except Exception:
            pass  # Column already exists
        # Create indexes for fast queries
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_swap_token_time 
            ON swap_events(token_address, block_time)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_swap_pool_time 
            ON swap_events(pool_address, block_time)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_swap_tx 
            ON swap_events(tx_signature)
            """
        )


def store_swap_event(
    token_address: str,
    tx_signature: str,
    block_time: float,
    price_usd: float,
    pool_address: Optional[str] = None,
    volume_usd: Optional[float] = None,
    amount_in: Optional[float] = None,
    amount_out: Optional[float] = None,
    base_mint: Optional[str] = None,
    quote_mint: Optional[str] = None,
    dex_program: Optional[str] = None,
    signer_wallet: Optional[str] = None,
) -> bool:
    """
    Store a swap event in the database.
    Returns True if stored successfully, False if duplicate or error.
    """
    _init_db()
    
    try:
        with _LOCK:
            with get_connection() as conn:
                # Use INSERT OR IGNORE to handle duplicates gracefully
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO swap_events 
                    (token_address, pool_address, tx_signature, block_time, price_usd, 
                     volume_usd, amount_in, amount_out, base_mint, quote_mint, dex_program, signer_wallet)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        token_address.lower(),
                        pool_address.lower() if pool_address else None,
                        tx_signature,
                        block_time,
                        price_usd,
                        volume_usd,
                        amount_in,
                        amount_out,
                        base_mint.lower() if base_mint else None,
                        quote_mint.lower() if quote_mint else None,
                        dex_program,
                        signer_wallet.lower() if signer_wallet else None,
                    ),
                )
                # Check if row was inserted by checking lastrowid
                return cursor.lastrowid is not None and cursor.lastrowid > 0
    except Exception as e:
        # Log error but don't fail silently
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error storing swap event: {e}")
        return False


def get_swap_events(
    token_address: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    pool_address: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Query swap events from the database.
    
    Args:
        token_address: Token mint address to filter by
        start_time: Start timestamp (Unix time)
        end_time: End timestamp (Unix time)
        pool_address: Optional pool address to filter by
        limit: Maximum number of results
    
    Returns:
        List of swap event dictionaries
    """
    _init_db()
    
    query = "SELECT * FROM swap_events WHERE token_address = ?"
    params: List[Any] = [token_address.lower()]
    
    if start_time is not None:
        query += " AND block_time >= ?"
        params.append(start_time)
    
    if end_time is not None:
        query += " AND block_time <= ?"
        params.append(end_time)
    
    if pool_address:
        query += " AND pool_address = ?"
        params.append(pool_address.lower())
    
    query += " ORDER BY block_time ASC"
    
    if limit:
        query += " LIMIT ?"
        params.append(limit)
    
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    
    return [
        {
            "token_address": row["token_address"],
            "pool_address": row["pool_address"],
            "tx_signature": row["tx_signature"],
            "block_time": row["block_time"],
            "price_usd": row["price_usd"],
            "volume_usd": row["volume_usd"],
            "amount_in": row["amount_in"],
            "amount_out": row["amount_out"],
            "base_mint": row["base_mint"],
            "quote_mint": row["quote_mint"],
            "dex_program": row["dex_program"],
            "signer_wallet": row.get("signer_wallet"),
        }
        for row in rows
    ]


def get_latest_swap_time(token_address: Optional[str] = None) -> Optional[float]:
    """
    Get the timestamp of the most recent swap event.
    If token_address is provided, returns latest for that token only.
    """
    _init_db()
    
    if token_address:
        query = "SELECT MAX(block_time) as max_time FROM swap_events WHERE token_address = ?"
        params = [token_address.lower()]
    else:
        query = "SELECT MAX(block_time) as max_time FROM swap_events"
        params = []
    
    with get_connection() as conn:
        row = conn.execute(query, params).fetchone()
        return row["max_time"] if row and row["max_time"] else None


def get_swap_count(
    token_address: Optional[str] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> int:
    """Get count of swap events matching criteria"""
    _init_db()
    
    query = "SELECT COUNT(*) as cnt FROM swap_events"
    params: List[Any] = []
    
    conditions = []
    if token_address:
        conditions.append("token_address = ?")
        params.append(token_address.lower())
    
    if start_time is not None:
        conditions.append("block_time >= ?")
        params.append(start_time)
    
    if end_time is not None:
        conditions.append("block_time <= ?")
        params.append(end_time)
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    with get_connection() as conn:
        row = conn.execute(query, params).fetchone()
        return row["cnt"] if row else 0


def cleanup_old_swaps(days_to_keep: int = 30) -> int:
    """
    Delete swap events older than specified days.
    Returns number of rows deleted.
    """
    _init_db()
    
    cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
    
    with _LOCK:
        with get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM swap_events WHERE block_time < ?",
                (cutoff_time,),
            )
            return cursor.rowcount
