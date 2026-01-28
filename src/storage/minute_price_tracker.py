from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional
from datetime import datetime

from .db import get_connection

_LOCK = threading.Lock()

# Token addresses for tracked tokens
PIPPIN_ADDRESS = "Dfh5DzRgSvvCFDoYc2ciTkMrbDfRKybA4SoFbPmApump"  # Correct Pippin token address
FARTCOIN_ADDRESS = "9BB6NFEcjBCtnNLFko2FqVQBq8HHM13kCyYcdQbgpump"
USOR_ADDRESS = "USoRyaQjch6E18nCdDvWoRgTo6osQs9MUd8JXEsspWR"
ONE_ADDRESS = "GMvCfcZg8YvkkQmwDaAzCtHDrrEtgE74nQpQ7xNabonk"
USELESS_ADDRESS = "Dz9mQ9NzkBcCsuGPFJ3r1bS4wgqKMHBPiVuniW8Mbonk"
AVICI_ADDRESS = "BANKJmvhT8tiJRsBSS1n2HryMBPvT5Ze4HU95DUAmeta"
GME_ADDRESS = "8wXtPeU6557ETkp9WHFY1n1EcU6NxDvbAggHGsMYiHsB"
GIGA_ADDRESS = "63LfDmNb3MQ8mw9MtZ2To9bEA2M71kZUUGq5tiJxcqj9"
DREAMS_ADDRESS = "GMzuntWYJLpNuCizrSR7ZXggiMdDzTNiEmSNHHunpump"

TRACKED_TOKENS = {
    "pippin": PIPPIN_ADDRESS,
    "fartcoin": FARTCOIN_ADDRESS,
    "usor": USOR_ADDRESS,
    "one": ONE_ADDRESS,
    "useless": USELESS_ADDRESS,
    "avici": AVICI_ADDRESS,
    "gme": GME_ADDRESS,
    "giga": GIGA_ADDRESS,
    "dreams": DREAMS_ADDRESS,
}

# 5-minute interval (300 seconds)
INTERVAL_SECONDS = 300


def _init_db() -> None:
    """Initialize minute_price_tracker table with indexes"""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS minute_price_tracker (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_address TEXT NOT NULL,
                interval_timestamp INTEGER NOT NULL,
                price_usd REAL NOT NULL,
                created_at REAL DEFAULT (strftime('%s','now')),
                UNIQUE(token_address, interval_timestamp)
            )
            """
        )
        # Create indexes for fast queries
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_interval_token_time 
            ON minute_price_tracker(token_address, interval_timestamp)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_interval_time 
            ON minute_price_tracker(interval_timestamp)
            """
        )


def store_price_snapshot(
    token_address: str,
    price_usd: float,
    timestamp: Optional[float] = None,
) -> bool:
    """
    Store a 5-minute price snapshot.
    The timestamp is rounded down to the nearest 5-minute interval.
    Returns True if stored successfully, False if duplicate or error.
    """
    _init_db()
    
    if timestamp is None:
        timestamp = time.time()
    
    # Round down to nearest 5-minute interval (300 seconds)
    interval_timestamp = int(timestamp // INTERVAL_SECONDS) * INTERVAL_SECONDS
    
    try:
        with _LOCK:
            with get_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO minute_price_tracker 
                    (token_address, interval_timestamp, price_usd)
                    VALUES (?, ?, ?)
                    """,
                    (
                        token_address.lower(),
                        interval_timestamp,
                        price_usd,
                    ),
                )
                return cursor.lastrowid is not None and cursor.lastrowid > 0
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error storing price snapshot: {e}")
        return False


def get_price_snapshots(
    token_address: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Query 5-minute price snapshots from the database.
    
    Args:
        token_address: Token mint address to filter by
        start_time: Start timestamp (Unix time)
        end_time: End timestamp (Unix time)
        limit: Maximum number of results
    
    Returns:
        List of price snapshot dictionaries with keys: token_address, interval_timestamp, price_usd
    """
    _init_db()
    
    query = "SELECT * FROM minute_price_tracker WHERE token_address = ?"
    params: List[Any] = [token_address.lower()]
    
    if start_time is not None:
        # Round down to nearest interval
        start_interval = int(start_time // INTERVAL_SECONDS) * INTERVAL_SECONDS
        query += " AND interval_timestamp >= ?"
        params.append(start_interval)
    
    if end_time is not None:
        # Round down to nearest interval
        end_interval = int(end_time // INTERVAL_SECONDS) * INTERVAL_SECONDS
        query += " AND interval_timestamp <= ?"
        params.append(end_interval)
    
    query += " ORDER BY interval_timestamp ASC"
    
    if limit:
        query += " LIMIT ?"
        params.append(limit)
    
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    
    return [
        {
            "token_address": row["token_address"],
            "interval_timestamp": row["interval_timestamp"],
            "price_usd": row["price_usd"],
            "time": float(row["interval_timestamp"]),  # For compatibility with candle format
        }
        for row in rows
    ]


def get_latest_price_snapshot(token_address: str) -> Optional[Dict[str, Any]]:
    """
    Get the most recent price snapshot for a token.
    """
    _init_db()
    
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT * FROM minute_price_tracker 
            WHERE token_address = ? 
            ORDER BY interval_timestamp DESC 
            LIMIT 1
            """,
            (token_address.lower(),),
        ).fetchone()
    
    if row:
        return {
            "token_address": row["token_address"],
            "interval_timestamp": row["interval_timestamp"],
            "price_usd": row["price_usd"],
            "time": float(row["interval_timestamp"]),
        }
    return None


def build_candles_from_snapshots(
    price_snapshots: List[Dict[str, Any]], 
    target_interval_seconds: int = 300
) -> List[Dict[str, Any]]:
    """
    Build OHLC candles from 5-minute price snapshots.
    
    Args:
        price_snapshots: List of price snapshot dictionaries
        target_interval_seconds: Target candle interval in seconds (300=5m, 900=15m, 3600=1h)
    
    Returns:
        List of candle dictionaries with keys: time, open, high, low, close, volume
    """
    if not price_snapshots:
        return []
    
    candles = {}
    
    for snapshot in price_snapshots:
        timestamp = snapshot["interval_timestamp"]
        price = snapshot["price_usd"]
        
        # Round down to nearest target interval
        candle_timestamp = int(timestamp // target_interval_seconds) * target_interval_seconds
        
        if candle_timestamp not in candles:
            candles[candle_timestamp] = {
                "time": candle_timestamp,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 0,  # Volume not tracked in snapshots
            }
        else:
            candle = candles[candle_timestamp]
            candle["high"] = max(candle["high"], price)
            candle["low"] = min(candle["low"], price)
            candle["close"] = price
    
    return sorted(candles.values(), key=lambda x: x["time"])


def cleanup_old_snapshots(days_to_keep: int = 30) -> int:
    """
    Delete price snapshots older than specified days.
    Returns number of rows deleted.
    """
    _init_db()
    
    cutoff_time = int((datetime.now().timestamp() - (days_to_keep * 24 * 3600)) // INTERVAL_SECONDS) * INTERVAL_SECONDS
    
    with _LOCK:
        with get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM minute_price_tracker WHERE interval_timestamp < ?",
                (cutoff_time,),
            )
            return cursor.rowcount
