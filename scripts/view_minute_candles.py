#!/usr/bin/env python3
"""
View candles built from 5-minute price snapshots for tracked tokens (pippin, fartcoin, usor).
"""

import sys
import os
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.storage.minute_price_tracker import (
    get_price_snapshots,
    build_candles_from_snapshots,
    TRACKED_TOKENS,
)


def view_candles(token_name: str, hours: int = 24, interval: str = "15m"):
    """View candles for a token"""
    token_address = TRACKED_TOKENS.get(token_name.lower())
    if not token_address:
        print(f"❌ Unknown token: {token_name}")
        return
    
    # Parse interval
    interval_map = {
        "5m": 300,
        "15m": 900,
        "1h": 3600,
    }
    interval_seconds = interval_map.get(interval.lower(), 900)
    
    # Calculate time range
    end_time = time.time()
    start_time = end_time - (hours * 3600)
    
    # Get 5-minute snapshots
    snapshots = get_price_snapshots(token_address, start_time, end_time)
    
    if not snapshots:
        print(f"❌ No price snapshot data found for {token_name}")
        return
    
    # Build candles from snapshots
    candles = build_candles_from_snapshots(snapshots, interval_seconds)
    
    print(f"\n📊 {token_name.upper()} - {interval} candles (last {hours} hours)")
    print(f"📈 {len(candles)} candles from {len(snapshots)} 5-minute snapshots")
    print("-" * 80)
    print(f"{'Time':<20} {'Open':<15} {'High':<15} {'Low':<15} {'Close':<15}")
    print("-" * 80)
    
    for candle in candles[-20:]:  # Show last 20 candles
        dt = datetime.fromtimestamp(candle["time"])
        print(
            f"{dt.strftime('%Y-%m-%d %H:%M'):<20} "
            f"${candle['open']:<14.8f} "
            f"${candle['high']:<14.8f} "
            f"${candle['low']:<14.8f} "
            f"${candle['close']:<14.8f}"
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="View price snapshot candles")
    parser.add_argument("token", choices=["pippin", "fartcoin", "usor"], help="Token to view")
    parser.add_argument("--hours", type=int, default=24, help="Hours of history")
    parser.add_argument("--interval", choices=["5m", "15m", "1h"], default="15m", help="Candle interval")
    
    args = parser.parse_args()
    view_candles(args.token, args.hours, args.interval)
