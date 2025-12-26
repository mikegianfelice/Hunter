#!/usr/bin/env python3
"""
Add a missing trade entry to trade_log.csv and update hunter_state.db
"""

import sys
import csv
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.monitoring.monitor_position import log_trade
from src.storage.performance import load_performance_data, replace_performance_data


def add_trade_entry(
    token_address: str,
    entry_price: float,
    exit_price: float,
    timestamp: str,
    reason: str = "partial_tp_50%",
    tx_hash: str = None
):
    """Add a trade entry to trade_log.csv"""
    log_file = project_root / "data" / "trade_log.csv"
    
    # Calculate PnL
    pnl_pct = ((exit_price - entry_price) / entry_price) * 100 if entry_price else 0.0
    
    row = {
        "timestamp": timestamp,
        "token": token_address,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl_pct": round(pnl_pct, 2),
        "reason": reason
    }
    
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_exists = log_file.exists()
        
        with open(log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        
        print(f"✅ Trade logged to trade_log.csv: {row}")
        
        # Also update performance_data.json if there's an open trade
        if tx_hash:
            perf_data = load_performance_data()
            trades = perf_data.get("trades", [])
            updated = False
            
            for trade in trades:
                if (trade.get("address", "").lower() == token_address.lower() and
                    trade.get("status") == "open" and
                    trade.get("chain", "").lower() == "solana"):
                    # Update sell_tx_hash for partial TP
                    if not trade.get("sell_tx_hash") or trade.get("sell_tx_hash") != tx_hash:
                        trade["sell_tx_hash"] = tx_hash
                        print(f"✅ Updated sell_tx_hash in performance_data.json")
                        updated = True
                    break
            
            if updated:
                replace_performance_data(perf_data)
                print(f"✅ Performance data updated")
        
        return True
    except Exception as e:
        print(f"❌ Error logging trade: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python scripts/add_missing_trade_entry.py <token_address> <entry_price> <exit_price> <timestamp> [reason] [tx_hash]")
        print("\nExample:")
        print("  python scripts/add_missing_trade_entry.py Dfh5DzRgSvvCFDoYc2ciTkMrbDfRKybA4SoFbPmApump 0.495 0.54 '2025-12-25 20:25:00' 'partial_tp_50%' 2kbwgSjPbKig5sFTvo6YoPKGVErMQ8bt9UDBrALz4SZct7T3JhnJjSnaFo23LwRB4Vo97y2WyNoaMuehCXTRj4B3")
        sys.exit(1)
    
    token_address = sys.argv[1]
    entry_price = float(sys.argv[2])
    exit_price = float(sys.argv[3])
    timestamp = sys.argv[4]
    reason = sys.argv[5] if len(sys.argv) > 5 else "partial_tp_50%"
    tx_hash = sys.argv[6] if len(sys.argv) > 6 else None
    
    add_trade_entry(token_address, entry_price, exit_price, timestamp, reason, tx_hash)

