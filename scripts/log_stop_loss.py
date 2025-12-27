#!/usr/bin/env python3
"""
Log a stop-loss transaction to all databases.
"""

import sys
import csv
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.storage.positions import load_positions, remove_position
from src.storage.performance import load_performance_data, replace_performance_data
from src.core.performance_tracker import performance_tracker

def log_stop_loss(token_address: str, entry_price: float, exit_price: float, 
                  timestamp_str: str, symbol: str = None, chain: str = "solana"):
    """
    Log a stop-loss transaction to all databases.
    
    Args:
        token_address: Token address
        entry_price: Entry price
        exit_price: Exit price
        timestamp_str: Timestamp in format "2025-12-27 08:50:00"
        symbol: Token symbol (optional)
        chain: Chain ID (default: solana)
    """
    print(f"📝 Logging stop-loss transaction...")
    print(f"   Token: {symbol or '?'} ({token_address})")
    print(f"   Chain: {chain.upper()}")
    print(f"   Entry: ${entry_price:.6f}")
    print(f"   Exit: ${exit_price:.6f}")
    print(f"   Time: {timestamp_str}\n")
    
    # Calculate PnL
    pnl_pct = ((exit_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0.0
    
    # Load positions to get position size
    positions = load_positions()
    position_key = token_address.lower()
    position_size_usd = 0.0
    
    if position_key in positions:
        position_data = positions[position_key]
        if isinstance(position_data, dict):
            position_size_usd = float(position_data.get("position_size_usd", 0))
            symbol = symbol or position_data.get("symbol", "?")
    
    pnl_usd = (pnl_pct / 100) * position_size_usd if position_size_usd > 0 else 0.0
    
    print(f"📊 PnL Calculation:")
    print(f"   PnL: ${pnl_usd:.2f} ({pnl_pct:.2f}%)")
    print()
    
    # 1. Add to trade_log.csv
    try:
        log_file = project_root / "data" / "trade_log.csv"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        row = {
            "timestamp": timestamp_str,
            "token": token_address,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_pct": round(pnl_pct, 2),
            "reason": "stop_loss"
        }
        
        file_exists = log_file.exists()
        with open(log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"✅ Added to trade_log.csv")
    except Exception as e:
        print(f"⚠️  Failed to write to trade_log.csv: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. Update performance_data.json
    try:
        perf_data = load_performance_data()
        trades = perf_data.get("trades", [])
        token_address_lower = token_address.lower()
        chain_id_lower = chain.lower()
        
        updated = False
        # First, try to find an open trade
        for trade in trades:
            if (trade.get("status") == "open" and 
                trade.get("address", "").lower() == token_address_lower and
                trade.get("chain", "").lower() == chain_id_lower):
                trade["status"] = "closed"
                trade["exit_time"] = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").isoformat()
                trade["exit_price"] = exit_price
                trade["pnl_usd"] = pnl_usd
                trade["pnl_percent"] = pnl_pct
                if not trade.get("sell_tx_hash"):
                    trade["sell_tx_hash"] = None
                updated = True
                print(f"✅ Updated trade {trade.get('id', '?')} in performance_data.json")
                break
        
        # If no open trade found, look for the most recent trade with matching entry price (within 1%)
        if not updated:
            for trade in reversed(trades):  # Check most recent first
                if (trade.get("address", "").lower() == token_address_lower and
                    trade.get("chain", "").lower() == chain_id_lower):
                    trade_entry_price = trade.get("entry_price", 0)
                    if abs(trade_entry_price - entry_price) / entry_price < 0.01:  # Within 1%
                        if trade.get("status") in ["open", "partial_tp"]:
                            trade["status"] = "closed"
                            trade["exit_time"] = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").isoformat()
                            trade["exit_price"] = exit_price
                            trade["pnl_usd"] = pnl_usd
                            trade["pnl_percent"] = pnl_pct
                            if not trade.get("sell_tx_hash"):
                                trade["sell_tx_hash"] = None
                            updated = True
                            print(f"✅ Updated trade {trade.get('id', '?')} in performance_data.json (found by entry price match)")
                            break
        
        # If still not found, create a new trade entry
        if not updated:
            from datetime import datetime as dt
            trade_id = f"{symbol.lower()}_{timestamp_str.replace(' ', '_').replace(':', '')}"
            new_trade = {
                "id": trade_id,
                "symbol": symbol,
                "address": token_address,
                "chain": chain_id_lower,
                "entry_time": "2025-12-25T15:06:26.603149",  # From open_positions.json
                "entry_price": entry_price,
                "position_size_usd": position_size_usd,
                "exit_time": datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").isoformat(),
                "exit_price": exit_price,
                "pnl_usd": pnl_usd,
                "pnl_percent": pnl_pct,
                "status": "closed",
                "sell_tx_hash": None
            }
            trades.append(new_trade)
            perf_data["trades"] = trades
            updated = True
            print(f"✅ Created new trade entry {trade_id} in performance_data.json")
        
        if updated:
            replace_performance_data(perf_data)
            print(f"✅ Performance data saved")
        else:
            print(f"⚠️  Could not update performance_data.json")
    except Exception as e:
        print(f"⚠️  Error updating performance_data.json: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. Remove from open_positions.json and hunter_state.db
    try:
        remove_position(position_key)
        print(f"✅ Removed from open_positions.json and hunter_state.db")
    except Exception as e:
        print(f"⚠️  Error removing position: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ Stop-loss transaction logged to all databases!")
    return True

if __name__ == "__main__":
    # Hardcoded values for this specific transaction
    token_address = "Dfh5DzRgSvvCFDoYc2ciTkMrbDfRKybA4SoFbPmApump"
    entry_price = 0.504900
    exit_price = 0.458000
    timestamp_str = "2025-12-27 08:50:00"
    symbol = "pippin"
    chain = "solana"
    
    log_stop_loss(token_address, entry_price, exit_price, timestamp_str, symbol, chain)

