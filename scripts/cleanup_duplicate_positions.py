#!/usr/bin/env python3
"""
Clean up duplicate positions in performance_data.json
Keep only the most recent open position per token address
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERFORMANCE_DATA_FILE = PROJECT_ROOT / "data" / "performance_data.json"


def cleanup_duplicates():
    """Remove duplicate open positions, keeping only the most recent per token"""
    try:
        if not PERFORMANCE_DATA_FILE.exists():
            print("❌ performance_data.json not found")
            return False
        
        # Load performance data
        with open(PERFORMANCE_DATA_FILE, "r") as f:
            perf_data = json.load(f)
        
        trades = perf_data.get("trades", [])
        open_trades = [t for t in trades if t.get("status") == "open"]
        closed_trades = [t for t in trades if t.get("status") != "open"]
        
        print(f"📊 Found {len(open_trades)} open trades and {len(closed_trades)} closed trades")
        
        # Group open trades by address
        positions_by_address = {}
        duplicates_to_close = []
        
        for trade in open_trades:
            address = trade.get("address", "").lower()
            if not address:
                continue
            
            entry_time = trade.get("entry_time", "")
            
            if address not in positions_by_address:
                positions_by_address[address] = trade
            else:
                # Compare with existing - keep the most recent
                existing_time = positions_by_address[address].get("entry_time", "")
                if entry_time > existing_time:
                    # New one is more recent, close the old one
                    duplicates_to_close.append(positions_by_address[address])
                    positions_by_address[address] = trade
                else:
                    # Existing is more recent, close this one
                    duplicates_to_close.append(trade)
        
        if duplicates_to_close:
            print(f"🔧 Closing {len(duplicates_to_close)} duplicate position(s):")
            for dup in duplicates_to_close:
                symbol = dup.get("symbol", "?")
                address = dup.get("address", "")[:8] + "..."
                entry_time = dup.get("entry_time", "")
                print(f"  - {symbol} ({address}) @ {entry_time}")
                
                # Mark as closed
                dup["status"] = "closed"
                dup["exit_time"] = entry_time  # Use entry time as exit time for duplicates
                dup["exit_price"] = dup.get("entry_price", 0)
                dup["pnl_usd"] = 0.0
                dup["pnl_percent"] = 0.0
        
        # Rebuild trades list
        unique_open_trades = list(positions_by_address.values())
        all_trades = unique_open_trades + duplicates_to_close + closed_trades
        
        # Sort by entry_time to maintain order
        all_trades.sort(key=lambda x: x.get("entry_time", ""))
        
        perf_data["trades"] = all_trades
        
        # Save back
        backup_file = PERFORMANCE_DATA_FILE.with_suffix(".json.bak")
        if backup_file.exists():
            backup_file.unlink()
        
        # Create backup
        with open(backup_file, "w") as f:
            json.dump(perf_data, f, indent=2)
        
        # Write cleaned data
        with open(PERFORMANCE_DATA_FILE, "w") as f:
            json.dump(perf_data, f, indent=2)
        
        print(f"✅ Cleaned up duplicates:")
        print(f"   - {len(unique_open_trades)} unique open position(s) remaining")
        print(f"   - {len(duplicates_to_close)} duplicate(s) marked as closed")
        print(f"   - Backup saved to {backup_file.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error cleaning up duplicates: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = cleanup_duplicates()
    sys.exit(0 if success else 1)

