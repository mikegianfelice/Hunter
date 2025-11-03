"""
Utility to sync positions between performance_data.json and open_positions.json
Ensures positions are always tracked even if initial logging fails.
"""
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PERFORMANCE_DATA_FILE = PROJECT_ROOT / "data" / "performance_data.json"
OPEN_POSITIONS_FILE = PROJECT_ROOT / "data" / "open_positions.json"


def sync_position_from_performance_data(token_address: str, symbol: str, chain_id: str, entry_price: float) -> bool:
    """Manually sync a position to open_positions.json"""
    try:
        OPEN_POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing positions
        if OPEN_POSITIONS_FILE.exists():
            try:
                with open(OPEN_POSITIONS_FILE, "r") as f:
                    positions = json.load(f) or {}
            except (json.JSONDecodeError, IOError):
                positions = {}
        else:
            positions = {}
        
        # Add or update position
        positions[token_address] = {
            "entry_price": float(entry_price),
            "chain_id": chain_id.lower(),
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }
        
        # Atomic write
        temp_file = OPEN_POSITIONS_FILE.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(positions, f, indent=2)
        temp_file.replace(OPEN_POSITIONS_FILE)
        
        print(f"📝 Synced position: {symbol} ({token_address[:8]}...{token_address[-8:]}) on {chain_id.upper()} @ ${entry_price:.6f}")
        return True
    except Exception as e:
        print(f"⚠️ Failed to sync position: {e}")
        return False


def sync_all_open_positions() -> Dict[str, bool]:
    """
    Sync all open positions from performance_data.json to open_positions.json
    Returns dict of {token_address: success_bool}
    """
    results = {}
    
    try:
        if not PERFORMANCE_DATA_FILE.exists():
            return results
        
        # Load performance data
        try:
            with open(PERFORMANCE_DATA_FILE, "r") as f:
                perf_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ Failed to read performance_data.json: {e}")
            return results
        
        # Load existing open positions
        if OPEN_POSITIONS_FILE.exists():
            try:
                with open(OPEN_POSITIONS_FILE, "r") as f:
                    open_positions = json.load(f) or {}
            except (json.JSONDecodeError, IOError):
                open_positions = {}
        else:
            open_positions = {}
        
        # Find all open trades in performance_data
        trades = perf_data.get("trades", [])
        synced_count = 0
        for trade in trades:
            if trade.get("status") == "open":
                address = trade.get("address")
                symbol = trade.get("symbol", "?")
                chain = trade.get("chain", "ethereum").lower()
                entry_price = float(trade.get("entry_price", 0))
                
                if not address or entry_price <= 0:
                    continue
                
                # Check if already in open_positions (and entry price matches)
                if address in open_positions:
                    existing_entry = open_positions[address].get("entry_price")
                    if existing_entry and abs(float(existing_entry) - entry_price) < 0.000001:  # Prices match
                        continue  # Already synced
                
                # Sync position
                success = sync_position_from_performance_data(address, symbol, chain, entry_price)
                results[address] = success
                if success:
                    synced_count += 1
                
                # Brief delay to avoid file lock issues
                time.sleep(0.1)
        
        if synced_count > 0:
            print(f"✅ Synced {synced_count} position(s) from performance_data.json to open_positions.json")
        
        return results
        
    except Exception as e:
        print(f"⚠️ Error syncing positions: {e}")
        return results


if __name__ == "__main__":
    # Run sync when called directly
    results = sync_all_open_positions()
    if results:
        print(f"Synced {sum(1 for v in results.values() if v)} position(s)")
    else:
        print("No positions to sync")

