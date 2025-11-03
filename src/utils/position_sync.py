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


def sync_position_from_performance_data(token_address: str, symbol: str, chain_id: str, entry_price: float, position_size_usd: float = None) -> bool:
    """Manually sync a position to open_positions.json"""
    try:
        # If position_size_usd not provided, try to get it from performance_data.json
        if position_size_usd is None:
            try:
                if PERFORMANCE_DATA_FILE.exists():
                    with open(PERFORMANCE_DATA_FILE, "r") as f:
                        perf_data = json.load(f)
                        trades = perf_data.get("trades", [])
                        # Find the most recent open trade for this token
                        matching_trades = [t for t in trades 
                                         if t.get("address", "").lower() == token_address.lower() 
                                         and t.get("status") == "open"]
                        if matching_trades:
                            # Sort by entry_time (most recent first)
                            matching_trades.sort(key=lambda x: x.get("entry_time", ""), reverse=True)
                            position_size_usd = matching_trades[0].get("position_size_usd", 0.0)
            except Exception:
                pass
        
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
        position_data = {
            "entry_price": float(entry_price),
            "chain_id": chain_id.lower(),
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }
        
        # Include position_size_usd if available
        if position_size_usd is not None and position_size_usd > 0:
            position_data["position_size_usd"] = float(position_size_usd)
        
        positions[token_address] = position_data
        
        # Atomic write
        temp_file = OPEN_POSITIONS_FILE.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(positions, f, indent=2)
        temp_file.replace(OPEN_POSITIONS_FILE)
        
        size_str = f" (${position_size_usd:.2f})" if position_size_usd else ""
        print(f"📝 Synced position: {symbol} ({token_address[:8]}...{token_address[-8:]}) on {chain_id.upper()} @ ${entry_price:.6f}{size_str}")
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
        # Group by address and keep only the most recent one per token
        trades = perf_data.get("trades", [])
        open_trades = [t for t in trades if t.get("status") == "open"]
        
        # Group by address and keep the most recent (by entry_time)
        positions_by_address = {}
        for trade in open_trades:
            address = trade.get("address", "").lower()
            if not address:
                continue
            
            entry_time = trade.get("entry_time", "")
            if address not in positions_by_address:
                positions_by_address[address] = trade
            else:
                # Keep the most recent one
                existing_time = positions_by_address[address].get("entry_time", "")
                if entry_time > existing_time:
                    positions_by_address[address] = trade
        
        synced_count = 0
        for address, trade in positions_by_address.items():
            symbol = trade.get("symbol", "?")
            chain = trade.get("chain", "ethereum").lower()
            entry_price = float(trade.get("entry_price", 0))
            position_size_usd = trade.get("position_size_usd", 0.0)
            
            if entry_price <= 0:
                continue
            
            # Sync position with position_size_usd
            success = sync_position_from_performance_data(address, symbol, chain, entry_price, position_size_usd)
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

