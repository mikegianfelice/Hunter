#!/usr/bin/env python3
"""
Quick script to remove a sold position from open_positions.json and database.
Use this when reconciliation hasn't run or failed.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.storage.positions import load_positions, remove_position
from src.core.performance_tracker import performance_tracker

def cleanup_position(token_address: str, trade_id: str = None):
    """Remove position from open_positions and close trade if needed"""
    print(f"🧹 Cleaning up position: {token_address}")
    
    # Load positions to find the key
    positions = load_positions()
    token_lower = token_address.lower()
    
    # Find position by address or trade_id
    position_key = None
    for key, pos_data in positions.items():
        if isinstance(pos_data, dict):
            pos_addr = (pos_data.get("address") or key).lower()
            pos_trade_id = pos_data.get("trade_id")
            if pos_addr == token_lower or (trade_id and pos_trade_id == trade_id):
                position_key = key
                break
        elif key.lower() == token_lower:
            position_key = key
            break
    
    if not position_key:
        print(f"❌ Position not found in open_positions.json")
        print(f"   Current positions: {list(positions.keys())}")
        return False
    
    print(f"✅ Found position with key: {position_key}")
    
    # Remove from positions
    removed = remove_position(position_key, token_address)
    if removed:
        print(f"✅ Removed position from open_positions.json and database")
    else:
        print(f"⚠️ Failed to remove position")
        return False
    
    # Check if trade needs to be closed in performance_tracker
    if trade_id:
        for trade in performance_tracker.trades:
            if trade.get("id") == trade_id and trade.get("status") == "open":
                print(f"⚠️ Trade {trade_id} is still marked as 'open' in performance_tracker")
                print(f"   This should be closed by reconciliation when it runs next")
                print(f"   Or you can manually close it using mark_position_closed.py")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/cleanup_sold_position.py <token_address> [trade_id]")
        print("\nExample:")
        print("  python3 scripts/cleanup_sold_position.py BANKJmvhT8tiJRsBSS1n2HryMBPvT5Ze4HU95DUAmeta AVICI_20260120_052403")
        sys.exit(1)
    
    token_address = sys.argv[1]
    trade_id = sys.argv[2] if len(sys.argv) > 2 else None
    
    cleanup_position(token_address, trade_id)
