#!/usr/bin/env python3
"""
Update a trade's sell_tx_hash in performance_data.json with a transaction signature.
Useful when a transaction succeeded on-chain but the hash wasn't captured during execution.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.storage.performance import load_performance_data, replace_performance_data


def update_trade_tx_hash(
    trade_id: Optional[str] = None,
    token_address: Optional[str] = None,
    tx_hash: str = None,
    entry_time: Optional[str] = None
) -> bool:
    """
    Update sell_tx_hash for a trade in performance_data.json
    
    Args:
        trade_id: Trade ID to update (e.g., "pippin_20251225_131500")
        token_address: Token address (alternative to trade_id)
        tx_hash: Transaction hash to set
        entry_time: Entry time filter (alternative to trade_id)
    
    Returns:
        True if updated, False if not found
    """
    if not tx_hash:
        print("❌ Transaction hash is required")
        return False
    
    perf_data = load_performance_data()
    trades = perf_data.get("trades", [])
    
    updated = False
    for trade in trades:
        match = False
        
        if trade_id and trade.get("id") == trade_id:
            match = True
        elif token_address and trade.get("address", "").lower() == token_address.lower():
            if entry_time:
                # Also match on entry_time if provided
                trade_entry_time = trade.get("entry_time", "")
                if entry_time in trade_entry_time or trade_entry_time.startswith(entry_time):
                    match = True
            else:
                match = True
        
        if match:
            current_hash = trade.get("sell_tx_hash")
            if current_hash:
                print(f"⚠️ Trade {trade.get('id')} already has sell_tx_hash: {current_hash}")
                if current_hash != tx_hash:
                    response = input(f"   Replace with {tx_hash}? (y/n): ")
                    if response.lower() != 'y':
                        continue
            
            trade["sell_tx_hash"] = tx_hash
            print(f"✅ Updated trade {trade.get('id')} with sell_tx_hash: {tx_hash}")
            updated = True
    
    if updated:
        replace_performance_data(perf_data)
        print(f"✅ Performance data saved")
        return True
    else:
        print(f"❌ No matching trade found")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/update_trade_tx_hash.py <tx_hash> [trade_id|token_address] [entry_time]")
        print("\nExamples:")
        print("  python scripts/update_trade_tx_hash.py 2kbwgSjPbKig5sFTvo6YoPKGVErMQ8bt9UDBrALz4SZct7T3JhnJjSnaFo23LwRB4Vo97y2WyNoaMuehCXTRj4B3 pippin_20251225_131500")
        print("  python scripts/update_trade_tx_hash.py 2kbwgSjPbKig5sFTvo6YoPKGVErMQ8bt9UDBrALz4SZct7T3JhnJjSnaFo23LwRB4Vo97y2WyNoaMuehCXTRj4B3 Dfh5DzRgSvvCFDoYc2ciTkMrbDfRKybA4SoFbPmApump 2025-12-25T13:15:00")
        sys.exit(1)
    
    tx_hash = sys.argv[1]
    identifier = sys.argv[2] if len(sys.argv) > 2 else None
    entry_time = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Determine if identifier is trade_id or token_address
    trade_id = None
    token_address = None
    
    if identifier:
        if len(identifier) > 20:  # Likely a token address
            token_address = identifier
        else:  # Likely a trade_id
            trade_id = identifier
    
    update_trade_tx_hash(trade_id=trade_id, token_address=token_address, tx_hash=tx_hash, entry_time=entry_time)

