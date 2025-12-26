#!/usr/bin/env python3
"""
Manually log a partial take-profit transaction that wasn't logged automatically.
This can happen when transaction verification fails but the transaction actually succeeded on-chain.
"""

import sys
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.monitoring.monitor_position import log_trade
from src.utils.helius_client import HeliusClient
from src.config.secrets import HELIUS_API_KEY


def get_transaction_details(tx_signature: str) -> Optional[dict]:
    """Get transaction details from Helius"""
    try:
        client = HeliusClient(HELIUS_API_KEY)
        tx = client.get_transaction(tx_signature)
        return tx
    except Exception as e:
        print(f"⚠️ Error fetching transaction: {e}")
        return None


def manual_log_partial_tp(
    token_address: str,
    entry_price: float,
    exit_price: float,
    partial_pct: float = 0.5,
    tx_signature: Optional[str] = None,
    timestamp: Optional[str] = None
):
    """
    Manually log a partial take-profit transaction
    
    Args:
        token_address: Token address
        entry_price: Entry price
        exit_price: Exit price at time of partial TP
        partial_pct: Percentage of position sold (default 0.5 for 50%)
        tx_signature: Optional transaction signature for verification
        timestamp: Optional timestamp (defaults to now)
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Verify transaction if signature provided
    if tx_signature:
        print(f"🔍 Verifying transaction {tx_signature}...")
        tx_details = get_transaction_details(tx_signature)
        if tx_details:
            print(f"✅ Transaction found on-chain")
            # Could add more verification here if needed
        else:
            print(f"⚠️ Could not verify transaction on-chain, but proceeding with manual log")
    
    # Log the trade
    reason = f"partial_tp_{partial_pct:.0%}"
    if tx_signature:
        reason += f"_manual_{tx_signature[:8]}"
    else:
        reason += "_manual"
    
    log_trade(token_address, entry_price, exit_price, reason)
    print(f"✅ Partial TP manually logged: {token_address[:8]}... at {exit_price} ({partial_pct*100:.0f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python scripts/manual_log_partial_tp.py <token_address> <entry_price> <exit_price> [partial_pct] [tx_signature] [timestamp]")
        print("\nExample:")
        print("  python scripts/manual_log_partial_tp.py Dfh5DzRgSvvCFDoYc2ciTkMrbDfRKybA4SoFbPmApump 0.495 0.54 0.5 2kbwgSjPbKig5sFTvo6YoPKGVErMQ8bt9UDBrALz4SZct7T3JhnJjSnaFo23LwRB4Vo97y2WyNoaMuehCXTRj4B3")
        sys.exit(1)
    
    token_address = sys.argv[1]
    entry_price = float(sys.argv[2])
    exit_price = float(sys.argv[3])
    partial_pct = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    tx_signature = sys.argv[5] if len(sys.argv) > 5 else None
    timestamp = sys.argv[6] if len(sys.argv) > 6 else None
    
    manual_log_partial_tp(token_address, entry_price, exit_price, partial_pct, tx_signature, timestamp)

