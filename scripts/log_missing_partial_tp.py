#!/usr/bin/env python3
"""
Log a missing partial TP transaction by fetching details from Helius and adding to trade_log.csv and hunter_state.db
"""

import sys
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helius_client import HeliusClient
from src.config.secrets import HELIUS_API_KEY, SOLANA_WALLET_ADDRESS
from src.monitoring.monitor_position import log_trade
from src.storage.performance import load_performance_data, replace_performance_data
from src.core.performance_tracker import performance_tracker


def extract_trade_info_from_transaction(tx: Dict[str, Any], wallet_address: str) -> Optional[Dict[str, Any]]:
    """Extract trade information from a normalized Helius transaction"""
    if not tx:
        return None
    
    # Get token transfers
    transfers = tx.get("transfers", [])
    if not transfers:
        return None
    
    # Find the token being sold (outgoing transfer)
    token_sold = None
    usdc_received = None
    
    for transfer in transfers:
        from_addr = transfer.get("from", "").lower()
        to_addr = transfer.get("to", "").lower()
        wallet_lower = wallet_address.lower()
        
        # Check if this is a sell (token going out, USDC/SOL coming in)
        if from_addr == wallet_lower:
            # Token being sold
            token_mint = transfer.get("mint", "").lower()
            if token_mint and token_mint != "so11111111111111111111111111111111111111112":  # Not SOL
                token_sold = {
                    "mint": token_mint,
                    "amount": transfer.get("amount", 0),
                    "decimals": transfer.get("decimals", 0)
                }
        elif to_addr == wallet_lower:
            # USDC or SOL received
            token_mint = transfer.get("mint", "").lower()
            if token_mint in ["epjfddbzamm6wwgekv6ymhzzrqgzvq8tp2ix9p7jz9bm",  # USDC
                             "so11111111111111111111111111111111111111112"]:  # SOL
                usdc_received = {
                    "mint": token_mint,
                    "amount": transfer.get("amount", 0),
                    "decimals": transfer.get("decimals", 0)
                }
    
    if not token_sold:
        return None
    
    # Get block time for timestamp
    block_time = tx.get("blockTime")
    if block_time:
        timestamp = datetime.fromtimestamp(block_time).strftime("%Y-%m-%d %H:%M:%S")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate price (we'll need to get current price or estimate)
    # For now, we'll need to fetch the current price of the token
    token_address = token_sold["mint"]
    
    return {
        "token_address": token_address,
        "timestamp": timestamp,
        "block_time": block_time,
        "token_amount_sold": token_sold["amount"] / (10 ** token_sold["decimals"]),
        "usdc_received": usdc_received["amount"] / (10 ** usdc_received["decimals"]) if usdc_received else None,
        "tx_signature": tx.get("signature")
    }


def log_missing_partial_tp(tx_signature: str, entry_price: Optional[float] = None):
    """
    Fetch transaction details and log the missing partial TP
    
    Args:
        tx_signature: Transaction signature
        entry_price: Optional entry price (if not provided, will try to find from performance_data)
    """
    print(f"🔍 Fetching transaction {tx_signature}...")
    
    try:
        client = HeliusClient(HELIUS_API_KEY)
        tx = client.get_transaction(tx_signature)
        
        if not tx:
            print(f"❌ Could not fetch transaction details")
            return False
        
        print(f"✅ Transaction found")
        
        # Extract trade info
        trade_info = extract_trade_info_from_transaction(tx, SOLANA_WALLET_ADDRESS)
        if not trade_info:
            print(f"❌ Could not extract trade information from transaction")
            return False
        
        token_address = trade_info["token_address"]
        timestamp = trade_info["timestamp"]
        
        print(f"📊 Token: {token_address[:8]}...")
        print(f"⏰ Timestamp: {timestamp}")
        print(f"💰 Amount sold: {trade_info['token_amount_sold']:.6f}")
        
        # Get exit price (price at time of sell)
        # Calculate from USDC received / tokens sold
        if trade_info["usdc_received"]:
            exit_price = trade_info["usdc_received"] / trade_info["token_amount_sold"]
            print(f"💵 Exit price: ${exit_price:.6f}")
        else:
            print(f"⚠️ Could not calculate exit price from transaction")
            # Try to get current price
            from src.monitoring.monitor_position import _fetch_token_price_multi_chain
            exit_price = _fetch_token_price_multi_chain(token_address)
            if exit_price <= 0:
                print(f"❌ Could not determine exit price")
                return False
        
        # Get entry price if not provided
        if entry_price is None:
            # Try to find from performance_data
            perf_data = load_performance_data()
            trades = perf_data.get("trades", [])
            
            # Find open trade for this token
            for trade in trades:
                if (trade.get("address", "").lower() == token_address.lower() and
                    trade.get("status") == "open" and
                    trade.get("chain", "").lower() == "solana"):
                    entry_price = trade.get("entry_price")
                    if entry_price:
                        print(f"📈 Found entry price from open position: ${entry_price:.6f}")
                        break
            
            if entry_price is None:
                print(f"⚠️ Could not find entry price. Please provide it manually.")
                print(f"   Usage: python scripts/log_missing_partial_tp.py <tx_signature> <entry_price>")
                return False
        
        # Calculate PnL
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100 if entry_price else 0.0
        
        # Log to trade_log.csv
        print(f"\n📝 Logging to trade_log.csv...")
        log_trade(token_address, entry_price, exit_price, f"partial_tp_50%_manual_{tx_signature[:8]}")
        
        # Update performance_data.json if there's an open trade
        perf_data = load_performance_data()
        trades = perf_data.get("trades", [])
        updated = False
        
        for trade in trades:
            if (trade.get("address", "").lower() == token_address.lower() and
                trade.get("status") == "open" and
                trade.get("chain", "").lower() == "solana"):
                # This is a partial TP, so mark it but don't close the position
                # Just update the sell_tx_hash if it's a new partial TP
                if not trade.get("sell_tx_hash"):
                    trade["sell_tx_hash"] = tx_signature
                    print(f"✅ Updated sell_tx_hash in performance_data.json")
                    updated = True
                break
        
        if updated:
            replace_performance_data(perf_data)
            print(f"✅ Performance data updated")
        
        print(f"\n✅ Partial TP logged successfully!")
        print(f"   Token: {token_address[:8]}...")
        print(f"   Entry: ${entry_price:.6f}")
        print(f"   Exit: ${exit_price:.6f}")
        print(f"   PnL: {pnl_pct:.2f}%")
        print(f"   Time: {timestamp}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/log_missing_partial_tp.py <tx_signature> [entry_price]")
        print("\nExample:")
        print("  python scripts/log_missing_partial_tp.py 2kbwgSjPbKig5sFTvo6YoPKGVErMQ8bt9UDBrALz4SZct7T3JhnJjSnaFo23LwRB4Vo97y2WyNoaMuehCXTRj4B3")
        sys.exit(1)
    
    tx_signature = sys.argv[1]
    entry_price = float(sys.argv[2]) if len(sys.argv) > 2 else None
    
    log_missing_partial_tp(tx_signature, entry_price)

