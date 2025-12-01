#!/usr/bin/env python3
"""
Manually close a specific position by selling the token.

Usage:
    python3 scripts/close_position.py <token_address> [--symbol SYMBOL] [--chain CHAIN]
    
Example:
    python3 scripts/close_position.py EjamcKN1PixSzm3GiFgUaqCFXBMy3F51JKmbUqNF99S --symbol Hajimi --chain solana
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.execution.solana_executor import sell_token_solana
from src.storage.positions import load_positions, remove_position
from src.storage.performance import load_performance_data, replace_performance_data
from src.core.performance_tracker import performance_tracker
from src.utils.position_sync import create_position_key, resolve_token_address
from datetime import datetime
import csv
from pathlib import Path

def close_position(token_address: str, symbol: str = None, chain: str = "solana", skip_confirm: bool = False):
    """
    Manually close a position by selling the token.
    
    Args:
        token_address: Token address to close
        symbol: Token symbol (optional, for display)
        chain: Chain ID (default: solana)
    """
    print(f"🔍 Closing position for token: {token_address}")
    if symbol:
        print(f"   Symbol: {symbol}")
    print(f"   Chain: {chain}\n")
    
    # Load positions
    positions = load_positions()
    # Position key is just the lowercase token address
    position_key = token_address.lower()
    
    # Check if position exists
    if position_key not in positions:
        print(f"❌ Position not found for token {token_address} on {chain}")
        print(f"   Position key: {position_key}")
        print(f"\n📋 Current open positions:")
        for key, pos in positions.items():
            pos_symbol = pos.get("symbol", "?") if isinstance(pos, dict) else "?"
            pos_addr = pos.get("address", key) if isinstance(pos, dict) else key
            print(f"   • {pos_symbol}: {pos_addr}")
        return False
    
    position_data = positions[position_key]
    if isinstance(position_data, dict):
        entry_price = float(position_data.get("entry_price", 0))
        position_size_usd = float(position_data.get("position_size_usd", 0))
        trade_id = position_data.get("trade_id")
        symbol = symbol or position_data.get("symbol", "?")
    else:
        entry_price = float(position_data) if position_data else 0
        position_size_usd = 0
        trade_id = None
        symbol = symbol or "?"
    
    print(f"📊 Position details:")
    print(f"   Entry price: ${entry_price:.6f}")
    print(f"   Position size: ${position_size_usd:.2f}")
    if trade_id:
        print(f"   Trade ID: {trade_id}")
    print()
    
    # Confirm before selling (unless skip_confirm is True)
    if not skip_confirm:
        response = input(f"⚠️  Are you sure you want to sell {symbol}? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            print("❌ Cancelled.")
            return False
    else:
        print(f"⚠️  Proceeding with sale of {symbol} (confirmation skipped)")
    
    # Sell the token
    print(f"\n💰 Selling {symbol}...")
    try:
        if chain.lower() == "solana":
            tx_hash = sell_token_solana(token_address, symbol)
        else:
            print(f"❌ Unsupported chain: {chain}")
            return False
        
        if not tx_hash:
            print("❌ Failed to sell token (no transaction hash returned)")
            return False
        
        print(f"✅ Sell transaction submitted: {tx_hash}")
        print(f"   Waiting for confirmation...")
        
        # Wait a moment for transaction to confirm
        import time
        time.sleep(3)
        
        # Get current price for PnL calculation
        from src.utils.utils import fetch_token_price_usd
        current_price = fetch_token_price_usd(token_address, chain) or entry_price
        
        # Calculate PnL
        if current_price and entry_price > 0:
            pnl_percent = ((current_price - entry_price) / entry_price) * 100
            pnl_usd = (pnl_percent / 100) * position_size_usd
        else:
            pnl_percent = 0
            pnl_usd = 0
        
        # Update performance tracker if trade_id exists
        if trade_id:
            try:
                performance_tracker.log_trade_exit(
                    trade_id,
                    current_price,
                    pnl_usd,
                    "manual_close",
                    additional_data={"sell_tx_hash": tx_hash}
                )
                print(f"✅ Performance tracker updated")
            except Exception as e:
                print(f"⚠️  Failed to update performance tracker: {e}")
                import traceback
                traceback.print_exc()
        
        # Write to trade_log.csv
        try:
            log_file = project_root / "data" / "trade_log.csv"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "token": token_address,
                "entry_price": entry_price,
                "exit_price": current_price,
                "pnl_pct": round(pnl_percent, 2),
                "reason": f"manual_close_pnl_{pnl_percent:.2f}%"
            }
            
            file_exists = log_file.exists()
            with open(log_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
            print(f"✅ Trade logged to trade_log.csv")
        except Exception as e:
            print(f"⚠️  Failed to write to trade_log.csv: {e}")
            import traceback
            traceback.print_exc()
        
        # Comprehensive cleanup: Remove from ALL databases
        try:
            # Remove from open_positions.json and hunter_state.db
            remove_position(position_key)
            print(f"✅ Position removed from open_positions.json and hunter_state.db")
            
            # Update performance_data.json to mark trade as closed
            try:
                perf_data = load_performance_data()
                trades = perf_data.get("trades", [])
                token_address_lower = token_address.lower()
                chain_id_lower = chain.lower()
                
                updated = False
                for trade in trades:
                    if (trade.get("status") in ["open", "manual_close"] and 
                        trade.get("address", "").lower() == token_address_lower and
                        trade.get("chain", "").lower() == chain_id_lower):
                        trade["status"] = "manual_close"
                        if not trade.get("exit_time"):
                            trade["exit_time"] = datetime.now().isoformat()
                        if not trade.get("exit_price"):
                            trade["exit_price"] = current_price
                        if not trade.get("pnl_usd"):
                            trade["pnl_usd"] = pnl_usd
                        if not trade.get("pnl_percent"):
                            trade["pnl_percent"] = pnl_percent
                        if not trade.get("sell_tx_hash"):
                            trade["sell_tx_hash"] = tx_hash
                        updated = True
                        print(f"✅ Marked trade {trade.get('id', '?')} as closed in performance_data.json")
                
                if updated:
                    replace_performance_data(perf_data)
                    print(f"✅ Performance data updated")
            except Exception as e:
                print(f"⚠️  Error updating performance_data.json: {e}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            print(f"⚠️  Error during cleanup: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n✅ Position closed successfully!")
        print(f"   Transaction: {tx_hash}")
        print(f"   Entry: ${entry_price:.6f} → Exit: ${current_price:.6f}")
        print(f"   PnL: ${pnl_usd:.2f} ({pnl_percent:.2f}%)")
        return True
        
    except Exception as e:
        print(f"❌ Error selling token: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manually close a position")
    parser.add_argument("token_address", help="Token address to close")
    parser.add_argument("--symbol", help="Token symbol (optional)")
    parser.add_argument("--chain", default="solana", help="Chain ID (default: solana)")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    
    args = parser.parse_args()
    
    success = close_position(args.token_address, args.symbol, args.chain, skip_confirm=args.yes)
    sys.exit(0 if success else 1)

