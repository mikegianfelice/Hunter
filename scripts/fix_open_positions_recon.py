#!/usr/bin/env python3
"""
Fix open_positions.json by:
1. Closing positions that are not in wallet (like PENGU)
2. Adding positions that are in wallet but not tracked (like ZEREBRO)
3. Using Helius to get accurate balances and transaction data
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.secrets import HELIUS_API_KEY, SOLANA_WALLET_ADDRESS
from src.utils.helius_client import HeliusClient
from src.storage.positions import load_positions as load_positions_store, replace_positions
from src.storage.performance import load_performance_data
from src.core.performance_tracker import performance_tracker

BALANCE_EPSILON = 1e-9
USDC_MINT = "epjfwdd5aufqssqem2qn1xzybapc8g4weggkzwytdt1v"
MIN_POSITION_SIZE_USD = 1.0  # Only track positions >= $1 USD

# Excluded tokens (native/gas tokens)
EXCLUDED_MINTS = {
    "so11111111111111111111111111111111111111112",  # SOL / wSOL
    USDC_MINT,  # USDC
    "es9vmfrzacermjfrf4h2fyd4kconky11mcce8benwnyb",  # USDT
    # ETH is not on Solana, but we exclude it for other chains
}

def get_token_symbol_from_trades(mint: str) -> str:
    """Try to get token symbol from performance data or trade log"""
    mint_lower = mint.lower()
    
    # Check performance tracker
    for trade in performance_tracker.trades:
        if (trade.get("address") or "").lower() == mint_lower:
            symbol = trade.get("symbol")
            if symbol:
                return symbol
    
    # Check performance data file
    try:
        perf_data = load_performance_data()
        trades = perf_data.get("trades", [])
        for trade in trades:
            if (trade.get("address") or "").lower() == mint_lower:
                symbol = trade.get("symbol")
                if symbol:
                    return symbol
    except Exception:
        pass
    
    # Return shortened mint as fallback
    return f"{mint[:8]}...{mint[-8:]}"

def get_token_price(mint: str) -> float:
    """Get current token price"""
    try:
        from src.execution.solana_executor import get_token_price_usd
        price = get_token_price_usd(mint)
        if price and price > 0:
            return price
    except Exception:
        pass
    
    try:
        from src.utils.utils import fetch_token_price_usd
        price = fetch_token_price_usd(mint)
        if price and price > 0:
            return price
    except Exception:
        pass
    
    return 0.0

def main():
    print("🔧 Fixing open_positions.json reconciliation issues...")
    print("=" * 70)
    
    if not HELIUS_API_KEY or not SOLANA_WALLET_ADDRESS:
        print("❌ Missing HELIUS_API_KEY or SOLANA_WALLET_ADDRESS")
        return
    
    # Get actual wallet balances
    print("\n📊 Fetching wallet balances from Helius...")
    client = HeliusClient(HELIUS_API_KEY)
    balances_data = client.get_address_balances(SOLANA_WALLET_ADDRESS)
    tokens = balances_data.get("tokens", [])
    
    # Build balance index (mint -> amount)
    wallet_balances = {}
    for token in tokens:
        mint = (token.get("mint") or "").lower()
        amount_raw = token.get("amount") or 0
        decimals = token.get("decimals") or 0
        if mint:
            amount = float(amount_raw) / (10 ** decimals) if decimals else float(amount_raw)
            if amount > BALANCE_EPSILON:
                wallet_balances[mint] = amount
    
    print(f"   Found {len(wallet_balances)} token(s) with balance > 0")
    
    # Load open positions
    open_positions = load_positions_store()
    print(f"\n📋 Current open_positions.json has {len(open_positions)} position(s)")
    
    # Step 1: Close positions not in wallet or too small
    print("\n" + "=" * 70)
    print("STEP 1: Closing positions not in wallet or too small")
    print("=" * 70)
    
    positions_to_remove = []
    for key, pos in open_positions.items():
        address = (pos.get("address") or key).lower()
        balance = wallet_balances.get(address, 0.0)
        symbol = pos.get("symbol", "?")
        position_size = pos.get("position_size_usd", 0.0)
        
        # Remove if no balance OR if position size is too small OR if excluded token
        should_remove = False
        reason = ""
        
        if address in EXCLUDED_MINTS:
            should_remove = True
            reason = "excluded token (USDC/SOL/ETH)"
        elif balance <= BALANCE_EPSILON:
            should_remove = True
            reason = "not in wallet"
        elif position_size < MIN_POSITION_SIZE_USD:
            should_remove = True
            reason = f"position size too small (${position_size:.6f} < ${MIN_POSITION_SIZE_USD:.2f})"
        
        if should_remove:
            print(f"❌ Closing {symbol} ({address[:8]}...{address[-8:]}) - {reason}")
            positions_to_remove.append(key)
            
            # Try to find and close the trade in performance tracker
            mint_lower = address.lower()
            for trade in performance_tracker.trades:
                if (trade.get("address") or "").lower() == mint_lower:
                    trade_id = trade.get("id")
                    status = trade.get("status", "open").lower()
                    if status == "open":
                        entry_price = trade.get("entry_price", 0.0) or 0.0
                        exit_price = entry_price  # Use entry price as fallback
                        
                        # Try to find sell transaction
                        try:
                            from src.core.helius_reconciliation import _HeliusContext
                            context = _HeliusContext(client, SOLANA_WALLET_ADDRESS, limit=200)
                            exit_tx = context.find_matching_transaction(trade, mint_lower, direction="sell")
                            if exit_tx:
                                transfers = exit_tx.get("tokenTransfers") or []
                                from src.core.helius_reconciliation import _aggregate_token_amount
                                usdc_received = _aggregate_token_amount(
                                    transfers, SOLANA_WALLET_ADDRESS, USDC_MINT, incoming=True
                                )
                                tokens_sold = _aggregate_token_amount(
                                    transfers, SOLANA_WALLET_ADDRESS, mint_lower, incoming=False
                                )
                                if tokens_sold > BALANCE_EPSILON and usdc_received > 0:
                                    exit_price = usdc_received / tokens_sold
                        except Exception as e:
                            print(f"   ⚠️  Could not find exit transaction: {e}")
                        
                        print(f"   Closing trade {trade_id} at exit price ${exit_price:.6f}")
                        performance_tracker.log_trade_exit(
                            trade_id,
                            exit_price,
                            0.0,
                            status="manual_close",
                        )
                        break
    
    # Remove closed positions
    for key in positions_to_remove:
        open_positions.pop(key, None)
    
    if positions_to_remove:
        print(f"\n✅ Closed {len(positions_to_remove)} position(s)")
    else:
        print("✅ No positions to close")
    
    # Step 2: Add missing positions from wallet
    print("\n" + "=" * 70)
    print("STEP 2: Adding missing positions from wallet")
    print("=" * 70)
    
    # Get existing position addresses
    existing_addresses = set()
    for key, pos in open_positions.items():
        address = (pos.get("address") or key).lower()
        existing_addresses.add(address)
    
    positions_added = 0
    positions_skipped = 0
    for mint, balance in wallet_balances.items():
        # Skip excluded tokens
        if mint in EXCLUDED_MINTS:
            continue
        if mint in existing_addresses:
            continue
        
        # Get symbol and price
        symbol = get_token_symbol_from_trades(mint)
        price = get_token_price(mint)
        value_usd = balance * price if price > 0 else 0.0
        
        # Skip if position size is too small
        if value_usd < MIN_POSITION_SIZE_USD:
            positions_skipped += 1
            continue
        
        print(f"➕ Adding {symbol} ({mint[:8]}...{mint[-8:]})")
        print(f"   Balance: {balance:.9f}, Price: ${price:.6f}, Value: ${value_usd:.2f}")
        
        # Create position entry
        position_key = mint.lower()
        open_positions[position_key] = {
            "address": mint,
            "symbol": symbol,
            "chain_id": "solana",
            "timestamp": datetime.now().isoformat(),
            "entry_price": price if price > 0 else 0.0,
            "position_size_usd": value_usd,
        }
        positions_added += 1
    
    if positions_added:
        print(f"\n✅ Added {positions_added} position(s)")
    else:
        print("✅ No new positions to add")
    
    if positions_skipped:
        print(f"⏭️  Skipped {positions_skipped} position(s) with value < ${MIN_POSITION_SIZE_USD:.2f}")
    
    # Step 3: Save updated positions
    print("\n" + "=" * 70)
    print("STEP 3: Saving updated open_positions.json")
    print("=" * 70)
    
    if positions_to_remove or positions_added:
        replace_positions(open_positions)
        print(f"✅ Saved {len(open_positions)} position(s) to open_positions.json")
        print(f"\n📊 Summary:")
        print(f"   • Closed: {len(positions_to_remove)}")
        print(f"   • Added: {positions_added}")
        print(f"   • Total positions: {len(open_positions)}")
    else:
        print("✅ No changes needed")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
