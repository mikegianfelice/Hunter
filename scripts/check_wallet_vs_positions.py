#!/usr/bin/env python3
"""
Check actual wallet balances vs open_positions.json to find discrepancies.
"""

import sys
import json
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.secrets import HELIUS_API_KEY, SOLANA_WALLET_ADDRESS
from src.utils.helius_client import HeliusClient
from src.storage.positions import load_positions as load_positions_store

def main():
    print("🔍 Checking wallet balances vs open_positions.json...")
    print("=" * 70)
    
    if not HELIUS_API_KEY or not SOLANA_WALLET_ADDRESS:
        print("❌ Missing HELIUS_API_KEY or SOLANA_WALLET_ADDRESS")
        return
    
    # Get actual wallet balances
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
            if amount > 1e-9:  # Only non-zero balances
                wallet_balances[mint] = amount
    
    print(f"\n📊 Wallet has {len(wallet_balances)} token(s) with balance > 0:")
    for mint, balance in sorted(wallet_balances.items()):
        print(f"   • {mint[:8]}...{mint[-8:]}: {balance:.9f}")
    
    # Load open positions
    open_positions = load_positions_store()
    print(f"\n📋 open_positions.json has {len(open_positions)} position(s):")
    for key, pos in open_positions.items():
        address = (pos.get("address") or key).lower()
        symbol = pos.get("symbol", "?")
        print(f"   • {symbol} ({address[:8]}...{address[-8:]}): {address}")
    
    # Compare
    print("\n" + "=" * 70)
    print("🔍 COMPARISON:")
    print("=" * 70)
    
    # Find positions in open_positions.json but NOT in wallet or too small
    missing_from_wallet = []
    for key, pos in open_positions.items():
        address = (pos.get("address") or key).lower()
        balance = wallet_balances.get(address, 0.0)
        position_size = pos.get("position_size_usd", 0.0)
        
        # Should be removed if: no balance, too small, or excluded token
        if address in EXCLUDED_MINTS:
            missing_from_wallet.append((key, pos, address, "excluded token"))
        elif balance <= BALANCE_EPSILON:
            missing_from_wallet.append((key, pos, address, "not in wallet"))
        elif position_size < MIN_POSITION_SIZE_USD:
            missing_from_wallet.append((key, pos, address, f"too small (${position_size:.6f})"))
    
    # Find tokens in wallet but NOT in open_positions.json
    # (excluding small positions and native tokens)
    missing_from_positions = []
    position_addresses = set()
    for key, pos in open_positions.items():
        address = (pos.get("address") or key).lower()
        position_addresses.add(address)
    
    for mint, balance in wallet_balances.items():
        # Skip excluded tokens
        if mint in EXCLUDED_MINTS:
            continue
        # Skip if already tracked
        if mint in position_addresses:
            continue
        # Only report if it would be worth tracking (we'll check price later)
        missing_from_positions.append((mint, balance))
    
    # Report discrepancies
    if missing_from_wallet:
        print("\n❌ Positions in open_positions.json that should be closed:")
        for item in missing_from_wallet:
            if len(item) == 4:
                key, pos, address, reason = item
            else:
                key, pos, address = item
                reason = "not in wallet"
            symbol = pos.get("symbol", "?")
            print(f"   • {symbol} ({address[:8]}...{address[-8:]}) - {reason}")
            if reason != "excluded token":
                print(f"     Entry: ${pos.get('entry_price', 0):.6f}, Size: ${pos.get('position_size_usd', 0):.2f}")
    
    if missing_from_positions:
        print("\n⚠️  Tokens in wallet but NOT in open_positions.json (should be added if >= $1):")
        for mint, balance in missing_from_positions:
            print(f"   • {mint[:8]}...{mint[-8:]}: {balance:.9f}")
    
    if not missing_from_wallet and not missing_from_positions:
        print("\n✅ No discrepancies found - wallet and open_positions.json match!")
    else:
        print(f"\n📝 Summary:")
        print(f"   • {len(missing_from_wallet)} position(s) need to be closed")
        print(f"   • {len(missing_from_positions)} token(s) need to be added")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
