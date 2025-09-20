#!/usr/bin/env python3
"""
Quick position verification script
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wallet_balance_checker import WalletBalanceChecker

def main():
    checker = WalletBalanceChecker()
    
    print("🔍 Quick Position Verification")
    print("=" * 40)
    
    # Get wallet balances
    wallet_balances = checker.get_solana_token_balances()
    tracked_positions = checker.load_open_positions()
    
    print(f"📊 Wallet: {len(wallet_balances)} tokens")
    print(f"📋 Bot tracking: {len(tracked_positions)} positions")
    
    # Show detailed breakdown
    print("\n📈 Detailed Holdings:")
    total_value = 0
    for balance in wallet_balances:
        print(f"   • {balance.symbol}: {balance.amount:,.6f} tokens (${balance.value_usd:.2f})")
        total_value += balance.value_usd
    
    print(f"\n💰 Total Value: ${total_value:.2f}")
    
    # Check for mismatches
    wallet_mints = {balance.mint for balance in wallet_balances}
    tracked_mints = set(tracked_positions.keys())
    
    missing = tracked_mints - wallet_mints
    extra = wallet_mints - tracked_mints
    
    if missing:
        print(f"\n❌ Tracked but missing from wallet ({len(missing)}):")
        for mint in missing:
            symbol = tracked_positions[mint].get("symbol", "UNKNOWN")
            print(f"   • {symbol} ({mint[:8]}...{mint[-8:]})")
    
    if extra:
        print(f"\n➕ In wallet but not tracked ({len(extra)}):")
        for mint in extra:
            balance = next((b for b in wallet_balances if b.mint == mint), None)
            if balance:
                print(f"   • {balance.symbol} ({balance.amount:.6f} tokens, ${balance.value_usd:.2f})")
    
    if not missing and not extra:
        print("\n✅ Perfect match! All tracked positions exist in wallet.")
    
    print("\n" + "=" * 40)

if __name__ == "__main__":
    main()
