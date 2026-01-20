#!/usr/bin/env python3
"""
Remove a token from all databases and storage files.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.positions import remove_position, load_positions
from src.storage.price_memory import load_price_memory, save_price_memory
from src.storage.cooldown import load_cooldown_log, save_cooldown_log
from src.storage.swap_events import get_connection
from src.storage.intents import load_all_trade_intents, delete_trade_intent
from src.storage.blacklist import (
    load_blacklist,
    save_blacklist,
    load_failures,
    save_failures,
    load_reasons,
    save_reasons,
)
from src.storage.delist import load_delisted_state, save_delisted_state
from src.storage.performance import load_performance_data, replace_performance_data
from src.storage.risk import load_balance_cache, save_balance_cache


def remove_token_from_all_storage(token_address: str, symbol: str = None) -> None:
    """
    Remove a token from all storage locations.
    
    Args:
        token_address: Token address (case-insensitive)
        symbol: Optional symbol for display purposes
    """
    token_address_lower = token_address.lower()
    display_name = symbol or token_address[:8] + "..." + token_address[-8:]
    
    print(f"🗑️  Removing {display_name} ({token_address_lower}) from all storage...")
    removed_count = 0
    
    # 1. Remove from positions
    print("\n1. Checking positions...")
    positions = load_positions()
    found_in_positions = False
    for position_key, pos_data in list(positions.items()):
        if isinstance(pos_data, dict):
            pos_addr = (pos_data.get("address") or position_key).lower()
        else:
            pos_addr = position_key.lower()
        
        if pos_addr == token_address_lower:
            found_in_positions = True
            removed = remove_position(position_key, token_address)
            if removed:
                print(f"   ✅ Removed from positions: {position_key}")
                removed_count += 1
            else:
                print(f"   ⚠️  Failed to remove position: {position_key}")
    
    if not found_in_positions:
        print("   ℹ️  Not found in positions")
    
    # 2. Remove from price_memory
    print("\n2. Checking price_memory...")
    price_memory = load_price_memory()
    if token_address_lower in price_memory:
        del price_memory[token_address_lower]
        save_price_memory(price_memory)
        print(f"   ✅ Removed from price_memory")
        removed_count += 1
    else:
        print("   ℹ️  Not found in price_memory")
    
    # 3. Remove from cooldown
    print("\n3. Checking cooldown...")
    cooldown_log = load_cooldown_log()
    if token_address_lower in cooldown_log:
        del cooldown_log[token_address_lower]
        save_cooldown_log(cooldown_log)
        print(f"   ✅ Removed from cooldown")
        removed_count += 1
    else:
        print("   ℹ️  Not found in cooldown")
    
    # 4. Remove from swap_events (SQLite only)
    print("\n4. Checking swap_events...")
    from src.storage.db import get_connection
    with get_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM swap_events WHERE token_address = ?",
            (token_address_lower,)
        )
        deleted_swaps = cursor.rowcount
        if deleted_swaps > 0:
            print(f"   ✅ Removed {deleted_swaps} swap events")
            removed_count += 1
        else:
            print("   ℹ️  Not found in swap_events")
    
    # 5. Remove from trade_intents
    print("\n5. Checking trade_intents...")
    intents = load_all_trade_intents()
    intent_removed = False
    for intent_id, intent_data in list(intents.items()):
        if isinstance(intent_data, dict):
            intent_addr = (intent_data.get("token_address") or intent_data.get("address") or "").lower()
            if intent_addr == token_address_lower:
                delete_trade_intent(intent_id)
                print(f"   ✅ Removed trade intent: {intent_id}")
                removed_count += 1
                intent_removed = True
    
    if not intent_removed:
        print("   ℹ️  Not found in trade_intents")
    
    # 6. Remove from blacklist
    print("\n6. Checking blacklist...")
    blacklist = load_blacklist()
    if token_address_lower in blacklist:
        blacklist.remove(token_address_lower)
        save_blacklist(blacklist)
        print(f"   ✅ Removed from blacklist")
        removed_count += 1
    else:
        print("   ℹ️  Not found in blacklist")
    
    # 7. Remove from blacklist_failures
    print("\n7. Checking blacklist_failures...")
    failures = load_failures()
    if token_address_lower in failures:
        del failures[token_address_lower]
        save_failures(failures)
        print(f"   ✅ Removed from blacklist_failures")
        removed_count += 1
    else:
        print("   ℹ️  Not found in blacklist_failures")
    
    # 8. Remove from blacklist_reasons
    print("\n8. Checking blacklist_reasons...")
    reasons = load_reasons()
    if token_address_lower in reasons:
        del reasons[token_address_lower]
        save_reasons(reasons)
        print(f"   ✅ Removed from blacklist_reasons")
        removed_count += 1
    else:
        print("   ℹ️  Not found in blacklist_reasons")
    
    # 9. Remove from delisted_tokens
    print("\n9. Checking delisted_tokens...")
    delisted_state = load_delisted_state()
    tokens = delisted_state.get("delisted_tokens", [])
    if token_address_lower in tokens:
        tokens.remove(token_address_lower)
        delisted_state["delisted_tokens"] = tokens
        delisted_state["remaining_count"] = len(tokens)
        save_delisted_state(delisted_state)
        print(f"   ✅ Removed from delisted_tokens")
        removed_count += 1
    else:
        print("   ℹ️  Not found in delisted_tokens")
    
    # 10. Check balance_cache (may contain token-specific data)
    print("\n10. Checking balance_cache...")
    balance_cache = load_balance_cache()
    updated_balance_cache = False
    if isinstance(balance_cache, dict):
        # Check if token address is a key
        if token_address_lower in balance_cache:
            del balance_cache[token_address_lower]
            updated_balance_cache = True
        # Check nested structures
        for key, value in list(balance_cache.items()):
            if isinstance(value, dict) and token_address_lower in value:
                del value[token_address_lower]
                updated_balance_cache = True
    
    if updated_balance_cache:
        save_balance_cache(balance_cache)
        print(f"   ✅ Removed from balance_cache")
        removed_count += 1
    else:
        print("   ℹ️  Not found in balance_cache")
    
    # 11. Check performance_data.json (trade history - we'll filter trades)
    print("\n11. Checking performance_data...")
    performance_data = load_performance_data()
    trades = performance_data.get("trades", [])
    original_count = len(trades)
    if isinstance(trades, list):
        trades = [
            trade for trade in trades
            if isinstance(trade, dict) and 
            (trade.get("token_address") or trade.get("address") or "").lower() != token_address_lower
        ]
        if len(trades) < original_count:
            performance_data["trades"] = trades
            replace_performance_data(performance_data)
            removed_trades = original_count - len(trades)
            print(f"   ✅ Removed {removed_trades} trade(s) from performance_data")
            removed_count += 1
        else:
            print("   ℹ️  Not found in performance_data trades")
    else:
        print("   ℹ️  No trades in performance_data")
    
    print(f"\n✅ Complete! Removed {display_name} from {removed_count} storage location(s)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python remove_token_from_db.py <token_address> [symbol]")
        print("\nExample:")
        print("  python remove_token_from_db.py 9BB6NFEcjBCtnNLFko2FqVQBq8HHM13kCyYcdQbgpump Fartcoin")
        sys.exit(1)
    
    token_address = sys.argv[1]
    symbol = sys.argv[2] if len(sys.argv) > 2 else None
    
    remove_token_from_all_storage(token_address, symbol)
