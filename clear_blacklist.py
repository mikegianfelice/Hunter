#!/usr/bin/env python3
"""
Clear blacklist and reset bot state to allow new trades
"""

import json
import os

def clear_blacklist():
    """Clear all blacklisted tokens"""
    
    # Clear blacklist.json if it exists
    if os.path.exists("blacklist.json"):
        os.remove("blacklist.json")
        print("🗑️ Removed blacklist.json")
    
    # Reset delisted_tokens.json to empty state
    empty_delisted = {
        "failure_counts": {},
        "delisted_tokens": [],
        "quick_cleaned_at": None,
        "removed_count": 0,
        "remaining_count": 0
    }
    
    with open("delisted_tokens.json", "w") as f:
        json.dump(empty_delisted, f, indent=2)
    print("🗑️ Reset delisted_tokens.json")
    
    # Clear cooldown log
    if os.path.exists("cooldown_log.json"):
        os.remove("cooldown_log.json")
        print("🗑️ Removed cooldown_log.json")
    
    # Clear open positions
    with open("open_positions.json", "w") as f:
        json.dump({}, f)
    print("🗑️ Reset open_positions.json")
    
    print("\n✅ Blacklist cleared! Bot should now be able to find new trading opportunities.")

if __name__ == "__main__":
    clear_blacklist()
