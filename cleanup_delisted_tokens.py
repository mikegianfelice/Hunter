#!/usr/bin/env python3
"""
Clean up delisted tokens list by checking which tokens are still actively trading
"""

import json
import requests
import time
from datetime import datetime

def check_token_status(address, symbol, chain_id="ethereum"):
    """Check if a token is still actively trading"""
    try:
        # Use DexScreener API to check current status
        if chain_id == "solana":
            # For Solana tokens, check if they have recent activity
            url = f"https://api.dexscreener.com/latest/dex/tokens/{address}"
        else:
            # For Ethereum and other chains
            url = f"https://api.dexscreener.com/latest/dex/tokens/{address}"
        
        headers = {"User-Agent": "Mozilla/5.0 (bot)"}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            pairs = data.get("pairs", [])
            
            if pairs:
                # Check if any pair has recent activity
                for pair in pairs:
                    volume_24h = float(pair.get("volume", {}).get("h24", 0))
                    liquidity_usd = float(pair.get("liquidity", {}).get("usd", 0))
                    
                    # If token has volume > $100 and liquidity > $500, consider it active
                    if volume_24h > 100 and liquidity_usd > 500:
                        return True, f"Active: vol=${volume_24h:.2f}, liq=${liquidity_usd:.2f}"
                
                return False, "No active pairs found"
            else:
                return False, "No pairs found"
        else:
            return False, f"API error: {response.status_code}"
            
    except Exception as e:
        return False, f"Error: {str(e)}"

def cleanup_delisted_tokens():
    """Clean up the delisted tokens list"""
    
    # Load current delisted tokens
    try:
        with open("delisted_tokens.json", "r") as f:
            data = json.load(f) or {}
    except FileNotFoundError:
        print("❌ delisted_tokens.json not found")
        return
    
    delisted_tokens = data.get("delisted_tokens", [])
    failure_counts = data.get("failure_counts", {})
    
    print(f"🔍 Checking {len(delisted_tokens)} delisted tokens...")
    
    # Create backup
    backup_data = {
        "backup_created": datetime.utcnow().isoformat(),
        "original_delisted_tokens": delisted_tokens.copy(),
        "original_failure_counts": failure_counts.copy()
    }
    
    with open("delisted_tokens_backup.json", "w") as f:
        json.dump(backup_data, f, indent=2)
    
    print("💾 Created backup: delisted_tokens_backup.json")
    
    # Check each token
    still_delisted = []
    now_active = []
    
    for i, address in enumerate(delisted_tokens):
        print(f"\n[{i+1}/{len(delisted_tokens)}] Checking {address[:20]}...")
        
        # Determine chain based on address length
        if len(address) in [43, 44]:
            chain_id = "solana"
        else:
            chain_id = "ethereum"
        
        is_active, status = check_token_status(address, "UNKNOWN", chain_id)
        
        if is_active:
            now_active.append(address)
            print(f"✅ {address[:20]}... is ACTIVE: {status}")
        else:
            still_delisted.append(address)
            print(f"❌ {address[:20]}... still DELISTED: {status}")
        
        # Add delay to avoid rate limiting
        time.sleep(1)
    
    # Update delisted tokens list
    updated_data = {
        "failure_counts": failure_counts,
        "delisted_tokens": still_delisted,
        "cleaned_at": datetime.utcnow().isoformat(),
        "removed_count": len(now_active),
        "remaining_count": len(still_delisted)
    }
    
    with open("delisted_tokens.json", "w") as f:
        json.dump(updated_data, f, indent=2)
    
    # Print summary
    print(f"\n🎯 CLEANUP SUMMARY:")
    print(f"• Original delisted tokens: {len(delisted_tokens)}")
    print(f"• Still delisted: {len(still_delisted)}")
    print(f"• Now active (removed): {len(now_active)}")
    print(f"• Reduction: {len(now_active)} tokens ({len(now_active)/len(delisted_tokens)*100:.1f}%)")
    
    if now_active:
        print(f"\n📋 REMOVED TOKENS (now active):")
        for addr in now_active:
            print(f"  - {addr}")
    
    print(f"\n✅ Updated delisted_tokens.json")
    print(f"💾 Backup saved as delisted_tokens_backup.json")

def quick_cleanup():
    """Quick cleanup - remove tokens that are likely false positives"""
    
    try:
        with open("delisted_tokens.json", "r") as f:
            data = json.load(f) or {}
    except FileNotFoundError:
        print("❌ delisted_tokens.json not found")
        return
    
    delisted_tokens = data.get("delisted_tokens", [])
    failure_counts = data.get("failure_counts", {})
    
    print(f"🔍 Quick cleanup of {len(delisted_tokens)} delisted tokens...")
    
    # Remove tokens that were added recently or have low failure counts
    still_delisted = []
    removed = []
    
    for address in delisted_tokens:
        failure_count = failure_counts.get(address, 0)
        
        # Keep tokens with high failure counts (likely actually delisted)
        if failure_count >= 3:
            still_delisted.append(address)
        else:
            removed.append(address)
            print(f"🔄 Removing {address[:20]}... (low failure count: {failure_count})")
    
    # Update file
    updated_data = {
        "failure_counts": failure_counts,
        "delisted_tokens": still_delisted,
        "quick_cleaned_at": datetime.utcnow().isoformat(),
        "removed_count": len(removed),
        "remaining_count": len(still_delisted)
    }
    
    with open("delisted_tokens.json", "w") as f:
        json.dump(updated_data, f, indent=2)
    
    print(f"\n🎯 QUICK CLEANUP SUMMARY:")
    print(f"• Original: {len(delisted_tokens)}")
    print(f"• Removed: {len(removed)}")
    print(f"• Remaining: {len(still_delisted)}")
    print(f"✅ Updated delisted_tokens.json")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        print("🚀 Running quick cleanup...")
        quick_cleanup()
    else:
        print("🚀 Running full cleanup (this may take a while)...")
        print("💡 Use 'python3 cleanup_delisted_tokens.py quick' for faster cleanup")
        cleanup_delisted_tokens()
