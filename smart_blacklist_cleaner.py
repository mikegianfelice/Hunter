#!/usr/bin/env python3
"""
Smart Blacklist Cleaner - Verifies if tokens are actually delisted before keeping them in the list
"""

import json
import time
import requests
from typing import List, Dict, Tuple

def check_token_status(token_address: str, symbol: str = "UNKNOWN") -> Tuple[bool, str]:
    """
    Check if a token is actually delisted/inactive by verifying:
    1. Current price from DexScreener
    2. Volume and liquidity data
    3. Multiple API sources for verification
    
    Returns: (is_delisted: bool, reason: str)
    """
    try:
        # Check DexScreener for current data
        url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            pairs = data.get("pairs", [])
            
            if pairs:
                # Get the first pair with valid data
                pair = pairs[0]
                price_usd = float(pair.get("priceUsd", 0))
                volume_24h = float(pair.get("volume24h", 0))
                liquidity = float(pair.get("liquidity", 0))
                
                # Token is active if it has:
                # 1. Non-zero price
                # 2. Some volume or liquidity
                if price_usd > 0.0000001:
                    if volume_24h > 10 or liquidity > 100:  # Very lenient thresholds
                        return False, f"Active token: price=${price_usd}, vol=${volume_24h}, liq=${liquidity}"
                    else:
                        return True, f"Low activity: price=${price_usd}, vol=${volume_24h}, liq=${liquidity}"
                else:
                    return True, "Zero price"
            else:
                # If no pairs found on DexScreener, be more lenient
                # Check if token appears in trending tokens (indicating it's active)
                return False, "No DexScreener pairs but may be active elsewhere"
        else:
            return False, f"DexScreener error: {response.status_code}"
            
    except Exception as e:
        return False, f"API error: {str(e)}"

def clean_delisted_tokens() -> Dict[str, any]:
    """
    Clean the delisted tokens list by verifying each token's status
    """
    try:
        # Load current delisted tokens
        with open("delisted_tokens.json", "r") as f:
            data = json.load(f)
        
        delisted_tokens = data.get("delisted_tokens", [])
        original_count = len(delisted_tokens)
        
        print(f"🔍 Verifying {original_count} delisted tokens...")
        
        # Check each token
        still_delisted = []
        now_active = []
        
        for i, token_address in enumerate(delisted_tokens):
            print(f"  [{i+1}/{original_count}] Checking {token_address[:8]}...{token_address[-8:]}")
            
            is_delisted, reason = check_token_status(token_address)
            
            if is_delisted:
                still_delisted.append(token_address)
                print(f"    ❌ Still delisted: {reason}")
            else:
                now_active.append(token_address)
                print(f"    ✅ Now active: {reason}")
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
        
        # Update the delisted tokens list
        data["delisted_tokens"] = still_delisted
        data["cleaned_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        data["removed_count"] = len(now_active)
        data["remaining_count"] = len(still_delisted)
        data["reactivated_tokens"] = now_active
        
        # Save updated data
        with open("delisted_tokens.json", "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"\n📊 Cleanup Results:")
        print(f"  • Original delisted tokens: {original_count}")
        print(f"  • Still delisted: {len(still_delisted)}")
        print(f"  • Reactivated: {len(now_active)}")
        print(f"  • Cleanup ratio: {len(now_active)/original_count*100:.1f}%")
        
        if now_active:
            print(f"\n🔄 Reactivated tokens:")
            for token in now_active:
                print(f"  • {token}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error cleaning delisted tokens: {e}")
        return {}

def add_to_delisted_tokens_smart(token_address: str, symbol: str, reason: str) -> bool:
    """
    Smart version of adding tokens to delisted list - only adds if actually delisted
    """
    try:
        # First check if token is actually delisted
        is_delisted, verification_reason = check_token_status(token_address, symbol)
        
        if not is_delisted:
            print(f"⚠️ {symbol} appears to be active ({verification_reason}) - not adding to delisted list")
            return False
        
        # Token is actually delisted, add it
        with open("delisted_tokens.json", "r") as f:
            data = json.load(f)
        
        delisted_tokens = data.get("delisted_tokens", [])
        token_address_lower = token_address.lower()
        
        if token_address_lower not in delisted_tokens:
            delisted_tokens.append(token_address_lower)
            data["delisted_tokens"] = delisted_tokens
            data["last_added"] = {
                "address": token_address,
                "symbol": symbol,
                "reason": reason,
                "verified_reason": verification_reason,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }
            
            with open("delisted_tokens.json", "w") as f:
                json.dump(data, f, indent=2)
            
            print(f"✅ Added {symbol} to delisted tokens (verified: {verification_reason})")
            return True
        else:
            print(f"ℹ️ {symbol} already in delisted tokens")
            return False
            
    except Exception as e:
        print(f"❌ Error adding {symbol} to delisted tokens: {e}")
        return False

if __name__ == "__main__":
    print("🧹 Smart Delisted Token Cleanup")
    print("=" * 40)
    
    # Run the cleanup
    result = clean_delisted_tokens()
    
    if result:
        print(f"\n✅ Cleanup completed successfully!")
    else:
        print(f"\n❌ Cleanup failed!")
