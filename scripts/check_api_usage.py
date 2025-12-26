#!/usr/bin/env python3
"""
Check current API call usage
"""

import json
from pathlib import Path
from datetime import datetime

def main():
    tracker_file = Path("data/api_call_tracker.json")
    
    if not tracker_file.exists():
        print("No API call tracking data found.")
        return
    
    data = json.loads(tracker_file.read_text())
    helius_calls = data.get('helius', 0)
    coingecko_calls = data.get('coingecko', 0)
    coincap_calls = data.get('coincap', 0)
    last_reset = data.get('last_reset', 0)
    
    helius_max = 30000
    coingecko_max = 330
    coincap_max = 130
    
    helius_remaining = helius_max - helius_calls
    coingecko_remaining = coingecko_max - coingecko_calls
    coincap_remaining = coincap_max - coincap_calls
    
    print("=" * 60)
    print("API Call Usage Report")
    print("=" * 60)
    print(f"\nHelius API:")
    print(f"  Calls today: {helius_calls}/{helius_max}")
    print(f"  Remaining: {helius_remaining}/{helius_max}")
    print(f"  Usage: {(helius_calls/helius_max)*100:.2f}%")
    
    print(f"\nCoinGecko API:")
    print(f"  Calls today: {coingecko_calls}/{coingecko_max}")
    print(f"  Remaining: {coingecko_remaining}/{coingecko_max}")
    print(f"  Usage: {(coingecko_calls/coingecko_max)*100:.2f}%")
    
    print(f"\nCoinCap API:")
    print(f"  Calls today: {coincap_calls}/{coincap_max}")
    print(f"  Remaining: {coincap_remaining}/{coincap_max}")
    print(f"  Usage: {(coincap_calls/coincap_max)*100:.2f}%")
    
    print(f"\nLast reset: {datetime.fromtimestamp(last_reset)}")
    
    # Warnings
    if coingecko_remaining < 50:
        print("\n⚠️  WARNING: Low CoinGecko API calls remaining!")
    elif coingecko_remaining < 100:
        print("\n⚠️  CAUTION: Less than 100 CoinGecko calls remaining")
    
    if coincap_remaining < 20:
        print("\n⚠️  WARNING: Low CoinCap API calls remaining!")
    elif coincap_remaining < 50:
        print("\n⚠️  CAUTION: Less than 50 CoinCap calls remaining")

if __name__ == "__main__":
    main()

