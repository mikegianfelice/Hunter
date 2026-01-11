#!/usr/bin/env python3
"""
Show Priority Tokens
Extracts and displays priority tokens from historical trades without requiring swap indexer.
"""

import sys
import json
from pathlib import Path
from collections import Counter
from datetime import datetime, timedelta
from typing import List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def load_helius_cache():
    """Load Helius wallet value cache (source of truth for trades)"""
    cache_file = PROJECT_ROOT / 'data' / 'helius_wallet_value_cache.json'
    if not cache_file.exists():
        return {"completed_trades": []}
    
    try:
        with open(cache_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading Helius cache: {e}")
        return {"completed_trades": []}

def get_established_tokens():
    """Get established tokens from token_scraper"""
    try:
        # Try to read ESTABLISHED_TOKENS directly from the file to avoid import issues
        token_scraper_file = PROJECT_ROOT / 'src' / 'utils' / 'token_scraper.py'
        if token_scraper_file.exists():
            content = token_scraper_file.read_text()
            # Extract ESTABLISHED_TOKENS dict using simple parsing
            # This is a fallback if import fails
            tokens = []
            in_dict = False
            for line in content.split('\n'):
                if 'ESTABLISHED_TOKENS = {' in line:
                    in_dict = True
                    continue
                if in_dict:
                    if line.strip() == '}':
                        break
                    # Extract address from lines like: "address": "SYMBOL",
                    if '"' in line and ':' in line:
                        parts = line.split(':')
                        if len(parts) >= 1:
                            addr = parts[0].strip().strip('"').strip("'")
                            if addr and len(addr) > 10:  # Valid address length
                                tokens.append(addr)
            return tokens
    except Exception:
        pass
    return []

def get_top_tokens_from_trades(
    days: int = 30,
    min_trades: int = 1,
    top_n: int = 50,
    chain_filter: str = "solana"
) -> List[Tuple[str, int, str]]:
    """Extract top tokens from Helius cache (source of truth)"""
    cache_data = load_helius_cache()
    completed_trades = cache_data.get('completed_trades', [])
    
    if not completed_trades:
        return []
    
    # Filter by date range (use UTC for timezone-aware comparison)
    from datetime import timezone
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    
    # Count token frequency
    token_counts = Counter()
    
    for trade in completed_trades:
        # Get buy_time for date filtering
        buy_time_str = trade.get('buy_time', '')
        if not buy_time_str:
            continue
        
        # Parse datetime (format: "2025-11-17 13:56:29+00:00")
        try:
            if 'T' in buy_time_str:
                buy_time = datetime.fromisoformat(buy_time_str.replace('Z', '+00:00'))
            else:
                buy_time = datetime.strptime(buy_time_str, "%Y-%m-%d %H:%M:%S%z")
            
            if buy_time < cutoff_date:
                continue
        except (ValueError, TypeError):
            continue
        
        # Get mint address (token address)
        mint = trade.get('mint', '').strip()
        if mint:
            token_counts[mint.lower()] += 1
    
    # Filter by minimum trades and get top N
    filtered = [
        (addr, count, 'UNKNOWN')  # Symbol not available in Helius cache
        for addr, count in token_counts.items()
        if count >= min_trades
    ]
    filtered.sort(key=lambda x: x[1], reverse=True)
    
    return filtered[:top_n]

def main():
    """Show priority tokens"""
    print("=" * 60)
    print("📊 PRIORITY TOKENS ANALYSIS")
    print("=" * 60)
    print()
    
    # Get config defaults
    from_trades_days = 30
    from_trades_min = 1
    from_trades_top_n = 50
    chain_filter = "solana"
    
    print(f"Configuration:")
    print(f"  - Look back: {from_trades_days} days")
    print(f"  - Minimum trades: {from_trades_min}")
    print(f"  - Top N: {from_trades_top_n}")
    print(f"  - Chain filter: {chain_filter}")
    print()
    
    # Get top tokens from trades
    print("📈 Top tokens from historical trades:")
    top_traded = get_top_tokens_from_trades(
        days=from_trades_days,
        min_trades=from_trades_min,
        top_n=from_trades_top_n,
        chain_filter=chain_filter
    )
    
    if top_traded:
        print(f"\nFound {len(top_traded)} tokens:\n")
        print(f"{'Rank':<6} {'Symbol':<12} {'Trades':<8} {'Address'}")
        print("-" * 60)
        for idx, (addr, count, symbol) in enumerate(top_traded, 1):
            addr_display = f"{addr[:8]}...{addr[-8:]}" if len(addr) > 16 else addr
            print(f"{idx:<6} {symbol:<12} {count:<8} {addr_display}")
    else:
        print("  No tokens found in historical trades")
    
    # Get established tokens (Solana only)
    print("\n" + "=" * 60)
    print("🏆 Established Solana tokens:")
    
    # Known Solana established tokens
    solana_established = [
        "So11111111111111111111111111111111111111112",  # SOL
        "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",  # BONK
        "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So",   # mSOL
    ]
    
    if solana_established:
        print(f"\nFound {len(solana_established)} Solana established tokens:\n")
        symbols = {
            "So11111111111111111111111111111111111111112": "SOL",
            "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263": "BONK",
            "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So": "mSOL",
        }
        for addr in solana_established:
            symbol = symbols.get(addr, "UNKNOWN")
            addr_display = f"{addr[:8]}...{addr[-8:]}" if len(addr) > 16 else addr
            print(f"  {symbol:<8} {addr_display}")
    else:
        print("  No established tokens found")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Summary:")
    priority_set = set(addr.lower() for addr, _, _ in top_traded)
    priority_set.update(addr.lower() for addr in solana_established)
    
    print(f"  - Top traded tokens: {len(top_traded)}")
    print(f"  - Established Solana tokens: {len(solana_established)}")
    print(f"  - Total unique priority tokens: {len(priority_set)}")
    print()

if __name__ == "__main__":
    main()
