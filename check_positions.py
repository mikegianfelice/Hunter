import json

def _fetch_token_price_multi_chain(token_address: str) -> float:
    """
    Fetch token price based on chain type.
    For now, we'll try to detect Solana vs Ethereum tokens.
    """
    try:
        # Try Solana price first (if it looks like a Solana address)
        if len(token_address) == 44:  # Solana addresses are 44 chars
            try:
                from solana_executor import get_token_price_usd
                price = get_token_price_usd(token_address)
                if price and price > 0:
                    print(f"🔗 Fetched Solana price for {token_address[:8]}...{token_address[-8:]}: ${price:.6f}")
                    return price
            except Exception as e:
                print(f"⚠️ Solana price fetch failed: {e}")
        
        # Fallback to Ethereum price fetching
        from utils import fetch_token_price_usd
        price = fetch_token_price_usd(token_address)
        if price and price > 0:
            print(f"🔗 Fetched Ethereum price for {token_address[:8]}...{token_address[-8:]}: ${price:.6f}")
            return price
            
        return None
    except Exception as e:
        print(f"⚠️ Price fetch failed for {token_address}: {e}")
        return None

# Load positions
with open('open_positions.json', 'r') as f:
    positions = json.load(f)

print('🔍 Current Position Analysis:')
print('=' * 50)

for addr, entry_price in positions.items():
    try:
        current_price = _fetch_token_price_multi_chain(addr)
        if current_price:
            gain_pct = ((current_price - entry_price) / entry_price) * 100
            print(f'Token: {addr[:8]}...{addr[-8:]}')
            print(f'  Entry: ${entry_price:.6f}')
            print(f'  Current: ${current_price:.6f}')
            print(f'  PnL: {gain_pct:.2f}%')
            
            # Check sell conditions
            if gain_pct >= 50:
                print(f'  🎯 TAKE PROFIT HIT! (+{gain_pct:.2f}% >= 50%)')
            elif gain_pct <= -25:
                print(f'  🛑 STOP LOSS HIT! ({gain_pct:.2f}% <= -25%)')
            else:
                print(f'  ⏳ Holding...')
            print()
        else:
            print(f'❌ Could not fetch price for {addr[:8]}...{addr[-8:]}')
    except Exception as e:
        print(f'⚠️ Error checking {addr[:8]}...{addr[-8:]}: {e}')
