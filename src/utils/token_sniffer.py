import requests
import time

def _get_dexscreener_pairs(token_address: str):
    """Get trading pairs from DexScreener for a token"""
    try:
        url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
        response = requests.get(url, timeout=12)
        if response.status_code == 200:
            data = response.json()
            return data.get("pairs") or []
    except Exception as e:
        print(f"⚠️ DexScreener API error for {token_address[:8]}...: {e}")
    return []

def check_token_safety(token_address, chain_id="ethereum"):
    """
    Check token safety using real market data from DexScreener.
    Analyzes liquidity, trading activity, and pair age to assess safety.
    No assumptions - uses real market data only.
    """
    try:
        # Get real trading pairs from DexScreener
        pairs = _get_dexscreener_pairs(token_address)
        
        if not pairs:
            print(f"❌ No pairs found on DexScreener for {token_address[:8]}...{token_address[-8:]}")
            return False
        
        # Find the pair with highest liquidity
        richest = max(pairs, key=lambda p: float((p.get("liquidity") or {}).get("usd") or 0))
        
        liquidity_usd = float((richest.get("liquidity") or {}).get("usd") or 0)
        txns = richest.get("txns", {}).get("h24", {})
        tx_count = int(txns.get("buys") or 0) + int(txns.get("sells") or 0)
        price_usd = float(richest.get("priceUsd") or 0)
        pair_created_at = richest.get("pairCreatedAt")
        
        # Calculate pair age (more recent = potentially riskier)
        is_new_pair = False
        if pair_created_at:
            age_ms = int(pair_created_at)
            current_ms = int(time.time() * 1000)
            age_hours = (current_ms - age_ms) / (1000 * 3600) if age_ms > 0 else 0
            is_new_pair = age_hours < 6  # Less than 6 hours old
        
        # Real safety criteria based on liquidity, trading activity, and pair age
        # Stricter thresholds for very new pairs
        min_liquidity = 30000 if is_new_pair else 20000
        min_txns = 120 if is_new_pair else 80
        
        is_safe = (
            liquidity_usd >= min_liquidity and
            tx_count >= min_txns and
            price_usd > 0
        )
        
        if is_safe:
            print(f"✅ Token safety confirmed: ${liquidity_usd:,.0f} liquidity, {tx_count} txns/24h, age={'new' if is_new_pair else 'established'}")
        else:
            print(f"⚠️ Token safety check failed: ${liquidity_usd:,.0f} liquidity, {tx_count} txns/24h, age={'new' if is_new_pair else 'established'}")
        
        return is_safe
        
    except Exception as e:
        print(f"❌ Error checking token safety for {token_address[:8]}...{token_address[-8:]}: {e}")
        return False  # Return False on error instead of assuming safe