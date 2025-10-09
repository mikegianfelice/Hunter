# strategy.py
import json
import os
import time
import yaml
import requests
from config_loader import get_config, get_config_bool, get_config_float, get_config_int

# Dynamic config loading
def get_config_values():
    """Get current configuration values dynamically"""
    return {
        'PRICE_MEM_TTL_SECS': get_config_int("price_memory_ttl_minutes", 15) * 60,
        'PRICE_MEM_PRUNE_SECS': get_config_int("price_memory_prune_hours", 24) * 3600,
        'BASE_TP': get_config_float("take_profit", 0.5),
        'TP_MIN': get_config_float("tp_min", 0.20),
        'TP_MAX': get_config_float("tp_max", 1.00),
        'MIN_MOMENTUM_PCT': get_config_float("min_momentum_pct", 0.003),
        'MIN_VOL_24H_BUY': get_config_float("min_volume_24h_for_buy", 50000),
        'MIN_LIQ_USD_BUY': get_config_float("min_liquidity_usd_for_buy", 50000),
        'MIN_PRICE_USD': get_config_float("min_price_usd", 0.0000001),
        'FASTPATH_VOL': get_config_float("fastpath_min_volume_24h", 100000),
        'FASTPATH_LIQ': get_config_float("fastpath_min_liquidity_usd", 100000),
        'FASTPATH_SENT': get_config_int("fastpath_min_sent_score", 30),
        'ENABLE_PRE_BUY_DELISTING_CHECK': get_config_bool("enable_pre_buy_delisting_check", False),
        'PRE_BUY_CHECK_SENSITIVITY': get_config("pre_buy_check_sensitivity", "lenient"),
        'PRE_BUY_CHECK_TIMEOUT': get_config_int("pre_buy_check_timeout", 10)
    }

PRICE_MEM_FILE = "price_memory.json"

def _now() -> int:
    return int(time.time())

def _load_price_mem() -> dict:
    if not os.path.exists(PRICE_MEM_FILE):
        return {}
    try:
        with open(PRICE_MEM_FILE, "r") as f:
            data = json.load(f) or {}
    except Exception:
        return {}
    return _prune_price_mem(data)

def _save_price_mem(mem: dict):
    try:
        with open(PRICE_MEM_FILE, "w") as f:
            json.dump(mem, f, indent=2)
    except Exception:
        pass

def _prune_price_mem(mem: dict) -> dict:
    now_ts = _now()
    config = get_config_values()
    pruned = {addr: info for addr, info in mem.items()
              if now_ts - int(info.get("ts", 0)) <= config['PRICE_MEM_PRUNE_SECS']}
    removed = len(mem) - len(pruned)
    if removed > 0:
        _save_price_mem(pruned)
        print(f"🧹 Pruned {removed} old entries from price_memory.json")
    return pruned

def prune_price_memory() -> int:
    if not os.path.exists(PRICE_MEM_FILE):
        return 0
    try:
        with open(PRICE_MEM_FILE, "r") as f:
            mem = json.load(f) or {}
    except Exception:
        return 0
    before = len(mem)
    pruned = _prune_price_mem(mem)
    return max(0, before - len(pruned))

def _pct_change(curr: float, prev: float) -> float:
    if prev <= 0:
        return 0.0
    return (curr - prev) / prev

def _add_to_delisted_tokens(address: str, symbol: str, reason: str):
    """
    Add a token to the delisted_tokens.json file with smart verification
    """
    try:
        # Import the smart verification function
        from smart_blacklist_cleaner import add_to_delisted_tokens_smart
        
        # Use smart verification to only add if actually delisted
        success = add_to_delisted_tokens_smart(address, symbol, reason)
        
        if not success:
            print(f"ℹ️ {symbol} appears to be active - not adding to delisted tokens")
            
    except Exception as e:
        print(f"⚠️ Failed to add {symbol} to delisted tokens: {e}")
        # Fallback to old method if smart verification fails
        try:
            # Load existing delisted tokens
            if os.path.exists("delisted_tokens.json"):
                with open("delisted_tokens.json", "r") as f:
                    data = json.load(f) or {}
            else:
                data = {"failure_counts": {}, "delisted_tokens": []}
            
            # Add the token address to delisted list
            delisted_tokens = data.get("delisted_tokens", [])
            token_address_lower = address.lower()
            
            if token_address_lower not in delisted_tokens:
                delisted_tokens.append(token_address_lower)
                data["delisted_tokens"] = delisted_tokens
                
                # Save updated data
                with open("delisted_tokens.json", "w") as f:
                    json.dump(data, f, indent=2)
                
                print(f"🛑 Added {symbol} ({address}) to delisted tokens: {reason}")
            else:
                print(f"ℹ️ {symbol} already in delisted tokens list")
                
        except Exception as fallback_error:
            print(f"❌ Both smart and fallback methods failed for {symbol}: {fallback_error}")

def _check_token_delisted(token: dict) -> bool:
    """
    Pre-buy check to detect if a token is likely delisted or inactive.
    Returns True if token appears to be delisted/inactive.
    Automatically adds delisted tokens to delisted_tokens.json.
    """
    config = get_config_values()
    
    # Check if pre-buy delisting check is enabled
    if not config['ENABLE_PRE_BUY_DELISTING_CHECK']:
        print(f"🔓 Pre-buy delisting check disabled - allowing {token.get('symbol', 'UNKNOWN')}")
        return False
    
    address = token.get("address", "")
    chain_id = token.get("chainId", "ethereum").lower()
    symbol = token.get("symbol", "")
    
    if not address:
        return True  # No address = likely invalid
    
    # Check if token is already in our delisted tokens list
    try:
        with open("delisted_tokens.json", "r") as f:
            data = json.load(f) or {}
            delisted_tokens = data.get("delisted_tokens", [])
            if address.lower() in [t.lower() for t in delisted_tokens]:
                print(f"🚨 Pre-buy check: {symbol} is already in delisted tokens list")
                return True
    except Exception as e:
        print(f"⚠️ Error reading delisted tokens: {e}")
        # Don't fail the check if we can't read the file
    
    # Check for Solana tokens (43-44 character addresses)
    if (len(address) in [43, 44]) and chain_id == "solana":
        return _check_solana_token_delisted(token)
    
    # For Ethereum tokens, try to get current price
    elif chain_id == "ethereum":
        return _check_ethereum_token_delisted(token)
    
    # For other chains, skip the check
    print(f"ℹ️ Pre-buy check: Skipping for {chain_id} chain")
    return False

def _check_solana_token_delisted(token: dict) -> bool:
    """
    Enhanced Solana token delisting check with better fallback logic.
    More lenient when DexScreener shows good data and when APIs are down.
    
    Returns True if token appears to be delisted/inactive.
    """
    config = get_config_values()
    address = token.get("address", "")
    symbol = token.get("symbol", "")
    volume_24h = float(token.get("volume24h", 0))
    liquidity = float(token.get("liquidity", 0))
    price_usd = float(token.get("priceUsd", 0))
    
    # Adjust thresholds based on sensitivity setting
    if config['PRE_BUY_CHECK_SENSITIVITY'] == "lenient":
        vol_threshold = 25
        liq_threshold = 100
        poor_vol_threshold = 5
        poor_liq_threshold = 25
    elif config['PRE_BUY_CHECK_SENSITIVITY'] == "strict":
        vol_threshold = 1000
        liq_threshold = 5000
        poor_vol_threshold = 50
        poor_liq_threshold = 200
    else:  # moderate (default)
        vol_threshold = 500
        liq_threshold = 1000
        poor_vol_threshold = 10
        poor_liq_threshold = 50
    
    # If token has good volume and liquidity from DexScreener, trust it
    if volume_24h > vol_threshold and liquidity > liq_threshold:
        print(f"✅ Pre-buy check: {symbol} has good volume (${volume_24h:.0f}) and liquidity (${liquidity:.0f}) - trusting DexScreener data")
        return False
    elif volume_24h > vol_threshold/2 and liquidity > liq_threshold/2:
        print(f"✅ Pre-buy check: {symbol} has moderate volume (${volume_24h:.0f}) and liquidity (${liquidity:.0f}) - trusting DexScreener data")
        return False
    elif volume_24h > vol_threshold/4 and liquidity > liq_threshold/4:
        print(f"✅ Pre-buy check: {symbol} has acceptable volume (${volume_24h:.0f}) and liquidity (${liquidity:.0f}) - trusting DexScreener data")
        return False
    
    # In lenient mode, trust DexScreener data more when APIs fail
    if config['PRE_BUY_CHECK_SENSITIVITY'] == "lenient":
        # If we have any reasonable volume/liquidity from DexScreener, trust it
        if volume_24h > 10 or liquidity > 50:
            print(f"✅ Pre-buy check: {symbol} has reasonable DexScreener data (vol: ${volume_24h:.0f}, liq: ${liquidity:.0f}) - trusting in lenient mode")
            return False
    
    # CRITICAL: If we have valid price from DexScreener, trust it over API checks
    # This prevents false delisting when APIs are rate-limited or down
    if price_usd > 0.0000001:  # Very low but non-zero price threshold
        # If DexScreener shows valid price AND decent volume/liquidity, trust it completely
        if volume_24h > poor_vol_threshold or liquidity > poor_liq_threshold:
            print(f"✅ Pre-buy check: {symbol} has valid DexScreener price (${price_usd}) and decent metrics (vol: ${volume_24h:.0f}, liq: ${liquidity:.0f}) - trusting data")
            return False
        else:
            print(f"✅ Pre-buy check: {symbol} has valid price (${price_usd}) from DexScreener but low metrics - trusting price data")
            return False
    
    # Only do API price verification if DexScreener price is missing/zero
    # Try to get current price from multiple sources to verify if token is actually delisted
    # Only mark as delisted if we can confirm it has zero price from multiple sources
    try:
        from solana_executor import SimpleSolanaExecutor
        executor = SimpleSolanaExecutor()
        current_price = executor.get_token_price_usd(address)
        
        # Note: executor may return 0.000001 as a fallback, not actual zero
        if current_price > 0.00001:  # Higher threshold to account for fallback value
            print(f"✅ Pre-buy check: {symbol} has current price ${current_price} - not delisted")
            return False
        elif current_price <= 0.00001:
            # Only mark as delisted if we have very poor metrics AND API also shows zero/fallback
            if volume_24h < poor_vol_threshold and liquidity < poor_liq_threshold:
                print(f"🚨 Pre-buy check: {symbol} has zero/unknown price and very low metrics - marking as delisted")
                _add_to_delisted_tokens(address, symbol, f"Zero price and low metrics (vol: ${volume_24h:.0f}, liq: ${liquidity:.0f})")
                return True
            else:
                print(f"⚠️ Pre-buy check: {symbol} has zero/unknown price but decent metrics - skipping but not blacklisting")
                return True
    except Exception as e:
        print(f"⚠️ Pre-buy check: Price verification failed for {symbol}: {e}")
        # Graceful degradation: When APIs fail, trust DexScreener data more
        # This prevents false rejections during API outages
        if volume_24h > 10 or liquidity > 50:
            print(f"✅ Pre-buy check: {symbol} price verification failed but has reasonable DexScreener data - allowing token (API outage resilience)")
            return False
        else:
            print(f"⚠️ Pre-buy check: {symbol} price verification failed and poor DexScreener data - skipping but not blacklisting (API may be down)")
            return True
    
    # If we're unsure, be conservative but don't blacklist
    print(f"⚠️ Pre-buy check: {symbol} has uncertain metrics - skipping but not blacklisting")
    return True

def _check_ethereum_token_delisted(token: dict) -> bool:
    """
    Enhanced Ethereum token delisting check with better error handling.
    More lenient when APIs are down but DexScreener shows good data.
    """
    config = get_config_values()
    address = token.get("address", "")
    symbol = token.get("symbol", "")
    volume_24h = float(token.get("volume24h", 0))
    liquidity = float(token.get("liquidity", 0))
    price_usd = float(token.get("priceUsd", 0))
    
    # Adjust thresholds based on sensitivity setting
    if config['PRE_BUY_CHECK_SENSITIVITY'] == "lenient":
        vol_threshold = 100
        liq_threshold = 500
        poor_vol_threshold = 10
        poor_liq_threshold = 50
    elif config['PRE_BUY_CHECK_SENSITIVITY'] == "strict":
        vol_threshold = 2000
        liq_threshold = 10000
        poor_vol_threshold = 100
        poor_liq_threshold = 500
    else:  # moderate (default)
        vol_threshold = 1000
        liq_threshold = 5000
        poor_vol_threshold = 50
        poor_liq_threshold = 200
    
    # If DexScreener shows good data, trust it even if price APIs fail
    if volume_24h > vol_threshold and liquidity > liq_threshold and price_usd > 0:
        print(f"✅ Pre-buy check: {symbol} has good DexScreener data - trusting DexScreener")
        return False
    elif volume_24h > vol_threshold/2 and liquidity > liq_threshold/2 and price_usd > 0:
        print(f"✅ Pre-buy check: {symbol} has moderate DexScreener data - trusting DexScreener")
        return False
    elif volume_24h > vol_threshold/4 and liquidity > liq_threshold/4 and price_usd > 0:
        print(f"✅ Pre-buy check: {symbol} has acceptable DexScreener data - trusting DexScreener")
        return False
    
    # In lenient mode, trust DexScreener data more when APIs fail
    if config['PRE_BUY_CHECK_SENSITIVITY'] == "lenient":
        # If we have any reasonable volume/liquidity from DexScreener, trust it
        if (volume_24h > 50 or liquidity > 200) and price_usd > 0:
            print(f"✅ Pre-buy check: {symbol} has reasonable DexScreener data (vol: ${volume_24h:.0f}, liq: ${liquidity:.0f}) - trusting in lenient mode")
            return False
    
    # Try multiple price sources with fallbacks
    current_price = _get_ethereum_token_price_with_fallbacks(address, symbol)
    
    if current_price is None:
        # In lenient mode, be much more trusting of DexScreener data when APIs fail
        if config['PRE_BUY_CHECK_SENSITIVITY'] == "lenient":
            if (volume_24h > 50 or liquidity > 200) and price_usd > 0:
                print(f"✅ Pre-buy check: {symbol} price verification failed but has reasonable DexScreener data - trusting in lenient mode")
                return False
            else:
                print(f"⚠️ Pre-buy check: {symbol} price verification failed and poor DexScreener data - skipping but not blacklisting")
                return True
        else:
            # If we can't get price from any source, check if we have good DexScreener data
            if volume_24h > vol_threshold/2 and liquidity > liq_threshold/2 and price_usd > 0:
                print(f"✅ Pre-buy check: {symbol} has good DexScreener data despite API failures - trusting DexScreener")
                return False
            elif volume_24h > vol_threshold/4 and liquidity > liq_threshold/4 and price_usd > 0:
                print(f"✅ Pre-buy check: {symbol} has moderate DexScreener data despite API failures - trusting DexScreener")
                return False
            
            # If we can't verify and don't have good DexScreener data, skip but don't blacklist unless metrics are very poor
            if volume_24h < poor_vol_threshold and liquidity < poor_liq_threshold:
                print(f"🚨 Pre-buy check: {symbol} price verification failed and very poor metrics - marking as delisted")
                _add_to_delisted_tokens(address, symbol, "Price verification failed and very poor metrics")
                return True
            else:
                print(f"⚠️ Pre-buy check: {symbol} price verification failed but has reasonable metrics - skipping but not blacklisting")
                return True
    
    if current_price == 0:
        # Only mark as delisted if we also have poor metrics
        if volume_24h < poor_vol_threshold and liquidity < poor_liq_threshold:
            print(f"🚨 Pre-buy check: {symbol} has zero price and poor metrics - likely delisted")
            _add_to_delisted_tokens(address, symbol, "Zero price detected")
            return True
        else:
            print(f"⚠️ Pre-buy check: {symbol} has zero price but decent metrics - skipping but not blacklisting")
            return True
        
    # Check if price is extremely low (potential delisting)
    if current_price < 0.0000001:
        # Only mark as delisted if we also have poor metrics
        if volume_24h < poor_vol_threshold and liquidity < poor_liq_threshold:
            print(f"🚨 Pre-buy check: {symbol} has suspiciously low price ${current_price} and poor metrics")
            _add_to_delisted_tokens(address, symbol, f"Low price: ${current_price}")
            return True
        else:
            print(f"⚠️ Pre-buy check: {symbol} has low price ${current_price} but decent metrics - skipping but not blacklisting")
            return True
        
    print(f"✅ Pre-buy check: {symbol} price verified at ${current_price}")
    return False

def _get_ethereum_token_price_with_fallbacks(address: str, symbol: str) -> float:
    """
    Get Ethereum token price with multiple fallback mechanisms.
    Returns None if all sources fail.
    """
    # Primary: Uniswap Graph API
    try:
        from utils import fetch_token_price_usd
        price = fetch_token_price_usd(address)
        if price and price > 0:
            return price
    except Exception as e:
        print(f"⚠️ Primary price source failed for {symbol}: {e}")
    
    # Fallback 1: Try DexScreener API
    try:
        url = f"https://api.dexscreener.com/latest/dex/tokens/{address}"
        response = requests.get(url, timeout=PRE_BUY_CHECK_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            pairs = data.get("pairs", [])
            if pairs:
                # Get the first pair with USDC or USDT
                for pair in pairs:
                    quote_token = pair.get("quoteToken", {}).get("symbol", "").upper()
                    if quote_token in ["USDC", "USDT"]:
                        price = float(pair.get("priceUsd", 0))
                        if price > 0:
                            print(f"✅ {symbol} price from DexScreener fallback: ${price}")
                            return price
                # If no USDC/USDT pair, use any pair with price
                for pair in pairs:
                    price = float(pair.get("priceUsd", 0))
                    if price > 0:
                        print(f"✅ {symbol} price from DexScreener fallback (non-USDC): ${price}")
                        return price
    except Exception as e:
        print(f"⚠️ DexScreener fallback failed for {symbol}: {e}")
    
    # Fallback 2: Try CoinGecko API (if we have the token ID)
    try:
        # This would require a mapping of addresses to CoinGecko IDs
        # For now, we'll skip this fallback
        pass
    except Exception as e:
        print(f"⚠️ CoinGecko fallback failed for {symbol}: {e}")
    
    print(f"⚠️ All price sources failed for {symbol}")
    return None

def _get_token_liquidity(token_address: str) -> float:
    """Get token liquidity from DexScreener"""
    try:
        url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            pairs = data.get("pairs", [])
            if pairs:
                # Get the first pair with liquidity
                for pair in pairs:
                    liquidity = float(pair.get("liquidity", {}).get("usd", 0))
                    if liquidity > 0:
                        return liquidity
    except Exception as e:
        print(f"⚠️ Could not get liquidity for {token_address}: {e}")
    return 0.0

def _check_jupiter_tradeable(token_address: str, symbol: str) -> bool:
    """
    Pre-check if token is tradeable on Jupiter before doing expensive evaluations.
    Returns True if token is tradeable on Jupiter.
    """
    try:
        # Validate Solana address format
        if len(token_address) != 44:
            print(f"❌ Jupiter pre-check: {symbol} invalid address length ({len(token_address)})")
            return False
        
        # Try a small quote to see if Jupiter supports this token
        url = "https://quote-api.jup.ag/v6/quote"
        params = {
            "inputMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
            "outputMint": token_address,
            "amount": "1000000",  # 1 USDC test amount
            "slippageBps": 100,  # 1% slippage
            "onlyDirectRoutes": "false",
            "asLegacyTransaction": "false"
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("data"):
                print(f"✅ Jupiter pre-check: {symbol} is tradeable")
                return True
            else:
                error_msg = data.get('error', 'Unknown error')
                print(f"❌ Jupiter pre-check: {symbol} not tradeable - {error_msg}")
                return False
        elif response.status_code == 400:
            try:
                error_data = response.json()
                error_msg = error_data.get('error', 'Bad Request')
                if "not tradable" in error_msg.lower() or "not tradeable" in error_msg.lower():
                    print(f"❌ Jupiter pre-check: {symbol} not tradeable - {error_msg}")
                    return False
                elif "cannot be parsed" in error_msg.lower() or "invalid" in error_msg.lower():
                    print(f"⚠️ Jupiter pre-check: {symbol} address validation issue - {error_msg}")
                    # For address parsing issues, let it pass (might be temporary or new token)
                    print(f"🔓 Allowing {symbol} to proceed despite Jupiter validation issue")
                    return True
                else:
                    print(f"⚠️ Jupiter pre-check: {symbol} 400 error - {error_msg}")
                    # For other 400 errors, let it pass (might be temporary)
                    return True
            except:
                print(f"⚠️ Jupiter pre-check: {symbol} 400 error - could not parse")
                # Let it pass for unparseable 400 errors
                return True
        else:
            print(f"⚠️ Jupiter pre-check: {symbol} status {response.status_code}")
            # Let it pass for other status codes (might be temporary)
            return True
            
    except Exception as e:
        print(f"⚠️ Jupiter pre-check failed for {symbol}: {e}")
        # Let it pass for exceptions (might be temporary)
        return True

def check_buy_signal(token: dict) -> bool:
    config = get_config_values()
    address = (token.get("address") or "").lower()
    price   = float(token.get("priceUsd") or 0.0)
    vol24h  = float(token.get("volume24h") or 0.0)
    liq_usd = float(token.get("liquidity") or 0.0)
    is_trusted = bool(token.get("is_trusted", False))
    chain_id = token.get("chainId", "ethereum").lower()

    if not address or price <= config['MIN_PRICE_USD']:
        print("📉 No address or price too low; skipping buy signal.")
        return False

    # Jupiter tradeability pre-check for Solana tokens (PRE-TRADE SAFETY CHECK)
    # This prevents attempting to buy tokens that cannot be traded on Jupiter
    if chain_id == "solana" and not is_trusted and config.get('ENABLE_JUPITER_PRE_CHECK', True):
        jupiter_result = _check_jupiter_tradeable(address, token.get("symbol", "UNKNOWN"))
        if not jupiter_result:
            if config.get('JUPITER_PRE_CHECK_STRICT', False):
                print("❌ Token not tradeable on Jupiter; skipping buy signal.")
                return False
            else:
                # Instead of rejecting, allow the token to proceed and let the actual swap attempt determine tradeability
                # Many new tokens are not immediately available on Jupiter but may be tradeable on other DEXs
                print("⚠️ Jupiter pre-check failed - allowing token to proceed (will attempt actual swap)")
                # Don't return False - let it continue to the actual swap attempt

    # Pre-buy delisting check (only if explicitly enabled in config)
    # This check is disabled by default as it can cause false rejections when APIs fail
    if config.get('ENABLE_PRE_BUY_DELISTING_CHECK', False):
        if _check_token_delisted(token):
            print("🚨 Token appears delisted/inactive; skipping buy signal.")
            return False
    else:
        print("🔓 Pre-buy delisting check disabled - skipping verification")

    # For trusted tokens, require milder depth floors
    min_vol = config['MIN_VOL_24H_BUY'] if not is_trusted else max(2000.0, config['MIN_VOL_24H_BUY'] * 0.5)
    min_liq = config['MIN_LIQ_USD_BUY'] if not is_trusted else max(2000.0, config['MIN_LIQ_USD_BUY'] * 0.5)
    
    # For multi-chain tokens, use lower requirements but not too aggressive
    if chain_id != "ethereum":
        min_vol = max(100.0, min_vol * 0.2)  # 20% of normal requirement (was 5%)
        min_liq = max(500.0, min_liq * 0.3)  # 30% of normal requirement (was 10%)

    if vol24h < min_vol or liq_usd < min_liq:
        print(f"🪫 Fails market depth: vol ${vol24h:,.0f} (need ≥ {min_vol:,.0f}), "
              f"liq ${liq_usd:,.0f} (need ≥ {min_liq:,.0f})")
        return False

    mem = _load_price_mem()
    entry = mem.get(address)
    now_ts = _now()
    mem[address] = {"price": price, "ts": now_ts}
    _save_price_mem(mem)

    # Trusted tokens: slightly easier momentum threshold
    momentum_need = config['MIN_MOMENTUM_PCT'] if not is_trusted else max(0.003, config['MIN_MOMENTUM_PCT'] * 0.5)  # e.g. 0.3%
    
    # Multi-chain tokens: even easier momentum threshold
    if chain_id != "ethereum":
        momentum_need = max(0.0001, momentum_need * 0.05)  # 5% of normal requirement for multi-chain (0.02% * 0.05 = 0.001% = 0.01%)
        print(f"🔓 Multi-chain momentum threshold: {momentum_need*100:.4f}%")

    # WETH is handled specially in executor.py - skip here
    if address == "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2":  # WETH
        print("🔓 WETH detected - will be handled in executor")
        return True

    if entry:
        prev_price = float(entry.get("price", 0.0))
        prev_ts    = int(entry.get("ts", 0))
        age = now_ts - prev_ts

        if prev_price > 0 and age <= config['PRICE_MEM_TTL_SECS']:
            mom = _pct_change(price, prev_price)
            print(f"📈 Momentum vs {age}s ago: {mom*100:.2f}% (need ≥ {momentum_need*100:.2f}%)")
            if mom >= momentum_need:
                print("✅ Momentum buy signal → TRUE")
                return True
            else:
                # Special case for Solana tokens with good volume/liquidity
                if chain_id == "solana" and vol24h >= 10000 and liq_usd >= 50000:
                    print("🔓 Solana token with good metrics - allowing despite zero momentum")
                    return True
                print("❌ Momentum insufficient.")
                return False
        else:
            print("ℹ️ Snapshot stale or missing, evaluating fast-path…")

    # Fast-path: for trusted tokens ignore sentiment; for others require sentiment
    sent_score    = int(token.get("sent_score") or 0)
    sent_mentions = int(token.get("sent_mentions") or 0)
    chain_id = token.get("chainId", "ethereum").lower()
    
    # Adjust requirements for non-Ethereum chains
    if chain_id != "ethereum":
        # Lower requirements for multi-chain tokens but not too aggressive
        fast_vol_ok = (vol24h >= config['FASTPATH_VOL'] * 0.01)   # 1% of Ethereum requirement (was 0.1%)
        fast_liq_ok = (liq_usd >= config['FASTPATH_LIQ'] * 0.02)  # 2% of Ethereum requirement (was 0.2%)
        fast_sent_ok = True  # Skip sentiment for non-Ethereum
        print(f"🔓 Multi-chain fast-path: vol ${vol24h:,.0f} (need ≥ {config['FASTPATH_VOL'] * 0.01:,.0f}), liq ${liq_usd:,.0f} (need ≥ {config['FASTPATH_LIQ'] * 0.02:,.0f})")
    else:
        # Original Ethereum requirements
        fast_vol_ok = (vol24h >= config['FASTPATH_VOL'])
        fast_liq_ok = (liq_usd >= config['FASTPATH_LIQ'])
        fast_sent_ok = (sent_score >= config['FASTPATH_SENT']) or (sent_mentions >= 3)

    if is_trusted:
        if fast_vol_ok and fast_liq_ok:
            print("🚀 Trusted fast-path (liq/vol only) → TRUE")
            return True
    else:
        if fast_vol_ok and fast_liq_ok and fast_sent_ok:
            print("🚀 Fast-path conditions met (liquidity/volume + sentiment) → TRUE")
            return True

    print("❌ No buy signal (no momentum yet and fast-path not met).")
    return False

def get_dynamic_take_profit(token: dict) -> float:
    config = get_config_values()
    tp = config['BASE_TP']
    vol24h = float(token.get("volume24h") or 0.0)
    sent_score = float(token.get("sent_score") or 0.0)
    mentions   = int(token.get("sent_mentions") or 0)

    if sent_score >= 75 or mentions >= 10:
        tp += 0.15
    elif sent_score <= 50 and mentions < 3:
        tp -= 0.10

    if vol24h >= 200_000:
        tp += 0.10
    elif vol24h < 20_000:
        tp -= 0.10

    tp = max(config['TP_MIN'], min(config['TP_MAX'], tp))
    print(f"🎯 Dynamic TP computed: {tp*100:.0f}% (base {config['BASE_TP']*100:.0f}%)")
    return tp