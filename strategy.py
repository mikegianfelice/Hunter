# strategy.py
import json
import os
import time
import yaml
import requests

# Load config
with open("config.yaml", "r") as f:
    _cfg = yaml.safe_load(f)

PRICE_MEM_FILE = "price_memory.json"
PRICE_MEM_TTL_SECS = int(_cfg.get("price_memory_ttl_minutes", 15)) * 60
PRICE_MEM_PRUNE_SECS = int(_cfg.get("price_memory_prune_hours", 24)) * 3600

BASE_TP = float(_cfg.get("take_profit", 0.5))
TP_MIN  = float(_cfg.get("tp_min", 0.20))
TP_MAX  = float(_cfg.get("tp_max", 1.00))

# Base thresholds
MIN_MOMENTUM_PCT = float(_cfg.get("min_momentum_pct", 0.003))           # Reduced from 0.5% to 0.3%
MIN_VOL_24H_BUY  = float(_cfg.get("min_volume_24h_for_buy", 1000))      # Reduced from 3000 to 1000
MIN_LIQ_USD_BUY  = float(_cfg.get("min_liquidity_usd_for_buy", 1000))   # Reduced from 3000 to 1000
MIN_PRICE_USD    = float(_cfg.get("min_price_usd", 0.0000001))

# Fast-path thresholds for first-seen tokens
FASTPATH_VOL   = float(_cfg.get("fastpath_min_volume_24h", 10000))  # Reduced from 50k to 10k
FASTPATH_LIQ   = float(_cfg.get("fastpath_min_liquidity_usd", 5000))  # Reduced from 25k to 5k
FASTPATH_SENT  = int(_cfg.get("fastpath_min_sent_score", 30))           # Reduced from 40 to 30

# Pre-buy delisting check
ENABLE_PRE_BUY_DELISTING_CHECK = bool(_cfg.get("enable_pre_buy_delisting_check", True))

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
    pruned = {addr: info for addr, info in mem.items()
              if now_ts - int(info.get("ts", 0)) <= PRICE_MEM_PRUNE_SECS}
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
    Add a token to the delisted_tokens.json file
    """
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
            
    except Exception as e:
        print(f"⚠️ Failed to add {symbol} to delisted tokens: {e}")

def _check_token_delisted(token: dict) -> bool:
    """
    Pre-buy check to detect if a token is likely delisted or inactive.
    Returns True if token appears to be delisted/inactive.
    Automatically adds delisted tokens to delisted_tokens.json.
    """
    address = token.get("address", "")
    chain_id = token.get("chainId", "ethereum").lower()
    symbol = token.get("symbol", "")
    
    if not address:
        return True  # No address = likely invalid
    
    # Check for Solana tokens (43-44 character addresses)
    if (len(address) in [43, 44]) and chain_id == "solana":
        try:
            # Try to get current price from Raydium API
            from solana_executor import get_token_price_usd
            current_price = get_token_price_usd(address)
            
            if current_price == 0 or current_price is None:
                print(f"⚠️ Pre-buy check: {symbol} has zero price (may be inactive)")
                # For Solana, be more lenient - zero price doesn't necessarily mean delisted
                # Many new tokens might not be in Raydium API yet
                return False  # Allow the trade to proceed
                
            # Check if price is extremely low (potential delisting)
            if current_price < 0.0000001:
                print(f"🚨 Pre-buy check: {symbol} has suspiciously low price ${current_price}")
                _add_to_delisted_tokens(address, symbol, f"Low price: ${current_price}")
                return True
                
            print(f"✅ Pre-buy check: {symbol} price verified at ${current_price}")
            return False
            
        except ImportError as e:
            print(f"⚠️ Pre-buy check failed for {symbol}: Missing Solana dependencies - {e}")
            print(f"💡 Install correct Solana version: pip install solana==0.27.0")
            # If we can't verify due to missing dependencies, be conservative and skip
            return True
        except Exception as e:
            print(f"⚠️ Pre-buy check failed for {symbol}: {e}")
            # If we can't verify, be conservative but allow the trade
            return False
    
    # For Ethereum tokens, try to get current price
    elif chain_id in ["ethereum", "base"]:
        try:
            from utils import fetch_token_price_usd
            current_price = fetch_token_price_usd(address)
            
            if current_price == 0 or current_price is None:
                print(f"🚨 Pre-buy check: {symbol} appears delisted (zero price)")
                _add_to_delisted_tokens(address, symbol, "Zero price from API")
                return True
                
            # Check if price is extremely low (potential delisting)
            if current_price < 0.0000001:
                print(f"🚨 Pre-buy check: {symbol} has suspiciously low price ${current_price}")
                _add_to_delisted_tokens(address, symbol, f"Low price: ${current_price}")
                return True
                
            print(f"✅ Pre-buy check: {symbol} price verified at ${current_price}")
            return False
            
        except Exception as e:
            print(f"⚠️ Pre-buy check failed for {symbol}: {e}")
            # If we can't verify, be conservative and skip
            return True
    
    # For other chains, skip the check for now
    print(f"ℹ️ Pre-buy check: Skipping for {chain_id} chain")
    return False

def check_buy_signal(token: dict) -> bool:
    address = (token.get("address") or "").lower()
    price   = float(token.get("priceUsd") or 0.0)
    vol24h  = float(token.get("volume24h") or 0.0)
    liq_usd = float(token.get("liquidity") or 0.0)
    is_trusted = bool(token.get("is_trusted", False))
    chain_id = token.get("chainId", "ethereum").lower()

    if not address or price <= MIN_PRICE_USD:
        print("📉 No address or price too low; skipping buy signal.")
        return False

    # Pre-buy delisting check
    if ENABLE_PRE_BUY_DELISTING_CHECK and _check_token_delisted(token):
        print("🚨 Token appears delisted/inactive; skipping buy signal.")
        return False

    # For trusted tokens, require milder depth floors
    min_vol = MIN_VOL_24H_BUY if not is_trusted else max(2000.0, MIN_VOL_24H_BUY * 0.5)
    min_liq = MIN_LIQ_USD_BUY if not is_trusted else max(2000.0, MIN_LIQ_USD_BUY * 0.5)
    
    # For multi-chain tokens, use even lower requirements
    if chain_id != "ethereum":
        min_vol = max(10.0, min_vol * 0.05)  # 5% of normal requirement
        min_liq = max(50.0, min_liq * 0.1)   # 10% of normal requirement

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
    momentum_need = MIN_MOMENTUM_PCT if not is_trusted else max(0.003, MIN_MOMENTUM_PCT * 0.5)  # e.g. 0.3%
    
    # Multi-chain tokens: even easier momentum threshold
    if chain_id != "ethereum":
        momentum_need = max(0.0001, momentum_need * 0.05)  # 5% of normal requirement for multi-chain
        print(f"🔓 Multi-chain momentum threshold: {momentum_need*100:.2f}%")

    # WETH is handled specially in executor.py - skip here
    if address == "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2":  # WETH
        print("🔓 WETH detected - will be handled in executor")
        return True

    if entry:
        prev_price = float(entry.get("price", 0.0))
        prev_ts    = int(entry.get("ts", 0))
        age = now_ts - prev_ts

        if prev_price > 0 and age <= PRICE_MEM_TTL_SECS:
            mom = _pct_change(price, prev_price)
            print(f"📈 Momentum vs {age}s ago: {mom*100:.2f}% (need ≥ {momentum_need*100:.2f}%)")
            if mom >= momentum_need:
                print("✅ Momentum buy signal → TRUE")
                return True
            else:
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
        # Much lower requirements for multi-chain tokens
        fast_vol_ok = (vol24h >= FASTPATH_VOL * 0.001)  # 0.1% of Ethereum requirement
        fast_liq_ok = (liq_usd >= FASTPATH_LIQ * 0.002)  # 0.2% of Ethereum requirement
        fast_sent_ok = True  # Skip sentiment for non-Ethereum
        print(f"🔓 Multi-chain fast-path: vol ${vol24h:,.0f} (need ≥ {FASTPATH_VOL * 0.001:,.0f}), liq ${liq_usd:,.0f} (need ≥ {FASTPATH_LIQ * 0.002:,.0f})")
    else:
        # Original Ethereum requirements
        fast_vol_ok = (vol24h >= FASTPATH_VOL)
        fast_liq_ok = (liq_usd >= FASTPATH_LIQ)
        fast_sent_ok = (sent_score >= FASTPATH_SENT) or (sent_mentions >= 3)

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
    tp = BASE_TP
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

    tp = max(TP_MIN, min(TP_MAX, tp))
    print(f"🎯 Dynamic TP computed: {tp*100:.0f}% (base {BASE_TP*100:.0f}%)")
    return tp