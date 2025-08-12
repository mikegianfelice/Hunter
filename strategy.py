# strategy.py
import json
import os
import time
import yaml

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
MIN_MOMENTUM_PCT = float(_cfg.get("min_momentum_pct", 0.005))           # Reduced from 0.8% to 0.5%
MIN_VOL_24H_BUY  = float(_cfg.get("min_volume_24h_for_buy", 3000))      # Reduced from 5000 to 3000
MIN_LIQ_USD_BUY  = float(_cfg.get("min_liquidity_usd_for_buy", 3000))   # Reduced from 5000 to 3000
MIN_PRICE_USD    = float(_cfg.get("min_price_usd", 0.0000001))

# Fast-path thresholds for first-seen tokens
FASTPATH_VOL   = float(_cfg.get("fastpath_min_volume_24h", 50000))  # Reduced from 100k to 50k
FASTPATH_LIQ   = float(_cfg.get("fastpath_min_liquidity_usd", 25000))  # Reduced from 50k to 25k
FASTPATH_SENT  = int(_cfg.get("fastpath_min_sent_score", 40))           # Reduced from 55 to 40

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

def check_buy_signal(token: dict) -> bool:
    address = (token.get("address") or "").lower()
    price   = float(token.get("priceUsd") or 0.0)
    vol24h  = float(token.get("volume24h") or 0.0)
    liq_usd = float(token.get("liquidity") or 0.0)
    is_trusted = bool(token.get("is_trusted", False))

    if not address or price <= MIN_PRICE_USD:
        print("📉 No address or price too low; skipping buy signal.")
        return False

    # For trusted tokens, require milder depth floors
    min_vol = MIN_VOL_24H_BUY if not is_trusted else max(2000.0, MIN_VOL_24H_BUY * 0.5)
    min_liq = MIN_LIQ_USD_BUY if not is_trusted else max(2000.0, MIN_LIQ_USD_BUY * 0.5)

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
    fast_liq_ok   = (vol24h >= FASTPATH_VOL and liq_usd >= FASTPATH_LIQ)

    if is_trusted:
        if fast_liq_ok:
            print("🚀 Trusted fast-path (liq/vol only) → TRUE")
            return True
    else:
        fast_sent_ok  = (sent_score >= FASTPATH_SENT) or (sent_mentions >= 3)
        if fast_liq_ok and fast_sent_ok:
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