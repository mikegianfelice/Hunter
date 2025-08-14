# token_scraper.py
import csv
import yaml
from datetime import datetime
from http_utils import get_json

# === Load config.yaml ===
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file) or {}

TELEGRAM_ENABLED = not config.get("test_mode", True)
TELEGRAM_BOT_TOKEN = config.get("telegram_bot_token")
TELEGRAM_CHAT_ID = config.get("telegram_chat_id")

# === Filters for TRADING (logging ignores these) ===
EXCLUDED_KEYWORDS = ["INU", "AI", "PEPE", "DOGE", "SHIBA"]
ENFORCE_KEYWORDS = False  # set False to allow all names - temporarily disabled for testing

# Promotional content filters
PROMOTIONAL_KEYWORDS = [
    "appstore", "playstore", "download", "available", "marketplace", "app", "quotes", 
    "motivation", "inspiration", "mindset", "growth", "success", "productivity",
    "trending", "viral", "explore", "positive", "selfimprovement", "nevergiveup",
    "daily", "best", "for", "you", "creators", "memes", "defi", "tba"
]

PRIMARY_URLS = [
    "https://api.dexscreener.com/latest/dex/search/?q=trending",
]
FALLBACK_URLS = [
    "https://api.dexscreener.com/latest/dex/search/?q=hot",
]

CSV_PATH = "trending_tokens.csv"
CSV_FIELDS = [
    "fetched_at", "symbol", "address", "dex", "chainId",
    "priceUsd", "volume24h", "liquidity"
]

def send_telegram_message(message):
    if not TELEGRAM_ENABLED:
        return
    import requests
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload, timeout=8)
    except Exception as e:
        print("⚠️ Telegram error:", e)

def is_promotional_content(symbol, description=""):
    """Check if content is promotional/spam rather than a real token"""
    if not symbol:
        return True
    
    # Check for promotional keywords in symbol
    symbol_lower = symbol.lower()
    for keyword in PROMOTIONAL_KEYWORDS:
        if keyword in symbol_lower:
            return True
    
    # Check for very long symbols (likely promotional text)
    if len(symbol) > 20:
        return True
    
    # Check for symbols with too many spaces (likely sentences)
    if symbol.count(' ') > 3:
        return True
    
    # Check for symbols with hashtags
    if '#' in symbol:
        return True
    
    # Check for symbols that look like URLs or app descriptions
    if any(word in symbol_lower for word in ['http', 'www', 'app', 'download', 'available']):
        return True
    
    return False

def is_valid_token_data(symbol, address, volume24h, liquidity):
    """Validate that we have real token data"""
    if not symbol or not address:
        return False
    
    # Must have some trading activity
    if volume24h <= 0 or liquidity <= 0:
        return False
    
    # Address should be a valid format (not empty or obviously wrong)
    if len(address) < 10:
        return False
    
    return True

def _append_all_to_csv(rows):
    write_header = False
    try:
        with open(CSV_PATH, "r", newline="") as _:
            pass
    except FileNotFoundError:
        write_header = True

    try:
        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"📁 Appended {len(rows)} rows to {CSV_PATH}")
    except Exception as e:
        print("⚠️ Failed to append CSV:", e)

def fetch_trending_tokens(limit=25):
    headers = {"User-Agent": "Mozilla/5.0 (bot)"}
    data = None

    for u in PRIMARY_URLS:
        try:
            data = get_json(u, headers=headers, timeout=10, retries=3, backoff=0.7)
            break
        except Exception as e:
            print(f"⚠️ Primary fetch failed ({u}): {e}")

    if not data:
        for u in FALLBACK_URLS:
            try:
                data = get_json(u, headers=headers, timeout=10, retries=3, backoff=0.7)
                print(f"🔁 Used fallback source: {u}")
                break
            except Exception as e:
                print(f"⚠️ Fallback fetch failed ({u}): {e}")

    if not data:
        print("❌ Error: all trending sources failed.")
        return []

    pairs = data.get("pairs", [])
    if pairs is None:
        pairs = []
    print(f"🔍 Found {len(pairs)} total trending tokens...")

    # 1) Log ALL chains to CSV (but filter out promotional content)
    fetched_at = datetime.utcnow().isoformat()
    all_rows = []
    valid_tokens_count = 0
    
    for pair in pairs:
        symbol = (pair.get("baseToken", {}) or {}).get("symbol") or ""
        addr = pair.get("pairAddress")
        dex = pair.get("dexId")
        chain = pair.get("chainId") or ""
        price = float(pair.get("priceUsd") or 0)
        vol24 = float((pair.get("volume", {}) or {}).get("h24") or 0)
        liq = float((pair.get("liquidity", {}) or {}).get("usd") or 0)
        
        # Skip promotional content
        if is_promotional_content(symbol):
            print(f"🚫 Skipping promotional content: {symbol[:50]}...")
            continue
            
        # Skip invalid token data
        if not is_valid_token_data(symbol, addr, vol24, liq):
            print(f"🚫 Skipping invalid token data: {symbol} (vol: ${vol24}, liq: ${liq})")
            continue
        
        valid_tokens_count += 1
        all_rows.append({
            "fetched_at": fetched_at,
            "symbol": symbol,
            "address": addr,
            "dex": dex,
            "chainId": chain,
            "priceUsd": price,
            "volume24h": vol24,
            "liquidity": liq
        })
    
    print(f"✅ Found {valid_tokens_count} valid tokens out of {len(pairs)} total pairs")

    if all_rows:
        _append_all_to_csv(all_rows)
    else:
        print("ℹ️ No pairs found to log.")

    # 2) Build FILTERED list for trading: Multi-chain support
    tokens_for_trading = []
    
    # Load supported chains from config
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f) or {}
        supported_chains = config.get("supported_chains", ["ethereum"])
    except Exception:
        supported_chains = ["ethereum"]  # fallback
    
    for row in all_rows:
        chain = (row["chainId"] or "").lower()
        if chain not in supported_chains:
            print(f"⛔ Skipping unsupported chain: {chain}")
            continue
        
        print(f"🔗 Processing {chain.upper()} token: {symbol}")

        symbol = (row["symbol"] or "").upper()
        vol_points = 2 if row["volume24h"] > 25_000 else 1 if row["volume24h"] > 5_000 else 0
        liq_points = 2 if row["liquidity"] > 15_000 else 1 if row["liquidity"] > 5_000 else 0

        blocked = any(k in symbol for k in EXCLUDED_KEYWORDS)
        clean_points = 0 if blocked else 2
        if blocked and ENFORCE_KEYWORDS:
            print(f"⛔ {symbol} — blocked keyword.")

        score = vol_points + liq_points + clean_points
        print(f"🧪 {symbol or '?'} | Vol: ${row['volume24h']:,.0f} ({vol_points}) | "
              f"LQ: ${row['liquidity']:,.0f} ({liq_points}) | "
              f"Clean: ({clean_points}) → Score: {score}/6")

        if score < 2:  # Reduced from 3 to 2
            continue

        tokens_for_trading.append({
            "symbol": symbol or "?",
            "address": row["address"],
            "dex": row["dex"],
            "chainId": row["chainId"],  # Add chainId to token object
            "priceUsd": row["priceUsd"],
            "volume24h": row["volume24h"],
            "liquidity": row["liquidity"],
        })

        if len(tokens_for_trading) >= limit:
            break

    if not tokens_for_trading:
        print("⚠️ No tokens passed filtering. Will try alternative sources...")
        # Don't inject WETH fallback - let the bot handle empty results gracefully
        return []

    return tokens_for_trading

if __name__ == "__main__":
    fetch_trending_tokens()