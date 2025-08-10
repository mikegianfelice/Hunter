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
ENFORCE_KEYWORDS = True  # set False to allow all names

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
        print("❌ Error: all trending sources failed. Injecting fallback.")
        _append_all_to_csv([{
            "fetched_at": datetime.utcnow().isoformat(),
            "symbol": "WETH",
            "address": "0xC02aaA39b223FE8D0a0e5C4F27eAD9083C756Cc2",
            "dex": "uniswap",
            "chainId": "ethereum",
            "priceUsd": 2000.0,
            "volume24h": 1_000_000,
            "liquidity": 50_000_000
        }])
        return [{
            "symbol": "WETH",
            "address": "0xC02aaA39b223FE8D0a0e5C4F27eAD9083C756Cc2",
            "dex": "uniswap",
            "priceUsd": 2000.0,
            "volume24h": 1_000_000,
            "liquidity": 50_000_000
        }]

    pairs = data.get("pairs", [])
    print(f"🔍 Found {len(pairs)} total trending tokens...")

    # 1) Log ALL chains to CSV
    fetched_at = datetime.utcnow().isoformat()
    all_rows = []
    for pair in pairs:
        symbol = (pair.get("baseToken", {}) or {}).get("symbol") or ""
        addr = pair.get("pairAddress")
        dex = pair.get("dexId")
        chain = pair.get("chainId") or ""
        price = float(pair.get("priceUsd") or 0)
        vol24 = float((pair.get("volume", {}) or {}).get("h24") or 0)
        liq = float((pair.get("liquidity", {}) or {}).get("usd") or 0)

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

    if all_rows:
        _append_all_to_csv(all_rows)
    else:
        print("ℹ️ No pairs found to log.")

    # 2) Build FILTERED list for trading: Ethereum only
    tokens_for_trading = []
    for row in all_rows:
        if (row["chainId"] or "").lower() != "ethereum":
            continue

        symbol = (row["symbol"] or "").upper()
        vol_points = 2 if row["volume24h"] > 50_000 else 1 if row["volume24h"] > 10_000 else 0
        liq_points = 2 if row["liquidity"] > 30_000 else 1 if row["liquidity"] > 10_000 else 0

        blocked = any(k in symbol for k in EXCLUDED_KEYWORDS)
        clean_points = 0 if blocked else 2
        if blocked and ENFORCE_KEYWORDS:
            print(f"⛔ {symbol} — blocked keyword.")

        score = vol_points + liq_points + clean_points
        print(f"🧪 {symbol or '?'} | Vol: ${row['volume24h']:,.0f} ({vol_points}) | "
              f"LQ: ${row['liquidity']:,.0f} ({liq_points}) | "
              f"Clean: ({clean_points}) → Score: {score}/6")

        if score < 3:
            continue

        tokens_for_trading.append({
            "symbol": symbol or "?",
            "address": row["address"],
            "dex": row["dex"],
            "priceUsd": row["priceUsd"],
            "volume24h": row["volume24h"],
            "liquidity": row["liquidity"],
        })

        if len(tokens_for_trading) >= limit:
            break

    if not tokens_for_trading:
        print("⚠️ No tokens passed filtering. Injecting fallback token for trading.")
        tokens_for_trading = [{
            "symbol": "WETH",
            "address": "0xC02aaA39b223FE8D0a0e5C4F27eAD9083C756Cc2",
            "dex": "uniswap",
            "priceUsd": 2000.0,
            "volume24h": 1_000_000,
            "liquidity": 50_000_000
        }]

    return tokens_for_trading

if __name__ == "__main__":
    fetch_trending_tokens()