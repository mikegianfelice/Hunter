import requests
import time

def check_liquidity_and_volume(token_address, min_liquidity=20000, min_volume=5000):
    try:
        url = f"https://api.dexscreener.com/latest/dex/pairs/ethereum/{token_address}"
        response = requests.get(url)

        if response.status_code != 200:
            print(f"❌ DexScreener error: {response.status_code}")
            return False

        data = response.json()
        pair_data = data.get("pair")

        if not pair_data:
            print(f"❌ No Uniswap data found for {token_address}")
            return False

        liquidity_usd = float(pair_data["liquidity"]["usd"])
        volume_usd = float(pair_data["volume"]["h24"])

        if liquidity_usd >= min_liquidity and volume_usd >= min_volume:
            print(f"✅ Liquidity: ${liquidity_usd:.0f}, Volume: ${volume_usd:.0f}")
            return True
        else:
            print(f"🚫 Insufficient liquidity/volume for {token_address}")
            return False

    except Exception as e:
        print(f"⚠️ Liquidity check failed: {e}")
        return False