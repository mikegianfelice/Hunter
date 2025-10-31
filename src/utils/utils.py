import json
import requests
import os
import time
from web3 import Web3
from pathlib import Path
from src.config.secrets import INFURA_URL
from src.utils.http_utils import get_json

# Public constants
UNISWAP_V2_ROUTER = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
WETH_ADDRESS      = "0xC02aaA39b223FE8D0A0E5C4F27eAD9083C756Cc2"
USDC_ADDRESS      = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"  # 6 decimals

# Web3 (no key needed; uses your RPC)
_w3 = Web3(Web3.HTTPProvider(INFURA_URL))

# Lazy-load router ABI
def _load_router_abi():
    path = Path("uniswap_router_abi.json")
    if not path.exists():
        raise FileNotFoundError("uniswap_router_abi.json not found in project root.")
    return json.loads(path.read_text())

def _router():
    return _w3.eth.contract(address=Web3.to_checksum_address(UNISWAP_V2_ROUTER), abi=_load_router_abi())

def fetch_token_price_usd(token_address: str):
    """
    Tries Uniswap v3 subgraph for token USD price: derivedETH * ethPriceUSD.
    Returns None on failure.
    """
    token_address = (token_address or "").lower()
    if not token_address:
        return None

    query = """
    {
      token(id: "%s") { derivedETH }
      bundle(id: "1") { ethPriceUSD }
    }
    """ % token_address

    url = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3-ethereum"
    try:
        r = requests.post(url, json={"query": query}, timeout=10)
        if r.status_code != 200:
            print(f"⚠️ GraphQL status {r.status_code}")
            return None
        data = r.json().get("data") or {}
        tok = data.get("token")
        bun = data.get("bundle")
        if not tok or not bun:
            print("⚠️ Graph fetch failed: missing token/bundle in GraphQL response")
            return None
        derived_eth = float(tok["derivedETH"])
        eth_usd = float(bun["ethPriceUSD"])
        return derived_eth * eth_usd
    except Exception as e:
        print(f"⚠️ Graph fetch exception: {e}")
        return None

def get_eth_price_usd() -> float:
    """
    Robust ETH/USD:
    1) Try Uniswap v3 subgraph via WETH token (fast).
    2) Fallback: on-chain Uniswap V2 getAmountsOut(1 WETH -> USDC).
    """
    # 1) Graph route via WETH
    px = fetch_token_price_usd(WETH_ADDRESS)
    if px and px > 0:
        return float(px)

    # 2) On-chain Uniswap V2 router quote
    try:
        router = _router()
        amount_in_wei = _w3.to_wei(1, "ether")
        path = [Web3.to_checksum_address(WETH_ADDRESS), Web3.to_checksum_address(USDC_ADDRESS)]
        amounts = router.functions.getAmountsOut(amount_in_wei, path).call()
        usdc_out = float(amounts[-1])  # USDC has 6 decimals
        eth_usd = usdc_out / 1_000_000.0
        if eth_usd <= 0:
            raise ValueError("Non-positive ETH/USD from router")
        return eth_usd
    except Exception as e:
        print(f"❌ Uniswap V2 router quote failed: {e}")
        # Last-ditch: return None; caller should handle
        return None

def _cache_sol_price(price: float):
    """Cache SOL price with timestamp"""
    try:
        cache_data = {
            'price': price,
            'timestamp': time.time()
        }
        with open("sol_price_cache.json", 'w') as f:
            json.dump(cache_data, f)
    except Exception:
        pass  # If caching fails, continue without it

def get_sol_price_usd() -> float:
    """
    Get SOL/USD price from multiple sources with retry logic and caching:
    1. CoinGecko API (primary)
    2. DexScreener API (fallback)
    3. Birdeye API (fallback)
    4. Jupiter quote API (fallback)
    5. Cached price (if recent)
    6. Fallback price (last resort)
    
    Returns a reasonable price even if all APIs fail to prevent trading halt.
    """
    # Check for cached price (valid for 5 minutes to reduce API calls)
    cache_file = "sol_price_cache.json"
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                cached_price = cache_data.get('price')
                cached_time = cache_data.get('timestamp', 0)
                current_time = time.time()
                
                # Use cached price if less than 1 hour old (increased from 5 minutes for API resilience)
                if cached_price and (current_time - cached_time) < 3600:
                    print(f"✅ SOL price from cache: ${cached_price}")
                    return cached_price
    except Exception:
        pass  # If cache fails, continue with API calls
    
    # Try CoinGecko first with retry
    for attempt in range(3):
        try:
            url = "https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd"
            data = get_json(url, timeout=15, retries=1)
            if data:
                # Check for rate limit error in response
                if "status" in data and data["status"].get("error_code") == 429:
                    print(f"⚠️ CoinGecko rate limited (attempt {attempt + 1}/3), trying fallback...")
                    if attempt < 2:
                        time.sleep(2)
                        continue
                else:
                    price = float(data.get("solana", {}).get("usd", 0))
                    if price > 0:
                        print(f"✅ SOL price from CoinGecko: ${price}")
                        _cache_sol_price(price)
                        return price
        except Exception as e:
            print(f"⚠️ CoinGecko SOL price error (attempt {attempt + 1}/3): {e}")
            if attempt < 2:
                time.sleep(1)
                continue
    
    # Fallback to DexScreener API with retry
    for attempt in range(3):
        try:
            url = "https://api.dexscreener.com/latest/dex/tokens/So11111111111111111111111111111111111111112"
            data = get_json(url, timeout=15, retries=1)
            if data:
                pairs = data.get("pairs", [])
                if pairs:
                    for pair in pairs:
                        if pair.get("quoteToken", {}).get("symbol") == "USDC":
                            price = float(pair.get("priceUsd", 0))
                            if price > 0:
                                print(f"✅ SOL price from DexScreener: ${price}")
                                _cache_sol_price(price)
                                return price
                    for pair in pairs:
                        price = float(pair.get("priceUsd", 0))
                        if price > 0:
                            print(f"✅ SOL price from DexScreener (non-USDC): ${price}")
                            _cache_sol_price(price)
                            return price
        except Exception as e:
            print(f"⚠️ DexScreener SOL price error (attempt {attempt + 1}/3): {e}")
        if attempt < 2:
            time.sleep(2)
    
    # Fallback to Birdeye API
    for attempt in range(2):
        try:
            url = "https://public-api.birdeye.so/public/price?address=So11111111111111111111111111111111111111112"
            data = get_json(url, timeout=15, retries=1)
            if data and data.get("success") and data.get("data", {}).get("value"):
                price = float(data["data"]["value"])
                print(f"✅ SOL price from Birdeye: ${price}")
                _cache_sol_price(price)
                return price
        except Exception as e:
            print(f"⚠️ Birdeye SOL price error (attempt {attempt + 1}/2): {e}")
            if attempt < 1:
                time.sleep(1)
    
    # Fallback to Jupiter quote API (simplified)
    # NOTE: Jupiter API endpoint has changed to api.jup.ag
    try:
        url = "https://api.jup.ag/v6/quote?inputMint=So11111111111111111111111111111111111111112&outputMint=EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v&amount=1000000000&slippageBps=50"
        data = get_json(url, timeout=8, retries=1)
        if data and data.get("swapUsdValue"):
            price = float(data["swapUsdValue"])
            print(f"✅ SOL price from Jupiter quote: ${price}")
            _cache_sol_price(price)
            return price
    except Exception as e:
        print(f"⚠️ Jupiter quote SOL price error: {e}")
    
    # Emergency fallback: Use a conservative price estimate to prevent trading halt
    # This allows the bot to continue trading even when all APIs are down
    emergency_price = 140.0  # Conservative SOL price estimate (update periodically)
    print(f"⚠️ All SOL price sources failed - using emergency fallback")
    print(f"💡 Using emergency SOL price: ${emergency_price}")
    print(f"💡 This allows trading to continue despite API issues")
    print(f"💡 Consider updating emergency_price in utils.py if SOL price changes significantly")
    
    # Cache the emergency price to reduce API spam
    _cache_sol_price(emergency_price)
    
    return emergency_price