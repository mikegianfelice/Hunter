import json
import requests
import os
import time
from web3 import Web3
from pathlib import Path
from src.config.secrets import GRAPH_API_KEY, INFURA_URL, UNISWAP_V3_DEPLOYMENT_ID
from src.utils.http_utils import get_json
from typing import Dict, Any, Optional

from src.storage.sol_price import load_sol_price_cache, save_sol_price_cache
from src.storage.btc_sol_ratio import load_btc_sol_ratio_cache, save_btc_sol_ratio_cache

# Public constants
UNISWAP_V2_ROUTER = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
WETH_ADDRESS      = "0xC02aaA39b223FE8D0A0E5C4F27eAD9083C756Cc2"
USDC_ADDRESS      = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"  # 6 decimals

# Web3 (no key needed; uses your RPC)
_w3 = Web3(Web3.HTTPProvider(INFURA_URL))

_ETH_PRICE_CACHE: Dict[str, float] = {
    "timestamp": 0.0,
    "price": 0.0,
}

# Lazy-load router ABI
def _load_router_abi():
    # Try multiple locations for the ABI file
    candidates = [
        Path("uniswap_router_abi.json"),
        Path("data/uniswap_router_abi.json"),
    ]
    
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text())
    
    raise FileNotFoundError("uniswap_router_abi.json not found in project root or data/.")

def _router():
    return _w3.eth.contract(address=Web3.to_checksum_address(UNISWAP_V2_ROUTER), abi=_load_router_abi())

def _build_graph_gateway_url(api_key: str, deployment_id: str) -> Optional[str]:
    api_key = (api_key or "").strip()
    deployment_id = (deployment_id or "").strip()
    if not api_key or not deployment_id:
        return None
    return f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/{deployment_id}"


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

    url = _build_graph_gateway_url(GRAPH_API_KEY, UNISWAP_V3_DEPLOYMENT_ID)
    if not url:
        print("⚠️ Graph fetch skipped: missing GRAPH_API_KEY or UNISWAP_V3_DEPLOYMENT_ID")
        return None
    try:
        payload = {"query": query}
        headers = {"Content-Type": "application/json"}
        r = requests.post(url, json=payload, headers=headers, timeout=10)
        if r.status_code != 200:
            print(f"⚠️ GraphQL status {r.status_code}")
            return None
        resp = r.json()
        errors = resp.get("errors")
        if errors:
            print(f"⚠️ Graph returned errors: {errors}")
            return None
        data = resp.get("data") or {}
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
    3) Fallback: CoinGecko ETH price.
    4) Emergency fallback to avoid None.
    """
    now = time.time()
    if _ETH_PRICE_CACHE["price"] > 0 and (now - _ETH_PRICE_CACHE["timestamp"]) < 60:
        return _ETH_PRICE_CACHE["price"]
    # 1) Graph route via WETH
    px = fetch_token_price_usd(WETH_ADDRESS)
    if px and px > 0:
        _ETH_PRICE_CACHE["price"] = float(px)
        _ETH_PRICE_CACHE["timestamp"] = time.time()
        return float(px)

    # 2) On-chain Uniswap V2 router quote
    try:
        router = _router()
        amount_in_wei = _w3.to_wei(1, "ether")
        path = [Web3.to_checksum_address(WETH_ADDRESS), Web3.to_checksum_address(USDC_ADDRESS)]
        amounts = router.functions.getAmountsOut(amount_in_wei, path).call()
        usdc_out = float(amounts[-1])  # USDC has 6 decimals
        eth_usd = usdc_out / 1_000_000.0
        if eth_usd > 0:
            _ETH_PRICE_CACHE["price"] = eth_usd
            _ETH_PRICE_CACHE["timestamp"] = time.time()
            return eth_usd
    except Exception as e:
        print(f"❌ Uniswap V2 router quote failed: {e}")

    # 3) CoinGecko fallback
    try:
        coingecko_key = os.getenv("COINGECKO_API_KEY", "").strip()
        base_url = "https://api.coingecko.com/api/v3/"
        url = f"{base_url}simple/price?ids=ethereum&vs_currencies=usd"
        headers = {}
        if coingecko_key:
            headers["x-cg-demo-api-key"] = coingecko_key
        data = get_json(url, headers=headers if headers else None, timeout=10, retries=1)
        if data:
            price = float(data.get("ethereum", {}).get("usd", 0))
            if price > 0:
                _ETH_PRICE_CACHE["price"] = price
                _ETH_PRICE_CACHE["timestamp"] = time.time()
                return price
    except Exception as e:
        print(f"⚠️ CoinGecko ETH price error: {e}")

    # 4) Emergency fallback
    print(f"⚠️ All ETH price sources failed - using emergency fallback of $3000")
    _ETH_PRICE_CACHE["price"] = 3000.0
    _ETH_PRICE_CACHE["timestamp"] = time.time()
    return 3000.0

def _cache_sol_price(price: float):
    """Cache SOL price with timestamp"""
    try:
        save_sol_price_cache(price)
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
    cache = load_sol_price_cache()
    try:
        cached_price = cache.get('price') if cache else None
        cached_time = cache.get('timestamp', 0) if cache else 0
        current_time = time.time()
                
        # Use cached price if less than 1 hour old (increased from 5 minutes for API resilience)
        if cached_price and (current_time - cached_time) < 3600:
            print(f"✅ SOL price from cache: ${cached_price}")
            return cached_price
    except Exception:
        pass  # If cache fails, continue with API calls
    
    # Try CoinGecko first with retry
    coingecko_key = os.getenv("COINGECKO_API_KEY", "").strip()
    coingecko_base = "https://api.coingecko.com/api/v3/"
    for attempt in range(3):
        try:
            url = f"{coingecko_base}simple/price?ids=solana&vs_currencies=usd"
            headers = {}
            if coingecko_key:
                headers["x-cg-demo-api-key"] = coingecko_key
            data = get_json(url, headers=headers if headers else None, timeout=15, retries=1)
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
                        from src.utils.api_tracker import track_coingecko_call
                        track_coingecko_call()
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

def get_btc_sol_ratio() -> float:
    """
    Get BTC/SOL exchange rate (how many SOL per 1 BTC) with 30-minute caching.
    
    Uses cached ratio if available and less than 30 minutes old.
    Otherwise fetches BTC and SOL prices and calculates the ratio.
    
    Returns:
        float: BTC/SOL ratio (e.g., 350.0 means 1 BTC = 350 SOL)
    """
    cache = load_btc_sol_ratio_cache()
    try:
        cached_ratio = cache.get('ratio') if cache else None
        cached_time = cache.get('timestamp', 0) if cache else 0
        current_time = time.time()
        
        # Use cached ratio if less than 30 minutes old (1800 seconds)
        if cached_ratio and cached_ratio > 0 and (current_time - cached_time) < 1800:
            print(f"✅ BTC/SOL ratio from cache: {cached_ratio:.4f}")
            return float(cached_ratio)
    except Exception:
        pass  # If cache fails, continue with calculation
    
    # Fetch BTC and SOL prices to calculate ratio
    try:
        from src.utils.market_data_fetcher import market_data_fetcher
        
        # Get BTC price in USD
        btc_price = market_data_fetcher.get_btc_price()
        if not btc_price or btc_price <= 0:
            # Fallback: use cached ratio even if expired, or estimate
            if cached_ratio and cached_ratio > 0:
                print(f"⚠️ BTC price fetch failed, using expired cached ratio: {cached_ratio:.4f}")
                return float(cached_ratio)
            # Very rough estimate: ~1 BTC = 350 SOL (adjust as needed)
            estimated_ratio = 350.0
            print(f"⚠️ Could not fetch BTC price, using estimated ratio: {estimated_ratio:.4f}")
            return estimated_ratio
        
        # Get SOL price in USD
        sol_price = get_sol_price_usd()
        if not sol_price or sol_price <= 0:
            # Fallback: use cached ratio even if expired, or estimate
            if cached_ratio and cached_ratio > 0:
                print(f"⚠️ SOL price fetch failed, using expired cached ratio: {cached_ratio:.4f}")
                return float(cached_ratio)
            estimated_ratio = 350.0
            print(f"⚠️ Could not fetch SOL price, using estimated ratio: {estimated_ratio:.4f}")
            return estimated_ratio
        
        # Calculate ratio: BTC price / SOL price
        ratio = btc_price / sol_price
        if ratio > 0:
            print(f"✅ BTC/SOL ratio calculated: {ratio:.4f} (BTC: ${btc_price:.2f}, SOL: ${sol_price:.2f})")
            save_btc_sol_ratio_cache(ratio)
            return float(ratio)
        else:
            # Invalid ratio, use cached or estimate
            if cached_ratio and cached_ratio > 0:
                print(f"⚠️ Invalid ratio calculated, using cached ratio: {cached_ratio:.4f}")
                return float(cached_ratio)
            estimated_ratio = 350.0
            print(f"⚠️ Invalid ratio, using estimated ratio: {estimated_ratio:.4f}")
            return estimated_ratio
            
    except Exception as e:
        print(f"⚠️ Error calculating BTC/SOL ratio: {e}")
        # Fallback to cached ratio if available
        if cached_ratio and cached_ratio > 0:
            print(f"⚠️ Using cached ratio as fallback: {cached_ratio:.4f}")
            return float(cached_ratio)
        # Last resort: estimated ratio
        estimated_ratio = 350.0
        print(f"⚠️ All methods failed, using estimated ratio: {estimated_ratio:.4f}")
        return estimated_ratio