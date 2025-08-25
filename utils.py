import json
import requests
from web3 import Web3
from pathlib import Path
from secrets import INFURA_URL

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

    url = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"
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

def get_sol_price_usd() -> float:
    """
    Get SOL/USD price from multiple sources:
    1. CoinGecko API (primary)
    2. Jupiter quote API (fallback)
    """
    # Try CoinGecko first
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            price = float(data.get("solana", {}).get("usd", 0))
            if price > 0:
                print(f"✅ SOL price from CoinGecko: ${price}")
                return price
    except Exception as e:
        print(f"⚠️ CoinGecko SOL price error: {e}")
    
    # Fallback to Jupiter quote API
    try:
        url = "https://quote-api.jup.ag/v6/quote"
        params = {
            "inputMint": "So11111111111111111111111111111111111111112",
            "outputMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "amount": "1000000000",  # 1 SOL in lamports
            "slippageBps": 50,
            "onlyDirectRoutes": False,
            "asLegacyTransaction": False
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("swapUsdValue"):
                price = float(data["swapUsdValue"])
                print(f"✅ SOL price from Jupiter quote: ${price}")
                return price
    except Exception as e:
        print(f"⚠️ Jupiter quote SOL price error: {e}")
    
    print("⚠️ Could not get SOL price from any source")
    return 0.0