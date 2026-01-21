# token_scraper_improved.py
import csv
import os
import yaml
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from src.utils.http_utils import get_json
from src.utils.address_utils import (
    is_evm_address,
    is_solana_address,
    detect_chain_from_address,
    normalize_evm_address,
    validate_chain_address_match,
)
from collections import defaultdict
from src.config.config_loader import get_config, get_config_bool, get_config_float

# Dynamic config loading
def get_token_scraper_config():
    """Get current configuration values dynamically"""
    return {
        'TELEGRAM_ENABLED': not get_config_bool("test_mode", True),
        'TELEGRAM_BOT_TOKEN': get_config("telegram_bot_token"),
        'TELEGRAM_CHAT_ID': get_config("telegram_chat_id")
    }

# === Enhanced Filters for TRADING ===
EXCLUDED_KEYWORDS = ["INU", "DOGE", "SHIBA", "SAFE", "ELON"]  # Removed "PEPE" to allow PEPE tokens
ENFORCE_KEYWORDS = True  # Re-enable keyword filtering for better quality

# Known established tokens (whitelist for priority) - EXCLUDING STABLECOINS
ESTABLISHED_TOKENS = {
    # Major cryptocurrencies (volatile, tradeable)
    "0xC02aaA39b223FE8D0A0E5C4F27eAD9083C756Cc2": "WETH",  # Wrapped Ethereum
    "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599": "WBTC",  # Wrapped Bitcoin
    "So11111111111111111111111111111111111111112": "SOL",   # Solana
    "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263": "BONK", # BONK
    "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So": "mSOL",  # Marinade Staked SOL
    
    # DeFi blue chips
    "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984": "UNI",   # Uniswap
    "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9": "AAVE",  # Aave
    "0xc00e94Cb662C3520282E6f5717214004A7f26888": "COMP",  # Compound
    "0x9f8F72aA9304c8B593d555F12eF6589cC3A579A2": "MKR",   # Maker
    "0x514910771AF9Ca656af840dff83E8264EcF986CA": "LINK",  # Chainlink
    "0xD533a949740bb3306d119CC777fa900bA034cd52": "CRV",   # Curve
    "0xba100000625a3754423978a60c9317c58a424e3D": "BAL",   # Balancer
    "0x6B3595068778DD592e39A122f4f5a5cF09C90fE2": "SUSHI", # SushiSwap
    "0x111111111117dC0aa78b770fA6A738034120C302": "1INCH", # 1inch
    "0x0bc529c00C6401aEF6D220BE8c6Ea1667F6Ad93e": "YFI",   # Yearn
    "0xC011a73ee8576Fb46F5E1c5751cA3B9Fe0af2a6F": "SNX",   # Synthetix
    "0x408e41876cCCDC0F92210600ef50372656052a38": "REN",   # Ren
    "0xdd974D5C2e2928deA5F71b9825b8b646686BD200": "KNC",   # Kyber
    
    # Layer 2 and scaling
    "0x4200000000000000000000000000000000000006": "WETH",  # WETH on Base
    "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1": "WETH",  # WETH on Arbitrum
    "0x4200000000000000000000000000000000000006": "WETH",  # WETH on Optimism
}

# Known tradeable tokens (legacy whitelist for testing)
KNOWN_TRADEABLE_TOKENS = list(ESTABLISHED_TOKENS.keys())

# Enhanced promotional content filters
PROMOTIONAL_KEYWORDS = [
    "appstore", "playstore", "download", "available", "marketplace", "app", "quotes", 
    "motivation", "inspiration", "mindset", "growth", "success", "productivity",
    "trending", "viral", "explore", "positive", "selfimprovement", "nevergiveup",
    "daily", "best", "for", "you", "creators", "memes", "defi", "tba", "launch",
    "presale", "airdrop", "whitelist", "ico", "ido", "fairlaunch", "stealth"
]

# Enhanced API sources for better diversity - EXCLUDING STABLECOINS
PRIMARY_URLS = [
        # Established and quality tokens (volatile, tradeable)
        "https://api.dexscreener.com/latest/dex/search/?q=ethereum",
        "https://api.dexscreener.com/latest/dex/search/?q=bitcoin",
        "https://api.dexscreener.com/latest/dex/search/?q=solana",
        "https://api.dexscreener.com/latest/dex/search/?q=weth",
        "https://api.dexscreener.com/latest/dex/search/?q=wbtc",
    "https://api.dexscreener.com/latest/dex/search/?q=uniswap",
    "https://api.dexscreener.com/latest/dex/search/?q=aave",
    "https://api.dexscreener.com/latest/dex/search/?q=compound",
    "https://api.dexscreener.com/latest/dex/search/?q=maker",
    "https://api.dexscreener.com/latest/dex/search/?q=chainlink",
    "https://api.dexscreener.com/latest/dex/search/?q=curve",
    "https://api.dexscreener.com/latest/dex/search/?q=balancer",
    "https://api.dexscreener.com/latest/dex/search/?q=sushiswap",
    "https://api.dexscreener.com/latest/dex/search/?q=1inch",
    "https://api.dexscreener.com/latest/dex/search/?q=yearn",
    "https://api.dexscreener.com/latest/dex/search/?q=synthetix",
    "https://api.dexscreener.com/latest/dex/search/?q=ren",
    "https://api.dexscreener.com/latest/dex/search/?q=kyber",
    
    # Trending and momentum tokens
    "https://api.dexscreener.com/latest/dex/search/?q=trending",
    "https://api.dexscreener.com/latest/dex/search/?q=hot",
    "https://api.dexscreener.com/latest/dex/search/?q=gaining",
    "https://api.dexscreener.com/latest/dex/search/?q=volume",
    "https://api.dexscreener.com/latest/dex/search/?q=liquidity",
    "https://api.dexscreener.com/latest/dex/search/?q=rising",
    "https://api.dexscreener.com/latest/dex/search/?q=popular",
    "https://api.dexscreener.com/latest/dex/search/?q=active",
    "https://api.dexscreener.com/latest/dex/search/?q=top",
    
    # DeFi blue chips and established protocols
    "https://api.dexscreener.com/latest/dex/search/?q=defi",
    "https://api.dexscreener.com/latest/dex/search/?q=dex",
    "https://api.dexscreener.com/latest/dex/search/?q=lending",
    "https://api.dexscreener.com/latest/dex/search/?q=staking",
    "https://api.dexscreener.com/latest/dex/search/?q=yield",
    "https://api.dexscreener.com/latest/dex/search/?q=governance",
    "https://api.dexscreener.com/latest/dex/search/?q=dao",
    "https://api.dexscreener.com/latest/dex/search/?q=oracle",
    "https://api.dexscreener.com/latest/dex/search/?q=bridge",
    "https://api.dexscreener.com/latest/dex/search/?q=layer2",
    "https://api.dexscreener.com/latest/dex/search/?q=scaling",
    
    # Quality categories
    "https://api.dexscreener.com/latest/dex/search/?q=gaming",
    "https://api.dexscreener.com/latest/dex/search/?q=nft",
    "https://api.dexscreener.com/latest/dex/search/?q=ai",
    "https://api.dexscreener.com/latest/dex/search/?q=privacy",
    "https://api.dexscreener.com/latest/dex/search/?q=infrastructure",
    "https://api.dexscreener.com/latest/dex/search/?q=storage",
    "https://api.dexscreener.com/latest/dex/search/?q=compute",
    "https://api.dexscreener.com/latest/dex/search/?q=identity",
    
    # Time-based searches for established tokens
    "https://api.dexscreener.com/latest/dex/search/?q=24h",
    "https://api.dexscreener.com/latest/dex/search/?q=7d",
    "https://api.dexscreener.com/latest/dex/search/?q=30d",
    
    # Meme tokens (reduced priority but still included)
    "https://api.dexscreener.com/latest/dex/search/?q=memes",
    "https://api.dexscreener.com/latest/dex/search/?q=moon",
    "https://api.dexscreener.com/latest/dex/search/?q=pump",
]

FALLBACK_URLS = [
    "https://api.dexscreener.com/latest/dex/search/?q=trending",
    "https://api.dexscreener.com/latest/dex/search/?q=hot",
    "https://api.dexscreener.com/latest/dex/search/?q=new",
    "https://api.dexscreener.com/latest/dex/search/?q=volume",
]

CSV_PATH = "data/trending_tokens.csv"
CSV_FIELDS = [
    "fetched_at", "symbol", "address", "dex", "chainId",
    "priceUsd", "volume24h", "liquidity",
    "priceChange5m", "priceChange1h", "priceChange24h"
]

def send_telegram_message(message):
    config = get_token_scraper_config()
    if not config['TELEGRAM_ENABLED']:
        return
    import requests
    url = f"https://api.telegram.org/bot{config['TELEGRAM_BOT_TOKEN']}/sendMessage"
    payload = {"chat_id": config['TELEGRAM_CHAT_ID'], "text": message}
    try:
        requests.post(url, data=payload, timeout=8)
    except Exception as e:
        print("⚠️ Telegram error:", e)

def is_promotional_content(symbol, description=""):
    """Enhanced check if content is promotional/spam rather than a real token"""
    if not symbol:
        return True
    
    # Check for promotional keywords in symbol
    symbol_lower = symbol.lower()
    for keyword in PROMOTIONAL_KEYWORDS:
        if keyword in symbol_lower:
            return True
    
    # Check for very long symbols (likely promotional text)
    if len(symbol) > 20:  # Increased from 15 to 20 for more opportunities
        return True
    
    # Check for symbols with too many spaces (likely sentences)
    if symbol.count(' ') > 3:  # Increased from 2 to 3 for more opportunities
        return True
    
    # Check for symbols with hashtags or special characters
    if any(char in symbol for char in ['#', '@', 'http', 'www', '.com', '.io']):
        return True
    
    # Check for symbols that look like URLs or app descriptions
    if any(word in symbol_lower for word in ['http', 'www', 'app', 'download', 'available', 'launch']):
        return True
    
    # Check for symbols that are just numbers or very short
    if len(symbol) < 2 or symbol.isdigit():
        return True
    
    return False

def is_valid_token_data(symbol, address, volume24h, liquidity):
    """Enhanced validation that we have real token data with strict liquidity requirements"""
    if not symbol or not address:
        return False
    
    # Must have some trading activity
    if volume24h <= 0 or liquidity <= 0:
        return False
    
    # Address should be a valid format (not empty or obviously wrong)
    if len(address) < 10:
        return False
    
    # EXCLUDE STABLECOINS - they don't have price volatility for trading
    # Stablecoins maintain ~$1 value, so no profit opportunity from price movements
    stablecoin_symbols = ['USDC', 'USDT', 'DAI', 'BUSD', 'TUSD', 'USDP', 'FRAX', 'LUSD', 'SUSD', 'GUSD']
    if symbol.upper() in stablecoin_symbols:
        return False
    
    # REDUCED minimum requirements to allow more trading opportunities
    # These match the config.yaml thresholds to ensure consistency
    if volume24h < 3000:  # Minimum $3k volume (matches config.yaml)
        return False
    
    if liquidity < 8000:  # Minimum $8k liquidity (matches config.yaml)
        return False
    
    return True

def calculate_token_score(symbol, volume24h, liquidity, chain_id, address=None):
    """Calculate a quality score for token filtering with established token prioritization"""
    score = 0
    
    # ESTABLISHED TOKEN PRIORITY (0-5 points) - HIGHEST PRIORITY
    # Only give bonus for exact address matches to prevent copycat tokens from getting false positives
    if address and address in ESTABLISHED_TOKENS:
        established_symbol = ESTABLISHED_TOKENS[address]
        score += 5  # Maximum priority for established tokens
        print(f"🏆 {symbol} - ESTABLISHED TOKEN BONUS (+5): {established_symbol}")
    # Removed symbol matching logic to prevent copycat tokens from getting false bonuses
    
    # Volume scoring (0-3 points) - balanced thresholds
    if volume24h >= 1000000:  # $1M+ volume (high priority)
        score += 3
    elif volume24h >= 100000:  # $100k+ volume
        score += 2
    elif volume24h >= 50000:  # $50k+ volume
        score += 1
    
    # Liquidity scoring (0-3 points) - balanced thresholds
    if liquidity >= 1000000:  # $1M+ liquidity (high priority)
        score += 3
    elif liquidity >= 200000:  # $200k+ liquidity
        score += 2
    elif liquidity >= 100000:  # $100k+ liquidity
        score += 1
    
    # Symbol quality scoring (0-2 points)
    symbol_lower = symbol.lower()
    
    # Penalize common spam symbols
    spam_indicators = ['hot', 'moon', 'safe', 'elon', 'inu', 'doge', 'shiba', 'pepe', 'pump', 'moon', 'rocket']
    if any(indicator in symbol_lower for indicator in spam_indicators):
        score -= 2  # Increased penalty for spam
    
    # Removed DeFi symbol matching to prevent copycat tokens from getting false bonuses
    # Only established tokens with exact address matches should get quality bonuses
    
    # Bonus for unique/interesting symbols
    if len(symbol) >= 4 and len(symbol) <= 8:
        score += 1
    
    # Chain-specific adjustments
    if chain_id == "ethereum":
        score += 1  # Bonus for Ethereum tokens
    elif chain_id in ["solana", "base"]:
        score += 0  # Neutral for other major chains
    else:
        score -= 1  # Penalty for less common chains
    
    return max(0, score)  # Ensure non-negative

def calculate_legitimacy_score(pair: dict) -> float:
    """
    Calculate legitimacy score (0-10) based on trading patterns.
    Higher scores indicate legitimate trading activity vs wash trading.
    
    Analyzes:
    - Buy/sell ratio balance
    - Transaction distribution consistency over time
    - Volume distribution consistency
    - Price movement correlation with volume
    - Average trade sizes
    - Token age
    
    Args:
        pair: DexScreener pair data dictionary
    
    Returns:
        float: Legitimacy score from 0-10
    """
    score = 0.0
    
    try:
        # Get transaction data
        txns_24h = pair.get("txns", {}).get("h24", {})
        buys_24h = int(txns_24h.get("buys", 0))
        sells_24h = int(txns_24h.get("sells", 0))
        total_txns = buys_24h + sells_24h
        
        txns_1h = pair.get("txns", {}).get("h1", {})
        buys_1h = int(txns_1h.get("buys", 0))
        sells_1h = int(txns_1h.get("sells", 0))
        
        # Get volume data
        volume24h = float((pair.get("volume") or {}).get("h24") or 0)
        volume_1h = float((pair.get("volume") or {}).get("h1") or 0)
        
        # Get price change data
        price_change = pair.get("priceChange", {})
        price_change_24h = float(price_change.get("h24", 0))
        
        # Get pair age
        pair_created_at = pair.get("pairCreatedAt")
        pair_age_hours = None
        if pair_created_at:
            age_ms = int(pair_created_at)
            current_ms = int(datetime.now().timestamp() * 1000)
            pair_age_hours = (current_ms - age_ms) / (1000 * 3600)
        
        # Indicator 1: Buy/Sell Ratio (0-2 points)
        if total_txns > 0:
            buy_sell_ratio = buys_24h / sells_24h if sells_24h > 0 else 0
            if 0.5 <= buy_sell_ratio <= 2.0:
                score += 2.0  # Balanced ratio - healthy trading
            elif 0.2 <= buy_sell_ratio <= 5.0:
                score += 1.0  # Reasonable ratio
        
        # Indicator 2: Transaction Distribution Consistency (0-2 points)
        if total_txns > 0 and (buys_1h + sells_1h) > 0:
            txn_rate_24h = total_txns / 24
            txn_rate_1h = buys_1h + sells_1h
            if txn_rate_24h > 0:
                consistency_ratio = txn_rate_1h / txn_rate_24h
                if 0.5 <= consistency_ratio <= 2.0:
                    score += 2.0  # Consistent activity over time
                elif 0.3 <= consistency_ratio <= 3.0:
                    score += 1.0  # Reasonable consistency
        
        # Indicator 3: Volume Distribution Consistency (0-2 points)
        if volume24h > 0 and volume_1h > 0:
            volume_per_hour_avg = volume24h / 24
            if volume_per_hour_avg > 0:
                volume_consistency = volume_1h / volume_per_hour_avg
                if 0.3 <= volume_consistency <= 3.0:
                    score += 2.0  # Consistent volume distribution
                elif 0.1 <= volume_consistency <= 5.0:
                    score += 1.0  # Reasonable distribution
        
        # Indicator 4: Price Movement Correlation (0-2 points)
        if abs(price_change_24h) > 50.0:
            score += 2.0  # Significant price movement with volume - likely legitimate
        elif abs(price_change_24h) > 10.0:
            score += 1.0  # Moderate price movement
        
        # Indicator 5: Average Trade Size (0-1 point)
        if total_txns > 0:
            avg_trade_size = volume24h / total_txns
            if 10 <= avg_trade_size <= 10000:
                score += 1.0  # Reasonable trade sizes (not bot-like or whale manipulation)
        
        # Indicator 6: Token Age (0-1 point)
        if pair_age_hours:
            if pair_age_hours >= 168:  # 7+ days old
                score += 1.0  # Established token - lower risk
        
    except Exception as e:
        # On error, return 0 (fail closed - don't bypass if we can't verify)
        return 0.0
    
    return min(10.0, score)  # Cap at 10

def ensure_symbol_diversity(tokens, max_same_symbol=3):
    """Ensure we don't get too many tokens with the same symbol"""
    symbol_counts = defaultdict(int)
    diverse_tokens = []
    
    for token in tokens:
        symbol = token.get("symbol", "").upper()
        if symbol_counts[symbol] < max_same_symbol:
            diverse_tokens.append(token)
            symbol_counts[symbol] += 1
        else:
            print(f"🔄 Skipping duplicate symbol: {symbol} (already have {symbol_counts[symbol]})")
    
    return diverse_tokens

def _append_all_to_csv(rows):
    write_header = False
    try:
        with open(CSV_PATH, "r", newline="") as _:
            pass
    except FileNotFoundError:
        write_header = True

    try:
        os.makedirs('data', exist_ok=True)
        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"📁 Appended {len(rows)} rows to {CSV_PATH}")
    except Exception as e:
        print("⚠️ Failed to append CSV:", e)

def test_token_tradeability(token_address: str, chain_id: str = "solana") -> bool:
    """Test if a token is actually tradeable on Jupiter or Raydium"""
    try:
        if chain_id.lower() == "solana":
            # Test Jupiter quote
            from jupiter_lib import JupiterCustomLib
            from src.config.secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY
            
            lib = JupiterCustomLib(SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY)
            SOL_MINT = "So11111111111111111111111111111111111111112"
            
            # Try a small quote
            quote = lib.get_quote(SOL_MINT, token_address, 1000000, slippage=0.10)  # 0.001 SOL
            if quote and quote.get('success'):
                return True
                
            # Try Raydium fallback
            from raydium_executor import get_raydium_executor
            executor = get_raydium_executor()
            if executor.check_token_tradeable_on_raydium(token_address):
                return True
                
        return False
    except Exception as e:
        print(f"⚠️ Tradeability test failed for {token_address}: {e}")
        return False

def fetch_jupiter_trending_tokens(intervals=None, limit_per_interval=100):
    """
    Fetch trending Solana tokens from Jupiter API.
    
    Args:
        intervals: List of intervals to fetch (default: ["1h", "6h", "24h"])
        limit_per_interval: Max tokens per interval (default: 100)
    
    Returns:
        List of Jupiter token objects, or empty list if failed
    """
    try:
        from src.config.secrets import JUPITER_API_KEY
        
        if not JUPITER_API_KEY:
            print("⚠️ Jupiter API key not found, skipping Jupiter discovery")
            return []
    except ImportError:
        print("⚠️ Could not import Jupiter API key, skipping Jupiter discovery")
        return []
    except Exception as e:
        print(f"⚠️ Error loading Jupiter API key: {e}, skipping Jupiter discovery")
        return []
    
    if intervals is None:
        intervals = get_config("token_discovery.jupiter_intervals", ["1h", "6h", "24h"])
    
    all_tokens = []
    seen_addresses = set()
    
    headers = {"x-api-key": JUPITER_API_KEY}
    
    for interval in intervals:
        try:
            url = f"https://api.jup.ag/tokens/v2/toptrending/{interval}?limit={limit_per_interval}"
            print(f"🔍 Fetching Jupiter {interval} trending tokens...")
            data = get_json(url, headers=headers, timeout=15, retries=2, backoff=1.0)
            
            if data and isinstance(data, list):
                count = 0
                for idx, token in enumerate(data):
                    token_id = token.get("id")
                    if token_id and token_id not in seen_addresses:
                        # Add metadata about trending position and interval
                        # Position is 1-indexed based on position in the trending list
                        token["_jupiter_trending_position"] = idx + 1
                        token["_jupiter_trending_interval"] = interval
                        token["_source"] = "jupiter"
                        all_tokens.append(token)
                        seen_addresses.add(token_id)
                        count += 1
                print(f"✅ Fetched {count} unique tokens from Jupiter {interval} trending")
            elif data:
                print(f"⚠️ Unexpected Jupiter API response format for {interval}")
        except Exception as e:
            print(f"⚠️ Failed to fetch Jupiter {interval} trending: {e}")
            continue
    
    print(f"🎯 Total Jupiter trending tokens: {len(all_tokens)}")
    return all_tokens

def transform_jupiter_token_to_common_format(jupiter_token):
    """
    Transform Jupiter token format to common format used by token scraper.
    
    Args:
        jupiter_token: Jupiter API token object
    
    Returns:
        Dictionary in common format, or None if transformation fails
    """
    try:
        token_id = jupiter_token.get("id")
        if not token_id:
            return None
        
        # Extract volume from stats24h
        stats24h = jupiter_token.get("stats24h", {})
        buy_volume = float(stats24h.get("buyVolume", 0) or 0)
        sell_volume = float(stats24h.get("sellVolume", 0) or 0)
        volume24h = buy_volume + sell_volume
        
        # Extract price changes
        stats5m = jupiter_token.get("stats5m", {})
        stats1h = jupiter_token.get("stats1h", {})
        price_change_5m = stats5m.get("priceChange")
        price_change_1h = stats1h.get("priceChange")
        price_change_24h = stats24h.get("priceChange")
        
        # Convert price changes from decimals (e.g., 0.16 = 16%) to percentages
        if price_change_5m is not None:
            price_change_5m = price_change_5m * 100
        if price_change_1h is not None:
            price_change_1h = price_change_1h * 100
        if price_change_24h is not None:
            price_change_24h = price_change_24h * 100
        
        # Get liquidity
        liquidity = float(jupiter_token.get("liquidity", 0) or 0)
        
        # Get price
        price = float(jupiter_token.get("usdPrice", 0) or 0)
        
        # Get symbol and name
        symbol = (jupiter_token.get("symbol") or "").upper()
        name = jupiter_token.get("name") or ""
        
        # Determine DEX (try to infer from firstPool or default to "jupiter")
        dex = "jupiter"
        first_pool = jupiter_token.get("firstPool", {})
        if first_pool:
            pool_id = first_pool.get("id", "")
            # Could add logic here to determine DEX from pool ID if needed
        
        return {
            "fetched_at": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "address": token_id,
            "dex": dex,
            "chainId": "solana",  # Jupiter only returns Solana tokens
            "priceUsd": price,
            "volume24h": volume24h,
            "liquidity": liquidity,
            "priceChange5m": price_change_5m if price_change_5m is not None else "",
            "priceChange1h": price_change_1h if price_change_1h is not None else "",
            "priceChange24h": price_change_24h if price_change_24h is not None else "",
            # Jupiter-specific metadata
            "jupiter_validated": True,
            "organic_score": jupiter_token.get("organicScore"),
            "holder_count": jupiter_token.get("holderCount"),
            "is_verified": jupiter_token.get("isVerified", False),
            "trending_position": jupiter_token.get("_jupiter_trending_position"),
            "trending_interval": jupiter_token.get("_jupiter_trending_interval"),
            "source": "jupiter",
            "name": name,
        }
    except Exception as e:
        print(f"⚠️ Error transforming Jupiter token: {e}")
        return None

def fetch_trending_tokens(limit=200):  # INCREASED for more opportunities
    """Enhanced token discovery with better diversity and quality filtering"""
    headers = {"User-Agent": "Mozilla/5.0 (bot)"}
    all_pairs = []
    jupiter_rows = []
    
    # Step 1: Fetch Jupiter trending tokens FIRST (primary source for Solana)
    jupiter_enabled = get_config_bool("token_discovery.jupiter_enabled", True)
    if jupiter_enabled:
        try:
            print("🚀 Fetching Jupiter trending tokens (primary source)...")
            jupiter_tokens = fetch_jupiter_trending_tokens()
            
            if jupiter_tokens:
                # Apply Jupiter-specific filters
                min_organic_score = get_config_float("token_discovery.jupiter_min_organic_score", 50.0)
                min_holders = get_config_float("token_discovery.jupiter_min_holders", 100.0)
                
                filtered_jupiter_tokens = []
                for token in jupiter_tokens:
                    organic_score = token.get("organicScore", 0)
                    holder_count = token.get("holderCount", 0)
                    
                    if organic_score >= min_organic_score and holder_count >= min_holders:
                        filtered_jupiter_tokens.append(token)
                    else:
                        print(f"🚫 Filtered Jupiter token {token.get('symbol', 'N/A')}: organic_score={organic_score:.1f} (min {min_organic_score}), holders={holder_count} (min {min_holders})")
                
                # Transform Jupiter tokens to common format
                for token in filtered_jupiter_tokens:
                    transformed = transform_jupiter_token_to_common_format(token)
                    if transformed:
                        jupiter_rows.append(transformed)
                
                print(f"✅ Processed {len(jupiter_rows)} Jupiter tokens after filtering")
        except Exception as e:
            print(f"⚠️ Jupiter discovery failed: {e}, continuing with DexScreener only")
    
    # Step 2: Fetch DexScreener tokens (secondary source, covers all chains)
    print("🔍 Fetching DexScreener tokens (secondary source)...")
    
    # Try multiple primary sources in a rotating deterministic order
    primary_urls = PRIMARY_URLS.copy()
    # Rotate list based on minute to avoid bias without randomness
    try:
        shift = int(time.time() // 60) % len(primary_urls)
        primary_urls = primary_urls[shift:] + primary_urls[:shift]
    except Exception:
        pass

    def _fetch(u: str):
        try:
            data = get_json(u, headers=headers, timeout=10, retries=3, backoff=0.7)
            return u, data
        except Exception as e:
            return u, None

    # Fetch up to 15 sources concurrently with a bounded pool
    urls_to_fetch = primary_urls[:15]
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(_fetch, u): u for u in urls_to_fetch}
        for fut in as_completed(futures):
            u, data = fut.result()
            if data and data.get("pairs"):
                pairs = data.get("pairs", [])
                all_pairs.extend(pairs)
                print(f"✅ Fetched {len(pairs)} tokens from {u}")
            else:
                print(f"⚠️ Primary fetch failed ({u})")

    # If no primary sources worked, try fallbacks
    if not all_pairs:
        for u in FALLBACK_URLS:
            try:
                data = get_json(u, headers=headers, timeout=10, retries=3, backoff=0.7)
                if data and data.get("pairs"):
                    pairs = data.get("pairs", [])
                    all_pairs.extend(pairs)
                    print(f"🔁 Used fallback source: {u} - {len(pairs)} tokens")
                break
            except Exception as e:
                print(f"⚠️ Fallback fetch failed ({u}): {e}")

    # Step 3: Process DexScreener pairs and merge with Jupiter tokens
    # Remove duplicates based on pair address
    unique_pairs = []
    seen_addresses = set()
    
    # Track Jupiter addresses to avoid duplicates
    jupiter_addresses = {row["address"] for row in jupiter_rows}
    
    for pair in all_pairs:
        base_token = (pair.get("baseToken", {}) or {})
        addr = base_token.get("address") or ""
        
        # Skip if this token was already found via Jupiter (Jupiter takes priority)
        if addr and addr in jupiter_addresses:
            continue
        
        pair_addr = pair.get("pairAddress")
        if pair_addr and pair_addr not in seen_addresses:
            unique_pairs.append(pair)
            seen_addresses.add(pair_addr)
    
    print(f"🔍 Found {len(unique_pairs)} unique DexScreener tokens (after deduplication with Jupiter)")
    print(f"📊 Total tokens: {len(jupiter_rows)} Jupiter + {len(unique_pairs)} DexScreener = {len(jupiter_rows) + len(unique_pairs)}")

    # Load supported chains from config early
    supported_chains = get_config("supported_chains", ["ethereum"])
    
    print(f"🔗 Supported chains: {supported_chains}")
    
    # Step 4: Process DexScreener pairs and combine with Jupiter tokens
    fetched_at = datetime.utcnow().isoformat()
    all_rows = jupiter_rows.copy()  # Start with Jupiter tokens
    valid_tokens_count = len(jupiter_rows)  # Count Jupiter tokens as valid
    unsupported_chain_count = 0
    
    # Process DexScreener pairs
    for pair in unique_pairs:
        base_token = (pair.get("baseToken", {}) or {})
        symbol = base_token.get("symbol") or ""
        # Use the actual token mint/contract address, not the pair/pool address
        addr = base_token.get("address") or ""
        dex = pair.get("dexId")
        chain = pair.get("chainId") or ""
        price = float(pair.get("priceUsd") or 0)
        vol24 = float((pair.get("volume", {}) or {}).get("h24") or 0)
        liq = float((pair.get("liquidity", {}) or {}).get("usd") or 0)
        
        # Extract price change data from DexScreener (percentages: e.g., 5.5 means 5.5%)
        price_change = (pair.get("priceChange", {}) or {})
        price_change_5m = price_change.get("m5")  # 5 minute price change percentage
        price_change_1h = price_change.get("h1")  # 1 hour price change percentage
        price_change_24h = price_change.get("h24")  # 24 hour price change percentage
        
        # Early chain filtering - skip unsupported chains immediately
        if chain.lower() not in supported_chains:
            unsupported_chain_count += 1
            continue
        
        # Enhanced promotional content filtering
        if is_promotional_content(symbol):
            print(f"🚫 Skipping promotional content: {symbol[:50]}...")
            continue
        
        # Reject tokens with suspiciously low prices (likely scams) - more lenient
        if price < 0.0000001:  # Less than $0.0000001 (more lenient)
            print(f"🚫 Skipping suspiciously low price token: {symbol} (${price})")
            continue
        
        # Reject tokens with suspicious volume/liquidity ratios (manipulation indicators)
        if liq > 0 and vol24 > 0:
            vol_liq_ratio = vol24 / liq
            max_vol_liq_ratio = get_config_float("max_volume_liquidity_ratio", 10.0)
            enable_legitimacy_bypass = get_config_bool("enable_legitimacy_bypass", True)
            min_legitimacy_for_bypass = get_config_float("legitimacy_bypass_threshold", 7.0)
            
            # Reject tokens with volume > threshold liquidity
            if vol_liq_ratio > max_vol_liq_ratio:
                # Check if token has high legitimacy score (allows bypass)
                if enable_legitimacy_bypass:
                    legitimacy_score = calculate_legitimacy_score(pair)
                    if legitimacy_score >= min_legitimacy_for_bypass:
                        print(f"✅ Allowing token with high volume/liquidity ratio due to high legitimacy score: {symbol} (ratio: {vol_liq_ratio:.2f}x, legitimacy: {legitimacy_score:.1f}/10)")
                        # Token passes - continue to next checks
                    else:
                        print(f"🚫 Skipping token with suspiciously high volume/liquidity ratio: {symbol} (ratio: {vol_liq_ratio:.2f}x, legitimacy: {legitimacy_score:.1f}/10 - below threshold)")
                        continue
                else:
                    print(f"🚫 Skipping token with suspiciously high volume/liquidity ratio: {symbol} (ratio: {vol_liq_ratio:.2f}x, vol: ${vol24:,.0f}, liq: ${liq:,.0f})")
                    continue
            # Reject tokens with extremely low volume/liquidity ratios (more lenient)
            if vol_liq_ratio < 0.05:  # Volume less than 5% of liquidity (more lenient)
                print(f"🚫 Skipping low volume/liquidity ratio: {symbol} (ratio: {vol_liq_ratio:.2f})")
                continue
            
        # Enhanced token data validation
        if not is_valid_token_data(symbol, addr, vol24, liq):
            print(f"🚫 Skipping invalid token data: {symbol} (vol: ${vol24}, liq: ${liq})")
            continue

        # Enhanced chain/address consistency validation and normalization
        is_valid, corrected_chain, error_message = validate_chain_address_match(addr, chain)
        
        if not is_valid:
            # Skip tokens with invalid chain/address combinations
            print(f"🚫 Skipping {symbol}: {error_message}")
            continue
        
        # Update chain if it was corrected
        if corrected_chain != chain.lower():
            print(f"🔧 Correcting chain for {symbol}: {chain} → {corrected_chain} (by address format)")
            chain = corrected_chain
        
        # Normalize EVM addresses
        if detect_chain_from_address(addr) == "evm":
            addr = normalize_evm_address(addr)
        
        valid_tokens_count += 1
        all_rows.append({
            "fetched_at": fetched_at,
            "symbol": symbol,
            "address": addr,
            "dex": dex,
            "chainId": chain,
            "priceUsd": price,
            "volume24h": vol24,
            "liquidity": liq,
            "priceChange5m": price_change_5m if price_change_5m is not None else "",
            "priceChange1h": price_change_1h if price_change_1h is not None else "",
            "priceChange24h": price_change_24h if price_change_24h is not None else "",
            "source": "dexscreener",  # Mark DexScreener tokens
            "jupiter_validated": False,  # Not validated by Jupiter
        })
    
    jupiter_count = len([r for r in all_rows if r.get("source") == "jupiter"])
    dexscreener_count = len([r for r in all_rows if r.get("source") == "dexscreener"])
    
    print(f"✅ Found {valid_tokens_count} valid tokens:")
    print(f"   🚀 Jupiter: {jupiter_count} tokens")
    print(f"   🔍 DexScreener: {dexscreener_count} tokens")
    print(f"⛔ Skipped {unsupported_chain_count} tokens from unsupported chains")

    if all_rows:
        _append_all_to_csv(all_rows)
    else:
        print("ℹ️ No pairs found to log.")

    # Enhanced trading token selection with scoring
    tokens_for_trading = []
    
    scored_tokens = []
    
    for row in all_rows:
        chain = (row["chainId"] or "").lower()
        
        symbol = (row["symbol"] or "").upper()
        volume24h = row["volume24h"]
        liquidity = row["liquidity"]
        
        # Calculate quality score with address for established token detection
        score = calculate_token_score(symbol, volume24h, liquidity, chain, row["address"])
        
        # Boost score for Jupiter-discovered tokens
        jupiter_boost = 0.0
        if row.get("jupiter_validated", False):
            jupiter_priority_boost = get_config_float("token_discovery.jupiter_priority_boost", 1.0)
            jupiter_boost = jupiter_priority_boost
            
            # Additional boosts for Jupiter tokens
            organic_score = row.get("organic_score", 0)
            if organic_score and organic_score > 80:
                jupiter_boost += 0.5  # High organic score bonus
            
            if row.get("is_verified", False):
                jupiter_boost += 0.3  # Verified token bonus
        
        score += jupiter_boost
        
        # Keyword filtering
        blocked = any(k in symbol for k in EXCLUDED_KEYWORDS)
        if blocked and ENFORCE_KEYWORDS:
            print(f"⛔ {symbol} — blocked keyword.")
            continue
        
        print(f"🧪 {symbol} | Vol: ${volume24h:,.0f} | LQ: ${liquidity:,.0f} | Score: {score}/8 | Chain: {chain}")
        
        # Only include tokens with good scores (balanced minimum for quality)
        if score >= 1:  # Minimum score of 1 for quality tokens (balanced)
            token_dict = {
                "symbol": symbol,
                "address": row["address"],
                "dex": row["dex"],
                "chainId": row["chainId"],
                "priceUsd": row["priceUsd"],
                "volume24h": volume24h,
                "liquidity": liquidity,
                "priceChange5m": row.get("priceChange5m", ""),
                "priceChange1h": row.get("priceChange1h", ""),
                "priceChange24h": row.get("priceChange24h", ""),
                "score": score,
                "source": row.get("source", "unknown"),
                "jupiter_validated": row.get("jupiter_validated", False),
            }
            
            # Add Jupiter-specific metadata if available
            if row.get("jupiter_validated"):
                token_dict["organic_score"] = row.get("organic_score")
                token_dict["holder_count"] = row.get("holder_count")
                token_dict["is_verified"] = row.get("is_verified", False)
                token_dict["trending_position"] = row.get("trending_position")
                token_dict["trending_interval"] = row.get("trending_interval")
            
            scored_tokens.append(token_dict)
        else:
            print(f"⛔ {symbol} rejected - score too low: {score}/8")
    
    # Prioritize tokens: Jupiter trending → established → known tradeable → unknown
    jupiter_trending = []
    established_tokens = []
    known_tradeable = []
    unknown_tokens = []
    
    for token in scored_tokens:
        # Highest priority: Jupiter-discovered tokens
        if token.get("jupiter_validated", False):
            jupiter_trending.append(token)
            pos = token.get("trending_position", "?")
            interval = token.get("trending_interval", "?")
            print(f"🚀 {token['symbol']} - JUPITER TRENDING (#{pos} in {interval}): Score {token['score']:.2f}")
        elif token["address"] in ESTABLISHED_TOKENS:
            established_tokens.append(token)
            print(f"🏆 {token['symbol']} - ESTABLISHED TOKEN (whitelisted): {ESTABLISHED_TOKENS[token['address']]}")
        elif token["address"] in KNOWN_TRADEABLE_TOKENS:
            known_tradeable.append(token)
            print(f"✅ {token['symbol']} - Known tradeable token (whitelisted)")
        else:
            unknown_tokens.append(token)
    
    # Sort by score (highest first) within each category
    # For Jupiter tokens, also sort by trending position (lower = better)
    jupiter_trending.sort(key=lambda x: (x["score"], -x.get("trending_position", 999)), reverse=True)
    established_tokens.sort(key=lambda x: x["score"], reverse=True)
    known_tradeable.sort(key=lambda x: x["score"], reverse=True)
    unknown_tokens.sort(key=lambda x: x["score"], reverse=True)
    
    # Combine: Jupiter trending first (highest priority), then established, then known tradeable, then unknown
    all_tokens = jupiter_trending + established_tokens + known_tradeable + unknown_tokens
    
    # Sort by volume and liquidity before applying diversity filter
    # This ensures we keep the highest quality tokens for each symbol
    all_tokens.sort(key=lambda x: (x.get("volume24h", 0) + x.get("liquidity", 0)), reverse=True)
    
    # Ensure symbol diversity (increased from 5 to 10 to allow more PUMP tokens)
    diverse_tokens = ensure_symbol_diversity(all_tokens, max_same_symbol=20)  # Increased for more opportunities
    
    # Apply tradeability filter but make it less aggressive
    from tradeability_checker import filter_tradeable_tokens
    tradeable_tokens = filter_tradeable_tokens(diverse_tokens, max_checks=25)  # INCREASED for more opportunities
    
    # Take top tokens up to limit
    tokens_for_trading = tradeable_tokens[:limit]
    
    # Remove internal metadata from final output
    for token in tokens_for_trading:
        token.pop("score", None)
        # Keep source and jupiter_validated for debugging, but remove other Jupiter metadata
        token.pop("organic_score", None)
        token.pop("holder_count", None)
        token.pop("trending_position", None)
        token.pop("trending_interval", None)
        token.pop("name", None)
    
    print(f"🎯 Selected {len(tokens_for_trading)} tradeable, high-quality tokens for trading")
    
    if not tokens_for_trading:
        print("⚠️ No tokens passed enhanced filtering. No trades this cycle.")
        # Don't fall back to any tokens - just return empty list
        tokens_for_trading = []

    return tokens_for_trading

if __name__ == "__main__":
    fetch_trending_tokens()
