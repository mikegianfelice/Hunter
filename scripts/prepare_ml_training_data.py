#!/usr/bin/env python3
"""
Prepare ML Training Data from trade_log.csv
Uses DexScreener API to fetch historical candles and creates training dataset
with advanced features optimized for small datasets (~200 trades)
"""

import sys
import os
import csv
import json
import time
import math
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import hashlib
from dotenv import load_dotenv
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Load environment variables from .env files (same pattern as rest of codebase)
system_env_path = project_root / "system" / ".env"
if system_env_path.exists():
    try:
        load_dotenv(system_env_path)
    except PermissionError:
        pass  # Silently skip if can't read
load_dotenv()  # Also load from root .env as fallback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# File paths
TRADE_LOG_FILE = project_root / "data" / "trade_log.csv"
OUTPUT_FILE = project_root / "data" / "ml_training_data.json"
CACHE_DIR = project_root / "data" / "candle_cache"
API_CALL_LOG = project_root / "data" / "ml_api_calls.log"

# Configuration
DEXSCREENER_BASE_URL = "https://api.dexscreener.com"
COINDESK_BASE_URL = "https://api.coindesk.com/v1"  # CoinDesk Data API
WINDOW_BEFORE_ENTRY_HOURS = 24  # Hours before entry to fetch
WINDOW_AFTER_EXIT_HOURS = 24    # Hours after exit to fetch
FAST_MODE = False  # Set to True for ±6h window
if FAST_MODE:
    WINDOW_BEFORE_ENTRY_HOURS = 6
    WINDOW_AFTER_EXIT_HOURS = 6

# Data source priority: on_chain_swaps > CoinDesk > CoinCap > market_data_fetcher (for non-Solana)
# Note: For Solana tokens, market_data_fetcher uses Helius RPC, so it's skipped
USE_ON_CHAIN_SWAPS = True  # Set to True to use on-chain swap events (recommended, no API limits)
USE_COINDESK_API = True  # Set to False to skip CoinDesk and use CoinCap/market_data_fetcher only

# ML Target Configuration
TARGET_PCT = 5.0   # +5% target
STOP_PCT = 3.0     # -3% stop
TIME_WINDOW_MINUTES = 60  # 60 minutes window

# Create cache directory
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def determine_chain_id(token_address: str) -> str:
    """Determine chain ID based on token address format"""
    # Solana addresses are 43-44 characters (base58)
    # Ethereum/Base addresses are 42 characters starting with 0x
    if len(token_address) in [43, 44]:
        return "solana"
    elif len(token_address) == 42 and token_address.startswith("0x"):
        # Default to ethereum (could be refined with config)
        return "ethereum"
    else:
        return "solana"  # Default


def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse timestamp from trade_log.csv format"""
    try:
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        try:
            return datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")


def estimate_exit_time(entry_time: datetime, pnl_pct: float, reason: str) -> datetime:
    """
    Estimate exit time based on trade characteristics.
    Since CSV doesn't have exit_time, we estimate it.
    """
    # For stop losses, typically faster exits (minutes to hours)
    if "stop_loss" in reason.lower():
        # Estimate based on typical stop loss timing (15 min to 4 hours)
        # More negative = faster exit typically
        if pnl_pct < -8:
            hours = 0.25  # 15 minutes
        elif pnl_pct < -7:
            hours = 1.0   # 1 hour
        else:
            hours = 2.0   # 2 hours
    elif "take_profit" in reason.lower():
        # Take profits can take longer (hours to days)
        if pnl_pct > 50:
            hours = 24.0  # 1 day
        elif pnl_pct > 20:
            hours = 12.0  # 12 hours
        elif pnl_pct > 10:
            hours = 6.0   # 6 hours
        else:
            hours = 3.0   # 3 hours
    else:
        # Default estimate
        hours = 4.0
    
    return entry_time + timedelta(hours=hours)


def discover_pair(token_address: str, chain: str) -> Optional[Dict]:
    """
    Step 1: Discover pair from DexScreener
    GET https://api.dexscreener.com/latest/dex/tokens/{TOKEN_ADDRESS}
    """
    try:
        url = f"{DEXSCREENER_BASE_URL}/latest/dex/tokens/{token_address}"
        logger.debug(f"Discovering pair: {url}")
        
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            logger.warning(f"DexScreener API error {response.status_code} for {token_address[:8]}...")
            return None
        
        data = response.json()
        pairs = data.get('pairs', [])
        
        if not pairs:
            logger.warning(f"No pairs found for {token_address[:8]}...")
            return None
        
        # Filter pairs by chain and find the most liquid one
        chain_pairs = [p for p in pairs if p.get('chainId', '').lower() == chain.lower()]
        
        if not chain_pairs:
            # Try to find any pair (chain might be different)
            chain_pairs = pairs
        
        # Sort by liquidity (descending) and take the first
        chain_pairs.sort(key=lambda x: float(x.get('liquidity', {}).get('usd', 0) or 0), reverse=True)
        
        pair = chain_pairs[0]
        logger.info(f"✅ Found pair: {pair.get('pairAddress', 'N/A')[:8]}... on {pair.get('dexId', 'N/A')}")
        
        return {
            'pairAddress': pair.get('pairAddress'),
            'chainId': pair.get('chainId', chain),
            'dexId': pair.get('dexId'),
            'liquidity': float(pair.get('liquidity', {}).get('usd', 0) or 0),
            'volume24h': float(pair.get('volume', {}).get('h24', 0) or 0),
        }
        
    except Exception as e:
        logger.error(f"Error discovering pair for {token_address[:8]}...: {e}")
        return None


def get_cache_key(pair_address: str, chain: str, interval: str, start_ts: int, end_ts: int) -> str:
    """Generate cache key for candles"""
    key_str = f"{pair_address}_{chain}_{interval}_{start_ts}_{end_ts}"
    return hashlib.md5(key_str.encode()).hexdigest()


def load_cached_candles(cache_key: str) -> Optional[List[Dict]]:
    """Load candles from cache if exists"""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                # Check if cache is still valid (24 hours)
                cache_time = data.get('cached_at', 0)
                if time.time() - cache_time < 86400:  # 24 hours
                    logger.debug(f"✅ Using cached candles: {cache_key[:8]}...")
                    return data.get('candles', [])
        except Exception as e:
            logger.debug(f"Error loading cache: {e}")
    return None


def save_candles_to_cache(cache_key: str, candles: List[Dict]):
    """Save candles to cache"""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    try:
        with open(cache_file, 'w') as f:
            json.dump({
                'candles': candles,
                'cached_at': time.time()
            }, f)
    except Exception as e:
        logger.debug(f"Error saving cache: {e}")


def fetch_coindesk_candles(token_address: str, chain: str,
                          start_time: datetime, end_time: datetime,
                          interval: str = "1m", pair_address: str = None) -> Optional[List[Dict]]:
    """
    Fetch historical candles from CoinDesk Data API
    
    CoinDesk provides minute-by-minute historical data for 10,000+ coins.
    Requires API key (set via COINDESK_API_KEY environment variable or config).
    
    Args:
        token_address: Token address
        chain: Chain ID
        start_time: Start of time window
        end_time: End of time window
        interval: Desired interval (1m, 5m, 15m, 1h)
        pair_address: Pair address (for caching)
    """
    try:
        # Check cache first
        start_ts = int(start_time.timestamp())
        end_ts = int(end_time.timestamp())
        cache_key = get_cache_key(pair_address or token_address, chain, interval, start_ts, end_ts)
        
        cached = load_cached_candles(cache_key)
        if cached:
            return cached
        
        # Get API key from .env file (COINDESK_API_KEY)
        api_key = (os.getenv('COINDESK_API_KEY') or '').strip()
        
        if not api_key:
            logger.debug("CoinDesk API key not found in .env (COINDESK_API_KEY), skipping CoinDesk API")
            return None
        
        # Map interval to CoinDesk format
        interval_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '1h': '1h',
        }
        coindesk_interval = interval_map.get(interval, '1m')
        
        # CoinDesk API endpoint format (check documentation for exact format)
        # Note: This is a placeholder - actual endpoint may vary
        # CoinDesk typically uses: /v1/data/ohlcv/{symbol}?start={start}&end={end}&interval={interval}
        
        # For now, we'll need to map token address to symbol
        # This is a limitation - CoinDesk may need symbol mapping
        # Try to get symbol from pair_info or use a mapping
        
        # Convert timestamps to ISO format or Unix timestamp (check API docs)
        start_iso = start_time.isoformat()
        end_iso = end_time.isoformat()
        
        # Try CoinDesk API endpoint
        # Note: Endpoint format based on CoinDesk Data API documentation
        # May need adjustment based on actual API structure
        # CoinDesk typically requires symbol mapping (not token addresses)
        # For DEX tokens, we may need to use pair identifier or symbol
        
        # Try multiple endpoint formats
        endpoints_to_try = [
            f"{COINDESK_BASE_URL}/data/ohlcv",
            f"{COINDESK_BASE_URL}/ohlcv",
            f"https://api.coindesk.com/v1/data/historical",
        ]
        
        # Try to map token to symbol (simplified - may need proper mapping)
        # For now, use token address as-is, but CoinDesk may require symbol
        symbol = token_address  # Placeholder - may need symbol lookup
        
        response = None
        successful_url = None
        for url in endpoints_to_try:
            # Try different parameter formats
            param_variants = [
                {
                    'symbol': symbol,
                    'start': start_iso,
                    'end': end_iso,
                    'interval': coindesk_interval,
                },
                {
                    'symbol': symbol,
                    'start': int(start_ts),
                    'end': int(end_ts),
                    'interval': coindesk_interval,
                },
            ]
            
            headers = {'X-API-Key': api_key} if api_key else {}
            if api_key:
                # Also try API key in params
                for params in param_variants:
                    params['api_key'] = api_key
                    try:
                        test_response = requests.get(url, params=params, headers=headers, timeout=10)
                        if test_response.status_code == 200:
                            response = test_response
                            successful_url = url
                            break
                    except:
                        continue
                if response:
                    break
            
            # Try without API key in params (only in headers)
            for params in param_variants:
                try:
                    test_response = requests.get(url, params=params, headers=headers, timeout=10)
                    if test_response.status_code == 200:
                        response = test_response
                        successful_url = url
                        break
                except:
                    continue
            if response:
                break
        
        if response and response.status_code == 200:
            data = response.json()
            candles = data.get('data', data.get('candles', []))
            
            if candles:
                # Convert to our format
                formatted_candles = []
                for candle in candles:
                    # CoinDesk format may vary - adjust based on actual response
                    formatted_candles.append({
                        'time': candle.get('timestamp', candle.get('time', 0)),
                        'timestamp': candle.get('timestamp', candle.get('time', 0)),
                        'open': float(candle.get('open', 0)),
                        'high': float(candle.get('high', 0)),
                        'low': float(candle.get('low', 0)),
                        'close': float(candle.get('close', 0)),
                        'volume': float(candle.get('volume', 0)),
                    })
                
                # Sort and filter
                formatted_candles.sort(key=lambda x: x['time'])
                formatted_candles = [c for c in formatted_candles 
                                    if start_ts <= c['time'] <= end_ts]
                
                if len(formatted_candles) >= 10:
                    save_candles_to_cache(cache_key, formatted_candles)
                    logger.info(f"✅ Got {len(formatted_candles)} {interval} candles from CoinDesk")
                    return formatted_candles
        
        logger.debug(f"CoinDesk API returned {response.status_code}, falling back")
        return None
        
    except Exception as e:
        logger.debug(f"CoinDesk API error: {e}")
        return None


# Note: CoinDesk API Integration
# To use CoinDesk API:
# 1. Register at https://developers.coindesk.com/ and get an API key
# 2. Set COINDESK_API_KEY environment variable or add to config.yaml:
#    coindesk_api_key: "your_api_key_here"
# 3. Verify the API endpoint format matches your CoinDesk plan
# 4. CoinDesk may require symbol mapping (not token addresses) - adjust as needed


def fetch_on_chain_swap_candles(token_address: str, chain: str,
                                start_time: datetime, end_time: datetime,
                                interval: str = "1m", pair_address: str = None) -> Optional[List[Dict]]:
    """
    Fetch historical candles from on-chain swap events (Step A - On-chain price pipeline).
    
    This method:
    1. Fetches swap events directly from the blockchain
    2. Reconstructs price from swap amounts
    3. Aggregates into OHLCV candles (1m, 5m, 15m, 1h)
    
    Works for Ethereum, Base, and Solana chains without requiring third-party APIs.
    
    Args:
        token_address: Token address
        chain: Chain ID (solana, ethereum, base)
        start_time: Start of time window
        end_time: End of time window
        interval: Desired interval (1m, 5m, 15m, 1h)
        pair_address: Pair address (for caching)
    """
    try:
        # Check cache first
        start_ts = int(start_time.timestamp())
        end_ts = int(end_time.timestamp())
        cache_key = get_cache_key(pair_address or token_address, chain, interval, start_ts, end_ts)
        
        cached = load_cached_candles(cache_key)
        if cached:
            return cached
        
        logger.info(f"  Fetching on-chain swap events for {token_address[:8]}... on {chain}")
        
        swaps = []
        if chain.lower() == "solana":
            swaps = _fetch_solana_swaps(token_address, start_time, end_time, pair_address)
        elif chain.lower() in ["ethereum", "base"]:
            swaps = _fetch_evm_swaps(token_address, chain, start_time, end_time, pair_address)
        
        if not swaps:
            logger.debug(f"No swap events found for {token_address[:8]}...")
            return None
        
        # Convert swaps to candles
        candles = _swaps_to_candles(swaps, start_time, end_time, interval)
        
        if len(candles) >= 10:
            save_candles_to_cache(cache_key, candles)
            logger.info(f"✅ Got {len(candles)} {interval} candles from on-chain swaps")
            return candles
        
        logger.debug(f"Insufficient candles from on-chain swaps ({len(candles)})")
        return None
        
    except Exception as e:
        logger.debug(f"On-chain swap fetch error: {e}")
        return None


def _fetch_solana_swaps(token_address: str, start_time: datetime, end_time: datetime,
                        pair_address: str = None) -> List[Dict]:
    """
    Fetch swap transactions from Solana DEX programs (Raydium, Orca, Jupiter, etc.)
    
    Strategy:
    1. If pair_address is provided, query transactions for that specific pool (most efficient)
    2. Otherwise, query transactions involving the token mint (less efficient but works)
    
    Uses pre/post token balances to extract swap data, which works across all DEX programs.
    """
    try:
        from solana.rpc.api import Client
        from solders.pubkey import Pubkey
        from src.config.secrets import SOLANA_RPC_URL
        
        # Use public RPC or configured RPC (but NOT Helius API endpoint)
        rpc_url = SOLANA_RPC_URL
        if "helius-rpc.com" in rpc_url or "helius.xyz" in rpc_url:
            # Use public RPC instead of Helius to avoid API usage
            logger.debug("Using public Solana RPC instead of Helius for on-chain swaps")
            rpc_url = "https://api.mainnet-beta.solana.com"
        
        client = Client(rpc_url)
        
        # USDC mint (common quote token on Solana)
        USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        WSOL_MINT = "So11111111111111111111111111111111111111112"
        
        swaps = []
        query_address = None
        
        # Prefer pair_address (pool account) if provided, otherwise use token address
        if pair_address:
            try:
                query_address = Pubkey.from_string(pair_address)
                logger.debug(f"Querying swaps from pool: {pair_address[:8]}...")
            except Exception:
                logger.debug(f"Invalid pair_address, falling back to token address")
                query_address = Pubkey.from_string(token_address)
        else:
            query_address = Pubkey.from_string(token_address)
        
        try:
            # Get signatures for the address (pool or token)
            # Limit to reasonable number to avoid too many RPC calls
            max_signatures = 1000
            signatures_response = client.get_signatures_for_address(
                query_address,
                limit=max_signatures,
            )
            
            if not signatures_response.value:
                logger.debug(f"No signatures found for address")
                return []
            
            logger.debug(f"Found {len(signatures_response.value)} signatures, parsing transactions...")
            
            # Parse transactions
            parsed_count = 0
            for sig_info in signatures_response.value:
                try:
                    # Filter by time window
                    if sig_info.block_time:
                        block_ts = sig_info.block_time
                        if block_ts < start_time.timestamp():
                            # Signatures are in reverse chronological order, so we can break early
                            break
                        if block_ts > end_time.timestamp():
                            continue
                    
                    # Get transaction with parsed format to access token balances
                    # Use requests directly for better control over JSON parsing
                    import json as json_lib
                    rpc_payload = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "getTransaction",
                        "params": [
                            str(sig_info.signature),
                            {
                                "encoding": "jsonParsed",
                                "maxSupportedTransactionVersion": 0
                            }
                        ]
                    }
                    
                    import requests as req_lib
                    rpc_response = req_lib.post(rpc_url, json=rpc_payload, timeout=10)
                    
                    if rpc_response.status_code == 200:
                        rpc_data = rpc_response.json()
                        tx_data = rpc_data.get('result')
                        
                        if tx_data and sig_info.block_time:
                            # tx_data is already a dict from JSON response
                            swap_data = _parse_solana_swap_tx(
                                tx_data,
                                token_address,
                                USDC_MINT,
                                WSOL_MINT,
                                sig_info.block_time
                            )
                        else:
                            swap_data = None
                    else:
                        swap_data = None
                        
                        if swap_data and swap_data.get('price', 0) > 0:
                            swaps.append(swap_data)
                            parsed_count += 1
                            
                            # Limit total swaps to avoid memory issues
                            if len(swaps) >= 5000:
                                logger.debug(f"Reached swap limit (5000), stopping")
                                break
                    
                    # Rate limiting: add small delay every N transactions
                    if parsed_count % 10 == 0 and parsed_count > 0:
                        time.sleep(0.1)  # Small delay to avoid rate limits
                        
                except Exception as e:
                    logger.debug(f"Error parsing Solana tx {sig_info.signature}: {e}")
                    continue
            
            logger.info(f"Parsed {parsed_count} swap events from {len(signatures_response.value)} signatures")
            
        except Exception as e:
            logger.debug(f"Error fetching Solana signatures: {e}")
        
        # Sort swaps by timestamp
        swaps.sort(key=lambda x: x.get('timestamp', 0))
        
        logger.info(f"Found {len(swaps)} valid Solana swap events")
        return swaps
        
    except Exception as e:
        logger.debug(f"Error fetching Solana swaps: {e}")
        return []


def _parse_solana_swap_tx(tx_data: Dict, token_address: str, usdc_mint: str, wsol_mint: str, block_time: float) -> Optional[Dict]:
    """
    Parse Solana transaction to extract swap data from token balance changes.
    
    This method works by analyzing pre/post token balances in the transaction,
    which works across all DEX programs (Raydium, Orca, Jupiter, etc.) without
    needing to parse specific instruction formats.
    
    Args:
        tx_data: Transaction data dict from Solana RPC getTransaction (JSON format)
        token_address: The token mint address we're tracking
        usdc_mint: USDC mint address (quote token)
        wsol_mint: Wrapped SOL mint address (alternative quote token)
        block_time: Transaction block time (Unix timestamp)
    
    Returns:
        Dict with swap data: timestamp, token_amount, quote_amount, price, volume_usd
    """
    try:
        # tx_data should be a dict from JSON-RPC response
        if not isinstance(tx_data, dict):
            return None
        
        meta = tx_data.get('meta', {})
        if not meta:
            return None
        
        # Check if transaction was successful
        if meta.get('err') is not None:
            return None  # Skip failed transactions
        
        # Get token balances (pre and post)
        pre_token_balances = meta.get('preTokenBalances', [])
        post_token_balances = meta.get('postTokenBalances', [])
        
        if not pre_token_balances or not post_token_balances:
            return None
        
        # Normalize mint addresses to lowercase for comparison
        token_address_lower = token_address.lower()
        usdc_mint_lower = usdc_mint.lower()
        wsol_mint_lower = wsol_mint.lower()
        
        # Create maps by account index for easy lookup
        pre_map = {}
        post_map = {}
        
        for bal in pre_token_balances:
            account_idx = bal.get('accountIndex')
            if account_idx is not None:
                ui_amount = bal.get('uiTokenAmount', {}).get('uiAmount', 0)
                mint = bal.get('mint', '').lower()
                if ui_amount is not None and mint:
                    pre_map[account_idx] = {
                        'amount': float(ui_amount),
                        'mint': mint
                    }
        
        for bal in post_token_balances:
            account_idx = bal.get('accountIndex')
            if account_idx is not None:
                ui_amount = bal.get('uiTokenAmount', {}).get('uiAmount', 0)
                mint = bal.get('mint', '').lower()
                if ui_amount is not None and mint:
                    post_map[account_idx] = {
                        'amount': float(ui_amount),
                        'mint': mint
                    }
        
        # Find balance changes for our token and quote tokens
        # Sum up all balance changes (there may be multiple accounts involved in the swap)
        token_amount = 0.0
        quote_amount = 0.0
        quote_mint = None
        
        # Check all accounts for balance changes
        all_accounts = set(pre_map.keys()) | set(post_map.keys())
        
        for account_idx in all_accounts:
            pre_bal = pre_map.get(account_idx, {'amount': 0.0, 'mint': ''})
            post_bal = post_map.get(account_idx, {'amount': 0.0, 'mint': ''})
            
            # Get mint from pre or post (should be the same)
            mint = pre_bal.get('mint') or post_bal.get('mint')
            if not mint:
                continue
            
            # Calculate amount change
            pre_amount = pre_bal.get('amount', 0.0)
            post_amount = post_bal.get('amount', 0.0)
            amount_change = post_amount - pre_amount
            
            # Skip if no change
            if abs(amount_change) < 0.000001:
                continue
            
            # Check if this is our target token
            if mint == token_address_lower:
                # Sum absolute changes (we care about total volume, not direction)
                token_amount += abs(amount_change)
            
            # Check if this is a quote token (USDC or SOL)
            elif mint == usdc_mint_lower or mint == wsol_mint_lower:
                # Sum absolute changes
                quote_amount += abs(amount_change)
                if quote_mint is None:
                    quote_mint = mint
        
        # Need both token and quote amounts to calculate price
        if token_amount <= 0 or quote_amount <= 0:
            return None
        
        # Calculate price: quote_amount / token_amount
        # This gives us the price in quote token units per token unit
        price = quote_amount / token_amount if token_amount > 0 else 0
        
        if price <= 0:
            return None
        
        # Calculate volume in USD
        # For USDC, amount is already in USD (6 decimals)
        # For SOL, we'd need SOL price, but for simplicity we'll use quote_amount
        volume_usd = quote_amount if quote_mint == usdc_mint_lower else quote_amount * 150.0  # Approximate SOL price
        
        return {
            'timestamp': block_time,
            'token_amount': token_amount,
            'quote_amount': quote_amount,
            'quote_mint': quote_mint,
            'price': price,
            'volume_usd': volume_usd,
        }
        
    except Exception as e:
        logger.debug(f"Error parsing Solana swap transaction: {e}")
        return None


def _fetch_evm_swaps(token_address: str, chain: str, start_time: datetime, end_time: datetime,
                     pair_address: str = None) -> List[Dict]:
    """Fetch swap events from EVM chains (Ethereum, Base) using Web3"""
    try:
        from web3 import Web3
        from src.config.secrets import INFURA_URL, BASE_RPC_URL
        
        # Get RPC URL for chain
        if chain.lower() == "ethereum":
            rpc_url = INFURA_URL or "https://eth.llamarpc.com"
        elif chain.lower() == "base":
            rpc_url = BASE_RPC_URL or "https://mainnet.base.org"
        else:
            return []
        
        if not rpc_url:
            logger.debug(f"No RPC URL configured for {chain}")
            return []
        
        w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not w3.is_connected():
            logger.debug(f"Failed to connect to {chain} RPC")
            return []
        
        # Uniswap V2 Swap event signature
        swap_topic = w3.keccak(text="Swap(address,address,uint256,uint256,uint256,uint256,address)").hex()
        
        # Get pair address if not provided (would need to query factory)
        # For now, assume pair_address is provided or we skip
        if not pair_address:
            logger.debug("Pair address required for EVM swaps")
            return []
        
        pair_contract = w3.to_checksum_address(pair_address)
        
        # Convert time range to block numbers
        latest_block = w3.eth.block_number
        start_block = _estimate_block_from_timestamp(w3, start_time.timestamp())
        end_block = min(_estimate_block_from_timestamp(w3, end_time.timestamp()), latest_block)
        
        # Fetch swap logs
        logs = w3.eth.get_logs({
            "fromBlock": max(start_block - 1000, 0),  # Add buffer
            "toBlock": end_block,
            "address": pair_contract,
            "topics": [swap_topic]
        })
        
        # Parse swap logs
        swaps = []
        for log in logs:
            try:
                # Decode swap event
                # Uniswap V2 Swap: Swap(address indexed sender, uint amount0In, uint amount1In, 
                #                      uint amount0Out, uint amount1Out, address indexed to)
                receipt = w3.eth.get_transaction_receipt(log['transactionHash'])
                block = w3.eth.get_block(log['blockNumber'])
                timestamp = block['timestamp']
                
                # Decode log data (amounts are in data, not topics)
                # Topic 0: event signature
                # Topic 1: sender
                # Topic 2: to
                # Data: amount0In (32 bytes) + amount1In (32 bytes) + amount0Out (32 bytes) + amount1Out (32 bytes)
                
                data = log['data']
                amount0_in = int(data[0:66], 16)
                amount1_in = int(data[66:130], 16)
                amount0_out = int(data[130:194], 16)
                amount1_out = int(data[194:258], 16)
                
                # Determine token0/token1 (would need to query pair contract)
                # Simplified: assume token0 is quote token (USDC/WETH) and token1 is token
                # In production, query pair contract to determine which is which
                
                # Calculate price: if swapping token -> quote, price = quote_out / token_in
                if amount0_in > 0 and amount1_out > 0:
                    # token -> quote (token is token1, quote is token0)
                    token_amount = amount1_out
                    quote_amount = amount0_in
                elif amount1_in > 0 and amount0_out > 0:
                    # quote -> token (token is token1, quote is token0)
                    token_amount = amount1_in
                    quote_amount = amount0_out
                else:
                    continue
                
                if token_amount > 0:
                    price = quote_amount / token_amount
                    
                    swaps.append({
                        'timestamp': timestamp,
                        'token_amount': token_amount,
                        'quote_amount': quote_amount,
                        'price': price,
                        'volume_usd': quote_amount / 1e6 if chain.lower() == "base" else quote_amount / 1e18,  # Simplified
                    })
            except Exception as e:
                logger.debug(f"Error parsing swap log: {e}")
                continue
        
        logger.info(f"Found {len(swaps)} {chain} swap events")
        return swaps
        
    except Exception as e:
        logger.debug(f"Error fetching EVM swaps: {e}")
        return []


def _estimate_block_from_timestamp(w3, timestamp: int) -> int:
    """Estimate block number from timestamp (approximate)"""
    try:
        latest_block = w3.eth.block_number
        latest_block_data = w3.eth.get_block('latest')
        latest_timestamp = latest_block_data['timestamp']
        current_time = time.time()
        
        # Average block time: ~12s for Ethereum, ~2s for Base
        block_time = 12 if 'ethereum' in str(w3.provider).lower() else 2
        
        # Estimate blocks to go back
        time_diff = current_time - timestamp
        blocks_back = int(time_diff / block_time)
        
        return max(latest_block - blocks_back, 0)
    except Exception:
        return 0


def _swaps_to_candles(swaps: List[Dict], start_time: datetime, end_time: datetime,
                      interval: str = "1m") -> List[Dict]:
    """
    Convert swap events to OHLCV candles using pandas resampling.
    
    This follows the Step A approach:
    1. Aggregate swaps by time interval
    2. Calculate OHLC (open=first, high=max, low=min, close=last)
    3. Sum volume for each interval
    """
    if not swaps:
        return []
    
    try:
        # Create DataFrame from swaps
        df = pd.DataFrame(swaps)
        
        # Ensure we have required columns
        if 'timestamp' not in df.columns or 'price' not in df.columns:
            logger.warning("Missing required columns in swaps data")
            return []
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        
        # Set datetime as index
        df.set_index('datetime', inplace=True)
        
        # Ensure volume_usd column exists
        if 'volume_usd' not in df.columns:
            df['volume_usd'] = 0.0
        
        # Map interval to pandas frequency
        freq_map = {
            '1m': '1T',   # 1 minute
            '5m': '5T',   # 5 minutes
            '15m': '15T', # 15 minutes
            '1h': '1H'    # 1 hour
        }
        freq = freq_map.get(interval, '1T')
        
        # Resample to candles using OHLCV aggregation
        # First, last, max, min for price (OHLC)
        # Sum for volume
        candles_df = df.resample(freq, label='left', closed='left').agg({
            'price': ['first', 'max', 'min', 'last'],
            'volume_usd': 'sum'
        })
        
        # Flatten column names
        candles_df.columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Filter to time range (ensure timezone aware comparison)
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=pd.Timestamp.now().tz)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=pd.Timestamp.now().tz)
            
        candles_df = candles_df[(candles_df.index >= start_time) & (candles_df.index <= end_time)]
        
        # Remove rows where open is NaN (no swaps in that interval)
        candles_df = candles_df.dropna(subset=['open'])
        
        # Convert to list of dicts
        candles = []
        for idx, row in candles_df.iterrows():
            try:
                candles.append({
                    'time': int(idx.timestamp()),
                    'timestamp': int(idx.timestamp()),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume'] if pd.notna(row['volume']) else 0),
                })
            except Exception as e:
                logger.debug(f"Error converting candle row: {e}")
                continue
        
        return candles
        
    except Exception as e:
        logger.error(f"Error converting swaps to candles: {e}", exc_info=True)
        return []


def fetch_coincap_candles(token_address: str, chain: str,
                          start_time: datetime, end_time: datetime,
                          interval: str = "1m", pair_address: str = None) -> Optional[List[Dict]]:
    """
    Fetch historical candles from CoinCap API (primary data source).
    
    CoinCap provides historical price data for major cryptocurrencies.
    Note: CoinCap only supports major tokens by asset ID, not arbitrary DEX token addresses.
    
    Args:
        token_address: Token address (may need to map to CoinCap asset ID)
        chain: Chain ID (solana, ethereum, etc.)
        start_time: Start of time window
        end_time: End of time window
        interval: Desired interval (1m, 5m, 15m, 1h)
        pair_address: Pair address (for caching)
    """
    try:
        # Check cache first
        start_ts = int(start_time.timestamp())
        end_ts = int(end_time.timestamp())
        cache_key = get_cache_key(pair_address or token_address, chain, interval, start_ts, end_ts)
        
        cached = load_cached_candles(cache_key)
        if cached:
            return cached
        
        # Import market_data_fetcher to use its CoinCap integration
        from src.utils.market_data_fetcher import market_data_fetcher
        
        # CoinCap API requires asset IDs, not token addresses
        # For major tokens, we can use market_data_fetcher's CoinCap integration
        # Map common tokens to CoinCap asset IDs
        asset_id_map = {
            'bitcoin': 'bitcoin',
            'ethereum': 'ethereum',
            'solana': 'solana',
            'usd-coin': 'usd-coin',
            'cardano': 'cardano',
            'polygon': 'polygon',
            'chainlink': 'chainlink',
            'uniswap': 'uniswap',
        }
        
        # Try to get asset ID from token address (simplified - may need better mapping)
        # For now, skip CoinCap for arbitrary DEX tokens
        logger.debug(f"CoinCap requires asset IDs, not token addresses. Skipping for {token_address[:8]}...")
        return None
        
    except Exception as e:
        logger.debug(f"CoinCap API error: {e}")
        return None


def fetch_dexscreener_candles(pair_address: str, chain: str, 
                              start_time: datetime, end_time: datetime,
                              interval: str = "1m", token_address: str = None) -> Optional[List[Dict]]:
    """
    Step 2: Fetch historical candles
    
    Note: DexScreener free API doesn't provide historical candles endpoint.
    Primary sources: On-chain swaps > CoinDesk API > CoinCap API > market_data_fetcher (The Graph/CoinGecko for Ethereum/Base only)
    
    Step A - On-chain price pipeline (recommended):
    - Fetches swap events directly from blockchain (no API limits)
    - Reconstructs price from swap amounts
    - Aggregates into OHLCV candles (1m, 5m, 15m, 1h)
    
    IMPORTANT: For Solana tokens, market_data_fetcher uses Helius RPC, so it's skipped to avoid Helius usage.
    
    Args:
        pair_address: Pair address (used for caching, not for API call)
        chain: Chain ID (solana, ethereum, etc.)
        start_time: Start of time window
        end_time: End of time window
        interval: Desired interval (1m, 5m, 15m)
        token_address: Token address for API call (required)
    """
    try:
        # Check cache first
        start_ts = int(start_time.timestamp())
        end_ts = int(end_time.timestamp())
        cache_key = get_cache_key(pair_address, chain, interval, start_ts, end_ts)
        
        cached = load_cached_candles(cache_key)
        if cached:
            return cached
        
        # Try on-chain swaps first (Step A - On-chain price pipeline, recommended)
        if USE_ON_CHAIN_SWAPS and token_address and pair_address:
            logger.info(f"  Trying on-chain swap events for historical candles...")
            swap_candles = fetch_on_chain_swap_candles(
                token_address, chain, start_time, end_time, interval, pair_address
            )
            if swap_candles:
                return swap_candles
        
        # Try CoinDesk API if enabled
        if USE_COINDESK_API and token_address:
            logger.info(f"  Trying CoinDesk API for historical candles...")
            coindesk_candles = fetch_coindesk_candles(
                token_address, chain, start_time, end_time, interval, pair_address
            )
            if coindesk_candles:
                return coindesk_candles
        
        # Try CoinCap API (requires asset ID mapping)
        if token_address:
            logger.info(f"  Trying CoinCap API for historical candles...")
            coincap_candles = fetch_coincap_candles(
                token_address, chain, start_time, end_time, interval, pair_address
            )
            if coincap_candles:
                return coincap_candles
        
        # For non-Solana chains, use market_data_fetcher (uses The Graph/CoinGecko, NOT Helius)
        if chain.lower() != "solana":
            logger.info(f"  Using market_data_fetcher (The Graph/CoinGecko) for {chain} chain")
            
            if not token_address:
                logger.warning(f"  Token address required for fallback, skipping {pair_address[:8]}...")
                return None
            
            # Import market_data_fetcher for fallback (only for Ethereum/Base, not Solana)
            from src.utils.market_data_fetcher import market_data_fetcher
            
            # Calculate hours needed
            hours = int((end_time - start_time).total_seconds() / 3600) + 1
            
            # Fetch using market_data_fetcher with target_timestamp
            # For Ethereum/Base, this uses The Graph or CoinGecko (NOT Helius)
            entry_ts = start_time.timestamp() + (hours * 3600 / 2)  # Middle of window
            candles = market_data_fetcher.get_candlestick_data(
                token_address=token_address,
                chain_id=chain,
                hours=hours,
                force_fetch=True,
                target_timestamp=entry_ts
            )
        else:
            # For Solana tokens, market_data_fetcher uses Helius - skip it to avoid Helius
            logger.warning(f"  Skipping market_data_fetcher for Solana token {token_address[:8]}... (uses Helius RPC)")
            logger.warning(f"  No alternative data source available for Solana DEX tokens (CoinCap/CoinDesk require asset IDs)")
            candles = None
        
        if not candles:
            logger.warning(f"No candles available for {token_address[:8]}...")
            return None
        
        # Convert to our format and filter by time window
        formatted_candles = []
        start_ts = start_time.timestamp()
        end_ts = end_time.timestamp()
        
        for candle in candles:
            # market_data_fetcher returns: {'time', 'open', 'high', 'low', 'close', 'volume'}
            candle_time = candle.get('time', candle.get('timestamp', 0))
            if isinstance(candle_time, str):
                try:
                    candle_time = datetime.fromisoformat(candle_time.replace('Z', '+00:00')).timestamp()
                except:
                    candle_time = 0
            
            # Filter to time window
            if start_ts <= candle_time <= end_ts:
                formatted_candles.append({
                    'time': candle_time,
                    'timestamp': candle_time,
                    'open': float(candle.get('open', 0)),
                    'high': float(candle.get('high', 0)),
                    'low': float(candle.get('low', 0)),
                    'close': float(candle.get('close', 0)),
                    'volume': float(candle.get('volume', 0)),
                })
        
        # Sort by timestamp
        formatted_candles.sort(key=lambda x: x['time'])
        
        # Resample to desired interval if needed
        # market_data_fetcher typically returns hourly candles, we need to resample
        if interval == "1m" and formatted_candles:
            # For 1m, we'd need to interpolate or fetch more granular data
            # For now, use what we have and note the limitation
            logger.debug(f"Note: Using hourly candles for 1m interval (interpolation not implemented)")
        elif interval == "5m" and formatted_candles:
            # Resample hourly to 5m (12 candles per hour)
            resampled = []
            for candle in formatted_candles:
                # Split each hour into 12 5-minute candles (simplified)
                for i in range(12):
                    resampled.append({
                        'time': candle['time'] + (i * 300),  # 5 minutes = 300 seconds
                        'timestamp': candle['timestamp'] + (i * 300),
                        'open': candle['open'] if i == 0 else candle['close'],
                        'high': candle['high'],
                        'low': candle['low'],
                        'close': candle['close'] if i == 11 else candle['open'],
                        'volume': candle['volume'] / 12,  # Distribute volume
                    })
            formatted_candles = [c for c in resampled if start_ts <= c['time'] <= end_ts]
        elif interval == "15m" and formatted_candles:
            # Resample hourly to 15m (4 candles per hour)
            resampled = []
            for candle in formatted_candles:
                for i in range(4):
                    resampled.append({
                        'time': candle['time'] + (i * 900),  # 15 minutes = 900 seconds
                        'timestamp': candle['timestamp'] + (i * 900),
                        'open': candle['open'] if i == 0 else candle['close'],
                        'high': candle['high'],
                        'low': candle['low'],
                        'close': candle['close'] if i == 3 else candle['open'],
                        'volume': candle['volume'] / 4,
                    })
            formatted_candles = [c for c in resampled if start_ts <= c['time'] <= end_ts]
        
        if len(formatted_candles) < 10:
            logger.warning(f"Insufficient candles ({len(formatted_candles)}) for {pair_address[:8]}...")
            return None
        
        # Cache the result
        save_candles_to_cache(cache_key, formatted_candles)
        
        logger.info(f"✅ Got {len(formatted_candles)} {interval} candles for {pair_address[:8]}...")
        return formatted_candles
        
    except Exception as e:
        logger.error(f"Error fetching DexScreener candles: {e}", exc_info=True)
        return None


def calculate_log_returns(candles: List[Dict], window: int = 1) -> float:
    """Calculate log returns over window"""
    if len(candles) < window + 1:
        return 0.0
    
    prices = [c['close'] for c in candles[-window-1:]]
    if prices[0] <= 0:
        return 0.0
    
    return math.log(prices[-1] / prices[0])


def calculate_ema(candles: List[Dict], period: int) -> float:
    """Calculate Exponential Moving Average"""
    if len(candles) < period:
        return candles[-1]['close'] if candles else 0.0
    
    prices = [c['close'] for c in candles[-period:]]
    multiplier = 2.0 / (period + 1)
    
    ema = prices[0]
    for price in prices[1:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))
    
    return ema


def calculate_rolling_std(candles: List[Dict], window: int = 20) -> float:
    """Calculate rolling standard deviation (volatility)"""
    if len(candles) < window:
        return 0.0
    
    prices = [c['close'] for c in candles[-window:]]
    if len(prices) < 2:
        return 0.0
    
    mean = sum(prices) / len(prices)
    variance = sum((p - mean) ** 2 for p in prices) / len(prices)
    return math.sqrt(variance)


def calculate_volume_delta(candles: List[Dict]) -> float:
    """Calculate volume delta (recent vs average)"""
    if len(candles) < 10:
        return 0.0
    
    recent_volumes = [c['volume'] for c in candles[-5:]]
    older_volumes = [c['volume'] for c in candles[-20:-5]]
    
    recent_avg = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 0.0
    older_avg = sum(older_volumes) / len(older_volumes) if older_volumes else 0.0
    
    if older_avg == 0:
        return 0.0
    
    return (recent_avg - older_avg) / older_avg


def calculate_volume_acceleration(candles: List[Dict]) -> float:
    """Calculate volume acceleration (rate of change)"""
    if len(candles) < 6:
        return 0.0
    
    volumes = [c['volume'] for c in candles[-6:]]
    if len(volumes) < 3:
        return 0.0
    
    # Calculate rate of change
    recent_change = (volumes[-1] - volumes[-3]) / volumes[-3] if volumes[-3] > 0 else 0.0
    older_change = (volumes[-3] - volumes[-6]) / volumes[-6] if volumes[-6] > 0 else 0.0
    
    return recent_change - older_change


def calculate_outcome_label(candles_1m: List[Dict], entry_price: float, 
                           entry_time: datetime, exit_time: datetime,
                           target_pct: float = TARGET_PCT, 
                           stop_pct: float = STOP_PCT,
                           time_window_minutes: int = TIME_WINDOW_MINUTES) -> Optional[int]:
    """
    Calculate outcome label: Does price move +X% before -Y% within N minutes?
    
    Returns:
        - 1 if hit +target_pct before -stop_pct within time_window
        - 0 if hit -stop_pct before +target_pct within time_window
        - None if neither hit within time_window (exclude from training)
    """
    if not candles_1m:
        return None
    
    entry_ts = entry_time.timestamp()
    window_end_ts = entry_ts + (time_window_minutes * 60)
    exit_ts = exit_time.timestamp()
    
    # Only check up to exit time or window end, whichever comes first
    check_end_ts = min(window_end_ts, exit_ts)
    
    target_price = entry_price * (1 + target_pct / 100.0)
    stop_price = entry_price * (1 - stop_pct / 100.0)
    
    # Find candles in the time window
    window_candles = [c for c in candles_1m 
                     if entry_ts <= c['time'] <= check_end_ts]
    
    if not window_candles:
        return None
    
    # Check which target was hit first
    target_hit = False
    stop_hit = False
    
    for candle in window_candles:
        high = candle['high']
        low = candle['low']
        
        # Check if target was hit in this candle
        if high >= target_price:
            target_hit = True
            break
        
        # Check if stop was hit in this candle
        if low <= stop_price:
            stop_hit = True
            break
    
    if target_hit:
        return 1
    elif stop_hit:
        return 0
    else:
        # Neither hit within time window - exclude from training
        return None


def extract_advanced_features(candles_1m: List[Dict], candles_5m: List[Dict], 
                             candles_15m: List[Dict], entry_price: float,
                             entry_time: datetime, pair_info: Dict) -> Dict:
    """
    Extract advanced features per requirements:
    - Log returns (1m, 5m, 15m)
    - EMAs (9, 21, 50)
    - Volatility (rolling std)
    - Candle range %
    - Volume features
    - Trade-relative features
    """
    try:
        features = {}
        
        # Price Features: Log returns
        features['log_return_1m'] = calculate_log_returns(candles_1m, window=1)
        features['log_return_5m'] = calculate_log_returns(candles_5m, window=1)
        features['log_return_15m'] = calculate_log_returns(candles_15m, window=1)
        
        # EMAs (9, 21, 50) - use 1m candles
        features['ema_9'] = calculate_ema(candles_1m, period=9)
        features['ema_21'] = calculate_ema(candles_1m, period=21)
        features['ema_50'] = calculate_ema(candles_1m, period=50)
        
        # Volatility (rolling std)
        features['volatility_1m'] = calculate_rolling_std(candles_1m, window=20)
        features['volatility_5m'] = calculate_rolling_std(candles_5m, window=20)
        
        # Candle range % (from most recent candle)
        if candles_1m:
            last_candle = candles_1m[-1]
            if last_candle['close'] > 0:
                features['candle_range_pct'] = (
                    (last_candle['high'] - last_candle['low']) / last_candle['close']
                ) * 100
            else:
                features['candle_range_pct'] = 0.0
        else:
            features['candle_range_pct'] = 0.0
        
        # Volume & Liquidity Features
        features['volume_delta'] = calculate_volume_delta(candles_1m)
        features['volume_acceleration'] = calculate_volume_acceleration(candles_1m)
        
        # Liquidity features (from pair_info)
        liquidity = pair_info.get('liquidity', 0)
        volume_24h = pair_info.get('volume24h', 0)
        
        if liquidity > 0:
            # Calculate volume/liquidity ratio
            features['volume_liquidity_ratio'] = volume_24h / liquidity if liquidity > 0 else 0.0
        else:
            features['volume_liquidity_ratio'] = 0.0
        
        # Estimate liquidity change (simplified - would need historical liquidity data)
        features['liquidity_change_pct'] = 0.0  # Placeholder
        
        # Trade-Relative Features
        # Minutes since pair creation (estimate from first candle)
        if candles_1m:
            first_candle_time = candles_1m[0]['time']
            pair_age_minutes = (entry_time.timestamp() - first_candle_time) / 60
            features['minutes_since_pair_creation'] = max(0, pair_age_minutes)
        else:
            features['minutes_since_pair_creation'] = 0.0
        
        # Liquidity at entry vs peak (simplified - use current liquidity as proxy)
        features['liquidity_at_entry_vs_peak'] = 1.0  # Placeholder
        
        # % move from entry (at entry time, this is 0)
        features['pct_move_from_entry'] = 0.0
        
        # Time since last high (find highest price before entry)
        if candles_1m:
            entry_ts = entry_time.timestamp()
            pre_entry_candles = [c for c in candles_1m if c['time'] < entry_ts]
            if pre_entry_candles:
                max_high = max(c['high'] for c in pre_entry_candles)
                max_high_candle = next(c for c in pre_entry_candles if c['high'] == max_high)
                time_since_high = (entry_ts - max_high_candle['time']) / 60  # minutes
                features['time_since_last_high'] = time_since_high
            else:
                features['time_since_last_high'] = 0.0
        else:
            features['time_since_last_high'] = 0.0
        
        # Additional derived features
        features['entry_price'] = entry_price
        features['price_vs_ema9'] = (entry_price - features['ema_9']) / features['ema_9'] if features['ema_9'] > 0 else 0.0
        features['price_vs_ema21'] = (entry_price - features['ema_21']) / features['ema_21'] if features['ema_21'] > 0 else 0.0
        features['price_vs_ema50'] = (entry_price - features['ema_50']) / features['ema_50'] if features['ema_50'] > 0 else 0.0
        
        # Time features
        features['hour_of_day'] = entry_time.hour
        features['day_of_week'] = entry_time.weekday()
        
        return features
        
    except Exception as e:
        logger.error(f"Error extracting features: {e}", exc_info=True)
        return {}


def load_trade_log() -> List[Dict]:
    """Load trades from trade_log.csv"""
    trades = []
    
    if not TRADE_LOG_FILE.exists():
        logger.error(f"Trade log file not found: {TRADE_LOG_FILE}")
        return trades
    
    try:
        with open(TRADE_LOG_FILE, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry_time = parse_timestamp(row['timestamp'])
                exit_time = estimate_exit_time(
                    entry_time, 
                    float(row['pnl_pct']), 
                    row['reason']
                )
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'token': row['token'],
                    'entry_price': float(row['entry_price']),
                    'exit_price': float(row['exit_price']),
                    'pnl_pct': float(row['pnl_pct']),
                    'reason': row['reason']
                })
        
        logger.info(f"Loaded {len(trades)} trades from trade_log.csv")
        return trades
        
    except Exception as e:
        logger.error(f"Error loading trade log: {e}")
        return []


def prepare_training_data():
    """Main function to prepare training data"""
    logger.info("=" * 60)
    logger.info("Starting ML Training Data Preparation (DexScreener)")
    logger.info("=" * 60)
    
    # Load trades
    trades = load_trade_log()
    if not trades:
        logger.error("No trades found. Exiting.")
        return
    
    logger.info(f"Processing {len(trades)} trades...")
    logger.info(f"Time window: -{WINDOW_BEFORE_ENTRY_HOURS}h to +{WINDOW_AFTER_EXIT_HOURS}h")
    logger.info(f"ML Target: +{TARGET_PCT}% before -{STOP_PCT}% within {TIME_WINDOW_MINUTES} minutes")
    logger.info("Press Ctrl+C to cancel, or wait 5 seconds to continue...")
    time.sleep(5)
    
    training_data = []
    api_call_count = 0
    successful_extractions = 0
    failed_extractions = 0
    
    # Log API calls
    api_log = []
    
    for i, trade in enumerate(trades, 1):
        try:
            logger.info(f"\n[{i}/{len(trades)}] Processing trade: {trade['token'][:8]}...")
            logger.info(f"  Entry: {trade['entry_time']} @ ${trade['entry_price']:.6f}")
            logger.info(f"  Exit: {trade['exit_time']} @ ${trade['exit_price']:.6f} ({trade['pnl_pct']:.2f}%)")
            logger.info(f"  Reason: {trade['reason']}")
            
            # Determine chain
            chain_id = determine_chain_id(trade['token'])
            logger.info(f"  Chain: {chain_id}")
            
            # Step 1: Discover pair
            start_time = time.time()
            pair_info = discover_pair(trade['token'], chain_id)
            api_call_count += 1
            
            if not pair_info:
                logger.warning(f"  ⚠️  Could not discover pair for {trade['token'][:8]}...")
                failed_extractions += 1
                continue
            
            pair_address = pair_info['pairAddress']
            logger.info(f"  Pair: {pair_address[:8]}... on {pair_info['dexId']}")
            
            # Step 2: Calculate time window
            start_window = trade['entry_time'] - timedelta(hours=WINDOW_BEFORE_ENTRY_HOURS)
            end_window = trade['exit_time'] + timedelta(hours=WINDOW_AFTER_EXIT_HOURS)
            
            # Step 3: Fetch candles for all intervals
            logger.info(f"  Fetching candles: {start_window} to {end_window}")
            
            candles_1m = fetch_dexscreener_candles(
                pair_address, chain_id, start_window, end_window, interval="1m",
                token_address=trade['token']
            )
            api_call_count += 1
            
            candles_5m = fetch_dexscreener_candles(
                pair_address, chain_id, start_window, end_window, interval="5m",
                token_address=trade['token']
            )
            api_call_count += 1
            
            candles_15m = fetch_dexscreener_candles(
                pair_address, chain_id, start_window, end_window, interval="15m",
                token_address=trade['token']
            )
            api_call_count += 1
            
            if not candles_1m or not candles_5m or not candles_15m:
                logger.warning(f"  ⚠️  Missing candle data (1m:{len(candles_1m) if candles_1m else 0}, "
                             f"5m:{len(candles_5m) if candles_5m else 0}, "
                             f"15m:{len(candles_15m) if candles_15m else 0})")
                failed_extractions += 1
                continue
            
            # Step 4: Extract features
            features = extract_advanced_features(
                candles_1m, candles_5m, candles_15m,
                trade['entry_price'], trade['entry_time'], pair_info
            )
            
            if not features:
                logger.warning(f"  ⚠️  Failed to extract features")
                failed_extractions += 1
                continue
            
            # Step 5: Calculate outcome label
            outcome_label = calculate_outcome_label(
                candles_1m, trade['entry_price'],
                trade['entry_time'], trade['exit_time']
            )
            
            if outcome_label is None:
                logger.warning(f"  ⚠️  Trade did not hit target or stop within time window - excluding")
                failed_extractions += 1
                continue
            
            # Create training sample
            training_sample = {
                'trade_index': i,
                'token': trade['token'],
                'pair_address': pair_address,
                'chain_id': chain_id,
                'entry_timestamp': trade['entry_time'].isoformat(),
                'exit_timestamp': trade['exit_time'].isoformat(),
                'entry_price': trade['entry_price'],
                'exit_price': trade['exit_price'],
                'pnl_pct': trade['pnl_pct'],
                'reason': trade['reason'],
                'outcome_label': outcome_label,  # 1 = hit target, 0 = hit stop
                'features': features,
                'pair_info': pair_info
            }
            
            training_data.append(training_sample)
            successful_extractions += 1
            
            logger.info(f"  ✅ Successfully extracted {len(features)} features")
            logger.info(f"  Outcome: {outcome_label} ({'Target Hit' if outcome_label == 1 else 'Stop Hit'})")
            
            # Small delay to be respectful to API
            time.sleep(0.2)
            
        except KeyboardInterrupt:
            logger.warning("\n⚠️  Interrupted by user. Saving progress...")
            break
        except Exception as e:
            logger.error(f"  ❌ Error processing trade: {e}", exc_info=True)
            failed_extractions += 1
            continue
    
    # Save training data
    logger.info("\n" + "=" * 60)
    logger.info("Saving training data...")
    
    output_data = {
        'metadata': {
            'total_trades': len(trades),
            'successful_extractions': successful_extractions,
            'failed_extractions': failed_extractions,
            'api_calls_made': api_call_count,
            'created_at': datetime.now().isoformat(),
            'source_file': str(TRADE_LOG_FILE),
            'window_before_hours': WINDOW_BEFORE_ENTRY_HOURS,
            'window_after_hours': WINDOW_AFTER_EXIT_HOURS,
            'ml_target': f"+{TARGET_PCT}% before -{STOP_PCT}% within {TIME_WINDOW_MINUTES} minutes"
        },
        'training_data': training_data,
        'api_call_log': api_log
    }
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"✅ Saved training data to {OUTPUT_FILE}")
    logger.info(f"   - {successful_extractions} successful extractions")
    logger.info(f"   - {failed_extractions} failed extractions")
    logger.info(f"   - {api_call_count} API calls made")
    
    # Save API call log separately
    API_CALL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(API_CALL_LOG, 'w') as f:
        json.dump(api_log, f, indent=2)
    
    logger.info(f"✅ Saved API call log to {API_CALL_LOG}")
    
    # Print summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("Summary Statistics:")
    logger.info("=" * 60)
    
    if training_data:
        target_hit = sum(1 for t in training_data if t['outcome_label'] == 1)
        stop_hit = sum(1 for t in training_data if t['outcome_label'] == 0)
        win_rate = (target_hit / len(training_data)) * 100 if training_data else 0
        
        logger.info(f"Total samples: {len(training_data)}")
        logger.info(f"Target Hit (+{TARGET_PCT}%): {target_hit} ({target_hit/len(training_data)*100:.1f}%)")
        logger.info(f"Stop Hit (-{STOP_PCT}%): {stop_hit} ({stop_hit/len(training_data)*100:.1f}%)")
        logger.info(f"Win Rate: {win_rate:.1f}%")
        
        # Feature statistics
        if training_data:
            sample_features = training_data[0]['features']
            logger.info(f"\nFeatures extracted ({len(sample_features)}):")
            for feature_name in sorted(sample_features.keys()):
                logger.info(f"  - {feature_name}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Training data preparation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        prepare_training_data()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)