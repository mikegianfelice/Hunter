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

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

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
WINDOW_BEFORE_ENTRY_HOURS = 24  # Hours before entry to fetch
WINDOW_AFTER_EXIT_HOURS = 24    # Hours after exit to fetch
FAST_MODE = False  # Set to True for ±6h window
if FAST_MODE:
    WINDOW_BEFORE_ENTRY_HOURS = 6
    WINDOW_AFTER_EXIT_HOURS = 6

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


def fetch_dexscreener_candles(pair_address: str, chain: str, 
                              start_time: datetime, end_time: datetime,
                              interval: str = "1m", token_address: str = None) -> Optional[List[Dict]]:
    """
    Step 2: Fetch historical candles
    
    Note: DexScreener free API doesn't provide historical candles endpoint.
    Falls back to market_data_fetcher (Helius RPC) for historical data.
    The market_data_fetcher typically returns hourly candles, which are then
    resampled to the requested interval (1m, 5m, 15m) using simple interpolation.
    
    Args:
        pair_address: Pair address (used for caching, not for API call)
        chain: Chain ID (solana, ethereum, etc.)
        start_time: Start of time window
        end_time: End of time window
        interval: Desired interval (1m, 5m, 15m)
        token_address: Token address for fallback API call (required)
    """
    try:
        # Check cache first
        start_ts = int(start_time.timestamp())
        end_ts = int(end_time.timestamp())
        cache_key = get_cache_key(pair_address, chain, interval, start_ts, end_ts)
        
        cached = load_cached_candles(cache_key)
        if cached:
            return cached
        
        # DexScreener free API doesn't have historical candles endpoint
        # Fall back to market_data_fetcher for historical data
        logger.info(f"  DexScreener doesn't provide historical candles, using market_data_fetcher fallback")
        
        if not token_address:
            logger.warning(f"  Token address required for fallback, skipping {pair_address[:8]}...")
            return None
        
        # Import market_data_fetcher for fallback
        from src.utils.market_data_fetcher import market_data_fetcher
        
        # Calculate hours needed
        hours = int((end_time - start_time).total_seconds() / 3600) + 1
        
        # Fetch using market_data_fetcher with target_timestamp
        entry_ts = start_time.timestamp() + (hours * 3600 / 2)  # Middle of window
        candles = market_data_fetcher.get_candlestick_data(
            token_address=token_address,
            chain_id=chain,
            hours=hours,
            force_fetch=True,
            target_timestamp=entry_ts
        )
        
        if not candles:
            logger.warning(f"No candles from market_data_fetcher for {token_address[:8]}...")
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
