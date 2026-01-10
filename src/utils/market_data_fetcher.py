#!/usr/bin/env python3
"""
Market Data Fetcher - Fetch real market data from multiple sources
"""

import os
import json
import time
import logging
import requests
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime, timedelta
import statistics
from pathlib import Path

from solana.rpc.api import Client
from solders.pubkey import Pubkey

# Load environment variables from system/.env (same pattern as src/config/secrets.py)
from dotenv import load_dotenv
PROJECT_ROOT = Path(__file__).resolve().parents[2]
system_env_path = PROJECT_ROOT / "system" / ".env"
if system_env_path.exists():
    try:
        load_dotenv(system_env_path)
    except PermissionError:
        pass  # Silently skip if can't read
load_dotenv()  # Also load from root .env as fallback

from src.config.secrets import GRAPH_API_KEY, UNISWAP_V3_DEPLOYMENT_ID
from src.utils.coingecko_helpers import ensure_vs_currency, DEFAULT_FIAT
from src.utils.api_tracker import get_tracker, track_coingecko_call, track_coincap_call, track_helius_call

# Configure logging
logger = logging.getLogger(__name__)


class MarketDataFetcher:
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes cache
        self.api_timeout = 15
        self.max_retries = 1
        self.global_snapshot_ttl = 300  # 5 minutes
        self._global_market_snapshot: Optional[Dict[str, Any]] = None
        self._global_snapshot_failure: Optional[float] = None
        self.market_chart_cache: Dict[str, Dict[str, Any]] = {}
        # Load cache duration from config (default 1 hour = 3600 seconds)
        from src.config.config_loader import get_config
        coingecko_cache_hours = get_config('api_rate_limiting.coingecko_cache_hours', 1)
        self.market_chart_cache_duration = coingecko_cache_hours * 3600  # Convert hours to seconds
        self._market_chart_bucket_seconds = 60  # align range queries to minute buckets
        # Load CoinGecko API key from environment
        self.coingecko_api_key = (os.getenv("COINGECKO_API_KEY") or "").strip()
        # Load CoinCap API key from environment
        self.coincap_api_key = (os.getenv("COINCAP_API_KEY") or "").strip()
        self._coingecko_base_url = "https://api.coingecko.com/api/v3/"
        # CoinCap Pro API v3 (requires API key - all calls need bearer token)
        self._coincap_base_url = "https://rest.coincap.io/v3/"
        self._binance_base_url = "https://api.binance.com/api/"
        self._coincap_available = True
        # Prefer non-CoinGecko sources to stay within free-tier limits
        self.prefer_coincap_for_prices = True
        # When False, CoinGecko market_chart/range is disabled and only CoinCap is used
        self.enable_coingecko_market_chart = True  # Enabled for Solana token lookups
        # When False, global metrics are fetched from CoinCap first; CoinGecko is fallback-only
        self.enable_coingecko_global = False
        
        # Helius API configuration
        self.helius_api_key = (os.getenv("HELIUS_API_KEY") or "").strip()
        self.helius_base_url = "https://api.helius.xyz/v0"
        
        # Separate caches for different APIs
        self.candlestick_cache_helius: Dict[str, Dict[str, any]] = {}
        self.candlestick_cache_coingecko: Dict[str, Dict[str, any]] = {}
        self.candlestick_cache_file = Path("data/candlestick_cache.json")
        
        # Cache durations - load from config with sensible defaults
        helius_cache_hours = get_config('helius_candlestick_settings.cache_duration_hours', 1)
        self.helius_cache_duration = helius_cache_hours * 3600  # Convert hours to seconds (default: 1 hour)
        self.coingecko_cache_duration = 3600  # 1 hour (conservative for CoinGecko)
        
        # Load routing thresholds for 15m candle policy
        from src.config.config_loader import get_config_int, get_config_float
        self.dex_max_swaps_guard = get_config_int('helius_15m_candle_policy.dex_api_max_swaps_guard', 3000)
        self.dex_max_response_mb = get_config_float('helius_15m_candle_policy.dex_api_max_response_size_mb', 15.0)
        self.dex_processing_timeout = get_config_float('helius_15m_candle_policy.dex_api_processing_timeout_seconds', 8.0)
        
        # API call tracking
        self.api_tracker = get_tracker()
        
        # Backfill rate limiting tracker (tracks tokens backfilled per cycle)
        self._backfill_tracker: Dict[str, float] = {}
        
        # Load persistent cache
        self._load_candlestick_cache()
        
    def get_btc_price(self) -> Optional[float]:
        """Get current BTC price in USD"""
        try:
            # Check cache first
            if 'btc_price' in self.cache:
                cached_data = self.cache['btc_price']
                if time.time() - cached_data['timestamp'] < self.cache_duration:
                    return cached_data['price']
            
            # Prefer CoinCap/Binance first to reduce CoinGecko usage
            if self.prefer_coincap_for_prices:
                price = self._get_price_from_coincap("bitcoin")
                if price is not None:
                    self.cache['btc_price'] = {'price': price, 'timestamp': time.time()}
                    return price

                price = self._get_price_from_binance("BTCUSDT")
                if price is not None:
                    self.cache['btc_price'] = {'price': price, 'timestamp': time.time()}
                    return price
            
            # Fallback to CoinGecko API
            url = f"{self._coingecko_base_url}simple/price?ids=bitcoin&vs_currencies=usd"
            data = self._fetch_json(url)
            if data and 'bitcoin' in data:
                price = float(data['bitcoin']['usd'])
                self.cache['btc_price'] = {'price': price, 'timestamp': time.time()}
                logger.info(f"✅ BTC price: ${price}")
                return price
            
            # Fallback to CoinGecko alternative endpoint
            url = f"{self._coingecko_base_url}coins/bitcoin?vs_currency={DEFAULT_FIAT}"
            data = self._fetch_json(url)
            if data and 'market_data' in data:
                price = float(data['market_data']['current_price']['usd'])
                self.cache['btc_price'] = {'price': price, 'timestamp': time.time()}
                logger.info(f"✅ BTC price: ${price}")
                return price
            
            # Final fallback to CoinCap/Binance if not already tried
            if not self.prefer_coincap_for_prices:
                price = self._get_price_from_coincap("bitcoin")
                if price is not None:
                    self.cache['btc_price'] = {'price': price, 'timestamp': time.time()}
                    return price
                price = self._get_price_from_binance("BTCUSDT")
                if price is not None:
                    self.cache['btc_price'] = {'price': price, 'timestamp': time.time()}
                    return price
                
        except Exception as e:
            logger.error(f"❌ Failed to fetch BTC price: {e}")
        
        return None
    
    def get_eth_price(self) -> Optional[float]:
        """Get current ETH price in USD"""
        try:
            # Check cache first
            if 'eth_price' in self.cache:
                cached_data = self.cache['eth_price']
                if time.time() - cached_data['timestamp'] < self.cache_duration:
                    return cached_data['price']
            
            # Prefer CoinCap/Binance first to reduce CoinGecko usage
            if self.prefer_coincap_for_prices:
                price = self._get_price_from_coincap("ethereum")
                if price is not None:
                    self.cache['eth_price'] = {'price': price, 'timestamp': time.time()}
                    return price

                price = self._get_price_from_binance("ETHUSDT")
                if price is not None:
                    self.cache['eth_price'] = {'price': price, 'timestamp': time.time()}
                    return price
            
            # Fallback to CoinGecko API
            url = f"{self._coingecko_base_url}simple/price?ids=ethereum&vs_currencies=usd"
            data = self._fetch_json(url)
            if data and 'ethereum' in data:
                price = float(data['ethereum']['usd'])
                self.cache['eth_price'] = {'price': price, 'timestamp': time.time()}
                logger.info(f"✅ ETH price: ${price}")
                return price
            
            # Fallback to CoinGecko alternative endpoint
            url = f"{self._coingecko_base_url}coins/ethereum?vs_currency={DEFAULT_FIAT}"
            data = self._fetch_json(url)
            if data and 'market_data' in data:
                price = float(data['market_data']['current_price']['usd'])
                self.cache['eth_price'] = {'price': price, 'timestamp': time.time()}
                logger.info(f"✅ ETH price: ${price}")
                return price
            
            # Final fallback to CoinCap/Binance if not already tried
            if not self.prefer_coincap_for_prices:
                price = self._get_price_from_coincap("ethereum")
                if price is not None:
                    self.cache['eth_price'] = {'price': price, 'timestamp': time.time()}
                    return price
                price = self._get_price_from_binance("ETHUSDT")
                if price is not None:
                    self.cache['eth_price'] = {'price': price, 'timestamp': time.time()}
                    return price
                
        except Exception as e:
            logger.error(f"❌ Failed to fetch ETH price: {e}")
        
        return None
    
    def get_btc_trend(self, hours: int = None) -> float:
        """Get BTC trend over last N hours (0-1 scale)
        
        Uses extended timeframe (7 days default) for regime detection accuracy.
        """
        # Default to 7 days (168 hours) for regime detection
        if hours is None:
            from src.config.config_loader import get_config
            hours = get_config('market_analysis_timeframes.btc_trend_hours', 168)
        
        try:
            data, from_timestamp, now = self._get_market_chart_range("bitcoin", hours)

            price_points: List[float] = []
            if data and 'prices' in data:
                price_points = [float(point[1]) for point in data['prices'] if len(point) >= 2]
            else:
                interval = self._select_coincap_interval(hours)
                history = self._get_history_from_coincap("bitcoin", from_timestamp, now, interval)
                if history:
                    price_points = [
                        float(point["priceUsd"])
                        for point in history
                        if point.get("priceUsd") is not None
                    ]

            if len(price_points) >= 2:
                start_price = price_points[0]
                end_price = price_points[-1]

                if start_price > 0:
                    change_pct = (end_price - start_price) / start_price
                    trend = max(0, min(1, 0.5 + change_pct))
                    return trend
                        
        except Exception as e:
            logger.error(f"❌ Failed to fetch BTC trend: {e}")
        
        return 0.5  # Neutral trend if unable to fetch
    
    def get_eth_trend(self, hours: int = None) -> float:
        """Get ETH trend over last N hours (0-1 scale)
        
        Uses extended timeframe (7 days default) for regime detection accuracy.
        """
        # Default to 7 days (168 hours) for regime detection
        if hours is None:
            from src.config.config_loader import get_config
            hours = get_config('market_analysis_timeframes.eth_trend_hours', 168)
        
        try:
            data, from_timestamp, now = self._get_market_chart_range("ethereum", hours)

            price_points: List[float] = []
            if data and 'prices' in data:
                price_points = [float(point[1]) for point in data['prices'] if len(point) >= 2]
            else:
                interval = self._select_coincap_interval(hours)
                history = self._get_history_from_coincap("ethereum", from_timestamp, now, interval)
                if history:
                    price_points = [
                        float(point["priceUsd"])
                        for point in history
                        if point.get("priceUsd") is not None
                    ]

            if len(price_points) >= 2:
                start_price = price_points[0]
                end_price = price_points[-1]

                if start_price > 0:
                    change_pct = (end_price - start_price) / start_price
                    trend = max(0, min(1, 0.5 + change_pct))
                    return trend
                        
        except Exception as e:
            logger.error(f"❌ Failed to fetch ETH trend: {e}")
        
        return 0.5  # Neutral trend if unable to fetch
    
    def get_market_volatility(self, hours: int = None) -> float:
        """Get market volatility index (0-1 scale)
        
        Uses extended timeframe (30 days default) for accurate volatility calculation.
        Needs 20-30 periods for reliable volatility measurement.
        """
        # Default to 30 days (720 hours) for accurate volatility
        if hours is None:
            from src.config.config_loader import get_config
            hours = get_config('market_analysis_timeframes.volatility_calculation_hours', 720)
            min_periods = get_config('market_analysis_timeframes.volatility_min_periods', 20)
        else:
            min_periods = 20  # Default minimum
        
        try:
            data, from_timestamp, now = self._get_market_chart_range("bitcoin", hours)

            prices: List[float] = []
            if data and 'prices' in data:
                prices = [float(p[1]) for p in data['prices']]
            else:
                interval = self._select_coincap_interval(hours)
                history = self._get_history_from_coincap("bitcoin", from_timestamp, now, interval)
                if history:
                    prices = [
                        float(point["priceUsd"])
                        for point in history
                        if point.get("priceUsd") is not None
                    ]

            if len(prices) >= min_periods:
                std_dev = statistics.stdev(prices)
                mean_price = statistics.mean(prices)

                if mean_price > 0:
                    volatility = std_dev / mean_price
                    volatility_normalized = min(1, volatility * 2)
                    
                    # Log confidence if we have less than ideal data
                    if len(prices) < 30:
                        confidence = len(prices) / 30.0
                        logger.info(f"✅ Market volatility: {volatility_normalized:.3f} (confidence: {confidence:.2f}, {len(prices)} periods)")
                    else:
                        logger.info(f"✅ Market volatility: {volatility_normalized:.3f} ({len(prices)} periods)")
                    
                    return volatility_normalized
            elif len(prices) >= 2:
                # Fallback: calculate with limited data but warn
                std_dev = statistics.stdev(prices)
                mean_price = statistics.mean(prices)
                if mean_price > 0:
                    volatility = std_dev / mean_price
                    volatility_normalized = min(1, volatility * 2)
                    confidence = len(prices) / min_periods
                    logger.warning(f"⚠️ Market volatility calculated with limited data: {volatility_normalized:.3f} (confidence: {confidence:.2f}, {len(prices)} < {min_periods} periods)")
                    return volatility_normalized
                        
        except Exception as e:
            logger.error(f"❌ Failed to fetch market volatility: {e}")
        
        return 0.5  # Medium volatility if unable to fetch
    
    def get_fear_greed_index(self) -> float:
        """Get Fear & Greed Index (0-1 scale, 0=extreme fear, 1=extreme greed)"""
        try:
            # Alternative Crypto Fear & Greed Index API
            url = "https://api.alternative.me/fng/"
            data = self._fetch_json(url)
            
            if data and 'data' in data and len(data['data']) > 0:
                value = int(data['data'][0]['value'])
                # Normalize 0-100 to 0-1
                normalized = value / 100.0
                logger.info(f"✅ Fear & Greed Index: {value}/100")
                return normalized
                
        except Exception as e:
            logger.error(f"❌ Failed to fetch Fear & Greed Index: {e}")
        
        return 0.5  # Neutral if unable to fetch
    
    def get_market_correlation(self, hours: int = None) -> float:
        """Get market correlation between BTC and ETH (0-1 scale)
        
        Uses extended timeframe (14 days default) for statistical significance.
        Needs 60+ data points for reliable correlation calculation.
        """
        # Default to 14 days (336 hours) for statistical significance
        if hours is None:
            from src.config.config_loader import get_config
            hours = get_config('market_analysis_timeframes.correlation_analysis_hours', 336)
            min_data_points = get_config('market_analysis_timeframes.correlation_min_data_points', 60)
        else:
            min_data_points = 60  # Default minimum
        
        try:
            btc_data, btc_from_timestamp, btc_now = self._get_market_chart_range("bitcoin", hours)
            eth_data, eth_from_timestamp, eth_now = self._get_market_chart_range("ethereum", hours)

            btc_prices: Dict[int, float] = {}
            eth_prices: Dict[int, float] = {}

            if btc_data and 'prices' in btc_data:
                btc_prices = {int(p[0] / 1000): float(p[1]) for p in btc_data['prices']}
            else:
                interval = self._select_coincap_interval(hours)
                history = self._get_history_from_coincap("bitcoin", btc_from_timestamp, btc_now, interval)
                if history:
                    btc_prices = {
                        int(point["time"] / 1000): float(point["priceUsd"])
                        for point in history
                        if point.get("time") is not None and point.get("priceUsd") is not None
                    }
            
            if eth_data and 'prices' in eth_data:
                eth_prices = {int(p[0] / 1000): float(p[1]) for p in eth_data['prices']}
            else:
                interval = self._select_coincap_interval(hours)
                history = self._get_history_from_coincap("ethereum", eth_from_timestamp, eth_now, interval)
                if history:
                    eth_prices = {
                        int(point["time"] / 1000): float(point["priceUsd"])
                        for point in history
                        if point.get("time") is not None and point.get("priceUsd") is not None
                    }

            if not btc_prices or not eth_prices:
                # Try shorter window if we have very little data
                if hours > 168:
                    logger.info(f"Retrying correlation with shorter time window ({hours}h -> 168h)")
                    return self.get_market_correlation(hours=168)
                elif hours > 24:
                    logger.info(f"Retrying correlation with shorter time window ({hours}h -> 24h)")
                    return self.get_market_correlation(hours=24)
                logger.warning("Insufficient data for correlation calculation")
                return 0.5
            
            # Find common timestamps (within 1 hour window)
            common_timestamps = []
            for btc_ts in btc_prices.keys():
                for eth_ts in eth_prices.keys():
                    if abs(btc_ts - eth_ts) < 3600:  # Within 1 hour
                        common_timestamps.append((btc_ts, eth_ts))
                        break
            
            # Require minimum data points for statistical significance
            if len(common_timestamps) < min_data_points:
                logger.warning(f"Insufficient data points for correlation ({len(common_timestamps)} < {min_data_points})")
                # Return neutral with reduced confidence
                return 0.5
            
            # Calculate returns for both assets
            btc_returns = []
            eth_returns = []
            
            sorted_timestamps = sorted(common_timestamps, key=lambda x: x[0])
            for i in range(1, len(sorted_timestamps)):
                prev_btc_ts, prev_eth_ts = sorted_timestamps[i-1]
                curr_btc_ts, curr_eth_ts = sorted_timestamps[i]
                
                prev_btc_price = btc_prices[prev_btc_ts]
                curr_btc_price = btc_prices[curr_btc_ts]
                prev_eth_price = eth_prices[prev_eth_ts]
                curr_eth_price = eth_prices[curr_eth_ts]
                
                if prev_btc_price > 0 and prev_eth_price > 0:
                    btc_return = (curr_btc_price - prev_btc_price) / prev_btc_price
                    eth_return = (curr_eth_price - prev_eth_price) / prev_eth_price
                    btc_returns.append(btc_return)
                    eth_returns.append(eth_return)
            
            if len(btc_returns) < min_data_points:
                logger.warning(f"Not enough return data for reliable correlation ({len(btc_returns)} < {min_data_points})")
                return 0.5
            
            # Calculate Pearson correlation coefficient
            mean_btc = statistics.mean(btc_returns)
            mean_eth = statistics.mean(eth_returns)
            
            numerator = sum((btc_returns[i] - mean_btc) * (eth_returns[i] - mean_eth) 
                          for i in range(len(btc_returns)))
            
            btc_variance = sum((r - mean_btc) ** 2 for r in btc_returns)
            eth_variance = sum((r - mean_eth) ** 2 for r in eth_returns)
            
            denominator = (btc_variance * eth_variance) ** 0.5
            
            if denominator == 0:
                return 0.5
            
            correlation = numerator / denominator
            
            # Normalize correlation (-1 to 1) to (0 to 1) scale
            # Where 0 = -1 correlation, 0.5 = 0 correlation, 1 = +1 correlation
            normalized = (correlation + 1) / 2
            
            # Calculate confidence based on data points
            data_quality = min(1.0, len(btc_returns) / min_data_points)
            confidence_note = f" (confidence: {data_quality:.2f}, {len(btc_returns)} data points)" if data_quality < 1.0 else ""
            
            logger.info(f"✅ Market correlation (BTC/ETH): {correlation:.3f} (normalized: {normalized:.3f}){confidence_note}")
            return max(0.0, min(1.0, normalized))
                
        except Exception as e:
            logger.error(f"❌ Failed to calculate market correlation: {e}")
        
        return 0.5
    
    def get_volume_trends(self, hours: int = None) -> float:
        """Get volume trends (0-1 scale) based on historical comparison
        
        Uses rolling average comparison (7-14 days) instead of static baselines.
        """
        # Default to 14 days (336 hours) with 7-day rolling window
        if hours is None:
            from src.config.config_loader import get_config
            hours = get_config('market_analysis_timeframes.volume_trend_hours', 336)
            rolling_window = get_config('market_analysis_timeframes.volume_trend_rolling_window', 168)
        else:
            rolling_window = hours // 2  # Use half the period as rolling window
        
        try:
            # Get current total market volume with caching
            snapshot = self._get_global_market_snapshot()
            data_dict = snapshot.get('data', {}) if snapshot else {}
            current_volume: Optional[float] = None

            if data_dict:
                volume_data = data_dict.get('total_volume', {})
                if isinstance(volume_data, dict) and 'usd' in volume_data:
                    current_volume = float(volume_data['usd'])

            if current_volume is None:
                logger.warning("Failed to fetch current volume data from all providers")
                return 0.5

            
            # Get historical volume data using BTC as proxy (most liquid asset)
            # We'll use BTC's volume history to estimate overall market trend
            logger.debug(f"🔍 Fetching BTC market chart data for {hours} hours...")
            btc_data, from_timestamp, now = self._get_market_chart_range("bitcoin", hours)

            volumes: List[float] = []
            if btc_data and 'total_volumes' in btc_data:
                volumes = [float(v[1]) for v in btc_data['total_volumes']]
                logger.debug(f"✅ Got {len(volumes)} volume data points from market_chart_range")

            if not volumes:
                logger.debug(f"⚠️ No volumes from market_chart_range (CoinCap /history endpoint doesn't provide volume data)")
                logger.debug(f"🔍 Falling back to CoinGecko for volume data...")
                
                # CoinCap Pro API v3 /history endpoint doesn't include volume data
                # We need to use CoinGecko market_chart/range for volume data
                # Clear cache to force fresh CoinGecko fetch (since cache might have CoinCap data without volumes)
                cache_key_to_clear = f"market_chart:bitcoin:{hours}:{int(time.time() // (3600 if hours >= 24 else 60)) * (3600 if hours >= 24 else 60)}"
                if cache_key_to_clear in self.market_chart_cache:
                    logger.debug(f"🔍 Clearing cached CoinCap data to force CoinGecko fetch for volumes")
                    del self.market_chart_cache[cache_key_to_clear]
                
                if self.enable_coingecko_market_chart and self._can_make_coingecko_call():
                    logger.debug(f"   CoinGecko available - fetching volume data...")
                    coingecko_data, _, _ = self._get_market_chart_range("bitcoin", hours)
                    if coingecko_data and 'total_volumes' in coingecko_data:
                        volumes = [float(v[1]) for v in coingecko_data['total_volumes']]
                        logger.debug(f"✅ Got {len(volumes)} volume data points from CoinGecko")
                    else:
                        logger.warning(f"⚠️ CoinGecko market_chart/range also failed to provide volume data")
                else:
                    coingecko_enabled = self.enable_coingecko_market_chart
                    coingecko_can_call = self._can_make_coingecko_call() if hasattr(self, '_can_make_coingecko_call') else False
                    logger.warning(f"⚠️ CoinGecko unavailable for volume fallback (enabled: {coingecko_enabled}, can_call: {coingecko_can_call})")

            if len(volumes) < 2:
                logger.warning(f"⚠️ Insufficient volume data: {len(volumes)} data points (need at least 2)")
                # Fallback: Use current volume relative to a reasonable baseline
                # Typical crypto market volume ranges from $50B to $200B+
                # Normalize assuming $50B = 0.0, $200B+ = 1.0
                baseline_volume = 50000000000  # $50B
                max_volume = 200000000000  # $200B
                
                if current_volume < baseline_volume:
                    trend = 0.0 + (current_volume / baseline_volume) * 0.3
                elif current_volume > max_volume:
                    trend = 1.0
                else:
                    trend = 0.3 + ((current_volume - baseline_volume) / (max_volume - baseline_volume)) * 0.7
                
                logger.warning(f"⚠️ Volume trend (estimated with static baseline): {trend:.3f} (current: ${current_volume/1e9:.1f}B, insufficient historical data)")
                return max(0.0, min(1.0, trend))
            
            # Use rolling average comparison instead of simple split
            # Compare recent rolling average vs older rolling average
            if len(volumes) < rolling_window * 2:
                # Not enough data for rolling comparison, use simple split
                mid_point = len(volumes) // 2
                older_volumes = volumes[:mid_point] if mid_point > 0 else volumes[:len(volumes)//2]
                recent_volumes = volumes[mid_point:] if mid_point > 0 else volumes[len(volumes)//2:]
            else:
                # Use rolling averages: compare last N periods vs previous N periods
                recent_volumes = volumes[-rolling_window:]
                older_volumes = volumes[-(rolling_window * 2):-rolling_window] if len(volumes) >= rolling_window * 2 else volumes[:rolling_window]
            
            if len(older_volumes) == 0 or len(recent_volumes) == 0:
                logger.warning("Insufficient volume history for trend calculation")
                return 0.5
            
            older_avg = statistics.mean(older_volumes)
            recent_avg = statistics.mean(recent_volumes)
            
            if older_avg == 0:
                return 0.5
            
            # Calculate percentage change
            volume_change_pct = (recent_avg - older_avg) / older_avg
            
            # Normalize to 0-1 scale
            # Assuming -50% to +50% change range maps to 0.0 to 1.0
            # More than +50% = 1.0, less than -50% = 0.0
            trend = max(0.0, min(1.0, 0.5 + volume_change_pct))
            
            # Calculate confidence based on data quality
            data_points = len(volumes)
            ideal_points = hours  # Ideally one data point per hour
            confidence = min(1.0, data_points / ideal_points) if ideal_points > 0 else 0.5
            
            confidence_note = f" (confidence: {confidence:.2f})" if confidence < 0.8 else ""
            logger.info(f"✅ Volume trend (rolling avg): {trend:.3f} ({volume_change_pct*100:+.1f}% change, recent: ${recent_avg/1e9:.1f}B, older: ${older_avg/1e9:.1f}B){confidence_note}")
            return trend
                
        except Exception as e:
            logger.error(f"❌ Failed to fetch volume trends: {e}")
        
        return 0.5
    
    def get_market_cap_trend(self, hours: int = 24) -> float:
        """Get market cap trend (0-1 scale) based on historical comparison"""
        try:
            # Get current total market cap with caching
            snapshot = self._get_global_market_snapshot()
            data_dict = snapshot.get('data', {}) if snapshot else {}
            current_market_cap: Optional[float] = None

            if data_dict:
                market_cap_data = data_dict.get('total_market_cap', {})
                if isinstance(market_cap_data, dict) and 'usd' in market_cap_data:
                    current_market_cap = float(market_cap_data['usd'])

            if current_market_cap is None:
                logger.warning("Failed to fetch current market cap data from all providers")
                return 0.5
            
            # Get historical market cap data using BTC as proxy
            btc_data, from_timestamp, now = self._get_market_chart_range("bitcoin", hours * 2)

            market_caps: List[float] = []
            if btc_data and 'market_caps' in btc_data:
                market_caps = [float(mc[1]) for mc in btc_data['market_caps']]

            if not market_caps and btc_data and 'prices' in btc_data:
                market_caps = [float(price[1]) for price in btc_data['prices']]

            if not market_caps:
                interval = self._select_coincap_interval(hours * 2)
                history = self._get_history_from_coincap("bitcoin", from_timestamp, now, interval)
                if history:
                    market_caps = [
                        float(point.get("marketCapUsd") or point.get("priceUsd"))
                        for point in history
                        if point.get("marketCapUsd") is not None or point.get("priceUsd") is not None
                    ]

            if len(market_caps) < 2:
                # Fallback: Use current market cap relative to a reasonable baseline
                # Typical crypto market cap ranges from $1T to $3T+
                # Normalize assuming $1T = 0.0, $3T+ = 1.0
                baseline_cap = 1000000000000  # $1T
                max_cap = 3000000000000  # $3T
                
                if current_market_cap < baseline_cap:
                    trend = 0.0 + (current_market_cap / baseline_cap) * 0.3
                elif current_market_cap > max_cap:
                    trend = 1.0
                else:
                    trend = 0.3 + ((current_market_cap - baseline_cap) / (max_cap - baseline_cap)) * 0.7
                
                logger.info(f"✅ Market cap trend (estimated): {trend:.3f} (current: ${current_market_cap/1e12:.2f}T)")
                return max(0.0, min(1.0, trend))
            
            # Split into two periods: older half and recent half
            mid_point = len(market_caps) // 2
            older_caps = market_caps[:mid_point] if mid_point > 0 else market_caps[:len(market_caps)//2]
            recent_caps = market_caps[mid_point:] if mid_point > 0 else market_caps[len(market_caps)//2:]
            
            if len(older_caps) == 0 or len(recent_caps) == 0:
                logger.warning("Insufficient market cap history for trend calculation")
                return 0.5
            
            older_avg = statistics.mean(older_caps)
            recent_avg = statistics.mean(recent_caps)
            
            if older_avg == 0:
                return 0.5
            
            # Calculate percentage change
            cap_change_pct = (recent_avg - older_avg) / older_avg
            
            # Normalize to 0-1 scale
            # Assuming -50% to +50% change range maps to 0.0 to 1.0
            # More than +50% = 1.0, less than -50% = 0.0
            trend = max(0.0, min(1.0, 0.5 + cap_change_pct))
            
            logger.info(f"✅ Market cap trend: {trend:.3f} ({cap_change_pct*100:+.1f}% change, recent avg: ${recent_avg/1e12:.2f}T, older avg: ${older_avg/1e12:.2f}T)")
            return trend
                
        except Exception as e:
            logger.error(f"❌ Failed to fetch market cap trend: {e}")
        
        return 0.5
    
    def _get_market_chart_range(self, asset_id: str, hours: int) -> Tuple[Optional[Dict], int, int]:
        """
        Retrieve and cache market_chart/range data for the requested asset.
        Uses CoinCap FIRST for BTC/ETH (free, unlimited), CoinGecko as fallback only.
        Returns a tuple of (data, from_timestamp, to_timestamp).
        """
        if hours <= 0:
            hours = 1

        # Use hour-based buckets for longer timeframes (better cache efficiency)
        if hours >= 24:
            bucket_seconds = 3600  # 1 hour buckets for longer timeframes
        else:
            bucket_seconds = self._market_chart_bucket_seconds
        
        bucket_now = int(time.time() // bucket_seconds) * bucket_seconds
        from_timestamp = bucket_now - (hours * 3600)
        cache_key = f"market_chart:{asset_id}:{hours}:{bucket_now}"

        # Check cache first
        cached = self.market_chart_cache.get(cache_key)
        current_time = time.time()
        if cached and current_time - cached["timestamp"] < self.market_chart_cache_duration:
            return cached["data"], from_timestamp, bucket_now

        # Try CoinCap FIRST for BTC/ETH (free, unlimited, reliable)
        # NOTE: CoinCap Pro API v3 /history endpoint doesn't include volume data
        # So if we need volumes, we'll continue to CoinGecko below
        coin_cap_prices = None
        if asset_id in ["bitcoin", "ethereum"]:
            logger.debug(f"🔍 Attempting CoinCap fetch for {asset_id} (interval: {self._select_coincap_interval(hours)}, hours: {hours})")
            logger.debug(f"   CoinCap API key present: {bool(self.coincap_api_key)}")
            interval = self._select_coincap_interval(hours)
            history = self._get_history_from_coincap(asset_id, from_timestamp, bucket_now, interval)
            if history and len(history) > 0:
                # Convert CoinCap format to CoinGecko-like format for compatibility
                coin_cap_prices = [[int(point["time"]), float(point["priceUsd"])] 
                                   for point in history if point.get("priceUsd") is not None]
                # CoinCap Pro API v3 /history endpoint doesn't include volumeUsd
                # Check available fields for debugging
                sample_point = history[0] if history else {}
                logger.debug(f"🔍 CoinCap history sample fields: {list(sample_point.keys())}")
                
                # CoinCap doesn't provide volume data, so we'll need CoinGecko for volumes
                # But if CoinCap provided prices and CoinGecko is disabled/rate-limited, return CoinCap prices
                if coin_cap_prices and (not self.enable_coingecko_market_chart or not self._can_make_coingecko_call()):
                    data = {"prices": coin_cap_prices}
                    self.market_chart_cache[cache_key] = {"data": data, "timestamp": current_time}
                    logger.debug(f"✅ Using CoinCap data for {asset_id} market chart ({len(coin_cap_prices)} price points, 0 volume points - CoinGecko unavailable)")
                    return data, from_timestamp, bucket_now
            else:
                logger.warning(f"⚠️ CoinCap fetch failed or returned no data for {asset_id}")

        # Only use CoinGecko if CoinCap failed AND CoinGecko is enabled
        if not self.enable_coingecko_market_chart:
            logger.debug(f"⚠️ CoinGecko market chart disabled, skipping fallback for {asset_id}")
            return None, from_timestamp, bucket_now
        
        # Check rate limit before making CoinGecko call
        coingecko_count = self.api_tracker.get_count('coingecko')
        if not self._can_make_coingecko_call():
            logger.warning(f"⚠️ CoinGecko rate limit reached ({coingecko_count}/330), skipping {asset_id} market chart fetch")
            return None, from_timestamp, bucket_now
        
        logger.debug(f"🔍 Attempting CoinGecko fetch for {asset_id} (current calls: {coingecko_count}/330)")
        logger.debug(f"   CoinGecko API key present: {bool(self.coingecko_api_key)}")

        # Prune stale cache entries opportunistically
        stale_keys = [
            key for key, entry in self.market_chart_cache.items()
            if current_time - entry["timestamp"] >= self.market_chart_cache_duration
        ]
        for key in stale_keys:
            self.market_chart_cache.pop(key, None)

        url = (
            f"{self._coingecko_base_url}coins/{asset_id}/market_chart/range"
            f"?vs_currency=usd&from={from_timestamp}&to={bucket_now}"
        )
        logger.debug(f"🔍 CoinGecko URL: {url[:100]}...")
        coingecko_data = self._fetch_json(url)
        if coingecko_data is not None:
            if isinstance(coingecko_data, dict):
                volume_count = len(coingecko_data.get('total_volumes', []))
                price_count = len(coingecko_data.get('prices', []))
                logger.debug(f"✅ CoinGecko returned data with {price_count} price points and {volume_count} volume points")
                
                # If we got CoinCap prices earlier but no volumes, merge CoinCap prices with CoinGecko volumes
                if coin_cap_prices and volume_count > 0:
                    logger.debug(f"🔗 Merging CoinCap prices ({len(coin_cap_prices)} points) with CoinGecko volumes ({volume_count} points)")
                    coingecko_data["prices"] = coin_cap_prices  # Use CoinCap prices (more reliable)
                    # Keep CoinGecko volumes
                
                if volume_count == 0:
                    logger.warning(f"⚠️ CoinGecko data exists but has no volume points. Available keys: {list(coingecko_data.keys())}")
            else:
                logger.warning(f"⚠️ CoinGecko returned non-dict data: {type(coingecko_data)}")
            self.market_chart_cache[cache_key] = {"data": coingecko_data, "timestamp": current_time}
            return coingecko_data, from_timestamp, bucket_now
        else:
            logger.warning(f"⚠️ CoinGecko fetch returned None for {asset_id} (check rate limits or API key)")
            # If we have CoinCap prices but CoinGecko failed, return CoinCap data anyway
            if coin_cap_prices:
                data = {"prices": coin_cap_prices}
                logger.debug(f"⚠️ Returning CoinCap price data only (no volumes available)")
                return data, from_timestamp, bucket_now

        return None, from_timestamp, bucket_now
    
    def _fetch_json(self, url: str) -> Optional[Dict]:
        """Fetch JSON data with retry logic, rate limit handling, and proper headers"""
        headers = {
            "Accept": "application/json",
            "User-Agent": "HunterBot/1.0"
        }
        
        # Check if this is a CoinGecko API call
        is_coingecko_call = "coingecko.com" in url
        # Check if this is a CoinCap API call (v3 uses rest.coincap.io)
        is_coincap_call = "rest.coincap.io" in url or "api.coincap.io" in url
        
        # Add CoinGecko API key if available
        if self.coingecko_api_key and is_coingecko_call:
            headers["x-cg-demo-api-key"] = self.coingecko_api_key
            
            # Check rate limit before making CoinGecko call
            if not self._can_make_coingecko_call():
                logger.warning(f"CoinGecko rate limit reached ({self.api_tracker.get_count('coingecko')}/330), skipping {url[:60]}...")
                return None
        
        # Add CoinCap API key if available (Bearer token format)
        # CoinCap Pro API v3 REQUIRES API key for all calls
        if is_coincap_call:
            if self.coincap_api_key:
                headers["Authorization"] = f"Bearer {self.coincap_api_key}"
                logger.debug(f"🔑 CoinCap API key added to headers (length: {len(self.coincap_api_key)})")
            else:
                logger.warning("❌ CoinCap API key not provided - CoinCap Pro API v3 requires API key for all calls")
                return None
        
        backoff = 1.0
        
        for attempt in range(self.max_retries):
            try:
                url, _ = ensure_vs_currency(url)
                response = requests.get(url, timeout=self.api_timeout, headers=headers)
                
                if response.status_code == 200:
                    # Track successful API calls
                    if is_coingecko_call:
                        track_coingecko_call()
                    elif is_coincap_call:
                        track_coincap_call()
                    logger.debug(f"✅ API call successful: {url[:80]}...")
                    return response.json()
                elif response.status_code == 429:
                    # Rate limited – log once and move on instead of retrying
                    api_name = "CoinGecko" if is_coingecko_call else "CoinCap"
                    logger.warning(
                        f"❌ Rate limited (429) on {api_name} API: {url[:60]}... Skipping retries"
                    )
                    return None
                elif response.status_code == 403:
                    api_name = "CoinGecko" if is_coingecko_call else "CoinCap"
                    logger.warning(f"❌ Access forbidden (403) on {api_name} API: {url[:60]}...")
                    return None
                else:
                    # Log other error status codes
                    api_name = "CoinGecko" if is_coingecko_call else "CoinCap"
                    logger.warning(
                        f"❌ {api_name} API returned status {response.status_code} for {url[:60]}... Response: {response.text[:200]}"
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 60)
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}/{self.max_retries} for {url[:60]}...")
                if attempt < self.max_retries - 1:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error on attempt {attempt + 1}/{self.max_retries}: {e}")
                if is_coincap_call:
                    if self._coincap_available:
                        logger.warning("Disabling CoinCap data source for this session due to network failures.")
                    self._coincap_available = False
                if attempt < self.max_retries - 1:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)
            except Exception as e:
                logger.warning(f"Unexpected error on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)
        
        return None

    def _get_price_from_coincap(self, asset_id: str) -> Optional[float]:
        """Fetch current price from CoinCap."""
        if not self._coincap_available:
            return None
        try:
            url = f"{self._coincap_base_url}assets/{asset_id}"
            data = self._fetch_json(url)
            if data and "data" in data and isinstance(data["data"], dict):
                price = data["data"].get("priceUsd")
                if price is not None:
                    value = float(price)
                    logger.info(f"✅ {asset_id.upper()} price from CoinCap: ${value}")
                    return value
        except Exception as exc:
            logger.warning(f"CoinCap price fetch failed for {asset_id}: {exc}")
        return None

    def _get_price_from_binance(self, symbol: str) -> Optional[float]:
        """Fetch current price from Binance public ticker endpoint."""
        try:
            url = f"{self._binance_base_url}v3/ticker/price?symbol={symbol}"
            data = self._fetch_json(url)
            if data and "price" in data:
                value = float(data["price"])
                logger.info(f"✅ {symbol} price from Binance: ${value}")
                return value
        except Exception as exc:
            logger.warning(f"Binance price fetch failed for {symbol}: {exc}")
        return None

    def _get_history_from_coincap(self, asset_id: str, start_ts: int, end_ts: int, interval: str = "h1") -> Optional[List[Dict]]:
        """Fetch historical price data from CoinCap."""
        if not self._coincap_available:
            logger.warning(f"⚠️ CoinCap marked as unavailable")
            return None
        try:
            # CoinCap expects millisecond timestamps
            start_ms = start_ts * 1000
            end_ms = end_ts * 1000
            url = (
                f"{self._coincap_base_url}assets/{asset_id}/history"
                f"?interval={interval}&start={start_ms}&end={end_ms}"
            )
            logger.debug(f"🔍 CoinCap URL: {url[:100]}...")
            data = self._fetch_json(url)
            if data:
                if "data" in data and isinstance(data["data"], list):
                    if data["data"]:
                        logger.info(f"✅ Retrieved {len(data['data'])} {asset_id} datapoints from CoinCap")
                        return data["data"]
                    else:
                        logger.warning(f"⚠️ CoinCap returned empty data list for {asset_id}")
                else:
                    logger.warning(f"⚠️ CoinCap response missing 'data' field or wrong type: {type(data.get('data'))}")
            else:
                logger.warning(f"⚠️ CoinCap _fetch_json returned None for {asset_id}")
        except Exception as exc:
            logger.warning(f"❌ CoinCap history fetch failed for {asset_id}: {exc}", exc_info=True)
        return None

    def _get_global_from_coincap(self) -> Optional[Dict]:
        """Fetch global market metrics from CoinCap.
        
        Note: CoinCap Pro API v3 does not have a /global endpoint.
        This method returns None to trigger fallback to CoinGecko.
        """
        # CoinCap Pro API v3 doesn't have a /global endpoint
        # Return None to use CoinGecko fallback
        return None

    def _get_global_market_snapshot(self) -> Optional[Dict]:
        """Return cached CoinGecko global snapshot or fetch a fresh one."""
        now = time.time()

        # Respect recent failure window to avoid hammering the API
        if self._global_snapshot_failure and now - self._global_snapshot_failure < 60:
            cached = self._global_market_snapshot
            if cached and now - cached.get("timestamp", 0) < self.global_snapshot_ttl:
                return cached.get("data")
            return cached.get("data") if cached else None

        cached_snapshot = self._global_market_snapshot
        if cached_snapshot and now - cached_snapshot.get("timestamp", 0) < self.global_snapshot_ttl:
            return cached_snapshot.get("data")

        # Prefer CoinCap first to reduce CoinGecko usage if enabled
        if not self.enable_coingecko_global:
            coincap_data = self._get_global_from_coincap()
            if coincap_data:
                normalized = {
                    "data": {
                        "total_market_cap": {
                            "usd": float(coincap_data.get("marketCapUsd")) if coincap_data.get("marketCapUsd") else None
                        },
                        "total_volume": {
                            "usd": float(coincap_data.get("volumeUsd24Hr")) if coincap_data.get("volumeUsd24Hr") else None
                        }
                    }
                }
                self._global_market_snapshot = {"timestamp": now, "data": normalized}
                self._global_snapshot_failure = None
                return normalized

        # CoinGecko global as primary or fallback
        url = f"{self._coingecko_base_url}global?vs_currency={DEFAULT_FIAT}"
        data = self._fetch_json(url)
        if data and isinstance(data, Dict):
            self._global_market_snapshot = {"timestamp": now, "data": data}
            self._global_snapshot_failure = None
            return data

        # Fallback to CoinCap; normalize structure to match CoinGecko response
        coincap_data = self._get_global_from_coincap()
        if coincap_data:
            normalized = {
                "data": {
                    "total_market_cap": {
                        "usd": float(coincap_data.get("marketCapUsd")) if coincap_data.get("marketCapUsd") else None
                    },
                    "total_volume": {
                        "usd": float(coincap_data.get("volumeUsd24Hr")) if coincap_data.get("volumeUsd24Hr") else None
                    }
                }
            }
            self._global_market_snapshot = {"timestamp": now, "data": normalized}
            self._global_snapshot_failure = None
            return normalized

        # Record failure to slow down repeated attempts
        self._global_snapshot_failure = now
        self._global_market_snapshot = {"timestamp": now, "data": None}
        return None

    @staticmethod
    def _select_coincap_interval(hours: int) -> str:
        """Pick an appropriate CoinCap interval based on requested hours."""
        if hours <= 6:
            return "m15"
        if hours <= 24:
            return "h1"
        if hours <= 72:
            return "h6"
        if hours <= 168:
            return "h12"
        return "d1"

    def _load_candlestick_cache(self):
        """Load persistent candlestick cache from disk"""
        if self.candlestick_cache_file.exists():
            try:
                data = json.loads(self.candlestick_cache_file.read_text())
                self.candlestick_cache_helius = data.get('helius', {})
                self.candlestick_cache_coingecko = data.get('coingecko', {})
                logger.info(f"Loaded candlestick cache: {len(self.candlestick_cache_helius)} Helius, {len(self.candlestick_cache_coingecko)} CoinGecko")
            except Exception as e:
                logger.error(f"Error loading candlestick cache: {e}")
                self.candlestick_cache_helius = {}
                self.candlestick_cache_coingecko = {}

    def _save_candlestick_cache(self):
        """Save candlestick cache to disk"""
        try:
            data = {
                'helius': self.candlestick_cache_helius,
                'coingecko': self.candlestick_cache_coingecko,
                'last_updated': time.time()
            }
            self.candlestick_cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.candlestick_cache_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Error saving candlestick cache: {e}")

    def get_candlestick_data(self, token_address: str, chain_id: str = "ethereum", 
                            hours: int = 24, force_fetch: bool = False,
                            target_timestamp: Optional[float] = None) -> Optional[List[Dict]]:
        """
        Get candlestick data using optimal API for each chain:
        - Solana: CoinGecko API (via token ID lookup)
        - Ethereum/Base: The Graph (free tier) or CoinGecko (sparingly)
        
        Args:
            target_timestamp: Unix timestamp to query historical data for (None = current time)
        """
        cache_key = f"{chain_id}:{token_address.lower()}:{hours}"
        if target_timestamp:
            cache_key += f":{int(target_timestamp)}"
        current_time = time.time()
        
        # SOLANA: Try indexed swaps first (fast, no API calls), then targeted backfill, then DEX API, then RPC
        if chain_id.lower() == "solana":
            # First, try indexed swap events from SQLite (fastest, no API calls)
            candles = self._get_solana_candles_from_indexed_swaps(
                token_address, hours, target_timestamp
            )
            
            # Check if we have sufficient data for accurate indicators
            from src.config.config_loader import get_config_int
            min_candles = get_config_int("swap_indexer.min_candles_for_accuracy", 35)
            
            if candles and len(candles) >= min_candles:
                logger.debug(f"✅ Got {len(candles)} candles from indexed swaps for {token_address[:8]}... (sufficient for accuracy)")
                return candles
            
            # If we have some data but not enough, try targeted backfill
            if candles and len(candles) >= 10:
                logger.debug(f"⚠️  Got {len(candles)} candles (need {min_candles}+), attempting targeted backfill for {token_address[:8]}...")
                candles = self._try_targeted_backfill(token_address, hours, candles, target_timestamp)
                if candles and len(candles) >= min_candles:
                    logger.debug(f"✅ After backfill: {len(candles)} candles for {token_address[:8]}...")
                    return candles
                # If backfill didn't help enough, continue with what we have
                if candles and len(candles) >= 10:
                    logger.debug(f"Using {len(candles)} candles (acceptable but not optimal)")
                    return candles
            
            # If no indexed data yet, add token to indexer for future cycles (defensive)
            if not candles or len(candles) < 10:
                try:
                    from src.config.config_loader import get_config
                    if get_config("swap_indexer.enabled", True):
                        from src.indexing.swap_indexer import get_indexer
                        indexer = get_indexer()
                        if indexer.running:
                            indexer.add_token(token_address)
                            logger.debug(f"Added {token_address[:8]}... to swap indexer (no indexed data yet, will build in background)")
                except Exception as e:
                    logger.debug(f"Could not add token to indexer: {e}")
            
            if self.helius_api_key:
                # Check rate limit before making expensive API calls
                from src.config.config_loader import get_config_int
                helius_calls = self.api_tracker.get_count('helius')
                rate_limit_threshold = get_config_int('helius_candlestick_settings.rate_limit_threshold', 25000)
                
                if helius_calls >= rate_limit_threshold:
                    logger.warning(
                        f"Helius API usage high ({helius_calls}/30000), using cached/memory data for {token_address[:8]}..."
                    )
                    # Try cache first (even if expired, might have usable data)
                    if cache_key in self.candlestick_cache_helius:
                        cached = self.candlestick_cache_helius[cache_key]
                        if cached.get('data'):
                            logger.debug(f"Using cached data despite rate limit for {token_address[:8]}...")
                            return cached['data']
                    # Fallback to memory
                    return self._get_solana_candles_from_memory(token_address, hours)
                
                # Try DEX API first (more efficient - 1 call vs potentially 100+ RPC calls)
                candles = self._get_solana_candles_from_helius(
                    token_address, hours, cache_key, force_fetch, target_timestamp
                )
                if candles:
                    return candles
                
                # Check rate limit again before expensive RPC fallback
                helius_calls = self.api_tracker.get_count('helius')
                if helius_calls >= rate_limit_threshold:
                    logger.warning(
                        f"Helius API usage high ({helius_calls}/30000), skipping RPC fallback for {token_address[:8]}..."
                    )
                    return self._get_solana_candles_from_memory(token_address, hours)
                
                # Fallback to RPC if DEX API fails or returns insufficient data
                logger.debug(f"DEX API failed or insufficient data, trying RPC fallback for {token_address[:8]}...")
                return self._get_solana_candles_from_rpc(
                    token_address, hours, cache_key, force_fetch, target_timestamp
                )
            else:
                logger.warning(f"No Helius API key configured for {token_address[:8]}..., using price_memory fallback")
                return self._get_solana_candles_from_memory(token_address, hours)
        
        # ETHEREUM/BASE: Use The Graph (free) or CoinGecko (sparingly)
        elif chain_id.lower() in ["ethereum", "base"]:
            # Try The Graph first (free, no CoinGecko usage)
            candles = self._get_ethereum_candles(token_address, chain_id.lower(), hours)
            if candles:
                return candles
            
            # Fallback to CoinGecko only if The Graph fails AND we have quota
            if self._can_make_coingecko_call():
                return self._get_candles_from_coingecko(token_address, chain_id, hours, cache_key)
        
        return None
    
    def _get_ethereum_candles(self, token_address: str, chain_id: str, hours: int) -> Optional[List[Dict]]:
        """Get real candlestick data from Uniswap subgraph"""
        try:
            # Use The Graph (Uniswap subgraph) - FREE tier works
            if chain_id == "ethereum":
                api_key = (GRAPH_API_KEY or "").strip()
                deployment_id = (UNISWAP_V3_DEPLOYMENT_ID or "").strip()
                if not api_key or not deployment_id:
                    logger.warning("Missing GRAPH_API_KEY or UNISWAP_V3_DEPLOYMENT_ID; cannot fetch Ethereum candles.")
                    return None
                subgraph_url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/{deployment_id}"
            elif chain_id == "base":
                subgraph_url = "https://api.studio.thegraph.com/query/48211/base-uniswap-v3/version/latest"
            else:
                return None
            
            # Get pair address for the token
            query = """
            {
              token(id: "%s") {
                id
                symbol
                derivedETH
                poolCount
                pools(first: 10, orderBy: reserveUSD, orderDirection: desc) {
                  id
                  token0 { id symbol }
                  token1 { id symbol }
                  reserveUSD
                  token0Price
                  token1Price
                  swaps(first: 100, orderBy: timestamp, orderDirection: desc) {
                    timestamp
                    amount0In
                    amount0Out
                    amount1In
                    amount1Out
                    amountUSD
                  }
                }
              }
            }
            """ % token_address.lower()
            
            data = self._fetch_graphql(subgraph_url, query)
            if not data or 'data' not in data or not data['data'].get('token'):
                logger.warning(f"No Uniswap data for {token_address[:8]}...")
                return None
            
            token_data = data['data']['token']
            if not token_data.get('pools'):
                logger.warning(f"No pools found for {token_address[:8]}...")
                return None
            
            # Get most liquid pool
            main_pool = token_data['pools'][0]
            swaps = main_pool.get('swaps', [])
            
            if len(swaps) < 2:
                logger.warning(f"Insufficient swap data for {token_address[:8]}...")
                return None
            
            # Process swaps into hourly candles
            candles = self._process_swaps_to_candles(swaps, hours)
            
            if candles:
                logger.info(f"✅ Retrieved {len(candles)} real candles from Uniswap for {token_address[:8]}...")
            
            return candles
            
        except Exception as e:
            logger.error(f"Error fetching Ethereum candles: {e}")
            return None
    
    def _get_solana_candles(self, token_address: str, hours: int) -> Optional[List[Dict]]:
        """Get real candlestick data from Solana DEX (legacy method - now uses Helius)"""
        # Redirect to Helius method (None for target_timestamp = current time)
        return self._get_solana_candles_from_helius(token_address, hours, f"solana:{token_address.lower()}:{hours}", False, None)
    
    def _get_solana_candles_from_helius(self, token_address: str, hours: int, 
                                       cache_key: str, force_fetch: bool = False,
                                       target_timestamp: Optional[float] = None) -> Optional[List[Dict]]:
        """
        Build candlesticks from Helius DEX swap transactions
        Uses Helius API (30k/day limit - use freely!)
        
        Args:
            target_timestamp: Unix timestamp to query historical data for (None = current time)
        """
        # Use target_timestamp if provided, otherwise use current time
        query_time = target_timestamp if target_timestamp else time.time()
        current_time = time.time()
        
        # Check cache first
        if not force_fetch and cache_key in self.candlestick_cache_helius:
            cached = self.candlestick_cache_helius[cache_key]
            cache_age = current_time - cached['timestamp']
            
            if cache_age < self.helius_cache_duration:
                logger.debug(f"✅ Using cached Helius candlestick data for {token_address[:8]}... (age: {cache_age/60:.1f}m)")
                return cached['data']
        
        if not self.helius_api_key:
            logger.debug("Helius API key not configured, using price_memory fallback")
            return self._get_solana_candles_from_memory(token_address, hours)
        
        # Check rate limit before making API call
        if not self._can_make_helius_call():
            helius_calls = self.api_tracker.get_count('helius')
            logger.warning(f"Helius rate limit reached ({helius_calls}/30000), using price_memory fallback for {token_address[:8]}...")
            return self._get_solana_candles_from_memory(token_address, hours)
        
        try:
            import time as time_module
            fetch_start_time = time_module.perf_counter()
            method_chosen = "dex_api"
            
            # Track API call BEFORE making request
            track_helius_call()
            
            # Use Helius DEX API to get swap transactions
            url = f"{self.helius_base_url}/addresses/{token_address}/transactions"
            params = {
                "api-key": self.helius_api_key,
                "mint": token_address,
                "type": "SWAP",
            }
            
            # Make request and capture response size
            api_call_start = time_module.perf_counter()
            response = requests.get(url, params=params, timeout=15)
            api_call_time = time_module.perf_counter() - api_call_start
            
            # Capture response size (before JSON parse)
            response_bytes = len(response.content) if hasattr(response, 'content') else 0
            response_size_mb = response_bytes / (1024 * 1024)
            
            if response.status_code != 200:
                error_msg = response.text if hasattr(response, 'text') else str(response.content)
                logger.error(f"Helius API error {response.status_code} for {token_address[:8]}...")
                logger.debug(f"Request URL: {url}")
                logger.debug(f"Request params: {params}")
                logger.debug(f"Response: {error_msg[:500]}")
                method_chosen = "rpc_fallback"
                # Switch to RPC method
                return self._get_solana_candles_from_rpc(token_address, hours, cache_key, force_fetch, target_timestamp)
            
            # Parse JSON and capture swap count
            json_parse_start = time_module.perf_counter()
            try:
                data = response.json()
            except Exception as e:
                logger.error(f"Failed to parse Helius API response as JSON for {token_address[:8]}...: {e}")
                logger.debug(f"Response content: {response.text[:500]}")
                method_chosen = "rpc_fallback"
                return self._get_solana_candles_from_rpc(token_address, hours, cache_key, force_fetch, target_timestamp)
            json_parse_time = time_module.perf_counter() - json_parse_start
            
            if isinstance(data, list):
                swaps = data
            elif isinstance(data, dict):
                swaps = data.get('transactions', [])
            else:
                swaps = []
            
            swaps_count_returned = len(swaps)
            
            # Check routing thresholds BEFORE client-side filtering
            if swaps_count_returned > self.dex_max_swaps_guard or response_size_mb > self.dex_max_response_mb:
                logger.warning(f"CANDLE_FETCH_METRICS: DEX payload too large: swaps={swaps_count_returned}, size={response_size_mb:.2f}MB, switching to RPC")
                method_chosen = "rpc_fallback"
                return self._get_solana_candles_from_rpc(token_address, hours, cache_key, force_fetch, target_timestamp)
            
            # Client-side filtering
            filter_start = time_module.perf_counter()
            if target_timestamp and swaps:
                end_time = int(target_timestamp)
                start_time_window = end_time - (hours * 3600)
                filtered_swaps = []
                for swap in swaps:
                    swap_time = swap.get('timestamp') or swap.get('blockTime')
                    if swap_time and start_time_window <= swap_time <= end_time:
                        filtered_swaps.append(swap)
                swaps = filtered_swaps
            else:
                # No filtering needed if no target_timestamp
                pass
            filter_time = time_module.perf_counter() - filter_start
            
            swaps_count_filtered = len(swaps)
            
            # Check processing time threshold
            processing_time_so_far = time_module.perf_counter() - fetch_start_time
            if processing_time_so_far > self.dex_processing_timeout:
                logger.warning(f"CANDLE_FETCH_METRICS: DEX processing too slow: {processing_time_so_far:.2f}s > {self.dex_processing_timeout}s, switching to RPC")
                method_chosen = "rpc_fallback"
                return self._get_solana_candles_from_rpc(token_address, hours, cache_key, force_fetch, target_timestamp)
            
            if not swaps or len(swaps) < 2:
                logger.debug(f"Insufficient swap data from Helius for {token_address[:8]}... (got {len(swaps)} swaps), using price_memory")
                return self._get_solana_candles_from_memory(token_address, hours)
            
            # Process swaps into 15m candles (now returns tuple)
            candle_build_start = time_module.perf_counter()
            candles, candle_metadata = self._process_helius_swaps_to_candles(swaps, hours, target_timestamp, token_address)
            candle_build_time = time_module.perf_counter() - candle_build_start
            
            total_processing_time = time_module.perf_counter() - fetch_start_time
            
            # Log instrumentation - INFO for switches/slow/large, DEBUG for normal
            if method_chosen == "rpc_fallback" or processing_time_so_far > 5.0 or swaps_count_returned > 2000:
                logger.info(
                    f"CANDLE_FETCH_METRICS: token={token_address[:8]}, method={method_chosen}, "
                    f"swaps_returned={swaps_count_returned}, swaps_filtered={swaps_count_filtered}, "
                    f"response_mb={response_size_mb:.2f}, api_time={api_call_time:.2f}s, "
                    f"json_parse_time={json_parse_time:.2f}s, filter_time={filter_time:.2f}s, "
                    f"total_time={total_processing_time:.2f}s"
                )
            else:
                logger.debug(
                    f"CANDLE_FETCH_METRICS: token={token_address[:8]}, method={method_chosen}, "
                    f"swaps_returned={swaps_count_returned}, swaps_filtered={swaps_count_filtered}, "
                    f"response_mb={response_size_mb:.2f}, api_time={api_call_time:.2f}s, "
                    f"total_time={total_processing_time:.2f}s"
                )
            
            # CANDLE_BUILD_METRICS - always DEBUG
            logger.debug(
                f"CANDLE_BUILD_METRICS: swaps={swaps_count_filtered}, candles={len(candles) if candles else 0}, "
                f"build_time={candle_build_time:.2f}s"
            )
            
            # CANDLE_SANITY - aggregate sanity metrics for verification
            if candles and len(candles) > 0:
                candle_count = len(candles)
                expected_candles = hours * 4
                missing_candles = max(0, expected_candles - candle_count)
                
                # First and last candle times (ISO format)
                from datetime import timezone as tz
                first_candle_time = datetime.fromtimestamp(candles[0]['time'], tz=tz.utc).isoformat()
                last_candle_time = datetime.fromtimestamp(candles[-1]['time'], tz=tz.utc).isoformat()
                time_span_hours = (candles[-1]['time'] - candles[0]['time']) / 3600.0
                
                # Price range across all candles
                min_price = min(c['low'] for c in candles)
                max_price = max(c['high'] for c in candles)
                
                # Count candles with zero volume
                empty_volume_candle_count = sum(1 for c in candles if c.get('volume', 0) == 0)
                
                # Count candles with invalid OHLC (high < low or close outside [low, high])
                invalid_ohlc_count = 0
                for c in candles:
                    high = c.get('high', 0)
                    low = c.get('low', 0)
                    close = c.get('close', 0)
                    if high < low or close < low or close > high:
                        invalid_ohlc_count += 1
                
                logger.debug(
                    f"CANDLE_SANITY: token={token_address[:8]}, candle_count={candle_count}, "
                    f"expected_candles={expected_candles}, missing_candles={missing_candles}, "
                    f"first_candle_time={first_candle_time}, last_candle_time={last_candle_time}, "
                    f"time_span_hours={time_span_hours:.2f}, min_price={min_price:.8f}, max_price={max_price:.8f}, "
                    f"empty_volume_candle_count={empty_volume_candle_count}, invalid_ohlc_count={invalid_ohlc_count}"
                )
            
            # Validate candle quality
            if not self._validate_candle_quality(candles, candle_metadata, hours):
                logger.warning(
                    f"CANDLE_QUALITY: Failed quality check for {token_address[:8]}: "
                    f"swaps={candle_metadata['swaps_processed']}, candles={candle_metadata['candles_created']}, "
                    f"non_empty={candle_metadata['non_empty_candles']}, "
                    f"avg_swaps_per_candle={candle_metadata['swaps_per_candle_avg']:.1f}"
                )
                return self._get_solana_candles_from_memory(token_address, hours)
            
            logger.debug(
                f"CANDLE_QUALITY: PASSED - swaps={candle_metadata['swaps_processed']}, "
                f"candles={candle_metadata['candles_created']}, non_empty={candle_metadata['non_empty_candles']}, "
                f"avg_swaps_per_candle={candle_metadata['swaps_per_candle_avg']:.1f}"
            )
            
            # Accept candles if quality check passed (minimum 16 candles = 4h)
            if candles and len(candles) >= 16:
                self.candlestick_cache_helius[cache_key] = {
                    'data': candles,
                    'timestamp': current_time,
                    'source': 'helius'
                }
                self._save_candlestick_cache()
                
                logger.info(f"✅ Built {len(candles)} 15m candles from Helius swaps for {token_address[:8]}...")
                return candles
            
            # Fallback to price_memory if Helius data insufficient
            logger.debug(f"Helius candles insufficient ({len(candles) if candles else 0} < 16), falling back to price_memory for {token_address[:8]}...")
            return self._get_solana_candles_from_memory(token_address, hours)
            
        except Exception as e:
            logger.error(f"Error fetching Helius candlestick data for {token_address[:8]}...: {e}", exc_info=True)
            return self._get_solana_candles_from_memory(token_address, hours)
    
    def _get_solana_candles_from_indexed_swaps(
        self, token_address: str, hours: int, target_timestamp: Optional[float] = None
    ) -> Optional[List[Dict]]:
        """
        Get candlestick data from indexed swap events in SQLite.
        This is the fastest method - no API calls, just database queries.
        
        Returns tuple: (candles, missing_hours) where missing_hours is hours that need backfill
        """
        try:
            from src.storage.swap_events import get_swap_events, get_latest_swap_time
            
            # Calculate time range
            query_time = target_timestamp if target_timestamp else time.time()
            end_time = query_time
            start_time = end_time - (hours * 3600)
            
            # Query swap events from database
            swaps = get_swap_events(
                token_address=token_address,
                start_time=start_time,
                end_time=end_time,
                limit=10000,  # Reasonable limit
            )
            
            if not swaps or len(swaps) < 2:
                logger.debug(f"No indexed swaps found for {token_address[:8]}... (need at least 2)")
                return None
            
            # Convert swaps to candles
            candles = self._process_swaps_to_candles_indexed(swaps, hours, start_time, end_time)
            
            if candles and len(candles) >= 2:
                logger.debug(f"Built {len(candles)} candles from {len(swaps)} indexed swaps for {token_address[:8]}...")
                return candles
            
            return None
            
        except Exception as e:
            logger.debug(f"Error getting candles from indexed swaps: {e}")
            return None
    
    def _get_missing_hours_for_backfill(
        self, token_address: str, hours: int, target_timestamp: Optional[float] = None
    ) -> Optional[float]:
        """
        Calculate how many hours of data are missing for accurate technical indicators.
        Returns hours needed for backfill, or None if sufficient data exists.
        """
        try:
            from src.storage.swap_events import get_latest_swap_time, get_swap_count
            from src.config.config_loader import get_config_int
            
            min_candles = get_config_int("swap_indexer.min_candles_for_accuracy", 35)
            query_time = target_timestamp if target_timestamp else time.time()
            end_time = query_time
            start_time = end_time - (hours * 3600)
            
            # Check how many candles we have
            candles = self._get_solana_candles_from_indexed_swaps(
                token_address, hours, target_timestamp
            )
            
            if candles and len(candles) >= min_candles:
                return None  # Sufficient data
            
            # Calculate missing hours
            if candles:
                missing_candles = min_candles - len(candles)
                missing_hours = missing_candles  # 1 candle per hour
            else:
                missing_hours = hours
            
            # Check latest swap time to see how far back we need to go
            latest_swap_time = get_latest_swap_time(token_address)
            if latest_swap_time:
                hours_since_latest = (end_time - latest_swap_time) / 3600
                # If we have recent data, only backfill the gap
                if hours_since_latest < hours:
                    missing_hours = min(missing_hours, hours - hours_since_latest)
            
            return missing_hours if missing_hours > 0 else None
            
        except Exception as e:
            logger.debug(f"Error calculating missing hours: {e}")
            return hours  # Default to full backfill if error
    
    def _try_targeted_backfill(
        self, token_address: str, hours: int, existing_candles: List[Dict], target_timestamp: Optional[float] = None
    ) -> Optional[List[Dict]]:
        """
        Attempt targeted backfill for missing hours only.
        Returns updated candles list with backfilled data, or None if backfill failed/limited.
        """
        try:
            from src.config.config_loader import get_config, get_config_int
            from src.indexing.swap_indexer import get_indexer
            
            if not get_config("swap_indexer.enabled", True):
                return existing_candles
            
            # Check if we're allowed to backfill (rate limiting)
            backfill_tracker = getattr(self, '_backfill_tracker', {})
            current_time = time.time()
            
            # Clean old entries (older than 1 hour)
            backfill_tracker = {
                k: v for k, v in backfill_tracker.items()
                if current_time - v < 3600
            }
            self._backfill_tracker = backfill_tracker
            
            max_per_cycle = get_config_int("swap_indexer.backfill_tokens_per_cycle", 2)
            if len(backfill_tracker) >= max_per_cycle:
                logger.debug(f"Skipping backfill for {token_address[:8]}... (rate limit: {len(backfill_tracker)}/{max_per_cycle} tokens this cycle)")
                return existing_candles
            
            # Calculate missing hours
            missing_hours = self._get_missing_hours_for_backfill(token_address, hours, target_timestamp)
            if not missing_hours or missing_hours <= 0:
                return existing_candles
            
            # Limit backfill to reasonable amount (max 48 hours per token)
            missing_hours = min(missing_hours, 48)
            
            # Perform targeted backfill
            indexer = get_indexer()
            if not indexer.running:
                return existing_candles
            
            stored = indexer.backfill_missing_hours(token_address, missing_hours)
            if stored > 0:
                # Track this backfill
                self._backfill_tracker[token_address] = current_time
                
                # Re-query indexed swaps to get updated candles
                updated_candles = self._get_solana_candles_from_indexed_swaps(
                    token_address, hours, target_timestamp
                )
                if updated_candles:
                    return updated_candles
            
            return existing_candles
            
        except Exception as e:
            logger.debug(f"Error in targeted backfill: {e}")
            return existing_candles
    
    def _process_swaps_to_candles_indexed(
        self, swaps: List[Dict], hours: int, start_time: float, end_time: float
    ) -> List[Dict]:
        """Convert indexed swap events to hourly candles"""
        if not swaps:
            return []
        
        candles = {}
        hour_start = int((start_time // 3600) * 3600)
        
        for swap in swaps:
            timestamp = float(swap.get('block_time', 0))
            price = float(swap.get('price_usd', 0))
            volume = float(swap.get('volume_usd', 0) or 0)
            
            if timestamp < hour_start or price <= 0:
                continue
            
            hour = int((timestamp // 3600) * 3600)
            
            if hour not in candles:
                candles[hour] = {
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': volume,
                    'time': hour
                }
            else:
                candle = candles[hour]
                candle['high'] = max(candle['high'], price)
                candle['low'] = min(candle['low'], price)
                candle['close'] = price
                candle['volume'] += volume
        
        return sorted(candles.values(), key=lambda x: x['time'])
    
    def _process_helius_swaps_to_candles(self, swaps: List[Dict], hours: int, target_timestamp: Optional[float] = None, token_address: Optional[str] = None) -> Tuple[List[Dict], Dict]:
        """Convert Helius swap transactions to 15-minute candles (NOT hourly!)
        Returns: (candles, metadata) where metadata includes quality metrics
        """
        metadata = {
            'swaps_processed': 0,
            'swaps_skipped_timestamp': 0,
            'swaps_skipped_price': 0,
            'candles_created': 0,
            'non_empty_candles': 0,
            'total_volume': 0.0,
            'swaps_per_candle_avg': 0.0
        }
        
        if not swaps:
            return [], metadata
        
        # Group swaps by 15-minute intervals (900 seconds), not hourly!
        candles = {}
        query_time = target_timestamp if target_timestamp else time.time()
        window_start = int(query_time - (hours * 3600))
        
        for swap in swaps:
            timestamp = swap.get('timestamp', swap.get('blockTime', 0))
            if timestamp < window_start:
                metadata['swaps_skipped_timestamp'] += 1
                continue
            
            price = self._extract_price_from_helius_swap(swap, token_address)
            if not price or price <= 0:
                metadata['swaps_skipped_price'] += 1
                continue
            
            # 15-minute interval: int((timestamp // 900) * 900)
            interval_15m = int((timestamp // 900) * 900)
            
            # Get or create candle
            if interval_15m not in candles:
                candles[interval_15m] = {
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': 0,
                    'time': interval_15m
                }
            
            # Always update OHLC (works for both new and existing candles)
            candle = candles[interval_15m]
            candle['high'] = max(candle['high'], price)
            candle['low'] = min(candle['low'], price)
            candle['close'] = price
            
            # Extract volume and add to candle
            volume_usd = self._extract_volume_from_helius_swap(swap)
            candle['volume'] += volume_usd
            metadata['swaps_processed'] += 1
        
        # Calculate metadata
        result = sorted(candles.values(), key=lambda x: x['time'])
        metadata['candles_created'] = len(result)
        metadata['non_empty_candles'] = sum(1 for c in result if c['volume'] > 0)
        metadata['total_volume'] = sum(c['volume'] for c in result)
        if result:
            metadata['swaps_per_candle_avg'] = metadata['swaps_processed'] / len(result)
        
        logger.debug(f"Processed {metadata['swaps_processed']}/{len(swaps)} swaps into {len(result)} 15m candles for {token_address[:8] if token_address else 'unknown'}...")
        return result, metadata
    
    def _extract_price_from_helius_swap(self, swap: Dict, token_address: Optional[str] = None) -> Optional[float]:
        """Extract price from Helius swap transaction"""
        try:
            # Helius provides token transfers with amounts
            token_transfers = swap.get('tokenTransfers', [])
            
            if len(token_transfers) < 2:
                return None
            
            # Supported base currencies (USDC, USDT, SOL)
            USDC_MINT = "epjfwdd5aufqssqem2qn1xzybapc8g4weggkzwytdt1v"
            USDT_MINT = "es9vmfrzwaerbvgtle3i33zq3f3kmbo2fdymgzcan4"  # USDT on Solana
            SOL_MINT = "so11111111111111111111111111111111111111112"  # Wrapped SOL
            
            base_amount = 0
            token_amount = 0
            target_token_mint = None
            
            # Use provided token_address if available, otherwise try to identify it
            if token_address:
                target_token_mint = token_address.lower()
            
            # If no token_address provided, identify which token we're pricing (the one that's not a base currency)
            base_mints = {USDC_MINT, USDT_MINT, SOL_MINT}
            if not target_token_mint:
                for transfer in token_transfers:
                    mint = transfer.get('mint', '').lower()
                    if mint not in base_mints:
                        target_token_mint = mint
                        break
            
            # Second pass: extract amounts
            for transfer in token_transfers:
                mint = transfer.get('mint', '').lower()
                amount = float(transfer.get('tokenAmount', 0))
                
                if mint == USDC_MINT or mint == USDT_MINT:
                    # USDC/USDT are worth $1, so use directly
                    base_amount = amount
                elif mint == SOL_MINT:
                    # SOL needs to be converted to USD - we'll need SOL price
                    # For now, skip SOL-based swaps or use a fallback
                    # TODO: Fetch SOL price and convert
                    continue
                elif mint == target_token_mint:
                    token_amount = amount
            
            # If we have SOL-based swap, try to get SOL price
            has_sol = any(t.get('mint', '').lower() == SOL_MINT for t in token_transfers)
            if has_sol and not base_amount:
                # Get real SOL price
                try:
                    from src.utils.utils import get_sol_price_usd
                    sol_price_usd = get_sol_price_usd() or 150.0  # Fallback to $150 if fetch fails
                except Exception:
                    sol_price_usd = 150.0  # Fallback
                
                for transfer in token_transfers:
                    mint = transfer.get('mint', '').lower()
                    amount = float(transfer.get('tokenAmount', 0))
                    if mint == SOL_MINT:
                        base_amount = amount * sol_price_usd
                    elif mint == target_token_mint:
                        token_amount = amount
            
            if base_amount > 0 and token_amount > 0:
                return base_amount / token_amount
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting price from Helius swap: {e}")
            return None
    
    def _extract_volume_from_helius_swap(self, swap: Dict) -> float:
        """Extract volume in USD from Helius swap"""
        try:
            token_transfers = swap.get('tokenTransfers', [])
            USDC_MINT = "epjfwdd5aufqssqem2qn1xzybapc8g4weggkzwytdt1v"
            USDT_MINT = "es9vmfrzwaerbvgtle3i33zq3f3kmbo2fdymgzcan4"  # USDT on Solana
            SOL_MINT = "so11111111111111111111111111111111111111112"  # Wrapped SOL
            
            # Sum USDC/USDT transfers (both worth $1)
            volume = 0.0
            sol_amount = 0.0
            
            for transfer in token_transfers:
                mint = transfer.get('mint', '').lower()
                amount = float(transfer.get('tokenAmount', 0))
                
                if mint == USDC_MINT or mint == USDT_MINT:
                    volume += amount  # USDC/USDT are worth $1
                elif mint == SOL_MINT:
                    sol_amount += amount
            
            # Convert SOL to USD
            if sol_amount > 0:
                try:
                    from src.utils.utils import get_sol_price_usd
                    sol_price_usd = get_sol_price_usd() or 150.0  # Fallback to $150 if fetch fails
                except Exception:
                    sol_price_usd = 150.0  # Fallback
                volume += sol_amount * sol_price_usd
            
            return volume
            
        except Exception:
            return 0.0
    
    def _validate_candle_quality(self, candles: List[Dict], metadata: Dict, hours: int) -> bool:
        """
        Validate candle quality for 15m candles.
        Returns True if candles meet quality thresholds.
        """
        from src.config.config_loader import get_config_int, get_config_float
        
        if not candles:
            return False
        
        # Load thresholds from config
        config_min_candles = get_config_int('helius_15m_candle_policy.min_candles', 16)  # Default 16
        # Scale min_candles with requested hours: max(config_min, hours*4, 16)
        # With config_min=16: 4h=max(16,16,16)=16, 6h=max(16,24,16)=24, 2h=max(16,8,16)=16
        min_candles = max(config_min_candles, hours * 4, 16)
        
        min_swaps_in_window = get_config_int('helius_15m_candle_policy.min_swaps_in_window', 40)
        # Default to 50% coverage (hours*4*0.5) if config not set, otherwise use config value
        default_min_non_empty = int(hours * 4 * 0.5)
        min_non_empty_candles = get_config_int('helius_15m_candle_policy.min_non_empty_candles', default_min_non_empty)
        min_time_coverage_hours_config = get_config_float('helius_15m_candle_policy.min_time_coverage_hours', 4.0)
        # Ensure min_time_coverage doesn't exceed requested hours
        min_time_coverage_hours = min(min_time_coverage_hours_config, hours)
        min_swaps_per_candle = get_config_float('helius_15m_candle_policy.min_swaps_per_candle_avg', 1.5)
        
        # Check 1: Minimum candles (scales with hours)
        if len(candles) < min_candles:
            logger.debug(f"CANDLE_QUALITY: Insufficient candles: {len(candles)} < {min_candles} (requested {hours}h, config floor={config_min_candles})")
            return False
        
        # Check 2: Minimum swaps in window
        if metadata['swaps_processed'] < min_swaps_in_window:
            logger.debug(f"CANDLE_QUALITY: Insufficient swaps: {metadata['swaps_processed']} < {min_swaps_in_window}")
            return False
        
        # Check 3: Minimum non-empty candles (coverage)
        if metadata['non_empty_candles'] < min_non_empty_candles:
            logger.debug(f"CANDLE_QUALITY: Insufficient non-empty candles: {metadata['non_empty_candles']} < {min_non_empty_candles}")
            return False
        
        # Check 4: Time coverage
        time_span = candles[-1]['time'] - candles[0]['time']
        min_time_coverage_seconds = min_time_coverage_hours * 3600
        if time_span < min_time_coverage_seconds:
            logger.debug(f"CANDLE_QUALITY: Insufficient time coverage: {time_span/3600:.1f}h < {min_time_coverage_hours}h")
            return False
        
        # Check 5: Average swaps per candle (density)
        if metadata['swaps_per_candle_avg'] < min_swaps_per_candle:
            logger.debug(f"CANDLE_QUALITY: Low swap density: {metadata['swaps_per_candle_avg']:.1f} < {min_swaps_per_candle}")
            return False
        
        return True
    
    def _get_solana_candles_from_memory(self, token_address: str, hours: int) -> Optional[List[Dict]]:
        """Build candles from price_memory (NO API CALLS)"""
        try:
            from src.storage.price_memory import load_price_memory
            
            price_memory = load_price_memory()
            token_data = price_memory.get(token_address.lower())
            
            if not token_data:
                return None
            
            # Try different keys
            prices = token_data.get('prices') or token_data.get('history') or []
            current_price = token_data.get('price') or token_data.get('last_price')
            
            if not prices and not current_price:
                return None
            
            # Build candles from price history
            current_time = time.time()
            hour_start = current_time - (hours * 3600)
            
            candles = {}
            
            # Add historical prices
            for entry in prices:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    timestamp, price = entry[0], entry[1]
                elif isinstance(entry, dict):
                    timestamp = entry.get('timestamp', entry.get('time', 0))
                    price = entry.get('price', 0)
                else:
                    continue
                
                if timestamp < hour_start:
                    continue
                
                hour = int((timestamp // 3600) * 3600)
                
                if hour not in candles:
                    candles[hour] = {
                        'open': price,
                        'high': price,
                        'low': price,
                        'close': price,
                        'volume': 0,
                        'time': hour
                    }
                else:
                    candle = candles[hour]
                    candle['high'] = max(candle['high'], price)
                    candle['low'] = min(candle['low'], price)
                    candle['close'] = price
            
            # Add CURRENT price to most recent candle
            if current_price:
                current_hour = int((current_time // 3600) * 3600)
                if current_hour not in candles:
                    candles[current_hour] = {
                        'open': current_price,
                        'high': current_price,
                        'low': current_price,
                        'close': current_price,
                        'volume': 0,
                        'time': current_hour
                    }
                else:
                    candle = candles[current_hour]
                    candle['high'] = max(candle['high'], current_price)
                    candle['low'] = min(candle['low'], current_price)
                    candle['close'] = current_price
            
            result = sorted(candles.values(), key=lambda x: x['time'])
            
            if len(result) >= 10:
                logger.debug(f"✅ Built {len(result)} candles from price_memory for {token_address[:8]}...")
                return result
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not build candles from memory: {e}")
            return None
    
    def _can_make_coingecko_call(self) -> bool:
        """Check if we can make a CoinGecko API call"""
        return self.api_tracker.can_make_call('coingecko', 330)
    
    def _can_make_helius_call(self) -> bool:
        """Check if we can make a Helius API call"""
        from src.config.config_loader import get_config_int
        helius_max = get_config_int('api_rate_limiting.helius_max_daily', 30000)
        return self.api_tracker.can_make_call('helius', helius_max)
    
    def _get_candles_from_coingecko(self, token_address: str, chain_id: str, 
                                hours: int, cache_key: str) -> Optional[List[Dict]]:
        """Get candlesticks from CoinGecko for Ethereum/Base tokens (sparingly, only when needed)"""
        # Check cache first
        if cache_key in self.candlestick_cache_coingecko:
            cached = self.candlestick_cache_coingecko[cache_key]
            if time.time() - cached['timestamp'] < self.coingecko_cache_duration:
                return cached['data']
        
        # For Ethereum/Base, we'd need CoinGecko token ID mapping
        # This is a placeholder - implement if needed for Ethereum tokens
        logger.debug(f"Skipping CoinGecko fetch for {token_address[:8]}... (Ethereum/Base not yet implemented)")
        return None
    
    def _get_solana_candles_from_rpc(self, token_address: str, hours: int,
                                     cache_key: str, force_fetch: bool = False,
                                     target_timestamp: Optional[float] = None) -> Optional[List[Dict]]:
        """
        Optimized RPC approach: Query token mint address directly for swap transactions.
        Much more efficient than querying DEX programs.
        
        Strategy:
        1. Query token mint address directly (gets all transactions involving the token)
        2. Filter for swap transactions by checking for DEX program interactions
        3. Extract price data from swaps
        4. Build OHLC candles
        """
        # Check cache first
        if not force_fetch and cache_key in self.candlestick_cache_helius:
            cached = self.candlestick_cache_helius[cache_key]
            cache_age = time.time() - cached['timestamp']
            if cache_age < self.helius_cache_duration:
                logger.debug(f"✅ Using cached RPC candlestick data for {token_address[:8]}...")
                return cached['data']
        
        if not self.helius_api_key:
            logger.debug("Helius API key not configured, using price_memory fallback")
            return self._get_solana_candles_from_memory(token_address, hours)
        
        try:
            # Use Helius RPC endpoint
            rpc_url = f"https://mainnet.helius-rpc.com/?api-key={self.helius_api_key}"
            client = Client(rpc_url)
            
            # Calculate time range
            query_time = target_timestamp if target_timestamp else time.time()
            end_time = int(query_time)
            start_time = end_time - (hours * 3600)
            
            token_pubkey = Pubkey.from_string(token_address)
            swaps = []
            
            # OPTIMIZATION: Find liquidity pools containing the token, then query pool transactions
            # Token mints don't have transactions - we need to query pool addresses
            logger.info(f"Finding liquidity pools for token {token_address[:8]}...")
            
            try:
                # Step 1: Find pools containing this token using DexScreener
                pool_addresses = self._find_pools_for_token(token_address)
                
                if not pool_addresses:
                    logger.debug(f"No pools found for token {token_address[:8]}...")
                    return self._try_jupiter_price_fallback(token_address, hours, target_timestamp)
                
                logger.info(f"Found {len(pool_addresses)} pools, querying transactions from pools...")
                
                # Step 2: Query transactions from pool addresses with pagination
                # Load pagination limits from config with sensible defaults
                from src.config.config_loader import get_config_int
                max_pools = get_config_int('helius_candlestick_settings.max_pools_to_query', 2)
                max_pages = get_config_int('helius_candlestick_settings.max_pagination_pages', 5)
                max_sigs_per_page = get_config_int('helius_candlestick_settings.max_signatures_per_page', 100)
                
                for pool_address in pool_addresses[:max_pools]:  # Limit pools (default: 2, was 5)
                    try:
                        pool_pubkey = Pubkey.from_string(pool_address)
                        
                        # Paginate backwards in time to reach historical data
                        all_signatures = []
                        before_signature = None
                        # Reduced pagination limits (default: 5 pages = 500 transactions max, was 20 pages = 10,000)
                        
                        logger.debug(f"Querying pool {pool_address[:8]}... with pagination (max {max_pages} pages)")
                        
                        for page in range(max_pages):
                            # Track RPC call before making request
                            track_helius_call()
                            
                            # Get signatures for the pool address, paginating backwards
                            sigs_response = client.get_signatures_for_address(
                                pool_pubkey,
                                limit=max_sigs_per_page,  # Reduced limit (default: 100, was 500)
                                before=before_signature  # Paginate using last signature from previous page
                            )
                            
                            if not sigs_response.value:
                                break  # No more transactions
                            
                            page_signatures = []
                            found_old_enough = False
                            
                            for sig_info in sigs_response.value:
                                if sig_info.block_time:
                                    if sig_info.block_time < start_time:
                                        # We've gone too far back, stop paginating
                                        found_old_enough = True
                                        break
                                    elif start_time <= sig_info.block_time <= end_time:
                                        # This transaction is in our target range
                                        page_signatures.append(sig_info.signature)
                                else:
                                    # No block_time, include it and filter later
                                    page_signatures.append(sig_info.signature)
                            
                            all_signatures.extend(page_signatures)
                            
                            # Check if we need to continue paginating
                            if found_old_enough:
                                # We've reached transactions older than our target, stop
                                break
                            
                            # Check if the oldest transaction in this page is still too recent
                            oldest_block_time = None
                            for sig_info in sigs_response.value:
                                if sig_info.block_time:
                                    if oldest_block_time is None or sig_info.block_time < oldest_block_time:
                                        oldest_block_time = sig_info.block_time
                            
                            if oldest_block_time and oldest_block_time < start_time:
                                # We've gone back far enough
                                break
                            
                            # Set up pagination for next page
                            if sigs_response.value:
                                before_signature = sigs_response.value[-1].signature
                            else:
                                break  # No more transactions
                            
                            logger.debug(f"Page {page + 1}: Found {len(page_signatures)} transactions in range, oldest: {oldest_block_time}")
                        
                        if not all_signatures:
                            logger.debug(f"No transactions in time range for pool {pool_address[:8]}...")
                            continue
                        
                        logger.info(f"Found {len(all_signatures)} transactions in target time range from pool {pool_address[:8]}...")
                        
                        # Fetch transactions in batches
                        batch_size = 50
                        for i in range(0, len(all_signatures), batch_size):
                            batch = all_signatures[i:i + batch_size]
                            
                            try:
                                # Track RPC call for transaction batch
                                track_helius_call()
                                
                                # Get transaction details
                                txs_response = client.get_transactions(
                                    batch,
                                    max_supported_transaction_version=0
                                )
                                
                                if not txs_response.value:
                                    continue
                                
                                # Process each transaction
                                for tx_result in txs_response.value:
                                    if not tx_result or not tx_result.transaction:
                                        continue
                                    
                                    tx = tx_result.transaction
                                    meta = tx_result.transaction.meta
                                    
                                    if not meta or meta.err:
                                        continue  # Skip failed transactions
                                    
                                    # Get block time (timestamp)
                                    block_time = tx_result.block_time
                                    if not block_time:
                                        continue
                                    
                                    # Filter by time range (double-check)
                                    if block_time < start_time or block_time > end_time:
                                        continue
                                    
                                    # Check if this is a swap transaction
                                    if self._is_swap_transaction(tx, meta):
                                        swap_data = self._extract_swap_from_transaction(
                                            tx, meta, token_pubkey, token_address
                                        )
                                        
                                        if swap_data:
                                            swaps.append({
                                                'timestamp': block_time,
                                                'price': swap_data['price'],
                                                'volume': swap_data.get('volume', 0),
                                                'signature': str(tx_result.transaction.signatures[0]) if tx_result.transaction.signatures else None
                                            })
                            
                            except Exception as e:
                                logger.debug(f"Error processing transaction batch from pool: {e}")
                                continue
                    
                    except Exception as e:
                        logger.debug(f"Error querying pool {pool_address[:8]}...: {e}")
                        continue
            
            except Exception as e:
                logger.error(f"Error finding pools or querying transactions: {e}", exc_info=True)
                return self._try_jupiter_price_fallback(token_address, hours, target_timestamp)
            
            if not swaps or len(swaps) < 2:
                logger.debug(f"Insufficient swap data from RPC for {token_address[:8]}... (got {len(swaps)} swaps)")
                # Try Jupiter price API as fallback
                return self._try_jupiter_price_fallback(token_address, hours, target_timestamp)
            
            # Sort swaps by timestamp
            swaps.sort(key=lambda x: x['timestamp'])
            
            # Build candles from swaps
            candles = self._process_swaps_to_candles_rpc(swaps, hours, start_time, end_time)
            
            if candles and len(candles) >= 10:
                # Cache the result
                self.candlestick_cache_helius[cache_key] = {
                    'data': candles,
                    'timestamp': time.time(),
                    'source': 'rpc'
                }
                self._save_candlestick_cache()
                
                # Note: RPC calls already tracked above (get_signatures_for_address and get_transactions)
                
                logger.info(f"✅ Built {len(candles)} candles from RPC swaps for {token_address[:8]}...")
                return candles
            
            # Fallback to Jupiter or price_memory
            return self._try_jupiter_price_fallback(token_address, hours, target_timestamp)
            
        except Exception as e:
            logger.error(f"Error fetching RPC candlestick data for {token_address[:8]}...: {e}", exc_info=True)
            return self._try_jupiter_price_fallback(token_address, hours, target_timestamp)
    
    def _extract_swap_from_transaction(self, tx, meta, token_pubkey: Pubkey, token_address: str) -> Optional[Dict]:
        """
        Extract swap price data from a Solana transaction.
        Looks for token transfers involving the target token and USDC/SOL.
        """
        try:
            if not meta.pre_token_balances or not meta.post_token_balances:
                return None
            
            # Find token balance changes for our token
            token_amount_change = 0
            usdc_amount_change = 0
            sol_amount_change = 0
            
            # USDC mint on Solana
            USDC_MINT = Pubkey.from_string("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")
            SOL_MINT_STR = "So11111111111111111111111111111111111111112"
            
            # Track balances
            pre_balances = {}
            post_balances = {}
            
            # Process pre-token balances
            for balance in meta.pre_token_balances:
                mint = balance.mint
                if mint:
                    owner = str(balance.owner) if balance.owner else None
                    ui_amount = float(balance.ui_token_amount.ui_amount) if balance.ui_token_amount else 0
                    pre_balances[(str(mint), owner)] = ui_amount
            
            # Process post-token balances
            for balance in meta.post_token_balances:
                mint = balance.mint
                if mint:
                    owner = str(balance.owner) if balance.owner else None
                    ui_amount = float(balance.ui_token_amount.ui_amount) if balance.ui_token_amount else 0
                    post_balances[(str(mint), owner)] = ui_amount
            
            # Calculate changes
            all_mints = set(pre_balances.keys()) | set(post_balances.keys())
            
            for (mint_str, owner) in all_mints:
                pre_amount = pre_balances.get((mint_str, owner), 0)
                post_amount = post_balances.get((mint_str, owner), 0)
                change = post_amount - pre_amount
                
                try:
                    mint_pubkey = Pubkey.from_string(mint_str)
                    
                    if mint_pubkey == token_pubkey:
                        token_amount_change += abs(change)
                    elif mint_pubkey == USDC_MINT:
                        usdc_amount_change += abs(change)
                    elif mint_str == SOL_MINT_STR:  # SOL
                        sol_amount_change += abs(change)
                except Exception:
                    # Skip invalid pubkeys
                    continue
            
            # Calculate price (prefer USDC, fallback to SOL)
            if token_amount_change > 0:
                if usdc_amount_change > 0:
                    price = usdc_amount_change / token_amount_change
                    volume = usdc_amount_change
                elif sol_amount_change > 0:
                    # Convert SOL to USD (rough estimate, could fetch SOL price)
                    sol_price_usd = 150  # Rough estimate, could be improved
                    price = (sol_amount_change * sol_price_usd) / token_amount_change
                    volume = sol_amount_change * sol_price_usd
                else:
                    return None
                
                return {
                    'price': price,
                    'volume': volume,
                    'token_amount': token_amount_change
                }
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting swap data: {e}")
            return None
    
    def _is_swap_transaction(self, tx, meta) -> bool:
        """
        Check if a transaction is a swap by looking for DEX program interactions.
        """
        try:
            # DEX Program IDs
            DEX_PROGRAMS = [
                "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",  # Raydium V4
                "27haf8L6oxUeXrHrgEgsexjSY5hbVUWEmvv9Nyxg8vQv",  # Raydium V3
                "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP",  # Orca V2
                "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc",  # Orca Whirlpool
                "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",  # Jupiter V6
            ]
            
            # Check transaction message for DEX program IDs
            if hasattr(tx, 'message') and hasattr(tx.message, 'account_keys'):
                account_keys = tx.message.account_keys
                for account_key in account_keys:
                    account_str = str(account_key)
                    if account_str in DEX_PROGRAMS:
                        return True
            
            # Also check if there are token balance changes (indicates swap)
            if meta.pre_token_balances and meta.post_token_balances:
                if len(meta.pre_token_balances) > 0 and len(meta.post_token_balances) > 0:
                    # Check if balances changed (swap indicator)
                    pre_total = sum(float(b.ui_token_amount.ui_amount) if b.ui_token_amount else 0 
                                   for b in meta.pre_token_balances)
                    post_total = sum(float(b.ui_token_amount.ui_amount) if b.ui_token_amount else 0 
                                    for b in meta.post_token_balances)
                    if abs(pre_total - post_total) > 0.0001:  # Significant change
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _find_pools_for_token(self, token_address: str) -> List[str]:
        """
        Find liquidity pool addresses that contain this token.
        Uses DexScreener API to discover pools.
        """
        pool_addresses = []
        
        try:
            # Use DexScreener API to find pools/pairs for this token
            url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                pairs = data.get('pairs', [])
                
                if not pairs:
                    logger.warning(f"DexScreener returned no pairs for {token_address[:8]}...")
                    return []
                
                # Extract pool addresses from pairs
                for pair in pairs[:10]:  # Get top 10 pairs
                    pair_address = pair.get('pairAddress')
                    if pair_address:
                        # Verify it's a valid Solana address format
                        if len(pair_address) in [43, 44]:  # Solana address length
                            pool_addresses.append(pair_address)
                        else:
                            logger.debug(f"Invalid address format from DexScreener: {pair_address[:20]}...")
                
                if pool_addresses:
                    logger.info(f"Found {len(pool_addresses)} pools from DexScreener for {token_address[:8]}...")
                    return pool_addresses
                else:
                    logger.warning(f"No valid pool addresses found in DexScreener response for {token_address[:8]}...")
                    logger.debug(f"DexScreener returned {len(pairs)} pairs, but no valid addresses")
                    if pairs:
                        logger.debug(f"Sample pair data: {list(pairs[0].keys())}")
            else:
                logger.warning(f"DexScreener API returned status {response.status_code} for {token_address[:8]}...")
                logger.debug(f"Response: {response.text[:500]}")
        
        except Exception as e:
            logger.error(f"DexScreener pool discovery failed for {token_address[:8]}...: {e}", exc_info=True)
        
        return pool_addresses
    
    def _try_jupiter_price_fallback(self, token_address: str, hours: int,
                                    target_timestamp: Optional[float] = None) -> Optional[List[Dict]]:
        """
        Try Jupiter Price API as fallback (provides current price, not historical).
        For historical data, we'd need Jupiter swap history which may not be available.
        Falls back to price_memory if Jupiter fails.
        """
        try:
            # Jupiter Price API (public, no key needed)
            url = "https://price.jup.ag/v4/price"
            params = {"ids": token_address}
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json().get('data', {})
                if token_address in data:
                    price_info = data[token_address]
                    current_price = float(price_info.get('price', 0))
                    
                    if current_price > 0:
                        logger.info(f"Got current price from Jupiter: ${current_price:.6f} for {token_address[:8]}...")
                        # Jupiter only provides current price, not historical
                        # Return None to fall back to price_memory
                        return None
            
            return None
        except Exception as e:
            logger.debug(f"Jupiter price API fallback failed: {e}")
            return self._get_solana_candles_from_memory(token_address, hours)
    
    def _process_swaps_to_candles_rpc(self, swaps: List[Dict], hours: int, 
                                      start_time: int, end_time: int) -> List[Dict]:
        """
        Convert swap transactions to hourly OHLC candles.
        """
        if not swaps:
            return []
        
        candles = {}
        
        for swap in swaps:
            timestamp = swap['timestamp']
            price = swap['price']
            volume = swap.get('volume', 0)
            
            # Round to hour
            hour = int((timestamp // 3600) * 3600)
            
            if hour not in candles:
                candles[hour] = {
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': volume,
                    'time': hour,
                    'timestamp': hour
                }
            else:
                candle = candles[hour]
                candle['high'] = max(candle['high'], price)
                candle['low'] = min(candle['low'], price)
                candle['close'] = price  # Last price in the hour
                candle['volume'] += volume
        
        # Fill gaps with previous close price
        result = []
        sorted_hours = sorted(candles.keys())
        
        for i, hour in enumerate(sorted_hours):
            candle = candles[hour]
            
            # If this isn't the first candle and there's a gap, fill it
            if i > 0:
                prev_hour = sorted_hours[i - 1]
                gap_hours = (hour - prev_hour) // 3600
                
                if gap_hours > 1:
                    prev_close = candles[prev_hour]['close']
                    # Fill gaps with previous close
                    for gap_hour in range(prev_hour + 3600, hour, 3600):
                        result.append({
                            'open': prev_close,
                            'high': prev_close,
                            'low': prev_close,
                            'close': prev_close,
                            'volume': 0,
                            'time': gap_hour,
                            'timestamp': gap_hour
                        })
            
            result.append(candle)
        
        return result
    
    def _process_swaps_to_candles(self, swaps: List[Dict], hours: int) -> List[Dict]:
        """Convert Uniswap swaps to hourly candles"""
        if not swaps:
            return []
        
        # Sort by timestamp
        swaps = sorted(swaps, key=lambda x: int(x.get('timestamp', 0)))
        
        # Group swaps by hour
        candles = {}
        current_time = time.time()
        hour_start = int(current_time - (hours * 3600))
        
        for swap in swaps:
            timestamp = int(swap.get('timestamp', 0))
            if timestamp < hour_start:
                continue
            
            hour = (timestamp // 3600) * 3600
            
            # Calculate price from swap amounts
            amount0_in = float(swap.get('amount0In', 0))
            amount0_out = float(swap.get('amount0Out', 0))
            amount1_in = float(swap.get('amount1In', 0))
            amount1_out = float(swap.get('amount1Out', 0))
            
            # Determine price from swap
            if amount0_in > 0 and amount1_out > 0:
                price = amount1_out / amount0_in
            elif amount1_in > 0 and amount0_out > 0:
                price = amount1_in / amount0_out
            else:
                continue
            
            if hour not in candles:
                candles[hour] = {
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': 0
                }
            
            candle = candles[hour]
            candle['high'] = max(candle['high'], price)
            candle['low'] = min(candle['low'], price)
            candle['close'] = price  # Last price in hour
            candle['volume'] += float(swap.get('amountUSD', 0))
        
        # Convert to list sorted by time
        result = []
        for hour in sorted(candles.keys()):
            candle = candles[hour]
            candle['time'] = hour
            result.append(candle)
        
        return result
    
    def _fetch_graphql(self, url: str, query: str) -> Optional[Dict]:
        """Fetch data from GraphQL endpoint"""
        try:
            response = requests.post(url, json={'query': query}, timeout=self.api_timeout)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.warning(f"GraphQL query failed: {e}")
        
        return None

    def get_support_resistance_levels(self, token_address: str, chain_id: str = "ethereum") -> Dict:
        """
        Calculate support and resistance levels from candlestick data
        """
        try:
            candles = self.get_candlestick_data(token_address, chain_id, hours=48)
            
            if not candles or len(candles) < 2:
                return {'support': None, 'resistance': None}
            
            # Calculate support and resistance from highs and lows
            highs = [candle['high'] for candle in candles]
            lows = [candle['low'] for candle in candles]
            
            # Support is recent low
            support = min(lows[-24:] if len(lows) > 24 else lows)
            
            # Resistance is recent high
            resistance = max(highs[-24:] if len(highs) > 24 else highs)
            
            return {
                'support': support,
                'resistance': resistance,
                'current_price': candles[-1]['close']
            }
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return {'support': None, 'resistance': None}

    def get_token_volume_cached(self, token_address: str, chain_id: str) -> Optional[float]:
        """
        Get token 24h volume in USD with caching to reduce API calls.
        
        Uses DexScreener API which is free and doesn't require API key.
        Caches results for 5 minutes (configurable) to match candlestick cache duration.
        
        Args:
            token_address: Token contract address
            chain_id: Chain identifier (solana, ethereum, base, etc.)
            
        Returns:
            24h volume in USD, or None if unavailable
        """
        from src.config.config_loader import get_config_float
        
        cache_duration = get_config_float("volume_cache_duration_seconds", 300)  # Default 5 minutes
        cache_key = f"volume:{chain_id}:{token_address.lower()}"
        current_time = time.time()
        
        # Check cache first
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if current_time - cached_data.get('timestamp', 0) < cache_duration:
                volume = cached_data.get('volume')
                if volume is not None:
                    logger.debug(f"✅ Using cached volume for {token_address[:8]}... (age: {(current_time - cached_data['timestamp'])/60:.1f}m)")
                    return float(volume)
        
        # Fetch fresh volume from DexScreener
        try:
            url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
            response = requests.get(url, timeout=self.api_timeout)
            
            if response.status_code == 200:
                data = response.json()
                pairs = data.get("pairs", [])
                
                if pairs:
                    # Find the pair with highest liquidity (most reliable)
                    best_pair = max(
                        pairs,
                        key=lambda p: float(p.get("liquidity", {}).get("usd", 0) or 0)
                    )
                    
                    # Get volume from volume.h24 field
                    volume_data = best_pair.get("volume", {})
                    volume_24h = volume_data.get("h24") if isinstance(volume_data, dict) else None
                    
                    if volume_24h:
                        volume_usd = float(volume_24h)
                        
                        # Cache the result
                        self.cache[cache_key] = {
                            'volume': volume_usd,
                            'timestamp': current_time
                        }
                        
                        logger.debug(f"✅ Fetched volume for {token_address[:8]}...: ${volume_usd:,.0f}")
                        return volume_usd
                    else:
                        logger.debug(f"⚠️ No volume data in DexScreener response for {token_address[:8]}...")
                else:
                    logger.debug(f"⚠️ No pairs found in DexScreener response for {token_address[:8]}...")
            else:
                logger.warning(f"DexScreener API returned status {response.status_code} for {token_address[:8]}...")
                
        except Exception as e:
            logger.warning(f"Error fetching volume from DexScreener for {token_address[:8]}...: {e}")
        
        return None


# Global instance
market_data_fetcher = MarketDataFetcher()
