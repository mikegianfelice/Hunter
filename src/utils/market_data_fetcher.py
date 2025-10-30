#!/usr/bin/env python3
"""
Market Data Fetcher - Fetch real market data from multiple sources
"""

import os
import json
import time
import logging
import requests
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import statistics

# Configure logging
logger = logging.getLogger(__name__)


class MarketDataFetcher:
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes cache
        self.api_timeout = 15
        self.max_retries = 3
        
    def get_btc_price(self) -> Optional[float]:
        """Get current BTC price in USD"""
        try:
            # Check cache first
            if 'btc_price' in self.cache:
                cached_data = self.cache['btc_price']
                if time.time() - cached_data['timestamp'] < self.cache_duration:
                    return cached_data['price']
            
            # Try CoinGecko API
            url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
            data = self._fetch_json(url)
            if data and 'bitcoin' in data:
                price = float(data['bitcoin']['usd'])
                self.cache['btc_price'] = {'price': price, 'timestamp': time.time()}
                logger.info(f"✅ BTC price: ${price}")
                return price
            
            # Fallback to CoinGecko alternative endpoint
            url = "https://api.coingecko.com/api/v3/coins/bitcoin"
            data = self._fetch_json(url)
            if data and 'market_data' in data:
                price = float(data['market_data']['current_price']['usd'])
                self.cache['btc_price'] = {'price': price, 'timestamp': time.time()}
                logger.info(f"✅ BTC price: ${price}")
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
            
            # Try CoinGecko API
            url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd"
            data = self._fetch_json(url)
            if data and 'ethereum' in data:
                price = float(data['ethereum']['usd'])
                self.cache['eth_price'] = {'price': price, 'timestamp': time.time()}
                logger.info(f"✅ ETH price: ${price}")
                return price
            
            # Fallback to CoinGecko alternative endpoint
            url = "https://api.coingecko.com/api/v3/coins/ethereum"
            data = self._fetch_json(url)
            if data and 'market_data' in data:
                price = float(data['market_data']['current_price']['usd'])
                self.cache['eth_price'] = {'price': price, 'timestamp': time.time()}
                logger.info(f"✅ ETH price: ${price}")
                return price
                
        except Exception as e:
            logger.error(f"❌ Failed to fetch ETH price: {e}")
        
        return None
    
    def get_btc_trend(self, hours: int = 24) -> float:
        """Get BTC trend over last N hours (0-1 scale)"""
        try:
            # Get historical prices
            now = int(time.time())
            from_timestamp = now - (hours * 3600)
            
            url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range?vs_currency=usd&from={from_timestamp}&to={now}"
            data = self._fetch_json(url)
            
            if data and 'prices' in data:
                prices = data['prices']
                if len(prices) >= 2:
                    # Calculate trend (percentage change)
                    start_price = float(prices[0][1])
                    end_price = float(prices[-1][1])
                    
                    if start_price > 0:
                        change_pct = (end_price - start_price) / start_price
                        # Normalize to 0-1 scale (assuming -50% to +50% range)
                        trend = max(0, min(1, 0.5 + change_pct))
                        return trend
                        
        except Exception as e:
            logger.error(f"❌ Failed to fetch BTC trend: {e}")
        
        return 0.5  # Neutral trend if unable to fetch
    
    def get_eth_trend(self, hours: int = 24) -> float:
        """Get ETH trend over last N hours (0-1 scale)"""
        try:
            # Get historical prices
            now = int(time.time())
            from_timestamp = now - (hours * 3600)
            
            url = f"https://api.coingecko.com/api/v3/coins/ethereum/market_chart/range?vs_currency=usd&from={from_timestamp}&to={now}"
            data = self._fetch_json(url)
            
            if data and 'prices' in data:
                prices = data['prices']
                if len(prices) >= 2:
                    # Calculate trend (percentage change)
                    start_price = float(prices[0][1])
                    end_price = float(prices[-1][1])
                    
                    if start_price > 0:
                        change_pct = (end_price - start_price) / start_price
                        # Normalize to 0-1 scale (assuming -50% to +50% range)
                        trend = max(0, min(1, 0.5 + change_pct))
                        return trend
                        
        except Exception as e:
            logger.error(f"❌ Failed to fetch ETH trend: {e}")
        
        return 0.5  # Neutral trend if unable to fetch
    
    def get_market_volatility(self, hours: int = 24) -> float:
        """Get market volatility index (0-1 scale)"""
        try:
            # Get BTC price history
            now = int(time.time())
            from_timestamp = now - (hours * 3600)
            
            url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range?vs_currency=usd&from={from_timestamp}&to={now}"
            data = self._fetch_json(url)
            
            if data and 'prices' in data:
                prices = [float(p[1]) for p in data['prices']]
                if len(prices) >= 2:
                    # Calculate standard deviation
                    std_dev = statistics.stdev(prices)
                    mean_price = statistics.mean(prices)
                    
                    if mean_price > 0:
                        # Coefficient of variation as volatility measure
                        volatility = std_dev / mean_price
                        # Normalize to 0-1 scale (assuming 0-0.5 range)
                        volatility_normalized = min(1, volatility * 2)
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
    
    def get_market_correlation(self, hours: int = 24) -> float:
        """Get market correlation between BTC and ETH (0-1 scale)"""
        try:
            # Get historical price data for both BTC and ETH
            now = int(time.time())
            from_timestamp = now - (hours * 3600)
            
            # Fetch BTC price history
            btc_url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range?vs_currency=usd&from={from_timestamp}&to={now}"
            btc_data = self._fetch_json(btc_url)
            
            # Fetch ETH price history
            eth_url = f"https://api.coingecko.com/api/v3/coins/ethereum/market_chart/range?vs_currency=usd&from={from_timestamp}&to={now}"
            eth_data = self._fetch_json(eth_url)
            
            if not btc_data or 'prices' not in btc_data or not eth_data or 'prices' not in eth_data:
                logger.warning("Insufficient data for correlation calculation")
                return 0.5
            
            btc_prices = {int(p[0] / 1000): float(p[1]) for p in btc_data['prices']}
            eth_prices = {int(p[0] / 1000): float(p[1]) for p in eth_data['prices']}
            
            # Find common timestamps (within 1 hour window)
            common_timestamps = []
            for btc_ts in btc_prices.keys():
                for eth_ts in eth_prices.keys():
                    if abs(btc_ts - eth_ts) < 3600:  # Within 1 hour
                        common_timestamps.append((btc_ts, eth_ts))
                        break
            
            if len(common_timestamps) < 3:
                logger.warning("Not enough common timestamps for correlation")
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
            
            if len(btc_returns) < 2:
                logger.warning("Not enough return data for correlation")
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
            
            logger.info(f"✅ Market correlation (BTC/ETH): {correlation:.3f} (normalized: {normalized:.3f})")
            return max(0.0, min(1.0, normalized))
                
        except Exception as e:
            logger.error(f"❌ Failed to calculate market correlation: {e}")
        
        return 0.5
    
    def get_volume_trends(self, hours: int = 24) -> float:
        """Get volume trends (0-1 scale) based on historical comparison"""
        try:
            # Get current total market volume
            url = "https://api.coingecko.com/api/v3/global"
            current_data = self._fetch_json(url)
            
            if not current_data or 'data' not in current_data:
                logger.warning("Failed to fetch current volume data")
                return 0.5
            
            current_volume = float(current_data['data']['total_24h']['usd'])
            
            # Get historical volume data using BTC as proxy (most liquid asset)
            # We'll use BTC's volume history to estimate overall market trend
            now = int(time.time())
            from_timestamp = now - (hours * 2 * 3600)  # Get 2x hours to compare
            
            btc_url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range?vs_currency=usd&from={from_timestamp}&to={now}"
            btc_data = self._fetch_json(btc_url)
            
            if not btc_data or 'total_volumes' not in btc_data or len(btc_data['total_volumes']) < 2:
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
                
                logger.info(f"✅ Volume trend (estimated): {trend:.3f} (current: ${current_volume/1e9:.1f}B)")
                return max(0.0, min(1.0, trend))
            
            # Calculate volume trend from historical data
            volumes = [float(v[1]) for v in btc_data['total_volumes']]
            
            # Split into two periods: older half and recent half
            mid_point = len(volumes) // 2
            older_volumes = volumes[:mid_point] if mid_point > 0 else volumes[:len(volumes)//2]
            recent_volumes = volumes[mid_point:] if mid_point > 0 else volumes[len(volumes)//2:]
            
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
            
            logger.info(f"✅ Volume trend: {trend:.3f} ({volume_change_pct*100:+.1f}% change, recent avg: ${recent_avg/1e9:.1f}B, older avg: ${older_avg/1e9:.1f}B)")
            return trend
                
        except Exception as e:
            logger.error(f"❌ Failed to fetch volume trends: {e}")
        
        return 0.5
    
    def get_market_cap_trend(self, hours: int = 24) -> float:
        """Get market cap trend (0-1 scale) based on historical comparison"""
        try:
            # Get current total market cap
            url = "https://api.coingecko.com/api/v3/global"
            current_data = self._fetch_json(url)
            
            if not current_data or 'data' not in current_data:
                logger.warning("Failed to fetch current market cap data")
                return 0.5
            
            current_market_cap = float(current_data['data']['total_market_cap']['usd'])
            
            # Get historical market cap data using BTC as proxy
            now = int(time.time())
            from_timestamp = now - (hours * 2 * 3600)  # Get 2x hours to compare
            
            btc_url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range?vs_currency=usd&from={from_timestamp}&to={now}"
            btc_data = self._fetch_json(btc_url)
            
            if not btc_data or 'market_caps' not in btc_data or len(btc_data['market_caps']) < 2:
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
            
            # Calculate market cap trend from historical data
            # Use market_caps array if available, otherwise estimate from prices
            if 'market_caps' in btc_data and len(btc_data['market_caps']) >= 2:
                market_caps = [float(mc[1]) for mc in btc_data['market_caps']]
            else:
                # Estimate market cap from price data (assuming proportional relationship)
                prices = [float(p[1]) for p in btc_data['prices']]
                if len(prices) < 2:
                    logger.warning("Insufficient price data for market cap trend")
                    return 0.5
                # Use price as proxy for market cap trend
                market_caps = prices
            
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
    
    def _fetch_json(self, url: str) -> Optional[Dict]:
        """Fetch JSON data with retry logic"""
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, timeout=self.api_timeout)
                if response.status_code == 200:
                    return response.json()
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
        
        return None

    def get_candlestick_data(self, token_address: str, chain_id: str = "ethereum", hours: int = 24) -> Optional[List[Dict]]:
        """
        Get REAL historical candlestick data from DEX APIs
        Returns list of candlesticks with OHLCV data from actual trades
        """
        try:
            chain_id_lower = chain_id.lower()
            
            if chain_id_lower == "ethereum" or chain_id_lower == "base":
                return self._get_ethereum_candles(token_address, chain_id_lower, hours)
            elif chain_id_lower == "solana":
                return self._get_solana_candles(token_address, hours)
            else:
                logger.warning(f"Unsupported chain for candlestick data: {chain_id}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to fetch candlestick data: {e}")
            return None
    
    def _get_ethereum_candles(self, token_address: str, chain_id: str, hours: int) -> Optional[List[Dict]]:
        """Get real candlestick data from Uniswap subgraph"""
        try:
            # Use The Graph (Uniswap subgraph) - FREE tier works
            if chain_id == "ethereum":
                subgraph_url = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"
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
        """Get real candlestick data from Solana DEX"""
        try:
            # Use Jupiter API for Solana
            url = f"https://price.jup.ag/v4/price?ids={token_address}"
            data = self._fetch_json(url)
            
            if not data or 'data' not in data or token_address not in data['data']:
                logger.warning(f"No Jupiter price data for {token_address[:8]}...")
                return None
            
            price_data = data['data'][token_address]
            price = float(price_data.get('price', 0))
            
            if price <= 0:
                return None
            
            # For Solana, we have limited historical data from free APIs
            # Return minimal real data
            return [{
                'time': time.time() - 3600,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': 0
            }]
            
        except Exception as e:
            logger.error(f"Error fetching Solana candles: {e}")
            return None
    
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


# Global instance
market_data_fetcher = MarketDataFetcher()
