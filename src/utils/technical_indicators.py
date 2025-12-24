#!/usr/bin/env python3
"""
Technical Indicators Calculator
Computes RSI, MACD, Bollinger Bands, VWAP, and other indicators from price history
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate technical indicators from OHLCV data"""
    
    def __init__(self):
        self.min_periods = {
            'rsi': 14,
            'macd': 26,  # Fast EMA period
            'macd_signal': 9,  # Signal line period
            'macd_slow': 12,  # Slow EMA period
            'bollinger': 20,
            'bollinger_std': 2,
            'volume_profile': 20,
            'price_action': 10
        }
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index (RSI)"""
        if len(prices) < period + 1:
            return 50.0  # Neutral default
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def calculate_macd(self, prices: pd.Series, 
                       fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if len(prices) < slow + signal:
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
        
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0,
            'signal': float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0,
            'histogram': float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0.0
        }
    
    def calculate_bollinger_bands(self, prices: pd.Series, 
                                  period: int = 20, std_dev: int = 2) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            current_price = float(prices.iloc[-1]) if len(prices) > 0 else 0.0
            return {
                'upper': current_price * 1.1,
                'middle': current_price,
                'lower': current_price * 0.9,
                'width': 0.0,
                'position': 0.5  # Middle of band
            }
        
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        band_width = (upper_band - lower_band) / sma
        
        current_price = float(prices.iloc[-1])
        current_upper = float(upper_band.iloc[-1])
        current_lower = float(lower_band.iloc[-1])
        current_middle = float(sma.iloc[-1])
        
        # Position: 0 = at lower band, 1 = at upper band
        if current_upper != current_lower:
            position = (current_price - current_lower) / (current_upper - current_lower)
        else:
            position = 0.5
        
        return {
            'upper': current_upper,
            'middle': current_middle,
            'lower': current_lower,
            'width': float(band_width.iloc[-1]) if not pd.isna(band_width.iloc[-1]) else 0.0,
            'position': max(0.0, min(1.0, position))
        }
    
    def calculate_volume_profile(self, prices: pd.Series, volumes: pd.Series, 
                                  bins: int = 20) -> Dict[str, float]:
        """Calculate volume profile (price levels with highest volume)"""
        if len(prices) < 10 or len(volumes) < 10:
            return {
                'poc': float(prices.iloc[-1]) if len(prices) > 0 else 0.0,  # Point of Control
                'vah': float(prices.iloc[-1]) if len(prices) > 0 else 0.0,  # Value Area High
                'val': float(prices.iloc[-1]) if len(prices) > 0 else 0.0,  # Value Area Low
                'volume_ratio': 1.0
            }
        
        # Create price bins
        price_min = float(prices.min())
        price_max = float(prices.max())
        if price_max == price_min:
            current_price = float(prices.iloc[-1])
            return {
                'poc': current_price,
                'vah': current_price,
                'val': current_price,
                'volume_ratio': 1.0
            }
        
        bin_size = (price_max - price_min) / bins
        price_bins = [price_min + i * bin_size for i in range(bins + 1)]
        
        # Aggregate volume by price bin
        volume_by_bin = {}
        for i in range(len(prices)):
            price = float(prices.iloc[i])
            volume = float(volumes.iloc[i]) if i < len(volumes) else 0.0
            
            # Find bin
            bin_idx = min(int((price - price_min) / bin_size), bins - 1)
            bin_center = price_bins[bin_idx] + bin_size / 2
            
            volume_by_bin[bin_center] = volume_by_bin.get(bin_center, 0) + volume
        
        if not volume_by_bin:
            current_price = float(prices.iloc[-1])
            return {
                'poc': current_price,
                'vah': current_price,
                'val': current_price,
                'volume_ratio': 1.0
            }
        
        # Find Point of Control (POC) - price level with highest volume
        poc = max(volume_by_bin, key=volume_by_bin.get)
        
        # Calculate Value Area (70% of volume)
        sorted_volumes = sorted(volume_by_bin.items(), key=lambda x: x[1], reverse=True)
        total_volume = sum(volume_by_bin.values())
        target_volume = total_volume * 0.7
        
        cumulative_volume = 0
        value_area_prices = []
        for price, volume in sorted_volumes:
            cumulative_volume += volume
            value_area_prices.append(price)
            if cumulative_volume >= target_volume:
                break
        
        vah = max(value_area_prices) if value_area_prices else poc
        val = min(value_area_prices) if value_area_prices else poc
        
        current_price = float(prices.iloc[-1])
        volume_ratio = volume_by_bin.get(poc, 0) / total_volume if total_volume > 0 else 1.0
        
        return {
            'poc': poc,
            'vah': vah,
            'val': val,
            'volume_ratio': volume_ratio,
            'current_vs_poc': (current_price - poc) / poc if poc > 0 else 0.0
        }
    
    def calculate_vwap(self, candles: List[Dict]) -> Dict[str, float]:
        """Calculate Volume Weighted Average Price (VWAP)"""
        if not candles or len(candles) < 1:
            return {
                'vwap': 0.0,
                'price_vs_vwap': 0.0,
                'is_above_vwap': False,
                'vwap_distance_pct': 0.0
            }
        
        try:
            # Calculate cumulative Typical Price × Volume and cumulative Volume
            cumulative_tpv = 0.0  # Typical Price × Volume
            cumulative_volume = 0.0
            
            for candle in candles:
                high = float(candle.get('high', 0))
                low = float(candle.get('low', 0))
                close = float(candle.get('close', 0))
                volume = float(candle.get('volume', 0))
                
                # Typical Price = (High + Low + Close) / 3
                # This is the standard VWAP calculation
                typical_price = (high + low + close) / 3
                
                cumulative_tpv += typical_price * volume
                cumulative_volume += volume
            
            if cumulative_volume == 0:
                current_price = float(candles[-1].get('close', 0))
                return {
                    'vwap': current_price,
                    'price_vs_vwap': 0.0,
                    'is_above_vwap': False,
                    'vwap_distance_pct': 0.0
                }
            
            vwap = cumulative_tpv / cumulative_volume
            current_price = float(candles[-1].get('close', 0))
            
            # Calculate price position relative to VWAP
            price_vs_vwap = ((current_price - vwap) / vwap * 100) if vwap > 0 else 0.0
            is_above_vwap = current_price > vwap
            vwap_distance_pct = abs(price_vs_vwap)
            
            return {
                'vwap': vwap,
                'price_vs_vwap': price_vs_vwap,  # Percentage above/below VWAP
                'is_above_vwap': is_above_vwap,
                'vwap_distance_pct': vwap_distance_pct
            }
            
        except Exception as e:
            logger.error(f"Error calculating VWAP: {e}")
            current_price = float(candles[-1].get('close', 0)) if candles else 0.0
            return {
                'vwap': current_price,
                'price_vs_vwap': 0.0,
                'is_above_vwap': False,
                'vwap_distance_pct': 0.0
            }
    
    def calculate_price_action_patterns(self, candles: List[Dict]) -> Dict[str, float]:
        """Detect price action patterns"""
        if len(candles) < 3:
            return {
                'trend_strength': 0.5,
                'volatility': 0.5,
                'momentum': 0.5,
                'support_resistance': 0.5
            }
        
        # Extract OHLC data
        closes = [float(c['close']) for c in candles]
        highs = [float(c['high']) for c in candles]
        lows = [float(c['low']) for c in candles]
        
        # Trend strength (based on higher highs and higher lows)
        higher_highs = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
        higher_lows = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])
        trend_strength = (higher_highs + higher_lows) / (2 * (len(highs) - 1))
        
        # Volatility (coefficient of variation)
        if len(closes) > 1:
            volatility = np.std(closes) / np.mean(closes) if np.mean(closes) > 0 else 0.0
            volatility = min(1.0, volatility * 10)  # Normalize to 0-1
        else:
            volatility = 0.5
        
        # Momentum (rate of change)
        if len(closes) >= 5:
            momentum = (closes[-1] - closes[-5]) / closes[-5] if closes[-5] > 0 else 0.0
            momentum = max(-1.0, min(1.0, momentum * 5))  # Normalize to -1 to 1, then scale to 0-1
            momentum = (momentum + 1) / 2  # Convert to 0-1 range
        else:
            momentum = 0.5
        
        # Support/Resistance (based on price clustering)
        price_ranges = [(lows[i], highs[i]) for i in range(len(lows))]
        if len(price_ranges) >= 3:
            # Find price levels where price spent most time
            all_prices = []
            for low, high in price_ranges[-10:]:  # Last 10 candles
                if high > low:
                    all_prices.extend([low, high])
            
            if all_prices:
                price_std = np.std(all_prices)
                price_mean = np.mean(all_prices)
                # Lower std relative to mean = stronger support/resistance
                support_resistance = 1.0 - min(1.0, price_std / price_mean if price_mean > 0 else 1.0)
            else:
                support_resistance = 0.5
        else:
            support_resistance = 0.5
        
        return {
            'trend_strength': max(0.0, min(1.0, trend_strength)),
            'volatility': max(0.0, min(1.0, volatility)),
            'momentum': max(0.0, min(1.0, momentum)),
            'support_resistance': max(0.0, min(1.0, support_resistance))
        }
    
    def calculate_all_indicators(self, candles: List[Dict], include_confidence: bool = True) -> Dict[str, any]:
        """Calculate all technical indicators from candlestick data with confidence scores"""
        if not candles or len(candles) < 2:
            result = self._get_default_indicators()
            if include_confidence:
                result['data_quality'] = {
                    'total_candles': 0,
                    'confidence_score': 0.0,
                    'is_approximation': True,
                    'warnings': ['Insufficient data: less than 2 candles']
                }
            return result
        
        try:
            # Convert to pandas DataFrame
            df = pd.DataFrame(candles)
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['high'] = pd.to_numeric(df['high'], errors='coerce')
            df['low'] = pd.to_numeric(df['low'], errors='coerce')
            df['volume'] = pd.to_numeric(df.get('volume', 0), errors='coerce').fillna(0)
            
            # Remove any rows with NaN
            df = df.dropna(subset=['close'])
            
            if len(df) < 2:
                result = self._get_default_indicators()
                if include_confidence:
                    result['data_quality'] = {
                        'total_candles': len(df),
                        'confidence_score': 0.0,
                        'is_approximation': True,
                        'warnings': ['Insufficient data after cleaning']
                    }
                return result
            
            prices = df['close']
            volumes = df['volume']
            num_candles = len(prices)
            
            # Calculate indicators
            rsi = self.calculate_rsi(prices)
            macd = self.calculate_macd(prices)
            bollinger = self.calculate_bollinger_bands(prices)
            volume_profile = self.calculate_volume_profile(prices, volumes)
            vwap = self.calculate_vwap(candles)
            price_action = self.calculate_price_action_patterns(candles)
            
            # Calculate confidence scores for each indicator
            confidence_scores = {}
            warnings = []
            
            # RSI confidence (needs 15+ periods)
            rsi_confidence = min(1.0, num_candles / 15.0) if num_candles < 15 else 1.0
            if num_candles < 15:
                warnings.append(f'RSI calculated with limited data ({num_candles} candles, needs 15+)')
            confidence_scores['rsi'] = rsi_confidence
            
            # MACD confidence (needs 35+ periods for accurate signal)
            macd_confidence = min(1.0, num_candles / 35.0) if num_candles < 35 else 1.0
            if num_candles < 35:
                warnings.append(f'MACD calculated with limited data ({num_candles} candles, needs 35+)')
            confidence_scores['macd'] = macd_confidence
            
            # Bollinger Bands confidence (needs 20+ periods)
            bb_confidence = min(1.0, num_candles / 20.0) if num_candles < 20 else 1.0
            if num_candles < 20:
                warnings.append(f'Bollinger Bands calculated with limited data ({num_candles} candles, needs 20+)')
            confidence_scores['bollinger'] = bb_confidence
            
            # MA-20 confidence
            ma20_confidence = min(1.0, num_candles / 20.0) if num_candles < 20 else 1.0
            confidence_scores['ma20'] = ma20_confidence
            
            # MA-50 confidence
            ma50_confidence = min(1.0, num_candles / 50.0) if num_candles < 50 else 1.0
            if num_candles < 50:
                warnings.append(f'MA-50 calculated with limited data ({num_candles} candles, needs 50+)')
            confidence_scores['ma50'] = ma50_confidence
            
            # VWAP confidence (works with any data, but better with more)
            vwap_confidence = min(1.0, num_candles / 24.0) if num_candles < 24 else 1.0
            confidence_scores['vwap'] = vwap_confidence
            
            # Overall confidence score (weighted average)
            overall_confidence = (
                rsi_confidence * 0.15 +
                macd_confidence * 0.25 +
                bb_confidence * 0.15 +
                ma20_confidence * 0.15 +
                ma50_confidence * 0.15 +
                vwap_confidence * 0.15
            )
            
            result = {
                'rsi': rsi,
                'macd': macd,
                'bollinger_bands': bollinger,
                'volume_profile': volume_profile,
                'vwap': vwap,
                'price_action': price_action,
                'moving_avg_20': float(prices.tail(20).mean()) if len(prices) >= 20 else float(prices.mean()),
                'moving_avg_50': float(prices.tail(50).mean()) if len(prices) >= 50 else float(prices.mean()),
                'current_price': float(prices.iloc[-1]),
                'price_change_24h': ((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100) if prices.iloc[0] > 0 else 0.0
            }
            
            if include_confidence:
                result['data_quality'] = {
                    'total_candles': num_candles,
                    'confidence_score': overall_confidence,
                    'is_approximation': overall_confidence < 0.7,
                    'indicator_confidence': confidence_scores,
                    'warnings': warnings if warnings else []
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            result = self._get_default_indicators()
            if include_confidence:
                result['data_quality'] = {
                    'total_candles': 0,
                    'confidence_score': 0.0,
                    'is_approximation': True,
                    'warnings': [f'Calculation error: {str(e)}']
                }
            return result
    
    def _get_default_indicators(self) -> Dict[str, any]:
        """Return default indicators when calculation fails"""
        return {
            'rsi': 50.0,
            'macd': {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0},
            'bollinger_bands': {'upper': 0.0, 'middle': 0.0, 'lower': 0.0, 'width': 0.0, 'position': 0.5},
            'volume_profile': {'poc': 0.0, 'vah': 0.0, 'val': 0.0, 'volume_ratio': 1.0, 'current_vs_poc': 0.0},
            'vwap': {'vwap': 0.0, 'price_vs_vwap': 0.0, 'is_above_vwap': False, 'vwap_distance_pct': 0.0},
            'price_action': {'trend_strength': 0.5, 'volatility': 0.5, 'momentum': 0.5, 'support_resistance': 0.5},
            'moving_avg_20': 0.0,
            'moving_avg_50': 0.0,
            'current_price': 0.0,
            'price_change_24h': 0.0
        }
    
    def calculate_multi_timeframe_indicators(self, candles_1h: List[Dict], 
                                            candles_4h: Optional[List[Dict]] = None,
                                            candles_24h: Optional[List[Dict]] = None) -> Dict[str, any]:
        """Calculate indicators across multiple timeframes for better analysis"""
        result = {
            'timeframe_1h': self.calculate_all_indicators(candles_1h, include_confidence=True),
            'timeframe_4h': None,
            'timeframe_24h': None,
            'multi_timeframe_summary': {}
        }
        
        if candles_4h and len(candles_4h) >= 2:
            result['timeframe_4h'] = self.calculate_all_indicators(candles_4h, include_confidence=True)
        
        if candles_24h and len(candles_24h) >= 2:
            result['timeframe_24h'] = self.calculate_all_indicators(candles_24h, include_confidence=True)
        
        # Create summary across timeframes
        if result['timeframe_1h']:
            tf1h = result['timeframe_1h']
            summary = {
                'rsi_trend': 'neutral',
                'macd_alignment': 'neutral',
                'trend_consensus': 'neutral',
                'confidence': tf1h.get('data_quality', {}).get('confidence_score', 0.0)
            }
            
            # Check RSI trend across timeframes
            rsi_values = []
            if tf1h.get('rsi'):
                rsi_values.append(('1h', tf1h['rsi']))
            if result['timeframe_4h'] and result['timeframe_4h'].get('rsi'):
                rsi_values.append(('4h', result['timeframe_4h']['rsi']))
            if result['timeframe_24h'] and result['timeframe_24h'].get('rsi'):
                rsi_values.append(('24h', result['timeframe_24h']['rsi']))
            
            if len(rsi_values) >= 2:
                # Check if RSI is trending in same direction
                if all(rsi < 50 for _, rsi in rsi_values):
                    summary['rsi_trend'] = 'bearish'
                elif all(rsi > 50 for _, rsi in rsi_values):
                    summary['rsi_trend'] = 'bullish'
            
            result['multi_timeframe_summary'] = summary
        
        return result


# Global instance
technical_indicators = TechnicalIndicators()

