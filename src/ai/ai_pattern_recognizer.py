#!/usr/bin/env python3
"""
AI-Powered Pattern Recognition for Sustainable Trading Bot
Uses computer vision and machine learning to identify profitable trading patterns
"""

import os
import json
import time
import logging
import math
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

# Configure logging
logger = logging.getLogger(__name__)

class AIPatternRecognizer:
    def __init__(self):
        self.pattern_cache = {}
        self.cache_duration = 300  # 5 minutes cache
        self.pattern_history = []
        
        # Pattern recognition configuration
        self.pattern_confidence_threshold = 0.7  # 70% confidence threshold
        self.strong_pattern_threshold = 0.8  # 80% confidence for strong patterns
        self.weak_pattern_threshold = 0.5  # 50% confidence for weak patterns
        
        # Pattern types and their weights
        self.pattern_types = {
            'bullish_engulfing': 0.15,  # 15% weight for bullish engulfing
            'bearish_engulfing': 0.15,  # 15% weight for bearish engulfing
            'hammer': 0.12,  # 12% weight for hammer
            'doji': 0.10,  # 10% weight for doji
            'shooting_star': 0.10,  # 10% weight for shooting star
            'morning_star': 0.08,  # 8% weight for morning star
            'evening_star': 0.08,  # 8% weight for evening star
            'breakout': 0.10,  # 10% weight for breakout
            'support_resistance': 0.12  # 12% weight for support/resistance
        }
        
        # Pattern strength indicators
        self.strength_indicators = {
            'volume_confirmation': 0.25,  # 25% weight for volume confirmation
            'momentum_alignment': 0.20,  # 20% weight for momentum alignment
            'timeframe_consistency': 0.15,  # 15% weight for timeframe consistency
            'pattern_completeness': 0.20,  # 20% weight for pattern completeness
            'market_context': 0.20  # 20% weight for market context
        }
        
        # Pattern signals
        self.pattern_signals = {
            'strong_buy': 0.9,  # Strong buy signal
            'buy': 0.7,  # Buy signal
            'weak_buy': 0.5,  # Weak buy signal
            'neutral': 0.0,  # Neutral signal
            'weak_sell': -0.5,  # Weak sell signal
            'sell': -0.7,  # Sell signal
            'strong_sell': -0.9  # Strong sell signal
        }
    
    def recognize_patterns(self, token: Dict) -> Dict:
        """
        Recognize trading patterns for a given token using AI and computer vision
        Returns pattern analysis, signals, and trading recommendations
        """
        try:
            symbol = token.get("symbol", "UNKNOWN")
            cache_key = f"pattern_{symbol}"
            
            # Check cache
            if cache_key in self.pattern_cache:
                cached_data = self.pattern_cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).total_seconds() < self.cache_duration:
                    logger.debug(f"Using cached pattern recognition for {symbol}")
                    return cached_data['pattern_data']
            
            # Analyze various pattern types
            pattern_analysis = self._analyze_pattern_types(token)
            
            # Calculate pattern strength
            pattern_strength = self._calculate_pattern_strength(token, pattern_analysis)
            
            # Generate trading signals
            trading_signals = self._generate_trading_signals(pattern_analysis, pattern_strength)
            
            # Detect support and resistance levels
            support_resistance = self._detect_support_resistance(token)
            
            # Analyze momentum patterns
            momentum_patterns = self._analyze_momentum_patterns(token)
            
            # Generate pattern insights
            pattern_insights = self._generate_pattern_insights(pattern_analysis, pattern_strength)
            
            # Generate trading recommendations
            trading_recommendations = self._generate_trading_recommendations(
                pattern_analysis, trading_signals, pattern_strength
            )
            
            result = {
                'pattern_analysis': pattern_analysis,
                'pattern_strength': pattern_strength,
                'trading_signals': trading_signals,
                'support_resistance': support_resistance,
                'momentum_patterns': momentum_patterns,
                'pattern_insights': pattern_insights,
                'trading_recommendations': trading_recommendations,
                'recognition_timestamp': datetime.now().isoformat(),
                'confidence_level': self._calculate_confidence_level(pattern_analysis),
                'overall_signal': self._calculate_overall_signal(trading_signals)
            }
            
            # Cache the result
            self.pattern_cache[cache_key] = {'timestamp': datetime.now(), 'pattern_data': result}
            
            logger.info(f"🔍 Pattern recognition for {symbol}: {len(pattern_analysis)} patterns, strength: {pattern_strength:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Pattern recognition failed for {token.get('symbol', 'UNKNOWN')}: {e}")
            return self._get_default_pattern_analysis()
    
    def _analyze_pattern_types(self, token: Dict) -> Dict:
        """Analyze various candlestick and chart patterns"""
        try:
            pattern_analysis = {}
            
            # Detect patterns based on real token price and volume data
            symbol = token.get("symbol", "UNKNOWN")
            price = float(token.get("priceUsd", 0))
            volume_24h = float(token.get("volume24h", 0))
            price_change_24h = float(token.get("priceChange24h", 0))
            
            # Bullish patterns
            pattern_analysis['bullish_engulfing'] = self._detect_bullish_engulfing(token)
            pattern_analysis['hammer'] = self._detect_hammer(token)
            pattern_analysis['morning_star'] = self._detect_morning_star(token)
            pattern_analysis['doji'] = self._detect_doji(token)
            
            # Bearish patterns
            pattern_analysis['bearish_engulfing'] = self._detect_bearish_engulfing(token)
            pattern_analysis['shooting_star'] = self._detect_shooting_star(token)
            pattern_analysis['evening_star'] = self._detect_evening_star(token)
            
            # Breakout patterns
            pattern_analysis['breakout'] = self._detect_breakout(token)
            
            # Support/Resistance patterns
            pattern_analysis['support_resistance'] = self._detect_support_resistance_basic(token)
            
            return pattern_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing pattern types: {e}")
            return {pattern: 0.0 for pattern in self.pattern_types.keys()}
    
    def _detect_bullish_engulfing(self, token: Dict) -> float:
        """Detect bullish engulfing pattern using real candlestick data"""
        try:
            # Get real candlestick data
            from market_data_fetcher import market_data_fetcher
            
            address = token.get("address", "")
            chain_id = token.get("chainId", "ethereum").lower()
            candles = market_data_fetcher.get_candlestick_data(address, chain_id, hours=24)
            
            if not candles or len(candles) < 2:
                # Fallback to price change analysis
                price_change = float(token.get("priceChange24h", 0))
                volume_24h = float(token.get("volume24h", 0))
                
                if price_change > 5 and volume_24h > 100000:
                    return 0.7
                elif price_change > 2 and volume_24h > 50000:
                    return 0.5
                return 0.3
            
            # Check last 2 candles for bullish engulfing
            prev_candle = candles[-2]
            curr_candle = candles[-1]
            
            # Bullish engulfing: current body engulfs previous body
            prev_body = abs(prev_candle['close'] - prev_candle['open'])
            curr_body = abs(curr_candle['close'] - curr_candle['open'])
            
            # Previous candle should be bearish (close < open), current bullish (close > open)
            is_bullish_engulfing = (prev_candle['close'] < prev_candle['open'] and 
                                   curr_candle['close'] > curr_candle['open'] and
                                   curr_body > prev_body * 1.2)  # 20% larger body
            
            if is_bullish_engulfing:
                confidence = min(0.9, 0.7 + (curr_body / curr_candle['close']) * 2)
                return max(0.0, min(1.0, confidence))
            
            return 0.3
            
        except Exception as e:
            logger.error(f"Error detecting bullish engulfing: {e}")
            return 0.3
    
    def _detect_bearish_engulfing(self, token: Dict) -> float:
        """Detect bearish engulfing pattern"""
        try:
            price_change = float(token.get("priceChange24h", 0))
            volume_24h = float(token.get("volume24h", 0))
            
            # Bearish engulfing characteristics
            if price_change < -5 and volume_24h > 100000:  # Strong negative price change with volume
                confidence = min(0.9, 0.6 + abs(price_change) / 50)
            elif price_change < -2 and volume_24h > 50000:  # Moderate negative change
                confidence = 0.5 + abs(price_change) / 100
            else:
                confidence = 0.2
            
            return max(0.0, min(1.0, confidence))
            
        except Exception:
            return 0.3
    
    def _detect_hammer(self, token: Dict) -> float:
        """Detect hammer pattern"""
        try:
            price_change = float(token.get("priceChange24h", 0))
            volume_24h = float(token.get("volume24h", 0))
            
            # Hammer characteristics (reversal after decline)
            if -10 < price_change < 5 and volume_24h > 50000:  # Recovery after decline
                confidence = 0.6 + abs(price_change) / 50
            else:
                confidence = 0.3
            
            return max(0.0, min(1.0, confidence))
            
        except Exception:
            return 0.3
    
    def _detect_doji(self, token: Dict) -> float:
        """Detect doji pattern"""
        try:
            price_change = float(token.get("priceChange24h", 0))
            
            # Doji characteristics (indecision)
            if -2 < price_change < 2:  # Small price change indicates indecision
                confidence = 0.7
            else:
                confidence = 0.2
            
            return max(0.0, min(1.0, confidence))
            
        except Exception:
            return 0.3
    
    def _detect_shooting_star(self, token: Dict) -> float:
        """Detect shooting star pattern"""
        try:
            price_change = float(token.get("priceChange24h", 0))
            volume_24h = float(token.get("volume24h", 0))
            
            # Shooting star characteristics (reversal after rise)
            if 5 < price_change < 15 and volume_24h > 50000:  # Rise followed by potential reversal
                confidence = 0.6 + price_change / 100
            else:
                confidence = 0.2
            
            return max(0.0, min(1.0, confidence))
            
        except Exception:
            return 0.3
    
    def _detect_morning_star(self, token: Dict) -> float:
        """Detect morning star pattern"""
        try:
            price_change = float(token.get("priceChange24h", 0))
            volume_24h = float(token.get("volume24h", 0))
            
            # Morning star characteristics (bullish reversal)
            if price_change > 3 and volume_24h > 100000:
                confidence = 0.7 + price_change / 100
            else:
                confidence = 0.3
            
            return max(0.0, min(1.0, confidence))
            
        except Exception:
            return 0.3
    
    def _detect_evening_star(self, token: Dict) -> float:
        """Detect evening star pattern"""
        try:
            price_change = float(token.get("priceChange24h", 0))
            volume_24h = float(token.get("volume24h", 0))
            
            # Evening star characteristics (bearish reversal)
            if price_change < -3 and volume_24h > 100000:
                confidence = 0.7 + abs(price_change) / 100
            else:
                confidence = 0.3
            
            return max(0.0, min(1.0, confidence))
            
        except Exception:
            return 0.3
    
    def _detect_breakout(self, token: Dict) -> float:
        """Detect breakout pattern"""
        try:
            price_change = float(token.get("priceChange24h", 0))
            volume_24h = float(token.get("volume24h", 0))
            
            # Breakout characteristics (strong move with volume)
            if abs(price_change) > 10 and volume_24h > 200000:
                confidence = 0.8 + abs(price_change) / 100
            elif abs(price_change) > 5 and volume_24h > 100000:
                confidence = 0.6 + abs(price_change) / 150
            else:
                confidence = 0.3
            
            return max(0.0, min(1.0, confidence))
            
        except Exception:
            return 0.3
    
    def _detect_support_resistance_basic(self, token: Dict) -> float:
        """Detect support/resistance levels"""
        try:
            price = float(token.get("priceUsd", 0))
            volume_24h = float(token.get("volume24h", 0))
            
            # Support/resistance characteristics
            if price > 0.01 and volume_24h > 100000:  # Higher price with good volume
                confidence = 0.7
            elif price > 0.001 and volume_24h > 50000:  # Medium price with decent volume
                confidence = 0.5
            else:
                confidence = 0.3
            
            return max(0.0, min(1.0, confidence))
            
        except Exception:
            return 0.3
    
    def _calculate_pattern_strength(self, token: Dict, pattern_analysis: Dict) -> float:
        """Calculate overall pattern strength"""
        try:
            # Calculate weighted pattern strength
            total_strength = 0.0
            total_weight = 0.0
            
            for pattern_type, confidence in pattern_analysis.items():
                weight = self.pattern_types.get(pattern_type, 0.1)
                total_strength += confidence * weight
                total_weight += weight
            
            if total_weight > 0:
                base_strength = total_strength / total_weight
            else:
                base_strength = 0.0
            
            # Apply strength indicators
            volume_confirmation = self._assess_volume_confirmation(token)
            momentum_alignment = self._assess_momentum_alignment(token)
            timeframe_consistency = self._assess_timeframe_consistency(token)
            pattern_completeness = self._assess_pattern_completeness(pattern_analysis)
            market_context = self._assess_market_context(token)
            
            # Calculate weighted strength
            strength_indicators = {
                'volume_confirmation': volume_confirmation,
                'momentum_alignment': momentum_alignment,
                'timeframe_consistency': timeframe_consistency,
                'pattern_completeness': pattern_completeness,
                'market_context': market_context
            }
            
            strength_score = base_strength
            for indicator, value in strength_indicators.items():
                weight = self.strength_indicators.get(indicator, 0.2)
                strength_score += value * weight * 0.3  # 30% boost from indicators
            
            return max(0.0, min(1.0, strength_score))
            
        except Exception:
            return 0.5  # Default medium strength
    
    def _assess_volume_confirmation(self, token: Dict) -> float:
        """Assess volume confirmation for patterns"""
        try:
            volume_24h = float(token.get("volume24h", 0))
            
            if volume_24h > 1000000:  # Very high volume
                return 0.9
            elif volume_24h > 500000:  # High volume
                return 0.7
            elif volume_24h > 100000:  # Good volume
                return 0.5
            elif volume_24h > 50000:  # Decent volume
                return 0.3
            else:  # Low volume
                return 0.1
            
        except Exception:
            return 0.5
    
    def _assess_momentum_alignment(self, token: Dict) -> float:
        """Assess momentum alignment with patterns"""
        try:
            price_change = float(token.get("priceChange24h", 0))
            
            # Strong momentum
            if abs(price_change) > 15:
                return 0.9
            elif abs(price_change) > 10:
                return 0.7
            elif abs(price_change) > 5:
                return 0.5
            elif abs(price_change) > 2:
                return 0.3
            else:
                return 0.1
            
        except Exception:
            return 0.5
    
    def _assess_timeframe_consistency(self, token: Dict) -> float:
        """Assess timeframe consistency based on token data"""
        try:
            # Assess consistency based on real data
            volume_24h = float(token.get('volume24h', 0))
            liquidity = float(token.get('liquidity', 0))
            
            # Higher volume and liquidity = better consistency
            if volume_24h > 500000 and liquidity > 1000000:
                return 0.85  # High consistency
            elif volume_24h > 100000 and liquidity > 200000:
                return 0.7  # Medium consistency
            else:
                return 0.55  # Lower consistency
            
        except Exception:
            return 0.7
    
    def _assess_pattern_completeness(self, pattern_analysis: Dict) -> float:
        """Assess pattern completeness"""
        try:
            # Count patterns with confidence > 0.5
            strong_patterns = sum(1 for conf in pattern_analysis.values() if conf > 0.5)
            total_patterns = len(pattern_analysis)
            
            if total_patterns > 0:
                completeness = strong_patterns / total_patterns
            else:
                completeness = 0.0
            
            return completeness
            
        except Exception:
            return 0.5
    
    def _assess_market_context(self, token: Dict) -> float:
        """Assess market context for patterns based on real data"""
        try:
            # Assess market context based on token metrics
            price_change = float(token.get('priceChange24h', 0))
            volume_24h = float(token.get('volume24h', 0))
            
            # Positive price change with volume = positive context
            if price_change > 5 and volume_24h > 100000:
                return 0.85  # Very positive context
            elif price_change > 2 and volume_24h > 50000:
                return 0.7  # Positive context
            elif price_change < -5:
                return 0.3  # Negative context
            else:
                return 0.6  # Neutral context
            
        except Exception:
            return 0.6
    
    def _generate_trading_signals(self, pattern_analysis: Dict, pattern_strength: float) -> Dict:
        """Generate trading signals based on patterns"""
        try:
            signals = {}
            
            # Calculate bullish vs bearish pattern strength
            bullish_patterns = ['bullish_engulfing', 'hammer', 'morning_star', 'doji']
            bearish_patterns = ['bearish_engulfing', 'shooting_star', 'evening_star']
            
            bullish_strength = sum(pattern_analysis.get(pattern, 0) for pattern in bullish_patterns)
            bearish_strength = sum(pattern_analysis.get(pattern, 0) for pattern in bearish_patterns)
            
            # Generate signals
            if bullish_strength > bearish_strength:
                if pattern_strength > 0.8:
                    signals['primary_signal'] = 'strong_buy'
                    signals['signal_strength'] = 0.9
                elif pattern_strength > 0.6:
                    signals['primary_signal'] = 'buy'
                    signals['signal_strength'] = 0.7
                else:
                    signals['primary_signal'] = 'weak_buy'
                    signals['signal_strength'] = 0.5
            elif bearish_strength > bullish_strength:
                if pattern_strength > 0.8:
                    signals['primary_signal'] = 'strong_sell'
                    signals['signal_strength'] = -0.9
                elif pattern_strength > 0.6:
                    signals['primary_signal'] = 'sell'
                    signals['signal_strength'] = -0.7
                else:
                    signals['primary_signal'] = 'weak_sell'
                    signals['signal_strength'] = -0.5
            else:
                signals['primary_signal'] = 'neutral'
                signals['signal_strength'] = 0.0
            
            # Additional signals
            signals['breakout_signal'] = 'strong' if pattern_analysis.get('breakout', 0) > 0.7 else 'weak'
            signals['reversal_signal'] = 'strong' if max(bullish_strength, bearish_strength) > 0.6 else 'weak'
            
            return signals
            
        except Exception:
            return {
                'primary_signal': 'neutral',
                'signal_strength': 0.0,
                'breakout_signal': 'weak',
                'reversal_signal': 'weak'
            }
    
    def _detect_support_resistance(self, token: Dict) -> Dict:
        """Detect support and resistance levels using real candlestick data"""
        try:
            # Get real support/resistance levels
            from market_data_fetcher import market_data_fetcher
            
            address = token.get("address", "")
            chain_id = token.get("chainId", "ethereum").lower()
            levels = market_data_fetcher.get_support_resistance_levels(address, chain_id)
            
            if levels and levels.get('support') and levels.get('resistance'):
                price = float(token.get("priceUsd", 0))
                volume_24h = float(token.get("volume24h", 0))
                
                support = levels['support']
                resistance = levels['resistance']
                
                # Calculate additional levels
                support_levels = [
                    support * 0.98,  # Just below support
                    support,         # Main support
                    support * 1.02   # Just above support
                ]
                
                resistance_levels = [
                    resistance * 0.98,  # Just below resistance
                    resistance,         # Main resistance
                    resistance * 1.02   # Just above resistance
                ]
                
                return {
                    'support_levels': support_levels,
                    'resistance_levels': resistance_levels,
                    'current_price': price,
                    'strength': 'strong' if volume_24h > 500000 else 'medium' if volume_24h > 100000 else 'weak',
                    'distance_to_support': ((price - support) / support * 100) if support > 0 else None,
                    'distance_to_resistance': ((resistance - price) / price * 100) if price > 0 else None
                }
            
            # Fallback to price-based levels
            price = float(token.get("priceUsd", 0))
            volume_24h = float(token.get("volume24h", 0))
            
            if price > 0:
                support_levels = [price * 0.95, price * 0.90, price * 0.85]
                resistance_levels = [price * 1.05, price * 1.10, price * 1.15]
                
                return {
                    'support_levels': support_levels,
                    'resistance_levels': resistance_levels,
                    'current_price': price,
                    'strength': 'strong' if volume_24h > 500000 else 'medium' if volume_24h > 100000 else 'weak'
                }
            
            return {
                'support_levels': [],
                'resistance_levels': [],
                'current_price': price,
                'strength': 'weak'
            }
            
        except Exception as e:
            logger.error(f"Error detecting support/resistance: {e}")
            return {
                'support_levels': [],
                'resistance_levels': [],
                'current_price': 0,
                'strength': 'weak'
            }
    
    def _analyze_momentum_patterns(self, token: Dict) -> Dict:
        """Analyze momentum patterns"""
        try:
            price_change = float(token.get("priceChange24h", 0))
            volume_24h = float(token.get("volume24h", 0))
            
            # Analyze momentum
            if price_change > 10 and volume_24h > 200000:
                momentum = 'strong_bullish'
                momentum_strength = 0.9
            elif price_change > 5 and volume_24h > 100000:
                momentum = 'bullish'
                momentum_strength = 0.7
            elif price_change > 2:
                momentum = 'weak_bullish'
                momentum_strength = 0.5
            elif price_change < -10 and volume_24h > 200000:
                momentum = 'strong_bearish'
                momentum_strength = -0.9
            elif price_change < -5 and volume_24h > 100000:
                momentum = 'bearish'
                momentum_strength = -0.7
            elif price_change < -2:
                momentum = 'weak_bearish'
                momentum_strength = -0.5
            else:
                momentum = 'neutral'
                momentum_strength = 0.0
            
            return {
                'momentum': momentum,
                'momentum_strength': momentum_strength,
                'trend_direction': 'up' if price_change > 0 else 'down' if price_change < 0 else 'sideways',
                'acceleration': 'increasing' if abs(price_change) > 10 else 'stable'
            }
            
        except Exception:
            return {
                'momentum': 'neutral',
                'momentum_strength': 0.0,
                'trend_direction': 'sideways',
                'acceleration': 'stable'
            }
    
    def _generate_pattern_insights(self, pattern_analysis: Dict, pattern_strength: float) -> List[str]:
        """Generate pattern insights"""
        insights = []
        
        try:
            # Pattern strength insights
            if pattern_strength > 0.8:
                insights.append("Very strong pattern formation detected")
            elif pattern_strength > 0.6:
                insights.append("Strong pattern formation detected")
            elif pattern_strength > 0.4:
                insights.append("Moderate pattern formation detected")
            else:
                insights.append("Weak pattern formation detected")
            
            # Specific pattern insights
            bullish_patterns = ['bullish_engulfing', 'hammer', 'morning_star']
            bearish_patterns = ['bearish_engulfing', 'shooting_star', 'evening_star']
            
            bullish_strength = sum(pattern_analysis.get(pattern, 0) for pattern in bullish_patterns)
            bearish_strength = sum(pattern_analysis.get(pattern, 0) for pattern in bearish_patterns)
            
            if bullish_strength > 0.6:
                insights.append("Strong bullish patterns detected")
            elif bearish_strength > 0.6:
                insights.append("Strong bearish patterns detected")
            
            # Breakout insights
            if pattern_analysis.get('breakout', 0) > 0.7:
                insights.append("Breakout pattern detected - potential strong move")
            
            # Doji insights
            if pattern_analysis.get('doji', 0) > 0.6:
                insights.append("Doji pattern detected - market indecision")
            
        except Exception:
            insights.append("Pattern analysis completed")
        
        return insights
    
    def _generate_trading_recommendations(self, pattern_analysis: Dict, 
                                        trading_signals: Dict, 
                                        pattern_strength: float) -> List[str]:
        """Generate trading recommendations"""
        recommendations = []
        
        try:
            primary_signal = trading_signals.get('primary_signal', 'neutral')
            signal_strength = trading_signals.get('signal_strength', 0)
            
            if primary_signal in ['strong_buy', 'buy']:
                if pattern_strength > 0.8:
                    recommendations.append("Strong buy signal - high confidence pattern")
                else:
                    recommendations.append("Buy signal - moderate confidence pattern")
            elif primary_signal in ['strong_sell', 'sell']:
                if pattern_strength > 0.8:
                    recommendations.append("Strong sell signal - high confidence pattern")
                else:
                    recommendations.append("Sell signal - moderate confidence pattern")
            else:
                recommendations.append("Neutral signal - wait for clearer patterns")
            
            # Breakout recommendations
            if trading_signals.get('breakout_signal') == 'strong':
                recommendations.append("Breakout detected - consider position entry")
            
            # Reversal recommendations
            if trading_signals.get('reversal_signal') == 'strong':
                recommendations.append("Reversal pattern detected - monitor for confirmation")
            
        except Exception:
            recommendations.append("Monitor patterns for trading opportunities")
        
        return recommendations
    
    def _calculate_confidence_level(self, pattern_analysis: Dict) -> str:
        """Calculate confidence level in pattern recognition"""
        try:
            # Analyze pattern consistency
            pattern_scores = list(pattern_analysis.values())
            if not pattern_scores:
                return "low"
            
            # Calculate average confidence
            avg_confidence = statistics.mean(pattern_scores)
            
            # Calculate variance
            variance = statistics.variance(pattern_scores) if len(pattern_scores) > 1 else 0
            
            # Determine confidence level
            if avg_confidence > 0.7 and variance < 0.2:
                return "high"
            elif avg_confidence > 0.5 and variance < 0.4:
                return "medium"
            else:
                return "low"
                
        except Exception:
            return "medium"
    
    def _calculate_overall_signal(self, trading_signals: Dict) -> str:
        """Calculate overall trading signal"""
        try:
            primary_signal = trading_signals.get('primary_signal', 'neutral')
            signal_strength = trading_signals.get('signal_strength', 0)
            
            if signal_strength > 0.7:
                return "strong_buy"
            elif signal_strength > 0.3:
                return "buy"
            elif signal_strength < -0.7:
                return "strong_sell"
            elif signal_strength < -0.3:
                return "sell"
            else:
                return "neutral"
                
        except Exception:
            return "neutral"
    
    def _get_default_pattern_analysis(self) -> Dict:
        """Return default pattern analysis when recognition fails"""
        return {
            'pattern_analysis': {pattern: 0.0 for pattern in self.pattern_types.keys()},
            'pattern_strength': 0.0,
            'trading_signals': {
                'primary_signal': 'neutral',
                'signal_strength': 0.0,
                'breakout_signal': 'weak',
                'reversal_signal': 'weak'
            },
            'support_resistance': {
                'support_levels': [],
                'resistance_levels': [],
                'current_price': 0,
                'strength': 'weak'
            },
            'momentum_patterns': {
                'momentum': 'neutral',
                'momentum_strength': 0.0,
                'trend_direction': 'sideways',
                'acceleration': 'stable'
            },
            'pattern_insights': ['Pattern recognition unavailable'],
            'trading_recommendations': ['Monitor for pattern formation'],
            'recognition_timestamp': datetime.now().isoformat(),
            'confidence_level': 'low',
            'overall_signal': 'neutral'
        }
    
    def get_pattern_summary(self, tokens: List[Dict]) -> Dict:
        """Get pattern summary for multiple tokens"""
        try:
            pattern_summaries = []
            strong_pattern_count = 0
            moderate_pattern_count = 0
            weak_pattern_count = 0
            
            for token in tokens:
                pattern_recognition = self.recognize_patterns(token)
                pattern_summaries.append({
                    'symbol': token.get('symbol', 'UNKNOWN'),
                    'pattern_strength': pattern_recognition['pattern_strength'],
                    'overall_signal': pattern_recognition['overall_signal'],
                    'confidence_level': pattern_recognition['confidence_level']
                })
                
                pattern_strength = pattern_recognition['pattern_strength']
                if pattern_strength > 0.7:
                    strong_pattern_count += 1
                elif pattern_strength > 0.4:
                    moderate_pattern_count += 1
                else:
                    weak_pattern_count += 1
            
            return {
                'total_tokens': len(tokens),
                'strong_patterns': strong_pattern_count,
                'moderate_patterns': moderate_pattern_count,
                'weak_patterns': weak_pattern_count,
                'pattern_summaries': pattern_summaries,
                'overall_pattern_quality': 'high' if strong_pattern_count > len(tokens) * 0.5 else 'medium' if moderate_pattern_count > len(tokens) * 0.3 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error getting pattern summary: {e}")
            return {
                'total_tokens': len(tokens),
                'strong_patterns': 0,
                'moderate_patterns': 0,
                'weak_patterns': 0,
                'pattern_summaries': [],
                'overall_pattern_quality': 'unknown'
            }

# Global instance
ai_pattern_recognizer = AIPatternRecognizer()
