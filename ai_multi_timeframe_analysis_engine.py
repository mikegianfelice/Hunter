#!/usr/bin/env python3
"""
AI-Powered Multi-Timeframe Analysis Engine for Sustainable Trading Bot
Analyzes multiple timeframes simultaneously for comprehensive market view and improved signal accuracy
"""

import os
import json
import time
import logging
import math
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import random

# Configure logging
logger = logging.getLogger(__name__)

class AIMultiTimeframeAnalysisEngine:
    def __init__(self):
        self.analysis_cache = {}
        self.cache_duration = 300  # 5 minutes cache for multi-timeframe analysis
        self.timeframe_history = deque(maxlen=1000)
        self.signal_history = deque(maxlen=500)
        self.confirmation_history = deque(maxlen=500)
        
        # Timeframe configuration
        self.timeframes = {
            '1m': {
                'name': '1 Minute',
                'weight': 0.10,  # 10% weight
                'characteristics': ['very_short_term', 'high_frequency', 'noise_prone'],
                'signal_strength': 'weak',
                'trend_reliability': 'low'
            },
            '5m': {
                'name': '5 Minutes',
                'weight': 0.15,  # 15% weight
                'characteristics': ['short_term', 'medium_frequency', 'moderate_noise'],
                'signal_strength': 'medium',
                'trend_reliability': 'medium'
            },
            '15m': {
                'name': '15 Minutes',
                'weight': 0.20,  # 20% weight
                'characteristics': ['short_term', 'low_frequency', 'low_noise'],
                'signal_strength': 'strong',
                'trend_reliability': 'high'
            },
            '1h': {
                'name': '1 Hour',
                'weight': 0.25,  # 25% weight
                'characteristics': ['medium_term', 'very_low_frequency', 'very_low_noise'],
                'signal_strength': 'very_strong',
                'trend_reliability': 'very_high'
            },
            '4h': {
                'name': '4 Hours',
                'weight': 0.20,  # 20% weight
                'characteristics': ['medium_term', 'ultra_low_frequency', 'ultra_low_noise'],
                'signal_strength': 'very_strong',
                'trend_reliability': 'very_high'
            },
            '1d': {
                'name': '1 Day',
                'weight': 0.10,  # 10% weight
                'characteristics': ['long_term', 'minimal_frequency', 'minimal_noise'],
                'signal_strength': 'extremely_strong',
                'trend_reliability': 'extremely_high'
            }
        }
        
        # Analysis factors (must sum to 1.0)
        self.analysis_factors = {
            'trend_analysis': 0.30,  # 30% weight for trend analysis
            'momentum_analysis': 0.25,  # 25% weight for momentum analysis
            'support_resistance': 0.20,  # 20% weight for support/resistance
            'volume_analysis': 0.15,  # 15% weight for volume analysis
            'volatility_analysis': 0.10  # 10% weight for volatility analysis
        }
        
        # Signal confirmation thresholds
        self.strong_confirmation_threshold = 0.8  # 80% strong confirmation
        self.medium_confirmation_threshold = 0.6  # 60% medium confirmation
        self.weak_confirmation_threshold = 0.4  # 40% weak confirmation
        
        # Trend strength thresholds
        self.very_strong_trend_threshold = 0.9  # 90% very strong trend
        self.strong_trend_threshold = 0.7  # 70% strong trend
        self.medium_trend_threshold = 0.5  # 50% medium trend
        self.weak_trend_threshold = 0.3  # 30% weak trend
        
        # Signal divergence thresholds
        self.major_divergence_threshold = 0.7  # 70% major divergence
        self.minor_divergence_threshold = 0.4  # 40% minor divergence
        
        # Timeframe alignment thresholds
        self.perfect_alignment_threshold = 0.9  # 90% perfect alignment
        self.good_alignment_threshold = 0.7  # 70% good alignment
        self.poor_alignment_threshold = 0.4  # 40% poor alignment
    
    def analyze_multi_timeframe(self, token: Dict, trade_amount: float, market_data: Dict = None) -> Dict:
        """
        Analyze multiple timeframes for comprehensive market view
        Returns multi-timeframe analysis with signal confirmation
        """
        # Provide default market_data if not provided
        if market_data is None:
            market_data = {'timestamp': datetime.now().isoformat()}
            
        try:
            symbol = token.get("symbol", "UNKNOWN")
            cache_key = f"multi_timeframe_{symbol}_{market_data.get('timestamp', 'current')}"
            
            # Check cache
            if cache_key in self.analysis_cache:
                cached_data = self.analysis_cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).total_seconds() < self.cache_duration:
                    logger.debug(f"Using cached multi-timeframe analysis for {symbol}")
                    return cached_data['analysis_data']
            
            # Analyze each timeframe
            timeframe_analyses = {}
            for timeframe, config in self.timeframes.items():
                timeframe_analyses[timeframe] = self._analyze_timeframe(
                    timeframe, config, token, market_data
                )
            
            # Calculate overall trend analysis
            trend_analysis = self._calculate_overall_trend_analysis(timeframe_analyses)
            
            # Calculate momentum analysis
            momentum_analysis = self._calculate_momentum_analysis(timeframe_analyses)
            
            # Calculate support/resistance analysis
            support_resistance_analysis = self._calculate_support_resistance_analysis(timeframe_analyses)
            
            # Calculate volume analysis
            volume_analysis = self._calculate_volume_analysis(timeframe_analyses)
            
            # Calculate volatility analysis
            volatility_analysis = self._calculate_volatility_analysis(timeframe_analyses)
            
            # Calculate signal confirmation
            signal_confirmation = self._calculate_signal_confirmation(timeframe_analyses)
            
            # Calculate timeframe alignment
            timeframe_alignment = self._calculate_timeframe_alignment(timeframe_analyses)
            
            # Detect signal divergences
            signal_divergences = self._detect_signal_divergences(timeframe_analyses)
            
            # Generate optimal timeframe recommendations
            optimal_timeframes = self._generate_optimal_timeframes(timeframe_analyses, signal_confirmation)
            
            # Calculate overall analysis score
            overall_score = self._calculate_overall_score(
                trend_analysis, momentum_analysis, support_resistance_analysis,
                volume_analysis, volatility_analysis, signal_confirmation
            )
            
            # Generate analysis insights
            analysis_insights = self._generate_analysis_insights(
                timeframe_analyses, trend_analysis, momentum_analysis,
                signal_confirmation, timeframe_alignment, signal_divergences
            )
            
            result = {
                'overall_score': overall_score,
                'trend_analysis': trend_analysis,
                'momentum_analysis': momentum_analysis,
                'support_resistance_analysis': support_resistance_analysis,
                'volume_analysis': volume_analysis,
                'volatility_analysis': volatility_analysis,
                'signal_confirmation': signal_confirmation,
                'timeframe_alignment': timeframe_alignment,
                'signal_divergences': signal_divergences,
                'optimal_timeframes': optimal_timeframes,
                'timeframe_analyses': timeframe_analyses,
                'analysis_insights': analysis_insights,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Cache the result
            self.analysis_cache[cache_key] = {'timestamp': datetime.now(), 'analysis_data': result}
            
            logger.info(f"📊 Multi-timeframe analysis for {symbol}: Score {overall_score:.2f}, Confirmation {signal_confirmation['confirmation_level']}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Multi-timeframe analysis failed for {token.get('symbol', 'UNKNOWN')}: {e}")
            return self._get_default_multi_timeframe_analysis(token, market_data)
    
    def _analyze_timeframe(self, timeframe: str, config: Dict, token: Dict, market_data: Dict) -> Dict:
        """Analyze a specific timeframe"""
        try:
            # Extract timeframe-specific data
            timeframe_data = market_data.get(f'{timeframe}_data', {})
            
            # Analyze trend
            trend_analysis = self._analyze_timeframe_trend(timeframe, timeframe_data)
            
            # Analyze momentum
            momentum_analysis = self._analyze_timeframe_momentum(timeframe, timeframe_data)
            
            # Analyze support/resistance
            support_resistance = self._analyze_timeframe_support_resistance(timeframe, timeframe_data)
            
            # Analyze volume
            volume_analysis = self._analyze_timeframe_volume(timeframe, timeframe_data)
            
            # Analyze volatility
            volatility_analysis = self._analyze_timeframe_volatility(timeframe, timeframe_data)
            
            # Calculate timeframe score
            timeframe_score = self._calculate_timeframe_score(
                trend_analysis, momentum_analysis, support_resistance,
                volume_analysis, volatility_analysis
            )
            
            # Determine signal strength
            signal_strength = self._determine_signal_strength(timeframe_score, config)
            
            return {
                'timeframe': timeframe,
                'timeframe_score': timeframe_score,
                'signal_strength': signal_strength,
                'trend_analysis': trend_analysis,
                'momentum_analysis': momentum_analysis,
                'support_resistance': support_resistance,
                'volume_analysis': volume_analysis,
                'volatility_analysis': volatility_analysis
            }
            
        except Exception:
            return {
                'timeframe': timeframe,
                'timeframe_score': 0.5,
                'signal_strength': 'medium',
                'trend_analysis': {'trend_direction': 'neutral', 'trend_strength': 0.5},
                'momentum_analysis': {'momentum_direction': 'neutral', 'momentum_strength': 0.5},
                'support_resistance': {'support_level': 0.5, 'resistance_level': 0.5},
                'volume_analysis': {'volume_trend': 'stable', 'volume_strength': 0.5},
                'volatility_analysis': {'volatility_level': 'moderate', 'volatility_trend': 'stable'}
            }
    
    def _analyze_timeframe_trend(self, timeframe: str, timeframe_data: Dict) -> Dict:
        """Analyze trend for a specific timeframe"""
        try:
            # Extract price data
            current_price = timeframe_data.get('current_price', 100)
            price_1_period_ago = timeframe_data.get('price_1_period_ago', 100)
            price_3_periods_ago = timeframe_data.get('price_3_periods_ago', 100)
            price_5_periods_ago = timeframe_data.get('price_5_periods_ago', 100)
            
            # Calculate price changes
            change_1_period = (current_price - price_1_period_ago) / price_1_period_ago if price_1_period_ago > 0 else 0
            change_3_periods = (current_price - price_3_periods_ago) / price_3_periods_ago if price_3_periods_ago > 0 else 0
            change_5_periods = (current_price - price_5_periods_ago) / price_5_periods_ago if price_5_periods_ago > 0 else 0
            
            # Calculate trend strength
            trend_strength = (
                abs(change_1_period) * 0.5 +
                abs(change_3_periods) * 0.3 +
                abs(change_5_periods) * 0.2
            )
            
            # Determine trend direction
            if change_1_period > 0.02:  # 2% increase
                trend_direction = "strong_uptrend"
            elif change_1_period > 0.005:  # 0.5% increase
                trend_direction = "uptrend"
            elif change_1_period > -0.005:  # 0.5% decrease
                trend_direction = "sideways"
            elif change_1_period > -0.02:  # 2% decrease
                trend_direction = "downtrend"
            else:
                trend_direction = "strong_downtrend"
            
            return {
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'change_1_period': change_1_period,
                'change_3_periods': change_3_periods,
                'change_5_periods': change_5_periods
            }
            
        except Exception:
            return {
                'trend_direction': 'sideways',
                'trend_strength': 0.5,
                'change_1_period': 0.0,
                'change_3_periods': 0.0,
                'change_5_periods': 0.0
            }
    
    def _analyze_timeframe_momentum(self, timeframe: str, timeframe_data: Dict) -> Dict:
        """Analyze momentum for a specific timeframe"""
        try:
            # Extract momentum data
            rsi = timeframe_data.get('rsi', 50)
            macd = timeframe_data.get('macd', 0)
            macd_signal = timeframe_data.get('macd_signal', 0)
            macd_histogram = timeframe_data.get('macd_histogram', 0)
            
            # Calculate momentum strength
            rsi_momentum = abs(rsi - 50) / 50  # Normalize to 0-1
            macd_momentum = abs(macd - macd_signal) / max(abs(macd), abs(macd_signal), 0.001)
            histogram_momentum = abs(macd_histogram) / max(abs(macd_histogram), 0.001)
            
            momentum_strength = (
                rsi_momentum * 0.4 +
                macd_momentum * 0.4 +
                histogram_momentum * 0.2
            )
            
            # Determine momentum direction
            if rsi > 70 and macd > macd_signal:
                momentum_direction = "strong_bullish"
            elif rsi > 60 and macd > macd_signal:
                momentum_direction = "bullish"
            elif rsi < 30 and macd < macd_signal:
                momentum_direction = "strong_bearish"
            elif rsi < 40 and macd < macd_signal:
                momentum_direction = "bearish"
            else:
                momentum_direction = "neutral"
            
            return {
                'momentum_direction': momentum_direction,
                'momentum_strength': momentum_strength,
                'rsi': rsi,
                'macd': macd,
                'macd_signal': macd_signal,
                'macd_histogram': macd_histogram
            }
            
        except Exception:
            return {
                'momentum_direction': 'neutral',
                'momentum_strength': 0.5,
                'rsi': 50,
                'macd': 0,
                'macd_signal': 0,
                'macd_histogram': 0
            }
    
    def _analyze_timeframe_support_resistance(self, timeframe: str, timeframe_data: Dict) -> Dict:
        """Analyze support/resistance for a specific timeframe"""
        try:
            # Extract price levels
            current_price = timeframe_data.get('current_price', 100)
            support_level = timeframe_data.get('support_level', 95)
            resistance_level = timeframe_data.get('resistance_level', 105)
            
            # Calculate distance to support/resistance
            support_distance = (current_price - support_level) / current_price if current_price > 0 else 0
            resistance_distance = (resistance_level - current_price) / current_price if current_price > 0 else 0
            
            # Calculate support/resistance strength
            support_strength = max(0.0, min(1.0, support_distance * 10))  # Normalize to 0-1
            resistance_strength = max(0.0, min(1.0, resistance_distance * 10))  # Normalize to 0-1
            
            # Determine support/resistance level
            if support_distance < 0.02:  # 2% from support
                level_type = "near_support"
            elif resistance_distance < 0.02:  # 2% from resistance
                level_type = "near_resistance"
            else:
                level_type = "between_levels"
            
            return {
                'support_level': support_level,
                'resistance_level': resistance_level,
                'support_distance': support_distance,
                'resistance_distance': resistance_distance,
                'support_strength': support_strength,
                'resistance_strength': resistance_strength,
                'level_type': level_type
            }
            
        except Exception:
            return {
                'support_level': 95,
                'resistance_level': 105,
                'support_distance': 0.05,
                'resistance_distance': 0.05,
                'support_strength': 0.5,
                'resistance_strength': 0.5,
                'level_type': 'between_levels'
            }
    
    def _analyze_timeframe_volume(self, timeframe: str, timeframe_data: Dict) -> Dict:
        """Analyze volume for a specific timeframe"""
        try:
            # Extract volume data
            current_volume = timeframe_data.get('current_volume', 1000000)
            avg_volume = timeframe_data.get('avg_volume', 1000000)
            volume_trend = timeframe_data.get('volume_trend', 0)
            
            # Calculate volume ratio
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Calculate volume strength
            volume_strength = min(1.0, volume_ratio / 3.0)  # Normalize to 0-1
            
            # Determine volume trend
            if volume_ratio > 2.0:  # 2x volume
                volume_trend_type = "very_high"
            elif volume_ratio > 1.5:  # 1.5x volume
                volume_trend_type = "high"
            elif volume_ratio > 1.0:  # 1x volume
                volume_trend_type = "normal"
            elif volume_ratio > 0.5:  # 0.5x volume
                volume_trend_type = "low"
            else:
                volume_trend_type = "very_low"
            
            return {
                'volume_trend': volume_trend_type,
                'volume_strength': volume_strength,
                'volume_ratio': volume_ratio,
                'current_volume': current_volume,
                'avg_volume': avg_volume
            }
            
        except Exception:
            return {
                'volume_trend': 'normal',
                'volume_strength': 0.5,
                'volume_ratio': 1.0,
                'current_volume': 1000000,
                'avg_volume': 1000000
            }
    
    def _analyze_timeframe_volatility(self, timeframe: str, timeframe_data: Dict) -> Dict:
        """Analyze volatility for a specific timeframe"""
        try:
            # Extract volatility data
            current_volatility = timeframe_data.get('volatility', 0.3)
            volatility_trend = timeframe_data.get('volatility_trend', 0)
            
            # Calculate volatility level
            if current_volatility > 0.6:  # 60% volatility
                volatility_level = "very_high"
            elif current_volatility > 0.4:  # 40% volatility
                volatility_level = "high"
            elif current_volatility > 0.2:  # 20% volatility
                volatility_level = "moderate"
            elif current_volatility > 0.1:  # 10% volatility
                volatility_level = "low"
            else:
                volatility_level = "very_low"
            
            # Determine volatility trend
            if volatility_trend > 0.1:  # 10% increase
                volatility_trend_type = "increasing"
            elif volatility_trend > -0.1:  # 10% decrease
                volatility_trend_type = "stable"
            else:
                volatility_trend_type = "decreasing"
            
            return {
                'volatility_level': volatility_level,
                'volatility_trend': volatility_trend_type,
                'current_volatility': current_volatility,
                'volatility_trend_value': volatility_trend
            }
            
        except Exception:
            return {
                'volatility_level': 'moderate',
                'volatility_trend': 'stable',
                'current_volatility': 0.3,
                'volatility_trend_value': 0.0
            }
    
    def _calculate_timeframe_score(self, trend_analysis: Dict, momentum_analysis: Dict,
                                 support_resistance: Dict, volume_analysis: Dict,
                                 volatility_analysis: Dict) -> float:
        """Calculate score for a specific timeframe"""
        try:
            # Weight the individual analysis scores
            timeframe_score = (
                trend_analysis.get('trend_strength', 0.5) * 0.3 +
                momentum_analysis.get('momentum_strength', 0.5) * 0.25 +
                (support_resistance.get('support_strength', 0.5) + support_resistance.get('resistance_strength', 0.5)) / 2 * 0.2 +
                volume_analysis.get('volume_strength', 0.5) * 0.15 +
                (1.0 - volatility_analysis.get('current_volatility', 0.3)) * 0.1  # Lower volatility is better
            )
            
            return max(0.0, min(1.0, timeframe_score))
            
        except Exception:
            return 0.5
    
    def _determine_signal_strength(self, timeframe_score: float, config: Dict) -> str:
        """Determine signal strength for a timeframe"""
        try:
            if timeframe_score > 0.8:
                return "very_strong"
            elif timeframe_score > 0.6:
                return "strong"
            elif timeframe_score > 0.4:
                return "medium"
            elif timeframe_score > 0.2:
                return "weak"
            else:
                return "very_weak"
                
        except Exception:
            return "medium"
    
    def _calculate_overall_trend_analysis(self, timeframe_analyses: Dict) -> Dict:
        """Calculate overall trend analysis across all timeframes"""
        try:
            trend_scores = []
            trend_directions = []
            
            for timeframe, analysis in timeframe_analyses.items():
                trend_analysis = analysis.get('trend_analysis', {})
                trend_scores.append(trend_analysis.get('trend_strength', 0.5))
                trend_directions.append(trend_analysis.get('trend_direction', 'sideways'))
            
            # Calculate weighted average trend strength
            weighted_trend_strength = 0
            total_weight = 0
            
            for timeframe, analysis in timeframe_analyses.items():
                weight = self.timeframes[timeframe]['weight']
                trend_strength = analysis.get('trend_analysis', {}).get('trend_strength', 0.5)
                weighted_trend_strength += trend_strength * weight
                total_weight += weight
            
            overall_trend_strength = weighted_trend_strength / total_weight if total_weight > 0 else 0.5
            
            # Determine overall trend direction
            bullish_count = sum(1 for direction in trend_directions if 'uptrend' in direction or 'bullish' in direction)
            bearish_count = sum(1 for direction in trend_directions if 'downtrend' in direction or 'bearish' in direction)
            
            if bullish_count > bearish_count:
                overall_trend_direction = "bullish"
            elif bearish_count > bullish_count:
                overall_trend_direction = "bearish"
            else:
                overall_trend_direction = "neutral"
            
            return {
                'overall_trend_direction': overall_trend_direction,
                'overall_trend_strength': overall_trend_strength,
                'bullish_timeframes': bullish_count,
                'bearish_timeframes': bearish_count,
                'neutral_timeframes': len(trend_directions) - bullish_count - bearish_count
            }
            
        except Exception:
            return {
                'overall_trend_direction': 'neutral',
                'overall_trend_strength': 0.5,
                'bullish_timeframes': 0,
                'bearish_timeframes': 0,
                'neutral_timeframes': 6
            }
    
    def _calculate_momentum_analysis(self, timeframe_analyses: Dict) -> Dict:
        """Calculate overall momentum analysis across all timeframes"""
        try:
            momentum_scores = []
            momentum_directions = []
            
            for timeframe, analysis in timeframe_analyses.items():
                momentum_analysis = analysis.get('momentum_analysis', {})
                momentum_scores.append(momentum_analysis.get('momentum_strength', 0.5))
                momentum_directions.append(momentum_analysis.get('momentum_direction', 'neutral'))
            
            # Calculate weighted average momentum strength
            weighted_momentum_strength = 0
            total_weight = 0
            
            for timeframe, analysis in timeframe_analyses.items():
                weight = self.timeframes[timeframe]['weight']
                momentum_strength = analysis.get('momentum_analysis', {}).get('momentum_strength', 0.5)
                weighted_momentum_strength += momentum_strength * weight
                total_weight += weight
            
            overall_momentum_strength = weighted_momentum_strength / total_weight if total_weight > 0 else 0.5
            
            # Determine overall momentum direction
            bullish_count = sum(1 for direction in momentum_directions if 'bullish' in direction)
            bearish_count = sum(1 for direction in momentum_directions if 'bearish' in direction)
            
            if bullish_count > bearish_count:
                overall_momentum_direction = "bullish"
            elif bearish_count > bullish_count:
                overall_momentum_direction = "bearish"
            else:
                overall_momentum_direction = "neutral"
            
            return {
                'overall_momentum_direction': overall_momentum_direction,
                'overall_momentum_strength': overall_momentum_strength,
                'bullish_timeframes': bullish_count,
                'bearish_timeframes': bearish_count,
                'neutral_timeframes': len(momentum_directions) - bullish_count - bearish_count
            }
            
        except Exception:
            return {
                'overall_momentum_direction': 'neutral',
                'overall_momentum_strength': 0.5,
                'bullish_timeframes': 0,
                'bearish_timeframes': 0,
                'neutral_timeframes': 6
            }
    
    def _calculate_support_resistance_analysis(self, timeframe_analyses: Dict) -> Dict:
        """Calculate overall support/resistance analysis across all timeframes"""
        try:
            support_levels = []
            resistance_levels = []
            support_strengths = []
            resistance_strengths = []
            
            for timeframe, analysis in timeframe_analyses.items():
                support_resistance = analysis.get('support_resistance', {})
                support_levels.append(support_resistance.get('support_level', 95))
                resistance_levels.append(support_resistance.get('resistance_level', 105))
                support_strengths.append(support_resistance.get('support_strength', 0.5))
                resistance_strengths.append(support_resistance.get('resistance_strength', 0.5))
            
            # Calculate average support/resistance levels
            avg_support_level = statistics.mean(support_levels)
            avg_resistance_level = statistics.mean(resistance_levels)
            avg_support_strength = statistics.mean(support_strengths)
            avg_resistance_strength = statistics.mean(resistance_strengths)
            
            return {
                'avg_support_level': avg_support_level,
                'avg_resistance_level': avg_resistance_level,
                'avg_support_strength': avg_support_strength,
                'avg_resistance_strength': avg_resistance_strength,
                'support_levels': support_levels,
                'resistance_levels': resistance_levels
            }
            
        except Exception:
            return {
                'avg_support_level': 95,
                'avg_resistance_level': 105,
                'avg_support_strength': 0.5,
                'avg_resistance_strength': 0.5,
                'support_levels': [95] * 6,
                'resistance_levels': [105] * 6
            }
    
    def _calculate_volume_analysis(self, timeframe_analyses: Dict) -> Dict:
        """Calculate overall volume analysis across all timeframes"""
        try:
            volume_scores = []
            volume_trends = []
            
            for timeframe, analysis in timeframe_analyses.items():
                volume_analysis = analysis.get('volume_analysis', {})
                volume_scores.append(volume_analysis.get('volume_strength', 0.5))
                volume_trends.append(volume_analysis.get('volume_trend', 'normal'))
            
            # Calculate weighted average volume strength
            weighted_volume_strength = 0
            total_weight = 0
            
            for timeframe, analysis in timeframe_analyses.items():
                weight = self.timeframes[timeframe]['weight']
                volume_strength = analysis.get('volume_analysis', {}).get('volume_strength', 0.5)
                weighted_volume_strength += volume_strength * weight
                total_weight += weight
            
            overall_volume_strength = weighted_volume_strength / total_weight if total_weight > 0 else 0.5
            
            # Determine overall volume trend
            high_volume_count = sum(1 for trend in volume_trends if 'high' in trend)
            low_volume_count = sum(1 for trend in volume_trends if 'low' in trend)
            
            if high_volume_count > low_volume_count:
                overall_volume_trend = "high"
            elif low_volume_count > high_volume_count:
                overall_volume_trend = "low"
            else:
                overall_volume_trend = "normal"
            
            return {
                'overall_volume_trend': overall_volume_trend,
                'overall_volume_strength': overall_volume_strength,
                'high_volume_timeframes': high_volume_count,
                'low_volume_timeframes': low_volume_count,
                'normal_volume_timeframes': len(volume_trends) - high_volume_count - low_volume_count
            }
            
        except Exception:
            return {
                'overall_volume_trend': 'normal',
                'overall_volume_strength': 0.5,
                'high_volume_timeframes': 0,
                'low_volume_timeframes': 0,
                'normal_volume_timeframes': 6
            }
    
    def _calculate_volatility_analysis(self, timeframe_analyses: Dict) -> Dict:
        """Calculate overall volatility analysis across all timeframes"""
        try:
            volatility_scores = []
            volatility_levels = []
            
            for timeframe, analysis in timeframe_analyses.items():
                volatility_analysis = analysis.get('volatility_analysis', {})
                volatility_scores.append(volatility_analysis.get('current_volatility', 0.3))
                volatility_levels.append(volatility_analysis.get('volatility_level', 'moderate'))
            
            # Calculate weighted average volatility
            weighted_volatility = 0
            total_weight = 0
            
            for timeframe, analysis in timeframe_analyses.items():
                weight = self.timeframes[timeframe]['weight']
                volatility = analysis.get('volatility_analysis', {}).get('current_volatility', 0.3)
                weighted_volatility += volatility * weight
                total_weight += weight
            
            overall_volatility = weighted_volatility / total_weight if total_weight > 0 else 0.3
            
            # Determine overall volatility level
            if overall_volatility > 0.6:
                overall_volatility_level = "very_high"
            elif overall_volatility > 0.4:
                overall_volatility_level = "high"
            elif overall_volatility > 0.2:
                overall_volatility_level = "moderate"
            elif overall_volatility > 0.1:
                overall_volatility_level = "low"
            else:
                overall_volatility_level = "very_low"
            
            return {
                'overall_volatility_level': overall_volatility_level,
                'overall_volatility': overall_volatility,
                'volatility_scores': volatility_scores,
                'volatility_levels': volatility_levels
            }
            
        except Exception:
            return {
                'overall_volatility_level': 'moderate',
                'overall_volatility': 0.3,
                'volatility_scores': [0.3] * 6,
                'volatility_levels': ['moderate'] * 6
            }
    
    def _calculate_signal_confirmation(self, timeframe_analyses: Dict) -> Dict:
        """Calculate signal confirmation across all timeframes"""
        try:
            # Count bullish and bearish signals
            bullish_signals = 0
            bearish_signals = 0
            neutral_signals = 0
            
            for timeframe, analysis in timeframe_analyses.items():
                trend_analysis = analysis.get('trend_analysis', {})
                momentum_analysis = analysis.get('momentum_analysis', {})
                
                trend_direction = trend_analysis.get('trend_direction', 'sideways')
                momentum_direction = momentum_analysis.get('momentum_direction', 'neutral')
                
                # Count signals
                if 'uptrend' in trend_direction or 'bullish' in momentum_direction:
                    bullish_signals += 1
                elif 'downtrend' in trend_direction or 'bearish' in momentum_direction:
                    bearish_signals += 1
                else:
                    neutral_signals += 1
            
            # Calculate confirmation percentage
            total_signals = bullish_signals + bearish_signals + neutral_signals
            if total_signals > 0:
                bullish_percentage = bullish_signals / total_signals
                bearish_percentage = bearish_signals / total_signals
                neutral_percentage = neutral_signals / total_signals
            else:
                bullish_percentage = 0.33
                bearish_percentage = 0.33
                neutral_percentage = 0.34
            
            # Determine confirmation level
            if bullish_percentage > 0.7:
                confirmation_direction = "bullish"
                confirmation_level = "strong"
            elif bearish_percentage > 0.7:
                confirmation_direction = "bearish"
                confirmation_level = "strong"
            elif bullish_percentage > 0.5:
                confirmation_direction = "bullish"
                confirmation_level = "medium"
            elif bearish_percentage > 0.5:
                confirmation_direction = "bearish"
                confirmation_level = "medium"
            else:
                confirmation_direction = "neutral"
                confirmation_level = "weak"
            
            return {
                'confirmation_direction': confirmation_direction,
                'confirmation_level': confirmation_level,
                'bullish_signals': bullish_signals,
                'bearish_signals': bearish_signals,
                'neutral_signals': neutral_signals,
                'bullish_percentage': bullish_percentage,
                'bearish_percentage': bearish_percentage,
                'neutral_percentage': neutral_percentage
            }
            
        except Exception:
            return {
                'confirmation_direction': 'neutral',
                'confirmation_level': 'weak',
                'bullish_signals': 2,
                'bearish_signals': 2,
                'neutral_signals': 2,
                'bullish_percentage': 0.33,
                'bearish_percentage': 0.33,
                'neutral_percentage': 0.34
            }
    
    def _calculate_timeframe_alignment(self, timeframe_analyses: Dict) -> Dict:
        """Calculate timeframe alignment"""
        try:
            # Get trend directions for each timeframe
            trend_directions = []
            for timeframe, analysis in timeframe_analyses.items():
                trend_analysis = analysis.get('trend_analysis', {})
                trend_directions.append(trend_analysis.get('trend_direction', 'sideways'))
            
            # Calculate alignment score
            bullish_count = sum(1 for direction in trend_directions if 'uptrend' in direction)
            bearish_count = sum(1 for direction in trend_directions if 'downtrend' in direction)
            total_count = len(trend_directions)
            
            # Calculate alignment percentage
            max_alignment = max(bullish_count, bearish_count)
            alignment_percentage = max_alignment / total_count if total_count > 0 else 0
            
            # Determine alignment level
            if alignment_percentage > 0.8:
                alignment_level = "excellent"
            elif alignment_percentage > 0.6:
                alignment_level = "good"
            elif alignment_percentage > 0.4:
                alignment_level = "fair"
            else:
                alignment_level = "poor"
            
            return {
                'alignment_level': alignment_level,
                'alignment_percentage': alignment_percentage,
                'bullish_alignment': bullish_count,
                'bearish_alignment': bearish_count,
                'total_timeframes': total_count
            }
            
        except Exception:
            return {
                'alignment_level': 'fair',
                'alignment_percentage': 0.5,
                'bullish_alignment': 3,
                'bearish_alignment': 3,
                'total_timeframes': 6
            }
    
    def _detect_signal_divergences(self, timeframe_analyses: Dict) -> List[Dict]:
        """Detect signal divergences between timeframes"""
        try:
            divergences = []
            
            # Compare short-term vs long-term signals
            short_term_timeframes = ['1m', '5m', '15m']
            long_term_timeframes = ['1h', '4h', '1d']
            
            # Get short-term and long-term signals
            short_term_signals = []
            long_term_signals = []
            
            for timeframe in short_term_timeframes:
                if timeframe in timeframe_analyses:
                    trend_analysis = timeframe_analyses[timeframe].get('trend_analysis', {})
                    short_term_signals.append(trend_analysis.get('trend_direction', 'sideways'))
            
            for timeframe in long_term_timeframes:
                if timeframe in timeframe_analyses:
                    trend_analysis = timeframe_analyses[timeframe].get('trend_analysis', {})
                    long_term_signals.append(trend_analysis.get('trend_direction', 'sideways'))
            
            # Detect divergences
            if short_term_signals and long_term_signals:
                short_bullish = sum(1 for signal in short_term_signals if 'uptrend' in signal)
                short_bearish = sum(1 for signal in short_term_signals if 'downtrend' in signal)
                long_bullish = sum(1 for signal in long_term_signals if 'uptrend' in signal)
                long_bearish = sum(1 for signal in long_term_signals if 'downtrend' in signal)
                
                # Check for bullish divergence (short-term bearish, long-term bullish)
                if short_bearish > short_bullish and long_bullish > long_bearish:
                    divergences.append({
                        'type': 'bullish_divergence',
                        'description': 'Short-term bearish, long-term bullish',
                        'strength': 'medium'
                    })
                
                # Check for bearish divergence (short-term bullish, long-term bearish)
                if short_bullish > short_bearish and long_bearish > long_bullish:
                    divergences.append({
                        'type': 'bearish_divergence',
                        'description': 'Short-term bullish, long-term bearish',
                        'strength': 'medium'
                    })
            
            return divergences
            
        except Exception:
            return []
    
    def _generate_optimal_timeframes(self, timeframe_analyses: Dict, signal_confirmation: Dict) -> List[Dict]:
        """Generate optimal timeframes for trading"""
        try:
            optimal_timeframes = []
            
            # Sort timeframes by score
            timeframe_scores = []
            for timeframe, analysis in timeframe_analyses.items():
                score = analysis.get('timeframe_score', 0.5)
                weight = self.timeframes[timeframe]['weight']
                weighted_score = score * weight
                timeframe_scores.append((timeframe, weighted_score, analysis))
            
            # Sort by weighted score
            timeframe_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Generate recommendations for top timeframes
            for timeframe, score, analysis in timeframe_scores[:3]:  # Top 3 timeframes
                signal_strength = analysis.get('signal_strength', 'medium')
                trend_analysis = analysis.get('trend_analysis', {})
                momentum_analysis = analysis.get('momentum_analysis', {})
                
                optimal_timeframes.append({
                    'timeframe': timeframe,
                    'score': score,
                    'signal_strength': signal_strength,
                    'trend_direction': trend_analysis.get('trend_direction', 'sideways'),
                    'momentum_direction': momentum_analysis.get('momentum_direction', 'neutral'),
                    'recommendation': 'optimal' if score > 0.7 else 'good' if score > 0.5 else 'fair'
                })
            
            return optimal_timeframes
            
        except Exception:
            return [{'timeframe': '15m', 'score': 0.5, 'signal_strength': 'medium', 'recommendation': 'fair'}]
    
    def _calculate_overall_score(self, trend_analysis: Dict, momentum_analysis: Dict,
                               support_resistance_analysis: Dict, volume_analysis: Dict,
                               volatility_analysis: Dict, signal_confirmation: Dict) -> float:
        """Calculate overall multi-timeframe analysis score"""
        try:
            # Weight the individual analysis scores
            overall_score = (
                trend_analysis.get('overall_trend_strength', 0.5) * self.analysis_factors['trend_analysis'] +
                momentum_analysis.get('overall_momentum_strength', 0.5) * self.analysis_factors['momentum_analysis'] +
                (support_resistance_analysis.get('avg_support_strength', 0.5) + support_resistance_analysis.get('avg_resistance_strength', 0.5)) / 2 * self.analysis_factors['support_resistance'] +
                volume_analysis.get('overall_volume_strength', 0.5) * self.analysis_factors['volume_analysis'] +
                (1.0 - volatility_analysis.get('overall_volatility', 0.3)) * self.analysis_factors['volatility_analysis']  # Lower volatility is better
            )
            
            return max(0.0, min(1.0, overall_score))
            
        except Exception:
            return 0.5
    
    def _generate_analysis_insights(self, timeframe_analyses: Dict, trend_analysis: Dict,
                                   momentum_analysis: Dict, signal_confirmation: Dict,
                                   timeframe_alignment: Dict, signal_divergences: List[Dict]) -> List[str]:
        """Generate analysis insights"""
        insights = []
        
        try:
            # Overall trend insights
            overall_trend_direction = trend_analysis.get('overall_trend_direction', 'neutral')
            overall_trend_strength = trend_analysis.get('overall_trend_strength', 0.5)
            insights.append(f"Overall trend: {overall_trend_direction} (strength: {overall_trend_strength:.2f})")
            
            # Momentum insights
            overall_momentum_direction = momentum_analysis.get('overall_momentum_direction', 'neutral')
            overall_momentum_strength = momentum_analysis.get('overall_momentum_strength', 0.5)
            insights.append(f"Overall momentum: {overall_momentum_direction} (strength: {overall_momentum_strength:.2f})")
            
            # Signal confirmation insights
            confirmation_direction = signal_confirmation.get('confirmation_direction', 'neutral')
            confirmation_level = signal_confirmation.get('confirmation_level', 'weak')
            insights.append(f"Signal confirmation: {confirmation_direction} ({confirmation_level})")
            
            # Timeframe alignment insights
            alignment_level = timeframe_alignment.get('alignment_level', 'fair')
            alignment_percentage = timeframe_alignment.get('alignment_percentage', 0.5)
            insights.append(f"Timeframe alignment: {alignment_level} ({alignment_percentage:.1%})")
            
            # Divergence insights
            if signal_divergences:
                insights.append(f"Signal divergences detected: {len(signal_divergences)}")
                for divergence in signal_divergences[:2]:  # Show top 2 divergences
                    insights.append(f"  • {divergence['description']}")
            else:
                insights.append("No signal divergences detected")
            
        except Exception:
            insights.append("Multi-timeframe analysis completed")
        
        return insights
    
    def _get_default_multi_timeframe_analysis(self, token: Dict, market_data: Dict) -> Dict:
        """Return default multi-timeframe analysis when analysis fails"""
        return {
            'overall_score': 0.5,
            'trend_analysis': {
                'overall_trend_direction': 'neutral',
                'overall_trend_strength': 0.5
            },
            'momentum_analysis': {
                'overall_momentum_direction': 'neutral',
                'overall_momentum_strength': 0.5
            },
            'support_resistance_analysis': {
                'avg_support_level': 95,
                'avg_resistance_level': 105
            },
            'volume_analysis': {
                'overall_volume_trend': 'normal',
                'overall_volume_strength': 0.5
            },
            'volatility_analysis': {
                'overall_volatility_level': 'moderate',
                'overall_volatility': 0.3
            },
            'signal_confirmation': {
                'confirmation_direction': 'neutral',
                'confirmation_level': 'weak'
            },
            'timeframe_alignment': {
                'alignment_level': 'fair',
                'alignment_percentage': 0.5
            },
            'signal_divergences': [],
            'optimal_timeframes': [{'timeframe': '15m', 'score': 0.5, 'recommendation': 'fair'}],
            'analysis_insights': ['Multi-timeframe analysis completed'],
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def get_multi_timeframe_summary(self, tokens: List[Dict], market_data_list: List[Dict]) -> Dict:
        """Get multi-timeframe summary for multiple tokens"""
        try:
            analysis_summaries = []
            excellent_scores = 0
            good_scores = 0
            fair_scores = 0
            poor_scores = 0
            
            for i, token in enumerate(tokens):
                market_data = market_data_list[i] if i < len(market_data_list) else {}
                analysis = self.analyze_multi_timeframe(token, 5.0, market_data)
                
                analysis_summaries.append({
                    'symbol': token.get('symbol', 'UNKNOWN'),
                    'overall_score': analysis['overall_score'],
                    'confirmation_level': analysis['signal_confirmation']['confirmation_level'],
                    'alignment_level': analysis['timeframe_alignment']['alignment_level']
                })
                
                overall_score = analysis['overall_score']
                if overall_score > 0.8:
                    excellent_scores += 1
                elif overall_score > 0.6:
                    good_scores += 1
                elif overall_score > 0.4:
                    fair_scores += 1
                else:
                    poor_scores += 1
            
            return {
                'total_tokens': len(tokens),
                'excellent_scores': excellent_scores,
                'good_scores': good_scores,
                'fair_scores': fair_scores,
                'poor_scores': poor_scores,
                'analysis_summaries': analysis_summaries,
                'overall_quality': 'excellent' if excellent_scores > len(tokens) * 0.5 else 'good' if good_scores > len(tokens) * 0.3 else 'fair'
            }
            
        except Exception as e:
            logger.error(f"Error getting multi-timeframe summary: {e}")
            return {
                'total_tokens': len(tokens),
                'excellent_scores': 0,
                'good_scores': 0,
                'fair_scores': 0,
                'poor_scores': 0,
                'analysis_summaries': [],
                'overall_quality': 'unknown'
            }

# Global instance
ai_multi_timeframe_analysis_engine = AIMultiTimeframeAnalysisEngine()
