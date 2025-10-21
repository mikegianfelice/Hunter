#!/usr/bin/env python3
"""
AI-Powered Market Regime Transition Detector for Sustainable Trading Bot
Detects market regime changes in real-time to prevent losses and capture opportunities
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIMarketRegimeTransitionDetector:
    def __init__(self):
        self.transition_cache = {}
        self.cache_duration = 60  # 1 minute cache for transition detection
        self.regime_history = deque(maxlen=1000)
        self.transition_history = deque(maxlen=500)
        self.early_warning_history = deque(maxlen=500)
        
        # Regime transition configuration
        self.regime_types = {
            'bull_market': {
                'name': 'Bull Market',
                'characteristics': ['rising_prices', 'high_volume', 'positive_sentiment', 'strong_momentum'],
                'transition_probability': 0.3,
                'stability_threshold': 0.7
            },
            'bear_market': {
                'name': 'Bear Market',
                'characteristics': ['falling_prices', 'high_volume', 'negative_sentiment', 'weak_momentum'],
                'transition_probability': 0.4,
                'stability_threshold': 0.6
            },
            'sideways_market': {
                'name': 'Sideways Market',
                'characteristics': ['stable_prices', 'moderate_volume', 'neutral_sentiment', 'weak_momentum'],
                'transition_probability': 0.5,
                'stability_threshold': 0.8
            },
            'volatile_market': {
                'name': 'Volatile Market',
                'characteristics': ['high_volatility', 'irregular_volume', 'mixed_sentiment', 'unpredictable_momentum'],
                'transition_probability': 0.6,
                'stability_threshold': 0.5
            },
            'recovery_market': {
                'name': 'Recovery Market',
                'characteristics': ['recovering_prices', 'increasing_volume', 'improving_sentiment', 'building_momentum'],
                'transition_probability': 0.4,
                'stability_threshold': 0.6
            }
        }
        
        # Transition detection weights (must sum to 1.0)
        self.transition_factors = {
            'price_momentum': 0.25,  # 25% weight for price momentum
            'volume_patterns': 0.20,  # 20% weight for volume patterns
            'sentiment_shifts': 0.20,  # 20% weight for sentiment shifts
            'volatility_changes': 0.15,  # 15% weight for volatility changes
            'correlation_breaks': 0.10,  # 10% weight for correlation breaks
            'news_impact': 0.10  # 10% weight for news impact
        }
        
        # Transition detection thresholds
        self.strong_transition_threshold = 0.8  # 80% strong transition signal
        self.medium_transition_threshold = 0.6  # 60% medium transition signal
        self.weak_transition_threshold = 0.4  # 40% weak transition signal
        self.early_warning_threshold = 0.3  # 30% early warning threshold
        
        # Regime stability thresholds
        self.high_stability_threshold = 0.8  # 80% high stability
        self.medium_stability_threshold = 0.6  # 60% medium stability
        self.low_stability_threshold = 0.4  # 40% low stability
        
        # Transition timing thresholds
        self.immediate_transition_threshold = 0.9  # 90% immediate transition
        self.near_term_transition_threshold = 0.7  # 70% near-term transition
        self.medium_term_transition_threshold = 0.5  # 50% medium-term transition
        
        # Early warning indicators
        self.early_warning_indicators = {
            'price_divergence': 0.15,  # 15% price divergence threshold
            'volume_anomaly': 2.0,  # 2x volume anomaly threshold
            'sentiment_shift': 0.3,  # 30% sentiment shift threshold
            'volatility_spike': 1.5,  # 1.5x volatility spike threshold
            'correlation_break': 0.4,  # 40% correlation break threshold
            'news_impact': 0.7  # 70% news impact threshold
        }
    
    def detect_regime_transition(self, token: Dict, trade_amount: float) -> Dict:
        """
        Detect potential market regime transitions
        Returns comprehensive transition analysis with early warning signals
        """
        try:
            symbol = token.get('symbol', 'UNKNOWN')
            cache_key = f"transition_{symbol}_{token.get('timestamp', 'current')}"
            
            # Check cache
            if cache_key in self.transition_cache:
                cached_data = self.transition_cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).total_seconds() < self.cache_duration:
                    logger.debug(f"Using cached transition analysis for {symbol}")
                    return cached_data['transition_data']
            
            # Create market data from token
            market_data = {
                'timestamp': datetime.now().isoformat(),
                'price': float(token.get('priceUsd', 0)),
                'volume': float(token.get('volume24h', 0)),
                'liquidity': float(token.get('liquidity', 0))
            }
            
            # Analyze transition components
            price_momentum_analysis = self._analyze_price_momentum(market_data)
            volume_pattern_analysis = self._analyze_volume_patterns(market_data)
            sentiment_shift_analysis = self._analyze_sentiment_shifts(market_data)
            volatility_change_analysis = self._analyze_volatility_changes(market_data)
            correlation_break_analysis = self._analyze_correlation_breaks(market_data)
            news_impact_analysis = self._analyze_news_impact(market_data)
            
            # Determine current regime from market data
            current_regime = self._determine_current_regime(market_data)
            
            # Calculate transition probability
            transition_probability = self._calculate_transition_probability(
                price_momentum_analysis, volume_pattern_analysis, sentiment_shift_analysis,
                volatility_change_analysis, correlation_break_analysis, news_impact_analysis
            )
            
            # Determine transition strength
            transition_strength = self._determine_transition_strength(transition_probability)
            
            # Predict most likely next regime
            next_regime_prediction = self._predict_next_regime(
                current_regime, transition_probability, market_data
            )
            
            # Calculate transition timing
            transition_timing = self._calculate_transition_timing(
                transition_probability, market_data
            )
            
            # Generate early warning signals
            early_warning_signals = self._generate_early_warning_signals(
                price_momentum_analysis, volume_pattern_analysis, sentiment_shift_analysis,
                volatility_change_analysis, correlation_break_analysis, news_impact_analysis
            )
            
            # Calculate regime stability
            regime_stability = self._calculate_regime_stability(
                current_regime, market_data, transition_probability
            )
            
            # Generate transition recommendations
            transition_recommendations = self._generate_transition_recommendations(
                current_regime, next_regime_prediction, transition_strength, transition_timing
            )
            
            # Generate transition insights
            transition_insights = self._generate_transition_insights(
                current_regime, next_regime_prediction, transition_strength,
                regime_stability, early_warning_signals
            )
            
            result = {
                'current_regime': current_regime,
                'predicted_regime': next_regime_prediction,
                'transition_probability': transition_probability,
                'transition_confidence': 'high' if transition_probability > 0.7 else 'medium' if transition_probability > 0.4 else 'low',
                'transition_strength': transition_strength,
                'transition_timing': transition_timing,
                'regime_stability': regime_stability,
                'early_warning_signals': early_warning_signals,
                'price_momentum_analysis': price_momentum_analysis,
                'volume_pattern_analysis': volume_pattern_analysis,
                'sentiment_shift_analysis': sentiment_shift_analysis,
                'volatility_change_analysis': volatility_change_analysis,
                'correlation_break_analysis': correlation_break_analysis,
                'news_impact_analysis': news_impact_analysis,
                'transition_recommendations': transition_recommendations,
                'transition_insights': transition_insights,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Cache the result
            self.transition_cache[cache_key] = {'timestamp': datetime.now(), 'transition_data': result}
            
            logger.info(f"🔄 Regime transition analysis: {current_regime} -> {next_regime_prediction} (strength: {transition_strength})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Regime transition detection failed: {e}")
            # Try to determine current regime for fallback, or use default
            try:
                current_regime = self._determine_current_regime(market_data)
            except:
                current_regime = "sideways_market"  # Default fallback regime
            return self._get_default_transition_analysis(current_regime)
    
    def _determine_current_regime(self, market_data: Dict) -> str:
        """Determine current market regime from market data"""
        try:
            # Simple regime determination based on price and volume
            price = market_data.get('price', 0)
            volume = market_data.get('volume', 0)
            
            if price > 0 and volume > 100000:  # High volume
                return 'bull_market'
            elif price > 0 and volume < 10000:  # Low volume
                return 'bear_market'
            else:
                return 'sideways_market'
        except Exception:
            return 'sideways_market'
    
    def _analyze_price_momentum(self, market_data: Dict) -> Dict:
        """Analyze price momentum for regime transition signals"""
        try:
            # Extract price data
            current_price = market_data.get('current_price', 100)
            price_24h_ago = market_data.get('price_24h_ago', 100)
            price_7d_ago = market_data.get('price_7d_ago', 100)
            price_30d_ago = market_data.get('price_30d_ago', 100)
            
            # Calculate price changes
            change_24h = (current_price - price_24h_ago) / price_24h_ago if price_24h_ago > 0 else 0
            change_7d = (current_price - price_7d_ago) / price_7d_ago if price_7d_ago > 0 else 0
            change_30d = (current_price - price_30d_ago) / price_30d_ago if price_30d_ago > 0 else 0
            
            # Calculate momentum indicators
            short_term_momentum = change_24h
            medium_term_momentum = change_7d
            long_term_momentum = change_30d
            
            # Calculate momentum divergence
            momentum_divergence = abs(short_term_momentum - medium_term_momentum)
            
            # Determine momentum characteristics
            if abs(short_term_momentum) > 0.1:  # 10% change
                momentum_characteristics = "strong"
                momentum_signal = "high_volatility"
            elif abs(short_term_momentum) > 0.05:  # 5% change
                momentum_characteristics = "moderate"
                momentum_signal = "medium_volatility"
            else:
                momentum_characteristics = "weak"
                momentum_signal = "low_volatility"
            
            # Calculate momentum score
            momentum_score = (
                abs(short_term_momentum) * 0.4 +
                abs(medium_term_momentum) * 0.3 +
                abs(long_term_momentum) * 0.3
            )
            
            return {
                'short_term_momentum': short_term_momentum,
                'medium_term_momentum': medium_term_momentum,
                'long_term_momentum': long_term_momentum,
                'momentum_divergence': momentum_divergence,
                'momentum_score': momentum_score,
                'momentum_characteristics': momentum_characteristics,
                'momentum_signal': momentum_signal
            }
            
        except Exception:
            return {
                'short_term_momentum': 0.0,
                'medium_term_momentum': 0.0,
                'long_term_momentum': 0.0,
                'momentum_divergence': 0.0,
                'momentum_score': 0.0,
                'momentum_characteristics': 'weak',
                'momentum_signal': 'low_volatility'
            }
    
    def _analyze_volume_patterns(self, market_data: Dict) -> Dict:
        """Analyze volume patterns for regime transition signals"""
        try:
            # Extract volume data
            current_volume = market_data.get('current_volume', 1000000)
            avg_volume_24h = market_data.get('avg_volume_24h', 1000000)
            avg_volume_7d = market_data.get('avg_volume_7d', 1000000)
            avg_volume_30d = market_data.get('avg_volume_30d', 1000000)
            
            # Calculate volume ratios
            volume_ratio_24h = current_volume / avg_volume_24h if avg_volume_24h > 0 else 1.0
            volume_ratio_7d = avg_volume_24h / avg_volume_7d if avg_volume_7d > 0 else 1.0
            volume_ratio_30d = avg_volume_7d / avg_volume_30d if avg_volume_30d > 0 else 1.0
            
            # Calculate volume momentum
            volume_momentum = (volume_ratio_24h - 1.0) * 0.5 + (volume_ratio_7d - 1.0) * 0.3 + (volume_ratio_30d - 1.0) * 0.2
            
            # Detect volume anomalies
            volume_anomaly = max(volume_ratio_24h, volume_ratio_7d, volume_ratio_30d)
            
            # Determine volume characteristics
            if volume_anomaly > 3.0:  # 3x volume spike
                volume_characteristics = "extreme"
                volume_signal = "high_anomaly"
            elif volume_anomaly > 2.0:  # 2x volume spike
                volume_characteristics = "high"
                volume_signal = "medium_anomaly"
            elif volume_anomaly > 1.5:  # 1.5x volume spike
                volume_characteristics = "moderate"
                volume_signal = "low_anomaly"
            else:
                volume_characteristics = "normal"
                volume_signal = "no_anomaly"
            
            # Calculate volume score
            volume_score = min(1.0, volume_anomaly / 5.0)  # Normalize to 0-1
            
            return {
                'volume_ratio_24h': volume_ratio_24h,
                'volume_ratio_7d': volume_ratio_7d,
                'volume_ratio_30d': volume_ratio_30d,
                'volume_momentum': volume_momentum,
                'volume_anomaly': volume_anomaly,
                'volume_score': volume_score,
                'volume_characteristics': volume_characteristics,
                'volume_signal': volume_signal
            }
            
        except Exception:
            return {
                'volume_ratio_24h': 1.0,
                'volume_ratio_7d': 1.0,
                'volume_ratio_30d': 1.0,
                'volume_momentum': 0.0,
                'volume_anomaly': 1.0,
                'volume_score': 0.2,
                'volume_characteristics': 'normal',
                'volume_signal': 'no_anomaly'
            }
    
    def _analyze_sentiment_shifts(self, market_data: Dict) -> Dict:
        """Analyze sentiment shifts for regime transition signals"""
        try:
            # Extract sentiment data
            current_sentiment = market_data.get('current_sentiment', 0.5)
            sentiment_24h_ago = market_data.get('sentiment_24h_ago', 0.5)
            sentiment_7d_ago = market_data.get('sentiment_7d_ago', 0.5)
            sentiment_30d_ago = market_data.get('sentiment_30d_ago', 0.5)
            
            # Calculate sentiment changes
            sentiment_change_24h = current_sentiment - sentiment_24h_ago
            sentiment_change_7d = current_sentiment - sentiment_7d_ago
            sentiment_change_30d = current_sentiment - sentiment_30d_ago
            
            # Calculate sentiment momentum
            sentiment_momentum = (
                sentiment_change_24h * 0.5 +
                sentiment_change_7d * 0.3 +
                sentiment_change_30d * 0.2
            )
            
            # Detect sentiment shifts
            sentiment_shift_magnitude = abs(sentiment_change_24h)
            
            # Determine sentiment characteristics
            if sentiment_shift_magnitude > 0.3:  # 30% sentiment shift
                sentiment_characteristics = "extreme"
                sentiment_signal = "major_shift"
            elif sentiment_shift_magnitude > 0.2:  # 20% sentiment shift
                sentiment_characteristics = "high"
                sentiment_signal = "significant_shift"
            elif sentiment_shift_magnitude > 0.1:  # 10% sentiment shift
                sentiment_characteristics = "moderate"
                sentiment_signal = "minor_shift"
            else:
                sentiment_characteristics = "stable"
                sentiment_signal = "no_shift"
            
            # Calculate sentiment score
            sentiment_score = min(1.0, sentiment_shift_magnitude * 2.0)  # Normalize to 0-1
            
            return {
                'sentiment_change_24h': sentiment_change_24h,
                'sentiment_change_7d': sentiment_change_7d,
                'sentiment_change_30d': sentiment_change_30d,
                'sentiment_momentum': sentiment_momentum,
                'sentiment_shift_magnitude': sentiment_shift_magnitude,
                'sentiment_score': sentiment_score,
                'sentiment_characteristics': sentiment_characteristics,
                'sentiment_signal': sentiment_signal
            }
            
        except Exception:
            return {
                'sentiment_change_24h': 0.0,
                'sentiment_change_7d': 0.0,
                'sentiment_change_30d': 0.0,
                'sentiment_momentum': 0.0,
                'sentiment_shift_magnitude': 0.0,
                'sentiment_score': 0.0,
                'sentiment_characteristics': 'stable',
                'sentiment_signal': 'no_shift'
            }
    
    def _analyze_volatility_changes(self, market_data: Dict) -> Dict:
        """Analyze volatility changes for regime transition signals"""
        try:
            # Extract volatility data
            current_volatility = market_data.get('current_volatility', 0.3)
            volatility_24h_ago = market_data.get('volatility_24h_ago', 0.3)
            volatility_7d_ago = market_data.get('volatility_7d_ago', 0.3)
            volatility_30d_ago = market_data.get('volatility_30d_ago', 0.3)
            
            # Calculate volatility changes
            volatility_change_24h = current_volatility - volatility_24h_ago
            volatility_change_7d = current_volatility - volatility_7d_ago
            volatility_change_30d = current_volatility - volatility_30d_ago
            
            # Calculate volatility momentum
            volatility_momentum = (
                volatility_change_24h * 0.5 +
                volatility_change_7d * 0.3 +
                volatility_change_30d * 0.2
            )
            
            # Detect volatility spikes
            volatility_spike = current_volatility / volatility_24h_ago if volatility_24h_ago > 0 else 1.0
            
            # Determine volatility characteristics
            if volatility_spike > 2.0:  # 2x volatility spike
                volatility_characteristics = "extreme"
                volatility_signal = "major_spike"
            elif volatility_spike > 1.5:  # 1.5x volatility spike
                volatility_characteristics = "high"
                volatility_signal = "significant_spike"
            elif volatility_spike > 1.2:  # 1.2x volatility spike
                volatility_characteristics = "moderate"
                volatility_signal = "minor_spike"
            else:
                volatility_characteristics = "stable"
                volatility_signal = "no_spike"
            
            # Calculate volatility score
            volatility_score = min(1.0, (volatility_spike - 1.0) * 2.0)  # Normalize to 0-1
            
            return {
                'volatility_change_24h': volatility_change_24h,
                'volatility_change_7d': volatility_change_7d,
                'volatility_change_30d': volatility_change_30d,
                'volatility_momentum': volatility_momentum,
                'volatility_spike': volatility_spike,
                'volatility_score': volatility_score,
                'volatility_characteristics': volatility_characteristics,
                'volatility_signal': volatility_signal
            }
            
        except Exception:
            return {
                'volatility_change_24h': 0.0,
                'volatility_change_7d': 0.0,
                'volatility_change_30d': 0.0,
                'volatility_momentum': 0.0,
                'volatility_spike': 1.0,
                'volatility_score': 0.0,
                'volatility_characteristics': 'stable',
                'volatility_signal': 'no_spike'
            }
    
    def _analyze_correlation_breaks(self, market_data: Dict) -> Dict:
        """Analyze correlation breaks for regime transition signals"""
        try:
            # Extract correlation data
            btc_correlation = market_data.get('btc_correlation', 0.5)
            eth_correlation = market_data.get('eth_correlation', 0.5)
            market_correlation = market_data.get('market_correlation', 0.5)
            
            # Calculate average correlation
            avg_correlation = (btc_correlation + eth_correlation + market_correlation) / 3
            
            # Calculate correlation breakdown
            correlation_breakdown = 1.0 - avg_correlation
            
            # Detect correlation breaks
            correlation_break_magnitude = abs(correlation_breakdown)
            
            # Determine correlation characteristics
            if correlation_break_magnitude > 0.5:  # 50% correlation breakdown
                correlation_characteristics = "extreme"
                correlation_signal = "major_break"
            elif correlation_break_magnitude > 0.3:  # 30% correlation breakdown
                correlation_characteristics = "high"
                correlation_signal = "significant_break"
            elif correlation_break_magnitude > 0.2:  # 20% correlation breakdown
                correlation_characteristics = "moderate"
                correlation_signal = "minor_break"
            else:
                correlation_characteristics = "stable"
                correlation_signal = "no_break"
            
            # Calculate correlation score
            correlation_score = min(1.0, correlation_break_magnitude * 2.0)  # Normalize to 0-1
            
            return {
                'btc_correlation': btc_correlation,
                'eth_correlation': eth_correlation,
                'market_correlation': market_correlation,
                'avg_correlation': avg_correlation,
                'correlation_breakdown': correlation_breakdown,
                'correlation_break_magnitude': correlation_break_magnitude,
                'correlation_score': correlation_score,
                'correlation_characteristics': correlation_characteristics,
                'correlation_signal': correlation_signal
            }
            
        except Exception:
            return {
                'btc_correlation': 0.5,
                'eth_correlation': 0.5,
                'market_correlation': 0.5,
                'avg_correlation': 0.5,
                'correlation_breakdown': 0.5,
                'correlation_break_magnitude': 0.5,
                'correlation_score': 0.5,
                'correlation_characteristics': 'moderate',
                'correlation_signal': 'minor_break'
            }
    
    def _analyze_news_impact(self, market_data: Dict) -> Dict:
        """Analyze news impact for regime transition signals"""
        try:
            # Extract news data
            news_sentiment = market_data.get('news_sentiment', 0.5)
            news_impact_score = market_data.get('news_impact_score', 0.5)
            breaking_news_count = market_data.get('breaking_news_count', 0)
            major_news_count = market_data.get('major_news_count', 0)
            
            # Calculate news impact
            news_impact = (
                news_sentiment * 0.3 +
                news_impact_score * 0.4 +
                (breaking_news_count / 10.0) * 0.2 +
                (major_news_count / 5.0) * 0.1
            )
            
            # Detect news impact magnitude
            news_impact_magnitude = abs(news_impact - 0.5) * 2.0  # Normalize to 0-1
            
            # Determine news characteristics
            if news_impact_magnitude > 0.8:  # 80% news impact
                news_characteristics = "extreme"
                news_signal = "major_impact"
            elif news_impact_magnitude > 0.6:  # 60% news impact
                news_characteristics = "high"
                news_signal = "significant_impact"
            elif news_impact_magnitude > 0.4:  # 40% news impact
                news_characteristics = "moderate"
                news_signal = "minor_impact"
            else:
                news_characteristics = "low"
                news_signal = "no_impact"
            
            return {
                'news_sentiment': news_sentiment,
                'news_impact_score': news_impact_score,
                'breaking_news_count': breaking_news_count,
                'major_news_count': major_news_count,
                'news_impact': news_impact,
                'news_impact_magnitude': news_impact_magnitude,
                'news_characteristics': news_characteristics,
                'news_signal': news_signal
            }
            
        except Exception:
            return {
                'news_sentiment': 0.5,
                'news_impact_score': 0.5,
                'breaking_news_count': 0,
                'major_news_count': 0,
                'news_impact': 0.5,
                'news_impact_magnitude': 0.0,
                'news_characteristics': 'low',
                'news_signal': 'no_impact'
            }
    
    def _calculate_transition_probability(self, price_momentum_analysis: Dict, volume_pattern_analysis: Dict,
                                        sentiment_shift_analysis: Dict, volatility_change_analysis: Dict,
                                        correlation_break_analysis: Dict, news_impact_analysis: Dict) -> float:
        """Calculate overall transition probability"""
        try:
            # Weight the individual analysis scores
            transition_probability = (
                price_momentum_analysis.get('momentum_score', 0.5) * self.transition_factors['price_momentum'] +
                volume_pattern_analysis.get('volume_score', 0.5) * self.transition_factors['volume_patterns'] +
                sentiment_shift_analysis.get('sentiment_score', 0.5) * self.transition_factors['sentiment_shifts'] +
                volatility_change_analysis.get('volatility_score', 0.5) * self.transition_factors['volatility_changes'] +
                correlation_break_analysis.get('correlation_score', 0.5) * self.transition_factors['correlation_breaks'] +
                news_impact_analysis.get('news_impact_magnitude', 0.5) * self.transition_factors['news_impact']
            )
            
            return max(0.0, min(1.0, transition_probability))
            
        except Exception:
            return 0.5
    
    def _determine_transition_strength(self, transition_probability: float) -> str:
        """Determine transition strength based on probability"""
        try:
            if transition_probability > self.strong_transition_threshold:
                return "strong"
            elif transition_probability > self.medium_transition_threshold:
                return "medium"
            elif transition_probability > self.weak_transition_threshold:
                return "weak"
            else:
                return "none"
                
        except Exception:
            return "none"
    
    def _predict_next_regime(self, current_regime: str, transition_probability: float, market_data: Dict) -> str:
        """Predict the most likely next regime"""
        try:
            # Get current regime characteristics
            current_regime_info = self.regime_types.get(current_regime, {})
            current_stability = current_regime_info.get('stability_threshold', 0.5)
            
            # If current regime is stable and transition probability is low, stay in current regime
            if transition_probability < 0.3 and current_stability > 0.7:
                return current_regime
            
            # Predict next regime based on market conditions
            if transition_probability > 0.7:
                # High transition probability - predict based on market conditions
                if market_data.get('current_sentiment', 0.5) > 0.7:
                    return 'bull_market'
                elif market_data.get('current_sentiment', 0.5) < 0.3:
                    return 'bear_market'
                elif market_data.get('current_volatility', 0.3) > 0.6:
                    return 'volatile_market'
                else:
                    return 'sideways_market'
            else:
                # Low transition probability - likely to stay in current regime
                return current_regime
                
        except Exception:
            return current_regime
    
    def _calculate_transition_timing(self, transition_probability: float, market_data: Dict) -> str:
        """Calculate when the transition is likely to occur"""
        try:
            if transition_probability > self.immediate_transition_threshold:
                return "immediate"
            elif transition_probability > self.near_term_transition_threshold:
                return "near_term"
            elif transition_probability > self.medium_term_transition_threshold:
                return "medium_term"
            else:
                return "long_term"
                
        except Exception:
            return "long_term"
    
    def _generate_early_warning_signals(self, price_momentum_analysis: Dict, volume_pattern_analysis: Dict,
                                      sentiment_shift_analysis: Dict, volatility_change_analysis: Dict,
                                      correlation_break_analysis: Dict, news_impact_analysis: Dict) -> List[str]:
        """Generate early warning signals for regime transition"""
        signals = []
        
        try:
            # Price momentum early warning
            if price_momentum_analysis.get('momentum_divergence', 0) > self.early_warning_indicators['price_divergence']:
                signals.append("Price momentum divergence detected")
            
            # Volume anomaly early warning
            if volume_pattern_analysis.get('volume_anomaly', 1) > self.early_warning_indicators['volume_anomaly']:
                signals.append("Volume anomaly detected")
            
            # Sentiment shift early warning
            if sentiment_shift_analysis.get('sentiment_shift_magnitude', 0) > self.early_warning_indicators['sentiment_shift']:
                signals.append("Sentiment shift detected")
            
            # Volatility spike early warning
            if volatility_change_analysis.get('volatility_spike', 1) > self.early_warning_indicators['volatility_spike']:
                signals.append("Volatility spike detected")
            
            # Correlation break early warning
            if correlation_break_analysis.get('correlation_break_magnitude', 0) > self.early_warning_indicators['correlation_break']:
                signals.append("Correlation breakdown detected")
            
            # News impact early warning
            if news_impact_analysis.get('news_impact_magnitude', 0) > self.early_warning_indicators['news_impact']:
                signals.append("High news impact detected")
            
        except Exception:
            signals.append("Monitoring regime transition indicators")
        
        return signals
    
    def _calculate_regime_stability(self, current_regime: str, market_data: Dict, transition_probability: float) -> float:
        """Calculate current regime stability"""
        try:
            # Get current regime stability threshold
            current_regime_info = self.regime_types.get(current_regime, {})
            base_stability = current_regime_info.get('stability_threshold', 0.5)
            
            # Adjust stability based on transition probability
            adjusted_stability = base_stability * (1.0 - transition_probability)
            
            return max(0.0, min(1.0, adjusted_stability))
            
        except Exception:
            return 0.5
    
    def _generate_transition_recommendations(self, current_regime: str, next_regime_prediction: str,
                                          transition_strength: str, transition_timing: str) -> List[str]:
        """Generate recommendations based on regime transition analysis"""
        recommendations = []
        
        try:
            # Transition strength recommendations
            if transition_strength == "strong":
                recommendations.append("Strong regime transition detected - prepare for major changes")
                recommendations.append("Consider reducing position sizes significantly")
                recommendations.append("Monitor for early exit signals")
            elif transition_strength == "medium":
                recommendations.append("Medium regime transition detected - monitor closely")
                recommendations.append("Consider conservative position sizing")
                recommendations.append("Prepare for potential regime change")
            elif transition_strength == "weak":
                recommendations.append("Weak regime transition detected - continue monitoring")
                recommendations.append("Maintain current strategy with caution")
            else:
                recommendations.append("No regime transition detected - continue current strategy")
            
            # Timing recommendations
            if transition_timing == "immediate":
                recommendations.append("Immediate regime transition expected - act quickly")
            elif transition_timing == "near_term":
                recommendations.append("Near-term regime transition expected - prepare within hours")
            elif transition_timing == "medium_term":
                recommendations.append("Medium-term regime transition expected - prepare within days")
            else:
                recommendations.append("Long-term regime transition expected - monitor over weeks")
            
            # Regime-specific recommendations
            if next_regime_prediction == "bull_market":
                recommendations.append("Bull market transition expected - consider aggressive strategies")
            elif next_regime_prediction == "bear_market":
                recommendations.append("Bear market transition expected - consider defensive strategies")
            elif next_regime_prediction == "volatile_market":
                recommendations.append("Volatile market transition expected - consider scalping strategies")
            elif next_regime_prediction == "sideways_market":
                recommendations.append("Sideways market transition expected - consider mean reversion strategies")
            
        except Exception:
            recommendations.append("Monitor regime transition indicators and adjust strategy accordingly")
        
        return recommendations
    
    def _generate_transition_insights(self, current_regime: str, next_regime_prediction: str,
                                    transition_strength: str, regime_stability: float,
                                    early_warning_signals: List[str]) -> List[str]:
        """Generate insights about regime transition"""
        insights = []
        
        try:
            # Current regime insights
            insights.append(f"Current regime: {current_regime}")
            insights.append(f"Regime stability: {regime_stability:.2f}")
            
            # Transition insights
            insights.append(f"Next regime prediction: {next_regime_prediction}")
            insights.append(f"Transition strength: {transition_strength}")
            
            # Early warning insights
            if early_warning_signals:
                insights.append(f"Early warning signals: {len(early_warning_signals)} detected")
                for signal in early_warning_signals[:3]:  # Show top 3 signals
                    insights.append(f"  • {signal}")
            else:
                insights.append("No early warning signals detected")
            
            # Regime transition insights
            if current_regime != next_regime_prediction:
                insights.append(f"Regime transition expected: {current_regime} -> {next_regime_prediction}")
            else:
                insights.append("No regime transition expected - current regime likely to continue")
            
        except Exception:
            insights.append("Regime transition analysis completed")
        
        return insights
    
    def _get_default_transition_analysis(self, current_regime: str) -> Dict:
        """Return default transition analysis when analysis fails"""
        return {
            'current_regime': current_regime,
            'next_regime_prediction': current_regime,
            'transition_probability': 0.3,
            'transition_strength': 'weak',
            'transition_timing': 'long_term',
            'regime_stability': 0.7,
            'early_warning_signals': ['Monitoring regime transition indicators'],
            'price_momentum_analysis': {'momentum_score': 0.3, 'momentum_signal': 'low_volatility'},
            'volume_pattern_analysis': {'volume_score': 0.3, 'volume_signal': 'no_anomaly'},
            'sentiment_shift_analysis': {'sentiment_score': 0.3, 'sentiment_signal': 'no_shift'},
            'volatility_change_analysis': {'volatility_score': 0.3, 'volatility_signal': 'no_spike'},
            'correlation_break_analysis': {'correlation_score': 0.3, 'correlation_signal': 'minor_break'},
            'news_impact_analysis': {'news_impact_magnitude': 0.3, 'news_signal': 'minor_impact'},
            'transition_recommendations': ['Monitor regime transition indicators'],
            'transition_insights': ['Regime transition analysis completed'],
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def get_transition_summary(self, market_data_list: List[Dict], current_regimes: List[str]) -> Dict:
        """Get transition summary for multiple market conditions"""
        try:
            transition_summaries = []
            strong_transitions = 0
            medium_transitions = 0
            weak_transitions = 0
            no_transitions = 0
            
            for i, market_data in enumerate(market_data_list):
                current_regime = current_regimes[i] if i < len(current_regimes) else 'sideways_market'
                
                transition_analysis = self.detect_regime_transition(market_data, current_regime)
                
                transition_summaries.append({
                    'current_regime': current_regime,
                    'next_regime_prediction': transition_analysis['next_regime_prediction'],
                    'transition_strength': transition_analysis['transition_strength'],
                    'transition_probability': transition_analysis['transition_probability']
                })
                
                transition_strength = transition_analysis['transition_strength']
                if transition_strength == 'strong':
                    strong_transitions += 1
                elif transition_strength == 'medium':
                    medium_transitions += 1
                elif transition_strength == 'weak':
                    weak_transitions += 1
                else:
                    no_transitions += 1
            
            return {
                'total_markets': len(market_data_list),
                'strong_transitions': strong_transitions,
                'medium_transitions': medium_transitions,
                'weak_transitions': weak_transitions,
                'no_transitions': no_transitions,
                'transition_summaries': transition_summaries,
                'overall_transition_risk': 'high' if strong_transitions > 0 else 'medium' if medium_transitions > 0 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error getting transition summary: {e}")
            return {
                'total_markets': len(market_data_list),
                'strong_transitions': 0,
                'medium_transitions': 0,
                'weak_transitions': 0,
                'no_transitions': 0,
                'transition_summaries': [],
                'overall_transition_risk': 'unknown'
            }

# Global instance
ai_market_regime_transition_detector = AIMarketRegimeTransitionDetector()
