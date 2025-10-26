#!/usr/bin/env python3
"""
AI-Powered Market Cycle Predictor for Sustainable Trading Bot
Predicts market cycles and seasonal patterns to optimize trading windows and avoid bad market periods
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

class AIMarketCyclePredictor:
    def __init__(self):
        self.cycle_cache = {}
        self.cache_duration = 3600  # 1 hour cache for cycle analysis
        self.cycle_history = deque(maxlen=1000)
        self.pattern_history = deque(maxlen=500)
        self.seasonal_history = deque(maxlen=500)
        
        # Market cycle configuration
        self.cycle_phases = {
            'accumulation': {
                'name': 'Accumulation Phase',
                'characteristics': ['low_volume', 'sideways_price', 'smart_money_buying'],
                'duration_days': 30,
                'trading_strategy': 'buy_and_hold',
                'risk_level': 'low'
            },
            'markup': {
                'name': 'Markup Phase',
                'characteristics': ['rising_prices', 'increasing_volume', 'public_interest'],
                'duration_days': 60,
                'trading_strategy': 'momentum',
                'risk_level': 'medium'
            },
            'distribution': {
                'name': 'Distribution Phase',
                'characteristics': ['high_volume', 'volatile_prices', 'smart_money_selling'],
                'duration_days': 30,
                'trading_strategy': 'take_profits',
                'risk_level': 'high'
            },
            'markdown': {
                'name': 'Markdown Phase',
                'characteristics': ['falling_prices', 'high_volume', 'panic_selling'],
                'duration_days': 45,
                'trading_strategy': 'avoid_trading',
                'risk_level': 'critical'
            }
        }
        
        # Seasonal patterns configuration
        self.seasonal_patterns = {
            'january_effect': {
                'name': 'January Effect',
                'months': [1],
                'characteristics': ['year_start_rally', 'tax_loss_selling_recovery'],
                'impact': 'positive',
                'strength': 'medium'
            },
            'summer_doldrums': {
                'name': 'Summer Doldrums',
                'months': [6, 7, 8],
                'characteristics': ['low_volume', 'sideways_movement', 'vacation_effect'],
                'impact': 'negative',
                'strength': 'weak'
            },
            'september_effect': {
                'name': 'September Effect',
                'months': [9],
                'characteristics': ['historical_weakness', 'back_to_school_selling'],
                'impact': 'negative',
                'strength': 'medium'
            },
            'october_crash': {
                'name': 'October Crash Season',
                'months': [10],
                'characteristics': ['high_volatility', 'crash_risk', 'historical_crashes'],
                'impact': 'negative',
                'strength': 'strong'
            },
            'november_rally': {
                'name': 'November Rally',
                'months': [11],
                'characteristics': ['year_end_rally', 'holiday_optimism'],
                'impact': 'positive',
                'strength': 'medium'
            },
            'december_rally': {
                'name': 'December Rally',
                'months': [12],
                'characteristics': ['year_end_optimism', 'window_dressing'],
                'impact': 'positive',
                'strength': 'strong'
            }
        }
        
        # Cycle prediction factors (must sum to 1.0)
        self.cycle_factors = {
            'price_momentum': 0.25,  # 25% weight for price momentum
            'volume_patterns': 0.20,  # 20% weight for volume patterns
            'sentiment_cycles': 0.20,  # 20% weight for sentiment cycles
            'seasonal_patterns': 0.15,  # 15% weight for seasonal patterns
            'market_structure': 0.10,  # 10% weight for market structure
            'external_factors': 0.10  # 10% weight for external factors
        }
        
        # Cycle phase thresholds
        self.strong_phase_threshold = 0.8  # 80% strong phase signal
        self.medium_phase_threshold = 0.6  # 60% medium phase signal
        self.weak_phase_threshold = 0.4  # 40% weak phase signal
        
        # Cycle transition thresholds
        self.imminent_transition_threshold = 0.9  # 90% imminent transition
        self.near_term_transition_threshold = 0.7  # 70% near-term transition
        self.medium_term_transition_threshold = 0.5  # 50% medium-term transition
        
        # Seasonal impact thresholds
        self.strong_seasonal_threshold = 0.8  # 80% strong seasonal impact
        self.medium_seasonal_threshold = 0.6  # 60% medium seasonal impact
        self.weak_seasonal_threshold = 0.4  # 40% weak seasonal impact
        
        # Cycle duration thresholds
        self.short_cycle_threshold = 30  # 30 days short cycle
        self.medium_cycle_threshold = 90  # 90 days medium cycle
        self.long_cycle_threshold = 180  # 180 days long cycle
        self.extended_cycle_threshold = 365  # 365 days extended cycle
    
    def predict_market_cycle(self, token: Dict, trade_amount: float, market_data: Dict = None) -> Dict:
        """
        Predict current market cycle phase and upcoming transitions
        Returns comprehensive cycle analysis with trading recommendations
        """
        # Provide default market_data if not provided
        if market_data is None:
            market_data = {
                'timestamp': datetime.now().isoformat(),
                'price': float(token.get('priceUsd', 0)),
                'volume': float(token.get('volume24h', 0)),
                'liquidity': float(token.get('liquidity', 0))
            }
            
        try:
            cache_key = f"cycle_{market_data.get('timestamp', 'current')}"
            
            # Check cache
            if cache_key in self.cycle_cache:
                cached_data = self.cycle_cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).total_seconds() < self.cache_duration:
                    logger.debug("Using cached cycle analysis")
                    return cached_data['cycle_data']
            
            # Analyze cycle components
            price_momentum_analysis = self._analyze_price_momentum_cycles(market_data, historical_data)
            volume_pattern_analysis = self._analyze_volume_pattern_cycles(market_data, historical_data)
            sentiment_cycle_analysis = self._analyze_sentiment_cycles(market_data, historical_data)
            seasonal_pattern_analysis = self._analyze_seasonal_patterns(market_data)
            market_structure_analysis = self._analyze_market_structure_cycles(market_data, historical_data)
            external_factor_analysis = self._analyze_external_factors(market_data)
            
            # Calculate current cycle phase
            current_cycle_phase = self._calculate_current_cycle_phase(
                price_momentum_analysis, volume_pattern_analysis, sentiment_cycle_analysis,
                seasonal_pattern_analysis, market_structure_analysis, external_factor_analysis
            )
            
            # Predict next cycle phase
            next_cycle_phase = self._predict_next_cycle_phase(
                current_cycle_phase, market_data, historical_data
            )
            
            # Calculate cycle transition probability
            transition_probability = self._calculate_transition_probability(
                current_cycle_phase, next_cycle_phase, market_data
            )
            
            # Calculate cycle timing
            cycle_timing = self._calculate_cycle_timing(
                current_cycle_phase, transition_probability, market_data
            )
            
            # Analyze seasonal patterns
            seasonal_analysis = self._analyze_current_seasonal_patterns(market_data)
            
            # Generate optimal trading windows
            optimal_trading_windows = self._generate_optimal_trading_windows(
                current_cycle_phase, seasonal_analysis, market_data
            )
            
            # Calculate cycle risk
            cycle_risk = self._calculate_cycle_risk(
                current_cycle_phase, transition_probability, seasonal_analysis
            )
            
            # Generate cycle recommendations
            cycle_recommendations = self._generate_cycle_recommendations(
                current_cycle_phase, next_cycle_phase, transition_probability,
                seasonal_analysis, cycle_risk
            )
            
            # Generate cycle insights
            cycle_insights = self._generate_cycle_insights(
                current_cycle_phase, next_cycle_phase, transition_probability,
                seasonal_analysis, cycle_timing
            )
            
            result = {
                'current_cycle_phase': current_cycle_phase,
                'next_cycle_phase': next_cycle_phase,
                'transition_probability': transition_probability,
                'cycle_timing': cycle_timing,
                'seasonal_analysis': seasonal_analysis,
                'optimal_trading_windows': optimal_trading_windows,
                'cycle_risk': cycle_risk,
                'price_momentum_analysis': price_momentum_analysis,
                'volume_pattern_analysis': volume_pattern_analysis,
                'sentiment_cycle_analysis': sentiment_cycle_analysis,
                'seasonal_pattern_analysis': seasonal_pattern_analysis,
                'market_structure_analysis': market_structure_analysis,
                'external_factor_analysis': external_factor_analysis,
                'cycle_recommendations': cycle_recommendations,
                'cycle_insights': cycle_insights,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Cache the result
            self.cycle_cache[cache_key] = {'timestamp': datetime.now(), 'cycle_data': result}
            
            logger.info(f"🔄 Cycle prediction: {current_cycle_phase} -> {next_cycle_phase} (transition: {transition_probability:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Market cycle prediction failed: {e}")
            return self._get_default_cycle_analysis(market_data)
    
    def _analyze_price_momentum_cycles(self, market_data: Dict, historical_data: Dict) -> Dict:
        """Analyze price momentum cycles"""
        try:
            # Extract price data
            current_price = market_data.get('current_price', 100)
            price_30d_ago = historical_data.get('price_30d_ago', 100)
            price_90d_ago = historical_data.get('price_90d_ago', 100)
            price_180d_ago = historical_data.get('price_180d_ago', 100)
            price_365d_ago = historical_data.get('price_365d_ago', 100)
            
            # Calculate price changes
            change_30d = (current_price - price_30d_ago) / price_30d_ago if price_30d_ago > 0 else 0
            change_90d = (current_price - price_90d_ago) / price_90d_ago if price_90d_ago > 0 else 0
            change_180d = (current_price - price_180d_ago) / price_180d_ago if price_180d_ago > 0 else 0
            change_365d = (current_price - price_365d_ago) / price_365d_ago if price_365d_ago > 0 else 0
            
            # Calculate momentum cycles
            short_term_momentum = change_30d
            medium_term_momentum = change_90d
            long_term_momentum = change_180d
            very_long_term_momentum = change_365d
            
            # Determine momentum cycle phase
            if short_term_momentum > 0.2 and medium_term_momentum > 0.1:  # 20% and 10% gains
                momentum_phase = "strong_uptrend"
                momentum_characteristics = "very_bullish"
            elif short_term_momentum > 0.1 and medium_term_momentum > 0.05:  # 10% and 5% gains
                momentum_phase = "uptrend"
                momentum_characteristics = "bullish"
            elif short_term_momentum < -0.2 and medium_term_momentum < -0.1:  # 20% and 10% losses
                momentum_phase = "strong_downtrend"
                momentum_characteristics = "very_bearish"
            elif short_term_momentum < -0.1 and medium_term_momentum < -0.05:  # 10% and 5% losses
                momentum_phase = "downtrend"
                momentum_characteristics = "bearish"
            else:
                momentum_phase = "sideways"
                momentum_characteristics = "neutral"
            
            # Calculate momentum score
            momentum_score = (
                short_term_momentum * 0.4 +
                medium_term_momentum * 0.3 +
                long_term_momentum * 0.2 +
                very_long_term_momentum * 0.1
            )
            
            return {
                'short_term_momentum': short_term_momentum,
                'medium_term_momentum': medium_term_momentum,
                'long_term_momentum': long_term_momentum,
                'very_long_term_momentum': very_long_term_momentum,
                'momentum_phase': momentum_phase,
                'momentum_characteristics': momentum_characteristics,
                'momentum_score': momentum_score
            }
            
        except Exception:
            return {
                'short_term_momentum': 0.0,
                'medium_term_momentum': 0.0,
                'long_term_momentum': 0.0,
                'very_long_term_momentum': 0.0,
                'momentum_phase': 'sideways',
                'momentum_characteristics': 'neutral',
                'momentum_score': 0.0
            }
    
    def _analyze_volume_pattern_cycles(self, market_data: Dict, historical_data: Dict) -> Dict:
        """Analyze volume pattern cycles"""
        try:
            # Extract volume data
            current_volume = market_data.get('current_volume', 1000000)
            volume_30d_ago = historical_data.get('volume_30d_ago', 1000000)
            volume_90d_ago = historical_data.get('volume_90d_ago', 1000000)
            volume_180d_ago = historical_data.get('volume_180d_ago', 1000000)
            volume_365d_ago = historical_data.get('volume_365d_ago', 1000000)
            
            # Calculate volume ratios
            volume_ratio_30d = current_volume / volume_30d_ago if volume_30d_ago > 0 else 1.0
            volume_ratio_90d = current_volume / volume_90d_ago if volume_90d_ago > 0 else 1.0
            volume_ratio_180d = current_volume / volume_180d_ago if volume_180d_ago > 0 else 1.0
            volume_ratio_365d = current_volume / volume_365d_ago if volume_365d_ago > 0 else 1.0
            
            # Calculate volume momentum
            volume_momentum = (
                (volume_ratio_30d - 1.0) * 0.4 +
                (volume_ratio_90d - 1.0) * 0.3 +
                (volume_ratio_180d - 1.0) * 0.2 +
                (volume_ratio_365d - 1.0) * 0.1
            )
            
            # Determine volume cycle phase
            if volume_ratio_30d > 2.0 and volume_ratio_90d > 1.5:  # 2x and 1.5x volume
                volume_phase = "high_volume_cycle"
                volume_characteristics = "very_active"
            elif volume_ratio_30d > 1.5 and volume_ratio_90d > 1.2:  # 1.5x and 1.2x volume
                volume_phase = "increasing_volume"
                volume_characteristics = "active"
            elif volume_ratio_30d < 0.5 and volume_ratio_90d < 0.7:  # 0.5x and 0.7x volume
                volume_phase = "low_volume_cycle"
                volume_characteristics = "inactive"
            elif volume_ratio_30d < 0.7 and volume_ratio_90d < 0.8:  # 0.7x and 0.8x volume
                volume_phase = "decreasing_volume"
                volume_characteristics = "declining"
            else:
                volume_phase = "normal_volume"
                volume_characteristics = "stable"
            
            # Calculate volume score
            volume_score = min(1.0, max(0.0, (volume_momentum + 1.0) / 2.0))  # Normalize to 0-1
            
            return {
                'volume_ratio_30d': volume_ratio_30d,
                'volume_ratio_90d': volume_ratio_90d,
                'volume_ratio_180d': volume_ratio_180d,
                'volume_ratio_365d': volume_ratio_365d,
                'volume_momentum': volume_momentum,
                'volume_phase': volume_phase,
                'volume_characteristics': volume_characteristics,
                'volume_score': volume_score
            }
            
        except Exception:
            return {
                'volume_ratio_30d': 1.0,
                'volume_ratio_90d': 1.0,
                'volume_ratio_180d': 1.0,
                'volume_ratio_365d': 1.0,
                'volume_momentum': 0.0,
                'volume_phase': 'normal_volume',
                'volume_characteristics': 'stable',
                'volume_score': 0.5
            }
    
    def _analyze_sentiment_cycles(self, market_data: Dict, historical_data: Dict) -> Dict:
        """Analyze sentiment cycles"""
        try:
            # Extract sentiment data
            current_sentiment = market_data.get('current_sentiment', 0.5)
            sentiment_30d_ago = historical_data.get('sentiment_30d_ago', 0.5)
            sentiment_90d_ago = historical_data.get('sentiment_90d_ago', 0.5)
            sentiment_180d_ago = historical_data.get('sentiment_180d_ago', 0.5)
            sentiment_365d_ago = historical_data.get('sentiment_365d_ago', 0.5)
            
            # Calculate sentiment changes
            sentiment_change_30d = current_sentiment - sentiment_30d_ago
            sentiment_change_90d = current_sentiment - sentiment_90d_ago
            sentiment_change_180d = current_sentiment - sentiment_180d_ago
            sentiment_change_365d = current_sentiment - sentiment_365d_ago
            
            # Calculate sentiment momentum
            sentiment_momentum = (
                sentiment_change_30d * 0.4 +
                sentiment_change_90d * 0.3 +
                sentiment_change_180d * 0.2 +
                sentiment_change_365d * 0.1
            )
            
            # Determine sentiment cycle phase
            if current_sentiment > 0.8 and sentiment_momentum > 0.1:  # 80% sentiment, 10% momentum
                sentiment_phase = "euphoria"
                sentiment_characteristics = "extreme_optimism"
            elif current_sentiment > 0.6 and sentiment_momentum > 0.05:  # 60% sentiment, 5% momentum
                sentiment_phase = "optimism"
                sentiment_characteristics = "positive"
            elif current_sentiment < 0.2 and sentiment_momentum < -0.1:  # 20% sentiment, -10% momentum
                sentiment_phase = "despair"
                sentiment_characteristics = "extreme_pessimism"
            elif current_sentiment < 0.4 and sentiment_momentum < -0.05:  # 40% sentiment, -5% momentum
                sentiment_phase = "pessimism"
                sentiment_characteristics = "negative"
            else:
                sentiment_phase = "neutral"
                sentiment_characteristics = "balanced"
            
            # Calculate sentiment score
            sentiment_score = current_sentiment
            
            return {
                'sentiment_change_30d': sentiment_change_30d,
                'sentiment_change_90d': sentiment_change_90d,
                'sentiment_change_180d': sentiment_change_180d,
                'sentiment_change_365d': sentiment_change_365d,
                'sentiment_momentum': sentiment_momentum,
                'sentiment_phase': sentiment_phase,
                'sentiment_characteristics': sentiment_characteristics,
                'sentiment_score': sentiment_score
            }
            
        except Exception:
            return {
                'sentiment_change_30d': 0.0,
                'sentiment_change_90d': 0.0,
                'sentiment_change_180d': 0.0,
                'sentiment_change_365d': 0.0,
                'sentiment_momentum': 0.0,
                'sentiment_phase': 'neutral',
                'sentiment_characteristics': 'balanced',
                'sentiment_score': 0.5
            }
    
    def _analyze_seasonal_patterns(self, market_data: Dict) -> Dict:
        """Analyze seasonal patterns"""
        try:
            current_date = datetime.now()
            current_month = current_date.month
            current_day = current_date.day
            
            # Check for seasonal patterns
            active_patterns = []
            seasonal_impact = 0.0
            
            for pattern_name, pattern_config in self.seasonal_patterns.items():
                if current_month in pattern_config['months']:
                    active_patterns.append({
                        'name': pattern_config['name'],
                        'impact': pattern_config['impact'],
                        'strength': pattern_config['strength']
                    })
                    
                    # Calculate seasonal impact
                    if pattern_config['impact'] == 'positive':
                        if pattern_config['strength'] == 'strong':
                            seasonal_impact += 0.3
                        elif pattern_config['strength'] == 'medium':
                            seasonal_impact += 0.2
                        else:
                            seasonal_impact += 0.1
                    else:  # negative impact
                        if pattern_config['strength'] == 'strong':
                            seasonal_impact -= 0.3
                        elif pattern_config['strength'] == 'medium':
                            seasonal_impact -= 0.2
                        else:
                            seasonal_impact -= 0.1
            
            # Determine seasonal phase
            if seasonal_impact > 0.2:  # 20% positive impact
                seasonal_phase = "positive_season"
                seasonal_characteristics = "favorable"
            elif seasonal_impact < -0.2:  # 20% negative impact
                seasonal_phase = "negative_season"
                seasonal_characteristics = "unfavorable"
            else:
                seasonal_phase = "neutral_season"
                seasonal_characteristics = "neutral"
            
            return {
                'active_patterns': active_patterns,
                'seasonal_impact': seasonal_impact,
                'seasonal_phase': seasonal_phase,
                'seasonal_characteristics': seasonal_characteristics,
                'current_month': current_month,
                'current_day': current_day
            }
            
        except Exception:
            return {
                'active_patterns': [],
                'seasonal_impact': 0.0,
                'seasonal_phase': 'neutral_season',
                'seasonal_characteristics': 'neutral',
                'current_month': datetime.now().month,
                'current_day': datetime.now().day
            }
    
    def _analyze_market_structure_cycles(self, market_data: Dict, historical_data: Dict) -> Dict:
        """Analyze market structure cycles"""
        try:
            # Extract market structure data
            current_market_cap = market_data.get('market_cap', 1000000000)
            market_cap_30d_ago = historical_data.get('market_cap_30d_ago', 1000000000)
            market_cap_90d_ago = historical_data.get('market_cap_90d_ago', 1000000000)
            
            # Calculate market cap changes
            market_cap_change_30d = (current_market_cap - market_cap_30d_ago) / market_cap_30d_ago if market_cap_30d_ago > 0 else 0
            market_cap_change_90d = (current_market_cap - market_cap_90d_ago) / market_cap_90d_ago if market_cap_90d_ago > 0 else 0
            
            # Calculate market structure momentum
            structure_momentum = (
                market_cap_change_30d * 0.6 +
                market_cap_change_90d * 0.4
            )
            
            # Determine market structure phase
            if structure_momentum > 0.2:  # 20% growth
                structure_phase = "expansion"
                structure_characteristics = "growing"
            elif structure_momentum > 0.05:  # 5% growth
                structure_phase = "growth"
                structure_characteristics = "moderate_growth"
            elif structure_momentum < -0.2:  # 20% decline
                structure_phase = "contraction"
                structure_characteristics = "shrinking"
            elif structure_momentum < -0.05:  # 5% decline
                structure_phase = "decline"
                structure_characteristics = "moderate_decline"
            else:
                structure_phase = "stable"
                structure_characteristics = "stable"
            
            # Calculate structure score
            structure_score = max(0.0, min(1.0, (structure_momentum + 0.5)))
            
            return {
                'market_cap_change_30d': market_cap_change_30d,
                'market_cap_change_90d': market_cap_change_90d,
                'structure_momentum': structure_momentum,
                'structure_phase': structure_phase,
                'structure_characteristics': structure_characteristics,
                'structure_score': structure_score
            }
            
        except Exception:
            return {
                'market_cap_change_30d': 0.0,
                'market_cap_change_90d': 0.0,
                'structure_momentum': 0.0,
                'structure_phase': 'stable',
                'structure_characteristics': 'stable',
                'structure_score': 0.5
            }
    
    def _analyze_external_factors(self, market_data: Dict) -> Dict:
        """Analyze external factors affecting market cycles"""
        try:
            # Extract external factor data
            news_sentiment = market_data.get('news_sentiment', 0.5)
            regulatory_environment = market_data.get('regulatory_environment', 0.5)
            economic_indicators = market_data.get('economic_indicators', 0.5)
            geopolitical_risk = market_data.get('geopolitical_risk', 0.5)
            
            # Calculate external factor impact
            external_impact = (
                news_sentiment * 0.3 +
                regulatory_environment * 0.25 +
                economic_indicators * 0.25 +
                (1.0 - geopolitical_risk) * 0.2  # Lower geopolitical risk is better
            )
            
            # Determine external factor phase
            if external_impact > 0.7:  # 70% positive impact
                external_phase = "very_favorable"
                external_characteristics = "excellent_conditions"
            elif external_impact > 0.5:  # 50% positive impact
                external_phase = "favorable"
                external_characteristics = "good_conditions"
            elif external_impact > 0.3:  # 30% positive impact
                external_phase = "neutral"
                external_characteristics = "neutral_conditions"
            elif external_impact > 0.1:  # 10% positive impact
                external_phase = "unfavorable"
                external_characteristics = "poor_conditions"
            else:
                external_phase = "very_unfavorable"
                external_characteristics = "very_poor_conditions"
            
            return {
                'news_sentiment': news_sentiment,
                'regulatory_environment': regulatory_environment,
                'economic_indicators': economic_indicators,
                'geopolitical_risk': geopolitical_risk,
                'external_impact': external_impact,
                'external_phase': external_phase,
                'external_characteristics': external_characteristics
            }
            
        except Exception:
            return {
                'news_sentiment': 0.5,
                'regulatory_environment': 0.5,
                'economic_indicators': 0.5,
                'geopolitical_risk': 0.5,
                'external_impact': 0.5,
                'external_phase': 'neutral',
                'external_characteristics': 'neutral_conditions'
            }
    
    def _calculate_current_cycle_phase(self, price_momentum_analysis: Dict, volume_pattern_analysis: Dict,
                                     sentiment_cycle_analysis: Dict, seasonal_pattern_analysis: Dict,
                                     market_structure_analysis: Dict, external_factor_analysis: Dict) -> str:
        """Calculate current market cycle phase"""
        try:
            # Weight the individual analysis scores
            cycle_score = (
                price_momentum_analysis.get('momentum_score', 0.5) * self.cycle_factors['price_momentum'] +
                volume_pattern_analysis.get('volume_score', 0.5) * self.cycle_factors['volume_patterns'] +
                sentiment_cycle_analysis.get('sentiment_score', 0.5) * self.cycle_factors['sentiment_cycles'] +
                (seasonal_pattern_analysis.get('seasonal_impact', 0.0) + 0.5) * self.cycle_factors['seasonal_patterns'] +
                market_structure_analysis.get('structure_score', 0.5) * self.cycle_factors['market_structure'] +
                external_factor_analysis.get('external_impact', 0.5) * self.cycle_factors['external_factors']
            )
            
            # Determine cycle phase based on score
            if cycle_score > 0.8:  # 80% score
                return "markup"
            elif cycle_score > 0.6:  # 60% score
                return "accumulation"
            elif cycle_score > 0.4:  # 40% score
                return "distribution"
            else:  # Below 40% score
                return "markdown"
                
        except Exception:
            return "accumulation"
    
    def _predict_next_cycle_phase(self, current_cycle_phase: str, market_data: Dict, historical_data: Dict) -> str:
        """Predict next market cycle phase"""
        try:
            # Get current cycle phase info
            current_phase_info = self.cycle_phases.get(current_cycle_phase, {})
            current_duration = current_phase_info.get('duration_days', 30)
            
            # Predict next phase based on cycle progression
            cycle_progression = {
                'accumulation': 'markup',
                'markup': 'distribution',
                'distribution': 'markdown',
                'markdown': 'accumulation'
            }
            
            # Get next phase
            next_phase = cycle_progression.get(current_cycle_phase, 'accumulation')
            
            # Adjust based on market conditions
            current_sentiment = market_data.get('current_sentiment', 0.5)
            if current_sentiment > 0.8:  # High sentiment
                if current_cycle_phase == 'markup':
                    next_phase = 'distribution'  # Skip to distribution
                elif current_cycle_phase == 'accumulation':
                    next_phase = 'markup'  # Accelerate to markup
            elif current_sentiment < 0.2:  # Low sentiment
                if current_cycle_phase == 'markdown':
                    next_phase = 'accumulation'  # Skip to accumulation
                elif current_cycle_phase == 'distribution':
                    next_phase = 'markdown'  # Accelerate to markdown
            
            return next_phase
            
        except Exception:
            return "accumulation"
    
    def _calculate_transition_probability(self, current_cycle_phase: str, next_cycle_phase: str, market_data: Dict) -> float:
        """Calculate probability of cycle transition"""
        try:
            # Base transition probability
            base_probability = 0.3  # 30% base probability
            
            # Adjust based on current phase duration
            current_phase_info = self.cycle_phases.get(current_cycle_phase, {})
            expected_duration = current_phase_info.get('duration_days', 30)
            
            # Simulate current phase duration (in real implementation, this would be tracked)
            current_duration = random.randint(1, expected_duration * 2)
            
            # Calculate duration-based probability
            if current_duration > expected_duration:
                duration_factor = min(1.0, (current_duration - expected_duration) / expected_duration)
            else:
                duration_factor = 0.0
            
            # Adjust based on market volatility
            current_volatility = market_data.get('current_volatility', 0.3)
            volatility_factor = min(1.0, current_volatility * 2.0)
            
            # Calculate final transition probability
            transition_probability = min(1.0, base_probability + duration_factor * 0.3 + volatility_factor * 0.2)
            
            return transition_probability
            
        except Exception:
            return 0.3
    
    def _calculate_cycle_timing(self, current_cycle_phase: str, transition_probability: float, market_data: Dict) -> str:
        """Calculate cycle transition timing"""
        try:
            if transition_probability > self.imminent_transition_threshold:
                return "imminent"
            elif transition_probability > self.near_term_transition_threshold:
                return "near_term"
            elif transition_probability > self.medium_term_transition_threshold:
                return "medium_term"
            else:
                return "long_term"
                
        except Exception:
            return "medium_term"
    
    def _analyze_current_seasonal_patterns(self, market_data: Dict) -> Dict:
        """Analyze current seasonal patterns"""
        try:
            current_date = datetime.now()
            current_month = current_date.month
            
            # Get active seasonal patterns
            active_patterns = []
            seasonal_impact = 0.0
            
            for pattern_name, pattern_config in self.seasonal_patterns.items():
                if current_month in pattern_config['months']:
                    active_patterns.append({
                        'name': pattern_config['name'],
                        'impact': pattern_config['impact'],
                        'strength': pattern_config['strength']
                    })
                    
                    # Calculate seasonal impact
                    if pattern_config['impact'] == 'positive':
                        if pattern_config['strength'] == 'strong':
                            seasonal_impact += 0.3
                        elif pattern_config['strength'] == 'medium':
                            seasonal_impact += 0.2
                        else:
                            seasonal_impact += 0.1
                    else:  # negative impact
                        if pattern_config['strength'] == 'strong':
                            seasonal_impact -= 0.3
                        elif pattern_config['strength'] == 'medium':
                            seasonal_impact -= 0.2
                        else:
                            seasonal_impact -= 0.1
            
            # Determine seasonal phase
            if seasonal_impact > 0.2:  # 20% positive impact
                seasonal_phase = "positive_season"
                seasonal_characteristics = "favorable"
            elif seasonal_impact < -0.2:  # 20% negative impact
                seasonal_phase = "negative_season"
                seasonal_characteristics = "unfavorable"
            else:
                seasonal_phase = "neutral_season"
                seasonal_characteristics = "neutral"
            
            return {
                'active_patterns': active_patterns,
                'seasonal_impact': seasonal_impact,
                'seasonal_phase': seasonal_phase,
                'seasonal_characteristics': seasonal_characteristics,
                'current_month': current_month
            }
            
        except Exception:
            return {
                'active_patterns': [],
                'seasonal_impact': 0.0,
                'seasonal_phase': 'neutral_season',
                'seasonal_characteristics': 'neutral',
                'current_month': datetime.now().month
            }
    
    def _generate_optimal_trading_windows(self, current_cycle_phase: str, seasonal_analysis: Dict, market_data: Dict) -> List[Dict]:
        """Generate optimal trading windows based on cycle and seasonal analysis"""
        try:
            trading_windows = []
            
            # Get current cycle phase info
            current_phase_info = self.cycle_phases.get(current_cycle_phase, {})
            trading_strategy = current_phase_info.get('trading_strategy', 'buy_and_hold')
            risk_level = current_phase_info.get('risk_level', 'medium')
            
            # Generate trading windows based on cycle phase
            if current_cycle_phase == "accumulation":
                trading_windows.append({
                    'window': 'current',
                    'quality': 'excellent',
                    'strategy': 'buy_and_hold',
                    'risk_level': 'low',
                    'recommendation': 'accumulate_positions'
                })
            elif current_cycle_phase == "markup":
                trading_windows.append({
                    'window': 'current',
                    'quality': 'good',
                    'strategy': 'momentum',
                    'risk_level': 'medium',
                    'recommendation': 'ride_trend'
                })
            elif current_cycle_phase == "distribution":
                trading_windows.append({
                    'window': 'current',
                    'quality': 'fair',
                    'strategy': 'take_profits',
                    'risk_level': 'high',
                    'recommendation': 'reduce_positions'
                })
            else:  # markdown
                trading_windows.append({
                    'window': 'current',
                    'quality': 'poor',
                    'strategy': 'avoid_trading',
                    'risk_level': 'critical',
                    'recommendation': 'exit_positions'
                })
            
            # Adjust based on seasonal patterns
            seasonal_impact = seasonal_analysis.get('seasonal_impact', 0.0)
            if seasonal_impact > 0.2:  # Positive seasonal impact
                for window in trading_windows:
                    window['quality'] = 'excellent' if window['quality'] == 'good' else window['quality']
                    window['recommendation'] = 'increase_activity'
            elif seasonal_impact < -0.2:  # Negative seasonal impact
                for window in trading_windows:
                    window['quality'] = 'poor' if window['quality'] == 'fair' else window['quality']
                    window['recommendation'] = 'reduce_activity'
            
            return trading_windows
            
        except Exception:
            return [{'window': 'current', 'quality': 'fair', 'strategy': 'monitor', 'risk_level': 'medium', 'recommendation': 'monitor_conditions'}]
    
    def _calculate_cycle_risk(self, current_cycle_phase: str, transition_probability: float, seasonal_analysis: Dict) -> str:
        """Calculate cycle risk level"""
        try:
            # Get current cycle phase risk
            current_phase_info = self.cycle_phases.get(current_cycle_phase, {})
            phase_risk = current_phase_info.get('risk_level', 'medium')
            
            # Adjust based on transition probability
            if transition_probability > 0.8:  # 80% transition probability
                if phase_risk == 'low':
                    cycle_risk = 'medium'
                elif phase_risk == 'medium':
                    cycle_risk = 'high'
                else:
                    cycle_risk = 'critical'
            elif transition_probability > 0.6:  # 60% transition probability
                if phase_risk == 'low':
                    cycle_risk = 'low'
                elif phase_risk == 'medium':
                    cycle_risk = 'medium'
                else:
                    cycle_risk = 'high'
            else:
                cycle_risk = phase_risk
            
            # Adjust based on seasonal patterns
            seasonal_impact = seasonal_analysis.get('seasonal_impact', 0.0)
            if seasonal_impact < -0.3:  # Strong negative seasonal impact
                if cycle_risk == 'low':
                    cycle_risk = 'medium'
                elif cycle_risk == 'medium':
                    cycle_risk = 'high'
                else:
                    cycle_risk = 'critical'
            
            return cycle_risk
            
        except Exception:
            return 'medium'
    
    def _generate_cycle_recommendations(self, current_cycle_phase: str, next_cycle_phase: str,
                                      transition_probability: float, seasonal_analysis: Dict,
                                      cycle_risk: str) -> List[str]:
        """Generate cycle-based recommendations"""
        recommendations = []
        
        try:
            # Current cycle phase recommendations
            current_phase_info = self.cycle_phases.get(current_cycle_phase, {})
            trading_strategy = current_phase_info.get('trading_strategy', 'buy_and_hold')
            
            if current_cycle_phase == "accumulation":
                recommendations.append("Accumulation phase - excellent buying opportunity")
                recommendations.append("Consider building positions gradually")
                recommendations.append("Focus on quality tokens with strong fundamentals")
            elif current_cycle_phase == "markup":
                recommendations.append("Markup phase - momentum trading opportunity")
                recommendations.append("Ride the trend but monitor for distribution signals")
                recommendations.append("Consider taking partial profits at resistance levels")
            elif current_cycle_phase == "distribution":
                recommendations.append("Distribution phase - reduce positions")
                recommendations.append("Take profits and prepare for potential decline")
                recommendations.append("Avoid new positions, focus on exit strategies")
            else:  # markdown
                recommendations.append("Markdown phase - avoid trading")
                recommendations.append("Exit positions and wait for accumulation phase")
                recommendations.append("Focus on capital preservation")
            
            # Transition recommendations
            if transition_probability > 0.8:  # 80% transition probability
                recommendations.append(f"High transition probability to {next_cycle_phase} phase")
                recommendations.append("Prepare for phase change and adjust strategy")
            elif transition_probability > 0.6:  # 60% transition probability
                recommendations.append(f"Medium transition probability to {next_cycle_phase} phase")
                recommendations.append("Monitor for transition signals")
            else:
                recommendations.append("Low transition probability - continue current strategy")
            
            # Seasonal recommendations
            seasonal_impact = seasonal_analysis.get('seasonal_impact', 0.0)
            if seasonal_impact > 0.2:  # Positive seasonal impact
                recommendations.append("Positive seasonal patterns detected - increase activity")
            elif seasonal_impact < -0.2:  # Negative seasonal impact
                recommendations.append("Negative seasonal patterns detected - reduce activity")
            
            # Risk recommendations
            if cycle_risk == "critical":
                recommendations.append("Critical cycle risk - avoid all trading")
            elif cycle_risk == "high":
                recommendations.append("High cycle risk - use small position sizes")
            elif cycle_risk == "medium":
                recommendations.append("Medium cycle risk - moderate position sizes")
            else:
                recommendations.append("Low cycle risk - normal position sizes")
            
        except Exception:
            recommendations.append("Monitor cycle conditions and adjust strategy accordingly")
        
        return recommendations
    
    def _generate_cycle_insights(self, current_cycle_phase: str, next_cycle_phase: str,
                               transition_probability: float, seasonal_analysis: Dict,
                               cycle_timing: str) -> List[str]:
        """Generate cycle insights"""
        insights = []
        
        try:
            # Current cycle insights
            current_phase_info = self.cycle_phases.get(current_cycle_phase, {})
            insights.append(f"Current cycle phase: {current_phase_info.get('name', current_cycle_phase)}")
            insights.append(f"Next predicted phase: {self.cycle_phases.get(next_cycle_phase, {}).get('name', next_cycle_phase)}")
            insights.append(f"Transition probability: {transition_probability:.1%}")
            insights.append(f"Transition timing: {cycle_timing}")
            
            # Seasonal insights
            seasonal_phase = seasonal_analysis.get('seasonal_phase', 'neutral_season')
            seasonal_impact = seasonal_analysis.get('seasonal_impact', 0.0)
            insights.append(f"Seasonal phase: {seasonal_phase}")
            insights.append(f"Seasonal impact: {seasonal_impact:.1%}")
            
            # Active seasonal patterns
            active_patterns = seasonal_analysis.get('active_patterns', [])
            if active_patterns:
                insights.append(f"Active seasonal patterns: {len(active_patterns)}")
                for pattern in active_patterns[:3]:  # Show top 3 patterns
                    insights.append(f"  • {pattern['name']} ({pattern['impact']}, {pattern['strength']})")
            else:
                insights.append("No active seasonal patterns")
            
        except Exception:
            insights.append("Cycle analysis completed")
        
        return insights
    
    def _get_default_cycle_analysis(self, market_data: Dict) -> Dict:
        """Return default cycle analysis when analysis fails"""
        return {
            'current_cycle_phase': 'accumulation',
            'next_cycle_phase': 'markup',
            'transition_probability': 0.3,
            'cycle_timing': 'medium_term',
            'seasonal_analysis': {
                'seasonal_phase': 'neutral_season',
                'seasonal_impact': 0.0,
                'active_patterns': []
            },
            'optimal_trading_windows': [{'window': 'current', 'quality': 'fair', 'strategy': 'monitor', 'risk_level': 'medium', 'recommendation': 'monitor_conditions'}],
            'cycle_risk': 'medium',
            'price_momentum_analysis': {'momentum_score': 0.5},
            'volume_pattern_analysis': {'volume_score': 0.5},
            'sentiment_cycle_analysis': {'sentiment_score': 0.5},
            'seasonal_pattern_analysis': {'seasonal_impact': 0.0},
            'market_structure_analysis': {'structure_score': 0.5},
            'external_factor_analysis': {'external_impact': 0.5},
            'cycle_recommendations': ['Monitor cycle conditions'],
            'cycle_insights': ['Cycle analysis completed'],
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def get_cycle_summary(self, market_data_list: List[Dict]) -> Dict:
        """Get cycle summary for multiple market conditions"""
        try:
            cycle_summaries = []
            accumulation_count = 0
            markup_count = 0
            distribution_count = 0
            markdown_count = 0
            
            for market_data in market_data_list:
                # Create a token-like dict from market_data for the prediction
                token_data = {
                    'priceUsd': market_data.get('price', 0),
                    'volume24h': market_data.get('volume', 0),
                    'liquidity': market_data.get('liquidity', 0),
                    'symbol': 'UNKNOWN'
                }
                cycle_analysis = self.predict_market_cycle(token_data, 5.0, market_data)
                
                cycle_summaries.append({
                    'current_cycle_phase': cycle_analysis['current_cycle_phase'],
                    'next_cycle_phase': cycle_analysis['next_cycle_phase'],
                    'transition_probability': cycle_analysis['transition_probability'],
                    'cycle_risk': cycle_analysis['cycle_risk']
                })
                
                current_phase = cycle_analysis['current_cycle_phase']
                if current_phase == 'accumulation':
                    accumulation_count += 1
                elif current_phase == 'markup':
                    markup_count += 1
                elif current_phase == 'distribution':
                    distribution_count += 1
                else:  # markdown
                    markdown_count += 1
            
            return {
                'total_markets': len(market_data_list),
                'accumulation_count': accumulation_count,
                'markup_count': markup_count,
                'distribution_count': distribution_count,
                'markdown_count': markdown_count,
                'cycle_summaries': cycle_summaries,
                'overall_cycle_phase': 'accumulation' if accumulation_count > markup_count else 'markup' if markup_count > distribution_count else 'distribution' if distribution_count > markdown_count else 'markdown'
            }
            
        except Exception as e:
            logger.error(f"Error getting cycle summary: {e}")
            return {
                'total_markets': len(market_data_list),
                'accumulation_count': 0,
                'markup_count': 0,
                'distribution_count': 0,
                'markdown_count': 0,
                'cycle_summaries': [],
                'overall_cycle_phase': 'unknown'
            }

# Global instance
ai_market_cycle_predictor = AIMarketCyclePredictor()
