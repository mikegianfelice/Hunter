#!/usr/bin/env python3
"""
AI-Powered Market Regime Detection for Sustainable Trading Bot
Uses machine learning to detect market conditions and adapt trading strategy
"""

import os
import json
import time
import logging
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

# Configure logging
logger = logging.getLogger(__name__)

class AIMarketRegimeDetector:
    def __init__(self):
        self.regime_cache = {}
        self.cache_duration = 1800  # 30 minutes cache
        self.api_timeout = 15
        self.max_retries = 3
        
        # Market regime categories
        self.regimes = {
            'bull_market': {
                'description': 'Strong upward trend with high confidence',
                'strategy': 'aggressive',
                'position_multiplier': 1.2,
                'quality_threshold_adjustment': -5,  # Lower threshold for more opportunities
                'risk_tolerance': 'high'
            },
            'bear_market': {
                'description': 'Strong downward trend with high confidence',
                'strategy': 'conservative',
                'position_multiplier': 0.7,
                'quality_threshold_adjustment': 10,  # Higher threshold for safety
                'risk_tolerance': 'low'
            },
            'sideways_market': {
                'description': 'Range-bound market with low volatility',
                'strategy': 'neutral',
                'position_multiplier': 1.0,
                'quality_threshold_adjustment': 0,
                'risk_tolerance': 'medium'
            },
            'high_volatility': {
                'description': 'Extreme volatility with uncertain direction',
                'strategy': 'cautious',
                'position_multiplier': 0.5,
                'quality_threshold_adjustment': 15,  # Much higher threshold
                'risk_tolerance': 'very_low'
            },
            'recovery_market': {
                'description': 'Recovering from downturn with mixed signals',
                'strategy': 'selective',
                'position_multiplier': 0.9,
                'quality_threshold_adjustment': 5,
                'risk_tolerance': 'medium'
            }
        }
        
        # Market indicators weights
        self.indicator_weights = {
            'btc_trend': 0.25,
            'sol_trend': 0.20,
            'market_correlation': 0.15,
            'volatility_index': 0.15,
            'volume_trends': 0.10,
            'fear_greed_index': 0.10,
            'market_cap_trend': 0.05
        }
    
    def detect_market_regime(self) -> Dict:
        """
        Detect current market regime using AI analysis
        Returns regime information and trading adjustments
        """
        now = time.time()
        cached = self.regime_cache.get('latest')
        if cached:
            cached_ts = cached.get('timestamp', 0)
            if now - cached_ts < self.cache_duration:
                return cached.get('data', self._get_default_regime())

        try:
            # Get market indicators
            indicators = self._collect_market_indicators()
            
            # Analyze indicators with AI
            regime_analysis = self._analyze_market_regime(indicators)
            
            # Get regime-specific trading adjustments
            regime_info = self.regimes[regime_analysis['regime']]
            
            result = {
                'regime': regime_analysis['regime'],
                'confidence': regime_analysis['confidence'],
                'description': regime_info['description'],
                'strategy': regime_info['strategy'],
                'position_multiplier': regime_info['position_multiplier'],
                'quality_threshold_adjustment': regime_info['quality_threshold_adjustment'],
                'risk_tolerance': regime_info['risk_tolerance'],
                'indicators': indicators,
                'analysis': regime_analysis,
                'timestamp': datetime.now().isoformat(),
                'recommendations': self._get_regime_recommendations(regime_analysis['regime'])
            }
            
            # Removed duplicate logging - regime detection is logged in main loop
            self.regime_cache['latest'] = {
                'timestamp': now,
                'data': result
            }
            return result
            
        except Exception as e:
            logger.error(f"❌ Market regime detection failed: {e}")
            return self._get_default_regime()
    
    def _collect_market_indicators(self) -> Dict:
        """Collect various market indicators for regime analysis"""
        indicators = {}
        
        try:
            # BTC trend analysis
            indicators['btc_trend'] = self._analyze_btc_trend()
            
            # SOL trend analysis (changed from ETH for Solana-focused trading)
            indicators['sol_trend'] = self._analyze_sol_trend()
            
            # Market correlation analysis
            indicators['market_correlation'] = self._analyze_market_correlation()
            
            # Volatility index
            indicators['volatility_index'] = self._calculate_volatility_index()
            
            # Volume trends
            indicators['volume_trends'] = self._analyze_volume_trends()
            
            # Fear & Greed Index (real data)
            indicators['fear_greed_index'] = self._get_fear_greed_index()
            
            # Market cap trend
            indicators['market_cap_trend'] = self._analyze_market_cap_trend()
            
        except Exception as e:
            logger.error(f"Error collecting market indicators: {e}")
            # Provide default values
            indicators = {
                'btc_trend': 0.5,
                'sol_trend': 0.5,
                'market_correlation': 0.5,
                'volatility_index': 0.5,
                'volume_trends': 0.5,
                'fear_greed_index': 0.5,
                'market_cap_trend': 0.5
            }
        
        return indicators
    
    def _analyze_btc_trend(self) -> float:
        """Analyze Bitcoin trend (0-1 scale) using extended timeframe"""
        try:
            # Use real BTC price data with extended timeframe for regime detection
            from src.utils.market_data_fetcher import market_data_fetcher
            from src.config.config_loader import get_config
            
            # Use extended timeframe (default 7 days) for better accuracy
            hours = get_config('market_analysis_timeframes.btc_trend_hours', 168)
            btc_trend = market_data_fetcher.get_btc_trend(hours=hours)
            return btc_trend
            
        except Exception as e:
            logger.error(f"Error analyzing BTC trend: {e}")
            return 0.5
    
    def _analyze_sol_trend(self) -> float:
        """Analyze Solana trend (0-1 scale) using extended timeframe"""
        try:
            # Use real SOL price data with extended timeframe for regime detection
            from src.utils.market_data_fetcher import market_data_fetcher
            from src.config.config_loader import get_config
            
            # Use extended timeframe (default 7 days) for better accuracy
            hours = get_config('market_analysis_timeframes.sol_trend_hours', 168)
            sol_trend = market_data_fetcher.get_sol_trend(hours=hours)
            return sol_trend
            
        except Exception as e:
            logger.error(f"Error analyzing SOL trend: {e}")
            return 0.5
    
    def _analyze_market_correlation(self) -> float:
        """Analyze market correlation (0-1 scale) using extended timeframe"""
        try:
            # Use real market data with extended timeframe for statistical significance
            from src.utils.market_data_fetcher import market_data_fetcher
            # Uses default 14 days (336 hours) for 60+ data points
            correlation = market_data_fetcher.get_market_correlation()
            return correlation
            
        except Exception as e:
            logger.error(f"Error analyzing market correlation: {e}")
            return 0.5
    
    def _calculate_volatility_index(self) -> float:
        """Calculate market volatility index (0-1 scale) using extended timeframe"""
        try:
            # Use real market volatility data with extended timeframe
            from src.utils.market_data_fetcher import market_data_fetcher
            # Uses default 30 days (720 hours) for accurate volatility
            volatility = market_data_fetcher.get_market_volatility()
            return volatility
            
        except Exception as e:
            logger.error(f"Error calculating volatility index: {e}")
            return 0.5
    
    def _analyze_volume_trends(self) -> float:
        """Analyze volume trends (0-1 scale) using rolling averages"""
        try:
            # Use real volume data with rolling average comparison
            from src.utils.market_data_fetcher import market_data_fetcher
            # Uses default 14 days (336 hours) with 7-day rolling window
            volume_trend = market_data_fetcher.get_volume_trends()
            return volume_trend
            
        except Exception as e:
            logger.error(f"Error analyzing volume trends: {e}")
            return 0.5
    
    def _get_fear_greed_index(self) -> float:
        """Get Fear & Greed Index (0-1 scale)"""
        try:
            # Use real Fear & Greed Index data
            from src.utils.market_data_fetcher import market_data_fetcher
            fear_greed = market_data_fetcher.get_fear_greed_index()
            return fear_greed
            
        except Exception as e:
            logger.error(f"Error getting Fear & Greed Index: {e}")
            return 0.5
    
    def _analyze_market_cap_trend(self) -> float:
        """Analyze total market cap trend (0-1 scale)"""
        try:
            # Use real market cap data
            from src.utils.market_data_fetcher import market_data_fetcher
            market_cap_trend = market_data_fetcher.get_market_cap_trend()
            return market_cap_trend
            
        except Exception as e:
            logger.error(f"Error analyzing market cap trend: {e}")
            return 0.5
    
    def _analyze_market_regime(self, indicators: Dict) -> Dict:
        """Analyze market regime using AI-like logic"""
        try:
            # Calculate weighted score for each regime
            regime_scores = {}
            
            # Bull market score
            bull_score = (
                indicators['btc_trend'] * 0.3 +
                indicators['sol_trend'] * 0.2 +
                indicators['market_correlation'] * 0.2 +
                indicators['volume_trends'] * 0.15 +
                indicators['fear_greed_index'] * 0.15
            )
            regime_scores['bull_market'] = bull_score
            
            # Bear market score
            bear_score = (
                (1 - indicators['btc_trend']) * 0.3 +
                (1 - indicators['sol_trend']) * 0.2 +
                indicators['market_correlation'] * 0.2 +
                (1 - indicators['volume_trends']) * 0.15 +
                (1 - indicators['fear_greed_index']) * 0.15
            )
            regime_scores['bear_market'] = bear_score
            
            # High volatility score
            volatility_score = (
                indicators['volatility_index'] * 0.4 +
                (1 - indicators['market_correlation']) * 0.3 +
                (1 - indicators['volume_trends']) * 0.3
            )
            regime_scores['high_volatility'] = volatility_score
            
            # Sideways market score
            sideways_score = (
                (1 - abs(indicators['btc_trend'] - 0.5)) * 0.3 +
                (1 - abs(indicators['sol_trend'] - 0.5)) * 0.2 +
                (1 - indicators['volatility_index']) * 0.3 +
                (1 - abs(indicators['fear_greed_index'] - 0.5)) * 0.2
            )
            regime_scores['sideways_market'] = sideways_score
            
            # Recovery market score
            recovery_score = (
                indicators['btc_trend'] * 0.25 +
                indicators['sol_trend'] * 0.25 +
                indicators['market_cap_trend'] * 0.25 +
                indicators['volume_trends'] * 0.25
            ) * 0.8  # Recovery is typically moderate
            regime_scores['recovery_market'] = recovery_score
            
            # Find the regime with highest score
            best_regime = max(regime_scores, key=regime_scores.get)
            confidence = regime_scores[best_regime]
            
            # Adjust confidence based on volatility
            if indicators['volatility_index'] > 0.8:
                confidence *= 0.8  # Lower confidence in high volatility
            
            return {
                'regime': best_regime,
                'confidence': min(0.95, confidence),
                'scores': regime_scores
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market regime: {e}")
            return {
                'regime': 'sideways_market',
                'confidence': 0.5,
                'scores': {}
            }
    
    def _get_regime_recommendations(self, regime: str) -> List[str]:
        """Get trading recommendations for specific regime"""
        recommendations = {
            'bull_market': [
                "Increase position sizes for high-quality tokens",
                "Lower quality thresholds to capture more opportunities",
                "Focus on momentum-based entries",
                "Consider higher take-profit targets"
            ],
            'bear_market': [
                "Reduce position sizes significantly",
                "Increase quality thresholds for safety",
                "Focus on defensive tokens only",
                "Use tighter stop-losses",
                "Consider shorter holding periods"
            ],
            'sideways_market': [
                "Maintain standard position sizes",
                "Focus on range-bound trading strategies",
                "Look for tokens with strong fundamentals",
                "Use moderate take-profit targets"
            ],
            'high_volatility': [
                "Minimize position sizes",
                "Use highest quality thresholds only",
                "Avoid trading during extreme volatility",
                "Focus on very liquid tokens only",
                "Use tight stop-losses"
            ],
            'recovery_market': [
                "Gradually increase position sizes",
                "Focus on tokens with strong recovery potential",
                "Use moderate quality thresholds",
                "Monitor for trend continuation"
            ]
        }
        
        return recommendations.get(regime, ["Monitor market conditions closely"])
    
    def _get_default_regime(self) -> Dict:
        """Return default regime when detection fails"""
        return {
            'regime': 'sideways_market',
            'confidence': 0.3,
            'description': 'Default regime - market conditions unclear',
            'strategy': 'neutral',
            'position_multiplier': 1.0,
            'quality_threshold_adjustment': 0,
            'risk_tolerance': 'medium',
            'indicators': {},
            'analysis': {},
            'timestamp': datetime.now().isoformat(),
            'recommendations': ["Monitor market conditions closely"]
        }
    
    def get_regime_insights(self) -> Dict:
        """Get insights about current market regime"""
        regime_data = self.detect_market_regime()
        
        insights = {
            'current_regime': regime_data['regime'],
            'confidence': regime_data['confidence'],
            'strategy_adjustment': regime_data['strategy'],
            'position_adjustment': f"{regime_data['position_multiplier']:.1f}x",
            'quality_threshold_change': f"{regime_data['quality_threshold_adjustment']:+d}",
            'risk_level': regime_data['risk_tolerance'],
            'recommendations': regime_data['recommendations'],
            'market_indicators': regime_data['indicators']
        }
        
        return insights
    
    def should_trade_in_current_regime(self) -> Tuple[bool, str]:
        """Determine if trading should proceed in current regime"""
        regime_data = self.detect_market_regime()
        regime = regime_data['regime']
        confidence = regime_data['confidence']
        
        # High volatility regime - be very cautious
        if regime == 'high_volatility' and confidence > 0.7:
            return False, "High volatility detected - trading paused for safety"
        
        # Bear market with high confidence - reduce trading
        if regime == 'bear_market' and confidence > 0.8:
            return False, "Strong bear market detected - trading paused"
        
        # All other regimes allow trading
        return True, f"Trading allowed in {regime} regime (confidence: {confidence:.2f})"

# Global instance
ai_market_regime_detector = AIMarketRegimeDetector()
