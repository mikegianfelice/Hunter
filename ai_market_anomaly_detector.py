#!/usr/bin/env python3
"""
AI-Powered Market Anomaly Detector for Sustainable Trading Bot
Detects unusual market conditions and opportunities to capture unique opportunities and avoid unusual risks
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

class AIMarketAnomalyDetector:
    def __init__(self):
        self.anomaly_cache = {}
        self.cache_duration = 300  # 5 minutes cache for anomaly detection
        self.anomaly_history = deque(maxlen=1000)
        self.opportunity_history = deque(maxlen=500)
        self.risk_history = deque(maxlen=500)
        
        # Anomaly types configuration
        self.anomaly_types = {
            'price_anomaly': {
                'name': 'Price Anomaly',
                'characteristics': ['unusual_price_movement', 'extreme_volatility', 'price_gaps'],
                'severity_levels': ['minor', 'moderate', 'major', 'extreme'],
                'detection_threshold': 0.7
            },
            'volume_anomaly': {
                'name': 'Volume Anomaly',
                'characteristics': ['unusual_volume_spike', 'volume_drop', 'irregular_patterns'],
                'severity_levels': ['minor', 'moderate', 'major', 'extreme'],
                'detection_threshold': 0.6
            },
            'liquidity_anomaly': {
                'name': 'Liquidity Anomaly',
                'characteristics': ['liquidity_drain', 'spread_widening', 'depth_changes'],
                'severity_levels': ['minor', 'moderate', 'major', 'extreme'],
                'detection_threshold': 0.8
            },
            'sentiment_anomaly': {
                'name': 'Sentiment Anomaly',
                'characteristics': ['sentiment_shift', 'social_media_spike', 'news_impact'],
                'severity_levels': ['minor', 'moderate', 'major', 'extreme'],
                'detection_threshold': 0.5
            },
            'arbitrage_opportunity': {
                'name': 'Arbitrage Opportunity',
                'characteristics': ['price_discrepancy', 'cross_exchange_spread', 'temporal_arbitrage'],
                'severity_levels': ['minor', 'moderate', 'major', 'extreme'],
                'detection_threshold': 0.9
            },
            'manipulation_pattern': {
                'name': 'Manipulation Pattern',
                'characteristics': ['wash_trading', 'pump_dump', 'spoofing', 'layering'],
                'severity_levels': ['minor', 'moderate', 'major', 'extreme'],
                'detection_threshold': 0.8
            }
        }
        
        # Anomaly detection factors (must sum to 1.0)
        self.detection_factors = {
            'price_analysis': 0.25,  # 25% weight for price analysis
            'volume_analysis': 0.20,  # 20% weight for volume analysis
            'liquidity_analysis': 0.20,  # 20% weight for liquidity analysis
            'sentiment_analysis': 0.15,  # 15% weight for sentiment analysis
            'pattern_analysis': 0.10,  # 10% weight for pattern analysis
            'correlation_analysis': 0.10  # 10% weight for correlation analysis
        }
        
        # Anomaly severity thresholds
        self.minor_anomaly_threshold = 0.3  # 30% minor anomaly
        self.moderate_anomaly_threshold = 0.5  # 50% moderate anomaly
        self.major_anomaly_threshold = 0.7  # 70% major anomaly
        self.extreme_anomaly_threshold = 0.9  # 90% extreme anomaly
        
        # Opportunity detection thresholds
        self.high_opportunity_threshold = 0.8  # 80% high opportunity
        self.medium_opportunity_threshold = 0.6  # 60% medium opportunity
        self.low_opportunity_threshold = 0.4  # 40% low opportunity
        
        # Risk detection thresholds
        self.critical_risk_threshold = 0.9  # 90% critical risk
        self.high_risk_threshold = 0.7  # 70% high risk
        self.medium_risk_threshold = 0.5  # 50% medium risk
        self.low_risk_threshold = 0.3  # 30% low risk
        
        # Anomaly persistence thresholds
        self.persistent_anomaly_threshold = 0.8  # 80% persistent anomaly
        self.temporary_anomaly_threshold = 0.4  # 40% temporary anomaly
        
        # Market correlation thresholds
        self.strong_correlation_threshold = 0.7  # 70% strong correlation
        self.weak_correlation_threshold = 0.3  # 30% weak correlation
        self.negative_correlation_threshold = -0.3  # -30% negative correlation
    
    def detect_market_anomalies(self, token: Dict, market_data: Dict, historical_data: Dict) -> Dict:
        """
        Detect market anomalies and unusual conditions
        Returns comprehensive anomaly analysis with opportunity and risk assessment
        """
        try:
            symbol = token.get("symbol", "UNKNOWN")
            cache_key = f"anomaly_{symbol}_{market_data.get('timestamp', 'current')}"
            
            # Check cache
            if cache_key in self.anomaly_cache:
                cached_data = self.anomaly_cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).total_seconds() < self.cache_duration:
                    logger.debug(f"Using cached anomaly detection for {symbol}")
                    return cached_data['anomaly_data']
            
            # Analyze anomaly components
            price_anomaly_analysis = self._analyze_price_anomalies(token, market_data, historical_data)
            volume_anomaly_analysis = self._analyze_volume_anomalies(token, market_data, historical_data)
            liquidity_anomaly_analysis = self._analyze_liquidity_anomalies(token, market_data, historical_data)
            sentiment_anomaly_analysis = self._analyze_sentiment_anomalies(token, market_data, historical_data)
            pattern_anomaly_analysis = self._analyze_pattern_anomalies(token, market_data, historical_data)
            correlation_anomaly_analysis = self._analyze_correlation_anomalies(token, market_data, historical_data)
            
            # Calculate overall anomaly score
            anomaly_score = self._calculate_anomaly_score(
                price_anomaly_analysis, volume_anomaly_analysis, liquidity_anomaly_analysis,
                sentiment_anomaly_analysis, pattern_anomaly_analysis, correlation_anomaly_analysis
            )
            
            # Determine anomaly severity
            anomaly_severity = self._determine_anomaly_severity(anomaly_score)
            
            # Detect opportunities
            opportunities = self._detect_opportunities(
                price_anomaly_analysis, volume_anomaly_analysis, liquidity_anomaly_analysis,
                sentiment_anomaly_analysis, pattern_anomaly_analysis, correlation_anomaly_analysis
            )
            
            # Detect risks
            risks = self._detect_risks(
                price_anomaly_analysis, volume_anomaly_analysis, liquidity_anomaly_analysis,
                sentiment_anomaly_analysis, pattern_anomaly_analysis, correlation_anomaly_analysis
            )
            
            # Calculate anomaly persistence
            anomaly_persistence = self._calculate_anomaly_persistence(
                anomaly_score, market_data, historical_data
            )
            
            # Generate anomaly recommendations
            anomaly_recommendations = self._generate_anomaly_recommendations(
                anomaly_severity, opportunities, risks, anomaly_persistence
            )
            
            # Generate anomaly insights
            anomaly_insights = self._generate_anomaly_insights(
                anomaly_score, anomaly_severity, opportunities, risks, anomaly_persistence
            )
            
            result = {
                'anomaly_score': anomaly_score,
                'anomaly_severity': anomaly_severity,
                'opportunities': opportunities,
                'risks': risks,
                'anomaly_persistence': anomaly_persistence,
                'price_anomaly_analysis': price_anomaly_analysis,
                'volume_anomaly_analysis': volume_anomaly_analysis,
                'liquidity_anomaly_analysis': liquidity_anomaly_analysis,
                'sentiment_anomaly_analysis': sentiment_anomaly_analysis,
                'pattern_anomaly_analysis': pattern_anomaly_analysis,
                'correlation_anomaly_analysis': correlation_anomaly_analysis,
                'anomaly_recommendations': anomaly_recommendations,
                'anomaly_insights': anomaly_insights,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Cache the result
            self.anomaly_cache[cache_key] = {'timestamp': datetime.now(), 'anomaly_data': result}
            
            logger.info(f"🔍 Anomaly detection for {symbol}: {anomaly_severity} (score: {anomaly_score:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Anomaly detection failed for {token.get('symbol', 'UNKNOWN')}: {e}")
            return self._get_default_anomaly_analysis(token, market_data, historical_data)
    
    def _analyze_price_anomalies(self, token: Dict, market_data: Dict, historical_data: Dict) -> Dict:
        """Analyze price anomalies"""
        try:
            # Extract price data
            current_price = market_data.get('current_price', 100)
            price_24h_ago = historical_data.get('price_24h_ago', 100)
            price_7d_ago = historical_data.get('price_7d_ago', 100)
            price_30d_ago = historical_data.get('price_30d_ago', 100)
            
            # Calculate price changes
            change_24h = (current_price - price_24h_ago) / price_24h_ago if price_24h_ago > 0 else 0
            change_7d = (current_price - price_7d_ago) / price_7d_ago if price_7d_ago > 0 else 0
            change_30d = (current_price - price_30d_ago) / price_30d_ago if price_30d_ago > 0 else 0
            
            # Calculate price volatility
            price_volatility = abs(change_24h)
            
            # Detect price anomalies
            price_anomaly_score = 0.0
            anomaly_indicators = []
            
            # Extreme price movement
            if abs(change_24h) > 0.5:  # 50% price change
                price_anomaly_score += 0.4
                anomaly_indicators.append('extreme_price_movement')
            elif abs(change_24h) > 0.2:  # 20% price change
                price_anomaly_score += 0.2
                anomaly_indicators.append('significant_price_movement')
            
            # Price gap detection
            if abs(change_24h) > 0.1 and abs(change_7d) < 0.05:  # 10% change in 24h but stable over 7d
                price_anomaly_score += 0.3
                anomaly_indicators.append('price_gap')
            
            # Volatility spike
            if price_volatility > 0.3:  # 30% volatility
                price_anomaly_score += 0.3
                anomaly_indicators.append('volatility_spike')
            
            # Determine anomaly severity
            if price_anomaly_score > 0.8:  # 80% anomaly score
                anomaly_severity = "extreme"
                anomaly_characteristics = "very_unusual"
            elif price_anomaly_score > 0.6:  # 60% anomaly score
                anomaly_severity = "major"
                anomaly_characteristics = "highly_unusual"
            elif price_anomaly_score > 0.4:  # 40% anomaly score
                anomaly_severity = "moderate"
                anomaly_characteristics = "somewhat_unusual"
            elif price_anomaly_score > 0.2:  # 20% anomaly score
                anomaly_severity = "minor"
                anomaly_characteristics = "slightly_unusual"
            else:
                anomaly_severity = "none"
                anomaly_characteristics = "normal"
            
            return {
                'price_anomaly_score': price_anomaly_score,
                'anomaly_severity': anomaly_severity,
                'anomaly_characteristics': anomaly_characteristics,
                'anomaly_indicators': anomaly_indicators,
                'price_volatility': price_volatility,
                'change_24h': change_24h,
                'change_7d': change_7d,
                'change_30d': change_30d
            }
            
        except Exception:
            return {
                'price_anomaly_score': 0.0,
                'anomaly_severity': 'none',
                'anomaly_characteristics': 'normal',
                'anomaly_indicators': [],
                'price_volatility': 0.0,
                'change_24h': 0.0,
                'change_7d': 0.0,
                'change_30d': 0.0
            }
    
    def _analyze_volume_anomalies(self, token: Dict, market_data: Dict, historical_data: Dict) -> Dict:
        """Analyze volume anomalies"""
        try:
            # Extract volume data
            current_volume = market_data.get('current_volume', 1000000)
            volume_24h_ago = historical_data.get('volume_24h_ago', 1000000)
            volume_7d_ago = historical_data.get('volume_7d_ago', 1000000)
            volume_30d_ago = historical_data.get('volume_30d_ago', 1000000)
            
            # Calculate volume ratios
            volume_ratio_24h = current_volume / volume_24h_ago if volume_24h_ago > 0 else 1.0
            volume_ratio_7d = current_volume / volume_7d_ago if volume_7d_ago > 0 else 1.0
            volume_ratio_30d = current_volume / volume_30d_ago if volume_30d_ago > 0 else 1.0
            
            # Calculate volume anomaly score
            volume_anomaly_score = 0.0
            anomaly_indicators = []
            
            # Volume spike detection
            if volume_ratio_24h > 5.0:  # 5x volume spike
                volume_anomaly_score += 0.5
                anomaly_indicators.append('extreme_volume_spike')
            elif volume_ratio_24h > 3.0:  # 3x volume spike
                volume_anomaly_score += 0.3
                anomaly_indicators.append('significant_volume_spike')
            elif volume_ratio_24h > 2.0:  # 2x volume spike
                volume_anomaly_score += 0.2
                anomaly_indicators.append('moderate_volume_spike')
            
            # Volume drop detection
            if volume_ratio_24h < 0.2:  # 80% volume drop
                volume_anomaly_score += 0.4
                anomaly_indicators.append('extreme_volume_drop')
            elif volume_ratio_24h < 0.5:  # 50% volume drop
                volume_anomaly_score += 0.2
                anomaly_indicators.append('significant_volume_drop')
            
            # Irregular volume patterns
            if volume_ratio_24h > 2.0 and volume_ratio_7d < 1.5:  # 24h spike but 7d normal
                volume_anomaly_score += 0.3
                anomaly_indicators.append('irregular_volume_pattern')
            
            # Determine anomaly severity
            if volume_anomaly_score > 0.8:  # 80% anomaly score
                anomaly_severity = "extreme"
                anomaly_characteristics = "very_unusual"
            elif volume_anomaly_score > 0.6:  # 60% anomaly score
                anomaly_severity = "major"
                anomaly_characteristics = "highly_unusual"
            elif volume_anomaly_score > 0.4:  # 40% anomaly score
                anomaly_severity = "moderate"
                anomaly_characteristics = "somewhat_unusual"
            elif volume_anomaly_score > 0.2:  # 20% anomaly score
                anomaly_severity = "minor"
                anomaly_characteristics = "slightly_unusual"
            else:
                anomaly_severity = "none"
                anomaly_characteristics = "normal"
            
            return {
                'volume_anomaly_score': volume_anomaly_score,
                'anomaly_severity': anomaly_severity,
                'anomaly_characteristics': anomaly_characteristics,
                'anomaly_indicators': anomaly_indicators,
                'volume_ratio_24h': volume_ratio_24h,
                'volume_ratio_7d': volume_ratio_7d,
                'volume_ratio_30d': volume_ratio_30d
            }
            
        except Exception:
            return {
                'volume_anomaly_score': 0.0,
                'anomaly_severity': 'none',
                'anomaly_characteristics': 'normal',
                'anomaly_indicators': [],
                'volume_ratio_24h': 1.0,
                'volume_ratio_7d': 1.0,
                'volume_ratio_30d': 1.0
            }
    
    def _analyze_liquidity_anomalies(self, token: Dict, market_data: Dict, historical_data: Dict) -> Dict:
        """Analyze liquidity anomalies"""
        try:
            # Extract liquidity data
            current_liquidity = market_data.get('liquidity', 100000)
            liquidity_24h_ago = historical_data.get('liquidity_24h_ago', 100000)
            liquidity_7d_ago = historical_data.get('liquidity_7d_ago', 100000)
            
            # Calculate liquidity changes
            liquidity_change_24h = (current_liquidity - liquidity_24h_ago) / liquidity_24h_ago if liquidity_24h_ago > 0 else 0
            liquidity_change_7d = (current_liquidity - liquidity_7d_ago) / liquidity_7d_ago if liquidity_7d_ago > 0 else 0
            
            # Calculate liquidity anomaly score
            liquidity_anomaly_score = 0.0
            anomaly_indicators = []
            
            # Liquidity drain detection
            if liquidity_change_24h < -0.5:  # 50% liquidity drain
                liquidity_anomaly_score += 0.6
                anomaly_indicators.append('extreme_liquidity_drain')
            elif liquidity_change_24h < -0.3:  # 30% liquidity drain
                liquidity_anomaly_score += 0.4
                anomaly_indicators.append('significant_liquidity_drain')
            elif liquidity_change_24h < -0.1:  # 10% liquidity drain
                liquidity_anomaly_score += 0.2
                anomaly_indicators.append('moderate_liquidity_drain')
            
            # Liquidity spike detection
            if liquidity_change_24h > 2.0:  # 200% liquidity spike
                liquidity_anomaly_score += 0.4
                anomaly_indicators.append('extreme_liquidity_spike')
            elif liquidity_change_24h > 1.0:  # 100% liquidity spike
                liquidity_anomaly_score += 0.2
                anomaly_indicators.append('significant_liquidity_spike')
            
            # Spread widening detection
            current_spread = market_data.get('spread', 0.01)
            spread_24h_ago = historical_data.get('spread_24h_ago', 0.01)
            spread_change = (current_spread - spread_24h_ago) / spread_24h_ago if spread_24h_ago > 0 else 0
            
            if spread_change > 2.0:  # 200% spread increase
                liquidity_anomaly_score += 0.3
                anomaly_indicators.append('extreme_spread_widening')
            elif spread_change > 1.0:  # 100% spread increase
                liquidity_anomaly_score += 0.2
                anomaly_indicators.append('significant_spread_widening')
            
            # Determine anomaly severity
            if liquidity_anomaly_score > 0.8:  # 80% anomaly score
                anomaly_severity = "extreme"
                anomaly_characteristics = "very_unusual"
            elif liquidity_anomaly_score > 0.6:  # 60% anomaly score
                anomaly_severity = "major"
                anomaly_characteristics = "highly_unusual"
            elif liquidity_anomaly_score > 0.4:  # 40% anomaly score
                anomaly_severity = "moderate"
                anomaly_characteristics = "somewhat_unusual"
            elif liquidity_anomaly_score > 0.2:  # 20% anomaly score
                anomaly_severity = "minor"
                anomaly_characteristics = "slightly_unusual"
            else:
                anomaly_severity = "none"
                anomaly_characteristics = "normal"
            
            return {
                'liquidity_anomaly_score': liquidity_anomaly_score,
                'anomaly_severity': anomaly_severity,
                'anomaly_characteristics': anomaly_characteristics,
                'anomaly_indicators': anomaly_indicators,
                'liquidity_change_24h': liquidity_change_24h,
                'liquidity_change_7d': liquidity_change_7d,
                'spread_change': spread_change
            }
            
        except Exception:
            return {
                'liquidity_anomaly_score': 0.0,
                'anomaly_severity': 'none',
                'anomaly_characteristics': 'normal',
                'anomaly_indicators': [],
                'liquidity_change_24h': 0.0,
                'liquidity_change_7d': 0.0,
                'spread_change': 0.0
            }
    
    def _analyze_sentiment_anomalies(self, token: Dict, market_data: Dict, historical_data: Dict) -> Dict:
        """Analyze sentiment anomalies"""
        try:
            # Extract sentiment data
            current_sentiment = market_data.get('current_sentiment', 0.5)
            sentiment_24h_ago = historical_data.get('sentiment_24h_ago', 0.5)
            sentiment_7d_ago = historical_data.get('sentiment_7d_ago', 0.5)
            
            # Calculate sentiment changes
            sentiment_change_24h = current_sentiment - sentiment_24h_ago
            sentiment_change_7d = current_sentiment - sentiment_7d_ago
            
            # Calculate sentiment anomaly score
            sentiment_anomaly_score = 0.0
            anomaly_indicators = []
            
            # Sentiment shift detection
            if abs(sentiment_change_24h) > 0.4:  # 40% sentiment shift
                sentiment_anomaly_score += 0.5
                anomaly_indicators.append('extreme_sentiment_shift')
            elif abs(sentiment_change_24h) > 0.2:  # 20% sentiment shift
                sentiment_anomaly_score += 0.3
                anomaly_indicators.append('significant_sentiment_shift')
            elif abs(sentiment_change_24h) > 0.1:  # 10% sentiment shift
                sentiment_anomaly_score += 0.2
                anomaly_indicators.append('moderate_sentiment_shift')
            
            # Social media spike detection
            social_media_activity = market_data.get('social_media_activity', 0.5)
            if social_media_activity > 0.8:  # 80% social media activity
                sentiment_anomaly_score += 0.3
                anomaly_indicators.append('social_media_spike')
            elif social_media_activity > 0.6:  # 60% social media activity
                sentiment_anomaly_score += 0.2
                anomaly_indicators.append('elevated_social_media_activity')
            
            # News impact detection
            news_impact = market_data.get('news_impact', 0.5)
            if news_impact > 0.8:  # 80% news impact
                sentiment_anomaly_score += 0.4
                anomaly_indicators.append('high_news_impact')
            elif news_impact > 0.6:  # 60% news impact
                sentiment_anomaly_score += 0.2
                anomaly_indicators.append('moderate_news_impact')
            
            # Determine anomaly severity
            if sentiment_anomaly_score > 0.8:  # 80% anomaly score
                anomaly_severity = "extreme"
                anomaly_characteristics = "very_unusual"
            elif sentiment_anomaly_score > 0.6:  # 60% anomaly score
                anomaly_severity = "major"
                anomaly_characteristics = "highly_unusual"
            elif sentiment_anomaly_score > 0.4:  # 40% anomaly score
                anomaly_severity = "moderate"
                anomaly_characteristics = "somewhat_unusual"
            elif sentiment_anomaly_score > 0.2:  # 20% anomaly score
                anomaly_severity = "minor"
                anomaly_characteristics = "slightly_unusual"
            else:
                anomaly_severity = "none"
                anomaly_characteristics = "normal"
            
            return {
                'sentiment_anomaly_score': sentiment_anomaly_score,
                'anomaly_severity': anomaly_severity,
                'anomaly_characteristics': anomaly_characteristics,
                'anomaly_indicators': anomaly_indicators,
                'sentiment_change_24h': sentiment_change_24h,
                'sentiment_change_7d': sentiment_change_7d,
                'social_media_activity': social_media_activity,
                'news_impact': news_impact
            }
            
        except Exception:
            return {
                'sentiment_anomaly_score': 0.0,
                'anomaly_severity': 'none',
                'anomaly_characteristics': 'normal',
                'anomaly_indicators': [],
                'sentiment_change_24h': 0.0,
                'sentiment_change_7d': 0.0,
                'social_media_activity': 0.5,
                'news_impact': 0.5
            }
    
    def _analyze_pattern_anomalies(self, token: Dict, market_data: Dict, historical_data: Dict) -> Dict:
        """Analyze pattern anomalies"""
        try:
            # Extract pattern data
            current_price = market_data.get('current_price', 100)
            price_patterns = market_data.get('price_patterns', [])
            volume_patterns = market_data.get('volume_patterns', [])
            
            # Calculate pattern anomaly score
            pattern_anomaly_score = 0.0
            anomaly_indicators = []
            
            # Unusual price patterns
            if 'head_and_shoulders' in price_patterns:
                pattern_anomaly_score += 0.3
                anomaly_indicators.append('head_and_shoulders_pattern')
            
            if 'double_top' in price_patterns:
                pattern_anomaly_score += 0.2
                anomaly_indicators.append('double_top_pattern')
            
            if 'double_bottom' in price_patterns:
                pattern_anomaly_score += 0.2
                anomaly_indicators.append('double_bottom_pattern')
            
            # Unusual volume patterns
            if 'volume_spike' in volume_patterns:
                pattern_anomaly_score += 0.3
                anomaly_indicators.append('volume_spike_pattern')
            
            if 'volume_drop' in volume_patterns:
                pattern_anomaly_score += 0.2
                anomaly_indicators.append('volume_drop_pattern')
            
            # Manipulation patterns
            if 'wash_trading' in price_patterns:
                pattern_anomaly_score += 0.5
                anomaly_indicators.append('wash_trading_pattern')
            
            if 'pump_dump' in price_patterns:
                pattern_anomaly_score += 0.4
                anomaly_indicators.append('pump_dump_pattern')
            
            if 'spoofing' in price_patterns:
                pattern_anomaly_score += 0.4
                anomaly_indicators.append('spoofing_pattern')
            
            # Determine anomaly severity
            if pattern_anomaly_score > 0.8:  # 80% anomaly score
                anomaly_severity = "extreme"
                anomaly_characteristics = "very_unusual"
            elif pattern_anomaly_score > 0.6:  # 60% anomaly score
                anomaly_severity = "major"
                anomaly_characteristics = "highly_unusual"
            elif pattern_anomaly_score > 0.4:  # 40% anomaly score
                anomaly_severity = "moderate"
                anomaly_characteristics = "somewhat_unusual"
            elif pattern_anomaly_score > 0.2:  # 20% anomaly score
                anomaly_severity = "minor"
                anomaly_characteristics = "slightly_unusual"
            else:
                anomaly_severity = "none"
                anomaly_characteristics = "normal"
            
            return {
                'pattern_anomaly_score': pattern_anomaly_score,
                'anomaly_severity': anomaly_severity,
                'anomaly_characteristics': anomaly_characteristics,
                'anomaly_indicators': anomaly_indicators,
                'price_patterns': price_patterns,
                'volume_patterns': volume_patterns
            }
            
        except Exception:
            return {
                'pattern_anomaly_score': 0.0,
                'anomaly_severity': 'none',
                'anomaly_characteristics': 'normal',
                'anomaly_indicators': [],
                'price_patterns': [],
                'volume_patterns': []
            }
    
    def _analyze_correlation_anomalies(self, token: Dict, market_data: Dict, historical_data: Dict) -> Dict:
        """Analyze correlation anomalies"""
        try:
            # Extract correlation data
            btc_correlation = market_data.get('btc_correlation', 0.5)
            eth_correlation = market_data.get('eth_correlation', 0.5)
            market_correlation = market_data.get('market_correlation', 0.5)
            
            # Calculate correlation anomaly score
            correlation_anomaly_score = 0.0
            anomaly_indicators = []
            
            # Correlation breakdown detection
            if btc_correlation < 0.2:  # 20% BTC correlation
                correlation_anomaly_score += 0.3
                anomaly_indicators.append('btc_correlation_breakdown')
            
            if eth_correlation < 0.2:  # 20% ETH correlation
                correlation_anomaly_score += 0.3
                anomaly_indicators.append('eth_correlation_breakdown')
            
            if market_correlation < 0.2:  # 20% market correlation
                correlation_anomaly_score += 0.4
                anomaly_indicators.append('market_correlation_breakdown')
            
            # Negative correlation detection
            if btc_correlation < -0.3:  # -30% BTC correlation
                correlation_anomaly_score += 0.4
                anomaly_indicators.append('negative_btc_correlation')
            
            if eth_correlation < -0.3:  # -30% ETH correlation
                correlation_anomaly_score += 0.4
                anomaly_indicators.append('negative_eth_correlation')
            
            # Determine anomaly severity
            if correlation_anomaly_score > 0.8:  # 80% anomaly score
                anomaly_severity = "extreme"
                anomaly_characteristics = "very_unusual"
            elif correlation_anomaly_score > 0.6:  # 60% anomaly score
                anomaly_severity = "major"
                anomaly_characteristics = "highly_unusual"
            elif correlation_anomaly_score > 0.4:  # 40% anomaly score
                anomaly_severity = "moderate"
                anomaly_characteristics = "somewhat_unusual"
            elif correlation_anomaly_score > 0.2:  # 20% anomaly score
                anomaly_severity = "minor"
                anomaly_characteristics = "slightly_unusual"
            else:
                anomaly_severity = "none"
                anomaly_characteristics = "normal"
            
            return {
                'correlation_anomaly_score': correlation_anomaly_score,
                'anomaly_severity': anomaly_severity,
                'anomaly_characteristics': anomaly_characteristics,
                'anomaly_indicators': anomaly_indicators,
                'btc_correlation': btc_correlation,
                'eth_correlation': eth_correlation,
                'market_correlation': market_correlation
            }
            
        except Exception:
            return {
                'correlation_anomaly_score': 0.0,
                'anomaly_severity': 'none',
                'anomaly_characteristics': 'normal',
                'anomaly_indicators': [],
                'btc_correlation': 0.5,
                'eth_correlation': 0.5,
                'market_correlation': 0.5
            }
    
    def _calculate_anomaly_score(self, price_anomaly_analysis: Dict, volume_anomaly_analysis: Dict,
                                liquidity_anomaly_analysis: Dict, sentiment_anomaly_analysis: Dict,
                                pattern_anomaly_analysis: Dict, correlation_anomaly_analysis: Dict) -> float:
        """Calculate overall anomaly score"""
        try:
            # Weight the individual anomaly scores
            anomaly_score = (
                price_anomaly_analysis.get('price_anomaly_score', 0.0) * self.detection_factors['price_analysis'] +
                volume_anomaly_analysis.get('volume_anomaly_score', 0.0) * self.detection_factors['volume_analysis'] +
                liquidity_anomaly_analysis.get('liquidity_anomaly_score', 0.0) * self.detection_factors['liquidity_analysis'] +
                sentiment_anomaly_analysis.get('sentiment_anomaly_score', 0.0) * self.detection_factors['sentiment_analysis'] +
                pattern_anomaly_analysis.get('pattern_anomaly_score', 0.0) * self.detection_factors['pattern_analysis'] +
                correlation_anomaly_analysis.get('correlation_anomaly_score', 0.0) * self.detection_factors['correlation_analysis']
            )
            
            return max(0.0, min(1.0, anomaly_score))
            
        except Exception:
            return 0.0
    
    def _determine_anomaly_severity(self, anomaly_score: float) -> str:
        """Determine anomaly severity level"""
        try:
            if anomaly_score > self.extreme_anomaly_threshold:  # 90% anomaly score
                return "extreme"
            elif anomaly_score > self.major_anomaly_threshold:  # 70% anomaly score
                return "major"
            elif anomaly_score > self.moderate_anomaly_threshold:  # 50% anomaly score
                return "moderate"
            elif anomaly_score > self.minor_anomaly_threshold:  # 30% anomaly score
                return "minor"
            else:
                return "none"
                
        except Exception:
            return "none"
    
    def _detect_opportunities(self, price_anomaly_analysis: Dict, volume_anomaly_analysis: Dict,
                            liquidity_anomaly_analysis: Dict, sentiment_anomaly_analysis: Dict,
                            pattern_anomaly_analysis: Dict, correlation_anomaly_analysis: Dict) -> List[Dict]:
        """Detect trading opportunities from anomalies"""
        opportunities = []
        
        try:
            # Price opportunity detection
            if price_anomaly_analysis.get('anomaly_severity') in ['moderate', 'major', 'extreme']:
                opportunities.append({
                    'type': 'price_opportunity',
                    'description': 'Unusual price movement detected',
                    'severity': price_anomaly_analysis.get('anomaly_severity'),
                    'confidence': price_anomaly_analysis.get('price_anomaly_score', 0.0),
                    'action': 'monitor_price_movement'
                })
            
            # Volume opportunity detection
            if volume_anomaly_analysis.get('anomaly_severity') in ['moderate', 'major', 'extreme']:
                opportunities.append({
                    'type': 'volume_opportunity',
                    'description': 'Unusual volume pattern detected',
                    'severity': volume_anomaly_analysis.get('anomaly_severity'),
                    'confidence': volume_anomaly_analysis.get('volume_anomaly_score', 0.0),
                    'action': 'monitor_volume_patterns'
                })
            
            # Liquidity opportunity detection
            if liquidity_anomaly_analysis.get('anomaly_severity') in ['moderate', 'major', 'extreme']:
                opportunities.append({
                    'type': 'liquidity_opportunity',
                    'description': 'Unusual liquidity conditions detected',
                    'severity': liquidity_anomaly_analysis.get('anomaly_severity'),
                    'confidence': liquidity_anomaly_analysis.get('liquidity_anomaly_score', 0.0),
                    'action': 'monitor_liquidity_conditions'
                })
            
            # Sentiment opportunity detection
            if sentiment_anomaly_analysis.get('anomaly_severity') in ['moderate', 'major', 'extreme']:
                opportunities.append({
                    'type': 'sentiment_opportunity',
                    'description': 'Unusual sentiment shift detected',
                    'severity': sentiment_anomaly_analysis.get('anomaly_severity'),
                    'confidence': sentiment_anomaly_analysis.get('sentiment_anomaly_score', 0.0),
                    'action': 'monitor_sentiment_changes'
                })
            
            # Pattern opportunity detection
            if pattern_anomaly_analysis.get('anomaly_severity') in ['moderate', 'major', 'extreme']:
                opportunities.append({
                    'type': 'pattern_opportunity',
                    'description': 'Unusual pattern detected',
                    'severity': pattern_anomaly_analysis.get('anomaly_severity'),
                    'confidence': pattern_anomaly_analysis.get('pattern_anomaly_score', 0.0),
                    'action': 'monitor_pattern_development'
                })
            
            # Correlation opportunity detection
            if correlation_anomaly_analysis.get('anomaly_severity') in ['moderate', 'major', 'extreme']:
                opportunities.append({
                    'type': 'correlation_opportunity',
                    'description': 'Unusual correlation breakdown detected',
                    'severity': correlation_anomaly_analysis.get('anomaly_severity'),
                    'confidence': correlation_anomaly_analysis.get('correlation_anomaly_score', 0.0),
                    'action': 'monitor_correlation_changes'
                })
            
        except Exception:
            opportunities.append({
                'type': 'general_opportunity',
                'description': 'Monitor market conditions',
                'severity': 'minor',
                'confidence': 0.5,
                'action': 'monitor_general_conditions'
            })
        
        return opportunities
    
    def _detect_risks(self, price_anomaly_analysis: Dict, volume_anomaly_analysis: Dict,
                     liquidity_anomaly_analysis: Dict, sentiment_anomaly_analysis: Dict,
                     pattern_anomaly_analysis: Dict, correlation_anomaly_analysis: Dict) -> List[Dict]:
        """Detect trading risks from anomalies"""
        risks = []
        
        try:
            # Price risk detection
            if price_anomaly_analysis.get('anomaly_severity') in ['major', 'extreme']:
                risks.append({
                    'type': 'price_risk',
                    'description': 'Extreme price movement risk',
                    'severity': price_anomaly_analysis.get('anomaly_severity'),
                    'confidence': price_anomaly_analysis.get('price_anomaly_score', 0.0),
                    'action': 'avoid_trading'
                })
            
            # Volume risk detection
            if volume_anomaly_analysis.get('anomaly_severity') in ['major', 'extreme']:
                risks.append({
                    'type': 'volume_risk',
                    'description': 'Extreme volume anomaly risk',
                    'severity': volume_anomaly_analysis.get('anomaly_severity'),
                    'confidence': volume_anomaly_analysis.get('volume_anomaly_score', 0.0),
                    'action': 'reduce_position_sizes'
                })
            
            # Liquidity risk detection
            if liquidity_anomaly_analysis.get('anomaly_severity') in ['major', 'extreme']:
                risks.append({
                    'type': 'liquidity_risk',
                    'description': 'Extreme liquidity anomaly risk',
                    'severity': liquidity_anomaly_analysis.get('anomaly_severity'),
                    'confidence': liquidity_anomaly_analysis.get('liquidity_anomaly_score', 0.0),
                    'action': 'avoid_trading'
                })
            
            # Sentiment risk detection
            if sentiment_anomaly_analysis.get('anomaly_severity') in ['major', 'extreme']:
                risks.append({
                    'type': 'sentiment_risk',
                    'description': 'Extreme sentiment anomaly risk',
                    'severity': sentiment_anomaly_analysis.get('anomaly_severity'),
                    'confidence': sentiment_anomaly_analysis.get('sentiment_anomaly_score', 0.0),
                    'action': 'monitor_closely'
                })
            
            # Pattern risk detection
            if pattern_anomaly_analysis.get('anomaly_severity') in ['major', 'extreme']:
                risks.append({
                    'type': 'pattern_risk',
                    'description': 'Extreme pattern anomaly risk',
                    'severity': pattern_anomaly_analysis.get('anomaly_severity'),
                    'confidence': pattern_anomaly_analysis.get('pattern_anomaly_score', 0.0),
                    'action': 'avoid_trading'
                })
            
            # Correlation risk detection
            if correlation_anomaly_analysis.get('anomaly_severity') in ['major', 'extreme']:
                risks.append({
                    'type': 'correlation_risk',
                    'description': 'Extreme correlation anomaly risk',
                    'severity': correlation_anomaly_analysis.get('anomaly_severity'),
                    'confidence': correlation_anomaly_analysis.get('correlation_anomaly_score', 0.0),
                    'action': 'monitor_closely'
                })
            
        except Exception:
            risks.append({
                'type': 'general_risk',
                'description': 'Monitor market conditions',
                'severity': 'minor',
                'confidence': 0.5,
                'action': 'monitor_general_conditions'
            })
        
        return risks
    
    def _calculate_anomaly_persistence(self, anomaly_score: float, market_data: Dict, historical_data: Dict) -> str:
        """Calculate anomaly persistence"""
        try:
            # Mock calculation - in real implementation, analyze historical anomaly patterns
            if anomaly_score > 0.8:  # 80% anomaly score
                return "persistent"
            elif anomaly_score > 0.5:  # 50% anomaly score
                return "moderate"
            else:
                return "temporary"
                
        except Exception:
            return "temporary"
    
    def _generate_anomaly_recommendations(self, anomaly_severity: str, opportunities: List[Dict],
                                        risks: List[Dict], anomaly_persistence: str) -> List[str]:
        """Generate anomaly-based recommendations"""
        recommendations = []
        
        try:
            # Severity-based recommendations
            if anomaly_severity == "extreme":
                recommendations.append("EXTREME ANOMALY DETECTED - Avoid trading")
                recommendations.append("Monitor market conditions closely")
                recommendations.append("Consider exiting positions")
            elif anomaly_severity == "major":
                recommendations.append("MAJOR ANOMALY DETECTED - Reduce position sizes")
                recommendations.append("Monitor for further anomalies")
                recommendations.append("Consider risk management")
            elif anomaly_severity == "moderate":
                recommendations.append("MODERATE ANOMALY DETECTED - Monitor closely")
                recommendations.append("Consider reducing position sizes")
                recommendations.append("Monitor for opportunity")
            elif anomaly_severity == "minor":
                recommendations.append("MINOR ANOMALY DETECTED - Monitor for changes")
                recommendations.append("Consider normal trading with caution")
            else:
                recommendations.append("No significant anomalies detected")
                recommendations.append("Continue normal trading operations")
            
            # Opportunity recommendations
            if opportunities:
                recommendations.append(f"Trading opportunities detected: {len(opportunities)}")
                for opportunity in opportunities[:3]:  # Top 3 opportunities
                    recommendations.append(f"  • {opportunity['description']}")
            
            # Risk recommendations
            if risks:
                recommendations.append(f"Trading risks detected: {len(risks)}")
                for risk in risks[:3]:  # Top 3 risks
                    recommendations.append(f"  • {risk['description']}")
            
            # Persistence recommendations
            if anomaly_persistence == "persistent":
                recommendations.append("Persistent anomalies detected - long-term monitoring required")
            elif anomaly_persistence == "moderate":
                recommendations.append("Moderate anomaly persistence - monitor for changes")
            else:
                recommendations.append("Temporary anomalies detected - monitor for resolution")
            
        except Exception:
            recommendations.append("Monitor anomaly conditions and adjust strategy accordingly")
        
        return recommendations
    
    def _generate_anomaly_insights(self, anomaly_score: float, anomaly_severity: str,
                                 opportunities: List[Dict], risks: List[Dict],
                                 anomaly_persistence: str) -> List[str]:
        """Generate anomaly insights"""
        insights = []
        
        try:
            # Anomaly insights
            insights.append(f"Anomaly score: {anomaly_score:.2f}")
            insights.append(f"Anomaly severity: {anomaly_severity}")
            insights.append(f"Anomaly persistence: {anomaly_persistence}")
            
            # Severity insights
            if anomaly_severity == "extreme":
                insights.append("Critical anomaly level - immediate attention required")
            elif anomaly_severity == "major":
                insights.append("High anomaly level - urgent attention required")
            elif anomaly_severity == "moderate":
                insights.append("Medium anomaly level - monitoring required")
            elif anomaly_severity == "minor":
                insights.append("Low anomaly level - watch for changes")
            else:
                insights.append("No significant anomalies - normal conditions")
            
            # Opportunity insights
            if opportunities:
                insights.append(f"Opportunities detected: {len(opportunities)}")
                for opportunity in opportunities[:3]:  # Top 3 opportunities
                    insights.append(f"  • {opportunity['type']}: {opportunity['description']}")
            else:
                insights.append("No significant opportunities detected")
            
            # Risk insights
            if risks:
                insights.append(f"Risks detected: {len(risks)}")
                for risk in risks[:3]:  # Top 3 risks
                    insights.append(f"  • {risk['type']}: {risk['description']}")
            else:
                insights.append("No significant risks detected")
            
            # Persistence insights
            if anomaly_persistence == "persistent":
                insights.append("Anomalies are persistent - long-term monitoring required")
            elif anomaly_persistence == "moderate":
                insights.append("Anomalies are moderately persistent - monitor for changes")
            else:
                insights.append("Anomalies are temporary - monitor for resolution")
            
        except Exception:
            insights.append("Anomaly detection analysis completed")
        
        return insights
    
    def _get_default_anomaly_analysis(self, token: Dict, market_data: Dict, historical_data: Dict) -> Dict:
        """Return default anomaly analysis when analysis fails"""
        return {
            'anomaly_score': 0.0,
            'anomaly_severity': 'none',
            'opportunities': [],
            'risks': [],
            'anomaly_persistence': 'temporary',
            'price_anomaly_analysis': {'price_anomaly_score': 0.0, 'anomaly_severity': 'none'},
            'volume_anomaly_analysis': {'volume_anomaly_score': 0.0, 'anomaly_severity': 'none'},
            'liquidity_anomaly_analysis': {'liquidity_anomaly_score': 0.0, 'anomaly_severity': 'none'},
            'sentiment_anomaly_analysis': {'sentiment_anomaly_score': 0.0, 'anomaly_severity': 'none'},
            'pattern_anomaly_analysis': {'pattern_anomaly_score': 0.0, 'anomaly_severity': 'none'},
            'correlation_anomaly_analysis': {'correlation_anomaly_score': 0.0, 'anomaly_severity': 'none'},
            'anomaly_recommendations': ['Monitor anomaly conditions'],
            'anomaly_insights': ['Anomaly detection analysis completed'],
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def get_anomaly_summary(self, tokens: List[Dict]) -> Dict:
        """Get anomaly summary for multiple tokens"""
        try:
            anomaly_summaries = []
            extreme_anomalies = 0
            major_anomalies = 0
            moderate_anomalies = 0
            minor_anomalies = 0
            no_anomalies = 0
            
            for i, token in enumerate(tokens):
                market_data = market_data_list[i] if i < len(market_data_list) else {}
                historical_data = historical_data_list[i] if i < len(historical_data_list) else {}
                
                analysis = self.detect_market_anomalies(token, market_data, historical_data)
                
                anomaly_summaries.append({
                    'symbol': token.get('symbol', 'UNKNOWN'),
                    'anomaly_score': analysis['anomaly_score'],
                    'anomaly_severity': analysis['anomaly_severity'],
                    'opportunities_count': len(analysis['opportunities']),
                    'risks_count': len(analysis['risks'])
                })
                
                severity = analysis['anomaly_severity']
                if severity == 'extreme':
                    extreme_anomalies += 1
                elif severity == 'major':
                    major_anomalies += 1
                elif severity == 'moderate':
                    moderate_anomalies += 1
                elif severity == 'minor':
                    minor_anomalies += 1
                else:
                    no_anomalies += 1
            
            return {
                'total_tokens': len(tokens),
                'extreme_anomalies': extreme_anomalies,
                'major_anomalies': major_anomalies,
                'moderate_anomalies': moderate_anomalies,
                'minor_anomalies': minor_anomalies,
                'no_anomalies': no_anomalies,
                'anomaly_summaries': anomaly_summaries,
                'overall_anomaly_risk': 'critical' if extreme_anomalies > 0 else 'high' if major_anomalies > 0 else 'medium' if moderate_anomalies > 0 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error getting anomaly summary: {e}")
            return {
                'total_tokens': len(tokens),
                'extreme_anomalies': 0,
                'major_anomalies': 0,
                'moderate_anomalies': 0,
                'minor_anomalies': 0,
                'no_anomalies': 0,
                'anomaly_summaries': [],
                'overall_anomaly_risk': 'unknown'
            }

# Global instance
ai_market_anomaly_detector = AIMarketAnomalyDetector()
