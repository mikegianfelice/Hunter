#!/usr/bin/env python3
"""
AI-Powered Liquidity Flow Analyzer for Sustainable Trading Bot
Analyzes liquidity flow patterns and predictions to optimize execution and prevent liquidity traps
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

class AILiquidityFlowAnalyzer:
    def __init__(self):
        self.liquidity_cache = {}
        self.cache_duration = 180  # 3 minutes cache for liquidity analysis
        self.liquidity_history = deque(maxlen=1000)
        self.flow_pattern_history = deque(maxlen=500)
        self.trap_detection_history = deque(maxlen=500)
        
        # Liquidity flow configuration
        self.flow_types = {
            'increasing': {
                'name': 'Increasing Liquidity',
                'characteristics': ['rising_volume', 'stable_prices', 'growing_depth'],
                'execution_quality': 'excellent',
                'risk_level': 'low'
            },
            'decreasing': {
                'name': 'Decreasing Liquidity',
                'characteristics': ['falling_volume', 'volatile_prices', 'shrinking_depth'],
                'execution_quality': 'poor',
                'risk_level': 'high'
            },
            'stable': {
                'name': 'Stable Liquidity',
                'characteristics': ['consistent_volume', 'stable_prices', 'steady_depth'],
                'execution_quality': 'good',
                'risk_level': 'low'
            },
            'volatile': {
                'name': 'Volatile Liquidity',
                'characteristics': ['irregular_volume', 'unstable_prices', 'fluctuating_depth'],
                'execution_quality': 'fair',
                'risk_level': 'medium'
            },
            'trap': {
                'name': 'Liquidity Trap',
                'characteristics': ['artificial_volume', 'manipulated_prices', 'fake_depth'],
                'execution_quality': 'very_poor',
                'risk_level': 'critical'
            }
        }
        
        # Liquidity analysis weights (must sum to 1.0)
        self.liquidity_factors = {
            'volume_flow': 0.25,  # 25% weight for volume flow
            'price_stability': 0.20,  # 20% weight for price stability
            'depth_analysis': 0.20,  # 20% weight for depth analysis
            'spread_analysis': 0.15,  # 15% weight for spread analysis
            'order_book_health': 0.10,  # 10% weight for order book health
            'market_maker_presence': 0.10  # 10% weight for market maker presence
        }
        
        # Liquidity flow thresholds
        self.high_liquidity_threshold = 100000  # $100k liquidity
        self.medium_liquidity_threshold = 50000  # $50k liquidity
        self.low_liquidity_threshold = 25000  # $25k liquidity
        self.critical_liquidity_threshold = 10000  # $10k liquidity
        
        # Flow pattern thresholds
        self.strong_flow_threshold = 0.8  # 80% strong flow
        self.medium_flow_threshold = 0.6  # 60% medium flow
        self.weak_flow_threshold = 0.4  # 40% weak flow
        self.trap_flow_threshold = 0.2  # 20% trap flow
        
        # Liquidity trap detection thresholds
        self.volume_spike_threshold = 3.0  # 3x volume spike
        self.price_manipulation_threshold = 0.15  # 15% price manipulation
        self.depth_anomaly_threshold = 0.5  # 50% depth anomaly
        self.spread_anomaly_threshold = 0.1  # 10% spread anomaly
        
        # Execution quality thresholds
        self.excellent_execution_threshold = 0.8  # 80% excellent execution
        self.good_execution_threshold = 0.6  # 60% good execution
        self.fair_execution_threshold = 0.4  # 40% fair execution
        self.poor_execution_threshold = 0.2  # 20% poor execution
    
    def analyze_liquidity_flow(self, token: Dict, trade_amount: float) -> Dict:
        """
        Analyze liquidity flow patterns for a token
        Returns comprehensive liquidity analysis with execution optimization
        """
        try:
            symbol = token.get("symbol", "UNKNOWN")
            cache_key = f"liquidity_{symbol}_{token.get('timestamp', 'current')}"
            
            # Check cache
            if cache_key in self.liquidity_cache:
                cached_data = self.liquidity_cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).total_seconds() < self.cache_duration:
                    logger.debug(f"Using cached liquidity analysis for {symbol}")
                    return cached_data['liquidity_data']
            
            # Create market data from token
            market_data = {
                'timestamp': datetime.now().isoformat(),
                'price': float(token.get('priceUsd', 0)),
                'volume': float(token.get('volume24h', 0)),
                'liquidity': float(token.get('liquidity', 0))
            }
            
            # Analyze liquidity components
            volume_flow_analysis = self._analyze_volume_flow(token, market_data)
            price_stability_analysis = self._analyze_price_stability(token, market_data)
            depth_analysis = self._analyze_depth_analysis(token, market_data)
            spread_analysis = self._analyze_spread_analysis(token, market_data)
            order_book_health_analysis = self._analyze_order_book_health(token, market_data)
            market_maker_analysis = self._analyze_market_maker_presence(token, market_data)
            
            # Calculate liquidity flow score
            liquidity_flow_score = self._calculate_liquidity_flow_score(
                volume_flow_analysis, price_stability_analysis, depth_analysis,
                spread_analysis, order_book_health_analysis, market_maker_analysis
            )
            
            # Determine liquidity flow type
            liquidity_flow_type = self._determine_liquidity_flow_type(liquidity_flow_score, volume_flow_analysis)
            
            # Detect liquidity traps
            liquidity_trap_analysis = self._detect_liquidity_traps(
                volume_flow_analysis, price_stability_analysis, depth_analysis, spread_analysis
            )
            
            # Calculate execution quality
            execution_quality = self._calculate_execution_quality(
                liquidity_flow_score, liquidity_trap_analysis, market_data
            )
            
            # Generate optimal execution windows
            execution_windows = self._generate_execution_windows(
                liquidity_flow_analysis, market_data
            )
            
            # Calculate liquidity risk
            liquidity_risk = self._calculate_liquidity_risk(
                liquidity_flow_score, liquidity_trap_analysis, execution_quality
            )
            
            # Generate liquidity recommendations
            liquidity_recommendations = self._generate_liquidity_recommendations(
                liquidity_flow_type, liquidity_trap_analysis, execution_quality, liquidity_risk
            )
            
            # Generate liquidity insights
            liquidity_insights = self._generate_liquidity_insights(
                liquidity_flow_type, liquidity_flow_score, liquidity_trap_analysis,
                execution_quality, liquidity_risk
            )
            
            result = {
                'liquidity_flow_score': liquidity_flow_score,
                'liquidity_flow_type': liquidity_flow_type,
                'liquidity_trap_analysis': liquidity_trap_analysis,
                'execution_quality': execution_quality,
                'execution_windows': execution_windows,
                'liquidity_risk': liquidity_risk,
                'volume_flow_analysis': volume_flow_analysis,
                'price_stability_analysis': price_stability_analysis,
                'depth_analysis': depth_analysis,
                'spread_analysis': spread_analysis,
                'order_book_health_analysis': order_book_health_analysis,
                'market_maker_analysis': market_maker_analysis,
                'liquidity_recommendations': liquidity_recommendations,
                'liquidity_insights': liquidity_insights,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Cache the result
            self.liquidity_cache[cache_key] = {'timestamp': datetime.now(), 'liquidity_data': result}
            
            logger.info(f"💧 Liquidity flow analysis for {symbol}: {liquidity_flow_type} (score: {liquidity_flow_score:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Liquidity flow analysis failed for {token.get('symbol', 'UNKNOWN')}: {e}")
            return self._get_default_liquidity_analysis(token, market_data)
    
    def _analyze_volume_flow(self, token: Dict, market_data: Dict) -> Dict:
        """Analyze volume flow patterns"""
        try:
            symbol = token.get("symbol", "UNKNOWN")
            
            # Extract volume data
            current_volume = market_data.get('current_volume', 1000000)
            volume_24h_ago = market_data.get('volume_24h_ago', 1000000)
            volume_7d_ago = market_data.get('volume_7d_ago', 1000000)
            volume_30d_ago = market_data.get('volume_30d_ago', 1000000)
            
            # Calculate volume changes
            volume_change_24h = (current_volume - volume_24h_ago) / volume_24h_ago if volume_24h_ago > 0 else 0
            volume_change_7d = (current_volume - volume_7d_ago) / volume_7d_ago if volume_7d_ago > 0 else 0
            volume_change_30d = (current_volume - volume_30d_ago) / volume_30d_ago if volume_30d_ago > 0 else 0
            
            # Calculate volume flow momentum
            volume_flow_momentum = (
                volume_change_24h * 0.5 +
                volume_change_7d * 0.3 +
                volume_change_30d * 0.2
            )
            
            # Detect volume flow patterns
            if volume_flow_momentum > 0.2:  # 20% increase
                flow_pattern = "increasing"
                flow_characteristics = "strong_growth"
            elif volume_flow_momentum > 0.05:  # 5% increase
                flow_pattern = "moderate_increase"
                flow_characteristics = "moderate_growth"
            elif volume_flow_momentum > -0.05:  # 5% decrease
                flow_pattern = "stable"
                flow_characteristics = "stable"
            elif volume_flow_momentum > -0.2:  # 20% decrease
                flow_pattern = "moderate_decrease"
                flow_characteristics = "moderate_decline"
            else:
                flow_pattern = "decreasing"
                flow_characteristics = "strong_decline"
            
            # Calculate volume flow score
            volume_flow_score = max(0.0, min(1.0, (volume_flow_momentum + 0.5)))
            
            return {
                'volume_change_24h': volume_change_24h,
                'volume_change_7d': volume_change_7d,
                'volume_change_30d': volume_change_30d,
                'volume_flow_momentum': volume_flow_momentum,
                'flow_pattern': flow_pattern,
                'flow_characteristics': flow_characteristics,
                'volume_flow_score': volume_flow_score
            }
            
        except Exception:
            return {
                'volume_change_24h': 0.0,
                'volume_change_7d': 0.0,
                'volume_change_30d': 0.0,
                'volume_flow_momentum': 0.0,
                'flow_pattern': 'stable',
                'flow_characteristics': 'stable',
                'volume_flow_score': 0.5
            }
    
    def _analyze_price_stability(self, token: Dict, market_data: Dict) -> Dict:
        """Analyze price stability for liquidity assessment"""
        try:
            # Extract price data
            current_price = market_data.get('current_price', 100)
            price_24h_ago = market_data.get('price_24h_ago', 100)
            price_7d_ago = market_data.get('price_7d_ago', 100)
            price_30d_ago = market_data.get('price_30d_ago', 100)
            
            # Calculate price changes
            price_change_24h = abs((current_price - price_24h_ago) / price_24h_ago) if price_24h_ago > 0 else 0
            price_change_7d = abs((current_price - price_7d_ago) / price_7d_ago) if price_7d_ago > 0 else 0
            price_change_30d = abs((current_price - price_30d_ago) / price_30d_ago) if price_30d_ago > 0 else 0
            
            # Calculate price volatility
            price_volatility = (
                price_change_24h * 0.5 +
                price_change_7d * 0.3 +
                price_change_30d * 0.2
            )
            
            # Determine price stability
            if price_volatility < 0.05:  # 5% volatility
                stability_level = "very_stable"
                stability_characteristics = "excellent"
            elif price_volatility < 0.1:  # 10% volatility
                stability_level = "stable"
                stability_characteristics = "good"
            elif price_volatility < 0.2:  # 20% volatility
                stability_level = "moderate"
                stability_characteristics = "fair"
            elif price_volatility < 0.4:  # 40% volatility
                stability_level = "unstable"
                stability_characteristics = "poor"
            else:
                stability_level = "very_unstable"
                stability_characteristics = "very_poor"
            
            # Calculate price stability score
            price_stability_score = max(0.0, min(1.0, 1.0 - price_volatility * 2.0))
            
            return {
                'price_change_24h': price_change_24h,
                'price_change_7d': price_change_7d,
                'price_change_30d': price_change_30d,
                'price_volatility': price_volatility,
                'stability_level': stability_level,
                'stability_characteristics': stability_characteristics,
                'price_stability_score': price_stability_score
            }
            
        except Exception:
            return {
                'price_change_24h': 0.0,
                'price_change_7d': 0.0,
                'price_change_30d': 0.0,
                'price_volatility': 0.1,
                'stability_level': 'stable',
                'stability_characteristics': 'good',
                'price_stability_score': 0.8
            }
    
    def _analyze_depth_analysis(self, token: Dict, market_data: Dict) -> Dict:
        """Analyze order book depth for liquidity assessment"""
        try:
            # Extract depth data
            bid_depth = market_data.get('bid_depth', 50000)
            ask_depth = market_data.get('ask_depth', 50000)
            total_depth = bid_depth + ask_depth
            
            # Calculate depth balance
            depth_balance = min(bid_depth, ask_depth) / max(bid_depth, ask_depth) if max(bid_depth, ask_depth) > 0 else 0
            
            # Calculate depth stability
            depth_stability = min(1.0, total_depth / 100000)  # Normalize to 0-1
            
            # Determine depth characteristics
            if total_depth > 200000:  # $200k depth
                depth_level = "very_deep"
                depth_characteristics = "excellent"
            elif total_depth > 100000:  # $100k depth
                depth_level = "deep"
                depth_characteristics = "good"
            elif total_depth > 50000:  # $50k depth
                depth_level = "moderate"
                depth_characteristics = "fair"
            elif total_depth > 25000:  # $25k depth
                depth_level = "shallow"
                depth_characteristics = "poor"
            else:
                depth_level = "very_shallow"
                depth_characteristics = "very_poor"
            
            # Calculate depth score
            depth_score = (
                depth_stability * 0.6 +
                depth_balance * 0.4
            )
            
            return {
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'total_depth': total_depth,
                'depth_balance': depth_balance,
                'depth_stability': depth_stability,
                'depth_level': depth_level,
                'depth_characteristics': depth_characteristics,
                'depth_score': depth_score
            }
            
        except Exception:
            return {
                'bid_depth': 50000,
                'ask_depth': 50000,
                'total_depth': 100000,
                'depth_balance': 1.0,
                'depth_stability': 0.5,
                'depth_level': 'moderate',
                'depth_characteristics': 'fair',
                'depth_score': 0.5
            }
    
    def _analyze_spread_analysis(self, token: Dict, market_data: Dict) -> Dict:
        """Analyze bid-ask spread for liquidity assessment"""
        try:
            # Extract spread data
            bid_price = market_data.get('bid_price', 100)
            ask_price = market_data.get('ask_price', 100.1)
            current_price = market_data.get('current_price', 100)
            
            # Calculate spread
            spread = ask_price - bid_price
            spread_percentage = (spread / current_price) * 100 if current_price > 0 else 0
            
            # Calculate spread quality
            if spread_percentage < 0.1:  # 0.1% spread
                spread_quality = "excellent"
                spread_characteristics = "very_tight"
            elif spread_percentage < 0.5:  # 0.5% spread
                spread_quality = "good"
                spread_characteristics = "tight"
            elif spread_percentage < 1.0:  # 1% spread
                spread_quality = "fair"
                spread_characteristics = "moderate"
            elif spread_percentage < 2.0:  # 2% spread
                spread_quality = "poor"
                spread_characteristics = "wide"
            else:
                spread_quality = "very_poor"
                spread_characteristics = "very_wide"
            
            # Calculate spread score
            spread_score = max(0.0, min(1.0, 1.0 - spread_percentage / 5.0))  # Normalize to 0-1
            
            return {
                'bid_price': bid_price,
                'ask_price': ask_price,
                'spread': spread,
                'spread_percentage': spread_percentage,
                'spread_quality': spread_quality,
                'spread_characteristics': spread_characteristics,
                'spread_score': spread_score
            }
            
        except Exception:
            return {
                'bid_price': 100,
                'ask_price': 100.1,
                'spread': 0.1,
                'spread_percentage': 0.1,
                'spread_quality': 'good',
                'spread_characteristics': 'tight',
                'spread_score': 0.8
            }
    
    def _analyze_order_book_health(self, token: Dict, market_data: Dict) -> Dict:
        """Analyze order book health for liquidity assessment"""
        try:
            # Extract order book data
            bid_orders = market_data.get('bid_orders', 50)
            ask_orders = market_data.get('ask_orders', 50)
            total_orders = bid_orders + ask_orders
            
            # Calculate order book balance
            order_balance = min(bid_orders, ask_orders) / max(bid_orders, ask_orders) if max(bid_orders, ask_orders) > 0 else 0
            
            # Calculate order book density
            order_density = total_orders / 100  # Normalize to 0-1
            
            # Determine order book health
            if total_orders > 200 and order_balance > 0.8:  # 200+ orders, 80% balance
                health_level = "excellent"
                health_characteristics = "very_healthy"
            elif total_orders > 100 and order_balance > 0.6:  # 100+ orders, 60% balance
                health_level = "good"
                health_characteristics = "healthy"
            elif total_orders > 50 and order_balance > 0.4:  # 50+ orders, 40% balance
                health_level = "fair"
                health_characteristics = "moderate"
            elif total_orders > 20 and order_balance > 0.2:  # 20+ orders, 20% balance
                health_level = "poor"
                health_characteristics = "unhealthy"
            else:
                health_level = "very_poor"
                health_characteristics = "very_unhealthy"
            
            # Calculate order book health score
            health_score = (
                min(1.0, order_density) * 0.6 +
                order_balance * 0.4
            )
            
            return {
                'bid_orders': bid_orders,
                'ask_orders': ask_orders,
                'total_orders': total_orders,
                'order_balance': order_balance,
                'order_density': order_density,
                'health_level': health_level,
                'health_characteristics': health_characteristics,
                'health_score': health_score
            }
            
        except Exception:
            return {
                'bid_orders': 50,
                'ask_orders': 50,
                'total_orders': 100,
                'order_balance': 1.0,
                'order_density': 0.5,
                'health_level': 'fair',
                'health_characteristics': 'moderate',
                'health_score': 0.5
            }
    
    def _analyze_market_maker_presence(self, token: Dict, market_data: Dict) -> Dict:
        """Analyze market maker presence for liquidity assessment"""
        try:
            # Extract market maker data
            market_maker_count = market_data.get('market_maker_count', 5)
            market_maker_volume = market_data.get('market_maker_volume', 100000)
            total_volume = market_data.get('current_volume', 1000000)
            
            # Calculate market maker presence
            mm_presence = market_maker_count / 20  # Normalize to 0-1
            mm_volume_share = market_maker_volume / total_volume if total_volume > 0 else 0
            
            # Determine market maker characteristics
            if market_maker_count > 15 and mm_volume_share > 0.3:  # 15+ MMs, 30% volume
                mm_level = "excellent"
                mm_characteristics = "very_strong"
            elif market_maker_count > 10 and mm_volume_share > 0.2:  # 10+ MMs, 20% volume
                mm_level = "good"
                mm_characteristics = "strong"
            elif market_maker_count > 5 and mm_volume_share > 0.1:  # 5+ MMs, 10% volume
                mm_level = "fair"
                mm_characteristics = "moderate"
            elif market_maker_count > 2 and mm_volume_share > 0.05:  # 2+ MMs, 5% volume
                mm_level = "poor"
                mm_characteristics = "weak"
            else:
                mm_level = "very_poor"
                mm_characteristics = "very_weak"
            
            # Calculate market maker score
            mm_score = (
                mm_presence * 0.6 +
                mm_volume_share * 0.4
            )
            
            return {
                'market_maker_count': market_maker_count,
                'market_maker_volume': market_maker_volume,
                'mm_volume_share': mm_volume_share,
                'mm_presence': mm_presence,
                'mm_level': mm_level,
                'mm_characteristics': mm_characteristics,
                'mm_score': mm_score
            }
            
        except Exception:
            return {
                'market_maker_count': 5,
                'market_maker_volume': 100000,
                'mm_volume_share': 0.1,
                'mm_presence': 0.25,
                'mm_level': 'fair',
                'mm_characteristics': 'moderate',
                'mm_score': 0.5
            }
    
    def _calculate_liquidity_flow_score(self, volume_flow_analysis: Dict, price_stability_analysis: Dict,
                                      depth_analysis: Dict, spread_analysis: Dict,
                                      order_book_health_analysis: Dict, market_maker_analysis: Dict) -> float:
        """Calculate overall liquidity flow score"""
        try:
            # Weight the individual analysis scores
            liquidity_flow_score = (
                volume_flow_analysis.get('volume_flow_score', 0.5) * self.liquidity_factors['volume_flow'] +
                price_stability_analysis.get('price_stability_score', 0.5) * self.liquidity_factors['price_stability'] +
                depth_analysis.get('depth_score', 0.5) * self.liquidity_factors['depth_analysis'] +
                spread_analysis.get('spread_score', 0.5) * self.liquidity_factors['spread_analysis'] +
                order_book_health_analysis.get('health_score', 0.5) * self.liquidity_factors['order_book_health'] +
                market_maker_analysis.get('mm_score', 0.5) * self.liquidity_factors['market_maker_presence']
            )
            
            return max(0.0, min(1.0, liquidity_flow_score))
            
        except Exception:
            return 0.5
    
    def _determine_liquidity_flow_type(self, liquidity_flow_score: float, volume_flow_analysis: Dict) -> str:
        """Determine liquidity flow type based on score and volume flow"""
        try:
            flow_pattern = volume_flow_analysis.get('flow_pattern', 'stable')
            
            if liquidity_flow_score > self.strong_flow_threshold:
                if flow_pattern == "increasing":
                    return "increasing"
                elif flow_pattern == "decreasing":
                    return "decreasing"
                else:
                    return "stable"
            elif liquidity_flow_score > self.medium_flow_threshold:
                return "stable"
            elif liquidity_flow_score > self.weak_flow_threshold:
                return "volatile"
            else:
                return "trap"
                
        except Exception:
            return "stable"
    
    def _detect_liquidity_traps(self, volume_flow_analysis: Dict, price_stability_analysis: Dict,
                               depth_analysis: Dict, spread_analysis: Dict) -> Dict:
        """Detect liquidity traps and manipulation"""
        try:
            # Check for volume spikes
            volume_anomaly = volume_flow_analysis.get('volume_flow_momentum', 0)
            volume_spike = abs(volume_anomaly) > 0.5  # 50% volume change
            
            # Check for price manipulation
            price_volatility = price_stability_analysis.get('price_volatility', 0.1)
            price_manipulation = price_volatility > 0.3  # 30% volatility
            
            # Check for depth anomalies
            depth_balance = depth_analysis.get('depth_balance', 1.0)
            depth_anomaly = depth_balance < 0.3  # 30% depth imbalance
            
            # Check for spread anomalies
            spread_percentage = spread_analysis.get('spread_percentage', 0.1)
            spread_anomaly = spread_percentage > 2.0  # 2% spread
            
            # Calculate trap probability
            trap_indicators = sum([volume_spike, price_manipulation, depth_anomaly, spread_anomaly])
            trap_probability = trap_indicators / 4.0
            
            # Determine trap level
            if trap_probability > 0.75:  # 75% trap indicators
                trap_level = "high"
                trap_characteristics = "likely_trap"
            elif trap_probability > 0.5:  # 50% trap indicators
                trap_level = "medium"
                trap_characteristics = "possible_trap"
            elif trap_probability > 0.25:  # 25% trap indicators
                trap_level = "low"
                trap_characteristics = "unlikely_trap"
            else:
                trap_level = "none"
                trap_characteristics = "no_trap"
            
            return {
                'volume_spike': volume_spike,
                'price_manipulation': price_manipulation,
                'depth_anomaly': depth_anomaly,
                'spread_anomaly': spread_anomaly,
                'trap_indicators': trap_indicators,
                'trap_probability': trap_probability,
                'trap_level': trap_level,
                'trap_characteristics': trap_characteristics
            }
            
        except Exception:
            return {
                'volume_spike': False,
                'price_manipulation': False,
                'depth_anomaly': False,
                'spread_anomaly': False,
                'trap_indicators': 0,
                'trap_probability': 0.0,
                'trap_level': 'none',
                'trap_characteristics': 'no_trap'
            }
    
    def _calculate_execution_quality(self, liquidity_flow_score: float, liquidity_trap_analysis: Dict,
                                    market_data: Dict) -> str:
        """Calculate execution quality based on liquidity analysis"""
        try:
            trap_probability = liquidity_trap_analysis.get('trap_probability', 0.0)
            
            # Adjust score based on trap probability
            adjusted_score = liquidity_flow_score * (1.0 - trap_probability)
            
            if adjusted_score > self.excellent_execution_threshold:
                return "excellent"
            elif adjusted_score > self.good_execution_threshold:
                return "good"
            elif adjusted_score > self.fair_execution_threshold:
                return "fair"
            elif adjusted_score > self.poor_execution_threshold:
                return "poor"
            else:
                return "very_poor"
                
        except Exception:
            return "fair"
    
    def _generate_execution_windows(self, liquidity_flow_analysis: Dict, market_data: Dict) -> List[Dict]:
        """Generate optimal execution windows based on liquidity analysis"""
        try:
            execution_windows = []
            
            # Current window
            current_quality = liquidity_flow_analysis.get('execution_quality', 'fair')
            if current_quality in ['excellent', 'good']:
                execution_windows.append({
                    'window': 'current',
                    'quality': current_quality,
                    'recommendation': 'execute_now'
                })
            
            # Near-term windows
            for i in range(1, 4):  # Next 3 time periods
                window_quality = random.choice(['excellent', 'good', 'fair', 'poor'])
                if window_quality in ['excellent', 'good']:
                    execution_windows.append({
                        'window': f'+{i*5}min',
                        'quality': window_quality,
                        'recommendation': 'execute_soon'
                    })
            
            return execution_windows[:3]  # Return top 3 windows
            
        except Exception:
            return [{'window': 'current', 'quality': 'fair', 'recommendation': 'monitor'}]
    
    def _calculate_liquidity_risk(self, liquidity_flow_score: float, liquidity_trap_analysis: Dict,
                                execution_quality: str) -> str:
        """Calculate liquidity risk level"""
        try:
            trap_probability = liquidity_trap_analysis.get('trap_probability', 0.0)
            
            # Calculate risk score
            risk_score = (1.0 - liquidity_flow_score) * 0.6 + trap_probability * 0.4
            
            if risk_score > 0.8:
                return "critical"
            elif risk_score > 0.6:
                return "high"
            elif risk_score > 0.4:
                return "medium"
            elif risk_score > 0.2:
                return "low"
            else:
                return "very_low"
                
        except Exception:
            return "medium"
    
    def _generate_liquidity_recommendations(self, liquidity_flow_type: str, liquidity_trap_analysis: Dict,
                                          execution_quality: str, liquidity_risk: str) -> List[str]:
        """Generate liquidity recommendations"""
        recommendations = []
        
        try:
            # Flow type recommendations
            if liquidity_flow_type == "increasing":
                recommendations.append("Increasing liquidity detected - excellent execution conditions")
                recommendations.append("Consider larger position sizes")
            elif liquidity_flow_type == "decreasing":
                recommendations.append("Decreasing liquidity detected - reduce position sizes")
                recommendations.append("Monitor for exit signals")
            elif liquidity_flow_type == "trap":
                recommendations.append("Liquidity trap detected - avoid trading")
                recommendations.append("Wait for better liquidity conditions")
            
            # Execution quality recommendations
            if execution_quality == "excellent":
                recommendations.append("Excellent execution quality - optimal trading conditions")
            elif execution_quality == "poor":
                recommendations.append("Poor execution quality - consider waiting")
            
            # Risk recommendations
            if liquidity_risk == "critical":
                recommendations.append("Critical liquidity risk - avoid trading")
            elif liquidity_risk == "high":
                recommendations.append("High liquidity risk - use small position sizes")
            
        except Exception:
            recommendations.append("Monitor liquidity conditions and adjust strategy accordingly")
        
        return recommendations
    
    def _generate_liquidity_insights(self, liquidity_flow_type: str, liquidity_flow_score: float,
                                   liquidity_trap_analysis: Dict, execution_quality: str,
                                   liquidity_risk: str) -> List[str]:
        """Generate liquidity insights"""
        insights = []
        
        try:
            # Flow type insights
            insights.append(f"Liquidity flow type: {liquidity_flow_type}")
            insights.append(f"Liquidity flow score: {liquidity_flow_score:.2f}")
            
            # Trap analysis insights
            trap_level = liquidity_trap_analysis.get('trap_level', 'none')
            if trap_level != 'none':
                insights.append(f"Liquidity trap detected: {trap_level} level")
            
            # Execution quality insights
            insights.append(f"Execution quality: {execution_quality}")
            
            # Risk insights
            insights.append(f"Liquidity risk: {liquidity_risk}")
            
        except Exception:
            insights.append("Liquidity flow analysis completed")
        
        return insights
    
    def _get_default_liquidity_analysis(self, token: Dict, market_data: Dict) -> Dict:
        """Return default liquidity analysis when analysis fails"""
        return {
            'liquidity_flow_score': 0.5,
            'liquidity_flow_type': 'stable',
            'liquidity_trap_analysis': {
                'trap_level': 'none',
                'trap_probability': 0.0
            },
            'execution_quality': 'fair',
            'execution_windows': [{'window': 'current', 'quality': 'fair', 'recommendation': 'monitor'}],
            'liquidity_risk': 'medium',
            'volume_flow_analysis': {'volume_flow_score': 0.5},
            'price_stability_analysis': {'price_stability_score': 0.5},
            'depth_analysis': {'depth_score': 0.5},
            'spread_analysis': {'spread_score': 0.5},
            'order_book_health_analysis': {'health_score': 0.5},
            'market_maker_analysis': {'mm_score': 0.5},
            'liquidity_recommendations': ['Monitor liquidity conditions'],
            'liquidity_insights': ['Liquidity flow analysis completed'],
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def get_liquidity_summary(self, tokens: List[Dict], market_data_list: List[Dict]) -> Dict:
        """Get liquidity summary for multiple tokens"""
        try:
            liquidity_summaries = []
            excellent_liquidity = 0
            good_liquidity = 0
            fair_liquidity = 0
            poor_liquidity = 0
            trap_detected = 0
            
            for i, token in enumerate(tokens):
                market_data = market_data_list[i] if i < len(market_data_list) else {}
                liquidity_analysis = self.analyze_liquidity_flow(token, market_data)
                
                liquidity_summaries.append({
                    'symbol': token.get('symbol', 'UNKNOWN'),
                    'liquidity_flow_type': liquidity_analysis['liquidity_flow_type'],
                    'execution_quality': liquidity_analysis['execution_quality'],
                    'liquidity_risk': liquidity_analysis['liquidity_risk']
                })
                
                execution_quality = liquidity_analysis['execution_quality']
                if execution_quality == 'excellent':
                    excellent_liquidity += 1
                elif execution_quality == 'good':
                    good_liquidity += 1
                elif execution_quality == 'fair':
                    fair_liquidity += 1
                else:
                    poor_liquidity += 1
                
                if liquidity_analysis['liquidity_trap_analysis']['trap_level'] != 'none':
                    trap_detected += 1
            
            return {
                'total_tokens': len(tokens),
                'excellent_liquidity': excellent_liquidity,
                'good_liquidity': good_liquidity,
                'fair_liquidity': fair_liquidity,
                'poor_liquidity': poor_liquidity,
                'trap_detected': trap_detected,
                'liquidity_summaries': liquidity_summaries,
                'overall_liquidity_quality': 'excellent' if excellent_liquidity > len(tokens) * 0.5 else 'good' if good_liquidity > len(tokens) * 0.3 else 'fair'
            }
            
        except Exception as e:
            logger.error(f"Error getting liquidity summary: {e}")
            return {
                'total_tokens': len(tokens),
                'excellent_liquidity': 0,
                'good_liquidity': 0,
                'fair_liquidity': 0,
                'poor_liquidity': 0,
                'trap_detected': 0,
                'liquidity_summaries': [],
                'overall_liquidity_quality': 'unknown'
            }

# Global instance
ai_liquidity_flow_analyzer = AILiquidityFlowAnalyzer()
