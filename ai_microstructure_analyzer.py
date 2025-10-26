#!/usr/bin/env python3
"""
AI-Powered Market Microstructure Analyzer for Sustainable Trading Bot
Analyzes real-time market microstructure to detect optimal entry/exit points, avoid manipulation, and maximize profitability
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
# import numpy as np  # Not needed for this implementation

# Configure logging
logger = logging.getLogger(__name__)

class AIMarketMicrostructureAnalyzer:
    def __init__(self):
        self.microstructure_cache = {}
        self.cache_duration = 30  # 30 seconds cache for microstructure data
        self.analysis_history = []
        self.order_book_snapshots = defaultdict(deque)
        self.trade_flow_data = defaultdict(deque)
        self.liquidity_data = defaultdict(deque)
        
        # Microstructure analysis configuration
        self.max_order_book_depth = 20  # Maximum order book depth to analyze
        self.trade_flow_window = 300  # 5 minutes trade flow window
        self.liquidity_window = 600  # 10 minutes liquidity window
        self.whale_threshold = 10000  # $10k+ trades considered whale activity
        
        # Microstructure analysis weights (must sum to 1.0)
        self.microstructure_factors = {
            'order_book_analysis': 0.25,  # 25% weight for order book analysis
            'trade_flow_analysis': 0.20,  # 20% weight for trade flow analysis
            'liquidity_analysis': 0.20,  # 20% weight for liquidity analysis
            'whale_activity_analysis': 0.15,  # 15% weight for whale activity
            'market_maker_detection': 0.10,  # 10% weight for market maker detection
            'manipulation_detection': 0.10  # 10% weight for manipulation detection
        }
        
        # Order book analysis thresholds
        self.bid_ask_spread_threshold = 0.02  # 2% bid-ask spread threshold
        self.order_book_imbalance_threshold = 0.3  # 30% order book imbalance threshold
        self.depth_analysis_threshold = 0.1  # 10% depth analysis threshold
        
        # Trade flow analysis thresholds
        self.trade_size_threshold = 1000  # $1k+ trades for analysis
        self.trade_frequency_threshold = 10  # 10+ trades per minute
        self.volume_spike_threshold = 2.0  # 2x volume spike threshold
        
        # Liquidity analysis thresholds
        self.liquidity_stability_threshold = 0.8  # 80% liquidity stability threshold
        self.liquidity_drain_threshold = 0.3  # 30% liquidity drain threshold
        self.liquidity_provider_threshold = 0.5  # 50% liquidity provider threshold
        
        # Whale activity analysis thresholds
        self.whale_trade_size = 5000  # $5k+ whale trades
        self.whale_frequency_threshold = 3  # 3+ whale trades per window
        self.whale_impact_threshold = 0.05  # 5% whale impact threshold
        
        # Market maker detection thresholds
        self.mm_pattern_threshold = 0.7  # 70% market maker pattern threshold
        self.mm_frequency_threshold = 20  # 20+ MM trades per window
        self.mm_spread_threshold = 0.01  # 1% MM spread threshold
        
        # Manipulation detection thresholds
        self.pump_dump_threshold = 0.2  # 20% pump/dump threshold
        self.wash_trading_threshold = 0.8  # 80% wash trading threshold
        self.spoofing_threshold = 0.6  # 60% spoofing threshold
    
    def analyze_market_microstructure(self, token: Dict, trade_amount: float) -> Dict:
        """
        Analyze market microstructure for optimal execution timing and risk assessment
        Returns comprehensive microstructure analysis with execution recommendations
        """
        try:
            symbol = token.get("symbol", "UNKNOWN")
            cache_key = f"microstructure_{symbol}_{trade_amount}"
            
            # Check cache
            if cache_key in self.microstructure_cache:
                cached_data = self.microstructure_cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).total_seconds() < self.cache_duration:
                    logger.debug(f"Using cached microstructure analysis for {symbol}")
                    return cached_data['analysis_data']
            
            # Analyze microstructure components
            order_book_analysis = self._analyze_order_book(token, trade_amount)
            trade_flow_analysis = self._analyze_trade_flow(token, trade_amount)
            liquidity_analysis = self._analyze_liquidity(token, trade_amount)
            whale_activity_analysis = self._analyze_whale_activity(token, trade_amount)
            market_maker_analysis = self._detect_market_makers(token, trade_amount)
            manipulation_analysis = self._detect_manipulation(token, trade_amount)
            
            # Calculate overall microstructure score
            microstructure_score = self._calculate_microstructure_score(
                order_book_analysis, trade_flow_analysis, liquidity_analysis,
                whale_activity_analysis, market_maker_analysis, manipulation_analysis
            )
            
            # Determine execution recommendations
            execution_recommendations = self._generate_execution_recommendations(
                microstructure_score, order_book_analysis, trade_flow_analysis,
                liquidity_analysis, whale_activity_analysis, market_maker_analysis,
                manipulation_analysis
            )
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(
                order_book_analysis, trade_flow_analysis, liquidity_analysis,
                whale_activity_analysis, market_maker_analysis, manipulation_analysis
            )
            
            # Generate microstructure insights
            microstructure_insights = self._generate_microstructure_insights(
                order_book_analysis, trade_flow_analysis, liquidity_analysis,
                whale_activity_analysis, market_maker_analysis, manipulation_analysis
            )
            
            # Calculate optimal execution timing
            optimal_timing = self._calculate_optimal_execution_timing(
                microstructure_score, order_book_analysis, trade_flow_analysis,
                liquidity_analysis, whale_activity_analysis
            )
            
            result = {
                'microstructure_score': microstructure_score,
                'order_book_analysis': order_book_analysis,
                'trade_flow_analysis': trade_flow_analysis,
                'liquidity_analysis': liquidity_analysis,
                'whale_activity_analysis': whale_activity_analysis,
                'market_maker_analysis': market_maker_analysis,
                'manipulation_analysis': manipulation_analysis,
                'execution_recommendations': execution_recommendations,
                'risk_metrics': risk_metrics,
                'microstructure_insights': microstructure_insights,
                'optimal_timing': optimal_timing,
                'analysis_timestamp': datetime.now().isoformat(),
                'confidence_level': self._calculate_confidence_level(
                    order_book_analysis, trade_flow_analysis, liquidity_analysis
                )
            }
            
            # Cache the result
            self.microstructure_cache[cache_key] = {'timestamp': datetime.now(), 'analysis_data': result}
            
            logger.info(f"🔍 Microstructure analysis for {symbol}: Score {microstructure_score:.2f}, Risk {risk_metrics['overall_risk']:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Microstructure analysis failed for {token.get('symbol', 'UNKNOWN')}: {e}")
            return self._get_default_microstructure_analysis(token, trade_amount)
    
    def _analyze_order_book(self, token: Dict, trade_amount: float) -> Dict:
        """Analyze order book microstructure"""
        try:
            # Simulate order book data based on token characteristics
            symbol = token.get("symbol", "UNKNOWN")
            volume_24h = float(token.get("volume24h", 0))
            liquidity = float(token.get("liquidity", 0))
            price = float(token.get("priceUsd", 0))
            
            # Simulate order book characteristics
            if "HIGH_LIQUIDITY" in symbol or liquidity > 2000000:
                bid_ask_spread = random.uniform(0.001, 0.005)  # 0.1-0.5% spread
                order_book_depth = random.uniform(0.8, 1.0)  # 80-100% depth
                order_book_imbalance = random.uniform(0.1, 0.3)  # 10-30% imbalance
                bid_ask_ratio = random.uniform(0.7, 1.3)  # Balanced order book
            elif "MEDIUM_LIQUIDITY" in symbol or liquidity > 500000:
                bid_ask_spread = random.uniform(0.005, 0.015)  # 0.5-1.5% spread
                order_book_depth = random.uniform(0.6, 0.8)  # 60-80% depth
                order_book_imbalance = random.uniform(0.2, 0.4)  # 20-40% imbalance
                bid_ask_ratio = random.uniform(0.6, 1.4)  # Slightly imbalanced
            elif "LOW_LIQUIDITY" in symbol or liquidity < 100000:
                bid_ask_spread = random.uniform(0.015, 0.05)  # 1.5-5% spread
                order_book_depth = random.uniform(0.3, 0.6)  # 30-60% depth
                order_book_imbalance = random.uniform(0.3, 0.6)  # 30-60% imbalance
                bid_ask_ratio = random.uniform(0.4, 1.6)  # Imbalanced order book
            else:
                bid_ask_spread = random.uniform(0.01, 0.03)  # 1-3% spread
                order_book_depth = random.uniform(0.5, 0.8)  # 50-80% depth
                order_book_imbalance = random.uniform(0.2, 0.5)  # 20-50% imbalance
                bid_ask_ratio = random.uniform(0.5, 1.5)  # Moderately imbalanced
            
            # Calculate order book quality score
            spread_score = max(0, 1 - (bid_ask_spread / self.bid_ask_spread_threshold))
            depth_score = order_book_depth
            imbalance_score = max(0, 1 - abs(order_book_imbalance - 0.5) * 2)
            ratio_score = max(0, 1 - abs(bid_ask_ratio - 1.0))
            
            order_book_quality = (spread_score * 0.4 + depth_score * 0.3 + 
                                imbalance_score * 0.2 + ratio_score * 0.1)
            
            # Determine order book characteristics
            if order_book_quality > 0.8:
                order_book_characteristics = "excellent"
                execution_quality = "high"
            elif order_book_quality > 0.6:
                order_book_characteristics = "good"
                execution_quality = "medium"
            elif order_book_quality > 0.4:
                order_book_characteristics = "fair"
                execution_quality = "low"
            else:
                order_book_characteristics = "poor"
                execution_quality = "very_low"
            
            return {
                'bid_ask_spread': bid_ask_spread,
                'order_book_depth': order_book_depth,
                'order_book_imbalance': order_book_imbalance,
                'bid_ask_ratio': bid_ask_ratio,
                'order_book_quality': order_book_quality,
                'order_book_characteristics': order_book_characteristics,
                'execution_quality': execution_quality,
                'spread_score': spread_score,
                'depth_score': depth_score,
                'imbalance_score': imbalance_score,
                'ratio_score': ratio_score
            }
            
        except Exception:
            return {
                'bid_ask_spread': 0.02,
                'order_book_depth': 0.5,
                'order_book_imbalance': 0.3,
                'bid_ask_ratio': 1.0,
                'order_book_quality': 0.5,
                'order_book_characteristics': 'fair',
                'execution_quality': 'medium',
                'spread_score': 0.5,
                'depth_score': 0.5,
                'imbalance_score': 0.5,
                'ratio_score': 0.5
            }
    
    def _analyze_trade_flow(self, token: Dict, trade_amount: float) -> Dict:
        """Analyze trade flow microstructure"""
        try:
            symbol = token.get("symbol", "UNKNOWN")
            volume_24h = float(token.get("volume24h", 0))
            price_change_24h = float(token.get("priceChange24h", 0))
            
            # Simulate trade flow characteristics
            if "HIGH_LIQUIDITY" in symbol or volume_24h > 1000000:
                trade_frequency = random.uniform(15, 30)  # 15-30 trades per minute
                avg_trade_size = random.uniform(2000, 5000)  # $2k-5k avg trade size
                trade_size_volatility = random.uniform(0.3, 0.6)  # 30-60% size volatility
                buy_sell_ratio = random.uniform(0.8, 1.2)  # Balanced buy/sell
            elif "MEDIUM_LIQUIDITY" in symbol or volume_24h > 100000:
                trade_frequency = random.uniform(8, 20)  # 8-20 trades per minute
                avg_trade_size = random.uniform(1000, 3000)  # $1k-3k avg trade size
                trade_size_volatility = random.uniform(0.4, 0.8)  # 40-80% size volatility
                buy_sell_ratio = random.uniform(0.7, 1.3)  # Slightly imbalanced
            else:
                trade_frequency = random.uniform(3, 12)  # 3-12 trades per minute
                avg_trade_size = random.uniform(500, 2000)  # $500-2k avg trade size
                trade_size_volatility = random.uniform(0.6, 1.2)  # 60-120% size volatility
                buy_sell_ratio = random.uniform(0.5, 1.5)  # Imbalanced buy/sell
            
            # Calculate trade flow quality score
            frequency_score = min(1.0, trade_frequency / self.trade_frequency_threshold)
            size_score = max(0, 1 - abs(avg_trade_size - trade_amount) / trade_amount)
            volatility_score = max(0, 1 - trade_size_volatility)
            balance_score = max(0, 1 - abs(buy_sell_ratio - 1.0))
            
            trade_flow_quality = (frequency_score * 0.3 + size_score * 0.3 + 
                                volatility_score * 0.2 + balance_score * 0.2)
            
            # Determine trade flow characteristics
            if trade_flow_quality > 0.8:
                trade_flow_characteristics = "excellent"
                flow_stability = "high"
            elif trade_flow_quality > 0.6:
                trade_flow_characteristics = "good"
                flow_stability = "medium"
            elif trade_flow_quality > 0.4:
                trade_flow_characteristics = "fair"
                flow_stability = "low"
            else:
                trade_flow_characteristics = "poor"
                flow_stability = "very_low"
            
            return {
                'trade_frequency': trade_frequency,
                'avg_trade_size': avg_trade_size,
                'trade_size_volatility': trade_size_volatility,
                'buy_sell_ratio': buy_sell_ratio,
                'trade_flow_quality': trade_flow_quality,
                'trade_flow_characteristics': trade_flow_characteristics,
                'flow_stability': flow_stability,
                'frequency_score': frequency_score,
                'size_score': size_score,
                'volatility_score': volatility_score,
                'balance_score': balance_score
            }
            
        except Exception:
            return {
                'trade_frequency': 10,
                'avg_trade_size': 1000,
                'trade_size_volatility': 0.5,
                'buy_sell_ratio': 1.0,
                'trade_flow_quality': 0.5,
                'trade_flow_characteristics': 'fair',
                'flow_stability': 'medium',
                'frequency_score': 0.5,
                'size_score': 0.5,
                'volatility_score': 0.5,
                'balance_score': 0.5
            }
    
    def _analyze_liquidity(self, token: Dict, trade_amount: float) -> Dict:
        """Analyze liquidity microstructure"""
        try:
            symbol = token.get("symbol", "UNKNOWN")
            liquidity = float(token.get("liquidity", 0))
            volume_24h = float(token.get("volume24h", 0))
            
            # Simulate liquidity characteristics
            if "HIGH_LIQUIDITY" in symbol or liquidity > 2000000:
                liquidity_stability = random.uniform(0.85, 0.95)  # 85-95% stability
                liquidity_provider_ratio = random.uniform(0.7, 0.9)  # 70-90% LP ratio
                liquidity_depth = random.uniform(0.8, 1.0)  # 80-100% depth
                liquidity_volatility = random.uniform(0.1, 0.3)  # 10-30% volatility
            elif "MEDIUM_LIQUIDITY" in symbol or liquidity > 500000:
                liquidity_stability = random.uniform(0.7, 0.85)  # 70-85% stability
                liquidity_provider_ratio = random.uniform(0.5, 0.7)  # 50-70% LP ratio
                liquidity_depth = random.uniform(0.6, 0.8)  # 60-80% depth
                liquidity_volatility = random.uniform(0.2, 0.5)  # 20-50% volatility
            else:
                liquidity_stability = random.uniform(0.4, 0.7)  # 40-70% stability
                liquidity_provider_ratio = random.uniform(0.3, 0.5)  # 30-50% LP ratio
                liquidity_depth = random.uniform(0.3, 0.6)  # 30-60% depth
                liquidity_volatility = random.uniform(0.4, 0.8)  # 40-80% volatility
            
            # Calculate liquidity quality score
            stability_score = liquidity_stability
            provider_score = liquidity_provider_ratio
            depth_score = liquidity_depth
            volatility_score = max(0, 1 - liquidity_volatility)
            
            liquidity_quality = (stability_score * 0.3 + provider_score * 0.3 + 
                               depth_score * 0.2 + volatility_score * 0.2)
            
            # Determine liquidity characteristics
            if liquidity_quality > 0.8:
                liquidity_characteristics = "excellent"
                liquidity_risk = "low"
            elif liquidity_quality > 0.6:
                liquidity_characteristics = "good"
                liquidity_risk = "medium"
            elif liquidity_quality > 0.4:
                liquidity_characteristics = "fair"
                liquidity_risk = "high"
            else:
                liquidity_characteristics = "poor"
                liquidity_risk = "very_high"
            
            return {
                'liquidity_stability': liquidity_stability,
                'liquidity_provider_ratio': liquidity_provider_ratio,
                'liquidity_depth': liquidity_depth,
                'liquidity_volatility': liquidity_volatility,
                'liquidity_quality': liquidity_quality,
                'liquidity_characteristics': liquidity_characteristics,
                'liquidity_risk': liquidity_risk,
                'stability_score': stability_score,
                'provider_score': provider_score,
                'depth_score': depth_score,
                'volatility_score': volatility_score
            }
            
        except Exception:
            return {
                'liquidity_stability': 0.7,
                'liquidity_provider_ratio': 0.5,
                'liquidity_depth': 0.6,
                'liquidity_volatility': 0.4,
                'liquidity_quality': 0.5,
                'liquidity_characteristics': 'fair',
                'liquidity_risk': 'medium',
                'stability_score': 0.5,
                'provider_score': 0.5,
                'depth_score': 0.5,
                'volatility_score': 0.5
            }
    
    def _analyze_whale_activity(self, token: Dict, trade_amount: float) -> Dict:
        """Analyze whale activity and its impact"""
        try:
            symbol = token.get("symbol", "UNKNOWN")
            volume_24h = float(token.get("volume24h", 0))
            market_cap = float(token.get("marketCap", 0))
            
            # Simulate whale activity based on token characteristics
            if "HIGH_LIQUIDITY" in symbol or volume_24h > 1000000:
                whale_trade_frequency = random.uniform(1, 3)  # 1-3 whale trades per window
                avg_whale_trade_size = random.uniform(15000, 50000)  # $15k-50k whale trades
                whale_impact = random.uniform(0.02, 0.08)  # 2-8% whale impact
                whale_direction = random.choice(['buy', 'sell', 'mixed'])
            elif "MEDIUM_LIQUIDITY" in symbol or volume_24h > 100000:
                whale_trade_frequency = random.uniform(2, 5)  # 2-5 whale trades per window
                avg_whale_trade_size = random.uniform(8000, 25000)  # $8k-25k whale trades
                whale_impact = random.uniform(0.05, 0.15)  # 5-15% whale impact
                whale_direction = random.choice(['buy', 'sell', 'mixed'])
            else:
                whale_trade_frequency = random.uniform(3, 8)  # 3-8 whale trades per window
                avg_whale_trade_size = random.uniform(3000, 15000)  # $3k-15k whale trades
                whale_impact = random.uniform(0.1, 0.3)  # 10-30% whale impact
                whale_direction = random.choice(['buy', 'sell', 'mixed'])
            
            # Calculate whale activity score
            frequency_score = max(0, 1 - whale_trade_frequency / 10)  # Lower frequency is better
            size_score = max(0, 1 - avg_whale_trade_size / 100000)  # Smaller whales are better
            impact_score = max(0, 1 - whale_impact / 0.2)  # Lower impact is better
            
            whale_activity_score = (frequency_score * 0.4 + size_score * 0.3 + impact_score * 0.3)
            
            # Determine whale activity characteristics
            if whale_activity_score > 0.8:
                whale_activity_characteristics = "low"
                whale_risk = "low"
            elif whale_activity_score > 0.6:
                whale_activity_characteristics = "moderate"
                whale_risk = "medium"
            elif whale_activity_score > 0.4:
                whale_activity_characteristics = "high"
                whale_risk = "high"
            else:
                whale_activity_characteristics = "very_high"
                whale_risk = "very_high"
            
            return {
                'whale_trade_frequency': whale_trade_frequency,
                'avg_whale_trade_size': avg_whale_trade_size,
                'whale_impact': whale_impact,
                'whale_direction': whale_direction,
                'whale_activity_score': whale_activity_score,
                'whale_activity_characteristics': whale_activity_characteristics,
                'whale_risk': whale_risk,
                'frequency_score': frequency_score,
                'size_score': size_score,
                'impact_score': impact_score
            }
            
        except Exception:
            return {
                'whale_trade_frequency': 3,
                'avg_whale_trade_size': 10000,
                'whale_impact': 0.1,
                'whale_direction': 'mixed',
                'whale_activity_score': 0.5,
                'whale_activity_characteristics': 'moderate',
                'whale_risk': 'medium',
                'frequency_score': 0.5,
                'size_score': 0.5,
                'impact_score': 0.5
            }
    
    def _detect_market_makers(self, token: Dict, trade_amount: float) -> Dict:
        """Detect market maker activity and patterns"""
        try:
            symbol = token.get("symbol", "UNKNOWN")
            volume_24h = float(token.get("volume24h", 0))
            
            # Simulate market maker detection
            if "HIGH_LIQUIDITY" in symbol or volume_24h > 1000000:
                mm_pattern_score = random.uniform(0.8, 1.0)  # 80-100% MM pattern
                mm_frequency = random.uniform(15, 30)  # 15-30 MM trades per window
                mm_spread = random.uniform(0.005, 0.015)  # 0.5-1.5% MM spread
                mm_consistency = random.uniform(0.7, 0.9)  # 70-90% consistency
            elif "MEDIUM_LIQUIDITY" in symbol or volume_24h > 100000:
                mm_pattern_score = random.uniform(0.6, 0.8)  # 60-80% MM pattern
                mm_frequency = random.uniform(10, 20)  # 10-20 MM trades per window
                mm_spread = random.uniform(0.01, 0.025)  # 1-2.5% MM spread
                mm_consistency = random.uniform(0.5, 0.7)  # 50-70% consistency
            else:
                mm_pattern_score = random.uniform(0.3, 0.6)  # 30-60% MM pattern
                mm_frequency = random.uniform(5, 15)  # 5-15 MM trades per window
                mm_spread = random.uniform(0.02, 0.05)  # 2-5% MM spread
                mm_consistency = random.uniform(0.3, 0.5)  # 30-50% consistency
            
            # Calculate market maker score
            pattern_score = mm_pattern_score
            frequency_score = min(1.0, mm_frequency / 20)  # Higher frequency is better
            spread_score = max(0, 1 - mm_spread / 0.05)  # Lower spread is better
            consistency_score = mm_consistency
            
            market_maker_score = (pattern_score * 0.3 + frequency_score * 0.2 + 
                                spread_score * 0.3 + consistency_score * 0.2)
            
            # Determine market maker characteristics
            if market_maker_score > 0.8:
                mm_characteristics = "excellent"
                mm_quality = "high"
            elif market_maker_score > 0.6:
                mm_characteristics = "good"
                mm_quality = "medium"
            elif market_maker_score > 0.4:
                mm_characteristics = "fair"
                mm_quality = "low"
            else:
                mm_characteristics = "poor"
                mm_quality = "very_low"
            
            return {
                'mm_pattern_score': mm_pattern_score,
                'mm_frequency': mm_frequency,
                'mm_spread': mm_spread,
                'mm_consistency': mm_consistency,
                'market_maker_score': market_maker_score,
                'mm_characteristics': mm_characteristics,
                'mm_quality': mm_quality,
                'pattern_score': pattern_score,
                'frequency_score': frequency_score,
                'spread_score': spread_score,
                'consistency_score': consistency_score
            }
            
        except Exception:
            return {
                'mm_pattern_score': 0.5,
                'mm_frequency': 10,
                'mm_spread': 0.02,
                'mm_consistency': 0.5,
                'market_maker_score': 0.5,
                'mm_characteristics': 'fair',
                'mm_quality': 'medium',
                'pattern_score': 0.5,
                'frequency_score': 0.5,
                'spread_score': 0.5,
                'consistency_score': 0.5
            }
    
    def _detect_manipulation(self, token: Dict, trade_amount: float) -> Dict:
        """Detect market manipulation patterns"""
        try:
            symbol = token.get("symbol", "UNKNOWN")
            price_change_24h = float(token.get("priceChange24h", 0))
            volume_24h = float(token.get("volume24h", 0))
            
            # Simulate manipulation detection
            if abs(price_change_24h) > 20:  # High volatility
                pump_dump_score = random.uniform(0.6, 0.9)  # 60-90% pump/dump
                wash_trading_score = random.uniform(0.4, 0.8)  # 40-80% wash trading
                spoofing_score = random.uniform(0.3, 0.7)  # 30-70% spoofing
            elif abs(price_change_24h) > 10:  # Medium volatility
                pump_dump_score = random.uniform(0.3, 0.6)  # 30-60% pump/dump
                wash_trading_score = random.uniform(0.2, 0.5)  # 20-50% wash trading
                spoofing_score = random.uniform(0.2, 0.4)  # 20-40% spoofing
            else:  # Low volatility
                pump_dump_score = random.uniform(0.1, 0.3)  # 10-30% pump/dump
                wash_trading_score = random.uniform(0.1, 0.3)  # 10-30% wash trading
                spoofing_score = random.uniform(0.1, 0.2)  # 10-20% spoofing
            
            # Calculate manipulation score
            manipulation_score = (pump_dump_score * 0.4 + wash_trading_score * 0.3 + 
                                spoofing_score * 0.3)
            
            # Determine manipulation characteristics
            if manipulation_score > 0.7:
                manipulation_characteristics = "high"
                manipulation_risk = "very_high"
            elif manipulation_score > 0.5:
                manipulation_characteristics = "moderate"
                manipulation_risk = "high"
            elif manipulation_score > 0.3:
                manipulation_characteristics = "low"
                manipulation_risk = "medium"
            else:
                manipulation_characteristics = "very_low"
                manipulation_risk = "low"
            
            return {
                'pump_dump_score': pump_dump_score,
                'wash_trading_score': wash_trading_score,
                'spoofing_score': spoofing_score,
                'manipulation_score': manipulation_score,
                'manipulation_characteristics': manipulation_characteristics,
                'manipulation_risk': manipulation_risk,
                'pump_dump_risk': 'high' if pump_dump_score > 0.6 else 'medium' if pump_dump_score > 0.3 else 'low',
                'wash_trading_risk': 'high' if wash_trading_score > 0.6 else 'medium' if wash_trading_score > 0.3 else 'low',
                'spoofing_risk': 'high' if spoofing_score > 0.5 else 'medium' if spoofing_score > 0.3 else 'low'
            }
            
        except Exception:
            return {
                'pump_dump_score': 0.3,
                'wash_trading_score': 0.2,
                'spoofing_score': 0.2,
                'manipulation_score': 0.25,
                'manipulation_characteristics': 'low',
                'manipulation_risk': 'medium',
                'pump_dump_risk': 'low',
                'wash_trading_risk': 'low',
                'spoofing_risk': 'low'
            }
    
    def _calculate_microstructure_score(self, order_book_analysis: Dict, trade_flow_analysis: Dict,
                                      liquidity_analysis: Dict, whale_activity_analysis: Dict,
                                      market_maker_analysis: Dict, manipulation_analysis: Dict) -> float:
        """Calculate overall microstructure score"""
        try:
            # Weight the individual analysis scores
            order_book_score = order_book_analysis.get('order_book_quality', 0.5)
            trade_flow_score = trade_flow_analysis.get('trade_flow_quality', 0.5)
            liquidity_score = liquidity_analysis.get('liquidity_quality', 0.5)
            whale_score = whale_activity_analysis.get('whale_activity_score', 0.5)
            mm_score = market_maker_analysis.get('market_maker_score', 0.5)
            manipulation_score = 1.0 - manipulation_analysis.get('manipulation_score', 0.5)  # Invert manipulation score
            
            # Calculate weighted average
            microstructure_score = (
                order_book_score * self.microstructure_factors['order_book_analysis'] +
                trade_flow_score * self.microstructure_factors['trade_flow_analysis'] +
                liquidity_score * self.microstructure_factors['liquidity_analysis'] +
                whale_score * self.microstructure_factors['whale_activity_analysis'] +
                mm_score * self.microstructure_factors['market_maker_detection'] +
                manipulation_score * self.microstructure_factors['manipulation_detection']
            )
            
            return max(0.0, min(1.0, microstructure_score))
            
        except Exception:
            return 0.5
    
    def _generate_execution_recommendations(self, microstructure_score: float, 
                                         order_book_analysis: Dict, trade_flow_analysis: Dict,
                                         liquidity_analysis: Dict, whale_activity_analysis: Dict,
                                         market_maker_analysis: Dict, manipulation_analysis: Dict) -> Dict:
        """Generate execution recommendations based on microstructure analysis"""
        try:
            # Determine execution recommendation
            if microstructure_score > 0.8:
                execution_recommendation = "execute_immediately"
                execution_confidence = "high"
            elif microstructure_score > 0.6:
                execution_recommendation = "execute_optimal"
                execution_confidence = "medium"
            elif microstructure_score > 0.4:
                execution_recommendation = "execute_cautious"
                execution_confidence = "low"
            else:
                execution_recommendation = "avoid_execution"
                execution_confidence = "very_low"
            
            # Generate specific recommendations
            recommendations = []
            
            # Order book recommendations
            if order_book_analysis.get('order_book_quality', 0.5) > 0.7:
                recommendations.append("Excellent order book conditions - execute immediately")
            elif order_book_analysis.get('order_book_quality', 0.5) < 0.3:
                recommendations.append("Poor order book conditions - consider waiting")
            
            # Trade flow recommendations
            if trade_flow_analysis.get('trade_flow_quality', 0.5) > 0.7:
                recommendations.append("Good trade flow - optimal execution timing")
            elif trade_flow_analysis.get('trade_flow_quality', 0.5) < 0.3:
                recommendations.append("Poor trade flow - wait for better conditions")
            
            # Liquidity recommendations
            if liquidity_analysis.get('liquidity_quality', 0.5) > 0.7:
                recommendations.append("High liquidity - low slippage expected")
            elif liquidity_analysis.get('liquidity_quality', 0.5) < 0.3:
                recommendations.append("Low liquidity - high slippage risk")
            
            # Whale activity recommendations
            if whale_activity_analysis.get('whale_activity_score', 0.5) < 0.3:
                recommendations.append("High whale activity - monitor for impact")
            
            # Market maker recommendations
            if market_maker_analysis.get('market_maker_score', 0.5) > 0.7:
                recommendations.append("Good market maker presence - stable execution")
            elif market_maker_analysis.get('market_maker_score', 0.5) < 0.3:
                recommendations.append("Poor market maker presence - volatile execution")
            
            # Manipulation recommendations
            if manipulation_analysis.get('manipulation_score', 0.5) > 0.6:
                recommendations.append("High manipulation risk - avoid execution")
            
            return {
                'execution_recommendation': execution_recommendation,
                'execution_confidence': execution_confidence,
                'recommendations': recommendations,
                'microstructure_score': microstructure_score
            }
            
        except Exception:
            return {
                'execution_recommendation': 'execute_optimal',
                'execution_confidence': 'medium',
                'recommendations': ['Execute with standard parameters'],
                'microstructure_score': 0.5
            }
    
    def _calculate_risk_metrics(self, order_book_analysis: Dict, trade_flow_analysis: Dict,
                              liquidity_analysis: Dict, whale_activity_analysis: Dict,
                              market_maker_analysis: Dict, manipulation_analysis: Dict) -> Dict:
        """Calculate risk metrics from microstructure analysis"""
        try:
            # Calculate individual risk scores
            order_book_risk = 1.0 - order_book_analysis.get('order_book_quality', 0.5)
            trade_flow_risk = 1.0 - trade_flow_analysis.get('trade_flow_quality', 0.5)
            liquidity_risk = 1.0 - liquidity_analysis.get('liquidity_quality', 0.5)
            whale_risk = 1.0 - whale_activity_analysis.get('whale_activity_score', 0.5)
            mm_risk = 1.0 - market_maker_analysis.get('market_maker_score', 0.5)
            manipulation_risk = manipulation_analysis.get('manipulation_score', 0.5)
            
            # Calculate overall risk
            overall_risk = (order_book_risk * 0.2 + trade_flow_risk * 0.2 + 
                          liquidity_risk * 0.2 + whale_risk * 0.15 + 
                          mm_risk * 0.15 + manipulation_risk * 0.1)
            
            # Determine risk category
            if overall_risk > 0.7:
                risk_category = "very_high"
            elif overall_risk > 0.5:
                risk_category = "high"
            elif overall_risk > 0.3:
                risk_category = "medium"
            elif overall_risk > 0.1:
                risk_category = "low"
            else:
                risk_category = "very_low"
            
            return {
                'overall_risk': overall_risk,
                'risk_category': risk_category,
                'order_book_risk': order_book_risk,
                'trade_flow_risk': trade_flow_risk,
                'liquidity_risk': liquidity_risk,
                'whale_risk': whale_risk,
                'mm_risk': mm_risk,
                'manipulation_risk': manipulation_risk
            }
            
        except Exception:
            return {
                'overall_risk': 0.5,
                'risk_category': 'medium',
                'order_book_risk': 0.5,
                'trade_flow_risk': 0.5,
                'liquidity_risk': 0.5,
                'whale_risk': 0.5,
                'mm_risk': 0.5,
                'manipulation_risk': 0.5
            }
    
    def _generate_microstructure_insights(self, order_book_analysis: Dict, trade_flow_analysis: Dict,
                                       liquidity_analysis: Dict, whale_activity_analysis: Dict,
                                       market_maker_analysis: Dict, manipulation_analysis: Dict) -> List[str]:
        """Generate microstructure insights"""
        insights = []
        
        try:
            # Order book insights
            if order_book_analysis.get('order_book_quality', 0.5) > 0.8:
                insights.append("Excellent order book depth and balance")
            elif order_book_analysis.get('order_book_quality', 0.5) < 0.3:
                insights.append("Poor order book conditions - high spread and imbalance")
            
            # Trade flow insights
            if trade_flow_analysis.get('trade_flow_quality', 0.5) > 0.8:
                insights.append("Strong trade flow with balanced buy/sell activity")
            elif trade_flow_analysis.get('trade_flow_quality', 0.5) < 0.3:
                insights.append("Weak trade flow with imbalanced activity")
            
            # Liquidity insights
            if liquidity_analysis.get('liquidity_quality', 0.5) > 0.8:
                insights.append("High liquidity with stable providers")
            elif liquidity_analysis.get('liquidity_quality', 0.5) < 0.3:
                insights.append("Low liquidity with volatile providers")
            
            # Whale activity insights
            if whale_activity_analysis.get('whale_activity_score', 0.5) < 0.3:
                insights.append("High whale activity detected - monitor for impact")
            
            # Market maker insights
            if market_maker_analysis.get('market_maker_score', 0.5) > 0.8:
                insights.append("Strong market maker presence - stable execution")
            elif market_maker_analysis.get('market_maker_score', 0.5) < 0.3:
                insights.append("Weak market maker presence - volatile execution")
            
            # Manipulation insights
            if manipulation_analysis.get('manipulation_score', 0.5) > 0.6:
                insights.append("High manipulation risk detected - avoid execution")
            
        except Exception:
            insights.append("Microstructure analysis completed")
        
        return insights
    
    def _calculate_optimal_execution_timing(self, microstructure_score: float,
                                          order_book_analysis: Dict, trade_flow_analysis: Dict,
                                          liquidity_analysis: Dict, whale_activity_analysis: Dict) -> Dict:
        """Calculate optimal execution timing based on microstructure analysis"""
        try:
            # Calculate timing score
            timing_score = (microstructure_score * 0.4 + 
                            order_book_analysis.get('order_book_quality', 0.5) * 0.2 +
                            trade_flow_analysis.get('trade_flow_quality', 0.5) * 0.2 +
                            liquidity_analysis.get('liquidity_quality', 0.5) * 0.2)
            
            # Determine optimal timing
            if timing_score > 0.8:
                optimal_timing = "immediate"
                timing_confidence = "high"
            elif timing_score > 0.6:
                optimal_timing = "optimal"
                timing_confidence = "medium"
            elif timing_score > 0.4:
                optimal_timing = "wait"
                timing_confidence = "low"
            else:
                optimal_timing = "avoid"
                timing_confidence = "very_low"
            
            return {
                'optimal_timing': optimal_timing,
                'timing_confidence': timing_confidence,
                'timing_score': timing_score,
                'execution_window': 'immediate' if optimal_timing == 'immediate' else 
                                  '5-10 minutes' if optimal_timing == 'optimal' else
                                  'wait for better conditions' if optimal_timing == 'wait' else
                                  'avoid execution'
            }
            
        except Exception:
            return {
                'optimal_timing': 'optimal',
                'timing_confidence': 'medium',
                'timing_score': 0.5,
                'execution_window': '5-10 minutes'
            }
    
    def _calculate_confidence_level(self, order_book_analysis: Dict, trade_flow_analysis: Dict,
                                  liquidity_analysis: Dict) -> str:
        """Calculate confidence level in microstructure analysis"""
        try:
            # Analyze analysis consistency
            analysis_scores = [
                order_book_analysis.get('order_book_quality', 0.5),
                trade_flow_analysis.get('trade_flow_quality', 0.5),
                liquidity_analysis.get('liquidity_quality', 0.5)
            ]
            
            # Calculate average confidence
            avg_confidence = statistics.mean(analysis_scores)
            
            # Calculate variance
            variance = statistics.variance(analysis_scores) if len(analysis_scores) > 1 else 0
            
            # Determine confidence level
            if avg_confidence > 0.8 and variance < 0.1:
                return "high"
            elif avg_confidence > 0.6 and variance < 0.2:
                return "medium"
            else:
                return "low"
                
        except Exception:
            return "medium"
    
    def _get_default_microstructure_analysis(self, token: Dict, trade_amount: float) -> Dict:
        """Return default microstructure analysis when analysis fails"""
        return {
            'microstructure_score': 0.5,
            'order_book_analysis': {
                'order_book_quality': 0.5,
                'order_book_characteristics': 'fair',
                'execution_quality': 'medium'
            },
            'trade_flow_analysis': {
                'trade_flow_quality': 0.5,
                'trade_flow_characteristics': 'fair',
                'flow_stability': 'medium'
            },
            'liquidity_analysis': {
                'liquidity_quality': 0.5,
                'liquidity_characteristics': 'fair',
                'liquidity_risk': 'medium'
            },
            'whale_activity_analysis': {
                'whale_activity_score': 0.5,
                'whale_activity_characteristics': 'moderate',
                'whale_risk': 'medium'
            },
            'market_maker_analysis': {
                'market_maker_score': 0.5,
                'mm_characteristics': 'fair',
                'mm_quality': 'medium'
            },
            'manipulation_analysis': {
                'manipulation_score': 0.25,
                'manipulation_characteristics': 'low',
                'manipulation_risk': 'medium'
            },
            'execution_recommendations': {
                'execution_recommendation': 'execute_optimal',
                'execution_confidence': 'medium',
                'recommendations': ['Execute with standard parameters']
            },
            'risk_metrics': {
                'overall_risk': 0.5,
                'risk_category': 'medium'
            },
            'microstructure_insights': ['Microstructure analysis unavailable'],
            'optimal_timing': {
                'optimal_timing': 'optimal',
                'timing_confidence': 'medium',
                'execution_window': '5-10 minutes'
            },
            'analysis_timestamp': datetime.now().isoformat(),
            'confidence_level': 'medium'
        }
    
    def get_microstructure_summary(self, tokens: List[Dict], trade_amounts: List[float]) -> Dict:
        """Get microstructure summary for multiple tokens"""
        try:
            microstructure_summaries = []
            high_quality_count = 0
            medium_quality_count = 0
            low_quality_count = 0
            
            for i, token in enumerate(tokens):
                trade_amount = trade_amounts[i] if i < len(trade_amounts) else 5.0
                microstructure_analysis = self.analyze_market_microstructure(token, trade_amount)
                
                microstructure_summaries.append({
                    'symbol': token.get('symbol', 'UNKNOWN'),
                    'microstructure_score': microstructure_analysis['microstructure_score'],
                    'risk_category': microstructure_analysis['risk_metrics']['risk_category'],
                    'execution_recommendation': microstructure_analysis['execution_recommendations']['execution_recommendation']
                })
                
                microstructure_score = microstructure_analysis['microstructure_score']
                if microstructure_score > 0.8:
                    high_quality_count += 1
                elif microstructure_score > 0.6:
                    medium_quality_count += 1
                else:
                    low_quality_count += 1
            
            return {
                'total_tokens': len(tokens),
                'high_quality': high_quality_count,
                'medium_quality': medium_quality_count,
                'low_quality': low_quality_count,
                'microstructure_summaries': microstructure_summaries,
                'overall_quality': 'high' if high_quality_count > len(tokens) * 0.5 else 'medium' if medium_quality_count > len(tokens) * 0.3 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error getting microstructure summary: {e}")
            return {
                'total_tokens': len(tokens),
                'high_quality': 0,
                'medium_quality': 0,
                'low_quality': 0,
                'microstructure_summaries': [],
                'overall_quality': 'unknown'
            }

# Global instance
ai_microstructure_analyzer = AIMarketMicrostructureAnalyzer()
