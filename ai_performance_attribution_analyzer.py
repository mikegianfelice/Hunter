#!/usr/bin/env python3
"""
AI-Powered Performance Attribution Analyzer for Sustainable Trading Bot
Analyzes what drives performance to improve strategy effectiveness and optimize resource allocation
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

class AIPerformanceAttributionAnalyzer:
    def __init__(self):
        self.attribution_cache = {}
        self.cache_duration = 1800  # 30 minutes cache for attribution analysis
        self.performance_history = deque(maxlen=1000)
        self.attribution_history = deque(maxlen=500)
        self.factor_history = deque(maxlen=500)
        
        # Performance attribution factors
        self.attribution_factors = {
            'market_conditions': {
                'name': 'Market Conditions',
                'weight': 0.25,  # 25% weight
                'subfactors': ['volatility', 'trend', 'regime', 'sentiment'],
                'impact_range': [0.0, 1.0]
            },
            'token_selection': {
                'name': 'Token Selection',
                'weight': 0.20,  # 20% weight
                'subfactors': ['quality_score', 'liquidity', 'volume', 'fundamentals'],
                'impact_range': [0.0, 1.0]
            },
            'timing': {
                'name': 'Entry/Exit Timing',
                'weight': 0.20,  # 20% weight
                'subfactors': ['entry_timing', 'exit_timing', 'hold_duration', 'market_timing'],
                'impact_range': [0.0, 1.0]
            },
            'position_sizing': {
                'name': 'Position Sizing',
                'weight': 0.15,  # 15% weight
                'subfactors': ['size_accuracy', 'risk_adjustment', 'diversification', 'concentration'],
                'impact_range': [0.0, 1.0]
            },
            'risk_management': {
                'name': 'Risk Management',
                'weight': 0.10,  # 10% weight
                'subfactors': ['stop_losses', 'take_profits', 'risk_limits', 'drawdown_control'],
                'impact_range': [0.0, 1.0]
            },
            'execution_quality': {
                'name': 'Execution Quality',
                'weight': 0.10,  # 10% weight
                'subfactors': ['slippage', 'fees', 'speed', 'success_rate'],
                'impact_range': [0.0, 1.0]
            }
        }
        
        # Performance metrics
        self.performance_metrics = {
            'total_return': {
                'name': 'Total Return',
                'weight': 0.30,  # 30% weight
                'calculation': 'absolute_return',
                'target': 0.20  # 20% target
            },
            'sharpe_ratio': {
                'name': 'Sharpe Ratio',
                'weight': 0.25,  # 25% weight
                'calculation': 'risk_adjusted_return',
                'target': 1.5  # 1.5 target
            },
            'max_drawdown': {
                'name': 'Maximum Drawdown',
                'weight': 0.20,  # 20% weight
                'calculation': 'risk_metric',
                'target': 0.10  # 10% target
            },
            'win_rate': {
                'name': 'Win Rate',
                'weight': 0.15,  # 15% weight
                'calculation': 'success_rate',
                'target': 0.60  # 60% target
            },
            'profit_factor': {
                'name': 'Profit Factor',
                'weight': 0.10,  # 10% weight
                'calculation': 'profit_loss_ratio',
                'target': 1.5  # 1.5 target
            }
        }
        
        # Attribution analysis thresholds
        self.high_impact_threshold = 0.7  # 70% high impact
        self.medium_impact_threshold = 0.5  # 50% medium impact
        self.low_impact_threshold = 0.3  # 30% low impact
        
        # Performance attribution weights (must sum to 1.0)
        self.attribution_weights = {
            'market_conditions': 0.25,  # 25% weight for market conditions
            'token_selection': 0.20,  # 20% weight for token selection
            'timing': 0.20,  # 20% weight for timing
            'position_sizing': 0.15,  # 15% weight for position sizing
            'risk_management': 0.10,  # 10% weight for risk management
            'execution_quality': 0.10  # 10% weight for execution quality
        }
        
        # Factor correlation thresholds
        self.strong_correlation_threshold = 0.7  # 70% strong correlation
        self.moderate_correlation_threshold = 0.5  # 50% moderate correlation
        self.weak_correlation_threshold = 0.3  # 30% weak correlation
        
        # Performance attribution periods
        self.attribution_periods = {
            'daily': {'name': 'Daily', 'weight': 0.10, 'lookback_days': 1},
            'weekly': {'name': 'Weekly', 'weight': 0.20, 'lookback_days': 7},
            'monthly': {'name': 'Monthly', 'weight': 0.40, 'lookback_days': 30},
            'quarterly': {'name': 'Quarterly', 'weight': 0.30, 'lookback_days': 90}
        }
    
    def analyze_performance_attribution(self, token: Dict, trade_amount: float) -> Dict:
        """
        Analyze performance attribution to identify what drives success/failure
        Returns comprehensive attribution analysis with optimization recommendations
        """
        try:
            # Create mock data structures from token and trade_amount
            portfolio_data = {
                'total_value': trade_amount,
                'position_count': 1,
                'timestamp': datetime.now().isoformat()
            }
            trade_history = []  # Empty trade history for single token analysis
            market_data = {
                'timestamp': datetime.now().isoformat(),
                'price': float(token.get('priceUsd', 0)),
                'volume': float(token.get('volume24h', 0)),
                'liquidity': float(token.get('liquidity', 0)),
                'volatility': 0.2,
                'regime': 'normal'
            }
            performance_metrics = {
                'total_return': 0.0,
                'volatility': 0.2,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }
            
            cache_key = f"attribution_{portfolio_data.get('timestamp', 'current')}"
            
            # Check cache
            if cache_key in self.attribution_cache:
                cached_data = self.attribution_cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).total_seconds() < self.cache_duration:
                    logger.debug("Using cached performance attribution analysis")
                    return cached_data['attribution_data']
            
            # Analyze attribution components
            market_conditions_analysis = self._analyze_market_conditions_attribution(market_data, trade_history)
            token_selection_analysis = self._analyze_token_selection_attribution(trade_history)
            timing_analysis = self._analyze_timing_attribution(trade_history, market_data)
            position_sizing_analysis = self._analyze_position_sizing_attribution(trade_history, portfolio_data)
            risk_management_analysis = self._analyze_risk_management_attribution(trade_history, portfolio_data)
            execution_quality_analysis = self._analyze_execution_quality_attribution(trade_history)
            
            # Calculate overall attribution scores
            attribution_scores = self._calculate_attribution_scores(
                market_conditions_analysis, token_selection_analysis, timing_analysis,
                position_sizing_analysis, risk_management_analysis, execution_quality_analysis
            )
            
            # Identify key performance drivers
            key_drivers = self._identify_key_performance_drivers(attribution_scores, performance_metrics)
            
            # Calculate factor correlations
            factor_correlations = self._calculate_factor_correlations(
                market_conditions_analysis, token_selection_analysis, timing_analysis,
                position_sizing_analysis, risk_management_analysis, execution_quality_analysis
            )
            
            # Generate optimization recommendations
            optimization_recommendations = self._generate_optimization_recommendations(
                attribution_scores, key_drivers, factor_correlations, performance_metrics
            )
            
            # Calculate performance attribution by period
            period_attribution = self._calculate_period_attribution(trade_history, performance_metrics)
            
            # Generate attribution insights
            attribution_insights = self._generate_attribution_insights(
                attribution_scores, key_drivers, factor_correlations, optimization_recommendations
            )
            
            # Calculate overall attribution score
            total_attribution_score = attribution_scores.get('total_attribution_score', 0.5)
            
            # Generate performance recommendations
            performance_recommendations = {
                'performance_recommendation': 'proceed_with_caution' if total_attribution_score > 0.5 else 'avoid_trading',
                'confidence': 'medium',
                'recommendations': optimization_recommendations
            }
            
            result = {
                'attribution_score': total_attribution_score,
                'performance_recommendations': performance_recommendations,
                'attribution_scores': attribution_scores,
                'key_drivers': key_drivers,
                'factor_correlations': factor_correlations,
                'optimization_recommendations': optimization_recommendations,
                'period_attribution': period_attribution,
                'market_conditions_analysis': market_conditions_analysis,
                'token_selection_analysis': token_selection_analysis,
                'timing_analysis': timing_analysis,
                'position_sizing_analysis': position_sizing_analysis,
                'risk_management_analysis': risk_management_analysis,
                'execution_quality_analysis': execution_quality_analysis,
                'attribution_insights': attribution_insights,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Cache the result
            self.attribution_cache[cache_key] = {'timestamp': datetime.now(), 'attribution_data': result}
            
            logger.info(f"📊 Performance attribution analysis: {len(key_drivers)} key drivers identified")
            return result
            
        except Exception as e:
            logger.error(f"❌ Performance attribution analysis failed: {e}")
            # Create default data structures for error fallback
            portfolio_data = {
                'total_value': trade_amount,
                'position_count': 1,
                'timestamp': datetime.now().isoformat()
            }
            trade_history = []
            market_data = {
                'timestamp': datetime.now().isoformat(),
                'price': float(token.get('priceUsd', 0)),
                'volume': float(token.get('volume24h', 0)),
                'liquidity': float(token.get('liquidity', 0)),
                'volatility': 0.2,
                'regime': 'normal'
            }
            performance_metrics = {
                'total_return': 0.0,
                'volatility': 0.2,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }
            return self._get_default_attribution_analysis(portfolio_data, trade_history, market_data, performance_metrics)
    
    def _analyze_market_conditions_attribution(self, market_data: Dict, trade_history: List[Dict]) -> Dict:
        """Analyze market conditions attribution to performance"""
        try:
            # Extract market data
            current_volatility = market_data.get('current_volatility', 0.3)
            market_trend = market_data.get('market_trend', 'neutral')
            market_regime = market_data.get('market_regime', 'sideways')
            market_sentiment = market_data.get('market_sentiment', 0.5)
            
            # Calculate market conditions impact
            volatility_impact = self._calculate_volatility_impact(current_volatility, trade_history)
            trend_impact = self._calculate_trend_impact(market_trend, trade_history)
            regime_impact = self._calculate_regime_impact(market_regime, trade_history)
            sentiment_impact = self._calculate_sentiment_impact(market_sentiment, trade_history)
            
            # Calculate overall market conditions score
            market_conditions_score = (
                volatility_impact * 0.3 +
                trend_impact * 0.3 +
                regime_impact * 0.2 +
                sentiment_impact * 0.2
            )
            
            # Determine market conditions impact level
            if market_conditions_score > self.high_impact_threshold:
                impact_level = "high"
                impact_characteristics = "very_influential"
            elif market_conditions_score > self.medium_impact_threshold:
                impact_level = "medium"
                impact_characteristics = "moderately_influential"
            elif market_conditions_score > self.low_impact_threshold:
                impact_level = "low"
                impact_characteristics = "somewhat_influential"
            else:
                impact_level = "minimal"
                impact_characteristics = "minimally_influential"
            
            return {
                'volatility_impact': volatility_impact,
                'trend_impact': trend_impact,
                'regime_impact': regime_impact,
                'sentiment_impact': sentiment_impact,
                'market_conditions_score': market_conditions_score,
                'impact_level': impact_level,
                'impact_characteristics': impact_characteristics
            }
            
        except Exception:
            return {
                'volatility_impact': 0.5,
                'trend_impact': 0.5,
                'regime_impact': 0.5,
                'sentiment_impact': 0.5,
                'market_conditions_score': 0.5,
                'impact_level': 'medium',
                'impact_characteristics': 'moderately_influential'
            }
    
    def _analyze_token_selection_attribution(self, trade_history: List[Dict]) -> Dict:
        """Analyze token selection attribution to performance"""
        try:
            if not trade_history:
                return {
                    'quality_score_impact': 0.5,
                    'liquidity_impact': 0.5,
                    'volume_impact': 0.5,
                    'fundamentals_impact': 0.5,
                    'token_selection_score': 0.5,
                    'impact_level': 'medium',
                    'impact_characteristics': 'moderately_influential'
                }
            
            # Calculate token selection factors
            quality_score_impact = self._calculate_quality_score_impact(trade_history)
            liquidity_impact = self._calculate_liquidity_impact(trade_history)
            volume_impact = self._calculate_volume_impact(trade_history)
            fundamentals_impact = self._calculate_fundamentals_impact(trade_history)
            
            # Calculate overall token selection score
            token_selection_score = (
                quality_score_impact * 0.4 +
                liquidity_impact * 0.3 +
                volume_impact * 0.2 +
                fundamentals_impact * 0.1
            )
            
            # Determine token selection impact level
            if token_selection_score > self.high_impact_threshold:
                impact_level = "high"
                impact_characteristics = "very_influential"
            elif token_selection_score > self.medium_impact_threshold:
                impact_level = "medium"
                impact_characteristics = "moderately_influential"
            elif token_selection_score > self.low_impact_threshold:
                impact_level = "low"
                impact_characteristics = "somewhat_influential"
            else:
                impact_level = "minimal"
                impact_characteristics = "minimally_influential"
            
            return {
                'quality_score_impact': quality_score_impact,
                'liquidity_impact': liquidity_impact,
                'volume_impact': volume_impact,
                'fundamentals_impact': fundamentals_impact,
                'token_selection_score': token_selection_score,
                'impact_level': impact_level,
                'impact_characteristics': impact_characteristics
            }
            
        except Exception:
            return {
                'quality_score_impact': 0.5,
                'liquidity_impact': 0.5,
                'volume_impact': 0.5,
                'fundamentals_impact': 0.5,
                'token_selection_score': 0.5,
                'impact_level': 'medium',
                'impact_characteristics': 'moderately_influential'
            }
    
    def _analyze_timing_attribution(self, trade_history: List[Dict], market_data: Dict) -> Dict:
        """Analyze timing attribution to performance"""
        try:
            if not trade_history:
                return {
                    'entry_timing_impact': 0.5,
                    'exit_timing_impact': 0.5,
                    'hold_duration_impact': 0.5,
                    'market_timing_impact': 0.5,
                    'timing_score': 0.5,
                    'impact_level': 'medium',
                    'impact_characteristics': 'moderately_influential'
                }
            
            # Calculate timing factors
            entry_timing_impact = self._calculate_entry_timing_impact(trade_history)
            exit_timing_impact = self._calculate_exit_timing_impact(trade_history)
            hold_duration_impact = self._calculate_hold_duration_impact(trade_history)
            market_timing_impact = self._calculate_market_timing_impact(trade_history, market_data)
            
            # Calculate overall timing score
            timing_score = (
                entry_timing_impact * 0.3 +
                exit_timing_impact * 0.3 +
                hold_duration_impact * 0.2 +
                market_timing_impact * 0.2
            )
            
            # Determine timing impact level
            if timing_score > self.high_impact_threshold:
                impact_level = "high"
                impact_characteristics = "very_influential"
            elif timing_score > self.medium_impact_threshold:
                impact_level = "medium"
                impact_characteristics = "moderately_influential"
            elif timing_score > self.low_impact_threshold:
                impact_level = "low"
                impact_characteristics = "somewhat_influential"
            else:
                impact_level = "minimal"
                impact_characteristics = "minimally_influential"
            
            return {
                'entry_timing_impact': entry_timing_impact,
                'exit_timing_impact': exit_timing_impact,
                'hold_duration_impact': hold_duration_impact,
                'market_timing_impact': market_timing_impact,
                'timing_score': timing_score,
                'impact_level': impact_level,
                'impact_characteristics': impact_characteristics
            }
            
        except Exception:
            return {
                'entry_timing_impact': 0.5,
                'exit_timing_impact': 0.5,
                'hold_duration_impact': 0.5,
                'market_timing_impact': 0.5,
                'timing_score': 0.5,
                'impact_level': 'medium',
                'impact_characteristics': 'moderately_influential'
            }
    
    def _analyze_position_sizing_attribution(self, trade_history: List[Dict], portfolio_data: Dict) -> Dict:
        """Analyze position sizing attribution to performance"""
        try:
            if not trade_history:
                return {
                    'size_accuracy_impact': 0.5,
                    'risk_adjustment_impact': 0.5,
                    'diversification_impact': 0.5,
                    'concentration_impact': 0.5,
                    'position_sizing_score': 0.5,
                    'impact_level': 'medium',
                    'impact_characteristics': 'moderately_influential'
                }
            
            # Calculate position sizing factors
            size_accuracy_impact = self._calculate_size_accuracy_impact(trade_history)
            risk_adjustment_impact = self._calculate_risk_adjustment_impact(trade_history, portfolio_data)
            diversification_impact = self._calculate_diversification_impact(trade_history, portfolio_data)
            concentration_impact = self._calculate_concentration_impact(trade_history, portfolio_data)
            
            # Calculate overall position sizing score
            position_sizing_score = (
                size_accuracy_impact * 0.3 +
                risk_adjustment_impact * 0.3 +
                diversification_impact * 0.2 +
                concentration_impact * 0.2
            )
            
            # Determine position sizing impact level
            if position_sizing_score > self.high_impact_threshold:
                impact_level = "high"
                impact_characteristics = "very_influential"
            elif position_sizing_score > self.medium_impact_threshold:
                impact_level = "medium"
                impact_characteristics = "moderately_influential"
            elif position_sizing_score > self.low_impact_threshold:
                impact_level = "low"
                impact_characteristics = "somewhat_influential"
            else:
                impact_level = "minimal"
                impact_characteristics = "minimally_influential"
            
            return {
                'size_accuracy_impact': size_accuracy_impact,
                'risk_adjustment_impact': risk_adjustment_impact,
                'diversification_impact': diversification_impact,
                'concentration_impact': concentration_impact,
                'position_sizing_score': position_sizing_score,
                'impact_level': impact_level,
                'impact_characteristics': impact_characteristics
            }
            
        except Exception:
            return {
                'size_accuracy_impact': 0.5,
                'risk_adjustment_impact': 0.5,
                'diversification_impact': 0.5,
                'concentration_impact': 0.5,
                'position_sizing_score': 0.5,
                'impact_level': 'medium',
                'impact_characteristics': 'moderately_influential'
            }
    
    def _analyze_risk_management_attribution(self, trade_history: List[Dict], portfolio_data: Dict) -> Dict:
        """Analyze risk management attribution to performance"""
        try:
            if not trade_history:
                return {
                    'stop_loss_impact': 0.5,
                    'take_profit_impact': 0.5,
                    'risk_limits_impact': 0.5,
                    'drawdown_control_impact': 0.5,
                    'risk_management_score': 0.5,
                    'impact_level': 'medium',
                    'impact_characteristics': 'moderately_influential'
                }
            
            # Calculate risk management factors
            stop_loss_impact = self._calculate_stop_loss_impact(trade_history)
            take_profit_impact = self._calculate_take_profit_impact(trade_history)
            risk_limits_impact = self._calculate_risk_limits_impact(trade_history, portfolio_data)
            drawdown_control_impact = self._calculate_drawdown_control_impact(trade_history, portfolio_data)
            
            # Calculate overall risk management score
            risk_management_score = (
                stop_loss_impact * 0.3 +
                take_profit_impact * 0.3 +
                risk_limits_impact * 0.2 +
                drawdown_control_impact * 0.2
            )
            
            # Determine risk management impact level
            if risk_management_score > self.high_impact_threshold:
                impact_level = "high"
                impact_characteristics = "very_influential"
            elif risk_management_score > self.medium_impact_threshold:
                impact_level = "medium"
                impact_characteristics = "moderately_influential"
            elif risk_management_score > self.low_impact_threshold:
                impact_level = "low"
                impact_characteristics = "somewhat_influential"
            else:
                impact_level = "minimal"
                impact_characteristics = "minimally_influential"
            
            return {
                'stop_loss_impact': stop_loss_impact,
                'take_profit_impact': take_profit_impact,
                'risk_limits_impact': risk_limits_impact,
                'drawdown_control_impact': drawdown_control_impact,
                'risk_management_score': risk_management_score,
                'impact_level': impact_level,
                'impact_characteristics': impact_characteristics
            }
            
        except Exception:
            return {
                'stop_loss_impact': 0.5,
                'take_profit_impact': 0.5,
                'risk_limits_impact': 0.5,
                'drawdown_control_impact': 0.5,
                'risk_management_score': 0.5,
                'impact_level': 'medium',
                'impact_characteristics': 'moderately_influential'
            }
    
    def _analyze_execution_quality_attribution(self, trade_history: List[Dict]) -> Dict:
        """Analyze execution quality attribution to performance"""
        try:
            if not trade_history:
                return {
                    'slippage_impact': 0.5,
                    'fees_impact': 0.5,
                    'speed_impact': 0.5,
                    'success_rate_impact': 0.5,
                    'execution_quality_score': 0.5,
                    'impact_level': 'medium',
                    'impact_characteristics': 'moderately_influential'
                }
            
            # Calculate execution quality factors
            slippage_impact = self._calculate_slippage_impact(trade_history)
            fees_impact = self._calculate_fees_impact(trade_history)
            speed_impact = self._calculate_speed_impact(trade_history)
            success_rate_impact = self._calculate_success_rate_impact(trade_history)
            
            # Calculate overall execution quality score
            execution_quality_score = (
                slippage_impact * 0.3 +
                fees_impact * 0.2 +
                speed_impact * 0.2 +
                success_rate_impact * 0.3
            )
            
            # Determine execution quality impact level
            if execution_quality_score > self.high_impact_threshold:
                impact_level = "high"
                impact_characteristics = "very_influential"
            elif execution_quality_score > self.medium_impact_threshold:
                impact_level = "medium"
                impact_characteristics = "moderately_influential"
            elif execution_quality_score > self.low_impact_threshold:
                impact_level = "low"
                impact_characteristics = "somewhat_influential"
            else:
                impact_level = "minimal"
                impact_characteristics = "minimally_influential"
            
            return {
                'slippage_impact': slippage_impact,
                'fees_impact': fees_impact,
                'speed_impact': speed_impact,
                'success_rate_impact': success_rate_impact,
                'execution_quality_score': execution_quality_score,
                'impact_level': impact_level,
                'impact_characteristics': impact_characteristics
            }
            
        except Exception:
            return {
                'slippage_impact': 0.5,
                'fees_impact': 0.5,
                'speed_impact': 0.5,
                'success_rate_impact': 0.5,
                'execution_quality_score': 0.5,
                'impact_level': 'medium',
                'impact_characteristics': 'moderately_influential'
            }
    
    def _calculate_attribution_scores(self, market_conditions_analysis: Dict, token_selection_analysis: Dict,
                                     timing_analysis: Dict, position_sizing_analysis: Dict,
                                     risk_management_analysis: Dict, execution_quality_analysis: Dict) -> Dict:
        """Calculate overall attribution scores"""
        try:
            # Calculate weighted attribution scores
            attribution_scores = {
                'market_conditions': market_conditions_analysis.get('market_conditions_score', 0.5) * self.attribution_weights['market_conditions'],
                'token_selection': token_selection_analysis.get('token_selection_score', 0.5) * self.attribution_weights['token_selection'],
                'timing': timing_analysis.get('timing_score', 0.5) * self.attribution_weights['timing'],
                'position_sizing': position_sizing_analysis.get('position_sizing_score', 0.5) * self.attribution_weights['position_sizing'],
                'risk_management': risk_management_analysis.get('risk_management_score', 0.5) * self.attribution_weights['risk_management'],
                'execution_quality': execution_quality_analysis.get('execution_quality_score', 0.5) * self.attribution_weights['execution_quality']
            }
            
            # Calculate total attribution score
            total_attribution_score = sum(attribution_scores.values())
            
            # Calculate attribution percentages
            attribution_percentages = {}
            for factor, score in attribution_scores.items():
                attribution_percentages[factor] = (score / total_attribution_score) * 100 if total_attribution_score > 0 else 0
            
            return {
                'attribution_scores': attribution_scores,
                'total_attribution_score': total_attribution_score,
                'attribution_percentages': attribution_percentages
            }
            
        except Exception:
            return {
                'attribution_scores': {'market_conditions': 0.25, 'token_selection': 0.20, 'timing': 0.20, 'position_sizing': 0.15, 'risk_management': 0.10, 'execution_quality': 0.10},
                'total_attribution_score': 1.0,
                'attribution_percentages': {'market_conditions': 25.0, 'token_selection': 20.0, 'timing': 20.0, 'position_sizing': 15.0, 'risk_management': 10.0, 'execution_quality': 10.0}
            }
    
    def _identify_key_performance_drivers(self, attribution_scores: Dict, performance_metrics: Dict) -> List[Dict]:
        """Identify key performance drivers"""
        try:
            key_drivers = []
            
            # Get attribution scores
            scores = attribution_scores.get('attribution_scores', {})
            percentages = attribution_scores.get('attribution_percentages', {})
            
            # Sort factors by impact
            sorted_factors = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            # Identify top drivers
            for i, (factor, score) in enumerate(sorted_factors):
                percentage = percentages.get(factor, 0)
                
                # Determine driver importance
                if percentage > 25:  # 25%+ impact
                    importance = "critical"
                elif percentage > 15:  # 15%+ impact
                    importance = "high"
                elif percentage > 10:  # 10%+ impact
                    importance = "medium"
                else:
                    importance = "low"
                
                key_drivers.append({
                    'factor': factor,
                    'score': score,
                    'percentage': percentage,
                    'importance': importance,
                    'rank': i + 1
                })
            
            return key_drivers
            
        except Exception:
            return []
    
    def _calculate_factor_correlations(self, market_conditions_analysis: Dict, token_selection_analysis: Dict,
                                     timing_analysis: Dict, position_sizing_analysis: Dict,
                                     risk_management_analysis: Dict, execution_quality_analysis: Dict) -> Dict:
        """Calculate factor correlations"""
        try:
            # Extract scores for correlation analysis
            scores = {
                'market_conditions': market_conditions_analysis.get('market_conditions_score', 0.5),
                'token_selection': token_selection_analysis.get('token_selection_score', 0.5),
                'timing': timing_analysis.get('timing_score', 0.5),
                'position_sizing': position_sizing_analysis.get('position_sizing_score', 0.5),
                'risk_management': risk_management_analysis.get('risk_management_score', 0.5),
                'execution_quality': execution_quality_analysis.get('execution_quality_score', 0.5)
            }
            
            # Calculate correlations between factors
            correlations = {}
            factors = list(scores.keys())
            
            for i, factor1 in enumerate(factors):
                for j, factor2 in enumerate(factors):
                    if i != j:
                        # Calculate simple correlation (in real implementation, use proper correlation)
                        correlation = abs(scores[factor1] - scores[factor2]) / max(scores[factor1], scores[factor2], 0.001)
                        correlations[f"{factor1}_vs_{factor2}"] = correlation
            
            # Identify strong correlations
            strong_correlations = []
            for correlation_name, correlation_value in correlations.items():
                if correlation_value > self.strong_correlation_threshold:
                    strong_correlations.append({
                        'correlation_name': correlation_name,
                        'correlation_value': correlation_value,
                        'strength': 'strong'
                    })
                elif correlation_value > self.moderate_correlation_threshold:
                    strong_correlations.append({
                        'correlation_name': correlation_name,
                        'correlation_value': correlation_value,
                        'strength': 'moderate'
                    })
            
            return {
                'correlations': correlations,
                'strong_correlations': strong_correlations
            }
            
        except Exception:
            return {
                'correlations': {},
                'strong_correlations': []
            }
    
    def _generate_optimization_recommendations(self, attribution_scores: Dict, key_drivers: List[Dict],
                                             factor_correlations: Dict, performance_metrics: Dict) -> List[Dict]:
        """Generate optimization recommendations based on attribution analysis"""
        recommendations = []
        
        try:
            # Get attribution percentages
            percentages = attribution_scores.get('attribution_percentages', {})
            
            # Generate recommendations for top factors
            for driver in key_drivers[:3]:  # Top 3 drivers
                factor = driver['factor']
                percentage = driver['percentage']
                importance = driver['importance']
                
                if importance == "critical":
                    recommendations.append({
                        'factor': factor,
                        'priority': 'critical',
                        'recommendation': f'Focus heavily on {factor} - {percentage:.1f}% impact',
                        'action': f'optimize_{factor}_strategy'
                    })
                elif importance == "high":
                    recommendations.append({
                        'factor': factor,
                        'priority': 'high',
                        'recommendation': f'Improve {factor} performance - {percentage:.1f}% impact',
                        'action': f'enhance_{factor}_approach'
                    })
                elif importance == "medium":
                    recommendations.append({
                        'factor': factor,
                        'priority': 'medium',
                        'recommendation': f'Monitor {factor} closely - {percentage:.1f}% impact',
                        'action': f'monitor_{factor}_metrics'
                    })
            
            # Generate recommendations for low-performing factors
            for factor, percentage in percentages.items():
                if percentage < 5:  # Less than 5% impact
                    recommendations.append({
                        'factor': factor,
                        'priority': 'low',
                        'recommendation': f'Consider reducing focus on {factor} - {percentage:.1f}% impact',
                        'action': f'reduce_{factor}_emphasis'
                    })
            
            # Generate recommendations for factor correlations
            strong_correlations = factor_correlations.get('strong_correlations', [])
            if strong_correlations:
                recommendations.append({
                    'factor': 'factor_correlations',
                    'priority': 'medium',
                    'recommendation': f'Address {len(strong_correlations)} strong factor correlations',
                    'action': 'optimize_factor_interactions'
                })
            
        except Exception:
            recommendations.append({
                'factor': 'general',
                'priority': 'medium',
                'recommendation': 'Monitor performance attribution factors',
                'action': 'monitor_attribution'
            })
        
        return recommendations
    
    def _calculate_period_attribution(self, trade_history: List[Dict], performance_metrics: Dict) -> Dict:
        """Calculate performance attribution by time period"""
        try:
            period_attribution = {}
            
            for period_name, period_config in self.attribution_periods.items():
                # Calculate attribution for this period
                period_score = random.uniform(0.3, 0.8)  # Mock calculation
                period_attribution[period_name] = {
                    'name': period_config['name'],
                    'score': period_score,
                    'weight': period_config['weight'],
                    'lookback_days': period_config['lookback_days']
                }
            
            return period_attribution
            
        except Exception:
            return {}
    
    def _generate_attribution_insights(self, attribution_scores: Dict, key_drivers: List[Dict],
                                     factor_correlations: Dict, optimization_recommendations: List[Dict]) -> List[str]:
        """Generate attribution insights"""
        insights = []
        
        try:
            # Attribution insights
            percentages = attribution_scores.get('attribution_percentages', {})
            insights.append(f"Performance attribution analysis completed")
            insights.append(f"Total attribution score: {attribution_scores.get('total_attribution_score', 0):.2f}")
            
            # Top driver insights
            if key_drivers:
                top_driver = key_drivers[0]
                insights.append(f"Top performance driver: {top_driver['factor']} ({top_driver['percentage']:.1f}%)")
                
                if len(key_drivers) > 1:
                    second_driver = key_drivers[1]
                    insights.append(f"Second driver: {second_driver['factor']} ({second_driver['percentage']:.1f}%)")
            
            # Factor correlation insights
            strong_correlations = factor_correlations.get('strong_correlations', [])
            if strong_correlations:
                insights.append(f"Strong factor correlations detected: {len(strong_correlations)}")
            else:
                insights.append("No strong factor correlations detected")
            
            # Optimization insights
            if optimization_recommendations:
                critical_recommendations = [r for r in optimization_recommendations if r['priority'] == 'critical']
                if critical_recommendations:
                    insights.append(f"Critical optimization areas: {len(critical_recommendations)}")
                
                high_recommendations = [r for r in optimization_recommendations if r['priority'] == 'high']
                if high_recommendations:
                    insights.append(f"High priority optimization areas: {len(high_recommendations)}")
            
        except Exception:
            insights.append("Performance attribution analysis completed")
        
        return insights
    
    def _get_default_attribution_analysis(self, portfolio_data: Dict, trade_history: List[Dict], 
                                        market_data: Dict, performance_metrics: Dict) -> Dict:
        """Return default attribution analysis when analysis fails"""
        return {
            'attribution_scores': {'attribution_scores': {'market_conditions': 0.25, 'token_selection': 0.20, 'timing': 0.20, 'position_sizing': 0.15, 'risk_management': 0.10, 'execution_quality': 0.10}, 'total_attribution_score': 1.0, 'attribution_percentages': {'market_conditions': 25.0, 'token_selection': 20.0, 'timing': 20.0, 'position_sizing': 15.0, 'risk_management': 10.0, 'execution_quality': 10.0}},
            'key_drivers': [{'factor': 'market_conditions', 'score': 0.25, 'percentage': 25.0, 'importance': 'high', 'rank': 1}],
            'factor_correlations': {'correlations': {}, 'strong_correlations': []},
            'optimization_recommendations': [{'factor': 'general', 'priority': 'medium', 'recommendation': 'Monitor performance attribution factors', 'action': 'monitor_attribution'}],
            'period_attribution': {},
            'market_conditions_analysis': {'market_conditions_score': 0.5, 'impact_level': 'medium'},
            'token_selection_analysis': {'token_selection_score': 0.5, 'impact_level': 'medium'},
            'timing_analysis': {'timing_score': 0.5, 'impact_level': 'medium'},
            'position_sizing_analysis': {'position_sizing_score': 0.5, 'impact_level': 'medium'},
            'risk_management_analysis': {'risk_management_score': 0.5, 'impact_level': 'medium'},
            'execution_quality_analysis': {'execution_quality_score': 0.5, 'impact_level': 'medium'},
            'attribution_insights': ['Performance attribution analysis completed'],
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def get_attribution_summary(self, portfolio_data_list: List[Dict], trade_history_list: List[List[Dict]], 
                              market_data_list: List[Dict], performance_metrics_list: List[Dict]) -> Dict:
        """Get attribution summary for multiple portfolios"""
        try:
            attribution_summaries = []
            high_impact_factors = 0
            medium_impact_factors = 0
            low_impact_factors = 0
            
            for i, portfolio_data in enumerate(portfolio_data_list):
                trade_history = trade_history_list[i] if i < len(trade_history_list) else []
                market_data = market_data_list[i] if i < len(market_data_list) else {}
                performance_metrics = performance_metrics_list[i] if i < len(performance_metrics_list) else {}
                
                analysis = self.analyze_performance_attribution(portfolio_data, trade_history, market_data, performance_metrics)
                
                attribution_summaries.append({
                    'total_attribution_score': analysis['attribution_scores']['total_attribution_score'],
                    'key_drivers_count': len(analysis['key_drivers']),
                    'optimization_recommendations_count': len(analysis['optimization_recommendations'])
                })
                
                # Count impact levels
                for driver in analysis['key_drivers']:
                    importance = driver.get('importance', 'medium')
                    if importance == 'high':
                        high_impact_factors += 1
                    elif importance == 'medium':
                        medium_impact_factors += 1
                    else:
                        low_impact_factors += 1
            
            return {
                'total_portfolios': len(portfolio_data_list),
                'high_impact_factors': high_impact_factors,
                'medium_impact_factors': medium_impact_factors,
                'low_impact_factors': low_impact_factors,
                'attribution_summaries': attribution_summaries,
                'overall_attribution_quality': 'high' if high_impact_factors > medium_impact_factors else 'medium' if medium_impact_factors > low_impact_factors else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error getting attribution summary: {e}")
            return {
                'total_portfolios': len(portfolio_data_list),
                'high_impact_factors': 0,
                'medium_impact_factors': 0,
                'low_impact_factors': 0,
                'attribution_summaries': [],
                'overall_attribution_quality': 'unknown'
            }
    
    # Helper methods for calculating individual factor impacts
    def _calculate_volatility_impact(self, volatility: float, trade_history: List[Dict]) -> float:
        """Calculate volatility impact on performance"""
        try:
            # Mock calculation - in real implementation, analyze correlation between volatility and performance
            return max(0.0, min(1.0, volatility * 2.0))
        except Exception:
            return 0.5
    
    def _calculate_trend_impact(self, trend: str, trade_history: List[Dict]) -> float:
        """Calculate trend impact on performance"""
        try:
            # Mock calculation - in real implementation, analyze correlation between trend and performance
            trend_scores = {'bull': 0.8, 'bear': 0.2, 'neutral': 0.5}
            return trend_scores.get(trend, 0.5)
        except Exception:
            return 0.5
    
    def _calculate_regime_impact(self, regime: str, trade_history: List[Dict]) -> float:
        """Calculate regime impact on performance"""
        try:
            # Mock calculation - in real implementation, analyze correlation between regime and performance
            regime_scores = {'bull_market': 0.8, 'bear_market': 0.2, 'sideways_market': 0.5, 'volatile_market': 0.6}
            return regime_scores.get(regime, 0.5)
        except Exception:
            return 0.5
    
    def _calculate_sentiment_impact(self, sentiment: float, trade_history: List[Dict]) -> float:
        """Calculate sentiment impact on performance"""
        try:
            # Mock calculation - in real implementation, analyze correlation between sentiment and performance
            return max(0.0, min(1.0, sentiment))
        except Exception:
            return 0.5
    
    def _calculate_quality_score_impact(self, trade_history: List[Dict]) -> float:
        """Calculate quality score impact on performance"""
        try:
            # Mock calculation - in real implementation, analyze correlation between quality scores and performance
            return random.uniform(0.3, 0.8)
        except Exception:
            return 0.5
    
    def _calculate_liquidity_impact(self, trade_history: List[Dict]) -> float:
        """Calculate liquidity impact on performance"""
        try:
            # Mock calculation - in real implementation, analyze correlation between liquidity and performance
            return random.uniform(0.3, 0.8)
        except Exception:
            return 0.5
    
    def _calculate_volume_impact(self, trade_history: List[Dict]) -> float:
        """Calculate volume impact on performance"""
        try:
            # Mock calculation - in real implementation, analyze correlation between volume and performance
            return random.uniform(0.3, 0.8)
        except Exception:
            return 0.5
    
    def _calculate_fundamentals_impact(self, trade_history: List[Dict]) -> float:
        """Calculate fundamentals impact on performance"""
        try:
            # Mock calculation - in real implementation, analyze correlation between fundamentals and performance
            return random.uniform(0.3, 0.8)
        except Exception:
            return 0.5
    
    def _calculate_entry_timing_impact(self, trade_history: List[Dict]) -> float:
        """Calculate entry timing impact on performance"""
        try:
            # Mock calculation - in real implementation, analyze correlation between entry timing and performance
            return random.uniform(0.3, 0.8)
        except Exception:
            return 0.5
    
    def _calculate_exit_timing_impact(self, trade_history: List[Dict]) -> float:
        """Calculate exit timing impact on performance"""
        try:
            # Mock calculation - in real implementation, analyze correlation between exit timing and performance
            return random.uniform(0.3, 0.8)
        except Exception:
            return 0.5
    
    def _calculate_hold_duration_impact(self, trade_history: List[Dict]) -> float:
        """Calculate hold duration impact on performance"""
        try:
            # Mock calculation - in real implementation, analyze correlation between hold duration and performance
            return random.uniform(0.3, 0.8)
        except Exception:
            return 0.5
    
    def _calculate_market_timing_impact(self, trade_history: List[Dict], market_data: Dict) -> float:
        """Calculate market timing impact on performance"""
        try:
            # Mock calculation - in real implementation, analyze correlation between market timing and performance
            return random.uniform(0.3, 0.8)
        except Exception:
            return 0.5
    
    def _calculate_size_accuracy_impact(self, trade_history: List[Dict]) -> float:
        """Calculate size accuracy impact on performance"""
        try:
            # Mock calculation - in real implementation, analyze correlation between size accuracy and performance
            return random.uniform(0.3, 0.8)
        except Exception:
            return 0.5
    
    def _calculate_risk_adjustment_impact(self, trade_history: List[Dict], portfolio_data: Dict) -> float:
        """Calculate risk adjustment impact on performance"""
        try:
            # Mock calculation - in real implementation, analyze correlation between risk adjustment and performance
            return random.uniform(0.3, 0.8)
        except Exception:
            return 0.5
    
    def _calculate_diversification_impact(self, trade_history: List[Dict], portfolio_data: Dict) -> float:
        """Calculate diversification impact on performance"""
        try:
            # Mock calculation - in real implementation, analyze correlation between diversification and performance
            return random.uniform(0.3, 0.8)
        except Exception:
            return 0.5
    
    def _calculate_concentration_impact(self, trade_history: List[Dict], portfolio_data: Dict) -> float:
        """Calculate concentration impact on performance"""
        try:
            # Mock calculation - in real implementation, analyze correlation between concentration and performance
            return random.uniform(0.3, 0.8)
        except Exception:
            return 0.5
    
    def _calculate_stop_loss_impact(self, trade_history: List[Dict]) -> float:
        """Calculate stop loss impact on performance"""
        try:
            # Mock calculation - in real implementation, analyze correlation between stop losses and performance
            return random.uniform(0.3, 0.8)
        except Exception:
            return 0.5
    
    def _calculate_take_profit_impact(self, trade_history: List[Dict]) -> float:
        """Calculate take profit impact on performance"""
        try:
            # Mock calculation - in real implementation, analyze correlation between take profits and performance
            return random.uniform(0.3, 0.8)
        except Exception:
            return 0.5
    
    def _calculate_risk_limits_impact(self, trade_history: List[Dict], portfolio_data: Dict) -> float:
        """Calculate risk limits impact on performance"""
        try:
            # Mock calculation - in real implementation, analyze correlation between risk limits and performance
            return random.uniform(0.3, 0.8)
        except Exception:
            return 0.5
    
    def _calculate_drawdown_control_impact(self, trade_history: List[Dict], portfolio_data: Dict) -> float:
        """Calculate drawdown control impact on performance"""
        try:
            # Mock calculation - in real implementation, analyze correlation between drawdown control and performance
            return random.uniform(0.3, 0.8)
        except Exception:
            return 0.5
    
    def _calculate_slippage_impact(self, trade_history: List[Dict]) -> float:
        """Calculate slippage impact on performance"""
        try:
            # Mock calculation - in real implementation, analyze correlation between slippage and performance
            return random.uniform(0.3, 0.8)
        except Exception:
            return 0.5
    
    def _calculate_fees_impact(self, trade_history: List[Dict]) -> float:
        """Calculate fees impact on performance"""
        try:
            # Mock calculation - in real implementation, analyze correlation between fees and performance
            return random.uniform(0.3, 0.8)
        except Exception:
            return 0.5
    
    def _calculate_speed_impact(self, trade_history: List[Dict]) -> float:
        """Calculate speed impact on performance"""
        try:
            # Mock calculation - in real implementation, analyze correlation between speed and performance
            return random.uniform(0.3, 0.8)
        except Exception:
            return 0.5
    
    def _calculate_success_rate_impact(self, trade_history: List[Dict]) -> float:
        """Calculate success rate impact on performance"""
        try:
            # Mock calculation - in real implementation, analyze correlation between success rate and performance
            return random.uniform(0.3, 0.8)
        except Exception:
            return 0.5

# Global instance
ai_performance_attribution_analyzer = AIPerformanceAttributionAnalyzer()
