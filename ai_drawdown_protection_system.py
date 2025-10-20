#!/usr/bin/env python3
"""
AI-Powered Drawdown Protection System for Sustainable Trading Bot
Prevents and manages drawdowns to protect capital and maintain trading discipline
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

class AIDrawdownProtectionSystem:
    def __init__(self):
        self.protection_cache = {}
        self.cache_duration = 60  # 1 minute cache for drawdown analysis
        self.drawdown_history = deque(maxlen=1000)
        self.protection_history = deque(maxlen=500)
        self.recovery_history = deque(maxlen=500)
        
        # Drawdown protection configuration
        self.drawdown_levels = {
            'minor': {
                'name': 'Minor Drawdown',
                'threshold': 0.05,  # 5% drawdown
                'severity': 'low',
                'action': 'monitor',
                'recovery_time': 1  # 1 day
            },
            'moderate': {
                'name': 'Moderate Drawdown',
                'threshold': 0.10,  # 10% drawdown
                'severity': 'medium',
                'action': 'reduce_risk',
                'recovery_time': 3  # 3 days
            },
            'significant': {
                'name': 'Significant Drawdown',
                'threshold': 0.20,  # 20% drawdown
                'severity': 'high',
                'action': 'halt_trading',
                'recovery_time': 7  # 7 days
            },
            'severe': {
                'name': 'Severe Drawdown',
                'threshold': 0.30,  # 30% drawdown
                'severity': 'critical',
                'action': 'emergency_exit',
                'recovery_time': 14  # 14 days
            }
        }
        
        # Protection triggers
        self.protection_triggers = {
            'daily_loss_limit': 0.05,  # 5% daily loss limit
            'weekly_loss_limit': 0.15,  # 15% weekly loss limit
            'monthly_loss_limit': 0.25,  # 25% monthly loss limit
            'consecutive_loss_limit': 5,  # 5 consecutive losses
            'portfolio_drawdown_limit': 0.20,  # 20% portfolio drawdown
            'single_trade_loss_limit': 0.10  # 10% single trade loss
        }
        
        # Recovery strategies
        self.recovery_strategies = {
            'conservative': {
                'name': 'Conservative Recovery',
                'position_size_multiplier': 0.5,  # 50% position size
                'risk_tolerance': 0.3,  # 30% risk tolerance
                'recovery_time': 7,  # 7 days
                'strategy': 'slow_and_steady'
            },
            'moderate': {
                'name': 'Moderate Recovery',
                'position_size_multiplier': 0.7,  # 70% position size
                'risk_tolerance': 0.5,  # 50% risk tolerance
                'recovery_time': 5,  # 5 days
                'strategy': 'balanced'
            },
            'aggressive': {
                'name': 'Aggressive Recovery',
                'position_size_multiplier': 1.0,  # 100% position size
                'risk_tolerance': 0.7,  # 70% risk tolerance
                'recovery_time': 3,  # 3 days
                'strategy': 'fast_recovery'
            }
        }
        
        # Drawdown analysis factors (must sum to 1.0)
        self.analysis_factors = {
            'portfolio_performance': 0.30,  # 30% weight for portfolio performance
            'individual_trade_performance': 0.25,  # 25% weight for individual trades
            'market_conditions': 0.20,  # 20% weight for market conditions
            'risk_metrics': 0.15,  # 15% weight for risk metrics
            'trading_frequency': 0.10  # 10% weight for trading frequency
        }
        
        # Drawdown thresholds
        self.minor_drawdown_threshold = 0.05  # 5% minor drawdown
        self.moderate_drawdown_threshold = 0.10  # 10% moderate drawdown
        self.significant_drawdown_threshold = 0.20  # 20% significant drawdown
        self.severe_drawdown_threshold = 0.30  # 30% severe drawdown
        
        # Protection activation thresholds
        self.immediate_protection_threshold = 0.15  # 15% immediate protection
        self.urgent_protection_threshold = 0.25  # 25% urgent protection
        self.emergency_protection_threshold = 0.35  # 35% emergency protection
        
        # Recovery thresholds
        self.full_recovery_threshold = 0.02  # 2% full recovery
        self.partial_recovery_threshold = 0.05  # 5% partial recovery
        self.recovery_momentum_threshold = 0.03  # 3% recovery momentum
    
    def analyze_drawdown_protection(self, portfolio_data: Dict, trade_history: List[Dict], market_data: Dict) -> Dict:
        """
        Analyze drawdown protection needs and generate protection strategies
        Returns comprehensive drawdown analysis with protection recommendations
        """
        try:
            cache_key = f"drawdown_{portfolio_data.get('timestamp', 'current')}"
            
            # Check cache
            if cache_key in self.protection_cache:
                cached_data = self.protection_cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).total_seconds() < self.cache_duration:
                    logger.debug("Using cached drawdown protection analysis")
                    return cached_data['protection_data']
            
            # Analyze drawdown components
            portfolio_performance_analysis = self._analyze_portfolio_performance(portfolio_data, trade_history)
            individual_trade_analysis = self._analyze_individual_trades(trade_history)
            market_conditions_analysis = self._analyze_market_conditions(market_data)
            risk_metrics_analysis = self._analyze_risk_metrics(portfolio_data, trade_history)
            trading_frequency_analysis = self._analyze_trading_frequency(trade_history)
            
            # Calculate current drawdown level
            current_drawdown = self._calculate_current_drawdown(portfolio_data, trade_history)
            
            # Determine drawdown severity
            drawdown_severity = self._determine_drawdown_severity(current_drawdown)
            
            # Calculate protection urgency
            protection_urgency = self._calculate_protection_urgency(
                current_drawdown, portfolio_performance_analysis, risk_metrics_analysis
            )
            
            # Generate protection strategies
            protection_strategies = self._generate_protection_strategies(
                drawdown_severity, protection_urgency, portfolio_data
            )
            
            # Calculate recovery potential
            recovery_potential = self._calculate_recovery_potential(
                current_drawdown, portfolio_performance_analysis, market_conditions_analysis
            )
            
            # Generate recovery strategies
            recovery_strategies = self._generate_recovery_strategies(
                current_drawdown, recovery_potential, portfolio_data
            )
            
            # Calculate risk adjustments
            risk_adjustments = self._calculate_risk_adjustments(
                current_drawdown, drawdown_severity, protection_urgency
            )
            
            # Generate protection recommendations
            protection_recommendations = self._generate_protection_recommendations(
                drawdown_severity, protection_urgency, protection_strategies, recovery_strategies
            )
            
            # Generate protection insights
            protection_insights = self._generate_protection_insights(
                current_drawdown, drawdown_severity, protection_urgency, recovery_potential
            )
            
            result = {
                'current_drawdown': current_drawdown,
                'drawdown_severity': drawdown_severity,
                'protection_urgency': protection_urgency,
                'protection_strategies': protection_strategies,
                'recovery_potential': recovery_potential,
                'recovery_strategies': recovery_strategies,
                'risk_adjustments': risk_adjustments,
                'portfolio_performance_analysis': portfolio_performance_analysis,
                'individual_trade_analysis': individual_trade_analysis,
                'market_conditions_analysis': market_conditions_analysis,
                'risk_metrics_analysis': risk_metrics_analysis,
                'trading_frequency_analysis': trading_frequency_analysis,
                'protection_recommendations': protection_recommendations,
                'protection_insights': protection_insights,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Cache the result
            self.protection_cache[cache_key] = {'timestamp': datetime.now(), 'protection_data': result}
            
            logger.info(f"🛡️ Drawdown protection analysis: {drawdown_severity} (drawdown: {current_drawdown:.1%})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Drawdown protection analysis failed: {e}")
            return self._get_default_drawdown_analysis(portfolio_data, trade_history, market_data)
    
    def _analyze_portfolio_performance(self, portfolio_data: Dict, trade_history: List[Dict]) -> Dict:
        """Analyze portfolio performance for drawdown assessment"""
        try:
            # Extract portfolio data
            current_value = portfolio_data.get('current_value', 10000)
            initial_value = portfolio_data.get('initial_value', 10000)
            peak_value = portfolio_data.get('peak_value', 10000)
            low_value = portfolio_data.get('low_value', 10000)
            
            # Calculate performance metrics
            total_return = (current_value - initial_value) / initial_value if initial_value > 0 else 0
            peak_return = (peak_value - initial_value) / initial_value if initial_value > 0 else 0
            current_drawdown = (peak_value - current_value) / peak_value if peak_value > 0 else 0
            max_drawdown = (peak_value - low_value) / peak_value if peak_value > 0 else 0
            
            # Calculate performance trends
            if total_return > 0.1:  # 10% gain
                performance_trend = "strong_growth"
                performance_characteristics = "excellent"
            elif total_return > 0.05:  # 5% gain
                performance_trend = "moderate_growth"
                performance_characteristics = "good"
            elif total_return > 0:  # Positive return
                performance_trend = "weak_growth"
                performance_characteristics = "fair"
            elif total_return > -0.05:  # 5% loss
                performance_trend = "weak_decline"
                performance_characteristics = "poor"
            elif total_return > -0.15:  # 15% loss
                performance_trend = "moderate_decline"
                performance_characteristics = "very_poor"
            else:  # 15%+ loss
                performance_trend = "strong_decline"
                performance_characteristics = "critical"
            
            # Calculate performance score
            performance_score = max(0.0, min(1.0, (total_return + 0.5)))  # Normalize to 0-1
            
            return {
                'total_return': total_return,
                'peak_return': peak_return,
                'current_drawdown': current_drawdown,
                'max_drawdown': max_drawdown,
                'performance_trend': performance_trend,
                'performance_characteristics': performance_characteristics,
                'performance_score': performance_score
            }
            
        except Exception:
            return {
                'total_return': 0.0,
                'peak_return': 0.0,
                'current_drawdown': 0.0,
                'max_drawdown': 0.0,
                'performance_trend': 'weak_growth',
                'performance_characteristics': 'fair',
                'performance_score': 0.5
            }
    
    def _analyze_individual_trades(self, trade_history: List[Dict]) -> Dict:
        """Analyze individual trade performance for drawdown assessment"""
        try:
            if not trade_history:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'profit_factor': 0.0,
                    'trade_score': 0.5
                }
            
            # Calculate trade statistics
            total_trades = len(trade_history)
            winning_trades = sum(1 for trade in trade_history if trade.get('pnl', 0) > 0)
            losing_trades = sum(1 for trade in trade_history if trade.get('pnl', 0) < 0)
            
            # Calculate win rate
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate average win and loss
            wins = [trade.get('pnl', 0) for trade in trade_history if trade.get('pnl', 0) > 0]
            losses = [abs(trade.get('pnl', 0)) for trade in trade_history if trade.get('pnl', 0) < 0]
            
            avg_win = statistics.mean(wins) if wins else 0
            avg_loss = statistics.mean(losses) if losses else 0
            
            # Calculate profit factor
            total_wins = sum(wins) if wins else 0
            total_losses = sum(losses) if losses else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0
            
            # Calculate trade score
            trade_score = (
                win_rate * 0.4 +
                min(1.0, profit_factor / 2.0) * 0.3 +
                min(1.0, avg_win / 100) * 0.2 +
                min(1.0, 1.0 - avg_loss / 100) * 0.1
            )
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'trade_score': trade_score
            }
            
        except Exception:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'trade_score': 0.5
            }
    
    def _analyze_market_conditions(self, market_data: Dict) -> Dict:
        """Analyze market conditions for drawdown assessment"""
        try:
            # Extract market data
            current_volatility = market_data.get('current_volatility', 0.3)
            market_sentiment = market_data.get('market_sentiment', 0.5)
            market_trend = market_data.get('market_trend', 'neutral')
            market_regime = market_data.get('market_regime', 'sideways')
            
            # Calculate market risk
            volatility_risk = min(1.0, current_volatility * 2.0)  # Normalize to 0-1
            sentiment_risk = abs(market_sentiment - 0.5) * 2.0  # Distance from neutral
            
            # Determine market risk level
            if volatility_risk > 0.8 and sentiment_risk > 0.6:  # High volatility and extreme sentiment
                market_risk_level = "very_high"
                market_characteristics = "extremely_risky"
            elif volatility_risk > 0.6 and sentiment_risk > 0.4:  # Medium-high volatility and sentiment
                market_risk_level = "high"
                market_characteristics = "very_risky"
            elif volatility_risk > 0.4 and sentiment_risk > 0.2:  # Medium volatility and sentiment
                market_risk_level = "medium"
                market_characteristics = "moderately_risky"
            elif volatility_risk > 0.2 and sentiment_risk > 0.1:  # Low volatility and sentiment
                market_risk_level = "low"
                market_characteristics = "somewhat_risky"
            else:
                market_risk_level = "very_low"
                market_characteristics = "low_risk"
            
            # Calculate market score
            market_score = max(0.0, min(1.0, 1.0 - (volatility_risk + sentiment_risk) / 2.0))
            
            return {
                'current_volatility': current_volatility,
                'market_sentiment': market_sentiment,
                'market_trend': market_trend,
                'market_regime': market_regime,
                'volatility_risk': volatility_risk,
                'sentiment_risk': sentiment_risk,
                'market_risk_level': market_risk_level,
                'market_characteristics': market_characteristics,
                'market_score': market_score
            }
            
        except Exception:
            return {
                'current_volatility': 0.3,
                'market_sentiment': 0.5,
                'market_trend': 'neutral',
                'market_regime': 'sideways',
                'volatility_risk': 0.3,
                'sentiment_risk': 0.2,
                'market_risk_level': 'medium',
                'market_characteristics': 'moderately_risky',
                'market_score': 0.5
            }
    
    def _analyze_risk_metrics(self, portfolio_data: Dict, trade_history: List[Dict]) -> Dict:
        """Analyze risk metrics for drawdown assessment"""
        try:
            # Extract risk data
            current_value = portfolio_data.get('current_value', 10000)
            initial_value = portfolio_data.get('initial_value', 10000)
            peak_value = portfolio_data.get('peak_value', 10000)
            
            # Calculate risk metrics
            current_drawdown = (peak_value - current_value) / peak_value if peak_value > 0 else 0
            max_drawdown = portfolio_data.get('max_drawdown', 0.0)
            var_95 = portfolio_data.get('var_95', 0.05)  # 5% VaR
            var_99 = portfolio_data.get('var_99', 0.10)  # 10% VaR
            
            # Calculate risk score
            risk_score = (
                current_drawdown * 0.4 +
                max_drawdown * 0.3 +
                var_95 * 0.2 +
                var_99 * 0.1
            )
            
            # Determine risk level
            if risk_score > 0.3:  # 30% risk score
                risk_level = "very_high"
                risk_characteristics = "extremely_risky"
            elif risk_score > 0.2:  # 20% risk score
                risk_level = "high"
                risk_characteristics = "very_risky"
            elif risk_score > 0.1:  # 10% risk score
                risk_level = "medium"
                risk_characteristics = "moderately_risky"
            elif risk_score > 0.05:  # 5% risk score
                risk_level = "low"
                risk_characteristics = "somewhat_risky"
            else:
                risk_level = "very_low"
                risk_characteristics = "low_risk"
            
            return {
                'current_drawdown': current_drawdown,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'var_99': var_99,
                'risk_score': risk_score,
                'risk_level': risk_level,
                'risk_characteristics': risk_characteristics
            }
            
        except Exception:
            return {
                'current_drawdown': 0.0,
                'max_drawdown': 0.0,
                'var_95': 0.05,
                'var_99': 0.10,
                'risk_score': 0.05,
                'risk_level': 'low',
                'risk_characteristics': 'low_risk'
            }
    
    def _analyze_trading_frequency(self, trade_history: List[Dict]) -> Dict:
        """Analyze trading frequency for drawdown assessment"""
        try:
            if not trade_history:
                return {
                    'trades_per_day': 0,
                    'trades_per_week': 0,
                    'trades_per_month': 0,
                    'frequency_level': 'low',
                    'frequency_characteristics': 'conservative',
                    'frequency_score': 0.5
                }
            
            # Calculate trading frequency
            total_trades = len(trade_history)
            days_trading = 30  # Assume 30 days of trading
            weeks_trading = 4  # Assume 4 weeks of trading
            months_trading = 1  # Assume 1 month of trading
            
            trades_per_day = total_trades / days_trading if days_trading > 0 else 0
            trades_per_week = total_trades / weeks_trading if weeks_trading > 0 else 0
            trades_per_month = total_trades / months_trading if months_trading > 0 else 0
            
            # Determine frequency level
            if trades_per_day > 10:  # 10+ trades per day
                frequency_level = "very_high"
                frequency_characteristics = "hyperactive"
            elif trades_per_day > 5:  # 5+ trades per day
                frequency_level = "high"
                frequency_characteristics = "very_active"
            elif trades_per_day > 2:  # 2+ trades per day
                frequency_level = "medium"
                frequency_characteristics = "active"
            elif trades_per_day > 0.5:  # 0.5+ trades per day
                frequency_level = "low"
                frequency_characteristics = "moderate"
            else:
                frequency_level = "very_low"
                frequency_characteristics = "conservative"
            
            # Calculate frequency score
            frequency_score = min(1.0, trades_per_day / 5.0)  # Normalize to 0-1
            
            return {
                'trades_per_day': trades_per_day,
                'trades_per_week': trades_per_week,
                'trades_per_month': trades_per_month,
                'frequency_level': frequency_level,
                'frequency_characteristics': frequency_characteristics,
                'frequency_score': frequency_score
            }
            
        except Exception:
            return {
                'trades_per_day': 0,
                'trades_per_week': 0,
                'trades_per_month': 0,
                'frequency_level': 'low',
                'frequency_characteristics': 'conservative',
                'frequency_score': 0.5
            }
    
    def _calculate_current_drawdown(self, portfolio_data: Dict, trade_history: List[Dict]) -> float:
        """Calculate current drawdown percentage"""
        try:
            current_value = portfolio_data.get('current_value', 10000)
            peak_value = portfolio_data.get('peak_value', 10000)
            
            if peak_value > 0:
                return (peak_value - current_value) / peak_value
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _determine_drawdown_severity(self, current_drawdown: float) -> str:
        """Determine drawdown severity level"""
        try:
            if current_drawdown >= self.severe_drawdown_threshold:  # 30%+ drawdown
                return "severe"
            elif current_drawdown >= self.significant_drawdown_threshold:  # 20%+ drawdown
                return "significant"
            elif current_drawdown >= self.moderate_drawdown_threshold:  # 10%+ drawdown
                return "moderate"
            elif current_drawdown >= self.minor_drawdown_threshold:  # 5%+ drawdown
                return "minor"
            else:
                return "none"
                
        except Exception:
            return "none"
    
    def _calculate_protection_urgency(self, current_drawdown: float, portfolio_performance_analysis: Dict,
                                    risk_metrics_analysis: Dict) -> str:
        """Calculate protection urgency level"""
        try:
            # Base urgency from drawdown level
            if current_drawdown >= self.emergency_protection_threshold:  # 35%+ drawdown
                base_urgency = "emergency"
            elif current_drawdown >= self.urgent_protection_threshold:  # 25%+ drawdown
                base_urgency = "urgent"
            elif current_drawdown >= self.immediate_protection_threshold:  # 15%+ drawdown
                base_urgency = "immediate"
            else:
                base_urgency = "monitor"
            
            # Adjust based on performance trend
            performance_trend = portfolio_performance_analysis.get('performance_trend', 'weak_growth')
            if performance_trend in ['strong_decline', 'moderate_decline']:
                if base_urgency == "monitor":
                    base_urgency = "immediate"
                elif base_urgency == "immediate":
                    base_urgency = "urgent"
            
            # Adjust based on risk level
            risk_level = risk_metrics_analysis.get('risk_level', 'low')
            if risk_level in ['very_high', 'high']:
                if base_urgency == "monitor":
                    base_urgency = "immediate"
                elif base_urgency == "immediate":
                    base_urgency = "urgent"
            
            return base_urgency
            
        except Exception:
            return "monitor"
    
    def _generate_protection_strategies(self, drawdown_severity: str, protection_urgency: str,
                                      portfolio_data: Dict) -> List[Dict]:
        """Generate protection strategies based on drawdown severity and urgency"""
        strategies = []
        
        try:
            # Immediate protection strategies
            if protection_urgency == "emergency":
                strategies.append({
                    'strategy': 'emergency_exit',
                    'description': 'Exit all positions immediately',
                    'priority': 'critical',
                    'action': 'halt_all_trading'
                })
                strategies.append({
                    'strategy': 'capital_preservation',
                    'description': 'Focus on capital preservation',
                    'priority': 'critical',
                    'action': 'reduce_risk_to_zero'
                })
            elif protection_urgency == "urgent":
                strategies.append({
                    'strategy': 'halt_trading',
                    'description': 'Stop all new trades',
                    'priority': 'high',
                    'action': 'pause_trading'
                })
                strategies.append({
                    'strategy': 'reduce_positions',
                    'description': 'Reduce existing positions',
                    'priority': 'high',
                    'action': 'cut_position_sizes'
                })
            elif protection_urgency == "immediate":
                strategies.append({
                    'strategy': 'reduce_risk',
                    'description': 'Reduce risk exposure',
                    'priority': 'medium',
                    'action': 'lower_position_sizes'
                })
                strategies.append({
                    'strategy': 'tighten_stops',
                    'description': 'Tighten stop losses',
                    'priority': 'medium',
                    'action': 'reduce_stop_losses'
                })
            else:  # monitor
                strategies.append({
                    'strategy': 'monitor_closely',
                    'description': 'Monitor portfolio closely',
                    'priority': 'low',
                    'action': 'increase_monitoring'
                })
            
            # Severity-based strategies
            if drawdown_severity == "severe":
                strategies.append({
                    'strategy': 'emergency_recovery',
                    'description': 'Implement emergency recovery plan',
                    'priority': 'critical',
                    'action': 'activate_emergency_protocols'
                })
            elif drawdown_severity == "significant":
                strategies.append({
                    'strategy': 'aggressive_protection',
                    'description': 'Implement aggressive protection measures',
                    'priority': 'high',
                    'action': 'activate_aggressive_protection'
                })
            elif drawdown_severity == "moderate":
                strategies.append({
                    'strategy': 'moderate_protection',
                    'description': 'Implement moderate protection measures',
                    'priority': 'medium',
                    'action': 'activate_moderate_protection'
                })
            else:  # minor or none
                strategies.append({
                    'strategy': 'preventive_protection',
                    'description': 'Implement preventive protection measures',
                    'priority': 'low',
                    'action': 'activate_preventive_protection'
                })
            
        except Exception:
            strategies.append({
                'strategy': 'monitor',
                'description': 'Monitor portfolio conditions',
                'priority': 'low',
                'action': 'monitor'
            })
        
        return strategies
    
    def _calculate_recovery_potential(self, current_drawdown: float, portfolio_performance_analysis: Dict,
                                    market_conditions_analysis: Dict) -> float:
        """Calculate recovery potential score"""
        try:
            # Base recovery potential from current drawdown
            base_potential = max(0.0, 1.0 - current_drawdown)  # Higher drawdown = lower potential
            
            # Adjust based on performance trend
            performance_trend = portfolio_performance_analysis.get('performance_trend', 'weak_growth')
            if performance_trend == 'strong_growth':
                performance_factor = 1.2
            elif performance_trend == 'moderate_growth':
                performance_factor = 1.1
            elif performance_trend == 'weak_growth':
                performance_factor = 1.0
            elif performance_trend == 'weak_decline':
                performance_factor = 0.9
            elif performance_trend == 'moderate_decline':
                performance_factor = 0.8
            else:  # strong_decline
                performance_factor = 0.7
            
            # Adjust based on market conditions
            market_score = market_conditions_analysis.get('market_score', 0.5)
            market_factor = 0.5 + market_score  # 0.5 to 1.5 range
            
            # Calculate final recovery potential
            recovery_potential = base_potential * performance_factor * market_factor
            
            return max(0.0, min(1.0, recovery_potential))
            
        except Exception:
            return 0.5
    
    def _generate_recovery_strategies(self, current_drawdown: float, recovery_potential: float,
                                    portfolio_data: Dict) -> List[Dict]:
        """Generate recovery strategies based on drawdown and recovery potential"""
        strategies = []
        
        try:
            # Recovery strategy based on potential
            if recovery_potential > 0.8:  # 80% recovery potential
                strategy_type = "aggressive"
                strategy_config = self.recovery_strategies['aggressive']
            elif recovery_potential > 0.6:  # 60% recovery potential
                strategy_type = "moderate"
                strategy_config = self.recovery_strategies['moderate']
            else:  # Below 60% recovery potential
                strategy_type = "conservative"
                strategy_config = self.recovery_strategies['conservative']
            
            strategies.append({
                'strategy': f'{strategy_type}_recovery',
                'description': strategy_config['name'],
                'position_size_multiplier': strategy_config['position_size_multiplier'],
                'risk_tolerance': strategy_config['risk_tolerance'],
                'recovery_time': strategy_config['recovery_time'],
                'approach': strategy_config['strategy']
            })
            
            # Additional recovery strategies based on drawdown level
            if current_drawdown > 0.2:  # 20%+ drawdown
                strategies.append({
                    'strategy': 'gradual_recovery',
                    'description': 'Gradual position size increase',
                    'position_size_multiplier': 0.3,
                    'risk_tolerance': 0.2,
                    'recovery_time': 14,
                    'approach': 'very_conservative'
                })
            elif current_drawdown > 0.1:  # 10%+ drawdown
                strategies.append({
                    'strategy': 'balanced_recovery',
                    'description': 'Balanced position size increase',
                    'position_size_multiplier': 0.5,
                    'risk_tolerance': 0.4,
                    'recovery_time': 7,
                    'approach': 'conservative'
                })
            else:  # Below 10% drawdown
                strategies.append({
                    'strategy': 'normal_recovery',
                    'description': 'Normal position size increase',
                    'position_size_multiplier': 0.8,
                    'risk_tolerance': 0.6,
                    'recovery_time': 3,
                    'approach': 'moderate'
                })
            
        except Exception:
            strategies.append({
                'strategy': 'monitor_recovery',
                'description': 'Monitor recovery conditions',
                'position_size_multiplier': 0.5,
                'risk_tolerance': 0.3,
                'recovery_time': 7,
                'approach': 'conservative'
            })
        
        return strategies
    
    def _calculate_risk_adjustments(self, current_drawdown: float, drawdown_severity: str,
                                  protection_urgency: str) -> Dict:
        """Calculate risk adjustments based on drawdown analysis"""
        try:
            # Base risk adjustments
            base_position_size = 1.0
            base_risk_tolerance = 1.0
            base_stop_loss = 0.05  # 5% stop loss
            
            # Adjust based on drawdown severity
            if drawdown_severity == "severe":
                position_size_multiplier = 0.1  # 10% position size
                risk_tolerance_multiplier = 0.1  # 10% risk tolerance
                stop_loss_multiplier = 0.5  # 2.5% stop loss
            elif drawdown_severity == "significant":
                position_size_multiplier = 0.3  # 30% position size
                risk_tolerance_multiplier = 0.3  # 30% risk tolerance
                stop_loss_multiplier = 0.7  # 3.5% stop loss
            elif drawdown_severity == "moderate":
                position_size_multiplier = 0.5  # 50% position size
                risk_tolerance_multiplier = 0.5  # 50% risk tolerance
                stop_loss_multiplier = 0.8  # 4% stop loss
            elif drawdown_severity == "minor":
                position_size_multiplier = 0.7  # 70% position size
                risk_tolerance_multiplier = 0.7  # 70% risk tolerance
                stop_loss_multiplier = 0.9  # 4.5% stop loss
            else:  # none
                position_size_multiplier = 1.0  # 100% position size
                risk_tolerance_multiplier = 1.0  # 100% risk tolerance
                stop_loss_multiplier = 1.0  # 5% stop loss
            
            # Adjust based on protection urgency
            if protection_urgency == "emergency":
                position_size_multiplier *= 0.1  # 10% of base
                risk_tolerance_multiplier *= 0.1  # 10% of base
                stop_loss_multiplier *= 0.5  # 50% of base
            elif protection_urgency == "urgent":
                position_size_multiplier *= 0.3  # 30% of base
                risk_tolerance_multiplier *= 0.3  # 30% of base
                stop_loss_multiplier *= 0.7  # 70% of base
            elif protection_urgency == "immediate":
                position_size_multiplier *= 0.5  # 50% of base
                risk_tolerance_multiplier *= 0.5  # 50% of base
                stop_loss_multiplier *= 0.8  # 80% of base
            
            # Calculate final adjustments
            adjusted_position_size = base_position_size * position_size_multiplier
            adjusted_risk_tolerance = base_risk_tolerance * risk_tolerance_multiplier
            adjusted_stop_loss = base_stop_loss * stop_loss_multiplier
            
            return {
                'position_size_multiplier': position_size_multiplier,
                'risk_tolerance_multiplier': risk_tolerance_multiplier,
                'stop_loss_multiplier': stop_loss_multiplier,
                'adjusted_position_size': adjusted_position_size,
                'adjusted_risk_tolerance': adjusted_risk_tolerance,
                'adjusted_stop_loss': adjusted_stop_loss
            }
            
        except Exception:
            return {
                'position_size_multiplier': 1.0,
                'risk_tolerance_multiplier': 1.0,
                'stop_loss_multiplier': 1.0,
                'adjusted_position_size': 1.0,
                'adjusted_risk_tolerance': 1.0,
                'adjusted_stop_loss': 0.05
            }
    
    def _generate_protection_recommendations(self, drawdown_severity: str, protection_urgency: str,
                                          protection_strategies: List[Dict], recovery_strategies: List[Dict]) -> List[str]:
        """Generate protection recommendations"""
        recommendations = []
        
        try:
            # Severity-based recommendations
            if drawdown_severity == "severe":
                recommendations.append("SEVERE DRAWDOWN DETECTED - Immediate action required")
                recommendations.append("Exit all positions immediately")
                recommendations.append("Halt all trading activities")
                recommendations.append("Focus on capital preservation")
            elif drawdown_severity == "significant":
                recommendations.append("SIGNIFICANT DRAWDOWN DETECTED - Urgent action required")
                recommendations.append("Reduce position sizes significantly")
                recommendations.append("Tighten stop losses")
                recommendations.append("Consider exiting losing positions")
            elif drawdown_severity == "moderate":
                recommendations.append("MODERATE DRAWDOWN DETECTED - Caution required")
                recommendations.append("Reduce position sizes")
                recommendations.append("Monitor positions closely")
                recommendations.append("Avoid new high-risk trades")
            elif drawdown_severity == "minor":
                recommendations.append("MINOR DRAWDOWN DETECTED - Monitor closely")
                recommendations.append("Consider reducing position sizes slightly")
                recommendations.append("Monitor for further deterioration")
            else:
                recommendations.append("No significant drawdown detected")
                recommendations.append("Continue normal trading operations")
                recommendations.append("Monitor for early warning signs")
            
            # Urgency-based recommendations
            if protection_urgency == "emergency":
                recommendations.append("EMERGENCY PROTECTION ACTIVATED")
                recommendations.append("All trading halted")
                recommendations.append("Emergency protocols activated")
            elif protection_urgency == "urgent":
                recommendations.append("URGENT PROTECTION REQUIRED")
                recommendations.append("Reduce risk exposure immediately")
                recommendations.append("Consider exiting positions")
            elif protection_urgency == "immediate":
                recommendations.append("IMMEDIATE PROTECTION REQUIRED")
                recommendations.append("Reduce position sizes")
                recommendations.append("Tighten risk management")
            else:
                recommendations.append("Monitor protection indicators")
                recommendations.append("Prepare for potential protection measures")
            
            # Strategy-based recommendations
            for strategy in protection_strategies[:3]:  # Top 3 strategies
                recommendations.append(f"Protection strategy: {strategy['description']}")
            
            for strategy in recovery_strategies[:2]:  # Top 2 recovery strategies
                recommendations.append(f"Recovery strategy: {strategy['description']}")
            
        except Exception:
            recommendations.append("Monitor drawdown conditions and adjust strategy accordingly")
        
        return recommendations
    
    def _generate_protection_insights(self, current_drawdown: float, drawdown_severity: str,
                                    protection_urgency: str, recovery_potential: float) -> List[str]:
        """Generate protection insights"""
        insights = []
        
        try:
            # Drawdown insights
            insights.append(f"Current drawdown: {current_drawdown:.1%}")
            insights.append(f"Drawdown severity: {drawdown_severity}")
            insights.append(f"Protection urgency: {protection_urgency}")
            insights.append(f"Recovery potential: {recovery_potential:.1%}")
            
            # Severity insights
            if drawdown_severity == "severe":
                insights.append("Critical drawdown level - immediate action required")
            elif drawdown_severity == "significant":
                insights.append("High drawdown level - urgent action required")
            elif drawdown_severity == "moderate":
                insights.append("Medium drawdown level - caution required")
            elif drawdown_severity == "minor":
                insights.append("Low drawdown level - monitor closely")
            else:
                insights.append("No significant drawdown - normal operations")
            
            # Urgency insights
            if protection_urgency == "emergency":
                insights.append("Emergency protection protocols activated")
            elif protection_urgency == "urgent":
                insights.append("Urgent protection measures required")
            elif protection_urgency == "immediate":
                insights.append("Immediate protection measures required")
            else:
                insights.append("Monitoring protection indicators")
            
            # Recovery insights
            if recovery_potential > 0.8:
                insights.append("High recovery potential - optimistic outlook")
            elif recovery_potential > 0.6:
                insights.append("Good recovery potential - positive outlook")
            elif recovery_potential > 0.4:
                insights.append("Moderate recovery potential - cautious outlook")
            else:
                insights.append("Low recovery potential - conservative approach required")
            
        except Exception:
            insights.append("Drawdown protection analysis completed")
        
        return insights
    
    def _get_default_drawdown_analysis(self, portfolio_data: Dict, trade_history: List[Dict], market_data: Dict) -> Dict:
        """Return default drawdown analysis when analysis fails"""
        return {
            'current_drawdown': 0.0,
            'drawdown_severity': 'none',
            'protection_urgency': 'monitor',
            'protection_strategies': [{'strategy': 'monitor', 'description': 'Monitor portfolio conditions', 'priority': 'low', 'action': 'monitor'}],
            'recovery_potential': 0.5,
            'recovery_strategies': [{'strategy': 'monitor_recovery', 'description': 'Monitor recovery conditions', 'position_size_multiplier': 0.5, 'risk_tolerance': 0.3, 'recovery_time': 7, 'approach': 'conservative'}],
            'risk_adjustments': {'position_size_multiplier': 1.0, 'risk_tolerance_multiplier': 1.0, 'stop_loss_multiplier': 1.0, 'adjusted_position_size': 1.0, 'adjusted_risk_tolerance': 1.0, 'adjusted_stop_loss': 0.05},
            'portfolio_performance_analysis': {'performance_score': 0.5},
            'individual_trade_analysis': {'trade_score': 0.5},
            'market_conditions_analysis': {'market_score': 0.5},
            'risk_metrics_analysis': {'risk_level': 'low'},
            'trading_frequency_analysis': {'frequency_score': 0.5},
            'protection_recommendations': ['Monitor drawdown conditions'],
            'protection_insights': ['Drawdown protection analysis completed'],
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def get_drawdown_summary(self, portfolio_data_list: List[Dict], trade_history_list: List[List[Dict]], market_data_list: List[Dict]) -> Dict:
        """Get drawdown summary for multiple portfolios"""
        try:
            drawdown_summaries = []
            severe_drawdowns = 0
            significant_drawdowns = 0
            moderate_drawdowns = 0
            minor_drawdowns = 0
            no_drawdowns = 0
            
            for i, portfolio_data in enumerate(portfolio_data_list):
                trade_history = trade_history_list[i] if i < len(trade_history_list) else []
                market_data = market_data_list[i] if i < len(market_data_list) else {}
                
                analysis = self.analyze_drawdown_protection(portfolio_data, trade_history, market_data)
                
                drawdown_summaries.append({
                    'current_drawdown': analysis['current_drawdown'],
                    'drawdown_severity': analysis['drawdown_severity'],
                    'protection_urgency': analysis['protection_urgency'],
                    'recovery_potential': analysis['recovery_potential']
                })
                
                severity = analysis['drawdown_severity']
                if severity == 'severe':
                    severe_drawdowns += 1
                elif severity == 'significant':
                    significant_drawdowns += 1
                elif severity == 'moderate':
                    moderate_drawdowns += 1
                elif severity == 'minor':
                    minor_drawdowns += 1
                else:
                    no_drawdowns += 1
            
            return {
                'total_portfolios': len(portfolio_data_list),
                'severe_drawdowns': severe_drawdowns,
                'significant_drawdowns': significant_drawdowns,
                'moderate_drawdowns': moderate_drawdowns,
                'minor_drawdowns': minor_drawdowns,
                'no_drawdowns': no_drawdowns,
                'drawdown_summaries': drawdown_summaries,
                'overall_drawdown_risk': 'critical' if severe_drawdowns > 0 else 'high' if significant_drawdowns > 0 else 'medium' if moderate_drawdowns > 0 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error getting drawdown summary: {e}")
            return {
                'total_portfolios': len(portfolio_data_list),
                'severe_drawdowns': 0,
                'significant_drawdowns': 0,
                'moderate_drawdowns': 0,
                'minor_drawdowns': 0,
                'no_drawdowns': 0,
                'drawdown_summaries': [],
                'overall_drawdown_risk': 'unknown'
            }

# Global instance
ai_drawdown_protection_system = AIDrawdownProtectionSystem()
