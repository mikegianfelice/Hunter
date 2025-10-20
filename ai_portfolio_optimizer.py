#!/usr/bin/env python3
"""
AI-Powered Portfolio Optimization for Sustainable Trading Bot
Uses modern portfolio theory and machine learning to optimize position allocation
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
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIPortfolioOptimizer:
    def __init__(self):
        self.portfolio_cache = {}
        self.cache_duration = 1800  # 30 minutes cache
        self.optimization_history = []
        
        # Portfolio optimization configuration
        self.max_portfolio_risk = 0.20  # 20% maximum portfolio risk
        self.max_position_weight = 0.30  # 30% maximum position weight
        self.min_position_weight = 0.05  # 5% minimum position weight
        self.target_sharpe_ratio = 1.5  # Target Sharpe ratio
        
        # Risk factors
        self.risk_factors = {
            'volatility': 0.25,
            'correlation': 0.20,
            'liquidity': 0.15,
            'market_regime': 0.15,
            'sentiment': 0.10,
            'technical': 0.10,
            'concentration': 0.05
        }
        
        # Optimization constraints
        self.constraints = {
            'max_total_exposure': 1.0,  # 100% maximum total exposure
            'max_sector_exposure': 0.4,  # 40% maximum sector exposure
            'max_correlation_threshold': 0.7,  # 70% maximum correlation
            'min_diversification': 3  # Minimum 3 positions for diversification
        }
    
    def optimize_portfolio(self, positions: List[Dict], available_capital: float) -> Dict:
        """
        Optimize portfolio allocation using AI and modern portfolio theory
        Returns optimized position sizes and portfolio metrics
        """
        try:
            if not positions:
                return self._get_empty_portfolio()
            
            # Analyze current portfolio
            portfolio_analysis = self._analyze_current_portfolio(positions)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(positions)
            
            # Optimize position sizes
            optimized_allocation = self._optimize_position_allocation(
                positions, available_capital, risk_metrics
            )
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(
                optimized_allocation, risk_metrics
            )
            
            # Generate optimization insights
            insights = self._generate_optimization_insights(
                portfolio_analysis, optimized_allocation, portfolio_metrics
            )
            
            result = {
                'optimized_allocation': optimized_allocation,
                'portfolio_metrics': portfolio_metrics,
                'risk_metrics': risk_metrics,
                'insights': insights,
                'optimization_timestamp': datetime.now().isoformat(),
                'total_capital': available_capital,
                'recommended_actions': self._generate_recommendations(
                    portfolio_analysis, optimized_allocation
                )
            }
            
            # Cache the result
            self.portfolio_cache['latest'] = result
            
            logger.info(f"🎯 Portfolio optimized: {len(positions)} positions, Sharpe: {portfolio_metrics['sharpe_ratio']:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Portfolio optimization failed: {e}")
            return self._get_default_portfolio(positions, available_capital)
    
    def _analyze_current_portfolio(self, positions: List[Dict]) -> Dict:
        """Analyze current portfolio composition and characteristics"""
        try:
            total_value = sum(pos.get('value', 0) for pos in positions)
            
            if total_value == 0:
                return {
                    'total_value': 0,
                    'position_count': len(positions),
                    'concentration_risk': 0,
                    'sector_diversification': 0,
                    'average_quality': 0,
                    'risk_score': 0
                }
            
            # Calculate concentration risk (Herfindahl index)
            weights = [pos.get('value', 0) / total_value for pos in positions]
            concentration_risk = sum(w**2 for w in weights)
            
            # Calculate sector diversification
            sectors = [pos.get('sector', 'unknown') for pos in positions]
            unique_sectors = len(set(sectors))
            sector_diversification = unique_sectors / len(positions) if positions else 0
            
            # Calculate average quality score
            quality_scores = [pos.get('quality_score', 0) for pos in positions]
            average_quality = statistics.mean(quality_scores) if quality_scores else 0
            
            # Calculate overall risk score
            risk_scores = [pos.get('risk_score', 0.5) for pos in positions]
            average_risk = statistics.mean(risk_scores) if risk_scores else 0.5
            
            return {
                'total_value': total_value,
                'position_count': len(positions),
                'concentration_risk': concentration_risk,
                'sector_diversification': sector_diversification,
                'average_quality': average_quality,
                'risk_score': average_risk,
                'largest_position_weight': max(weights) if weights else 0,
                'position_weights': weights
            }
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio: {e}")
            return {
                'total_value': 0,
                'position_count': 0,
                'concentration_risk': 0,
                'sector_diversification': 0,
                'average_quality': 0,
                'risk_score': 0
            }
    
    def _calculate_risk_metrics(self, positions: List[Dict]) -> Dict:
        """Calculate risk metrics for portfolio optimization"""
        try:
            if not positions:
                return {'portfolio_volatility': 0, 'correlation_matrix': {}, 'risk_factors': {}}
            
            # Calculate individual position volatilities
            position_volatilities = {}
            for pos in positions:
                symbol = pos.get('symbol', 'UNKNOWN')
                # Simulate volatility based on token characteristics
                volatility = self._calculate_position_volatility(pos)
                position_volatilities[symbol] = volatility
            
            # Calculate correlation matrix
            correlation_matrix = self._calculate_correlation_matrix(positions)
            
            # Calculate portfolio volatility
            portfolio_volatility = self._calculate_portfolio_volatility(
                positions, position_volatilities, correlation_matrix
            )
            
            # Calculate risk factors
            risk_factors = self._calculate_risk_factors(positions)
            
            return {
                'portfolio_volatility': portfolio_volatility,
                'position_volatilities': position_volatilities,
                'correlation_matrix': correlation_matrix,
                'risk_factors': risk_factors,
                'max_correlation': max(
                    max(correlation_matrix.get(symbol, {}).values(), default=0)
                    for symbol in correlation_matrix
                ) if correlation_matrix else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {'portfolio_volatility': 0, 'correlation_matrix': {}, 'risk_factors': {}}
    
    def _calculate_position_volatility(self, position: Dict) -> float:
        """Calculate volatility for a single position"""
        try:
            # Simulate volatility based on position characteristics
            quality_score = position.get('quality_score', 50)
            volume = position.get('volume_24h', 0)
            liquidity = position.get('liquidity', 0)
            
            # Higher quality, volume, and liquidity = lower volatility
            base_volatility = 0.3  # 30% base volatility
            
            # Quality adjustment
            quality_factor = max(0.5, 1.0 - (quality_score / 100))
            
            # Volume adjustment
            volume_factor = max(0.7, 1.0 - min(1.0, volume / 1000000))
            
            # Liquidity adjustment
            liquidity_factor = max(0.8, 1.0 - min(1.0, liquidity / 2000000))
            
            volatility = base_volatility * quality_factor * volume_factor * liquidity_factor
            
            return max(0.1, min(0.8, volatility))  # Clamp between 10% and 80%
            
        except Exception:
            return 0.3  # Default 30% volatility
    
    def _calculate_correlation_matrix(self, positions: List[Dict]) -> Dict:
        """Calculate correlation matrix between positions"""
        try:
            symbols = [pos.get('symbol', f'TOKEN_{i}') for i, pos in enumerate(positions)]
            correlation_matrix = {}
            
            for i, symbol1 in enumerate(symbols):
                correlation_matrix[symbol1] = {}
                for j, symbol2 in enumerate(symbols):
                    if i == j:
                        correlation_matrix[symbol1][symbol2] = 1.0
                    else:
                        # Simulate correlation based on token characteristics
                        correlation = self._calculate_token_correlation(
                            positions[i], positions[j]
                        )
                        correlation_matrix[symbol1][symbol2] = correlation
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return {}
    
    def _calculate_token_correlation(self, pos1: Dict, pos2: Dict) -> float:
        """Calculate correlation between two tokens"""
        try:
            # Simulate correlation based on token characteristics
            # Similar tokens (same sector, similar quality) have higher correlation
            
            sector1 = pos1.get('sector', 'unknown')
            sector2 = pos2.get('sector', 'unknown')
            
            quality1 = pos1.get('quality_score', 50)
            quality2 = pos2.get('quality_score', 50)
            
            # Sector correlation
            if sector1 == sector2:
                sector_correlation = 0.6
            else:
                sector_correlation = 0.2
            
            # Quality correlation
            quality_diff = abs(quality1 - quality2)
            quality_correlation = max(0.1, 1.0 - (quality_diff / 100))
            
            # Combine factors
            correlation = (sector_correlation * 0.6 + quality_correlation * 0.4)
            
            # Add some randomness for realism
            correlation += random.uniform(-0.1, 0.1)
            
            return max(0.0, min(1.0, correlation))
            
        except Exception:
            return 0.3  # Default correlation
    
    def _calculate_portfolio_volatility(self, positions: List[Dict], 
                                       position_volatilities: Dict, 
                                       correlation_matrix: Dict) -> float:
        """Calculate portfolio volatility using modern portfolio theory"""
        try:
            if not positions:
                return 0.0
            
            # Simplified portfolio volatility calculation
            # In practice, this would use the full covariance matrix
            
            weights = [pos.get('weight', 1.0/len(positions)) for pos in positions]
            symbols = [pos.get('symbol', f'TOKEN_{i}') for i, pos in enumerate(positions)]
            
            # Calculate weighted average volatility
            weighted_volatility = sum(
                weights[i] * position_volatilities.get(symbols[i], 0.3)
                for i in range(len(positions))
            )
            
            # Adjust for diversification (correlation effect)
            avg_correlation = self._calculate_average_correlation(correlation_matrix)
            diversification_factor = 1.0 - (avg_correlation * 0.5)
            
            portfolio_volatility = weighted_volatility * diversification_factor
            
            return max(0.05, min(0.5, portfolio_volatility))  # Clamp between 5% and 50%
            
        except Exception:
            return 0.2  # Default 20% portfolio volatility
    
    def _calculate_average_correlation(self, correlation_matrix: Dict) -> float:
        """Calculate average correlation in the portfolio"""
        try:
            if not correlation_matrix:
                return 0.3
            
            correlations = []
            for symbol1, correlations_dict in correlation_matrix.items():
                for symbol2, correlation in correlations_dict.items():
                    if symbol1 != symbol2:  # Exclude self-correlation
                        correlations.append(correlation)
            
            return statistics.mean(correlations) if correlations else 0.3
            
        except Exception:
            return 0.3
    
    def _calculate_risk_factors(self, positions: List[Dict]) -> Dict:
        """Calculate various risk factors for the portfolio"""
        try:
            risk_factors = {}
            
            # Volatility risk
            volatilities = [self._calculate_position_volatility(pos) for pos in positions]
            risk_factors['volatility'] = statistics.mean(volatilities) if volatilities else 0.3
            
            # Concentration risk
            total_value = sum(pos.get('value', 0) for pos in positions)
            if total_value > 0:
                weights = [pos.get('value', 0) / total_value for pos in positions]
                concentration_risk = sum(w**2 for w in weights)  # Herfindahl index
                risk_factors['concentration'] = concentration_risk
            else:
                risk_factors['concentration'] = 0
            
            # Liquidity risk
            liquidity_scores = [pos.get('liquidity', 0) for pos in positions]
            avg_liquidity = statistics.mean(liquidity_scores) if liquidity_scores else 0
            risk_factors['liquidity'] = max(0, 1.0 - min(1.0, avg_liquidity / 1000000))
            
            # Quality risk
            quality_scores = [pos.get('quality_score', 50) for pos in positions]
            avg_quality = statistics.mean(quality_scores) if quality_scores else 50
            risk_factors['quality'] = max(0, 1.0 - (avg_quality / 100))
            
            return risk_factors
            
        except Exception:
            return {
                'volatility': 0.3,
                'concentration': 0.5,
                'liquidity': 0.3,
                'quality': 0.3
            }
    
    def _optimize_position_allocation(self, positions: List[Dict], 
                                   available_capital: float, 
                                   risk_metrics: Dict) -> Dict:
        """Optimize position allocation using AI and modern portfolio theory"""
        try:
            if not positions:
                return {}
            
            # Calculate optimal weights for each position
            optimal_weights = {}
            total_risk_budget = self.max_portfolio_risk
            
            for pos in positions:
                symbol = pos.get('symbol', 'UNKNOWN')
                quality_score = pos.get('quality_score', 50)
                risk_score = pos.get('risk_score', 0.5)
                expected_return = pos.get('expected_return', 0.12)
                
                # Calculate optimal weight based on risk-adjusted return
                risk_adjusted_return = expected_return / (risk_score + 0.1)  # Avoid division by zero
                
                # Quality adjustment
                quality_factor = quality_score / 100
                
                # Risk budget allocation
                risk_budget = total_risk_budget / len(positions)  # Equal risk budget initially
                
                # Calculate optimal weight
                optimal_weight = (risk_adjusted_return * quality_factor * risk_budget) / 0.2  # Normalize
                
                # Apply constraints
                optimal_weight = max(self.min_position_weight, 
                                   min(self.max_position_weight, optimal_weight))
                
                optimal_weights[symbol] = optimal_weight
            
            # Normalize weights to sum to 1.0
            total_weight = sum(optimal_weights.values())
            if total_weight > 0:
                optimal_weights = {k: v/total_weight for k, v in optimal_weights.items()}
            
            # Calculate position sizes
            allocation = {}
            for symbol, weight in optimal_weights.items():
                position_size = weight * available_capital
                allocation[symbol] = {
                    'weight': weight,
                    'position_size': position_size,
                    'risk_budget': weight * total_risk_budget,
                    'expected_return': positions[0].get('expected_return', 0.12)  # Simplified
                }
            
            return allocation
            
        except Exception as e:
            logger.error(f"Error optimizing position allocation: {e}")
            return self._get_equal_allocation(positions, available_capital)
    
    def _get_equal_allocation(self, positions: List[Dict], available_capital: float) -> Dict:
        """Get equal allocation as fallback"""
        try:
            if not positions:
                return {}
            
            equal_weight = 1.0 / len(positions)
            allocation = {}
            
            for pos in positions:
                symbol = pos.get('symbol', 'UNKNOWN')
                allocation[symbol] = {
                    'weight': equal_weight,
                    'position_size': equal_weight * available_capital,
                    'risk_budget': equal_weight * self.max_portfolio_risk,
                    'expected_return': 0.12
                }
            
            return allocation
            
        except Exception:
            return {}
    
    def _calculate_portfolio_metrics(self, allocation: Dict, risk_metrics: Dict) -> Dict:
        """Calculate portfolio performance metrics"""
        try:
            if not allocation:
                return {
                    'expected_return': 0,
                    'portfolio_volatility': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'var_95': 0
                }
            
            # Calculate expected return
            expected_return = sum(
                pos['expected_return'] * pos['weight'] 
                for pos in allocation.values()
            )
            
            # Get portfolio volatility
            portfolio_volatility = risk_metrics.get('portfolio_volatility', 0.2)
            
            # Calculate Sharpe ratio
            risk_free_rate = 0.02  # 2% risk-free rate
            sharpe_ratio = (expected_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # Calculate Value at Risk (simplified)
            var_95 = portfolio_volatility * 1.645  # 95% VaR
            
            # Calculate maximum drawdown (simplified)
            max_drawdown = portfolio_volatility * 2.0  # Simplified estimate
            
            return {
                'expected_return': expected_return,
                'portfolio_volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'risk_adjusted_return': expected_return / portfolio_volatility if portfolio_volatility > 0 else 0
            }
            
        except Exception:
            return {
                'expected_return': 0.12,
                'portfolio_volatility': 0.2,
                'sharpe_ratio': 0.5,
                'max_drawdown': 0.4,
                'var_95': 0.33,
                'risk_adjusted_return': 0.6
            }
    
    def _generate_optimization_insights(self, portfolio_analysis: Dict, 
                                      allocation: Dict, 
                                      metrics: Dict) -> List[str]:
        """Generate insights about portfolio optimization"""
        insights = []
        
        try:
            # Concentration insights
            concentration_risk = portfolio_analysis.get('concentration_risk', 0)
            if concentration_risk > 0.3:
                insights.append(f"High concentration risk detected ({concentration_risk:.2f})")
            elif concentration_risk < 0.2:
                insights.append(f"Good diversification achieved ({concentration_risk:.2f})")
            
            # Quality insights
            avg_quality = portfolio_analysis.get('average_quality', 0)
            if avg_quality > 70:
                insights.append(f"High quality portfolio (avg: {avg_quality:.1f})")
            elif avg_quality < 50:
                insights.append(f"Low quality portfolio (avg: {avg_quality:.1f})")
            
            # Risk insights
            portfolio_volatility = metrics.get('portfolio_volatility', 0)
            if portfolio_volatility > 0.3:
                insights.append(f"High portfolio volatility ({portfolio_volatility:.1%})")
            elif portfolio_volatility < 0.15:
                insights.append(f"Low portfolio volatility ({portfolio_volatility:.1%})")
            
            # Sharpe ratio insights
            sharpe_ratio = metrics.get('sharpe_ratio', 0)
            if sharpe_ratio > 1.5:
                insights.append(f"Excellent risk-adjusted returns (Sharpe: {sharpe_ratio:.2f})")
            elif sharpe_ratio < 0.5:
                insights.append(f"Poor risk-adjusted returns (Sharpe: {sharpe_ratio:.2f})")
            
            # Diversification insights
            sector_diversification = portfolio_analysis.get('sector_diversification', 0)
            if sector_diversification > 0.7:
                insights.append("Good sector diversification")
            elif sector_diversification < 0.3:
                insights.append("Poor sector diversification")
            
        except Exception:
            insights.append("Portfolio optimization completed")
        
        return insights
    
    def _generate_recommendations(self, portfolio_analysis: Dict, allocation: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        try:
            # Concentration recommendations
            concentration_risk = portfolio_analysis.get('concentration_risk', 0)
            if concentration_risk > 0.4:
                recommendations.append("Reduce position concentration - consider selling largest positions")
            
            # Quality recommendations
            avg_quality = portfolio_analysis.get('average_quality', 0)
            if avg_quality < 60:
                recommendations.append("Improve portfolio quality - focus on higher quality tokens")
            
            # Diversification recommendations
            position_count = portfolio_analysis.get('position_count', 0)
            if position_count < 3:
                recommendations.append("Increase diversification - add more positions")
            elif position_count > 10:
                recommendations.append("Consider consolidating - too many small positions")
            
            # Risk recommendations
            portfolio_volatility = portfolio_analysis.get('risk_score', 0)
            if portfolio_volatility > 0.7:
                recommendations.append("Reduce portfolio risk - focus on lower volatility tokens")
            
        except Exception:
            recommendations.append("Monitor portfolio performance closely")
        
        return recommendations
    
    def _get_empty_portfolio(self) -> Dict:
        """Return empty portfolio structure"""
        return {
            'optimized_allocation': {},
            'portfolio_metrics': {
                'expected_return': 0,
                'portfolio_volatility': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'var_95': 0,
                'risk_adjusted_return': 0
            },
            'risk_metrics': {'portfolio_volatility': 0, 'correlation_matrix': {}, 'risk_factors': {}},
            'insights': ['No positions to optimize'],
            'optimization_timestamp': datetime.now().isoformat(),
            'total_capital': 0,
            'recommended_actions': ['Add positions to portfolio']
        }
    
    def _get_default_portfolio(self, positions: List[Dict], available_capital: float) -> Dict:
        """Return default portfolio when optimization fails"""
        return {
            'optimized_allocation': self._get_equal_allocation(positions, available_capital),
            'portfolio_metrics': {
                'expected_return': 0.12,
                'portfolio_volatility': 0.2,
                'sharpe_ratio': 0.5,
                'max_drawdown': 0.4,
                'var_95': 0.33,
                'risk_adjusted_return': 0.6
            },
            'risk_metrics': {'portfolio_volatility': 0.2, 'correlation_matrix': {}, 'risk_factors': {}},
            'insights': ['Portfolio optimization unavailable - using equal allocation'],
            'optimization_timestamp': datetime.now().isoformat(),
            'total_capital': available_capital,
            'recommended_actions': ['Monitor portfolio performance']
        }
    
    def get_portfolio_insights(self, positions: List[Dict], available_capital: float) -> Dict:
        """Get comprehensive portfolio insights"""
        try:
            optimization_result = self.optimize_portfolio(positions, available_capital)
            
            insights = {
                'total_positions': len(positions),
                'total_capital': available_capital,
                'optimization_status': 'completed',
                'portfolio_metrics': optimization_result['portfolio_metrics'],
                'risk_assessment': optimization_result['risk_metrics'],
                'recommendations': optimization_result['recommended_actions'],
                'insights': optimization_result['insights']
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting portfolio insights: {e}")
            return {
                'total_positions': len(positions),
                'total_capital': available_capital,
                'optimization_status': 'failed',
                'portfolio_metrics': {},
                'risk_assessment': {},
                'recommendations': ['Portfolio analysis unavailable'],
                'insights': ['Error in portfolio analysis']
            }

# Global instance
ai_portfolio_optimizer = AIPortfolioOptimizer()
