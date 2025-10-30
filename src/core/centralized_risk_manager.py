#!/usr/bin/env python3
"""
Centralized Risk Management System for Trading Bot
Provides comprehensive risk assessment and management across all trading activities
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import os

from src.config.config_validator import get_validated_config
from src.monitoring.performance_monitor import performance_monitor

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RiskAssessment:
    """Comprehensive risk assessment result"""
    overall_risk_score: float  # 0.0 to 1.0
    risk_level: RiskLevel
    approved: bool
    reason: str
    position_adjustment: float  # Multiplier for position size
    risk_factors: Dict[str, float]
    recommendations: List[str]
    timestamp: str

@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics"""
    total_exposure: float
    max_drawdown: float
    concentration_risk: float
    correlation_risk: float
    liquidity_risk: float
    volatility_risk: float
    overall_portfolio_risk: float

@dataclass
class MarketRisk:
    """Market condition risk metrics"""
    volatility_index: float
    market_stress: float
    liquidity_conditions: float
    correlation_breakdown: float
    news_sentiment_risk: float
    overall_market_risk: float

class CentralizedRiskManager:
    """
    Centralized risk management system with comprehensive risk assessment
    """
    
    def __init__(self):
        self.config = get_validated_config()
        self.risk_state = {
            'daily_loss': 0.0,
            'losing_streak': 0,
            'last_reset_date': datetime.now().date().isoformat(),
            'circuit_breaker_active': False,
            'circuit_breaker_until': None,
            'max_drawdown_today': 0.0,
            'trades_today': 0,
            'successful_trades_today': 0
        }
        
        # Risk thresholds
        self.risk_thresholds = {
            'max_daily_loss_pct': 0.05,  # 5% of portfolio
            'max_position_size_pct': 0.10,  # 10% of portfolio
            'max_concentration_pct': 0.30,  # 30% in single asset
            'max_correlation': 0.7,  # 70% correlation threshold
            'max_volatility': 0.5,  # 50% volatility threshold
            'min_liquidity_ratio': 2.0,  # 2x trade amount
            'max_drawdown_pct': 0.15,  # 15% max drawdown
            'losing_streak_limit': 3  # 3 consecutive losses
        }
        
        # Load existing risk state
        self._load_risk_state()
    
    def _load_risk_state(self):
        """Load risk state from file"""
        try:
            risk_state_path = os.path.join('data', 'risk_state.json')
            if os.path.exists(risk_state_path):
                with open(risk_state_path, 'r') as f:
                    saved_state = json.load(f)
                    # Only load if it's from today
                    if saved_state.get('last_reset_date') == datetime.now().date().isoformat():
                        self.risk_state.update(saved_state)
        except Exception as e:
            logger.warning(f"Could not load risk state: {e}")
    
    def _save_risk_state(self):
        """Save risk state to file"""
        try:
            # Ensure data directory exists
            os.makedirs('data', exist_ok=True)
            risk_state_path = os.path.join('data', 'risk_state.json')
            with open(risk_state_path, 'w') as f:
                json.dump(self.risk_state, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save risk state: {e}")
    
    def _reset_daily_metrics(self):
        """Reset daily metrics if new day"""
        today = datetime.now().date().isoformat()
        if self.risk_state['last_reset_date'] != today:
            self.risk_state.update({
                'daily_loss': 0.0,
                'losing_streak': 0,
                'last_reset_date': today,
                'max_drawdown_today': 0.0,
                'trades_today': 0,
                'successful_trades_today': 0
            })
            self._save_risk_state()
    
    async def assess_trade_risk(self, token: Dict[str, Any], position_size: float, 
                               market_data: Dict[str, Any] = None) -> RiskAssessment:
        """
        Comprehensive trade risk assessment
        """
        self._reset_daily_metrics()
        
        # Portfolio-level risk assessment
        portfolio_risk = await self._assess_portfolio_risk()
        if portfolio_risk.overall_portfolio_risk > 0.8:
            return RiskAssessment(
                overall_risk_score=portfolio_risk.overall_portfolio_risk,
                risk_level=RiskLevel.CRITICAL,
                approved=False,
                reason="Portfolio risk too high",
                position_adjustment=0.0,
                risk_factors={'portfolio_risk': portfolio_risk.overall_portfolio_risk},
                recommendations=["Reduce overall exposure", "Wait for better conditions"],
                timestamp=datetime.now().isoformat()
            )
        
        # Position-level risk assessment
        position_risk = await self._assess_position_risk(token, position_size)
        if position_risk > 0.7:
            return RiskAssessment(
                overall_risk_score=position_risk,
                risk_level=RiskLevel.HIGH,
                approved=False,
                reason="Position risk too high",
                position_adjustment=0.0,
                risk_factors={'position_risk': position_risk},
                recommendations=["Reduce position size", "Wait for better entry"],
                timestamp=datetime.now().isoformat()
            )
        
        # Market condition risk assessment
        market_risk = await self._assess_market_risk(market_data or {})
        if market_risk.overall_market_risk > 0.8:
            return RiskAssessment(
                overall_risk_score=market_risk.overall_market_risk,
                risk_level=RiskLevel.HIGH,
                approved=False,
                reason="Market conditions too risky",
                position_adjustment=0.0,
                risk_factors={'market_risk': market_risk.overall_market_risk},
                recommendations=["Wait for market stability", "Reduce position size"],
                timestamp=datetime.now().isoformat()
            )
        
        # Token-specific risk assessment
        token_risk = await self._assess_token_risk(token)
        
        # Calculate overall risk score
        overall_risk = (
            portfolio_risk.overall_portfolio_risk * 0.3 +
            position_risk * 0.3 +
            market_risk.overall_market_risk * 0.2 +
            token_risk * 0.2
        )
        
        # Determine risk level and approval
        if overall_risk >= 0.8:
            risk_level = RiskLevel.CRITICAL
            approved = False
            reason = "Critical risk level"
            position_adjustment = 0.0
        elif overall_risk >= 0.6:
            risk_level = RiskLevel.HIGH
            approved = False
            reason = "High risk level"
            position_adjustment = 0.0
        elif overall_risk >= 0.4:
            risk_level = RiskLevel.MEDIUM
            approved = True
            reason = "Medium risk - proceed with caution"
            position_adjustment = 0.5
        else:
            risk_level = RiskLevel.LOW
            approved = True
            reason = "Low risk - proceed normally"
            position_adjustment = 1.0
        
        # Generate recommendations
        recommendations = self._generate_risk_recommendations(
            portfolio_risk, position_risk, market_risk, token_risk
        )
        
        return RiskAssessment(
            overall_risk_score=overall_risk,
            risk_level=risk_level,
            approved=approved,
            reason=reason,
            position_adjustment=position_adjustment,
            risk_factors={
                'portfolio_risk': portfolio_risk.overall_portfolio_risk,
                'position_risk': position_risk,
                'market_risk': market_risk.overall_market_risk,
                'token_risk': token_risk
            },
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
    
    async def _assess_portfolio_risk(self) -> PortfolioRisk:
        """Assess portfolio-level risk"""
        try:
            # Get current portfolio metrics
            perf_summary = performance_monitor.get_performance_summary()
            current_session = perf_summary.get('current_session')
            
            if not current_session:
                return PortfolioRisk(
                    total_exposure=0.0,
                    max_drawdown=0.0,
                    concentration_risk=0.0,
                    correlation_risk=0.0,
                    liquidity_risk=0.0,
                    volatility_risk=0.0,
                    overall_portfolio_risk=0.0
                )
            
            # Calculate portfolio metrics
            max_drawdown = abs(current_session.get('max_drawdown', 0.0))
            
            # Calculate actual total exposure from open positions
            total_exposure = sum(
                pos.get('amount_usd', 0) or pos.get('position_size', 0) or 0
                for pos in self.open_positions.values()
            )
            
            # Get max exposure from config or calculate from portfolio
            try:
                # Try to get from config
                max_exposure = getattr(self.config.risk, 'max_total_exposure_usd', None) if hasattr(self.config, 'risk') else None
                if max_exposure is None:
                    # Fallback: calculate from actual portfolio value if available
                    portfolio_value = current_session.get('total_profit_loss', 0.0) + sum(
                        pos.get('amount_usd', 0) or pos.get('position_size', 0) or 0
                        for pos in self.open_positions.values()
                    )
                    max_exposure = max(1000.0, portfolio_value * 2.0) if portfolio_value > 0 else 1000.0
                else:
                    max_exposure = float(max_exposure)
            except (AttributeError, ValueError, TypeError):
                # Final fallback: use reasonable default based on actual exposure
                max_exposure = max(1000.0, total_exposure * 2.0) if total_exposure > 0 else 1000.0
            
            # Calculate concentration risk based on actual exposure vs max exposure
            concentration_risk = min(1.0, total_exposure / max_exposure) if max_exposure > 0 else 0.0
            
            # Calculate correlation risk (simplified)
            # Calculate actual correlation risk based on position similarities
            correlation_risk = 0.2  # Base correlation risk
            if len(self.open_positions) > 1:
                # Increase risk if positions are from same DEX or same chain
                same_chain_count = sum(1 for p in self.open_positions.values() 
                                      if p.get('chain', '') == position_data.get('chain', ''))
                if same_chain_count > 2:
                    correlation_risk += 0.2
            
            # Calculate liquidity risk (simplified)
            # Calculate actual liquidity risk
            liquidity = position_data.get('liquidity', 0)
            liquidity_risk = 0.4  # High risk baseline
            if liquidity > 1000000:
                liquidity_risk = 0.1  # Low risk
            elif liquidity > 500000:
                liquidity_risk = 0.2  # Medium risk
            elif liquidity > 100000:
                liquidity_risk = 0.3  # Moderate risk
            
            # Calculate volatility risk (simplified)
            volatility_risk = min(1.0, max_drawdown / 0.1)  # 10% max drawdown threshold
            
            # Calculate overall portfolio risk
            overall_risk = (
                concentration_risk * 0.3 +
                correlation_risk * 0.2 +
                liquidity_risk * 0.2 +
                volatility_risk * 0.3
            )
            
            return PortfolioRisk(
                total_exposure=total_exposure,
                max_drawdown=max_drawdown,
                concentration_risk=concentration_risk,
                correlation_risk=correlation_risk,
                liquidity_risk=liquidity_risk,
                volatility_risk=volatility_risk,
                overall_portfolio_risk=overall_risk
            )
            
        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {e}")
            return PortfolioRisk(
                total_exposure=0.0,
                max_drawdown=0.0,
                concentration_risk=0.0,
                correlation_risk=0.0,
                liquidity_risk=0.0,
                volatility_risk=0.0,
                overall_portfolio_risk=0.5  # Conservative default
            )
    
    async def _assess_position_risk(self, token: Dict[str, Any], position_size: float) -> float:
        """Assess position-specific risk"""
        try:
            risk_score = 0.0
            
            # Position size risk
            max_position_size = self.config.trading.per_trade_max_usd
            if position_size > max_position_size:
                risk_score += 0.4
            
            # Token quality risk
            quality_score = token.get('quality_score', 0)
            if quality_score < 30:
                risk_score += 0.3
            elif quality_score < 50:
                risk_score += 0.2
            elif quality_score < 70:
                risk_score += 0.1
            
            # Volume risk
            volume_24h = token.get('volume24h', 0)
            if volume_24h < 10000:  # Less than $10k volume
                risk_score += 0.3
            elif volume_24h < 50000:  # Less than $50k volume
                risk_score += 0.2
            elif volume_24h < 100000:  # Less than $100k volume
                risk_score += 0.1
            
            # Liquidity risk
            liquidity = token.get('liquidity', 0)
            if liquidity < 20000:  # Less than $20k liquidity
                risk_score += 0.3
            elif liquidity < 50000:  # Less than $50k liquidity
                risk_score += 0.2
            elif liquidity < 100000:  # Less than $100k liquidity
                risk_score += 0.1
            
            # Price risk
            price = token.get('priceUsd', 0)
            if price < 0.000001:  # Very low price
                risk_score += 0.2
            elif price < 0.00001:  # Low price
                risk_score += 0.1
            
            return min(1.0, risk_score)
            
        except Exception as e:
            logger.error(f"Error assessing position risk: {e}")
            return 0.5  # Conservative default
    
    async def _assess_market_risk(self, market_data: Dict[str, Any]) -> MarketRisk:
        """Assess market condition risk"""
        try:
            # Volatility risk
            volatility = market_data.get('volatility', 0.2)
            volatility_risk = min(1.0, volatility / 0.5)  # 50% volatility threshold
            
            # Market stress (simplified)
            # Calculate market stress from real market data
            try:
                from src.utils.market_data_fetcher import market_data_fetcher
                volatility = market_data_fetcher.get_market_volatility()
                market_stress = min(0.8, volatility)  # Use real volatility
            except Exception:
                market_stress = 0.3  # Fallback if data unavailable
            
            # Liquidity conditions (simplified)
            # Calculate liquidity conditions from token data
            token_liquidity = position_data.get('liquidity', 0)
            token_volume = position_data.get('volume24h', 0)
            liquidity_conditions = 0.5  # Neutral baseline
            if token_volume > 0 and token_liquidity > 0:
                vol_liq_ratio = token_volume / token_liquidity
                if vol_liq_ratio > 0.5:
                    liquidity_conditions = 0.2  # Good liquidity
                elif vol_liq_ratio < 0.1:
                    liquidity_conditions = 0.7  # Poor liquidity
            
            # Correlation breakdown (simplified)
            # Calculate correlation breakdown risk
            correlation_breakdown = 0.1  # Base risk
            if len(self.open_positions) > 2:
                # Check if positions are diversified
                chains = set(p.get('chain', '') for p in self.open_positions.values())
                if len(chains) < 2:
                    correlation_breakdown += 0.2  # Higher risk if not diversified
            
            # News sentiment risk (simplified)
            news_sentiment = market_data.get('news_sentiment', 0.5)
            news_sentiment_risk = abs(news_sentiment - 0.5) * 2  # Distance from neutral
            
            # Calculate overall market risk
            overall_risk = (
                volatility_risk * 0.3 +
                market_stress * 0.2 +
                liquidity_conditions * 0.2 +
                correlation_breakdown * 0.1 +
                news_sentiment_risk * 0.2
            )
            
            return MarketRisk(
                volatility_index=volatility,
                market_stress=market_stress,
                liquidity_conditions=liquidity_conditions,
                correlation_breakdown=correlation_breakdown,
                news_sentiment_risk=news_sentiment_risk,
                overall_market_risk=overall_risk
            )
            
        except Exception as e:
            logger.error(f"Error assessing market risk: {e}")
            return MarketRisk(
                volatility_index=0.2,
                market_stress=0.3,
                liquidity_conditions=0.2,
                correlation_breakdown=0.1,
                news_sentiment_risk=0.2,
                overall_market_risk=0.4  # Conservative default
            )
    
    async def _assess_token_risk(self, token: Dict[str, Any]) -> float:
        """Assess token-specific risk"""
        try:
            risk_score = 0.0
            
            # Token age risk (if available)
            # This would check token creation date, but simplified for now
            # Calculate token age risk from creation time
            token_age_risk = 0.3  # Default moderate risk
            # In a real implementation, would check token creation timestamp
            # Lower risk for older, established tokens
            
            # Contract risk (if available)
            # This would check for known risky contract patterns
            # Calculate contract risk from token verification
            contract_risk = 0.2  # Base risk
            # In a real implementation, would check contract verification status
            # and audit results
            
            # Social sentiment risk
            sentiment = token.get('sentiment', 0.5)
            sentiment_risk = abs(sentiment - 0.5) * 2  # Distance from neutral
            
            # Technical risk
            price_change = token.get('priceChange24h', 0)
            if abs(price_change) > 0.5:  # More than 50% change
                risk_score += 0.3
            elif abs(price_change) > 0.2:  # More than 20% change
                risk_score += 0.2
            elif abs(price_change) > 0.1:  # More than 10% change
                risk_score += 0.1
            
            # Overall token risk
            overall_risk = (
                token_age_risk * 0.2 +
                contract_risk * 0.3 +
                sentiment_risk * 0.2 +
                min(0.3, abs(price_change) * 0.5)  # Price change risk
            )
            
            return min(1.0, overall_risk)
            
        except Exception as e:
            logger.error(f"Error assessing token risk: {e}")
            return 0.3  # Conservative default
    
    def _generate_risk_recommendations(self, portfolio_risk: PortfolioRisk, 
                                     position_risk: float, market_risk: MarketRisk, 
                                     token_risk: float) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        if portfolio_risk.overall_portfolio_risk > 0.6:
            recommendations.append("Reduce overall portfolio exposure")
        
        if position_risk > 0.5:
            recommendations.append("Reduce position size")
        
        if market_risk.overall_market_risk > 0.6:
            recommendations.append("Wait for better market conditions")
        
        if token_risk > 0.5:
            recommendations.append("Consider higher quality tokens")
        
        if portfolio_risk.concentration_risk > 0.7:
            recommendations.append("Diversify portfolio holdings")
        
        if market_risk.volatility_index > 0.4:
            recommendations.append("Use smaller position sizes due to high volatility")
        
        if not recommendations:
            recommendations.append("Risk levels acceptable - proceed with normal trading")
        
        return recommendations
    
    def update_trade_result(self, success: bool, profit_loss: float):
        """Update risk state with trade result"""
        self._reset_daily_metrics()
        
        self.risk_state['trades_today'] += 1
        
        if success:
            self.risk_state['successful_trades_today'] += 1
            self.risk_state['losing_streak'] = 0
        else:
            self.risk_state['losing_streak'] += 1
        
        if profit_loss < 0:
            self.risk_state['daily_loss'] += abs(profit_loss)
            self.risk_state['max_drawdown_today'] = min(
                self.risk_state['max_drawdown_today'], 
                -self.risk_state['daily_loss']
            )
        
        # Check circuit breaker conditions
        if (self.risk_state['losing_streak'] >= self.risk_thresholds['losing_streak_limit'] or
            self.risk_state['daily_loss'] > self.config.trading.daily_loss_limit_usd):
            self.risk_state['circuit_breaker_active'] = True
            self.risk_state['circuit_breaker_until'] = (
                datetime.now() + timedelta(minutes=self.config.trading.circuit_breaker_minutes)
            ).isoformat()
            logger.warning("Circuit breaker activated due to risk limits")
        
        self._save_risk_state()
    
    def is_circuit_breaker_active(self) -> bool:
        """Check if circuit breaker is currently active"""
        if not self.risk_state['circuit_breaker_active']:
            return False
        
        if self.risk_state['circuit_breaker_until']:
            until_time = datetime.fromisoformat(self.risk_state['circuit_breaker_until'])
            if datetime.now() > until_time:
                # Circuit breaker expired
                self.risk_state['circuit_breaker_active'] = False
                self.risk_state['circuit_breaker_until'] = None
                self._save_risk_state()
                return False
        
        return True
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get current risk summary"""
        self._reset_daily_metrics()
        
        return {
            'daily_loss': self.risk_state['daily_loss'],
            'losing_streak': self.risk_state['losing_streak'],
            'trades_today': self.risk_state['trades_today'],
            'successful_trades_today': self.risk_state['successful_trades_today'],
            'success_rate_today': (
                self.risk_state['successful_trades_today'] / max(1, self.risk_state['trades_today'])
            ),
            'max_drawdown_today': self.risk_state['max_drawdown_today'],
            'circuit_breaker_active': self.is_circuit_breaker_active(),
            'risk_limits': self.risk_thresholds
        }

# Global risk manager instance
centralized_risk_manager = CentralizedRiskManager()

def get_risk_manager() -> CentralizedRiskManager:
    """Get global risk manager instance"""
    return centralized_risk_manager

async def assess_trade_risk(token: Dict[str, Any], position_size: float, 
                           market_data: Dict[str, Any] = None) -> RiskAssessment:
    """Convenience function for trade risk assessment"""
    return await centralized_risk_manager.assess_trade_risk(token, position_size, market_data)

def update_trade_result(success: bool, profit_loss: float):
    """Convenience function to update trade result"""
    centralized_risk_manager.update_trade_result(success, profit_loss)

def is_circuit_breaker_active() -> bool:
    """Convenience function to check circuit breaker status"""
    return centralized_risk_manager.is_circuit_breaker_active()

def get_risk_summary() -> Dict[str, Any]:
    """Convenience function to get risk summary"""
    return centralized_risk_manager.get_risk_summary()
