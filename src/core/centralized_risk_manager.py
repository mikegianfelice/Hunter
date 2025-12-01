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

from src.config.config_validator import get_validated_config
from src.config.config_loader import get_config_bool, get_config_float
from src.monitoring.performance_monitor import performance_monitor
from src.storage.positions import load_positions as load_positions_store, replace_positions
from src.storage.risk import load_risk_state as load_risk_state_store, save_risk_state as save_risk_state_store

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
        
        # Initialize open positions tracking
        self.positions_file = "data/open_positions.json"
        self.open_positions = self._load_open_positions()
        
        # Load existing risk state
        self._load_risk_state()
    
    def _load_open_positions(self) -> Dict[str, Any]:
        """Load open positions from persistent storage"""
        try:
            return load_positions_store()
        except Exception as e:
            logger.warning(f"Could not load open positions: {e}")
            return {}
    
    def _load_risk_state(self):
        """Load risk state from persistent storage"""
        try:
            stored = load_risk_state_store() or {}
            today = datetime.now().date().isoformat()
            if stored.get('last_reset_date') == today:
                self.risk_state.update(stored)
        except Exception as e:
            logger.warning(f"Could not load risk state: {e}")

    def _save_risk_state(self):
        """Persist risk state via storage layer"""
        try:
            save_risk_state_store(self.risk_state)
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
        
        # Check holder concentration (optional hard block)
        fail_closed = get_config_bool("holder_concentration_fail_closed", True)
        block_on_high_risk = get_config_bool("holder_concentration_block_on_high_risk", True)
        if block_on_high_risk:
            try:
                from src.utils.holder_concentration_checker import check_holder_concentration
                
                token_address = token.get("address", "")
                chain_id = token.get("chainId", "solana")
                
                if token_address and get_config_bool("enable_holder_concentration_check", True):
                    holder_check = check_holder_concentration(token_address, chain_id)
                    
                    if holder_check and holder_check.get("error"):
                        error_type = holder_check.get("error_type", "unknown")
                        error_msg = holder_check['error']
                        
                        # Don't block trades for rate limit errors - these are RPC issues, not concentration issues
                        is_rate_limit = error_type == "rate_limit"
                        
                        if fail_closed and not is_rate_limit:
                            logger.warning(f"Holder concentration check error (fail-closed) for {token_address} on {chain_id}: {error_msg}")
                            from src.monitoring.structured_logger import log_error
                            log_error("risk.holder_concentration_blocked",
                                     f"Trade blocked: holder concentration check failed for {token.get('symbol', 'UNKNOWN')}: {error_msg}",
                                     symbol=token.get('symbol', 'UNKNOWN'),
                                     token_address=token_address,
                                     chain_id=chain_id,
                                     error=error_msg,
                                     error_type=error_type)
                            return RiskAssessment(
                                overall_risk_score=0.95,
                                risk_level=RiskLevel.CRITICAL,
                                approved=False,
                                reason=f"Holder concentration check failed: {error_msg}",
                                position_adjustment=0.0,
                                risk_factors={
                                    'holder_concentration_error': error_msg,
                                    'holder_concentration_blocked': True
                                },
                                recommendations=["Retry later; holder concentration data unavailable"],
                                timestamp=datetime.now().isoformat()
                            )
                        elif is_rate_limit:
                            # Rate limit error - log warning but don't block trade
                            logger.warning(f"Holder concentration check rate limited for {token_address} on {chain_id}: {error_msg}. Allowing trade to proceed.")
                            from src.monitoring.structured_logger import log_warning
                            log_warning("risk.holder_concentration_rate_limited",
                                       f"Holder concentration check rate limited for {token.get('symbol', 'UNKNOWN')}: {error_msg}. Trade allowed to proceed.",
                                       symbol=token.get('symbol', 'UNKNOWN'),
                                       token_address=token_address,
                                       chain_id=chain_id,
                                       error=error_msg)
                            # Continue with risk assessment without blocking
                    if holder_check and not holder_check.get("error"):
                        threshold = get_config_float("holder_concentration_threshold", 60.0)
                        percentage = holder_check.get("top_10_percentage", 0)
                        
                        # Log the check result for debugging
                        logger.info(f"Holder concentration check for {token.get('symbol', 'UNKNOWN')}: {percentage:.2f}% (threshold: {threshold:.2f}%)")
                        
                        if percentage >= threshold:
                            from src.monitoring.structured_logger import log_error
                            log_error("risk.holder_concentration_blocked",
                                     f"Trade blocked: high holder concentration for {token.get('symbol', 'UNKNOWN')}: {percentage:.1f}% (threshold: {threshold:.1f}%)",
                                     symbol=token.get('symbol', 'UNKNOWN'),
                                     token_address=token_address,
                                     chain_id=chain_id,
                                     percentage=percentage,
                                     threshold=threshold)
                            return RiskAssessment(
                                overall_risk_score=0.9,
                                risk_level=RiskLevel.CRITICAL,
                                approved=False,
                                reason=f"High holder concentration: top 10 holders own {percentage:.1f}% (threshold: {threshold:.1f}%)",
                                position_adjustment=0.0,
                                risk_factors={
                                    'holder_concentration_pct': percentage,
                                    'holder_concentration_risk': 1.0,
                                    'holder_concentration_blocked': True
                                },
                                recommendations=["Token has high holder concentration - potential rug pull risk"],
                                timestamp=datetime.now().isoformat()
                            )
            except Exception as e:
                # If fail_closed is True, block trades when check fails due to exception
                token_address = token.get("address", "")
                chain_id = token.get("chainId", "solana")
                if fail_closed:
                    logger.error(f"Holder concentration check exception (fail-closed) for {token_address} on {chain_id}: {e}")
                    from src.monitoring.structured_logger import log_error
                    log_error("risk.holder_concentration_blocked",
                             f"Trade blocked: holder concentration check exception for {token.get('symbol', 'UNKNOWN')}: {str(e)}",
                             symbol=token.get('symbol', 'UNKNOWN'),
                             token_address=token_address,
                             chain_id=chain_id,
                             exception=str(e))
                    return RiskAssessment(
                        overall_risk_score=0.95,
                        risk_level=RiskLevel.CRITICAL,
                        approved=False,
                        reason=f"Holder concentration check failed with exception: {str(e)}",
                        position_adjustment=0.0,
                        risk_factors={
                            'holder_concentration_error': str(e),
                            'holder_concentration_blocked': True
                        },
                        recommendations=["Holder concentration check unavailable - trade blocked for safety"],
                        timestamp=datetime.now().isoformat()
                    )
                else:
                    logger.warning(f"Holder concentration check failed: {e}, continuing with risk assessment (fail-open mode)")
        
        # Token-specific risk assessment
        token_risk = await self._assess_token_risk(token)
        
        # Calculate overall risk score
        risk_weight = get_config_float("holder_concentration_risk_weight", 0.15)
        base_weights_sum = 0.3 + 0.3 + 0.2 + 0.2
        # Adjust weights to accommodate holder concentration
        adjusted_weight = 0.2 - risk_weight  # Reduce token_risk weight
        overall_risk = (
            portfolio_risk.overall_portfolio_risk * 0.3 +
            position_risk * 0.3 +
            market_risk.overall_market_risk * 0.2 +
            token_risk * adjusted_weight
        )
        
        # Add holder concentration risk if available
        holder_concentration_risk = 0.0
        holder_concentration_pct = 0.0
        try:
            from src.utils.holder_concentration_checker import check_holder_concentration
            
            token_address = token.get("address", "")
            chain_id = token.get("chainId", "solana")
            
            if token_address and get_config_bool("enable_holder_concentration_check", True):
                holder_check = check_holder_concentration(token_address, chain_id)
                
                if holder_check and holder_check.get("error"):
                    error_type = holder_check.get("error_type", "unknown")
                    error_msg = holder_check['error']
                    is_rate_limit = error_type == "rate_limit"
                    
                    # Don't add risk for rate limit errors - these are RPC issues, not concentration issues
                    if fail_closed and not is_rate_limit:
                        holder_concentration_risk = 1.0
                        holder_concentration_pct = 0.0
                        logger.warning(f"Holder concentration risk weighting hit error (fail-closed) for {token_address} on {chain_id}: {error_msg}")
                        overall_risk += holder_concentration_risk * risk_weight
                    elif is_rate_limit:
                        # Rate limit error - don't add risk, just log
                        logger.info(f"Holder concentration check rate limited for {token_address} on {chain_id}: {error_msg}. Not adding risk penalty.")
                        # Continue without adding risk
                if holder_check and not holder_check.get("error"):
                    threshold = get_config_float("holder_concentration_threshold", 60.0)
                    percentage = holder_check.get("top_10_percentage", 0)
                    holder_concentration_pct = percentage
                    
                    # Calculate concentration risk (0.0 to 1.0)
                    if percentage >= threshold:
                        holder_concentration_risk = 1.0
                    elif percentage >= threshold * 0.8:
                        holder_concentration_risk = 0.8
                    elif percentage >= threshold * 0.6:
                        holder_concentration_risk = 0.5
                    elif percentage >= threshold * 0.4:
                        holder_concentration_risk = 0.3
                    else:
                        holder_concentration_risk = 0.1
                    
                    # Add to overall risk
                    overall_risk += holder_concentration_risk * risk_weight
        except Exception as e:
            logger.warning(f"Holder concentration check failed in risk calculation: {e}")
        
        overall_risk = min(1.0, overall_risk)
        
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
            portfolio_risk, position_risk, market_risk, token_risk, holder_concentration_pct
        )
        
        # Build risk factors dict
        risk_factors = {
            'portfolio_risk': portfolio_risk.overall_portfolio_risk,
            'position_risk': position_risk,
            'market_risk': market_risk.overall_market_risk,
            'token_risk': token_risk
        }
        
        # Add holder concentration if checked
        if holder_concentration_pct > 0:
            risk_factors['holder_concentration_pct'] = holder_concentration_pct
            risk_factors['holder_concentration_risk'] = holder_concentration_risk
        
        return RiskAssessment(
            overall_risk_score=overall_risk,
            risk_level=risk_level,
            approved=approved,
            reason=reason,
            position_adjustment=position_adjustment,
            risk_factors=risk_factors,
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
            
            # Reload open positions to get latest data
            self.open_positions = self._load_open_positions()
            positions = self.open_positions or {}
            
            # Calculate portfolio metrics
            max_drawdown = abs(current_session.get('max_drawdown', 0.0))
            
            # Calculate actual total exposure from open positions
            total_exposure = 0.0
            for pos in positions.values():
                amount = float(
                    pos.get('position_size_usd', 0) or
                    pos.get('amount_usd', 0) or
                    pos.get('position_size', 0) or
                    0.0
                )
                total_exposure += amount
            
            # Get max exposure from config or calculate from portfolio
            try:
                # Try to get from config
                max_exposure = getattr(self.config.risk, 'max_total_exposure_usd', None) if hasattr(self.config, 'risk') else None
                if max_exposure is None:
                    # Fallback: calculate from actual portfolio value if available
                    portfolio_value = float(current_session.get('total_profit_loss', 0.0)) + total_exposure
                    max_exposure = max(1000.0, portfolio_value * 2.0) if portfolio_value > 0 else 1000.0
                else:
                    max_exposure = float(max_exposure)
            except (AttributeError, ValueError, TypeError):
                # Final fallback: use reasonable default based on actual exposure
                max_exposure = max(1000.0, total_exposure * 2.0) if total_exposure > 0 else 1000.0
            
            # Calculate concentration risk based on actual exposure vs max exposure
            concentration_risk = min(1.0, total_exposure / max_exposure) if max_exposure > 0 else 0.0
            
            # Calculate correlation risk based on chain diversity
            correlation_risk = 0.2  # Base correlation risk
            if len(positions) > 1:
                # Increase risk if positions are concentrated on same chain
                chains = [p.get('chain', '') or p.get('chainId', '') for p in positions.values()]
                unique_chains = len(set(chains))
                if unique_chains <= 1:
                    correlation_risk += 0.2
            
            # Calculate liquidity risk from average position liquidity
            liquidity_risk = 0.4  # High risk baseline
            liquidity_values = []
            for pos in positions.values():
                liq = float(pos.get('liquidity', 0) or 0.0)
                if liq > 0:
                    liquidity_values.append(liq)
            
            if liquidity_values:
                avg_liquidity = sum(liquidity_values) / len(liquidity_values)
                if avg_liquidity > 1000000:
                    liquidity_risk = 0.1  # Low risk
                elif avg_liquidity > 500000:
                    liquidity_risk = 0.2  # Medium risk
                elif avg_liquidity > 100000:
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
            # Calculate liquidity conditions from market data
            token_liquidity = float(market_data.get('liquidity', 0) or 0.0)
            token_volume = float(market_data.get('volume24h', 0) or 0.0)
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
            # Reload positions to get latest data
            self.open_positions = self._load_open_positions()
            if len(self.open_positions) > 2:
                # Check if positions are diversified
                chains = set(p.get('chain', '') or p.get('chainId', '') for p in self.open_positions.values())
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
                                     token_risk: float, holder_concentration_pct: float = 0.0) -> List[str]:
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
        
        # Holder concentration recommendations
        if holder_concentration_pct > 0:
            threshold = get_config_float("holder_concentration_threshold", 60.0)
            if holder_concentration_pct >= threshold:
                recommendations.append(f"⚠️ High holder concentration: {holder_concentration_pct:.1f}% - potential rug pull risk")
            elif holder_concentration_pct >= threshold * 0.8:
                recommendations.append(f"⚠️ Moderate holder concentration: {holder_concentration_pct:.1f}% - monitor closely")
            elif holder_concentration_pct >= threshold * 0.6:
                recommendations.append(f"ℹ️ Some holder concentration: {holder_concentration_pct:.1f}% - acceptable but watch for changes")
        
        if not recommendations:
            recommendations.append("Risk levels acceptable - proceed with normal trading")
        
        return recommendations
    
    def update_trade_result(self, success: bool, profit_loss: float, error_type: Optional[str] = None, token_address: Optional[str] = None):
        """
        Update risk state with trade result
        
        Args:
            success: Whether the trade was successful
            profit_loss: Profit/loss from the trade
            error_type: Type of error if failed. Categories:
                - "gate": Protective gates (time window, risk checks) - don't count as failures
                - "token": Token-specific issues (should blacklist token, not count toward circuit breaker)
                - "systemic": Systemic issues (network, wallet, RPC) - count toward circuit breaker
                - None: Unknown/legacy - count as systemic for safety
            token_address: Token address for blacklisting token-specific failures
        """
        self._reset_daily_metrics()
        
        # Gate failures (time window, risk checks) don't count as trade attempts
        if not success and error_type == "gate":
            logger.info(f"Gate failure ignored (not counting as trade failure): {error_type}")
            return  # Don't update any metrics for gate failures
        
        # Only count actual trade attempts
        self.risk_state['trades_today'] += 1
        
        if success:
            self.risk_state['successful_trades_today'] += 1
            self.risk_state['losing_streak'] = 0
        else:
            # Token-specific failures: blacklist the token but don't count toward circuit breaker
            if error_type == "token" and token_address:
                try:
                    from src.storage.blacklist import load_blacklist, save_blacklist
                    blacklist = load_blacklist()
                    blacklist.add(token_address.lower())
                    save_blacklist(blacklist)
                    logger.warning(f"Token-specific failure: blacklisted {token_address[:8]}... (not counting toward circuit breaker)")
                except Exception as e:
                    logger.error(f"Failed to blacklist token {token_address}: {e}")
                # Don't increment losing streak for token-specific failures
                return
            
            # Systemic failures: count toward circuit breaker
            self.risk_state['losing_streak'] += 1
        
        if profit_loss < 0:
            self.risk_state['daily_loss'] += abs(profit_loss)
            self.risk_state['max_drawdown_today'] = min(
                self.risk_state['max_drawdown_today'], 
                -self.risk_state['daily_loss']
            )
        
        # Check circuit breaker conditions (only for systemic failures)
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

def update_trade_result(success: bool, profit_loss: float, error_type: Optional[str] = None, token_address: Optional[str] = None):
    """Convenience function to update trade result"""
    centralized_risk_manager.update_trade_result(success, profit_loss, error_type, token_address)

def is_circuit_breaker_active() -> bool:
    """Convenience function to check circuit breaker status"""
    return centralized_risk_manager.is_circuit_breaker_active()

def get_risk_summary() -> Dict[str, Any]:
    """Convenience function to get risk summary"""
    return centralized_risk_manager.get_risk_summary()
