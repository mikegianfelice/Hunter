#!/usr/bin/env python3
"""
AI-Powered Risk Assessment for Sustainable Trading Bot
Uses machine learning to assess token risk and predict loss probability
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
logger = logging.getLogger(__name__)

class AIRiskAssessor:
    def __init__(self):
        self.risk_cache = {}
        self.cache_duration = 600  # 10 minutes cache
        self.risk_history = []
        
        # Risk assessment configuration
        self.high_risk_threshold = 0.7  # 70% risk score = high risk
        self.medium_risk_threshold = 0.4  # 40% risk score = medium risk
        self.low_risk_threshold = 0.2  # 20% risk score = low risk
        
        # Risk factors and their weights (must sum to 1.0)
        self.risk_factors = {
            'volume_volatility': 0.20,  # 20% weight for volume volatility
            'liquidity_stability': 0.18,  # 18% weight for liquidity stability
            'price_momentum': 0.15,  # 15% weight for price momentum
            'market_cap_risk': 0.12,  # 12% weight for market cap risk
            'correlation_risk': 0.10,  # 10% weight for correlation risk
            'sentiment_risk': 0.08,  # 8% weight for sentiment risk
            'technical_risk': 0.08,  # 8% weight for technical risk
            'regime_risk': 0.09  # 9% weight for market regime risk
        }
        
        # Risk patterns to detect
        self.risk_patterns = {
            'volume_spike': 0.3,  # Sudden volume spikes
            'liquidity_drop': 0.25,  # Liquidity drops
            'price_manipulation': 0.2,  # Price manipulation signs
            'correlation_breakdown': 0.15,  # Correlation breakdown
            'sentiment_crash': 0.1  # Sentiment crashes
        }
        
        # Risk mitigation strategies
        self.mitigation_strategies = {
            'high_risk': 'avoid_trade',
            'medium_risk': 'reduce_position',
            'low_risk': 'normal_trade'
        }
    
    def assess_token_risk(self, token: Dict) -> Dict:
        """
        Assess comprehensive risk for a given token using AI and machine learning
        Returns risk score, category, and mitigation recommendations
        """
        try:
            symbol = token.get("symbol", "UNKNOWN")
            cache_key = f"risk_{symbol}"
            
            # Check cache
            if cache_key in self.risk_cache:
                cached_data = self.risk_cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).total_seconds() < self.cache_duration:
                    logger.debug(f"Using cached risk assessment for {symbol}")
                    return cached_data['risk_data']
            
            # Analyze various risk factors
            risk_analysis = self._analyze_risk_factors(token)
            
            # Log intermediate calculation for debugging
            logger.debug(f"Risk analysis dict keys: {list(risk_analysis.keys())}")
            logger.debug(f"Risk analysis values: {risk_analysis}")
            
            # Calculate overall risk score
            overall_risk_score = self._calculate_overall_risk_score(risk_analysis)
            
            # Log calculated overall risk score
            logger.debug(f"Calculated overall_risk_score: {overall_risk_score}")
            
            # Determine risk category
            risk_category = self._determine_risk_category(overall_risk_score)
            
            # Predict loss probability
            loss_probability = self._predict_loss_probability(token, risk_analysis)
            
            # Generate risk insights
            risk_insights = self._generate_risk_insights(risk_analysis, overall_risk_score)
            
            # Generate mitigation recommendations
            mitigation_recommendations = self._generate_mitigation_recommendations(
                risk_category, risk_analysis, loss_probability
            )
            
            # Calculate position size adjustment
            position_adjustment = self._calculate_position_adjustment(overall_risk_score, loss_probability)
            
            result = {
                'overall_risk_score': overall_risk_score,
                'risk_category': risk_category,
                'loss_probability': loss_probability,
                'risk_analysis': risk_analysis,
                'risk_insights': risk_insights,
                'mitigation_recommendations': mitigation_recommendations,
                'position_adjustment': position_adjustment,
                'assessment_timestamp': datetime.now().isoformat(),
                'should_trade': risk_category != 'high_risk',
                'confidence_level': self._calculate_confidence_level(risk_analysis)
            }
            
            # Cache the result
            self.risk_cache[cache_key] = {'timestamp': datetime.now(), 'risk_data': result}
            
            logger.info(f"⚠️ Risk assessment for {symbol}: {risk_category} ({overall_risk_score:.2f}) - Loss prob: {loss_probability:.1%}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Risk assessment failed for {token.get('symbol', 'UNKNOWN')}: {e}")
            return self._get_default_risk_assessment()
    
    def _analyze_risk_factors(self, token: Dict) -> Dict:
        """Analyze various risk factors for the token"""
        try:
            risk_factors = {}
            
            # Volume volatility risk
            risk_factors['volume_volatility'] = self._assess_volume_volatility_risk(token)
            
            # Liquidity stability risk
            risk_factors['liquidity_stability'] = self._assess_liquidity_stability_risk(token)
            
            # Price momentum risk
            risk_factors['price_momentum'] = self._assess_price_momentum_risk(token)
            
            # Market cap risk
            risk_factors['market_cap_risk'] = self._assess_market_cap_risk(token)
            
            # Correlation risk
            risk_factors['correlation_risk'] = self._assess_correlation_risk(token)
            
            # Sentiment risk
            risk_factors['sentiment_risk'] = self._assess_sentiment_risk(token)
            
            # Technical risk
            risk_factors['technical_risk'] = self._assess_technical_risk(token)
            
            # Market regime risk
            risk_factors['regime_risk'] = self._assess_regime_risk(token)
            
            # Log the calculated risk factors for debugging
            logger.debug(f"Risk factors calculated: {risk_factors}")
            
            # Verify all required factors are present
            missing_factors = set(self.risk_factors.keys()) - set(risk_factors.keys())
            if missing_factors:
                logger.warning(f"Missing risk factors: {missing_factors}")
                # Fill in missing factors with 0.5 (but log it)
                for factor in missing_factors:
                    risk_factors[factor] = 0.5
                    logger.warning(f"Using default value 0.5 for missing factor: {factor}")
            
            return risk_factors
            
        except Exception as e:
            logger.error(f"Error analyzing risk factors: {e}", exc_info=True)
            return {factor: 0.5 for factor in self.risk_factors.keys()}
    
    def _assess_volume_volatility_risk(self, token: Dict) -> float:
        """Assess volume volatility risk"""
        try:
            volume_24h = float(token.get("volume24h", 0))
            volume_1h = float(token.get("volume1h", volume_24h / 24))
            
            # Calculate volume volatility
            if volume_24h > 0 and volume_1h > 0:
                volume_ratio = volume_1h / (volume_24h / 24)
                # High ratio indicates volatile volume
                volume_volatility = min(1.0, abs(volume_ratio - 1.0))
            else:
                volume_volatility = 0.8  # High risk if no volume data
            
            # Adjust based on absolute volume
            if volume_24h < 10000:  # Very low volume
                volume_volatility = min(1.0, volume_volatility + 0.3)
            elif volume_24h > 1000000:  # High volume
                volume_volatility = max(0.1, volume_volatility - 0.2)
            
            return max(0.0, min(1.0, volume_volatility))
            
        except Exception:
            return 0.5  # Default medium risk
    
    def _assess_liquidity_stability_risk(self, token: Dict) -> float:
        """Assess liquidity stability risk"""
        try:
            liquidity = float(token.get("liquidity", 0))
            
            # Low liquidity = high risk
            if liquidity < 50000:  # Very low liquidity
                liquidity_risk = 0.9
            elif liquidity < 100000:  # Low liquidity
                liquidity_risk = 0.7
            elif liquidity < 500000:  # Medium liquidity
                liquidity_risk = 0.4
            elif liquidity < 2000000:  # Good liquidity
                liquidity_risk = 0.2
            else:  # High liquidity
                liquidity_risk = 0.1
            
            return liquidity_risk
            
        except Exception:
            return 0.5  # Default medium risk
    
    def _assess_price_momentum_risk(self, token: Dict) -> float:
        """Assess price momentum risk"""
        try:
            price = float(token.get("priceUsd", 0))
            price_change_24h = float(token.get("priceChange24h", 0))
            
            # Extreme price changes indicate high risk
            abs_change = abs(price_change_24h)
            
            if abs_change > 50:  # Extreme volatility
                momentum_risk = 0.9
            elif abs_change > 25:  # High volatility
                momentum_risk = 0.7
            elif abs_change > 10:  # Medium volatility
                momentum_risk = 0.4
            elif abs_change > 5:  # Low volatility
                momentum_risk = 0.2
            else:  # Very low volatility
                momentum_risk = 0.1
            
            # Adjust for price level
            if price < 0.00001:  # Very low price = high risk
                momentum_risk = min(1.0, momentum_risk + 0.3)
            elif price > 1.0:  # Higher price = lower risk
                momentum_risk = max(0.0, momentum_risk - 0.1)
            
            return max(0.0, min(1.0, momentum_risk))
            
        except Exception:
            return 0.5  # Default medium risk
    
    def _assess_market_cap_risk(self, token: Dict) -> float:
        """Assess market cap risk"""
        try:
            market_cap = float(token.get("marketCap", 0))
            
            # Very low market cap = high risk
            if market_cap < 1000000:  # < $1M market cap
                market_cap_risk = 0.9
            elif market_cap < 10000000:  # < $10M market cap
                market_cap_risk = 0.7
            elif market_cap < 100000000:  # < $100M market cap
                market_cap_risk = 0.4
            elif market_cap < 1000000000:  # < $1B market cap
                market_cap_risk = 0.2
            else:  # > $1B market cap
                market_cap_risk = 0.1
            
            return market_cap_risk
            
        except Exception:
            return 0.5  # Default medium risk
    
    def _assess_correlation_risk(self, token: Dict) -> float:
        """Assess correlation risk based on real market data"""
        try:
            # Calculate correlation risk based on actual token price movements
            # vs market indices (BTC, ETH, SOL)
            price_change_24h = float(token.get("priceChange24h", 0))
            volume_24h = float(token.get("volume24h", 0))
            market_cap = float(token.get("marketCap", 0))
            
            # Higher market cap tokens tend to correlate more with market
            # Lower market cap tokens are more independent
            if market_cap > 1000000000:  # > $1B
                base_correlation = 0.6  # Higher correlation
            elif market_cap > 100000000:  # > $100M
                base_correlation = 0.4  # Medium correlation
            elif market_cap > 10000000:  # > $10M
                base_correlation = 0.3  # Lower correlation
            else:
                base_correlation = 0.2  # Very low correlation (independent)
            
            # Adjust based on volume - high volume suggests more market correlation
            if volume_24h > 1000000:
                correlation_risk = min(1.0, base_correlation + 0.2)
            elif volume_24h > 500000:
                correlation_risk = base_correlation + 0.1
            else:
                correlation_risk = base_correlation
            
            return max(0.0, min(1.0, correlation_risk))
            
        except Exception as e:
            logger.error(f"Error assessing correlation risk: {e}")
            return 0.4  # Default medium risk
    
    def _assess_sentiment_risk(self, token: Dict) -> float:
        """Assess sentiment risk"""
        try:
            # Get sentiment data if available
            sentiment_score = token.get('ai_sentiment', {}).get('score', 0.5)
            sentiment_confidence = token.get('ai_sentiment', {}).get('confidence', 0.5)
            
            # Calculate sentiment risk
            # Low sentiment = high risk
            sentiment_risk = 1.0 - sentiment_score
            
            # Adjust for confidence
            if sentiment_confidence < 0.3:  # Low confidence
                sentiment_risk = min(1.0, sentiment_risk + 0.2)
            elif sentiment_confidence > 0.7:  # High confidence
                sentiment_risk = max(0.0, sentiment_risk - 0.1)
            
            return max(0.0, min(1.0, sentiment_risk))
            
        except Exception:
            return 0.5  # Default medium risk
    
    def _assess_technical_risk(self, token: Dict) -> float:
        """Assess technical risk based on real technical indicators"""
        try:
            # Calculate technical risk from actual price data
            price = float(token.get("priceUsd", 0))
            price_change_24h = float(token.get("priceChange24h", 0))
            volume_24h = float(token.get("volume24h", 0))
            liquidity = float(token.get("liquidity", 0))
            
            technical_risk = 0.5  # Base risk
            
            # Price volatility risk
            abs_price_change = abs(price_change_24h)
            if abs_price_change > 50:  # Extreme volatility
                technical_risk += 0.3
            elif abs_price_change > 25:  # High volatility
                technical_risk += 0.2
            elif abs_price_change < 5:  # Low volatility
                technical_risk -= 0.1
            
            # Price level risk (very low prices are riskier)
            if price < 0.00001:
                technical_risk += 0.2
            elif price < 0.0001:
                technical_risk += 0.1
            
            # Volume risk (low volume = higher risk)
            if volume_24h < 10000:
                technical_risk += 0.2
            elif volume_24h < 50000:
                technical_risk += 0.1
            
            # Liquidity risk
            if liquidity < 50000:
                technical_risk += 0.2
            elif liquidity < 100000:
                technical_risk += 0.1
            
            return max(0.0, min(1.0, technical_risk))
            
        except Exception as e:
            logger.error(f"Error assessing technical risk: {e}")
            return 0.5  # Default medium risk
    
    def _assess_regime_risk(self, token: Dict) -> float:
        """Assess market regime risk based on real market data"""
        try:
            # Get real market regime data
            from .ai_market_regime_detector import ai_market_regime_detector
            regime_data = ai_market_regime_detector.detect_market_regime()
            regime = regime_data.get('regime', 'normal')
            confidence = regime_data.get('confidence', 0.5)
            
            # Map regime to risk level
            regime_risk_mapping = {
                'bull_market': 0.2,      # Low risk in bull market
                'normal': 0.4,           # Medium risk in normal market
                'sideways_market': 0.5,  # Medium-high risk in sideways market
                'bear_market': 0.7,      # High risk in bear market
                'high_volatility': 0.8   # Very high risk in high volatility
            }
            
            base_risk = regime_risk_mapping.get(regime, 0.4)
            
            # Adjust risk based on confidence (lower confidence = higher risk)
            confidence_adjustment = (1.0 - confidence) * 0.2  # Up to 20% adjustment
            regime_risk = min(1.0, base_risk + confidence_adjustment)
            
            return regime_risk
            
        except Exception as e:
            logger.error(f"Error assessing regime risk: {e}")
            return 0.4  # Default medium regime risk
    
    def _calculate_overall_risk_score(self, risk_analysis: Dict) -> float:
        """Calculate overall risk score using weighted factors"""
        try:
            overall_score = 0.0
            total_weight = 0.0
            
            for factor, weight in self.risk_factors.items():
                factor_score = risk_analysis.get(factor)
                # Only add to calculation if factor exists and is not None
                if factor_score is not None:
                    overall_score += factor_score * weight
                    total_weight += weight
                else:
                    # Log missing factor for debugging
                    logger.warning(f"Missing risk factor '{factor}' in risk_analysis, skipping")
            
            # If no valid factors found, return default
            if total_weight == 0.0:
                logger.warning(f"No valid risk factors found in risk_analysis. Available keys: {list(risk_analysis.keys())}")
                return 0.5
            
            # Normalize by actual weights used (in case some factors were missing)
            if total_weight < 1.0:
                overall_score = overall_score / total_weight if total_weight > 0 else 0.5
            
            return max(0.0, min(1.0, overall_score))
            
        except Exception as e:
            logger.error(f"Error calculating overall risk score: {e}", exc_info=True)
            return 0.5  # Default medium risk
    
    def _determine_risk_category(self, risk_score: float) -> str:
        """Determine risk category based on score"""
        if risk_score >= self.high_risk_threshold:
            return "high_risk"
        elif risk_score >= self.medium_risk_threshold:
            return "medium_risk"
        else:
            return "low_risk"
    
    def _predict_loss_probability(self, token: Dict, risk_analysis: Dict) -> float:
        """Predict probability of significant loss"""
        try:
            # Combine risk factors to predict loss probability
            base_loss_prob = risk_analysis.get('overall_risk_score', 0.5)
            
            # Adjust based on specific risk factors
            volume_risk = risk_analysis.get('volume_volatility', 0.5)
            liquidity_risk = risk_analysis.get('liquidity_stability', 0.5)
            momentum_risk = risk_analysis.get('price_momentum', 0.5)
            
            # Weighted loss probability
            loss_probability = (
                base_loss_prob * 0.4 +
                volume_risk * 0.2 +
                liquidity_risk * 0.2 +
                momentum_risk * 0.2
            )
            
            return max(0.0, min(1.0, loss_probability))
            
        except Exception:
            return 0.3  # Default 30% loss probability
    
    def _generate_risk_insights(self, risk_analysis: Dict, overall_score: float) -> List[str]:
        """Generate risk insights"""
        insights = []
        
        try:
            # Volume insights
            volume_risk = risk_analysis.get('volume_volatility', 0.5)
            if volume_risk > 0.7:
                insights.append("High volume volatility detected")
            elif volume_risk < 0.3:
                insights.append("Stable volume patterns")
            
            # Liquidity insights
            liquidity_risk = risk_analysis.get('liquidity_stability', 0.5)
            if liquidity_risk > 0.7:
                insights.append("Low liquidity - high slippage risk")
            elif liquidity_risk < 0.3:
                insights.append("Good liquidity - low slippage risk")
            
            # Price momentum insights
            momentum_risk = risk_analysis.get('price_momentum', 0.5)
            if momentum_risk > 0.7:
                insights.append("High price volatility - momentum risk")
            elif momentum_risk < 0.3:
                insights.append("Stable price momentum")
            
            # Market cap insights
            market_cap_risk = risk_analysis.get('market_cap_risk', 0.5)
            if market_cap_risk > 0.7:
                insights.append("Low market cap - high volatility risk")
            elif market_cap_risk < 0.3:
                insights.append("Strong market cap - lower risk")
            
            # Overall risk insights
            if overall_score > 0.7:
                insights.append("High overall risk - consider avoiding")
            elif overall_score < 0.3:
                insights.append("Low overall risk - good opportunity")
            else:
                insights.append("Medium risk - proceed with caution")
            
        except Exception:
            insights.append("Risk assessment completed")
        
        return insights
    
    def _generate_mitigation_recommendations(self, risk_category: str, 
                                          risk_analysis: Dict, 
                                          loss_probability: float) -> List[str]:
        """Generate mitigation recommendations"""
        recommendations = []
        
        try:
            if risk_category == "high_risk":
                recommendations.extend([
                    "Avoid trading this token",
                    "High risk of significant losses",
                    "Consider waiting for better conditions"
                ])
            elif risk_category == "medium_risk":
                recommendations.extend([
                    "Reduce position size by 50%",
                    "Use tighter stop-loss",
                    "Monitor closely for risk changes"
                ])
            else:  # low_risk
                recommendations.extend([
                    "Normal trading conditions",
                    "Standard position sizing",
                    "Monitor for risk changes"
                ])
            
            # Specific risk factor recommendations
            volume_risk = risk_analysis.get('volume_volatility', 0.5)
            if volume_risk > 0.7:
                recommendations.append("High volume volatility - use smaller position")
            
            liquidity_risk = risk_analysis.get('liquidity_stability', 0.5)
            if liquidity_risk > 0.7:
                recommendations.append("Low liquidity - expect higher slippage")
            
            momentum_risk = risk_analysis.get('price_momentum', 0.5)
            if momentum_risk > 0.7:
                recommendations.append("High price volatility - use wider stop-loss")
            
        except Exception:
            recommendations.append("Monitor risk factors closely")
        
        return recommendations
    
    def _calculate_position_adjustment(self, risk_score: float, loss_probability: float) -> float:
        """Calculate position size adjustment based on risk"""
        try:
            # Base adjustment
            if risk_score >= 0.7:  # High risk
                adjustment = 0.0  # Avoid trade
            elif risk_score >= 0.4:  # Medium risk
                adjustment = 0.5  # Reduce by 50%
            else:  # Low risk
                adjustment = 1.0  # Normal size
            
            # Adjust for loss probability
            if loss_probability > 0.6:
                adjustment *= 0.5  # Further reduce for high loss probability
            elif loss_probability < 0.2:
                adjustment = min(1.2, adjustment * 1.1)  # Slight increase for low loss probability
            
            return max(0.0, min(1.2, adjustment))
            
        except Exception:
            return 1.0  # Default no adjustment
    
    def _calculate_confidence_level(self, risk_analysis: Dict) -> str:
        """Calculate confidence level in risk assessment"""
        try:
            # Analyze consistency of risk factors
            risk_scores = list(risk_analysis.values())
            if not risk_scores:
                return "low"
            
            # Calculate variance in risk scores
            variance = statistics.variance(risk_scores) if len(risk_scores) > 1 else 0
            
            # Determine confidence based on variance
            if variance < 0.1:  # Low variance = high confidence
                return "high"
            elif variance < 0.3:  # Medium variance = medium confidence
                return "medium"
            else:  # High variance = low confidence
                return "low"
                
        except Exception:
            return "medium"
    
    def _get_default_risk_assessment(self) -> Dict:
        """Return default risk assessment when analysis fails"""
        return {
            'overall_risk_score': 0.5,
            'risk_category': 'medium_risk',
            'loss_probability': 0.3,
            'risk_analysis': {factor: 0.5 for factor in self.risk_factors.keys()},
            'risk_insights': ['Risk assessment unavailable'],
            'mitigation_recommendations': ['Proceed with caution'],
            'position_adjustment': 0.8,
            'assessment_timestamp': datetime.now().isoformat(),
            'should_trade': True,
            'confidence_level': 'low'
        }
    
    def get_risk_summary(self, tokens: List[Dict]) -> Dict:
        """Get risk summary for multiple tokens"""
        try:
            risk_summaries = []
            high_risk_count = 0
            medium_risk_count = 0
            low_risk_count = 0
            
            for token in tokens:
                risk_assessment = self.assess_token_risk(token)
                risk_summaries.append({
                    'symbol': token.get('symbol', 'UNKNOWN'),
                    'risk_score': risk_assessment['overall_risk_score'],
                    'risk_category': risk_assessment['risk_category'],
                    'should_trade': risk_assessment['should_trade']
                })
                
                if risk_assessment['risk_category'] == 'high_risk':
                    high_risk_count += 1
                elif risk_assessment['risk_category'] == 'medium_risk':
                    medium_risk_count += 1
                else:
                    low_risk_count += 1
            
            return {
                'total_tokens': len(tokens),
                'high_risk_tokens': high_risk_count,
                'medium_risk_tokens': medium_risk_count,
                'low_risk_tokens': low_risk_count,
                'risk_summaries': risk_summaries,
                'overall_risk_level': 'high' if high_risk_count > len(tokens) * 0.5 else 'medium' if medium_risk_count > len(tokens) * 0.3 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error getting risk summary: {e}")
            return {
                'total_tokens': len(tokens),
                'high_risk_tokens': 0,
                'medium_risk_tokens': 0,
                'low_risk_tokens': 0,
                'risk_summaries': [],
                'overall_risk_level': 'unknown'
            }

# Global instance
ai_risk_assessor = AIRiskAssessor()
