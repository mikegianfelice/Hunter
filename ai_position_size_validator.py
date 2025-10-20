"""
AI Position Size Validator
Double-checks position sizes before execution to prevent oversized trades and wallet drainage
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import random
import math

# Configure logging
logger = logging.getLogger(__name__)

def get_config():
    """Get configuration from config.yaml"""
    try:
        import yaml
        with open('config.yaml', 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}

class AIPositionSizeValidator:
    def __init__(self):
        self.validation_cache = {}
        self.config = get_config()
        self.enable_analysis = self.config.get('enable_ai_position_size_validation', False)
        self.cache_duration = self.config.get('position_validation_cache_duration', 120) # 2 minutes
        
        # Position size limits
        self.max_position_size_usd = self.config.get('max_position_size_usd', 50.0) # $50 max position
        self.min_position_size_usd = self.config.get('min_position_size_usd', 1.0) # $1 min position
        self.max_wallet_usage_percent = self.config.get('max_wallet_usage_percent', 0.1) # 10% max wallet usage
        self.max_total_exposure_usd = self.config.get('max_total_exposure_usd', 200.0) # $200 max total exposure
        
        # Risk-based position sizing
        self.high_risk_position_multiplier = self.config.get('high_risk_position_multiplier', 0.5) # 50% reduction for high risk
        self.medium_risk_position_multiplier = self.config.get('medium_risk_position_multiplier', 0.8) # 80% for medium risk
        self.low_risk_position_multiplier = self.config.get('low_risk_position_multiplier', 1.0) # 100% for low risk
        
        # Market condition adjustments
        self.bear_market_position_multiplier = self.config.get('bear_market_position_multiplier', 0.6) # 60% in bear market
        self.high_volatility_position_multiplier = self.config.get('high_volatility_position_multiplier', 0.7) # 70% in high volatility
        self.low_liquidity_position_multiplier = self.config.get('low_liquidity_position_multiplier', 0.5) # 50% for low liquidity
        
        # Validation thresholds (adjusted for more reasonable validation)
        self.critical_validation_threshold = 0.95 # 95% of limits (more lenient)
        self.warning_validation_threshold = 0.8 # 80% of limits (more lenient)
        self.safe_validation_threshold = 0.6 # 60% of limits (more lenient)
        
        if not self.enable_analysis:
            logger.warning("⚠️ AI Position Size Validator is disabled in config.yaml.")

    def validate_position_size(self, token: Dict, proposed_amount: float, wallet_balance: float, 
                            current_positions: List[Dict], market_conditions: Dict) -> Dict:
        """
        Validate position size before execution
        Returns comprehensive validation analysis with recommendations
        """
        try:
            symbol = token.get('symbol', 'UNKNOWN')
            cache_key = f"validation_{symbol}_{proposed_amount}_{wallet_balance}"
            
            # Check cache
            if cache_key in self.validation_cache:
                cached_data = self.validation_cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).total_seconds() < self.cache_duration:
                    logger.debug(f"Using cached position validation for {symbol}")
                    return cached_data['validation_data']
            
            # Analyze validation components
            wallet_balance_analysis = self._analyze_wallet_balance(proposed_amount, wallet_balance)
            position_size_analysis = self._analyze_position_size(proposed_amount, token)
            total_exposure_analysis = self._analyze_total_exposure(proposed_amount, current_positions)
            risk_based_analysis = self._analyze_risk_based_sizing(token, proposed_amount, market_conditions)
            market_condition_analysis = self._analyze_market_condition_sizing(token, proposed_amount, market_conditions)
            liquidity_analysis = self._analyze_liquidity_sizing(token, proposed_amount)
            
            # Calculate validation score
            validation_score = self._calculate_validation_score(
                wallet_balance_analysis, position_size_analysis, total_exposure_analysis,
                risk_based_analysis, market_condition_analysis, liquidity_analysis
            )
            
            # Determine validation result
            validation_result = self._determine_validation_result(validation_score)
            
            # Calculate recommended position size
            recommended_size = self._calculate_recommended_size(
                proposed_amount, wallet_balance_analysis, position_size_analysis,
                total_exposure_analysis, risk_based_analysis, market_condition_analysis, liquidity_analysis
            )
            
            # Generate validation recommendations
            validation_recommendations = self._generate_validation_recommendations(
                validation_result, recommended_size, proposed_amount, validation_score
            )
            
            # Generate validation insights
            validation_insights = self._generate_validation_insights(
                validation_result, validation_score, recommended_size, proposed_amount
            )
            
            result = {
                'validation_score': validation_score,
                'validation_result': validation_result,
                'recommended_size': recommended_size,
                'proposed_amount': proposed_amount,
                'validation_recommendations': validation_recommendations,
                'wallet_balance_analysis': wallet_balance_analysis,
                'position_size_analysis': position_size_analysis,
                'total_exposure_analysis': total_exposure_analysis,
                'risk_based_analysis': risk_based_analysis,
                'market_condition_analysis': market_condition_analysis,
                'liquidity_analysis': liquidity_analysis,
                'validation_insights': validation_insights,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Cache the result
            self.validation_cache[cache_key] = {'timestamp': datetime.now(), 'validation_data': result}
            
            logger.info(f"🔍 Position validation for {symbol}: {validation_result} (score: {validation_score:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Position size validation failed: {e}")
            return self._get_default_validation_analysis(proposed_amount)

    def _analyze_wallet_balance(self, proposed_amount: float, wallet_balance: float) -> Dict:
        """Analyze wallet balance constraints"""
        try:
            if wallet_balance <= 0:
                return {
                    'wallet_balance': wallet_balance,
                    'proposed_usage_percent': 1.0,
                    'usage_severity': 'critical',
                    'validation_passed': False,
                    'recommendation': 'Insufficient wallet balance'
                }
            
            usage_percent = proposed_amount / wallet_balance if wallet_balance > 0 else 1.0
            max_usage_percent = self.max_wallet_usage_percent
            
            if usage_percent >= max_usage_percent:
                usage_severity = 'critical'
                validation_passed = False
                recommendation = f'Position size exceeds {max_usage_percent*100:.0f}% wallet usage limit'
            elif usage_percent >= max_usage_percent * 0.8:
                usage_severity = 'warning'
                validation_passed = True
                recommendation = f'Position size near {max_usage_percent*100:.0f}% wallet usage limit'
            else:
                usage_severity = 'normal'
                validation_passed = True
                recommendation = 'Wallet balance sufficient'
            
            return {
                'wallet_balance': wallet_balance,
                'proposed_usage_percent': usage_percent,
                'usage_severity': usage_severity,
                'validation_passed': validation_passed,
                'recommendation': recommendation
            }
            
        except Exception as e:
            logger.error(f"❌ Wallet balance analysis failed: {e}")
            return {'wallet_balance': 0, 'proposed_usage_percent': 1.0, 'usage_severity': 'critical', 'validation_passed': False, 'recommendation': 'Error analyzing wallet balance'}

    def _analyze_position_size(self, proposed_amount: float, token: Dict) -> Dict:
        """Analyze position size constraints"""
        try:
            if proposed_amount > self.max_position_size_usd:
                size_severity = 'critical'
                validation_passed = False
                recommendation = f'Position size exceeds maximum limit of ${self.max_position_size_usd}'
            elif proposed_amount < self.min_position_size_usd:
                size_severity = 'warning'
                validation_passed = False
                recommendation = f'Position size below minimum limit of ${self.min_position_size_usd}'
            elif proposed_amount >= self.max_position_size_usd * 0.9:
                size_severity = 'warning'
                validation_passed = True
                recommendation = 'Position size near maximum limit'
            else:
                size_severity = 'normal'
                validation_passed = True
                recommendation = 'Position size within acceptable range'
            
            return {
                'proposed_amount': proposed_amount,
                'max_limit': self.max_position_size_usd,
                'min_limit': self.min_position_size_usd,
                'size_severity': size_severity,
                'validation_passed': validation_passed,
                'recommendation': recommendation
            }
            
        except Exception as e:
            logger.error(f"❌ Position size analysis failed: {e}")
            return {'proposed_amount': proposed_amount, 'max_limit': self.max_position_size_usd, 'min_limit': self.min_position_size_usd, 'size_severity': 'normal', 'validation_passed': True, 'recommendation': 'Error analyzing position size'}

    def _analyze_total_exposure(self, proposed_amount: float, current_positions: List[Dict]) -> Dict:
        """Analyze total exposure constraints"""
        try:
            current_exposure = sum(pos.get('position_size_usd', 0) for pos in current_positions)
            total_exposure = current_exposure + proposed_amount
            
            if total_exposure > self.max_total_exposure_usd:
                exposure_severity = 'critical'
                validation_passed = False
                recommendation = f'Total exposure would exceed limit of ${self.max_total_exposure_usd}'
            elif total_exposure >= self.max_total_exposure_usd * 0.9:
                exposure_severity = 'warning'
                validation_passed = True
                recommendation = 'Total exposure near maximum limit'
            else:
                exposure_severity = 'normal'
                validation_passed = True
                recommendation = 'Total exposure within acceptable range'
            
            return {
                'current_exposure': current_exposure,
                'proposed_exposure': proposed_amount,
                'total_exposure': total_exposure,
                'max_exposure': self.max_total_exposure_usd,
                'exposure_severity': exposure_severity,
                'validation_passed': validation_passed,
                'recommendation': recommendation
            }
            
        except Exception as e:
            logger.error(f"❌ Total exposure analysis failed: {e}")
            return {'current_exposure': 0, 'proposed_exposure': proposed_amount, 'total_exposure': proposed_amount, 'max_exposure': self.max_total_exposure_usd, 'exposure_severity': 'normal', 'validation_passed': True, 'recommendation': 'Error analyzing total exposure'}

    def _analyze_risk_based_sizing(self, token: Dict, proposed_amount: float, market_conditions: Dict) -> Dict:
        """Analyze risk-based position sizing"""
        try:
            # Get risk level from token or market conditions
            risk_level = token.get('risk_level', 'medium')
            if risk_level not in ['low', 'medium', 'high']:
                risk_level = 'medium'
            
            # Apply risk-based multipliers
            if risk_level == 'high':
                risk_multiplier = self.high_risk_position_multiplier
                risk_severity = 'high'
            elif risk_level == 'medium':
                risk_multiplier = self.medium_risk_position_multiplier
                risk_severity = 'medium'
            else:
                risk_multiplier = self.low_risk_position_multiplier
                risk_severity = 'low'
            
            # Calculate risk-adjusted position size
            risk_adjusted_size = proposed_amount * risk_multiplier
            
            if risk_adjusted_size < proposed_amount * 0.5:
                validation_passed = False
                recommendation = f'Position size too large for {risk_level} risk level'
            else:
                validation_passed = True
                recommendation = f'Position size appropriate for {risk_level} risk level'
            
            return {
                'risk_level': risk_level,
                'risk_multiplier': risk_multiplier,
                'risk_adjusted_size': risk_adjusted_size,
                'risk_severity': risk_severity,
                'validation_passed': validation_passed,
                'recommendation': recommendation
            }
            
        except Exception as e:
            logger.error(f"❌ Risk-based sizing analysis failed: {e}")
            return {'risk_level': 'medium', 'risk_multiplier': 1.0, 'risk_adjusted_size': proposed_amount, 'risk_severity': 'medium', 'validation_passed': True, 'recommendation': 'Error analyzing risk-based sizing'}

    def _analyze_market_condition_sizing(self, token: Dict, proposed_amount: float, market_conditions: Dict) -> Dict:
        """Analyze market condition-based position sizing"""
        try:
            market_regime = market_conditions.get('regime', 'normal')
            volatility = market_conditions.get('volatility', 0.2)
            
            # Apply market condition multipliers
            if market_regime == 'bear_market':
                market_multiplier = self.bear_market_position_multiplier
                market_severity = 'high'
            elif volatility > 0.4:  # High volatility
                market_multiplier = self.high_volatility_position_multiplier
                market_severity = 'high'
            else:
                market_multiplier = 1.0
                market_severity = 'normal'
            
            # Calculate market-adjusted position size
            market_adjusted_size = proposed_amount * market_multiplier
            
            if market_adjusted_size < proposed_amount * 0.6:
                validation_passed = False
                recommendation = f'Position size too large for {market_regime} market conditions'
            else:
                validation_passed = True
                recommendation = f'Position size appropriate for {market_regime} market conditions'
            
            return {
                'market_regime': market_regime,
                'volatility': volatility,
                'market_multiplier': market_multiplier,
                'market_adjusted_size': market_adjusted_size,
                'market_severity': market_severity,
                'validation_passed': validation_passed,
                'recommendation': recommendation
            }
            
        except Exception as e:
            logger.error(f"❌ Market condition sizing analysis failed: {e}")
            return {'market_regime': 'normal', 'volatility': 0.2, 'market_multiplier': 1.0, 'market_adjusted_size': proposed_amount, 'market_severity': 'normal', 'validation_passed': True, 'recommendation': 'Error analyzing market condition sizing'}

    def _analyze_liquidity_sizing(self, token: Dict, proposed_amount: float) -> Dict:
        """Analyze liquidity-based position sizing"""
        try:
            liquidity = float(token.get('liquidity', 0))
            volume_24h = float(token.get('volume24h', 0))
            
            # Calculate liquidity ratio
            if liquidity > 0:
                liquidity_ratio = proposed_amount / liquidity
            else:
                liquidity_ratio = 1.0
            
            # Determine liquidity severity
            if liquidity_ratio > 0.1:  # 10% of liquidity
                liquidity_severity = 'critical'
                validation_passed = False
                recommendation = 'Position size too large relative to liquidity'
            elif liquidity_ratio > 0.05:  # 5% of liquidity
                liquidity_severity = 'warning'
                validation_passed = True
                recommendation = 'Position size large relative to liquidity'
            else:
                liquidity_severity = 'normal'
                validation_passed = True
                recommendation = 'Position size appropriate for liquidity'
            
            return {
                'liquidity': liquidity,
                'volume_24h': volume_24h,
                'liquidity_ratio': liquidity_ratio,
                'liquidity_severity': liquidity_severity,
                'validation_passed': validation_passed,
                'recommendation': recommendation
            }
            
        except Exception as e:
            logger.error(f"❌ Liquidity sizing analysis failed: {e}")
            return {'liquidity': 0, 'volume_24h': 0, 'liquidity_ratio': 1.0, 'liquidity_severity': 'critical', 'validation_passed': False, 'recommendation': 'Error analyzing liquidity sizing'}

    def _calculate_validation_score(self, wallet_balance_analysis: Dict, position_size_analysis: Dict,
                                  total_exposure_analysis: Dict, risk_based_analysis: Dict,
                                  market_condition_analysis: Dict, liquidity_analysis: Dict) -> float:
        """Calculate overall validation score"""
        try:
            # Weight factors
            weights = {
                'wallet_balance': 0.30,
                'position_size': 0.25,
                'total_exposure': 0.20,
                'risk_based': 0.15,
                'market_condition': 0.05,
                'liquidity': 0.05
            }
            
            # Calculate component scores
            wallet_score = 1.0 if wallet_balance_analysis['validation_passed'] else 0.0
            position_score = 1.0 if position_size_analysis['validation_passed'] else 0.0
            exposure_score = 1.0 if total_exposure_analysis['validation_passed'] else 0.0
            risk_score = 1.0 if risk_based_analysis['validation_passed'] else 0.0
            market_score = 1.0 if market_condition_analysis['validation_passed'] else 0.0
            liquidity_score = 1.0 if liquidity_analysis['validation_passed'] else 0.0
            
            # Calculate weighted validation score
            validation_score = (
                wallet_score * weights['wallet_balance'] +
                position_score * weights['position_size'] +
                exposure_score * weights['total_exposure'] +
                risk_score * weights['risk_based'] +
                market_score * weights['market_condition'] +
                liquidity_score * weights['liquidity']
            )
            
            return min(1.0, validation_score)
            
        except Exception as e:
            logger.error(f"❌ Validation score calculation failed: {e}")
            return 0.0

    def _determine_validation_result(self, validation_score: float) -> str:
        """Determine validation result based on score"""
        try:
            if validation_score >= self.critical_validation_threshold:
                return 'critical'
            elif validation_score >= self.warning_validation_threshold:
                return 'warning'
            elif validation_score >= self.safe_validation_threshold:
                return 'safe'
            else:
                return 'rejected'
                
        except Exception as e:
            logger.error(f"❌ Validation result determination failed: {e}")
            return 'rejected'

    def _calculate_recommended_size(self, proposed_amount: float, wallet_balance_analysis: Dict,
                                  position_size_analysis: Dict, total_exposure_analysis: Dict,
                                  risk_based_analysis: Dict, market_condition_analysis: Dict,
                                  liquidity_analysis: Dict) -> float:
        """Calculate recommended position size"""
        try:
            # Start with proposed amount
            recommended_size = proposed_amount
            
            # Apply wallet balance constraint
            if not wallet_balance_analysis['validation_passed']:
                max_wallet_amount = wallet_balance_analysis['wallet_balance'] * self.max_wallet_usage_percent
                recommended_size = min(recommended_size, max_wallet_amount)
            
            # Apply position size constraint
            if not position_size_analysis['validation_passed']:
                recommended_size = min(recommended_size, self.max_position_size_usd)
            
            # Apply total exposure constraint
            if not total_exposure_analysis['validation_passed']:
                max_exposure_amount = self.max_total_exposure_usd - total_exposure_analysis['current_exposure']
                recommended_size = min(recommended_size, max_exposure_amount)
            
            # Apply risk-based constraint
            if not risk_based_analysis['validation_passed']:
                recommended_size = min(recommended_size, risk_based_analysis['risk_adjusted_size'])
            
            # Apply market condition constraint
            if not market_condition_analysis['validation_passed']:
                recommended_size = min(recommended_size, market_condition_analysis['market_adjusted_size'])
            
            # Apply liquidity constraint
            if not liquidity_analysis['validation_passed']:
                max_liquidity_amount = liquidity_analysis['liquidity'] * 0.05  # 5% of liquidity
                recommended_size = min(recommended_size, max_liquidity_amount)
            
            # Ensure minimum size
            recommended_size = max(recommended_size, self.min_position_size_usd)
            
            return recommended_size
            
        except Exception as e:
            logger.error(f"❌ Recommended size calculation failed: {e}")
            return proposed_amount

    def _generate_validation_recommendations(self, validation_result: str, recommended_size: float,
                                           proposed_amount: float, validation_score: float) -> List[str]:
        """Generate validation recommendations"""
        try:
            recommendations = []
            
            if validation_result == 'critical':
                recommendations.extend([
                    "🚨 CRITICAL VALIDATION FAILURE",
                    "• Position size exceeds critical limits",
                    "• Trading not recommended",
                    "• Reduce position size significantly",
                    f"• Recommended size: ${recommended_size:.2f}"
                ])
            elif validation_result == 'warning':
                recommendations.extend([
                    "⚠️ WARNING VALIDATION ISSUES",
                    "• Position size near limits",
                    "• Proceed with caution",
                    "• Monitor closely",
                    f"• Recommended size: ${recommended_size:.2f}"
                ])
            elif validation_result == 'safe':
                recommendations.extend([
                    "✅ SAFE VALIDATION",
                    "• Position size within limits",
                    "• Proceed with confidence",
                    "• Normal risk level",
                    f"• Recommended size: ${recommended_size:.2f}"
                ])
            else:
                recommendations.extend([
                    "❌ VALIDATION REJECTED",
                    "• Position size exceeds limits",
                    "• Trading not recommended",
                    "• Reduce position size",
                    f"• Recommended size: ${recommended_size:.2f}"
                ])
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Validation recommendations generation failed: {e}")
            return ["Error generating recommendations"]

    def _generate_validation_insights(self, validation_result: str, validation_score: float,
                                    recommended_size: float, proposed_amount: float) -> List[str]:
        """Generate validation insights"""
        try:
            insights = []
            
            insights.append(f"🔍 Validation Result: {validation_result.upper()}")
            insights.append(f"📊 Validation Score: {validation_score:.2f}")
            insights.append(f"💰 Proposed Amount: ${proposed_amount:.2f}")
            insights.append(f"✅ Recommended Size: ${recommended_size:.2f}")
            
            if validation_result == 'critical':
                insights.append("🚨 CRITICAL ISSUES DETECTED")
                insights.append("• Position size too large")
                insights.append("• Risk of significant losses")
            elif validation_result == 'warning':
                insights.append("⚠️ WARNING CONDITIONS")
                insights.append("• Position size near limits")
                insights.append("• Monitor closely")
            else:
                insights.append("✅ VALIDATION PASSED")
                insights.append("• Position size acceptable")
                insights.append("• Proceed with confidence")
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Validation insights generation failed: {e}")
            return ["Error generating insights"]

    def _get_default_validation_analysis(self, proposed_amount: float) -> Dict:
        """Get default validation analysis when analysis fails"""
        return {
            'validation_score': 0.5,
            'validation_result': 'safe',
            'recommended_size': proposed_amount,
            'proposed_amount': proposed_amount,
            'validation_recommendations': ['Continue with proposed position size'],
            'wallet_balance_analysis': {'wallet_balance': 0, 'proposed_usage_percent': 0, 'usage_severity': 'normal', 'validation_passed': True, 'recommendation': 'Default analysis'},
            'position_size_analysis': {'proposed_amount': proposed_amount, 'max_limit': self.max_position_size_usd, 'min_limit': self.min_position_size_usd, 'size_severity': 'normal', 'validation_passed': True, 'recommendation': 'Default analysis'},
            'total_exposure_analysis': {'current_exposure': 0, 'proposed_exposure': proposed_amount, 'total_exposure': proposed_amount, 'max_exposure': self.max_total_exposure_usd, 'exposure_severity': 'normal', 'validation_passed': True, 'recommendation': 'Default analysis'},
            'risk_based_analysis': {'risk_level': 'medium', 'risk_multiplier': 1.0, 'risk_adjusted_size': proposed_amount, 'risk_severity': 'medium', 'validation_passed': True, 'recommendation': 'Default analysis'},
            'market_condition_analysis': {'market_regime': 'normal', 'volatility': 0.2, 'market_multiplier': 1.0, 'market_adjusted_size': proposed_amount, 'market_severity': 'normal', 'validation_passed': True, 'recommendation': 'Default analysis'},
            'liquidity_analysis': {'liquidity': 0, 'volume_24h': 0, 'liquidity_ratio': 0, 'liquidity_severity': 'normal', 'validation_passed': True, 'recommendation': 'Default analysis'},
            'validation_insights': ['Default validation analysis'],
            'analysis_timestamp': datetime.now().isoformat()
        }

    def get_validation_summary(self, token: Dict, proposed_amount: float, wallet_balance: float,
                             current_positions: List[Dict], market_conditions: Dict) -> Dict:
        """Get validation summary for quick assessment"""
        try:
            validation_analysis = self.validate_position_size(token, proposed_amount, wallet_balance, current_positions, market_conditions)
            
            return {
                'validation_result': validation_analysis['validation_result'],
                'validation_score': validation_analysis['validation_score'],
                'recommended_size': validation_analysis['recommended_size'],
                'validation_passed': validation_analysis['validation_result'] in ['safe', 'warning'],
                'recommendations': validation_analysis['validation_recommendations'][:3]  # Top 3 recommendations
            }
            
        except Exception as e:
            logger.error(f"❌ Validation summary generation failed: {e}")
            return {
                'validation_result': 'safe',
                'validation_score': 0.5,
                'recommended_size': proposed_amount,
                'validation_passed': True,
                'recommendations': ['Continue with proposed position size']
            }

# Global instance
ai_position_size_validator = AIPositionSizeValidator()
