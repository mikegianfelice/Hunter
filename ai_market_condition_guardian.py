"""
AI Market Condition Guardian
Blocks trading during unfavorable market conditions, crashes, manipulation, and extreme volatility
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from market_data_fetcher import market_data_fetcher
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

class AIMarketConditionGuardian:
    def __init__(self):
        self.guardian_cache = {}
        self.config = get_config()
        self.enable_analysis = self.config.get('enable_ai_market_condition_guardian', False)
        self.cache_duration = self.config.get('market_guardian_cache_duration', 60) # 1 minute
        
        # Market condition thresholds
        self.extreme_volatility_threshold = self.config.get('extreme_volatility_threshold', 0.50) # 50% volatility
        self.high_volatility_threshold = self.config.get('high_volatility_threshold', 0.30) # 30% volatility
        self.critical_liquidity_threshold = self.config.get('critical_liquidity_threshold', 0.1) # 10% liquidity drop
        self.low_liquidity_threshold = self.config.get('low_liquidity_threshold', 0.3) # 30% liquidity drop
        
        # Market crash detection thresholds
        self.critical_price_drop_threshold = self.config.get('critical_price_drop_threshold', 0.20) # 20% price drop
        self.significant_price_drop_threshold = self.config.get('significant_price_drop_threshold', 0.10) # 10% price drop
        self.volume_spike_threshold = self.config.get('volume_spike_threshold', 5.0) # 5x volume spike
        
        # Manipulation detection thresholds
        self.manipulation_price_spike_threshold = self.config.get('manipulation_price_spike_threshold', 0.15) # 15% price spike
        self.manipulation_volume_spike_threshold = self.config.get('manipulation_volume_spike_threshold', 3.0) # 3x volume spike
        self.manipulation_pattern_threshold = self.config.get('manipulation_pattern_threshold', 0.8) # 80% pattern match
        
        # News impact thresholds
        self.critical_news_impact_threshold = self.config.get('critical_news_impact_threshold', 0.9) # 90% negative impact
        self.significant_news_impact_threshold = self.config.get('significant_news_impact_threshold', 0.7) # 70% negative impact
        
        # Market condition levels
        self.condition_levels = {
            'normal': {'level': 0, 'action': 'allow_trading'},
            'caution': {'level': 1, 'action': 'reduce_position_sizes'},
            'warning': {'level': 2, 'action': 'halt_high_risk_trades'},
            'danger': {'level': 3, 'action': 'halt_all_trading'},
            'critical': {'level': 4, 'action': 'emergency_shutdown'}
        }
        
        if not self.enable_analysis:
            logger.warning("⚠️ AI Market Condition Guardian is disabled in config.yaml.")

    def check_market_conditions(self, market_data: Dict, token_data: Dict, 
                              news_data: Dict, historical_data: Dict) -> Dict:
        """
        Check market conditions for trading safety
        Returns comprehensive market condition analysis with trading recommendations
        """
        try:
            symbol = token_data.get('symbol', 'UNKNOWN')
            cache_key = f"guardian_{symbol}_{market_data.get('timestamp', 'current')}"
            
            # Check cache
            if cache_key in self.guardian_cache:
                cached_data = self.guardian_cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).total_seconds() < self.cache_duration:
                    logger.debug(f"Using cached market condition analysis for {symbol}")
                    return cached_data['guardian_data']
            
            # Analyze market condition components
            volatility_analysis = self._analyze_market_volatility(market_data, token_data)
            liquidity_analysis = self._analyze_market_liquidity(market_data, token_data)
            crash_detection_analysis = self._analyze_crash_conditions(market_data, token_data, historical_data)
            manipulation_analysis = self._analyze_manipulation_signals(market_data, token_data)
            news_impact_analysis = self._analyze_news_impact(market_data, news_data)
            correlation_analysis = self._analyze_market_correlation(market_data, historical_data)
            
            # Calculate overall market condition score
            condition_score = self._calculate_condition_score(
                volatility_analysis, liquidity_analysis, crash_detection_analysis,
                manipulation_analysis, news_impact_analysis, correlation_analysis
            )
            
            # Determine market condition level
            condition_level = self._determine_condition_level(condition_score)
            
            # Generate trading recommendations
            trading_recommendations = self._generate_trading_recommendations(
                condition_level, volatility_analysis, liquidity_analysis,
                crash_detection_analysis, manipulation_analysis, news_impact_analysis
            )
            
            # Calculate trading safety
            trading_safety = self._calculate_trading_safety(
                condition_score, condition_level, volatility_analysis, crash_detection_analysis
            )
            
            # Generate market insights
            market_insights = self._generate_market_insights(
                condition_level, condition_score, trading_safety
            )
            
            result = {
                'condition_score': condition_score,
                'condition_level': condition_level,
                'trading_safety': trading_safety,
                'trading_recommendations': trading_recommendations,
                'volatility_analysis': volatility_analysis,
                'liquidity_analysis': liquidity_analysis,
                'crash_detection_analysis': crash_detection_analysis,
                'manipulation_analysis': manipulation_analysis,
                'news_impact_analysis': news_impact_analysis,
                'correlation_analysis': correlation_analysis,
                'market_insights': market_insights,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Cache the result
            self.guardian_cache[cache_key] = {'timestamp': datetime.now(), 'guardian_data': result}
            
            logger.info(f"🛡️ Market condition analysis for {symbol}: {condition_level} (score: {condition_score:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Market condition analysis failed: {e}")
            return self._get_default_guardian_analysis()

    def _analyze_market_volatility(self, market_data: Dict, token_data: Dict) -> Dict:
        """Analyze market volatility conditions"""
        try:
            # Calculate volatility using real market data
            volatility = market_data_fetcher.get_market_volatility(hours=24)
            
            if volatility >= self.extreme_volatility_threshold:
                volatility_severity = 'extreme'
                volatility_urgency = 'critical'
                recommendation = 'Extreme volatility detected - avoid trading'
            elif volatility >= self.high_volatility_threshold:
                volatility_severity = 'high'
                volatility_urgency = 'urgent'
                recommendation = 'High volatility detected - reduce position sizes'
            else:
                volatility_severity = 'normal'
                volatility_urgency = 'normal'
                recommendation = 'Volatility within acceptable range'
            
            return {
                'volatility': volatility,
                'volatility_severity': volatility_severity,
                'volatility_urgency': volatility_urgency,
                'recommendation': recommendation,
                'trading_allowed': volatility_severity not in ['extreme']
            }
            
        except Exception as e:
            logger.error(f"❌ Market volatility analysis failed: {e}")
            return {'volatility': 0, 'volatility_severity': 'normal', 'volatility_urgency': 'normal', 'recommendation': 'Error analyzing volatility', 'trading_allowed': True}

    def _analyze_market_liquidity(self, market_data: Dict, token_data: Dict) -> Dict:
        """Analyze market liquidity conditions"""
        try:
            current_liquidity = float(token_data.get('liquidity', 0))
            historical_liquidity = float(token_data.get('historical_liquidity', current_liquidity))
            
            if historical_liquidity > 0:
                liquidity_change = (current_liquidity - historical_liquidity) / historical_liquidity
            else:
                liquidity_change = 0
            
            if liquidity_change <= -self.critical_liquidity_threshold:
                liquidity_severity = 'critical'
                liquidity_urgency = 'critical'
                recommendation = 'Critical liquidity drop detected - halt trading'
            elif liquidity_change <= -self.low_liquidity_threshold:
                liquidity_severity = 'low'
                liquidity_urgency = 'urgent'
                recommendation = 'Low liquidity detected - reduce position sizes'
            else:
                liquidity_severity = 'normal'
                liquidity_urgency = 'normal'
                recommendation = 'Liquidity within acceptable range'
            
            return {
                'current_liquidity': current_liquidity,
                'historical_liquidity': historical_liquidity,
                'liquidity_change': liquidity_change,
                'liquidity_severity': liquidity_severity,
                'liquidity_urgency': liquidity_urgency,
                'recommendation': recommendation,
                'trading_allowed': liquidity_severity not in ['critical']
            }
            
        except Exception as e:
            logger.error(f"❌ Market liquidity analysis failed: {e}")
            return {'current_liquidity': 0, 'historical_liquidity': 0, 'liquidity_change': 0, 'liquidity_severity': 'normal', 'liquidity_urgency': 'normal', 'recommendation': 'Error analyzing liquidity', 'trading_allowed': True}

    def _analyze_crash_conditions(self, market_data: Dict, token_data: Dict, historical_data: Dict) -> Dict:
        """Analyze market crash conditions"""
        try:
            current_price = float(token_data.get('priceUsd', 0))
            historical_price = float(token_data.get('historical_price', current_price))
            current_volume = float(token_data.get('volume24h', 0))
            historical_volume = float(token_data.get('historical_volume', current_volume))
            
            # Calculate price change
            if historical_price > 0:
                price_change = (current_price - historical_price) / historical_price
            else:
                price_change = 0
            
            # Calculate volume spike
            if historical_volume > 0:
                volume_spike = current_volume / historical_volume
            else:
                volume_spike = 1.0
            
            # Determine crash severity
            if price_change <= -self.critical_price_drop_threshold and volume_spike >= self.volume_spike_threshold:
                crash_severity = 'critical'
                crash_urgency = 'critical'
                recommendation = 'Critical market crash detected - emergency halt required'
            elif price_change <= -self.significant_price_drop_threshold:
                crash_severity = 'significant'
                crash_urgency = 'urgent'
                recommendation = 'Significant price drop detected - halt trading'
            else:
                crash_severity = 'normal'
                crash_urgency = 'normal'
                recommendation = 'No crash conditions detected'
            
            return {
                'price_change': price_change,
                'volume_spike': volume_spike,
                'crash_severity': crash_severity,
                'crash_urgency': crash_urgency,
                'recommendation': recommendation,
                'trading_allowed': crash_severity not in ['critical', 'significant']
            }
            
        except Exception as e:
            logger.error(f"❌ Crash condition analysis failed: {e}")
            return {'price_change': 0, 'volume_spike': 1.0, 'crash_severity': 'normal', 'crash_urgency': 'normal', 'recommendation': 'Error analyzing crash conditions', 'trading_allowed': True}

    def _analyze_manipulation_signals(self, market_data: Dict, token_data: Dict) -> Dict:
        """Analyze market manipulation signals"""
        try:
            current_price = float(token_data.get('priceUsd', 0))
            historical_price = float(token_data.get('historical_price', current_price))
            current_volume = float(token_data.get('volume24h', 0))
            historical_volume = float(token_data.get('historical_volume', current_volume))
            
            # Calculate price spike
            if historical_price > 0:
                price_spike = abs(current_price - historical_price) / historical_price
            else:
                price_spike = 0
            
            # Calculate volume spike
            if historical_volume > 0:
                volume_spike = current_volume / historical_volume
            else:
                volume_spike = 1.0
            
            # Calculate manipulation probability
            manipulation_probability = (price_spike + (volume_spike - 1)) / 2
            
            if manipulation_probability >= self.manipulation_pattern_threshold:
                manipulation_severity = 'high'
                manipulation_urgency = 'urgent'
                recommendation = 'High manipulation probability detected - avoid trading'
            elif manipulation_probability >= self.manipulation_pattern_threshold * 0.7:
                manipulation_severity = 'medium'
                manipulation_urgency = 'warning'
                recommendation = 'Medium manipulation probability - proceed with caution'
            else:
                manipulation_severity = 'low'
                manipulation_urgency = 'normal'
                recommendation = 'Low manipulation probability - normal trading'
            
            return {
                'price_spike': price_spike,
                'volume_spike': volume_spike,
                'manipulation_probability': manipulation_probability,
                'manipulation_severity': manipulation_severity,
                'manipulation_urgency': manipulation_urgency,
                'recommendation': recommendation,
                'trading_allowed': manipulation_severity not in ['high']
            }
            
        except Exception as e:
            logger.error(f"❌ Manipulation signal analysis failed: {e}")
            return {'price_spike': 0, 'volume_spike': 1.0, 'manipulation_probability': 0, 'manipulation_severity': 'low', 'manipulation_urgency': 'normal', 'recommendation': 'Error analyzing manipulation signals', 'trading_allowed': True}

    def _analyze_news_impact(self, market_data: Dict, news_data: Dict) -> Dict:
        """Analyze news impact on market conditions"""
        try:
            # Approximate news impact deterministically from price and volume changes
            price_change = abs(float(token_data.get('price_change_24h', 0)))
            vol = float(token_data.get('volume24h', 0))
            liq = float(token_data.get('liquidity', max(vol, 1)))
            news_impact = max(0.0, min(1.0, (price_change / 20.0) + min(0.5, vol / max(liq, 1_000_000))))
            news_sentiment = news_data.get('sentiment', 'neutral')
            
            if news_impact >= self.critical_news_impact_threshold and news_sentiment == 'negative':
                news_severity = 'critical'
                news_urgency = 'critical'
                recommendation = 'Critical negative news impact - halt trading'
            elif news_impact >= self.significant_news_impact_threshold and news_sentiment == 'negative':
                news_severity = 'significant'
                news_urgency = 'urgent'
                recommendation = 'Significant negative news impact - reduce trading'
            else:
                news_severity = 'normal'
                news_urgency = 'normal'
                recommendation = 'News impact within acceptable range'
            
            return {
                'news_impact': news_impact,
                'news_sentiment': news_sentiment,
                'news_severity': news_severity,
                'news_urgency': news_urgency,
                'recommendation': recommendation,
                'trading_allowed': news_severity not in ['critical', 'significant']
            }
            
        except Exception as e:
            logger.error(f"❌ News impact analysis failed: {e}")
            return {'news_impact': 0, 'news_sentiment': 'neutral', 'news_severity': 'normal', 'news_urgency': 'normal', 'recommendation': 'Error analyzing news impact', 'trading_allowed': True}

    def _analyze_market_correlation(self, market_data: Dict, historical_data: Dict) -> Dict:
        """Analyze market correlation breakdown"""
        try:
            # Estimate correlation breakdown using BTC/ETH trend divergence
            btc_trend = market_data_fetcher.get_btc_trend(hours=24)
            eth_trend = market_data_fetcher.get_eth_trend(hours=24)
            correlation_breakdown = min(0.5, abs(btc_trend - eth_trend))
            
            if correlation_breakdown >= 0.4:
                correlation_severity = 'high'
                correlation_urgency = 'urgent'
                recommendation = 'High correlation breakdown - market instability detected'
            elif correlation_breakdown >= 0.2:
                correlation_severity = 'medium'
                correlation_urgency = 'warning'
                recommendation = 'Medium correlation breakdown - monitor closely'
            else:
                correlation_severity = 'low'
                correlation_urgency = 'normal'
                recommendation = 'Correlation within normal range'
            
            return {
                'correlation_breakdown': correlation_breakdown,
                'correlation_severity': correlation_severity,
                'correlation_urgency': correlation_urgency,
                'recommendation': recommendation,
                'trading_allowed': correlation_severity not in ['high']
            }
            
        except Exception as e:
            logger.error(f"❌ Market correlation analysis failed: {e}")
            return {'correlation_breakdown': 0, 'correlation_severity': 'low', 'correlation_urgency': 'normal', 'recommendation': 'Error analyzing market correlation', 'trading_allowed': True}

    def _calculate_condition_score(self, volatility_analysis: Dict, liquidity_analysis: Dict,
                                 crash_detection_analysis: Dict, manipulation_analysis: Dict,
                                 news_impact_analysis: Dict, correlation_analysis: Dict) -> float:
        """Calculate overall market condition score"""
        try:
            # Weight factors
            weights = {
                'volatility': 0.25,
                'liquidity': 0.20,
                'crash_detection': 0.25,
                'manipulation': 0.15,
                'news_impact': 0.10,
                'correlation': 0.05
            }
            
            # Calculate component scores
            volatility_score = 1.0 if volatility_analysis['trading_allowed'] else 0.0
            liquidity_score = 1.0 if liquidity_analysis['trading_allowed'] else 0.0
            crash_score = 1.0 if crash_detection_analysis['trading_allowed'] else 0.0
            manipulation_score = 1.0 if manipulation_analysis['trading_allowed'] else 0.0
            news_score = 1.0 if news_impact_analysis['trading_allowed'] else 0.0
            correlation_score = 1.0 if correlation_analysis['trading_allowed'] else 0.0
            
            # Calculate weighted condition score
            condition_score = (
                volatility_score * weights['volatility'] +
                liquidity_score * weights['liquidity'] +
                crash_score * weights['crash_detection'] +
                manipulation_score * weights['manipulation'] +
                news_score * weights['news_impact'] +
                correlation_score * weights['correlation']
            )
            
            return min(1.0, condition_score)
            
        except Exception as e:
            logger.error(f"❌ Condition score calculation failed: {e}")
            return 0.0

    def _determine_condition_level(self, condition_score: float) -> str:
        """Determine market condition level based on score"""
        try:
            if condition_score >= 0.9:
                return 'normal'
            elif condition_score >= 0.7:
                return 'caution'
            elif condition_score >= 0.5:
                return 'warning'
            elif condition_score >= 0.3:
                return 'danger'
            else:
                return 'critical'
                
        except Exception as e:
            logger.error(f"❌ Condition level determination failed: {e}")
            return 'danger'

    def _generate_trading_recommendations(self, condition_level: str, volatility_analysis: Dict,
                                        liquidity_analysis: Dict, crash_detection_analysis: Dict,
                                        manipulation_analysis: Dict, news_impact_analysis: Dict) -> List[str]:
        """Generate trading recommendations based on market conditions"""
        try:
            recommendations = []
            
            if condition_level == 'critical':
                recommendations.extend([
                    "🚨 CRITICAL MARKET CONDITIONS",
                    "• Halt all trading immediately",
                    "• Close all open positions",
                    "• Activate emergency protocols",
                    "• Monitor for market recovery"
                ])
            elif condition_level == 'danger':
                recommendations.extend([
                    "🚨 DANGEROUS MARKET CONDITIONS",
                    "• Halt all new trading",
                    "• Close high-risk positions",
                    "• Reduce exposure significantly",
                    "• Monitor market conditions closely"
                ])
            elif condition_level == 'warning':
                recommendations.extend([
                    "⚠️ WARNING MARKET CONDITIONS",
                    "• Reduce position sizes",
                    "• Avoid high-risk trades",
                    "• Monitor volatility closely",
                    "• Prepare for potential halt"
                ])
            elif condition_level == 'caution':
                recommendations.extend([
                    "⚠️ CAUTION MARKET CONDITIONS",
                    "• Reduce position sizes",
                    "• Avoid high-risk strategies",
                    "• Monitor market conditions",
                    "• Prepare contingency plans"
                ])
            else:
                recommendations.extend([
                    "✅ NORMAL MARKET CONDITIONS",
                    "• Continue normal trading",
                    "• Monitor for changes",
                    "• Maintain current strategy"
                ])
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Trading recommendations generation failed: {e}")
            return ["Error generating recommendations"]

    def _calculate_trading_safety(self, condition_score: float, condition_level: str,
                                volatility_analysis: Dict, crash_detection_analysis: Dict) -> str:
        """Calculate trading safety level"""
        try:
            if condition_level in ['critical', 'danger']:
                return 'unsafe'
            elif condition_level == 'warning':
                return 'risky'
            elif condition_level == 'caution':
                return 'cautious'
            else:
                return 'safe'
                
        except Exception as e:
            logger.error(f"❌ Trading safety calculation failed: {e}")
            return 'cautious'

    def _generate_market_insights(self, condition_level: str, condition_score: float,
                                trading_safety: str) -> List[str]:
        """Generate market insights"""
        try:
            insights = []
            
            insights.append(f"🛡️ Market Condition: {condition_level.upper()}")
            insights.append(f"📊 Condition Score: {condition_score:.2f}")
            insights.append(f"🔒 Trading Safety: {trading_safety.upper()}")
            
            if condition_level in ['critical', 'danger']:
                insights.append("🚨 CRITICAL MARKET CONDITIONS")
                insights.append("• Trading not recommended")
                insights.append("• Market instability detected")
            elif condition_level == 'warning':
                insights.append("⚠️ WARNING MARKET CONDITIONS")
                insights.append("• Proceed with extreme caution")
                insights.append("• Monitor closely")
            else:
                insights.append("✅ NORMAL MARKET CONDITIONS")
                insights.append("• Safe trading conditions")
                insights.append("• Monitor for changes")
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Market insights generation failed: {e}")
            return ["Error generating insights"]

    def _get_default_guardian_analysis(self) -> Dict:
        """Get default guardian analysis when analysis fails"""
        return {
            'condition_score': 0.5,
            'condition_level': 'caution',
            'trading_safety': 'cautious',
            'trading_recommendations': ['Proceed with caution'],
            'volatility_analysis': {'volatility': 0, 'volatility_severity': 'normal', 'volatility_urgency': 'normal', 'recommendation': 'Default analysis', 'trading_allowed': True},
            'liquidity_analysis': {'current_liquidity': 0, 'historical_liquidity': 0, 'liquidity_change': 0, 'liquidity_severity': 'normal', 'liquidity_urgency': 'normal', 'recommendation': 'Default analysis', 'trading_allowed': True},
            'crash_detection_analysis': {'price_change': 0, 'volume_spike': 1.0, 'crash_severity': 'normal', 'crash_urgency': 'normal', 'recommendation': 'Default analysis', 'trading_allowed': True},
            'manipulation_analysis': {'price_spike': 0, 'volume_spike': 1.0, 'manipulation_probability': 0, 'manipulation_severity': 'low', 'manipulation_urgency': 'normal', 'recommendation': 'Default analysis', 'trading_allowed': True},
            'news_impact_analysis': {'news_impact': 0, 'news_sentiment': 'neutral', 'news_severity': 'normal', 'news_urgency': 'normal', 'recommendation': 'Default analysis', 'trading_allowed': True},
            'correlation_analysis': {'correlation_breakdown': 0, 'correlation_severity': 'low', 'correlation_urgency': 'normal', 'recommendation': 'Default analysis', 'trading_allowed': True},
            'market_insights': ['Default market condition analysis'],
            'analysis_timestamp': datetime.now().isoformat()
        }

    def get_guardian_summary(self, market_data: Dict, token_data: Dict, 
                           news_data: Dict, historical_data: Dict) -> Dict:
        """Get guardian summary for quick assessment"""
        try:
            guardian_analysis = self.check_market_conditions(market_data, token_data, news_data, historical_data)
            
            return {
                'condition_level': guardian_analysis['condition_level'],
                'condition_score': guardian_analysis['condition_score'],
                'trading_safety': guardian_analysis['trading_safety'],
                'trading_allowed': guardian_analysis['condition_level'] in ['normal', 'caution'],
                'recommendations': guardian_analysis['trading_recommendations'][:3]  # Top 3 recommendations
            }
            
        except Exception as e:
            logger.error(f"❌ Guardian summary generation failed: {e}")
            return {
                'condition_level': 'caution',
                'condition_score': 0.5,
                'trading_safety': 'cautious',
                'trading_allowed': True,
                'recommendations': ['Proceed with caution']
            }

# Global instance
ai_market_condition_guardian = AIMarketConditionGuardian()
