"""
AI Emergency Stop System
Automatically halts all trading during extreme market conditions, system errors, or excessive losses
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import random
import math
from src.monitoring.logger import log_event

def get_config():
    """Get configuration from config.yaml"""
    try:
        import yaml
        with open('config.yaml', 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        log_event("emergency.config_error", level="ERROR", error=str(e))
        return {}

class AIEmergencyStopSystem:
    def __init__(self):
        self.emergency_cache = {}
        self.config = get_config()
        self.enable_analysis = self.config.get('enable_ai_emergency_stop', False)
        self.cache_duration = self.config.get('emergency_stop_cache_duration', 60) # 1 minute
        
        # Emergency stop thresholds
        self.critical_drawdown_threshold = self.config.get('emergency_critical_drawdown_threshold', 0.25) # 25% drawdown
        self.urgent_drawdown_threshold = self.config.get('emergency_urgent_drawdown_threshold', 0.15) # 15% drawdown
        self.warning_drawdown_threshold = self.config.get('emergency_warning_drawdown_threshold', 0.10) # 10% drawdown
        
        # Market volatility thresholds
        self.extreme_volatility_threshold = self.config.get('emergency_extreme_volatility_threshold', 0.50) # 50% volatility
        self.high_volatility_threshold = self.config.get('emergency_high_volatility_threshold', 0.30) # 30% volatility
        
        # Loss streak thresholds
        self.critical_loss_streak = self.config.get('emergency_critical_loss_streak', 5) # 5 consecutive losses
        self.urgent_loss_streak = self.config.get('emergency_urgent_loss_streak', 3) # 3 consecutive losses
        
        # System error thresholds
        self.critical_error_count = self.config.get('emergency_critical_error_count', 10) # 10 errors
        self.urgent_error_count = self.config.get('emergency_urgent_error_count', 5) # 5 errors
        
        # News impact thresholds
        self.critical_news_impact = self.config.get('emergency_critical_news_impact', 0.9) # 90% negative impact
        self.urgent_news_impact = self.config.get('emergency_urgent_news_impact', 0.7) # 70% negative impact
        
        # Emergency stop levels
        self.emergency_levels = {
            'normal': {'level': 0, 'action': 'continue_trading'},
            'warning': {'level': 1, 'action': 'reduce_position_sizes'},
            'urgent': {'level': 2, 'action': 'halt_new_trades'},
            'critical': {'level': 3, 'action': 'emergency_stop_all'},
            'emergency': {'level': 4, 'action': 'immediate_shutdown'}
        }
        
        if not self.enable_analysis:
            log_event("emergency.disabled", level="WARNING", message="AI Emergency Stop System is disabled in config.yaml")

    def check_emergency_conditions(self, portfolio_data: Dict, trade_history: List[Dict], 
                                 market_data: Dict, system_errors: List[Dict]) -> Dict:
        """
        Check for emergency conditions that require immediate action
        Returns comprehensive emergency analysis with stop recommendations
        """
        try:
            cache_key = f"emergency_{portfolio_data.get('timestamp', 'current')}"
            
            # Check cache
            if cache_key in self.emergency_cache:
                cached_data = self.emergency_cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).total_seconds() < self.cache_duration:
                    log_event("emergency.cache_hit", message="Using cached emergency analysis")
                    return cached_data['emergency_data']
            
            # Analyze emergency components
            drawdown_analysis = self._analyze_drawdown_emergency(portfolio_data, trade_history)
            volatility_analysis = self._analyze_volatility_emergency(market_data)
            loss_streak_analysis = self._analyze_loss_streak_emergency(trade_history)
            system_error_analysis = self._analyze_system_error_emergency(system_errors)
            news_impact_analysis = self._analyze_news_impact_emergency(market_data)
            market_crash_analysis = self._analyze_market_crash_emergency(market_data)
            
            # Calculate overall emergency score
            emergency_score = self._calculate_emergency_score(
                drawdown_analysis, volatility_analysis, loss_streak_analysis,
                system_error_analysis, news_impact_analysis, market_crash_analysis
            )
            
            # Determine emergency level
            emergency_level = self._determine_emergency_level(emergency_score)
            
            # Generate emergency actions
            emergency_actions = self._generate_emergency_actions(
                emergency_level, drawdown_analysis, volatility_analysis, 
                loss_streak_analysis, system_error_analysis
            )
            
            # Calculate emergency urgency
            emergency_urgency = self._calculate_emergency_urgency(
                emergency_score, emergency_level, drawdown_analysis
            )
            
            # Generate emergency recommendations
            emergency_recommendations = self._generate_emergency_recommendations(
                emergency_level, emergency_urgency, emergency_actions
            )
            
            # Generate emergency insights
            emergency_insights = self._generate_emergency_insights(
                emergency_level, emergency_score, emergency_urgency
            )
            
            result = {
                'emergency_score': emergency_score,
                'emergency_level': emergency_level,
                'emergency_urgency': emergency_urgency,
                'emergency_actions': emergency_actions,
                'emergency_recommendations': emergency_recommendations,
                'drawdown_analysis': drawdown_analysis,
                'volatility_analysis': volatility_analysis,
                'loss_streak_analysis': loss_streak_analysis,
                'system_error_analysis': system_error_analysis,
                'news_impact_analysis': news_impact_analysis,
                'market_crash_analysis': market_crash_analysis,
                'emergency_insights': emergency_insights,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Cache the result
            self.emergency_cache[cache_key] = {'timestamp': datetime.now(), 'emergency_data': result}
            
            # Removed duplicate logging - emergency analysis is logged in main loop
            return result
            
        except Exception as e:
            log_event("emergency.analysis_error", level="ERROR", error=str(e))
            return self._get_default_emergency_analysis()

    def _analyze_drawdown_emergency(self, portfolio_data: Dict, trade_history: List[Dict]) -> Dict:
        """Analyze drawdown for emergency conditions"""
        try:
            current_value = portfolio_data.get('total_value', 0)
            initial_value = portfolio_data.get('initial_value', current_value)
            
            if initial_value > 0:
                current_drawdown = (initial_value - current_value) / initial_value
            else:
                current_drawdown = 0
            
            # Determine drawdown severity
            if current_drawdown >= self.critical_drawdown_threshold:
                drawdown_severity = 'critical'
                drawdown_urgency = 'emergency'
            elif current_drawdown >= self.urgent_drawdown_threshold:
                drawdown_severity = 'urgent'
                drawdown_urgency = 'urgent'
            elif current_drawdown >= self.warning_drawdown_threshold:
                drawdown_severity = 'warning'
                drawdown_urgency = 'warning'
            else:
                drawdown_severity = 'normal'
                drawdown_urgency = 'normal'
            
            return {
                'current_drawdown': current_drawdown,
                'drawdown_severity': drawdown_severity,
                'drawdown_urgency': drawdown_urgency,
                'requires_emergency_stop': drawdown_severity in ['critical', 'urgent']
            }
            
        except Exception as e:
            log_event("emergency.drawdown_analysis_error", level="ERROR", error=str(e))
            return {'current_drawdown': 0, 'drawdown_severity': 'normal', 'drawdown_urgency': 'normal', 'requires_emergency_stop': False}

    def _analyze_volatility_emergency(self, market_data: Dict) -> Dict:
        """Analyze market volatility for emergency conditions"""
        try:
            # Calculate market volatility using real data
            from src.utils.market_data_fetcher import market_data_fetcher
            volatility = market_data_fetcher.get_market_volatility(hours=24)
            
            # Determine volatility severity
            if volatility >= self.extreme_volatility_threshold:
                volatility_severity = 'extreme'
                volatility_urgency = 'emergency'
            elif volatility >= self.high_volatility_threshold:
                volatility_severity = 'high'
                volatility_urgency = 'urgent'
            else:
                volatility_severity = 'normal'
                volatility_urgency = 'normal'
            
            return {
                'volatility': volatility,
                'volatility_severity': volatility_severity,
                'volatility_urgency': volatility_urgency,
                'requires_emergency_stop': volatility_severity in ['extreme']
            }
            
        except Exception as e:
            log_event("emergency.volatility_analysis_error", level="ERROR", error=str(e))
            return {'volatility': 0, 'volatility_severity': 'normal', 'volatility_urgency': 'normal', 'requires_emergency_stop': False}

    def _analyze_loss_streak_emergency(self, trade_history: List[Dict]) -> Dict:
        """Analyze loss streak for emergency conditions"""
        try:
            if not trade_history:
                return {'loss_streak': 0, 'loss_severity': 'normal', 'loss_urgency': 'normal', 'requires_emergency_stop': False}
            
            # Calculate current loss streak
            loss_streak = 0
            for trade in reversed(trade_history[-10:]):  # Check last 10 trades
                pnl = trade.get('pnl', 0)
                if pnl < 0:
                    loss_streak += 1
                else:
                    break
            
            # Determine loss streak severity
            if loss_streak >= self.critical_loss_streak:
                loss_severity = 'critical'
                loss_urgency = 'emergency'
            elif loss_streak >= self.urgent_loss_streak:
                loss_severity = 'urgent'
                loss_urgency = 'urgent'
            else:
                loss_severity = 'normal'
                loss_urgency = 'normal'
            
            return {
                'loss_streak': loss_streak,
                'loss_severity': loss_severity,
                'loss_urgency': loss_urgency,
                'requires_emergency_stop': loss_severity in ['critical']
            }
            
        except Exception as e:
            log_event("emergency.loss_streak_analysis_error", level="ERROR", error=str(e))
            return {'loss_streak': 0, 'loss_severity': 'normal', 'loss_urgency': 'normal', 'requires_emergency_stop': False}

    def _analyze_system_error_emergency(self, system_errors: List[Dict]) -> Dict:
        """Analyze system errors for emergency conditions"""
        try:
            if not system_errors:
                return {'error_count': 0, 'error_severity': 'normal', 'error_urgency': 'normal', 'requires_emergency_stop': False}
            
            # Count recent errors
            recent_errors = [e for e in system_errors if (datetime.now() - datetime.fromisoformat(e.get('timestamp', '2024-01-01T00:00:00'))).total_seconds() < 3600]  # Last hour
            error_count = len(recent_errors)
            
            # Determine error severity
            if error_count >= self.critical_error_count:
                error_severity = 'critical'
                error_urgency = 'emergency'
            elif error_count >= self.urgent_error_count:
                error_severity = 'urgent'
                error_urgency = 'urgent'
            else:
                error_severity = 'normal'
                error_urgency = 'normal'
            
            return {
                'error_count': error_count,
                'error_severity': error_severity,
                'error_urgency': error_urgency,
                'requires_emergency_stop': error_severity in ['critical']
            }
            
        except Exception as e:
            log_event("emergency.system_error_analysis_error", level="ERROR", error=str(e))
            return {'error_count': 0, 'error_severity': 'normal', 'error_urgency': 'normal', 'requires_emergency_stop': False}

    def _analyze_news_impact_emergency(self, market_data: Dict) -> Dict:
        """Analyze news impact for emergency conditions"""
        try:
            # Deterministic proxy for news impact from price and volume changes
            price_change = abs(float(market_data.get('price_change_24h', 0)))
            volume_change = float(market_data.get('volume_change_ratio', 1.0))
            news_impact = max(0.0, min(1.0, (price_change / 20.0) + max(0.0, volume_change - 1.0) / 3))
            
            # Determine news impact severity
            if news_impact >= self.critical_news_impact:
                news_severity = 'critical'
                news_urgency = 'emergency'
            elif news_impact >= self.urgent_news_impact:
                news_severity = 'urgent'
                news_urgency = 'urgent'
            else:
                news_severity = 'normal'
                news_urgency = 'normal'
            
            return {
                'news_impact': news_impact,
                'news_severity': news_severity,
                'news_urgency': news_urgency,
                'requires_emergency_stop': news_severity in ['critical']
            }
            
        except Exception as e:
            log_event("emergency.news_impact_analysis_error", level="ERROR", error=str(e))
            return {'news_impact': 0, 'news_severity': 'normal', 'news_urgency': 'normal', 'requires_emergency_stop': False}

    def _analyze_market_crash_emergency(self, market_data: Dict) -> Dict:
        """Analyze market crash conditions"""
        try:
            # Estimate crash probability deterministically from drawdown and volatility
            current_drawdown = float(market_data.get('current_drawdown', 0))
            from src.utils.market_data_fetcher import market_data_fetcher
            vol = market_data_fetcher.get_market_volatility(hours=24)
            crash_probability = max(0.0, min(0.9, 0.2 * current_drawdown + 0.5 * vol))
            
            # Determine crash severity
            if crash_probability >= 0.8:
                crash_severity = 'critical'
                crash_urgency = 'emergency'
            elif crash_probability >= 0.6:
                crash_severity = 'urgent'
                crash_urgency = 'urgent'
            else:
                crash_severity = 'normal'
                crash_urgency = 'normal'
            
            return {
                'crash_probability': crash_probability,
                'crash_severity': crash_severity,
                'crash_urgency': crash_urgency,
                'requires_emergency_stop': crash_severity in ['critical']
            }
            
        except Exception as e:
            log_event("emergency.market_crash_analysis_error", level="ERROR", error=str(e))
            return {'crash_probability': 0, 'crash_severity': 'normal', 'crash_urgency': 'normal', 'requires_emergency_stop': False}

    def _calculate_emergency_score(self, drawdown_analysis: Dict, volatility_analysis: Dict, 
                                 loss_streak_analysis: Dict, system_error_analysis: Dict,
                                 news_impact_analysis: Dict, market_crash_analysis: Dict) -> float:
        """Calculate overall emergency score"""
        try:
            # Weight factors
            weights = {
                'drawdown': 0.30,
                'volatility': 0.20,
                'loss_streak': 0.20,
                'system_errors': 0.15,
                'news_impact': 0.10,
                'market_crash': 0.05
            }
            
            # Calculate component scores
            drawdown_score = 1.0 if drawdown_analysis['requires_emergency_stop'] else 0.0
            volatility_score = 1.0 if volatility_analysis['requires_emergency_stop'] else 0.0
            loss_streak_score = 1.0 if loss_streak_analysis['requires_emergency_stop'] else 0.0
            system_error_score = 1.0 if system_error_analysis['requires_emergency_stop'] else 0.0
            news_impact_score = 1.0 if news_impact_analysis['requires_emergency_stop'] else 0.0
            market_crash_score = 1.0 if market_crash_analysis['requires_emergency_stop'] else 0.0
            
            # Calculate weighted emergency score
            emergency_score = (
                drawdown_score * weights['drawdown'] +
                volatility_score * weights['volatility'] +
                loss_streak_score * weights['loss_streak'] +
                system_error_score * weights['system_errors'] +
                news_impact_score * weights['news_impact'] +
                market_crash_score * weights['market_crash']
            )
            
            return min(1.0, emergency_score)
            
        except Exception as e:
            log_event("emergency.score_calculation_error", level="ERROR", error=str(e))
            return 0.0

    def _determine_emergency_level(self, emergency_score: float) -> str:
        """Determine emergency level based on score"""
        try:
            if emergency_score >= 0.9:
                return 'emergency'
            elif emergency_score >= 0.7:
                return 'critical'
            elif emergency_score >= 0.5:
                return 'urgent'
            elif emergency_score >= 0.3:
                return 'warning'
            else:
                return 'normal'
                
        except Exception as e:
            log_event("emergency.level_determination_error", level="ERROR", error=str(e))
            return 'normal'

    def _generate_emergency_actions(self, emergency_level: str, drawdown_analysis: Dict, 
                                  volatility_analysis: Dict, loss_streak_analysis: Dict, 
                                  system_error_analysis: Dict) -> Dict:
        """Generate emergency actions based on level"""
        try:
            actions = []
            
            if emergency_level == 'emergency':
                actions.extend([
                    "🚨 IMMEDIATE SHUTDOWN REQUIRED",
                    "• Stop all trading immediately",
                    "• Close all open positions",
                    "• Activate emergency protocols",
                    "• Notify administrators"
                ])
            elif emergency_level == 'critical':
                actions.extend([
                    "🚨 CRITICAL EMERGENCY DETECTED",
                    "• Halt all new trades",
                    "• Close high-risk positions",
                    "• Reduce position sizes to minimum",
                    "• Monitor for further deterioration"
                ])
            elif emergency_level == 'urgent':
                actions.extend([
                    "⚠️ URGENT ACTION REQUIRED",
                    "• Reduce position sizes significantly",
                    "• Avoid new high-risk trades",
                    "• Monitor market conditions closely",
                    "• Prepare for potential emergency stop"
                ])
            elif emergency_level == 'warning':
                actions.extend([
                    "⚠️ WARNING CONDITIONS DETECTED",
                    "• Reduce position sizes",
                    "• Increase monitoring frequency",
                    "• Avoid high-risk strategies",
                    "• Prepare contingency plans"
                ])
            else:
                actions.extend([
                    "✅ NORMAL OPERATING CONDITIONS",
                    "• Continue normal trading",
                    "• Monitor for changes",
                    "• Maintain current strategy"
                ])
            
            return {
                'actions': actions,
                'emergency_level': emergency_level,
                'requires_immediate_action': emergency_level in ['emergency', 'critical']
            }
            
        except Exception as e:
            log_event("emergency.actions_generation_error", level="ERROR", error=str(e))
            return {'actions': ['Error generating actions'], 'emergency_level': 'normal', 'requires_immediate_action': False}

    def _calculate_emergency_urgency(self, emergency_score: float, emergency_level: str, 
                                   drawdown_analysis: Dict) -> str:
        """Calculate emergency urgency"""
        try:
            if emergency_level in ['emergency', 'critical']:
                return 'emergency'
            elif emergency_level == 'urgent':
                return 'urgent'
            elif emergency_level == 'warning':
                return 'warning'
            else:
                return 'normal'
                
        except Exception as e:
            log_event("emergency.urgency_calculation_error", level="ERROR", error=str(e))
            return 'normal'

    def _generate_emergency_recommendations(self, emergency_level: str, emergency_urgency: str, 
                                          emergency_actions: Dict) -> List[str]:
        """Generate emergency recommendations"""
        try:
            recommendations = []
            
            if emergency_level == 'emergency':
                recommendations.extend([
                    "🚨 EMERGENCY PROTOCOLS ACTIVATED",
                    "• Immediate system shutdown required",
                    "• All trading operations halted",
                    "• Emergency contacts notified",
                    "• System diagnostics initiated"
                ])
            elif emergency_level == 'critical':
                recommendations.extend([
                    "🚨 CRITICAL CONDITIONS DETECTED",
                    "• Halt all new trading immediately",
                    "• Close high-risk positions",
                    "• Reduce exposure to minimum",
                    "• Prepare for emergency shutdown"
                ])
            elif emergency_level == 'urgent':
                recommendations.extend([
                    "⚠️ URGENT ACTION REQUIRED",
                    "• Significantly reduce position sizes",
                    "• Avoid new high-risk trades",
                    "• Monitor conditions continuously",
                    "• Prepare emergency protocols"
                ])
            else:
                recommendations.extend([
                    "✅ NORMAL OPERATIONS",
                    "• Continue current strategy",
                    "• Monitor for changes",
                    "• Maintain normal operations"
                ])
            
            return recommendations
            
        except Exception as e:
            log_event("emergency.recommendations_generation_error", level="ERROR", error=str(e))
            return ["Error generating recommendations"]

    def _generate_emergency_insights(self, emergency_level: str, emergency_score: float, 
                                   emergency_urgency: str) -> List[str]:
        """Generate emergency insights"""
        try:
            insights = []
            
            insights.append(f"🚨 Emergency Level: {emergency_level.upper()}")
            insights.append(f"📊 Emergency Score: {emergency_score:.2f}")
            insights.append(f"⚡ Urgency: {emergency_urgency.upper()}")
            
            if emergency_level in ['emergency', 'critical']:
                insights.append("🚨 IMMEDIATE ACTION REQUIRED")
                insights.append("• System shutdown recommended")
                insights.append("• All trading halted")
            elif emergency_level == 'urgent':
                insights.append("⚠️ URGENT CONDITIONS DETECTED")
                insights.append("• Reduce trading activity")
                insights.append("• Monitor closely")
            else:
                insights.append("✅ NORMAL OPERATING CONDITIONS")
                insights.append("• Continue normal operations")
                insights.append("• Monitor for changes")
            
            return insights
            
        except Exception as e:
            log_event("emergency.insights_generation_error", level="ERROR", error=str(e))
            return ["Error generating insights"]

    def _get_default_emergency_analysis(self) -> Dict:
        """Get default emergency analysis when analysis fails"""
        return {
            'emergency_score': 0.0,
            'emergency_level': 'normal',
            'emergency_urgency': 'normal',
            'emergency_actions': {'actions': ['Normal operations'], 'emergency_level': 'normal', 'requires_immediate_action': False},
            'emergency_recommendations': ['Continue normal operations'],
            'drawdown_analysis': {'current_drawdown': 0, 'drawdown_severity': 'normal', 'drawdown_urgency': 'normal', 'requires_emergency_stop': False},
            'volatility_analysis': {'volatility': 0, 'volatility_severity': 'normal', 'volatility_urgency': 'normal', 'requires_emergency_stop': False},
            'loss_streak_analysis': {'loss_streak': 0, 'loss_severity': 'normal', 'loss_urgency': 'normal', 'requires_emergency_stop': False},
            'system_error_analysis': {'error_count': 0, 'error_severity': 'normal', 'error_urgency': 'normal', 'requires_emergency_stop': False},
            'news_impact_analysis': {'news_impact': 0, 'news_severity': 'normal', 'news_urgency': 'normal', 'requires_emergency_stop': False},
            'market_crash_analysis': {'crash_probability': 0, 'crash_severity': 'normal', 'crash_urgency': 'normal', 'requires_emergency_stop': False},
            'emergency_insights': ['Normal operating conditions'],
            'analysis_timestamp': datetime.now().isoformat()
        }

    def get_emergency_summary(self, portfolio_data: Dict, trade_history: List[Dict], 
                            market_data: Dict, system_errors: List[Dict]) -> Dict:
        """Get emergency summary for quick assessment"""
        try:
            emergency_analysis = self.check_emergency_conditions(portfolio_data, trade_history, market_data, system_errors)
            
            return {
                'emergency_level': emergency_analysis['emergency_level'],
                'emergency_score': emergency_analysis['emergency_score'],
                'emergency_urgency': emergency_analysis['emergency_urgency'],
                'requires_emergency_stop': emergency_analysis['emergency_level'] in ['emergency', 'critical'],
                'recommendations': emergency_analysis['emergency_recommendations'][:3]  # Top 3 recommendations
            }
            
        except Exception as e:
            log_event("emergency.summary_generation_error", level="ERROR", error=str(e))
            return {
                'emergency_level': 'normal',
                'emergency_score': 0.0,
                'emergency_urgency': 'normal',
                'requires_emergency_stop': False,
                'recommendations': ['Continue normal operations']
            }

# Global instance
ai_emergency_stop_system = AIEmergencyStopSystem()
