"""
AI Trade Execution Monitor
Monitors trade execution success and handles failures, retries, and stuck transactions
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

class AITradeExecutionMonitor:
    def __init__(self):
        self.monitoring_cache = {}
        self.config = get_config()
        self.enable_analysis = self.config.get('enable_ai_trade_execution_monitoring', False)
        self.cache_duration = self.config.get('execution_monitoring_cache_duration', 60) # 1 minute
        
        # Execution monitoring thresholds
        self.max_execution_time_seconds = self.config.get('max_execution_time_seconds', 300) # 5 minutes
        self.max_retry_attempts = self.config.get('max_retry_attempts', 3)
        self.retry_delay_seconds = self.config.get('retry_delay_seconds', 30) # 30 seconds
        self.stuck_transaction_threshold = self.config.get('stuck_transaction_threshold', 600) # 10 minutes
        
        # Gas optimization thresholds
        self.low_gas_threshold = self.config.get('low_gas_threshold', 0.001) # 0.1% gas price
        self.high_gas_threshold = self.config.get('high_gas_threshold', 0.01) # 1% gas price
        self.gas_optimization_enabled = self.config.get('gas_optimization_enabled', True)
        
        # Slippage monitoring thresholds
        self.max_slippage_threshold = self.config.get('max_slippage_threshold', 0.05) # 5% slippage
        self.high_slippage_threshold = self.config.get('high_slippage_threshold', 0.03) # 3% slippage
        self.slippage_monitoring_enabled = self.config.get('slippage_monitoring_enabled', True)
        
        # Execution success thresholds
        self.success_rate_threshold = self.config.get('success_rate_threshold', 0.8) # 80% success rate
        self.warning_success_rate = self.config.get('warning_success_rate', 0.9) # 90% success rate
        
        # Execution monitoring levels
        self.monitoring_levels = {
            'normal': {'level': 0, 'action': 'continue_monitoring'},
            'warning': {'level': 1, 'action': 'increase_monitoring'},
            'urgent': {'level': 2, 'action': 'reduce_execution_frequency'},
            'critical': {'level': 3, 'action': 'halt_execution'},
            'emergency': {'level': 4, 'action': 'emergency_shutdown'}
        }
        
        if not self.enable_analysis:
            logger.warning("⚠️ AI Trade Execution Monitor is disabled in config.yaml.")

    def monitor_trade_execution(self, trade_data: Dict, execution_history: List[Dict], 
                              market_conditions: Dict) -> Dict:
        """
        Monitor trade execution and handle failures
        Returns comprehensive execution monitoring analysis
        """
        try:
            trade_id = trade_data.get('trade_id', 'unknown')
            cache_key = f"execution_{trade_id}_{trade_data.get('timestamp', 'current')}"
            
            # Check cache
            if cache_key in self.monitoring_cache:
                cached_data = self.monitoring_cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).total_seconds() < self.cache_duration:
                    logger.debug(f"Using cached execution monitoring for {trade_id}")
                    return cached_data['monitoring_data']
            
            # Analyze execution components
            execution_status_analysis = self._analyze_execution_status(trade_data)
            execution_time_analysis = self._analyze_execution_time(trade_data)
            gas_optimization_analysis = self._analyze_gas_optimization(trade_data, market_conditions)
            slippage_analysis = self._analyze_slippage(trade_data)
            retry_analysis = self._analyze_retry_requirements(trade_data, execution_history)
            stuck_transaction_analysis = self._analyze_stuck_transactions(trade_data)
            
            # Calculate execution monitoring score
            monitoring_score = self._calculate_monitoring_score(
                execution_status_analysis, execution_time_analysis, gas_optimization_analysis,
                slippage_analysis, retry_analysis, stuck_transaction_analysis
            )
            
            # Determine monitoring level
            monitoring_level = self._determine_monitoring_level(monitoring_score)
            
            # Generate execution recommendations
            execution_recommendations = self._generate_execution_recommendations(
                monitoring_level, execution_status_analysis, execution_time_analysis,
                gas_optimization_analysis, slippage_analysis, retry_analysis
            )
            
            # Calculate execution urgency
            execution_urgency = self._calculate_execution_urgency(
                monitoring_score, monitoring_level, execution_status_analysis
            )
            
            # Generate execution insights
            execution_insights = self._generate_execution_insights(
                monitoring_level, monitoring_score, execution_urgency
            )
            
            result = {
                'monitoring_score': monitoring_score,
                'monitoring_level': monitoring_level,
                'execution_urgency': execution_urgency,
                'execution_recommendations': execution_recommendations,
                'execution_status_analysis': execution_status_analysis,
                'execution_time_analysis': execution_time_analysis,
                'gas_optimization_analysis': gas_optimization_analysis,
                'slippage_analysis': slippage_analysis,
                'retry_analysis': retry_analysis,
                'stuck_transaction_analysis': stuck_transaction_analysis,
                'execution_insights': execution_insights,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Cache the result
            self.monitoring_cache[cache_key] = {'timestamp': datetime.now(), 'monitoring_data': result}
            
            logger.info(f"🔍 Execution monitoring for {trade_id}: {monitoring_level} (score: {monitoring_score:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Trade execution monitoring failed: {e}")
            return self._get_default_monitoring_analysis(trade_data)

    def _analyze_execution_status(self, trade_data: Dict) -> Dict:
        """Analyze execution status"""
        try:
            status = trade_data.get('status', 'pending')
            success = trade_data.get('success', False)
            
            if status == 'completed' and success:
                status_severity = 'success'
                status_urgency = 'normal'
                recommendation = 'Execution completed successfully'
            elif status == 'failed':
                status_severity = 'critical'
                status_urgency = 'urgent'
                recommendation = 'Execution failed - retry required'
            elif status == 'pending':
                status_severity = 'warning'
                status_urgency = 'warning'
                recommendation = 'Execution pending - monitor closely'
            else:
                status_severity = 'unknown'
                status_urgency = 'warning'
                recommendation = 'Unknown execution status - investigate'
            
            return {
                'status': status,
                'success': success,
                'status_severity': status_severity,
                'status_urgency': status_urgency,
                'recommendation': recommendation
            }
            
        except Exception as e:
            logger.error(f"❌ Execution status analysis failed: {e}")
            return {'status': 'unknown', 'success': False, 'status_severity': 'unknown', 'status_urgency': 'warning', 'recommendation': 'Error analyzing execution status'}

    def _analyze_execution_time(self, trade_data: Dict) -> Dict:
        """Analyze execution time"""
        try:
            start_time = trade_data.get('start_time', datetime.now())
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time)
            
            current_time = datetime.now()
            execution_time = (current_time - start_time).total_seconds()
            
            if execution_time > self.max_execution_time_seconds:
                time_severity = 'critical'
                time_urgency = 'urgent'
                recommendation = f'Execution time exceeded {self.max_execution_time_seconds}s - consider cancellation'
            elif execution_time > self.max_execution_time_seconds * 0.8:
                time_severity = 'warning'
                time_urgency = 'warning'
                recommendation = f'Execution time approaching {self.max_execution_time_seconds}s limit'
            else:
                time_severity = 'normal'
                time_urgency = 'normal'
                recommendation = 'Execution time within acceptable range'
            
            return {
                'execution_time': execution_time,
                'max_execution_time': self.max_execution_time_seconds,
                'time_severity': time_severity,
                'time_urgency': time_urgency,
                'recommendation': recommendation
            }
            
        except Exception as e:
            logger.error(f"❌ Execution time analysis failed: {e}")
            return {'execution_time': 0, 'max_execution_time': self.max_execution_time_seconds, 'time_severity': 'normal', 'time_urgency': 'normal', 'recommendation': 'Error analyzing execution time'}

    def _analyze_gas_optimization(self, trade_data: Dict, market_conditions: Dict) -> Dict:
        """Analyze gas optimization"""
        try:
            if not self.gas_optimization_enabled:
                return {'gas_optimization_enabled': False, 'gas_severity': 'normal', 'gas_urgency': 'normal', 'recommendation': 'Gas optimization disabled'}
            
            gas_price = trade_data.get('gas_price', 0)
            gas_limit = trade_data.get('gas_limit', 0)
            estimated_gas_cost = gas_price * gas_limit
            
            # Calculate gas efficiency
            if estimated_gas_cost > 0:
                gas_efficiency = 1.0 / estimated_gas_cost  # Higher is better
            else:
                gas_efficiency = 0
            
            if gas_efficiency < self.low_gas_threshold:
                gas_severity = 'critical'
                gas_urgency = 'urgent'
                recommendation = 'Gas costs too high - optimize gas settings'
            elif gas_efficiency < self.high_gas_threshold:
                gas_severity = 'warning'
                gas_urgency = 'warning'
                recommendation = 'Gas costs high - consider optimization'
            else:
                gas_severity = 'normal'
                gas_urgency = 'normal'
                recommendation = 'Gas costs within acceptable range'
            
            return {
                'gas_price': gas_price,
                'gas_limit': gas_limit,
                'estimated_gas_cost': estimated_gas_cost,
                'gas_efficiency': gas_efficiency,
                'gas_severity': gas_severity,
                'gas_urgency': gas_urgency,
                'recommendation': recommendation
            }
            
        except Exception as e:
            logger.error(f"❌ Gas optimization analysis failed: {e}")
            return {'gas_price': 0, 'gas_limit': 0, 'estimated_gas_cost': 0, 'gas_efficiency': 0, 'gas_severity': 'normal', 'gas_urgency': 'normal', 'recommendation': 'Error analyzing gas optimization'}

    def _analyze_slippage(self, trade_data: Dict) -> Dict:
        """Analyze slippage"""
        try:
            if not self.slippage_monitoring_enabled:
                return {'slippage_monitoring_enabled': False, 'slippage_severity': 'normal', 'slippage_urgency': 'normal', 'recommendation': 'Slippage monitoring disabled'}
            
            expected_price = trade_data.get('expected_price', 0)
            actual_price = trade_data.get('actual_price', 0)
            
            if expected_price > 0 and actual_price > 0:
                slippage = abs(actual_price - expected_price) / expected_price
            else:
                slippage = 0
            
            if slippage > self.max_slippage_threshold:
                slippage_severity = 'critical'
                slippage_urgency = 'urgent'
                recommendation = f'Slippage exceeded {self.max_slippage_threshold*100:.1f}% - execution may be unfavorable'
            elif slippage > self.high_slippage_threshold:
                slippage_severity = 'warning'
                slippage_urgency = 'warning'
                recommendation = f'Slippage high at {slippage*100:.1f}% - monitor closely'
            else:
                slippage_severity = 'normal'
                slippage_urgency = 'normal'
                recommendation = f'Slippage within acceptable range at {slippage*100:.1f}%'
            
            return {
                'expected_price': expected_price,
                'actual_price': actual_price,
                'slippage': slippage,
                'slippage_severity': slippage_severity,
                'slippage_urgency': slippage_urgency,
                'recommendation': recommendation
            }
            
        except Exception as e:
            logger.error(f"❌ Slippage analysis failed: {e}")
            return {'expected_price': 0, 'actual_price': 0, 'slippage': 0, 'slippage_severity': 'normal', 'slippage_urgency': 'normal', 'recommendation': 'Error analyzing slippage'}

    def _analyze_retry_requirements(self, trade_data: Dict, execution_history: List[Dict]) -> Dict:
        """Analyze retry requirements"""
        try:
            trade_id = trade_data.get('trade_id', 'unknown')
            retry_count = trade_data.get('retry_count', 0)
            
            # Count failed attempts for this trade
            failed_attempts = len([h for h in execution_history if h.get('trade_id') == trade_id and not h.get('success', False)])
            
            if retry_count >= self.max_retry_attempts:
                retry_severity = 'critical'
                retry_urgency = 'urgent'
                recommendation = f'Maximum retry attempts ({self.max_retry_attempts}) exceeded - manual intervention required'
            elif retry_count >= self.max_retry_attempts * 0.8:
                retry_severity = 'warning'
                retry_urgency = 'warning'
                recommendation = f'Approaching maximum retry attempts ({retry_count}/{self.max_retry_attempts})'
            else:
                retry_severity = 'normal'
                retry_urgency = 'normal'
                recommendation = f'Retry attempts within limits ({retry_count}/{self.max_retry_attempts})'
            
            return {
                'retry_count': retry_count,
                'failed_attempts': failed_attempts,
                'max_retry_attempts': self.max_retry_attempts,
                'retry_severity': retry_severity,
                'retry_urgency': retry_urgency,
                'recommendation': recommendation
            }
            
        except Exception as e:
            logger.error(f"❌ Retry requirements analysis failed: {e}")
            return {'retry_count': 0, 'failed_attempts': 0, 'max_retry_attempts': self.max_retry_attempts, 'retry_severity': 'normal', 'retry_urgency': 'normal', 'recommendation': 'Error analyzing retry requirements'}

    def _analyze_stuck_transactions(self, trade_data: Dict) -> Dict:
        """Analyze stuck transactions"""
        try:
            start_time = trade_data.get('start_time', datetime.now())
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time)
            
            current_time = datetime.now()
            transaction_age = (current_time - start_time).total_seconds()
            
            if transaction_age > self.stuck_transaction_threshold:
                stuck_severity = 'critical'
                stuck_urgency = 'urgent'
                recommendation = f'Transaction stuck for {transaction_age/60:.1f} minutes - consider cancellation'
            elif transaction_age > self.stuck_transaction_threshold * 0.8:
                stuck_severity = 'warning'
                stuck_urgency = 'warning'
                recommendation = f'Transaction aging ({transaction_age/60:.1f} minutes) - monitor closely'
            else:
                stuck_severity = 'normal'
                stuck_urgency = 'normal'
                recommendation = f'Transaction age normal ({transaction_age/60:.1f} minutes)'
            
            return {
                'transaction_age': transaction_age,
                'stuck_threshold': self.stuck_transaction_threshold,
                'stuck_severity': stuck_severity,
                'stuck_urgency': stuck_urgency,
                'recommendation': recommendation
            }
            
        except Exception as e:
            logger.error(f"❌ Stuck transaction analysis failed: {e}")
            return {'transaction_age': 0, 'stuck_threshold': self.stuck_transaction_threshold, 'stuck_severity': 'normal', 'stuck_urgency': 'normal', 'recommendation': 'Error analyzing stuck transactions'}

    def _calculate_monitoring_score(self, execution_status_analysis: Dict, execution_time_analysis: Dict,
                                 gas_optimization_analysis: Dict, slippage_analysis: Dict,
                                 retry_analysis: Dict, stuck_transaction_analysis: Dict) -> float:
        """Calculate overall monitoring score"""
        try:
            # Weight factors
            weights = {
                'execution_status': 0.30,
                'execution_time': 0.25,
                'gas_optimization': 0.15,
                'slippage': 0.15,
                'retry_requirements': 0.10,
                'stuck_transactions': 0.05
            }
            
            # Calculate component scores
            status_score = 1.0 if execution_status_analysis['status_severity'] == 'success' else 0.0
            time_score = 1.0 if execution_time_analysis['time_severity'] == 'normal' else 0.0
            gas_score = 1.0 if gas_optimization_analysis['gas_severity'] == 'normal' else 0.0
            slippage_score = 1.0 if slippage_analysis['slippage_severity'] == 'normal' else 0.0
            retry_score = 1.0 if retry_analysis['retry_severity'] == 'normal' else 0.0
            stuck_score = 1.0 if stuck_transaction_analysis['stuck_severity'] == 'normal' else 0.0
            
            # Calculate weighted monitoring score
            monitoring_score = (
                status_score * weights['execution_status'] +
                time_score * weights['execution_time'] +
                gas_score * weights['gas_optimization'] +
                slippage_score * weights['slippage'] +
                retry_score * weights['retry_requirements'] +
                stuck_score * weights['stuck_transactions']
            )
            
            return min(1.0, monitoring_score)
            
        except Exception as e:
            logger.error(f"❌ Monitoring score calculation failed: {e}")
            return 0.0

    def _determine_monitoring_level(self, monitoring_score: float) -> str:
        """Determine monitoring level based on score"""
        try:
            if monitoring_score >= 0.9:
                return 'normal'
            elif monitoring_score >= 0.7:
                return 'warning'
            elif monitoring_score >= 0.5:
                return 'urgent'
            elif monitoring_score >= 0.3:
                return 'critical'
            else:
                return 'emergency'
                
        except Exception as e:
            logger.error(f"❌ Monitoring level determination failed: {e}")
            return 'critical'

    def _generate_execution_recommendations(self, monitoring_level: str, execution_status_analysis: Dict,
                                          execution_time_analysis: Dict, gas_optimization_analysis: Dict,
                                          slippage_analysis: Dict, retry_analysis: Dict) -> List[str]:
        """Generate execution recommendations"""
        try:
            recommendations = []
            
            if monitoring_level == 'emergency':
                recommendations.extend([
                    "🚨 EMERGENCY EXECUTION CONDITIONS",
                    "• Halt all trading immediately",
                    "• Cancel all pending transactions",
                    "• Activate emergency protocols",
                    "• Manual intervention required"
                ])
            elif monitoring_level == 'critical':
                recommendations.extend([
                    "🚨 CRITICAL EXECUTION ISSUES",
                    "• Stop new trade executions",
                    "• Cancel stuck transactions",
                    "• Investigate execution failures",
                    "• Reduce execution frequency"
                ])
            elif monitoring_level == 'urgent':
                recommendations.extend([
                    "⚠️ URGENT EXECUTION ISSUES",
                    "• Monitor executions closely",
                    "• Optimize gas settings",
                    "• Reduce position sizes",
                    "• Check network conditions"
                ])
            elif monitoring_level == 'warning':
                recommendations.extend([
                    "⚠️ WARNING EXECUTION CONDITIONS",
                    "• Increase monitoring frequency",
                    "• Optimize execution parameters",
                    "• Monitor gas costs",
                    "• Check slippage levels"
                ])
            else:
                recommendations.extend([
                    "✅ NORMAL EXECUTION CONDITIONS",
                    "• Continue normal execution",
                    "• Monitor for changes",
                    "• Maintain current settings"
                ])
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Execution recommendations generation failed: {e}")
            return ["Error generating recommendations"]

    def _calculate_execution_urgency(self, monitoring_score: float, monitoring_level: str,
                                   execution_status_analysis: Dict) -> str:
        """Calculate execution urgency"""
        try:
            if monitoring_level in ['emergency', 'critical']:
                return 'emergency'
            elif monitoring_level == 'urgent':
                return 'urgent'
            elif monitoring_level == 'warning':
                return 'warning'
            else:
                return 'normal'
                
        except Exception as e:
            logger.error(f"❌ Execution urgency calculation failed: {e}")
            return 'normal'

    def _generate_execution_insights(self, monitoring_level: str, monitoring_score: float,
                                   execution_urgency: str) -> List[str]:
        """Generate execution insights"""
        try:
            insights = []
            
            insights.append(f"🔍 Monitoring Level: {monitoring_level.upper()}")
            insights.append(f"📊 Monitoring Score: {monitoring_score:.2f}")
            insights.append(f"⚡ Execution Urgency: {execution_urgency.upper()}")
            
            if monitoring_level in ['emergency', 'critical']:
                insights.append("🚨 CRITICAL EXECUTION ISSUES")
                insights.append("• Immediate action required")
                insights.append("• Halt trading operations")
            elif monitoring_level == 'urgent':
                insights.append("⚠️ URGENT EXECUTION ISSUES")
                insights.append("• Monitor closely")
                insights.append("• Optimize settings")
            else:
                insights.append("✅ NORMAL EXECUTION CONDITIONS")
                insights.append("• Continue operations")
                insights.append("• Monitor for changes")
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Execution insights generation failed: {e}")
            return ["Error generating insights"]

    def _get_default_monitoring_analysis(self, trade_data: Dict) -> Dict:
        """Get default monitoring analysis when analysis fails"""
        return {
            'monitoring_score': 0.5,
            'monitoring_level': 'normal',
            'execution_urgency': 'normal',
            'execution_recommendations': ['Continue normal execution'],
            'execution_status_analysis': {'status': 'unknown', 'success': False, 'status_severity': 'normal', 'status_urgency': 'normal', 'recommendation': 'Default analysis'},
            'execution_time_analysis': {'execution_time': 0, 'max_execution_time': self.max_execution_time_seconds, 'time_severity': 'normal', 'time_urgency': 'normal', 'recommendation': 'Default analysis'},
            'gas_optimization_analysis': {'gas_price': 0, 'gas_limit': 0, 'estimated_gas_cost': 0, 'gas_efficiency': 0, 'gas_severity': 'normal', 'gas_urgency': 'normal', 'recommendation': 'Default analysis'},
            'slippage_analysis': {'expected_price': 0, 'actual_price': 0, 'slippage': 0, 'slippage_severity': 'normal', 'slippage_urgency': 'normal', 'recommendation': 'Default analysis'},
            'retry_analysis': {'retry_count': 0, 'failed_attempts': 0, 'max_retry_attempts': self.max_retry_attempts, 'retry_severity': 'normal', 'retry_urgency': 'normal', 'recommendation': 'Default analysis'},
            'stuck_transaction_analysis': {'transaction_age': 0, 'stuck_threshold': self.stuck_transaction_threshold, 'stuck_severity': 'normal', 'stuck_urgency': 'normal', 'recommendation': 'Default analysis'},
            'execution_insights': ['Default execution monitoring analysis'],
            'analysis_timestamp': datetime.now().isoformat()
        }

    def get_monitoring_summary(self, trade_data: Dict, execution_history: List[Dict], 
                             market_conditions: Dict) -> Dict:
        """Get monitoring summary for quick assessment"""
        try:
            monitoring_analysis = self.monitor_trade_execution(trade_data, execution_history, market_conditions)
            
            return {
                'monitoring_level': monitoring_analysis['monitoring_level'],
                'monitoring_score': monitoring_analysis['monitoring_score'],
                'execution_urgency': monitoring_analysis['execution_urgency'],
                'requires_immediate_action': monitoring_analysis['monitoring_level'] in ['emergency', 'critical'],
                'recommendations': monitoring_analysis['execution_recommendations'][:3]  # Top 3 recommendations
            }
            
        except Exception as e:
            logger.error(f"❌ Monitoring summary generation failed: {e}")
            return {
                'monitoring_level': 'normal',
                'monitoring_score': 0.5,
                'execution_urgency': 'normal',
                'requires_immediate_action': False,
                'recommendations': ['Continue normal execution']
            }

# Global instance
ai_trade_execution_monitor = AITradeExecutionMonitor()
