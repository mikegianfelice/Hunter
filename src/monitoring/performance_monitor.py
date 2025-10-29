#!/usr/bin/env python3
"""
Performance Monitoring System for Trading Bot
Tracks key metrics, performance indicators, and system health
"""

import time
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class TradeMetrics:
    """Individual trade performance metrics"""
    timestamp: str
    symbol: str
    chain: str
    amount_usd: float
    success: bool
    execution_time_ms: float
    profit_loss_usd: float
    quality_score: float
    risk_score: float
    error_message: Optional[str] = None

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: str
    cpu_usage_percent: float
    memory_usage_mb: float
    active_connections: int
    api_response_time_ms: float
    ai_module_health: Dict[str, bool]
    circuit_breaker_states: Dict[str, str]

@dataclass
class TradingSession:
    """Trading session summary"""
    session_id: str
    start_time: str
    end_time: Optional[str]
    total_trades: int
    successful_trades: int
    failed_trades: int
    total_profit_loss: float
    max_drawdown: float
    win_rate: float
    avg_execution_time: float
    best_trade: Optional[TradeMetrics]
    worst_trade: Optional[TradeMetrics]

class PerformanceMonitor:
    """
    Comprehensive performance monitoring system
    """
    
    def __init__(self, max_history_size: int = 10000):
        self.max_history_size = max_history_size
        self.trade_history: deque = deque(maxlen=max_history_size)
        self.system_metrics: deque = deque(maxlen=1000)
        self.current_session: Optional[TradingSession] = None
        self.session_history: List[TradingSession] = []
        
        # Performance counters
        self.counters = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_profit_loss': 0.0,
            'total_execution_time': 0.0,
            'api_calls': 0,
            'ai_module_calls': 0,
            'errors': 0
        }
        
        # Real-time metrics
        self.real_time_metrics = {
            'trades_per_hour': 0,
            'success_rate': 0.0,
            'avg_profit_per_trade': 0.0,
            'current_drawdown': 0.0,
            'system_health_score': 100.0
        }
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Start monitoring thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
    
    def start_trading_session(self, session_id: str = None):
        """Start a new trading session"""
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        
        with self.lock:
            if self.current_session is not None:
                self.end_trading_session()
            
            self.current_session = TradingSession(
                session_id=session_id,
                start_time=datetime.now().isoformat(),
                end_time=None,
                total_trades=0,
                successful_trades=0,
                failed_trades=0,
                total_profit_loss=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                avg_execution_time=0.0,
                best_trade=None,
                worst_trade=None
            )
            
            logger.info(f"Started trading session: {session_id}")
    
    def end_trading_session(self):
        """End current trading session"""
        with self.lock:
            if self.current_session is None:
                return
            
            self.current_session.end_time = datetime.now().isoformat()
            
            # Calculate session metrics
            if self.current_session.total_trades > 0:
                self.current_session.win_rate = (
                    self.current_session.successful_trades / self.current_session.total_trades
                )
                
                # Calculate average execution time
                session_trades = [
                    trade for trade in self.trade_history
                    if trade.timestamp >= self.current_session.start_time
                ]
                if session_trades:
                    self.current_session.avg_execution_time = statistics.mean(
                        [trade.execution_time_ms for trade in session_trades]
                    )
                    
                    # Find best and worst trades
                    profitable_trades = [t for t in session_trades if t.profit_loss_usd > 0]
                    losing_trades = [t for t in session_trades if t.profit_loss_usd < 0]
                    
                    if profitable_trades:
                        self.current_session.best_trade = max(
                            profitable_trades, key=lambda t: t.profit_loss_usd
                        )
                    if losing_trades:
                        self.current_session.worst_trade = min(
                            losing_trades, key=lambda t: t.profit_loss_usd
                        )
            
            self.session_history.append(self.current_session)
            logger.info(f"Ended trading session: {self.current_session.session_id}")
            self.current_session = None
    
    def record_trade(self, trade_metrics: TradeMetrics):
        """Record a trade and update metrics"""
        with self.lock:
            # Add to history
            self.trade_history.append(trade_metrics)
            
            # Update counters
            self.counters['total_trades'] += 1
            self.counters['total_execution_time'] += trade_metrics.execution_time_ms
            self.counters['total_profit_loss'] += trade_metrics.profit_loss_usd
            
            if trade_metrics.success:
                self.counters['successful_trades'] += 1
            else:
                self.counters['failed_trades'] += 1
                self.counters['errors'] += 1
            
            # Update current session
            if self.current_session is not None:
                self.current_session.total_trades += 1
                self.current_session.total_profit_loss += trade_metrics.profit_loss_usd
                
                if trade_metrics.success:
                    self.current_session.successful_trades += 1
                else:
                    self.current_session.failed_trades += 1
                
                # Update max drawdown
                if trade_metrics.profit_loss_usd < 0:
                    self.current_session.max_drawdown = min(
                        self.current_session.max_drawdown,
                        trade_metrics.profit_loss_usd
                    )
            
            # Update real-time metrics
            self._update_real_time_metrics()
    
    def record_system_metrics(self, system_metrics: SystemMetrics):
        """Record system performance metrics"""
        with self.lock:
            self.system_metrics.append(system_metrics)
            self._update_real_time_metrics()
    
    def record_api_call(self, endpoint: str, response_time_ms: float, success: bool):
        """Record API call metrics"""
        with self.lock:
            self.counters['api_calls'] += 1
            if not success:
                self.counters['errors'] += 1
    
    def record_ai_module_call(self, module_name: str, execution_time_ms: float, success: bool):
        """Record AI module call metrics"""
        with self.lock:
            self.counters['ai_module_calls'] += 1
            if not success:
                self.counters['errors'] += 1
    
    def _update_real_time_metrics(self):
        """Update real-time performance metrics"""
        # Calculate trades per hour
        now = time.time()
        one_hour_ago = now - 3600
        recent_trades = [
            trade for trade in self.trade_history
            if time.mktime(datetime.fromisoformat(trade.timestamp).timetuple()) > one_hour_ago
        ]
        self.real_time_metrics['trades_per_hour'] = len(recent_trades)
        
        # Calculate success rate
        if self.counters['total_trades'] > 0:
            self.real_time_metrics['success_rate'] = (
                self.counters['successful_trades'] / self.counters['total_trades']
            )
        
        # Calculate average profit per trade
        if self.counters['total_trades'] > 0:
            self.real_time_metrics['avg_profit_per_trade'] = (
                self.counters['total_profit_loss'] / self.counters['total_trades']
            )
        
        # Calculate current drawdown
        if self.current_session:
            self.real_time_metrics['current_drawdown'] = self.current_session.max_drawdown
        
        # Calculate system health score
        self.real_time_metrics['system_health_score'] = self._calculate_health_score()
    
    def _calculate_health_score(self) -> float:
        """Calculate overall system health score (0-100)"""
        score = 100.0
        
        # Deduct for errors
        if self.counters['total_trades'] > 0:
            error_rate = self.counters['errors'] / self.counters['total_trades']
            score -= error_rate * 50  # Up to 50 points for errors
        
        # Deduct for low success rate
        if self.counters['total_trades'] > 10:
            success_rate = self.counters['successful_trades'] / self.counters['total_trades']
            if success_rate < 0.5:  # Less than 50% success rate
                score -= (0.5 - success_rate) * 40  # Up to 20 points for low success rate
        
        # Deduct for high execution time
        if self.counters['total_trades'] > 0:
            avg_execution_time = self.counters['total_execution_time'] / self.counters['total_trades']
            if avg_execution_time > 10000:  # More than 10 seconds
                score -= min(20, (avg_execution_time - 10000) / 1000)  # Up to 20 points
        
        return max(0.0, min(100.0, score))
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Record system metrics every 30 seconds
                system_metrics = SystemMetrics(
                    timestamp=datetime.now().isoformat(),
                    cpu_usage_percent=self._get_cpu_usage(),
                    memory_usage_mb=self._get_memory_usage(),
                    active_connections=self._get_active_connections(),
                    api_response_time_ms=self._get_avg_api_response_time(),
                    ai_module_health=self._get_ai_module_health(),
                    circuit_breaker_states=self._get_circuit_breaker_states()
                )
                self.record_system_metrics(system_metrics)
                
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        try:
            import psutil
            return psutil.cpu_percent()
        except ImportError:
            return 0.0
        except Exception:
            return 0.0
    
    def _get_memory_usage(self) -> float:
        """Get memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
        except Exception:
            return 0.0
    
    def _get_active_connections(self) -> int:
        """Get number of active network connections"""
        try:
            import psutil
            process = psutil.Process()
            return len(process.connections())
        except ImportError:
            return 0
        except Exception:
            return 0
    
    def _get_avg_api_response_time(self) -> float:
        """Get average API response time"""
        # This would be implemented based on your API client
        return 0.0
    
    def _get_ai_module_health(self) -> Dict[str, bool]:
        """Get AI module health status"""
        try:
            from ai_circuit_breaker import check_ai_module_health
            health = check_ai_module_health()
            return {
                module: state['state'] != 'OPEN'
                for module, state in health['module_states'].items()
            }
        except ImportError:
            return {}
    
    def _get_circuit_breaker_states(self) -> Dict[str, str]:
        """Get circuit breaker states"""
        try:
            from ai_circuit_breaker import circuit_breaker_manager
            return {
                name: breaker.state.value
                for name, breaker in circuit_breaker_manager.breakers.items()
            }
        except ImportError:
            return {}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        with self.lock:
            return {
                'counters': self.counters.copy(),
                'real_time_metrics': self.real_time_metrics.copy(),
                'current_session': asdict(self.current_session) if self.current_session else None,
                'total_sessions': len(self.session_history),
                'health_score': self.real_time_metrics['system_health_score']
            }
    
    def get_trade_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trade history"""
        with self.lock:
            recent_trades = list(self.trade_history)[-limit:]
            return [asdict(trade) for trade in recent_trades]
    
    def get_session_history(self) -> List[Dict[str, Any]]:
        """Get session history"""
        with self.lock:
            return [asdict(session) for session in self.session_history]
    
    def export_metrics(self, file_path: str):
        """Export metrics to JSON file"""
        with self.lock:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'performance_summary': self.get_performance_summary(),
                'trade_history': self.get_trade_history(1000),
                'session_history': self.get_session_history()
            }
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Metrics exported to {file_path}")
    
    def stop_monitoring(self):
        """Stop monitoring and cleanup"""
        self.monitoring_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    return performance_monitor

def record_trade_metrics(symbol: str, chain: str, amount_usd: float, 
                        success: bool, execution_time_ms: float, 
                        profit_loss_usd: float, quality_score: float, 
                        risk_score: float, error_message: str = None):
    """Convenience function to record trade metrics"""
    trade_metrics = TradeMetrics(
        timestamp=datetime.now().isoformat(),
        symbol=symbol,
        chain=chain,
        amount_usd=amount_usd,
        success=success,
        execution_time_ms=execution_time_ms,
        profit_loss_usd=profit_loss_usd,
        quality_score=quality_score,
        risk_score=risk_score,
        error_message=error_message
    )
    performance_monitor.record_trade(trade_metrics)

def get_performance_summary() -> Dict[str, Any]:
    """Get performance summary"""
    return performance_monitor.get_performance_summary()

def start_trading_session(session_id: str = None):
    """Start a new trading session"""
    performance_monitor.start_trading_session(session_id)

def end_trading_session():
    """End current trading session"""
    performance_monitor.end_trading_session()
