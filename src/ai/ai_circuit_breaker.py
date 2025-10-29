#!/usr/bin/env python3
"""
AI Circuit Breaker System for Trading Bot
Provides fault tolerance and graceful degradation for AI modules
"""

import time
import logging
from typing import Dict, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Circuit is open, calls fail fast
    HALF_OPEN = "HALF_OPEN"  # Testing if service is back

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 3
    recovery_timeout: int = 300  # 5 minutes
    expected_exception: type = Exception
    success_threshold: int = 2  # For half-open state

class AICircuitBreaker:
    """
    Circuit breaker pattern implementation for AI modules
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
            else:
                logger.warning(f"Circuit breaker {self.name} is OPEN - failing fast")
                return self._get_fallback_response()
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.config.expected_exception as e:
            self._on_failure(e)
            logger.warning(f"Circuit breaker {self.name} caught exception: {e}")
            return self._get_fallback_response()
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful call"""
        self.last_success_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info(f"Circuit breaker {self.name} reset to CLOSED")
        else:
            self.failure_count = 0
    
    def _on_failure(self, exception: Exception):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(f"Circuit breaker {self.name} opened due to {self.failure_count} failures")
    
    def _get_fallback_response(self) -> Dict[str, Any]:
        """Get fallback response when circuit is open"""
        return {
            'success': False,
            'error': f'Circuit breaker {self.name} is open',
            'fallback': True,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time,
            'last_success_time': self.last_success_time
        }

class AICircuitBreakerManager:
    """
    Manages multiple circuit breakers for different AI modules
    """
    
    def __init__(self):
        self.breakers: Dict[str, AICircuitBreaker] = {}
        self._setup_default_breakers()
    
    def _setup_default_breakers(self):
        """Setup circuit breakers for all AI modules"""
        ai_modules = [
            'sentiment_analyzer',
            'risk_assessor', 
            'price_predictor',
            'market_regime_detector',
            'execution_optimizer',
            'microstructure_analyzer',
            'portfolio_optimizer',
            'pattern_recognizer',
            'market_intelligence',
            'predictive_analytics',
            'dynamic_strategy_selector',
            'risk_prediction_prevention',
            'market_regime_transition_detector',
            'liquidity_flow_analyzer',
            'multi_timeframe_analysis',
            'market_cycle_predictor',
            'drawdown_protection',
            'performance_attribution',
            'market_anomaly_detector',
            'portfolio_rebalancing',
            'emergency_stop',
            'position_size_validator',
            'trade_execution_monitor',
            'market_condition_guardian'
        ]
        
        for module in ai_modules:
            self.breakers[module] = AICircuitBreaker(
                name=module,
                config=CircuitBreakerConfig(
                    failure_threshold=3,
                    recovery_timeout=300,
                    success_threshold=2
                )
            )
    
    def get_breaker(self, module_name: str) -> AICircuitBreaker:
        """Get circuit breaker for specific module"""
        if module_name not in self.breakers:
            self.breakers[module_name] = AICircuitBreaker(module_name)
        return self.breakers[module_name]
    
    def call_with_breaker(self, module_name: str, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        breaker = self.get_breaker(module_name)
        return breaker.call(func, *args, **kwargs)
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get states of all circuit breakers"""
        return {name: breaker.get_state() for name, breaker in self.breakers.items()}
    
    def reset_breaker(self, module_name: str):
        """Reset specific circuit breaker"""
        if module_name in self.breakers:
            self.breakers[module_name].state = CircuitState.CLOSED
            self.breakers[module_name].failure_count = 0
            self.breakers[module_name].success_count = 0
            logger.info(f"Reset circuit breaker for {module_name}")
    
    def reset_all_breakers(self):
        """Reset all circuit breakers"""
        for breaker in self.breakers.values():
            breaker.state = CircuitState.CLOSED
            breaker.failure_count = 0
            breaker.success_count = 0
        logger.info("Reset all circuit breakers")

# Global circuit breaker manager instance
circuit_breaker_manager = AICircuitBreakerManager()

def with_circuit_breaker(module_name: str):
    """
    Decorator to add circuit breaker protection to AI module methods
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            return circuit_breaker_manager.call_with_breaker(module_name, func, *args, **kwargs)
        return wrapper
    return decorator

def check_ai_module_health() -> Dict[str, Any]:
    """
    Check health of all AI modules using circuit breakers
    """
    health_status = {
        'overall_healthy': True,
        'unhealthy_modules': [],
        'module_states': {}
    }
    
    for module_name, breaker in circuit_breaker_manager.breakers.items():
        state = breaker.get_state()
        health_status['module_states'][module_name] = state
        
        if state['state'] == CircuitState.OPEN.value:
            health_status['overall_healthy'] = False
            health_status['unhealthy_modules'].append(module_name)
    
    return health_status

def get_ai_module_status() -> str:
    """
    Get human-readable status of AI modules
    """
    health = check_ai_module_health()
    
    if health['overall_healthy']:
        return "✅ All AI modules healthy"
    else:
        unhealthy = health['unhealthy_modules']
        return f"⚠️ {len(unhealthy)} AI modules unhealthy: {', '.join(unhealthy)}"
