#!/usr/bin/env python3
"""
A/B Testing Framework for Feature Weights
Tests different weight combinations and tracks performance
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import time

logger = logging.getLogger(__name__)

@dataclass
class WeightConfig:
    """Configuration for feature weights"""
    name: str
    weights: Dict[str, float]
    created_at: str
    active: bool = True
    trades_count: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0

class ABTestingFramework:
    """A/B testing framework for weight combinations"""
    
    def __init__(self):
        self.config_file = Path("data/ab_test_configs.json")
        self.results_file = Path("data/ab_test_results.json")
        self.configs: Dict[str, WeightConfig] = {}
        self.results: List[Dict] = []
        self.current_variant: Optional[str] = None
        self.load_configs()
    
    def load_configs(self):
        """Load A/B test configurations"""
        if self.config_file.exists():
            try:
                data = json.loads(self.config_file.read_text())
                self.configs = {
                    name: WeightConfig(**config)
                    for name, config in data.items()
                }
            except Exception as e:
                logger.error(f"Error loading A/B test configs: {e}")
                self.configs = {}
        
        # Initialize default configs if none exist
        if not self.configs:
            self._initialize_default_configs()
    
    def _initialize_default_configs(self):
        """Initialize default weight configurations"""
        # Baseline (current weights)
        baseline = WeightConfig(
            name="baseline",
            weights={
                'price_momentum': 0.25,
                'volume_trend': 0.20,
                'liquidity_stability': 0.15,
                'sentiment_score': 0.15,
                'market_regime': 0.10,
                'technical_indicators': 0.10,
                'volatility_pattern': 0.05
            },
            created_at=datetime.now().isoformat()
        )
        
        # Variant A: More weight on technical indicators
        variant_a = WeightConfig(
            name="variant_technical",
            weights={
                'price_momentum': 0.20,
                'volume_trend': 0.15,
                'liquidity_stability': 0.15,
                'sentiment_score': 0.10,
                'market_regime': 0.10,
                'technical_indicators': 0.25,  # Increased
                'volatility_pattern': 0.05
            },
            created_at=datetime.now().isoformat()
        )
        
        # Variant B: More weight on sentiment
        variant_b = WeightConfig(
            name="variant_sentiment",
            weights={
                'price_momentum': 0.20,
                'volume_trend': 0.15,
                'liquidity_stability': 0.15,
                'sentiment_score': 0.25,  # Increased
                'market_regime': 0.10,
                'technical_indicators': 0.10,
                'volatility_pattern': 0.05
            },
            created_at=datetime.now().isoformat()
        )
        
        # Variant C: Balanced with new features
        variant_c = WeightConfig(
            name="variant_balanced",
            weights={
                'price_momentum': 0.15,
                'volume_trend': 0.15,
                'liquidity_stability': 0.10,
                'sentiment_score': 0.15,
                'market_regime': 0.10,
                'technical_indicators': 0.15,
                'volatility_pattern': 0.05,
                'rsi': 0.05,  # New
                'macd_signal': 0.05,  # New
                'volume_profile': 0.05  # New
            },
            created_at=datetime.now().isoformat()
        )
        
        self.configs = {
            'baseline': baseline,
            'variant_technical': variant_a,
            'variant_sentiment': variant_b,
            'variant_balanced': variant_c
        }
        self.save_configs()
    
    def save_configs(self):
        """Save configurations to file"""
        try:
            data = {
                name: asdict(config)
                for name, config in self.configs.items()
            }
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            self.config_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Error saving A/B test configs: {e}")
    
    def get_active_config(self) -> WeightConfig:
        """Get the currently active weight configuration"""
        active_configs = [c for c in self.configs.values() if c.active]
        
        if not active_configs:
            return self.configs.get('baseline', list(self.configs.values())[0])
        
        # Simple round-robin selection
        if not self.current_variant or self.current_variant not in [c.name for c in active_configs]:
            self.current_variant = active_configs[0].name
        
        return self.configs[self.current_variant]
    
    def record_trade_result(self, config_name: str, symbol: str, 
                           pnl: float, success: bool):
        """Record trade result for A/B testing"""
        if config_name not in self.configs:
            return
        
        config = self.configs[config_name]
        config.trades_count += 1
        config.total_pnl += pnl
        
        if success:
            config.wins += 1
        else:
            config.losses += 1
        
        # Save result
        result = {
            'config_name': config_name,
            'symbol': symbol,
            'pnl': pnl,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'win_rate': config.wins / config.trades_count if config.trades_count > 0 else 0.0,
            'total_trades': config.trades_count
        }
        
        self.results.append(result)
        self._save_results()
        self.save_configs()
    
    def _save_results(self):
        """Save test results"""
        try:
            if len(self.results) > 1000:
                self.results = self.results[-1000:]
            
            self.results_file.parent.mkdir(parents=True, exist_ok=True)
            self.results_file.write_text(json.dumps(self.results, indent=2))
        except Exception as e:
            logger.error(f"Error saving A/B test results: {e}")
    
    def get_best_config(self) -> Optional[WeightConfig]:
        """Get the best performing configuration"""
        active_configs = [c for c in self.configs.values() if c.active and c.trades_count >= 10]
        
        if not active_configs:
            return None
        
        scored_configs = []
        for config in active_configs:
            win_rate = config.wins / config.trades_count if config.trades_count > 0 else 0.0
            avg_pnl = config.total_pnl / config.trades_count if config.trades_count > 0 else 0.0
            
            score = (win_rate * 0.6) + (min(1.0, max(0.0, (avg_pnl + 10) / 20)) * 0.4)
            scored_configs.append((score, config))
        
        if not scored_configs:
            return None
        
        scored_configs.sort(reverse=True, key=lambda x: x[0])
        return scored_configs[0][1]
    
    def get_performance_report(self) -> Dict:
        """Get performance report for all configurations"""
        report = {
            'total_configs': len(self.configs),
            'active_configs': len([c for c in self.configs.values() if c.active]),
            'configs': []
        }
        
        for name, config in self.configs.items():
            win_rate = config.wins / config.trades_count if config.trades_count > 0 else 0.0
            avg_pnl = config.total_pnl / config.trades_count if config.trades_count > 0 else 0.0
            
            report['configs'].append({
                'name': name,
                'active': config.active,
                'trades': config.trades_count,
                'wins': config.wins,
                'losses': config.losses,
                'win_rate': win_rate,
                'total_pnl': config.total_pnl,
                'avg_pnl': avg_pnl,
                'weights': config.weights
            })
        
        return report


# Global instance
ab_testing = ABTestingFramework()

