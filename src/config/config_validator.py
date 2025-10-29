#!/usr/bin/env python3
"""
Configuration Validation System for Trading Bot
Provides schema validation and type checking for all configuration values
"""

import yaml
import os
from typing import Dict, Any, List, Optional, Union, Literal
from pydantic import BaseModel, Field, model_validator
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TradingConfig(BaseModel):
    """Core trading configuration validation"""
    trade_amount_usd: float = Field(ge=1.0, le=1000.0, description="Trade amount in USD")
    max_concurrent_positions: int = Field(ge=1, le=20, description="Maximum concurrent positions")
    daily_loss_limit_usd: float = Field(ge=10.0, le=10000.0, description="Daily loss limit in USD")
    per_trade_max_usd: float = Field(ge=1.0, le=1000.0, description="Maximum per trade amount")
    min_wallet_balance_buffer: float = Field(ge=0.01, le=0.5, description="Wallet balance buffer")
    
    # Take profit and stop loss
    take_profit: float = Field(ge=0.05, le=2.0, description="Take profit percentage")
    stop_loss: float = Field(ge=0.05, le=1.0, description="Stop loss percentage")
    trailing_stop_percent: float = Field(ge=0.01, le=0.5, description="Trailing stop percentage")
    
    # Risk management
    max_losing_streak: int = Field(ge=1, le=10, description="Maximum losing streak")
    circuit_breaker_minutes: int = Field(ge=5, le=1440, description="Circuit breaker duration")
    
    @model_validator(mode='after')
    def stop_loss_less_than_take_profit(self):
        if self.stop_loss >= self.take_profit:
            raise ValueError('Stop loss must be less than take profit')
        return self

class ChainConfig(BaseModel):
    """Blockchain configuration validation"""
    supported_chains: List[Literal["ethereum", "solana", "base"]] = Field(
        min_items=1, 
        description="Supported blockchain networks"
    )
    
    # Gas settings
    gas_blocks: int = Field(ge=1, le=10, description="Gas blocks")
    gas_reward_percentile: int = Field(ge=1, le=100, description="Gas reward percentile")
    gas_basefee_headroom: float = Field(ge=0.0, le=1.0, description="Gas basefee headroom")
    gas_priority_min_gwei: int = Field(ge=1, le=1000, description="Minimum gas priority")
    gas_priority_max_gwei: int = Field(ge=1, le=10000, description="Maximum gas priority")
    gas_ceiling_gwei: int = Field(ge=100, le=100000, description="Gas ceiling")
    gas_multiplier: float = Field(ge=0.1, le=10.0, description="Gas multiplier")
    
    @model_validator(mode='after')
    def max_gas_greater_than_min(self):
        if self.gas_priority_max_gwei <= self.gas_priority_min_gwei:
            raise ValueError('Max gas priority must be greater than min gas priority')
        return self

class AIConfig(BaseModel):
    """AI module configuration validation"""
    enable_ai_sentiment_analysis: bool = Field(default=False, description="Enable AI sentiment analysis")
    enable_ai_price_prediction: bool = Field(default=True, description="Enable AI price prediction")
    enable_ai_risk_assessment: bool = Field(default=True, description="Enable AI risk assessment")
    enable_ai_execution_optimization: bool = Field(default=True, description="Enable AI execution optimization")
    enable_ai_microstructure_analysis: bool = Field(default=True, description="Enable AI microstructure analysis")
    enable_ai_portfolio_optimization: bool = Field(default=True, description="Enable AI portfolio optimization")
    
    # Cache durations (in seconds)
    sentiment_cache_duration: int = Field(ge=60, le=3600, description="Sentiment cache duration")
    prediction_cache_duration: int = Field(ge=60, le=3600, description="Prediction cache duration")
    risk_cache_duration: int = Field(ge=60, le=3600, description="Risk cache duration")
    
    # API timeouts (in seconds)
    api_timeout_seconds: int = Field(ge=5, le=60, description="API timeout")
    api_retry_attempts: int = Field(ge=1, le=10, description="API retry attempts")

class QualityConfig(BaseModel):
    """Token quality scoring configuration validation"""
    min_quality_score: int = Field(ge=0, le=100, description="Minimum quality score")
    min_volume_24h_for_buy: float = Field(ge=1000, le=1000000, description="Minimum 24h volume")
    min_liquidity_usd_for_buy: float = Field(ge=1000, le=1000000, description="Minimum liquidity")
    min_price_usd: float = Field(ge=0.000001, le=1.0, description="Minimum token price")
    
    # Quality scoring weights (must sum to 1.0)
    quality_volume_weight: float = Field(ge=0.0, le=1.0, description="Volume weight in quality score")
    quality_liquidity_weight: float = Field(ge=0.0, le=1.0, description="Liquidity weight in quality score")
    quality_price_weight: float = Field(ge=0.0, le=1.0, description="Price weight in quality score")
    
    @model_validator(mode='after')
    def weights_sum_to_one(self):
        total = self.quality_volume_weight + self.quality_liquidity_weight + self.quality_price_weight
        if abs(total - 1.0) > 0.01:  # Allow small floating point errors
            raise ValueError('Quality scoring weights must sum to 1.0')
        return self

class MonitoringConfig(BaseModel):
    """Monitoring and logging configuration validation"""
    enable_performance_tracking: bool = Field(default=True, description="Enable performance tracking")
    performance_data_file: str = Field(default="performance_data.json", description="Performance data file")
    performance_report_days: int = Field(ge=1, le=365, description="Performance report days")
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    enable_structured_logging: bool = Field(default=True, description="Enable structured logging")
    
    # Alerts
    enable_telegram_alerts: bool = Field(default=True, description="Enable Telegram alerts")
    enable_performance_alerts: bool = Field(default=True, description="Enable performance alerts")

class BotConfig(BaseModel):
    """Main bot configuration validation"""
    test_mode: bool = Field(default=False, description="Test mode flag")
    async_trading_enabled: bool = Field(default=False, description="Enable async trading mode")
    
    # Sub-configurations
    trading: TradingConfig = Field(default_factory=TradingConfig)
    chains: ChainConfig = Field(default_factory=ChainConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    # Additional validation
    @model_validator(mode='after')
    def validate_test_mode(self):
        if self.test_mode and self.trading.trade_amount_usd > 10:
            logger.warning("Test mode: Reducing trade amount to $10 for safety")
            self.trading.trade_amount_usd = 10.0
        return self

class ConfigValidator:
    """
    Configuration validation and management system
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.validated_config: Optional[BotConfig] = None
        self.raw_config: Optional[Dict[str, Any]] = None
    
    def load_and_validate(self) -> BotConfig:
        """
        Load configuration from file and validate against schema
        """
        try:
            # Load raw YAML
            with open(self.config_path, 'r') as f:
                self.raw_config = yaml.safe_load(f)
            
            # Validate against schema
            self.validated_config = BotConfig(**self.raw_config)
            
            logger.info("Configuration validation successful")
            return self.validated_config
            
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in configuration file: {e}")
            raise
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def get_config(self) -> BotConfig:
        """
        Get validated configuration
        """
        if self.validated_config is None:
            return self.load_and_validate()
        return self.validated_config
    
    def validate_specific_section(self, section: str) -> Dict[str, Any]:
        """
        Validate specific configuration section
        """
        if self.validated_config is None:
            self.load_and_validate()
        
        section_config = getattr(self.validated_config, section, None)
        if section_config is None:
            raise ValueError(f"Configuration section '{section}' not found")
        
        return section_config.dict()
    
    def get_trading_config(self) -> TradingConfig:
        """Get trading configuration"""
        return self.get_config().trading
    
    def get_ai_config(self) -> AIConfig:
        """Get AI configuration"""
        return self.get_config().ai
    
    def get_chain_config(self) -> ChainConfig:
        """Get chain configuration"""
        return self.get_config().chains
    
    def get_quality_config(self) -> QualityConfig:
        """Get quality configuration"""
        return self.get_config().quality
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration"""
        return self.get_config().monitoring
    
    def validate_environment(self) -> Dict[str, Any]:
        """
        Validate environment variables and dependencies
        """
        validation_results = {
            'environment_variables': {},
            'dependencies': {},
            'overall_valid': True
        }
        
        # Check required environment variables
        required_env_vars = [
            'INFURA_URL',
            'WALLET_ADDRESS', 
            'PRIVATE_KEY',
            'SOLANA_PRIVATE_KEY',
            'SOLANA_RPC_URL'
        ]
        
        for var in required_env_vars:
            value = os.getenv(var)
            validation_results['environment_variables'][var] = {
                'present': value is not None,
                'value': '***' if value else None
            }
            if not value:
                validation_results['overall_valid'] = False
        
        # Check Python dependencies
        required_packages = [
            'web3',
            'solana',
            'requests',
            'pandas',
            'yaml'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                validation_results['dependencies'][package] = {'installed': True}
            except ImportError:
                validation_results['dependencies'][package] = {'installed': False}
                validation_results['overall_valid'] = False
        
        return validation_results
    
    def generate_config_template(self, output_path: str = "config_template.yaml"):
        """
        Generate a configuration template with default values
        """
        template_config = BotConfig()
        
        with open(output_path, 'w') as f:
            yaml.dump(template_config.dict(), f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration template generated: {output_path}")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current configuration
        """
        if self.validated_config is None:
            self.load_and_validate()
        
        config = self.validated_config
        
        return {
            'test_mode': config.test_mode,
            'trading': {
                'trade_amount_usd': config.trading.trade_amount_usd,
                'max_positions': config.trading.max_concurrent_positions,
                'daily_loss_limit': config.trading.daily_loss_limit_usd,
                'take_profit': config.trading.take_profit,
                'stop_loss': config.trading.stop_loss
            },
            'chains': {
                'supported': config.chains.supported_chains,
                'gas_multiplier': config.chains.gas_multiplier
            },
            'ai': {
                'sentiment_enabled': config.ai.enable_ai_sentiment_analysis,
                'prediction_enabled': config.ai.enable_ai_price_prediction,
                'risk_enabled': config.ai.enable_ai_risk_assessment
            },
            'quality': {
                'min_score': config.quality.min_quality_score,
                'min_volume': config.quality.min_volume_24h_for_buy,
                'min_liquidity': config.quality.min_liquidity_usd_for_buy
            }
        }

# Global configuration validator instance
config_validator = ConfigValidator()

def get_validated_config() -> BotConfig:
    """Get validated configuration instance"""
    return config_validator.get_config()

def validate_config() -> bool:
    """Validate configuration and return success status"""
    try:
        config_validator.load_and_validate()
        return True
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False

def get_config_summary() -> Dict[str, Any]:
    """Get configuration summary"""
    return config_validator.get_config_summary()
