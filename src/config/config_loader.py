# config_loader.py
import yaml
import os
from typing import Any, Dict

# Sentinel object to distinguish between "key not found" and "key exists but is None"
_NOT_FOUND = object()

class ConfigLoader:
    """Dynamic configuration loader that can reload config when needed"""
    
    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = config_file
        self._config = None
        self._last_modified = 0
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                current_modified = os.path.getmtime(self.config_file)
                if current_modified > self._last_modified or self._config is None:
                    with open(self.config_file, "r") as f:
                        self._config = yaml.safe_load(f) or {}
                    self._last_modified = current_modified
                    print(f"🔄 Configuration reloaded from {self.config_file}")
        except Exception as e:
            print(f"⚠️ Error loading config: {e}")
            if self._config is None:
                self._config = {}
    
    def _get_nested(self, config: Dict, key: str, default: Any = None) -> Any:
        """Get a nested configuration value using dot notation (e.g., 'time_window_scheduler.pause_on_volatility_spike')
        
        Returns a special sentinel object if the key path doesn't exist, so we can distinguish
        between 'key not found' (return default) and 'key exists but value is None' (return None).
        """
        keys = key.split('.')
        value = config
        for i, k in enumerate(keys):
            if isinstance(value, dict):
                if k not in value:
                    # Key path doesn't exist - return sentinel
                    return _NOT_FOUND
                value = value[k]
            else:
                # Intermediate value is not a dict - path doesn't exist
                return _NOT_FOUND
        return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value, reloading config if needed. Supports dot notation for nested keys."""
        self._load_config()
        # Try dot notation first (for nested keys)
        if '.' in key:
            result = self._get_nested(self._config, key, default)
            if result is not _NOT_FOUND:
                # Key path exists (even if value is None)
                return result if result is not None else default
            # Key path doesn't exist, fall through to direct key access for backward compatibility
        # Fall back to direct key access (for backward compatibility)
        return self._config.get(key, default)
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get a boolean configuration value"""
        return bool(self.get(key, default))
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get an integer configuration value"""
        return int(self.get(key, default))
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get a float configuration value"""
        return float(self.get(key, default))
    
    def get_list(self, key: str, default: list = None) -> list:
        """Get a list configuration value"""
        if default is None:
            default = []
        return self.get(key, default)
    
    def reload(self):
        """Force reload of configuration"""
        self._last_modified = 0
        self._load_config()

# Global config loader instance
config_loader = ConfigLoader()

# Convenience functions for backward compatibility
def get_config(key: str, default: Any = None) -> Any:
    return config_loader.get(key, default)

def get_config_bool(key: str, default: bool = False) -> bool:
    return config_loader.get_bool(key, default)

def get_config_int(key: str, default: int = 0) -> int:
    return config_loader.get_int(key, default)

def get_config_float(key: str, default: float = 0.0) -> float:
    return config_loader.get_float(key, default)

def get_config_list(key: str, default: list = None) -> list:
    return config_loader.get_list(key, default)

def reload_config():
    """Force reload of configuration"""
    config_loader.reload()

def get_config_values():
    """Get current configuration values dynamically"""
    return {
        'PRICE_MEM_TTL_SECS': get_config_int("price_memory_ttl_minutes", 15) * 60,
        'PRICE_MEM_PRUNE_SECS': get_config_int("price_memory_prune_hours", 24) * 3600,
        'BASE_TP': get_config_float("take_profit", 0.5),
        'TP_MIN': get_config_float("tp_min", 0.20),
        'TP_MAX': get_config_float("tp_max", 1.00),
        'MIN_MOMENTUM_PCT': get_config_float("min_momentum_pct", 0.003),
        'MIN_VOL_24H_BUY': get_config_float("min_volume_24h_for_buy", 50000),
        'MIN_LIQ_USD_BUY': get_config_float("min_liquidity_usd_for_buy", 50000),
        'MIN_PRICE_USD': get_config_float("min_price_usd", 0.0000001),
        'FASTPATH_VOL': get_config_float("fastpath_min_volume_24h", 100000),
        'FASTPATH_LIQ': get_config_float("fastpath_min_liquidity_usd", 100000),
        'FASTPATH_SENT': get_config_int("fastpath_min_sent_score", 30),
        'ENABLE_PRE_BUY_DELISTING_CHECK': get_config_bool("enable_pre_buy_delisting_check", False),
        'PRE_BUY_CHECK_SENSITIVITY': get_config("pre_buy_check_sensitivity", "lenient"),
        'PRE_BUY_CHECK_TIMEOUT': get_config_int("pre_buy_check_timeout", 10),
        'ENABLE_EXTERNAL_MOMENTUM': get_config_bool("enable_external_momentum", True),
        'EXTERNAL_MOMENTUM_PRIMARY_TIMEFRAME': get_config("external_momentum_primary_timeframe", "h1"),
        'EXTERNAL_MOMENTUM_FALLBACK_TIMEFRAME': get_config("external_momentum_fallback_timeframe", "m5"),
        'EXTERNAL_MOMENTUM_MIN_TIMEFRAME_WEIGHT': get_config_float("external_momentum_min_timeframe_weight", 0.3),
        'USE_MULTI_TIMEFRAME_MOMENTUM': get_config_bool("use_multi_timeframe_momentum", True),
        'EXTERNAL_MOMENTUM_M5_WEIGHT': get_config_float("external_momentum_m5_weight", 0.3),
        'EXTERNAL_MOMENTUM_H1_WEIGHT': get_config_float("external_momentum_h1_weight", 0.5),
        'EXTERNAL_MOMENTUM_H24_WEIGHT': get_config_float("external_momentum_h24_weight", 0.2),
        'REQUIRE_MOMENTUM_ALIGNMENT': get_config_bool("require_momentum_alignment", True),
        'REQUIRE_POSITIVE_24H_MOMENTUM': get_config_bool("require_positive_24h_momentum", True),
        'MIN_MOMENTUM_ACCELERATION': get_config_float("min_momentum_acceleration", 0.002),
        'MIN_MOMENTUM_5M_VELOCITY': get_config_float("min_momentum_5m_velocity", 0.025),  # NEW: 5m velocity check
        'ENABLE_VOLUME_MOMENTUM_CHECK': get_config_bool("enable_volume_momentum_check", True),
        'MIN_VOLUME_CHANGE_1H': get_config_float("min_volume_change_1h", 0.1),
        'ENABLE_RSI_FILTER': get_config_bool("enable_rsi_filter", True),
        'RSI_OVERBOUGHT_THRESHOLD': get_config_float("rsi_overbought_threshold", 70)
    }
