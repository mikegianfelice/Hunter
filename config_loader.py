# config_loader.py
import yaml
import os
from typing import Any, Dict

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
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value, reloading config if needed"""
        self._load_config()
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
        'PRE_BUY_CHECK_TIMEOUT': get_config_int("pre_buy_check_timeout", 10)
    }
