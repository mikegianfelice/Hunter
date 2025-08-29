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
