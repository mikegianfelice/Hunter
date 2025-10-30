# advanced_cache.py
"""
Advanced caching utilities for the trading bot
"""

import time
import json
import hashlib
from typing import Any, Dict, Optional, Union
from functools import wraps

# Simple in-memory cache
_cache: Dict[str, Dict[str, Any]] = {}

def get_cache() -> Dict[str, Dict[str, Any]]:
    """Get the cache instance"""
    return _cache

def cache_get(key: str, default: Any = None) -> Any:
    """Get value from cache"""
    if key in _cache:
        entry = _cache[key]
        if time.time() - entry['timestamp'] < entry['ttl']:
            return entry['value']
        else:
            del _cache[key]
    return default

def cache_set(key: str, value: Any, ttl: int = 300) -> None:
    """Set value in cache with TTL"""
    _cache[key] = {
        'value': value,
        'timestamp': time.time(),
        'ttl': ttl
    }

def cache_clear() -> None:
    """Clear all cache entries"""
    _cache.clear()

def cache_key(*args, **kwargs) -> str:
    """Generate cache key from arguments"""
    key_data = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(key_data.encode()).hexdigest()

def cached(ttl: int = 300):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{cache_key(*args, **kwargs)}"
            result = cache_get(key)
            if result is not None:
                return result
            
            result = func(*args, **kwargs)
            cache_set(key, result, ttl)
            return result
        return wrapper
    return decorator
