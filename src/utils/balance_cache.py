"""
Token balance caching utility to reduce RPC calls
"""
import time
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from src.config.config_loader import get_config_int, get_config_bool

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CACHE_FILE = DATA_DIR / "token_balance_cache.json"

# In-memory cache for faster access
_token_balance_cache: Dict[str, Dict[str, any]] = {}
_cache_loaded = False

def _load_cache() -> Dict[str, Dict[str, any]]:
    """Load token balance cache from disk"""
    global _token_balance_cache, _cache_loaded
    
    if _cache_loaded:
        return _token_balance_cache
    
    if CACHE_FILE.exists():
        try:
            _token_balance_cache = json.loads(CACHE_FILE.read_text(encoding="utf-8") or "{}") or {}
        except Exception:
            _token_balance_cache = {}
    else:
        _token_balance_cache = {}
    
    _cache_loaded = True
    return _token_balance_cache

def _save_cache():
    """Save token balance cache to disk"""
    global _token_balance_cache
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(_token_balance_cache, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"⚠️ Failed to save token balance cache: {e}")

def get_cached_token_balance(token_address: str, chain_id: str, cache_ttl_seconds: int = None) -> Tuple[Optional[float], bool]:
    """
    Get cached token balance if available and fresh
    
    Args:
        token_address: Token contract address
        chain_id: Chain identifier (solana, ethereum, base)
        cache_ttl_seconds: Cache TTL in seconds (defaults to config value)
    
    Returns:
        tuple: (balance: float or None, is_valid: bool)
    """
    if cache_ttl_seconds is None:
        cache_ttl_seconds = get_config_int("balance_cache_ttl_seconds", 300)
    
    cache = _load_cache()
    cache_key = f"{chain_id.lower()}:{token_address.lower()}"
    
    if cache_key not in cache:
        return None, False
    
    cached_entry = cache[cache_key]
    cached_balance = cached_entry.get("balance")
    cached_timestamp = cached_entry.get("timestamp", 0)
    
    if cached_balance is None:
        return None, False
    
    # Check if cache is still valid (within TTL)
    age_seconds = time.time() - cached_timestamp
    if age_seconds > cache_ttl_seconds:
        return cached_balance, False  # Cache exists but is stale
    
    return cached_balance, True  # Cache is valid

def update_token_balance_cache(token_address: str, chain_id: str, balance: float):
    """Update token balance cache with new successful balance check"""
    global _token_balance_cache
    cache = _load_cache()
    cache_key = f"{chain_id.lower()}:{token_address.lower()}"
    
    cache[cache_key] = {
        "balance": float(balance),
        "timestamp": time.time()
    }
    
    _token_balance_cache = cache
    _save_cache()

def invalidate_token_balance_cache(token_address: str, chain_id: str = None):
    """Invalidate cache for a specific token (useful after trades)"""
    global _token_balance_cache
    cache = _load_cache()
    
    if chain_id:
        cache_key = f"{chain_id.lower()}:{token_address.lower()}"
        cache.pop(cache_key, None)
    else:
        # Invalidate for all chains
        keys_to_remove = [k for k in cache.keys() if k.endswith(f":{token_address.lower()}")]
        for key in keys_to_remove:
            cache.pop(key, None)
    
    _token_balance_cache = cache
    _save_cache()
