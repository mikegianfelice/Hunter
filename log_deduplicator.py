# log_deduplicator.py
import time
import hashlib
from typing import Dict, Set

# Log deduplication cache
_log_cache: Dict[str, float] = {}
_log_cache_ttl = 300  # 5 minutes

# Rate limiting for frequent log messages
_rate_limits: Dict[str, list] = {}
_rate_limit_window = 60  # 1 minute
_max_logs_per_window = 10  # Max 10 logs per minute per type

def _cleanup_log_cache():
    """Remove old log entries from the cache"""
    current_time = time.time()
    global _log_cache
    _log_cache = {
        log_id: timestamp for log_id, timestamp in _log_cache.items()
        if current_time - timestamp < _log_cache_ttl
    }

def _get_log_fingerprint(message: str, level: str = "INFO") -> str:
    """Create a fingerprint for log deduplication that ignores minor variations"""
    # Normalize message by removing timestamps, addresses, and other variable data
    normalized = f"{level}:{message.lower()}"
    # Remove common variable patterns
    import re
    normalized = re.sub(r'0x[a-fA-F0-9]{8,}', 'ADDRESS', normalized)
    normalized = re.sub(r'\$[\d,]+\.?\d*', 'AMOUNT', normalized)
    normalized = re.sub(r'\d+\.\d+%', 'PERCENT', normalized)
    normalized = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', 'TIMESTAMP', normalized)
    normalized = re.sub(r'\d+\.\d+', 'NUMBER', normalized)
    normalized = re.sub(r'[a-fA-F0-9]{8,}', 'HASH', normalized)  # Remove long hex strings
    return hashlib.md5(normalized.encode()).hexdigest()

def _check_log_rate_limit(log_type: str) -> bool:
    """Check if we're within rate limits for this log type"""
    current_time = time.time()
    global _rate_limits
    
    # Clean old entries
    _rate_limits = {
        log_type: timestamps for log_type, timestamps in _rate_limits.items()
        if any(t > current_time - _rate_limit_window for t in timestamps)
    }
    
    # Check current rate
    if log_type not in _rate_limits:
        _rate_limits[log_type] = []
    
    recent_logs = [t for t in _rate_limits[log_type] if t > current_time - _rate_limit_window]
    
    if len(recent_logs) >= _max_logs_per_window:
        return False
    
    # Add current log
    _rate_limits[log_type].append(current_time)
    return True

def should_log(message: str, level: str = "INFO", log_type: str = "general") -> bool:
    """
    Check if a log message should be logged based on deduplication and rate limiting.
    Returns True if the message should be logged, False if it should be skipped.
    """
    # Check rate limiting first
    if not _check_log_rate_limit(log_type):
        return False
    
    # Check deduplication
    _cleanup_log_cache()
    fingerprint = _get_log_fingerprint(message, level)
    
    if fingerprint in _log_cache:
        return False  # Skip duplicate log
    
    # Add to cache
    _log_cache[fingerprint] = time.time()
    return True

def get_dedup_stats() -> Dict[str, int]:
    """Get statistics about deduplication cache"""
    return {
        "cached_logs": len(_log_cache),
        "rate_limited_types": len(_rate_limits)
    }
