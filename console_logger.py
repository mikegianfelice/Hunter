# console_logger.py
import time
from typing import Dict, Set
from log_deduplicator import should_log

# Console message deduplication
_console_cache: Dict[str, float] = {}
_console_cache_ttl = 60  # 1 minute for console messages

def _cleanup_console_cache():
    """Remove old console messages from the cache"""
    current_time = time.time()
    global _console_cache
    _console_cache = {
        msg_id: timestamp for msg_id, timestamp in _console_cache.items()
        if current_time - timestamp < _console_cache_ttl
    }

def _get_console_fingerprint(message: str) -> str:
    """Create a fingerprint for console message deduplication"""
    import hashlib
    import re
    
    # Normalize message by removing variable data
    normalized = message.lower()
    normalized = re.sub(r'0x[a-fA-F0-9]{8,}', 'ADDRESS', normalized)
    normalized = re.sub(r'\$[\d,]+\.?\d*', 'AMOUNT', normalized)
    normalized = re.sub(r'\d+\.\d+%', 'PERCENT', normalized)
    normalized = re.sub(r'\d+\.\d+', 'NUMBER', normalized)
    normalized = re.sub(r'[a-fA-F0-9]{8,}', 'HASH', normalized)
    return hashlib.md5(normalized.encode()).hexdigest()

def console_print(message: str, log_type: str = "general", force: bool = False):
    """
    Print a message to console with deduplication.
    - message: text to print
    - log_type: category for rate limiting
    - force: if True, bypass deduplication
    """
    if not force:
        # Check deduplication
        _cleanup_console_cache()
        fingerprint = _get_console_fingerprint(message)
        
        if fingerprint in _console_cache:
            return  # Skip duplicate message
        
        # Add to cache
        _console_cache[fingerprint] = time.time()
        
        # Check rate limiting
        if not should_log(message, "INFO", log_type):
            return  # Skip rate-limited message
    
    print(message)

def console_error(message: str, log_type: str = "error", force: bool = False):
    """Print an error message to console with deduplication"""
    console_print(f"❌ {message}", log_type, force)

def console_warning(message: str, log_type: str = "warning", force: bool = False):
    """Print a warning message to console with deduplication"""
    console_print(f"⚠️ {message}", log_type, force)

def console_success(message: str, log_type: str = "success", force: bool = False):
    """Print a success message to console with deduplication"""
    console_print(f"✅ {message}", log_type, force)

def console_info(message: str, log_type: str = "info", force: bool = False):
    """Print an info message to console with deduplication"""
    console_print(f"ℹ️ {message}", log_type, force)
