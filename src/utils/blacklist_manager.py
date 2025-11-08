# blacklist_manager.py
import json
import os
import time
from typing import Dict, Set, Tuple
from src.storage.delist import load_delisted_state

BLACKLIST_FILE = "data/blacklist.json"
FAILURE_LOG_FILE = "data/blacklist_failures.json"

# Configuration
MAX_FAILURES_BEFORE_BLACKLIST = 3  # Number of failures before blacklisting
FAILURE_RESET_HOURS = 24  # Hours after which failures are reset
BLACKLIST_REVIEW_HOURS = 168  # 7 days - review blacklisted tokens

def _load() -> Set[str]:
    if not os.path.exists(BLACKLIST_FILE):
        return set()
    try:
        with open(BLACKLIST_FILE, "r") as f:
            data = json.load(f) or []
            return {a.lower() for a in data if isinstance(a, str)}
    except Exception as e:
        print(f"⚠️ Error loading blacklist: {e}")
        return set()

def _save(s: Set[str]):
    try:
        os.makedirs('data', exist_ok=True)
        with open(BLACKLIST_FILE, "w") as f:
            json.dump(sorted(list(s)), f, indent=2)
    except Exception as e:
        print(f"⚠️ Error saving blacklist: {e}")

def _load_failures() -> Dict[str, Dict]:
    """Load failure tracking data"""
    if not os.path.exists(FAILURE_LOG_FILE):
        return {}
    try:
        with open(FAILURE_LOG_FILE, "r") as f:
            return json.load(f) or {}
    except Exception as e:
        print(f"⚠️ Error loading failure log: {e}")
        return {}

def _save_failures(failures: Dict[str, Dict]):
    """Save failure tracking data"""
    try:
        os.makedirs('data', exist_ok=True)
        with open(FAILURE_LOG_FILE, "w") as f:
            json.dump(failures, f, indent=2)
    except Exception as e:
        print(f"⚠️ Error saving failure log: {e}")

def _cleanup_old_failures(failures: Dict[str, Dict]) -> Dict[str, Dict]:
    """Remove old failure entries based on FAILURE_RESET_HOURS"""
    current_time = time.time()
    cutoff_time = current_time - (FAILURE_RESET_HOURS * 3600)
    
    cleaned = {}
    for address, failure_data in failures.items():
        last_failure = failure_data.get("last_failure", 0)
        if last_failure > cutoff_time:
            cleaned[address] = failure_data
    
    removed_count = len(failures) - len(cleaned)
    if removed_count > 0:
        print(f"🧹 Cleaned up {removed_count} old failure entries")
    
    return cleaned

def _should_blacklist_for_failure(address: str, failure_type: str) -> bool:
    """
    Determine if a token should be blacklisted based on failure history.
    Returns True if the token should be blacklisted.
    """
    failures = _load_failures()
    failures = _cleanup_old_failures(failures)
    
    address_lower = address.lower()
    failure_data = failures.get(address_lower, {
        "count": 0,
        "last_failure": 0,
        "failure_types": []
    })
    
    current_time = time.time()
    
    # Update failure data
    failure_data["count"] += 1
    failure_data["last_failure"] = current_time
    if failure_type not in failure_data["failure_types"]:
        failure_data["failure_types"].append(failure_type)
    
    failures[address_lower] = failure_data
    _save_failures(failures)
    
    # Check if we should blacklist
    if failure_data["count"] >= MAX_FAILURES_BEFORE_BLACKLIST:
        print(f"🛑 Token {address} has {failure_data['count']} failures - blacklisting")
        return True
    
    print(f"⚠️ Token {address} has {failure_data['count']}/{MAX_FAILURES_BEFORE_BLACKLIST} failures")
    return False

def is_blacklisted(address: str) -> bool:
    # Check regular blacklist
    if (address or "").lower() in _load():
        return True
    
    # Check delisted tokens
    try:
        state = load_delisted_state()
        delisted_tokens = state.get("delisted_tokens", [])
        return (address or "").lower() in delisted_tokens
    except Exception as e:
        print(f"⚠️ Error checking delisted tokens: {e}")
        return False
    
    return False

def add_to_blacklist(address: str, reason: str = "Unknown"):
    """Add a token to blacklist with reason tracking"""
    s = _load()
    addr_lower = (address or "").lower()
    s.add(addr_lower)
    _save(s)
    
    # Log the blacklist reason
    try:
        blacklist_log_file = "data/blacklist_reasons.json"
        os.makedirs('data', exist_ok=True)
        if os.path.exists(blacklist_log_file):
            with open(blacklist_log_file, "r") as f:
                reasons = json.load(f) or {}
        else:
            reasons = {}
        
        reasons[addr_lower] = {
            "reason": reason,
            "timestamp": time.time(),
            "address": address
        }
        
        with open(blacklist_log_file, "w") as f:
            json.dump(reasons, f, indent=2)
    except Exception as e:
        print(f"⚠️ Error logging blacklist reason: {e}")
    
    print(f"🛑 Token blacklisted: {address} (reason: {reason})")

def remove_from_blacklist(address: str, reason: str = "Manual removal"):
    """Remove a token from blacklist and clear failure history"""
    s = _load()
    addr_lower = (address or "").lower()
    if addr_lower in s:
        s.remove(addr_lower)
        _save(s)
        
        # Clear failure history
        failures = _load_failures()
        if addr_lower in failures:
            del failures[addr_lower]
            _save_failures(failures)
        
        print(f"🔓 Removed from blacklist: {address} (reason: {reason})")
        return True
    return False

def record_failure(address: str, failure_type: str, should_blacklist: bool = False) -> bool:
    """
    Record a failure for a token and optionally blacklist it.
    Returns True if the token was blacklisted.
    """
    if should_blacklist or _should_blacklist_for_failure(address, failure_type):
        add_to_blacklist(address, f"Multiple {failure_type} failures")
        return True
    return False

def get_failure_stats(address: str) -> Dict:
    """Get failure statistics for a token"""
    failures = _load_failures()
    address_lower = address.lower()
    return failures.get(address_lower, {
        "count": 0,
        "last_failure": 0,
        "failure_types": []
    })

def review_blacklisted_tokens() -> int:
    """
    Review blacklisted tokens and potentially remove old entries.
    Returns the number of tokens removed.
    """
    current_time = time.time()
    cutoff_time = current_time - (BLACKLIST_REVIEW_HOURS * 3600)
    
    try:
        blacklist_log_file = "data/blacklist_reasons.json"
        if os.path.exists(blacklist_log_file):
            with open(blacklist_log_file, "r") as f:
                reasons = json.load(f) or {}
        else:
            reasons = {}

        removed_count = 0
        for addr_lower, reason_data in list(reasons.items()):
            timestamp = reason_data.get("timestamp", 0)
            if timestamp < cutoff_time:
                if remove_from_blacklist(addr_lower, "Automatic review - old entry"):
                    removed_count += 1
                del reasons[addr_lower]

        with open(blacklist_log_file, "w") as f:
            json.dump(reasons, f, indent=2)

        if removed_count > 0:
            print(f"🔄 Automatically removed {removed_count} old blacklisted tokens")

        return removed_count

    except Exception as e:
        print(f"⚠️ Error reviewing blacklisted tokens: {e}")
        return 0

def get_blacklist_stats() -> Dict:
    """Get statistics about the blacklist"""
    blacklist = _load()
    failures = _load_failures()
    
    return {
        "blacklisted_count": len(blacklist),
        "failure_tracking_count": len(failures),
        "max_failures_before_blacklist": MAX_FAILURES_BEFORE_BLACKLIST,
        "failure_reset_hours": FAILURE_RESET_HOURS,
        "blacklist_review_hours": BLACKLIST_REVIEW_HOURS
    }