# blacklist_manager.py
import time
from typing import Dict, Set, Tuple

from src.storage.blacklist import (
    load_blacklist,
    save_blacklist,
    load_failures,
    save_failures,
    load_reasons,
    save_reasons,
)
from src.storage.delist import load_delisted_state

# Configuration
MAX_FAILURES_BEFORE_BLACKLIST = 3  # Number of failures before blacklisting
FAILURE_RESET_HOURS = 24  # Hours after which failures are reset
BLACKLIST_REVIEW_HOURS = 168  # 7 days - review blacklisted tokens


def _load() -> Set[str]:
    return set(load_blacklist())


def _save(entries: Set[str]):
    save_blacklist(entries)


def _load_failures() -> Dict[str, Dict]:
    return load_failures()


def _save_failures(failures: Dict[str, Dict]):
    save_failures(failures)


def _load_reasons() -> Dict[str, Dict]:
    return load_reasons()


def _save_reasons(reasons: Dict[str, Dict]):
    save_reasons(reasons)


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
    """Determine if a token should be blacklisted based on failure history."""
    failures = _cleanup_old_failures(_load_failures())

    address_lower = address.lower()
    failure_data = failures.get(
        address_lower,
        {
            "count": 0,
            "last_failure": 0,
            "failure_types": [],
        },
    )

    current_time = time.time()

    failure_data["count"] += 1
    failure_data["last_failure"] = current_time
    if failure_type not in failure_data["failure_types"]:
        failure_data["failure_types"].append(failure_type)

    failures[address_lower] = failure_data
    _save_failures(failures)

    if failure_data["count"] >= MAX_FAILURES_BEFORE_BLACKLIST:
        print(f"🛑 Token {address} has {failure_data['count']} failures - blacklisting")
        return True

    print(f"⚠️ Token {address} has {failure_data['count']}/{MAX_FAILURES_BEFORE_BLACKLIST} failures")
    return False


def is_blacklisted(address: str) -> bool:
    if (address or "").lower() in _load():
        return True

    try:
        state = load_delisted_state()
        delisted_tokens = state.get("delisted_tokens", [])
        return (address or "").lower() in delisted_tokens
    except Exception as e:
        print(f"⚠️ Error checking delisted tokens: {e}")
        return False


def add_to_blacklist(address: str, reason: str = "Unknown"):
    entries = _load()
    addr_lower = (address or "").lower()
    entries.add(addr_lower)
    _save(entries)

    reasons = _load_reasons()
    reasons[addr_lower] = {
        "reason": reason,
        "timestamp": time.time(),
        "address": address,
    }
    _save_reasons(reasons)

    print(f"🛑 Token blacklisted: {address} (reason: {reason})")


def remove_from_blacklist(address: str, reason: str = "Manual removal"):
    entries = _load()
    addr_lower = (address or "").lower()
    if addr_lower in entries:
        entries.remove(addr_lower)
        _save(entries)

        failures = _load_failures()
        if addr_lower in failures:
            del failures[addr_lower]
            _save_failures(failures)

        reasons = _load_reasons()
        if addr_lower in reasons:
            del reasons[addr_lower]
            _save_reasons(reasons)

        print(f"🔓 Removed from blacklist: {address} (reason: {reason})")
        return True
    return False


def record_failure(address: str, failure_type: str, should_blacklist: bool = False) -> bool:
    if should_blacklist or _should_blacklist_for_failure(address, failure_type):
        add_to_blacklist(address, f"Multiple {failure_type} failures")
        return True
    return False


def get_failure_stats(address: str) -> Dict:
    failures = _load_failures()
    address_lower = address.lower()
    return failures.get(
        address_lower,
        {
            "count": 0,
            "last_failure": 0,
            "failure_types": [],
        },
    )


def review_blacklisted_tokens() -> int:
    current_time = time.time()
    cutoff_time = current_time - (BLACKLIST_REVIEW_HOURS * 3600)

    try:
        reasons = _load_reasons()

        removed_count = 0
        for addr_lower, reason_data in list(reasons.items()):
            timestamp = reason_data.get("timestamp", 0)
            if timestamp < cutoff_time:
                if remove_from_blacklist(addr_lower, "Automatic review - old entry"):
                    removed_count += 1
                del reasons[addr_lower]

        _save_reasons(reasons)

        if removed_count > 0:
            print(f"🔄 Automatically removed {removed_count} old blacklisted tokens")

        return removed_count

    except Exception as e:
        print(f"⚠️ Error reviewing blacklisted tokens: {e}")
        return 0


def get_blacklist_stats() -> Dict:
    blacklist = _load()
    failures = _load_failures()
    return {
        "blacklisted_count": len(blacklist),
        "max_failures_before_blacklist": MAX_FAILURES_BEFORE_BLACKLIST,
        "failure_entries": len(failures),
        "blacklist_review_hours": BLACKLIST_REVIEW_HOURS,
    }