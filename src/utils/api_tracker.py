#!/usr/bin/env python3
"""
Centralized API call tracker for all modules.
Tracks API calls across the entire codebase for CoinGecko, CoinCap, and Helius.
"""

import json
import time
import threading
from pathlib import Path
from typing import Dict, Optional

# Thread-safe lock for concurrent access
_lock = threading.Lock()

# Singleton instance
_tracker_instance: Optional['APICallTracker'] = None

# API rate limits (Helius Developer plan: 300k/day, 50 RPC req/s)
API_LIMITS = {
    'coingecko': 330,  # 300 calls/day, but code uses 330
    'coincap': 130,
    'helius': 300000  # Helius Developer plan daily limit
}


class APICallTracker:
    """Thread-safe API call tracker that persists to disk"""
    
    def __init__(self, tracker_file: Path = None):
        if tracker_file is None:
            tracker_file = Path("data/api_call_tracker.json")
        self.tracker_file = tracker_file
        self._data = self._load()
    
    def _load(self) -> Dict:
        """Load tracker data from disk, reset if new day"""
        if self.tracker_file.exists():
            try:
                data = json.loads(self.tracker_file.read_text())
                last_reset = data.get('last_reset', 0)
                # Reset if new day (86400 seconds = 24 hours)
                if time.time() - last_reset > 86400:
                    return {
                        'helius': 0, 
                        'coingecko': 0, 
                        'coincap': 0,
                        'last_reset': time.time()
                    }
                return data
            except Exception:
                pass
        return {
            'helius': 0, 
            'coingecko': 0, 
            'coincap': 0,
            'last_reset': time.time()
        }
    
    def _save(self):
        """Save tracker data to disk"""
        try:
            # Reset if new day
            if time.time() - self._data.get('last_reset', 0) > 86400:
                self._data = {
                    'helius': 0, 
                    'coingecko': 0, 
                    'coincap': 0,
                    'last_reset': time.time()
                }
            
            self.tracker_file.parent.mkdir(parents=True, exist_ok=True)
            self.tracker_file.write_text(json.dumps(self._data, indent=2))
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error saving API tracker: {e}")
    
    def increment(self, api_name: str, count: int = 1):
        """Increment API call count (thread-safe)"""
        with _lock:
            self._data[api_name] = self._data.get(api_name, 0) + count
            self._save()
    
    def get_count(self, api_name: str) -> int:
        """Get current API call count"""
        with _lock:
            return self._data.get(api_name, 0)
    
    def get_all_counts(self) -> Dict:
        """Get all API call counts"""
        with _lock:
            return self._data.copy()
    
    def can_make_call(self, api_name: str, max_calls: Optional[int] = None) -> bool:
        """Check if we can make an API call (under rate limit)"""
        if max_calls is None:
            max_calls = API_LIMITS.get(api_name, 1000)
        with _lock:
            return self._data.get(api_name, 0) < max_calls
    
    def get_remaining(self, api_name: str, max_calls: Optional[int] = None) -> int:
        """Get remaining API calls"""
        if max_calls is None:
            max_calls = API_LIMITS.get(api_name, 1000)
        with _lock:
            return max(0, max_calls - self._data.get(api_name, 0))


def get_tracker() -> APICallTracker:
    """Get singleton tracker instance"""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = APICallTracker()
    return _tracker_instance


# Convenience functions for each API
def track_coingecko_call():
    """Track a CoinGecko API call"""
    tracker = get_tracker()
    tracker.increment('coingecko')


def track_coincap_call():
    """Track a CoinCap API call"""
    tracker = get_tracker()
    tracker.increment('coincap')


def track_helius_call():
    """Track a Helius API call"""
    tracker = get_tracker()
    tracker.increment('helius')

