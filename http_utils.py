# http_utils.py
import time, random
import requests
from typing import Optional, Dict, Any

DEFAULT_TIMEOUT = 15  # Increased from 10 to 15 seconds
DEFAULT_RETRIES = 4  # Increased from 3 to 4 retries
DEFAULT_BACKOFF = 0.8  # Increased from 0.6 for better rate limiting
DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0 (bot)"}

# Circuit breaker state
_circuit_breaker = {
    "failures": 0,
    "last_failure_time": 0,
    "is_open": False,
    "threshold": 10,  # Open circuit after 10 consecutive failures
    "reset_timeout": 60  # Reset circuit after 60 seconds
}

def _sleep(i, backoff):
    # exponential backoff with jitter
    delay = (backoff * (2 ** (i - 1))) + random.uniform(0, backoff / 3)
    time.sleep(delay)

def _check_circuit_breaker(url: str) -> bool:
    """Check if circuit breaker should prevent request"""
    current_time = time.time()
    
    # Check if circuit is open
    if _circuit_breaker["is_open"]:
        # Check if enough time has passed to try again
        if current_time - _circuit_breaker["last_failure_time"] > _circuit_breaker["reset_timeout"]:
            print(f"🔄 Circuit breaker reset, attempting request to {url[:50]}...")
            _circuit_breaker["is_open"] = False
            _circuit_breaker["failures"] = 0
            return True
        else:
            print(f"⚠️ Circuit breaker OPEN - skipping request to {url[:50]}...")
            return False
    
    return True

def _record_success():
    """Record successful request"""
    _circuit_breaker["failures"] = 0
    _circuit_breaker["is_open"] = False

def _record_failure():
    """Record failed request and potentially open circuit"""
    _circuit_breaker["failures"] += 1
    _circuit_breaker["last_failure_time"] = time.time()
    
    if _circuit_breaker["failures"] >= _circuit_breaker["threshold"]:
        _circuit_breaker["is_open"] = True
        print(f"⚠️ Circuit breaker OPENED after {_circuit_breaker['failures']} consecutive failures")

def get_json(url, headers=None, timeout=DEFAULT_TIMEOUT, retries=DEFAULT_RETRIES, backoff=DEFAULT_BACKOFF) -> Optional[Dict[Any, Any]]:
    """
    GET request with retry logic, exponential backoff, and circuit breaker
    Returns None if circuit breaker is open or all retries fail
    """
    # Check circuit breaker
    if not _check_circuit_breaker(url):
        return None
    
    h = DEFAULT_HEADERS.copy()
    if headers:
        h.update(headers)

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=h, timeout=timeout)
            resp.raise_for_status()
            _record_success()
            return resp.json()
        except requests.exceptions.ConnectionError as e:
            last_err = e
            print(f"⚠️ Connection error (attempt {attempt}/{retries}): {type(e).__name__}")
            if attempt < retries:
                _sleep(attempt, backoff)
            else:
                _record_failure()
        except requests.exceptions.Timeout as e:
            last_err = e
            print(f"⚠️ Timeout error (attempt {attempt}/{retries}): {timeout}s exceeded")
            if attempt < retries:
                _sleep(attempt, backoff)
            else:
                _record_failure()
        except requests.exceptions.HTTPError as e:
            # For HTTP errors, don't retry if it's a client error (4xx)
            if hasattr(e, 'response') and 400 <= e.response.status_code < 500:
                _record_success()  # Don't count client errors as failures
                raise e
            last_err = e
            if attempt < retries:
                _sleep(attempt, backoff)
            else:
                _record_failure()
        except requests.exceptions.RequestException as e:
            last_err = e
            if attempt < retries:
                _sleep(attempt, backoff)
            else:
                _record_failure()
    
    raise last_err

def post_json(url, payload, headers=None, timeout=DEFAULT_TIMEOUT, retries=DEFAULT_RETRIES, backoff=DEFAULT_BACKOFF) -> Optional[Dict[Any, Any]]:
    """
    POST request with retry logic, exponential backoff, and circuit breaker
    Returns None if circuit breaker is open or all retries fail
    """
    # Check circuit breaker
    if not _check_circuit_breaker(url):
        return None
    
    h = DEFAULT_HEADERS.copy()
    h["Content-Type"] = "application/json"
    if headers:
        h.update(headers)

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=payload, headers=h, timeout=timeout)
            resp.raise_for_status()
            _record_success()
            return resp.json()
        except requests.exceptions.ConnectionError as e:
            last_err = e
            print(f"⚠️ Connection error (attempt {attempt}/{retries}): {type(e).__name__}")
            if attempt < retries:
                _sleep(attempt, backoff)
            else:
                _record_failure()
        except requests.exceptions.Timeout as e:
            last_err = e
            print(f"⚠️ Timeout error (attempt {attempt}/{retries}): {timeout}s exceeded")
            if attempt < retries:
                _sleep(attempt, backoff)
            else:
                _record_failure()
        except requests.exceptions.HTTPError as e:
            # For HTTP errors, don't retry if it's a client error (4xx)
            if hasattr(e, 'response') and 400 <= e.response.status_code < 500:
                _record_success()  # Don't count client errors as failures
                raise e
            last_err = e
            if attempt < retries:
                _sleep(attempt, backoff)
            else:
                _record_failure()
        except requests.exceptions.RequestException as e:
            last_err = e
            if attempt < retries:
                _sleep(attempt, backoff)
            else:
                _record_failure()
    
    raise last_err

def get_circuit_breaker_status() -> Dict[str, Any]:
    """Get current circuit breaker status for diagnostics"""
    return {
        "is_open": _circuit_breaker["is_open"],
        "failures": _circuit_breaker["failures"],
        "last_failure_time": _circuit_breaker["last_failure_time"],
        "threshold": _circuit_breaker["threshold"]
    }

def reset_circuit_breaker():
    """Manually reset circuit breaker"""
    _circuit_breaker["failures"] = 0
    _circuit_breaker["is_open"] = False
    _circuit_breaker["last_failure_time"] = 0
    print("✅ Circuit breaker manually reset")