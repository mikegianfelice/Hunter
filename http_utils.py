# http_utils.py
import time, random
import requests

DEFAULT_TIMEOUT = 10
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 0.6  # seconds, exponential
DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0 (bot)"}

def _sleep(i, backoff):
    # exponential backoff with jitter
    delay = (backoff * (2 ** (i - 1))) + random.uniform(0, backoff / 3)
    time.sleep(delay)

def get_json(url, headers=None, timeout=DEFAULT_TIMEOUT, retries=DEFAULT_RETRIES, backoff=DEFAULT_BACKOFF):
    h = DEFAULT_HEADERS.copy()
    if headers:
        h.update(headers)

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=h, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            last_err = e
            if attempt < retries:
                _sleep(attempt, backoff)
            else:
                break
    raise last_err

def post_json(url, payload, headers=None, timeout=DEFAULT_TIMEOUT, retries=DEFAULT_RETRIES, backoff=DEFAULT_BACKOFF):
    h = DEFAULT_HEADERS.copy()
    h["Content-Type"] = "application/json"
    if headers:
        h.update(headers)

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=payload, headers=h, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            last_err = e
            if attempt < retries:
                _sleep(attempt, backoff)
            else:
                break
    raise last_err