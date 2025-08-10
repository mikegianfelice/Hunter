# blacklist_manager.py
import json
import os

BLACKLIST_FILE = "blacklist.json"

def _load():
    if not os.path.exists(BLACKLIST_FILE):
        return set()
    try:
        with open(BLACKLIST_FILE, "r") as f:
            data = json.load(f) or []
            return {a.lower() for a in data if isinstance(a, str)}
    except Exception:
        return set()

def _save(s: set):
    try:
        with open(BLACKLIST_FILE, "w") as f:
            json.dump(sorted(list(s)), f, indent=2)
    except Exception:
        pass

def is_blacklisted(address: str) -> bool:
    return (address or "").lower() in _load()

def add_to_blacklist(address: str):
    s = _load()
    s.add((address or "").lower())
    _save(s)
    print(f"🛑 Token blacklisted: {address}")

def remove_from_blacklist(address: str):
    s = _load()
    addr = (address or "").lower()
    if addr in s:
        s.remove(addr)
        _save(s)
        print(f"🔓 Removed from blacklist (trusted): {address}")