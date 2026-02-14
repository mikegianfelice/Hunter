"""
Token Trade State — per-token cooldown, blacklist, and daily trade counter.

Persisted to data/token_trade_state.json.  Loaded lazily on first access.
All public functions use (token_address, chain_id) as the composite key so the
same mint on different chains is tracked independently.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.config.config_loader import (
    get_config_float,
    get_config_int,
)

# ---------------------------------------------------------------------------
# Persistence paths
# ---------------------------------------------------------------------------
_STATE_FILE = Path("data/token_trade_state.json")
_STATE: Dict[str, dict] = {}
_LOADED = False

# Retention: prune entries older than 7 days on every save
_RETENTION_SECONDS = 7 * 24 * 60 * 60


def _key(token_address: str, chain_id: str) -> str:
    return f"{chain_id.lower()}:{token_address.lower()}"


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------

def _load() -> None:
    global _STATE, _LOADED
    if _LOADED:
        return
    try:
        if _STATE_FILE.exists():
            with open(_STATE_FILE, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                _STATE = data
    except Exception as e:
        print(f"[token_trade_state] WARNING: load failed: {e}")
    _LOADED = True


def _save() -> None:
    """Atomic write with pruning."""
    _prune()
    try:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(_STATE_FILE) + ".tmp")
        with open(tmp, "w") as f:
            json.dump(_STATE, f, indent=2)
        os.replace(tmp, _STATE_FILE)
    except Exception as e:
        print(f"[token_trade_state] WARNING: save failed: {e}")


def _prune() -> None:
    """Remove entries not updated in 7 days."""
    cutoff = time.time() - _RETENTION_SECONDS
    keys_to_remove = [
        k for k, v in _STATE.items()
        if v.get("last_trade_ts", 0) < cutoff
    ]
    for k in keys_to_remove:
        del _STATE[k]


def _get(token_address: str, chain_id: str) -> dict:
    _load()
    k = _key(token_address, chain_id)
    if k not in _STATE:
        _STATE[k] = {
            "last_trade_ts": 0,
            "cooldown_until": 0,
            "consecutive_losses": 0,
            "blacklist_until": 0,
            "trades_24h": [],  # list of timestamps
        }
    return _STATE[k]


# ---------------------------------------------------------------------------
# Public: entry eligibility check
# ---------------------------------------------------------------------------

def is_token_allowed(
    token_address: str,
    chain_id: str,
) -> Tuple[bool, str]:
    """
    Return (allowed: bool, reason: str).

    Checks cooldown, blacklist, and daily trade cap.
    """
    now = time.time()
    rec = _get(token_address, chain_id)

    # --- blacklist ---
    bl_until = rec.get("blacklist_until", 0)
    if now < bl_until:
        remaining = int(bl_until - now)
        return False, f"blacklisted ({remaining}s remaining, consecutive_losses={rec.get('consecutive_losses', 0)})"

    # --- cooldown ---
    cd_until = rec.get("cooldown_until", 0)
    if now < cd_until:
        remaining = int(cd_until - now)
        return False, f"cooldown ({remaining}s remaining)"

    # --- daily trade cap ---
    max_per_day = get_config_int("max_trades_per_token_per_day", 3)
    trades_24h = rec.get("trades_24h", [])
    day_ago = now - 86400
    trades_24h = [ts for ts in trades_24h if ts > day_ago]
    rec["trades_24h"] = trades_24h  # prune in-place
    if len(trades_24h) >= max_per_day:
        return False, f"daily_cap ({len(trades_24h)}/{max_per_day} trades in 24h)"

    return True, ""


# ---------------------------------------------------------------------------
# Public: record trade close
# ---------------------------------------------------------------------------

def record_trade_close(
    token_address: str,
    chain_id: str,
    pnl_usd: float,
) -> None:
    """
    Called after a position is fully closed.

    * Always sets cooldown.
    * If pnl < 0: increments consecutive_losses; may set blacklist.
    * If pnl >= 0: resets consecutive_losses.
    """
    now = time.time()
    rec = _get(token_address, chain_id)

    # --- record timestamp ---
    rec["last_trade_ts"] = now
    trades_24h = rec.get("trades_24h", [])
    trades_24h.append(now)
    rec["trades_24h"] = trades_24h

    # --- cooldown (always) ---
    cooldown_minutes = get_config_float("token_cooldown_minutes", 240)
    rec["cooldown_until"] = now + cooldown_minutes * 60

    # --- consecutive losses / blacklist ---
    if pnl_usd < 0:
        rec["consecutive_losses"] = rec.get("consecutive_losses", 0) + 1
        bl_threshold = get_config_int("blacklist_after_consecutive_losses", 2)
        if rec["consecutive_losses"] >= bl_threshold:
            bl_hours = get_config_float("blacklist_hours", 24)
            rec["blacklist_until"] = now + bl_hours * 3600
            print(
                f"[token_trade_state] BLACKLISTED {_key(token_address, chain_id)} "
                f"for {bl_hours}h (consecutive_losses={rec['consecutive_losses']})"
            )
    else:
        rec["consecutive_losses"] = 0

    _save()


# ---------------------------------------------------------------------------
# Public: introspection (for tests / dry-run)
# ---------------------------------------------------------------------------

def get_state(token_address: str, chain_id: str) -> dict:
    """Return a copy of the current state dict for a token."""
    return dict(_get(token_address, chain_id))


def reset_all() -> None:
    """Clear all state (useful for tests)."""
    global _STATE, _LOADED
    _STATE = {}
    _LOADED = True
    _save()
