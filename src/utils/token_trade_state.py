"""
Token Trade State — per-token cooldown, blacklist, and daily trade counter.

Persisted to data/token_trade_state.json.  Loaded lazily on first access.
All public functions use (token_address, chain_id) as the composite key so the
same mint on different chains is tracked independently.

KEY DESIGN DECISIONS (2026-02-21 overhaul):
  - Blacklist after 1 consecutive loss (configurable, default 1).
  - Token cooldown is 24 hours after any trade close (configurable).
  - Max 1 trade per token per rolling 24 h (configurable).
  - Early-scout and confirm-add share the SAME counter — no separate tracking.
  - Prefer-new-tokens flag: tokens traded in the last 48 h are deprioritised.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config.config_loader import (
    get_config_float,
    get_config_int,
)
from src.monitoring.structured_logger import log_info, log_warning

# ---------------------------------------------------------------------------
# Persistence paths
# ---------------------------------------------------------------------------
_STATE_FILE = Path("data/token_trade_state.json")
_STATE: Dict[str, dict] = {}
_LOADED = False

_RETENTION_SECONDS = 7 * 24 * 60 * 60  # prune entries older than 7 days


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
        log_warning("token_trade_state.load_error", f"Load failed: {e}")
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
        log_warning("token_trade_state.save_error", f"Save failed: {e}")


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
            "trades_24h": [],
            "last_pnl_usd": None,
            "total_losses": 0,
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
    Checks blacklist → cooldown → daily trade cap (in that order).
    """
    now = time.time()
    rec = _get(token_address, chain_id)

    # --- blacklist ---
    bl_until = rec.get("blacklist_until", 0)
    if now < bl_until:
        remaining_h = (bl_until - now) / 3600
        return False, (
            f"blacklisted ({remaining_h:.1f}h remaining, "
            f"consecutive_losses={rec.get('consecutive_losses', 0)}, "
            f"total_losses={rec.get('total_losses', 0)})"
        )

    # --- cooldown ---
    cd_until = rec.get("cooldown_until", 0)
    if now < cd_until:
        remaining_m = (cd_until - now) / 60
        return False, f"cooldown ({remaining_m:.0f}min remaining)"

    # --- daily trade cap (unified across all entry lanes) ---
    max_per_day = get_config_int("max_trades_per_token_per_day", 1)
    trades_24h = rec.get("trades_24h", [])
    day_ago = now - 86400
    trades_24h = [ts for ts in trades_24h if ts > day_ago]
    rec["trades_24h"] = trades_24h
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

    * Always sets cooldown (24 h default).
    * If pnl < 0: increments consecutive_losses + total_losses; blacklists
      after threshold (default: 1 loss).
    * If pnl >= 0: resets consecutive_losses (total_losses is never reset).
    """
    now = time.time()
    rec = _get(token_address, chain_id)
    k = _key(token_address, chain_id)

    rec["last_trade_ts"] = now
    rec["last_pnl_usd"] = round(pnl_usd, 4)
    trades_24h = rec.get("trades_24h", [])
    trades_24h.append(now)
    rec["trades_24h"] = trades_24h

    # --- cooldown (always applied) ---
    cooldown_minutes = get_config_float("token_cooldown_minutes", 1440)  # 24 h default
    rec["cooldown_until"] = now + cooldown_minutes * 60

    # --- consecutive losses / blacklist ---
    if pnl_usd < 0:
        rec["consecutive_losses"] = rec.get("consecutive_losses", 0) + 1
        rec["total_losses"] = rec.get("total_losses", 0) + 1
        bl_threshold = get_config_int("blacklist_after_consecutive_losses", 1)
        if rec["consecutive_losses"] >= bl_threshold:
            bl_hours = get_config_float("blacklist_hours", 24)
            rec["blacklist_until"] = now + bl_hours * 3600
            log_warning(
                "token_trade_state.blacklisted",
                f"BLACKLISTED {k} for {bl_hours}h "
                f"(consecutive_losses={rec['consecutive_losses']}, "
                f"total_losses={rec['total_losses']}, "
                f"last_pnl=${pnl_usd:.2f})",
                token=k,
                consecutive_losses=rec["consecutive_losses"],
                total_losses=rec["total_losses"],
                blacklist_hours=bl_hours,
            )
    else:
        rec["consecutive_losses"] = 0

    _save()


# ---------------------------------------------------------------------------
# Public: recently-traded token detection (prefer new tokens)
# ---------------------------------------------------------------------------

def was_recently_traded(token_address: str, chain_id: str, hours: float = 48) -> bool:
    """Return True if this token had any trade in the last ``hours`` hours."""
    rec = _get(token_address, chain_id)
    last_ts = rec.get("last_trade_ts", 0)
    return (time.time() - last_ts) < hours * 3600


def get_recent_token_addresses(chain_id: str, hours: float = 48) -> List[str]:
    """Return list of token addresses traded in the last ``hours`` hours."""
    _load()
    cutoff = time.time() - hours * 3600
    prefix = f"{chain_id.lower()}:"
    result = []
    for k, v in _STATE.items():
        if k.startswith(prefix) and v.get("last_trade_ts", 0) > cutoff:
            result.append(k[len(prefix):])
    return result


# ---------------------------------------------------------------------------
# Public: introspection
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
