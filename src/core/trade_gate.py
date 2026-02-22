"""
Trade Gate — centralised gatekeeper for ALL trade executions.

Every trade attempt must call `trade_gate_check()` BEFORE execution.
The gate enforces:

1. **Daily trade cap**       – hard limit on trades per rolling 24 h window
2. **Global cooldown**       – minimum seconds between any two trade executions
3. **Time-of-day filter**    – only allow trading during configured UTC hour windows
4. **Entry quality score**   – multi-factor composite score with minimum threshold

State is persisted to ``data/trade_gate_state.json`` so caps survive restarts.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config.config_loader import get_config, get_config_bool, get_config_float, get_config_int
from src.monitoring.structured_logger import log_info, log_warning

_STATE_FILE = Path("data/trade_gate_state.json")
_state: Dict = {}
_loaded = False


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _load_state() -> None:
    global _state, _loaded
    if _loaded:
        return
    try:
        if _STATE_FILE.exists():
            with open(_STATE_FILE, "r") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                _state = data
    except Exception as exc:
        log_warning("trade_gate.load_error", f"Could not load trade gate state: {exc}")
    _loaded = True


def _save_state() -> None:
    try:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(_STATE_FILE) + ".tmp")
        with open(tmp, "w") as fh:
            json.dump(_state, fh, indent=2)
        os.replace(tmp, _STATE_FILE)
    except Exception as exc:
        log_warning("trade_gate.save_error", f"Could not save trade gate state: {exc}")


def _prune_old_entries() -> None:
    """Remove trade timestamps older than 24 h."""
    cutoff = time.time() - 86_400
    trades = _state.get("trades_24h", [])
    _state["trades_24h"] = [ts for ts in trades if ts > cutoff]


# ---------------------------------------------------------------------------
# Config helpers  (all read from config.yaml → trade_gate section)
# ---------------------------------------------------------------------------

def _cfg_int(key: str, default: int) -> int:
    return get_config_int(f"trade_gate.{key}", default)


def _cfg_float(key: str, default: float) -> float:
    return get_config_float(f"trade_gate.{key}", default)


def _cfg_bool(key: str, default: bool) -> bool:
    return get_config_bool(f"trade_gate.{key}", default)


def _cfg(key: str, default=None):
    return get_config(f"trade_gate.{key}", default)


# ---------------------------------------------------------------------------
# 1.  Daily trade cap
# ---------------------------------------------------------------------------

def _check_daily_cap() -> Tuple[bool, str]:
    _load_state()
    _prune_old_entries()
    max_trades = _cfg_int("max_trades_per_day", 5)
    current = len(_state.get("trades_24h", []))
    if current >= max_trades:
        return False, f"daily_cap_reached ({current}/{max_trades} trades in 24h)"
    return True, f"daily_cap_ok ({current}/{max_trades})"


# ---------------------------------------------------------------------------
# 2.  Global cooldown
# ---------------------------------------------------------------------------

def _check_global_cooldown() -> Tuple[bool, str]:
    _load_state()
    cooldown_sec = _cfg_float("global_cooldown_seconds", 1800)  # 30 min default
    last_trade_ts = _state.get("last_trade_ts", 0)
    elapsed = time.time() - last_trade_ts
    if elapsed < cooldown_sec:
        remaining = int(cooldown_sec - elapsed)
        return False, f"global_cooldown ({remaining}s remaining of {int(cooldown_sec)}s)"
    return True, "global_cooldown_ok"


# ---------------------------------------------------------------------------
# 3.  Time-of-day filter
# ---------------------------------------------------------------------------

def _check_time_of_day() -> Tuple[bool, str]:
    if not _cfg_bool("enable_time_filter", True):
        return True, "time_filter_disabled"

    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour

    allowed_windows: List[List[int]] = _cfg("allowed_hours_utc", [[6, 12], [15, 19]])
    blocked_hours: List[int] = _cfg("blocked_hours_utc", [21, 22, 23])

    if hour in blocked_hours:
        return False, f"blocked_hour (UTC {hour:02d}:00 is in blocked_hours)"

    for window in allowed_windows:
        if len(window) == 2 and window[0] <= hour < window[1]:
            return True, f"allowed_window (UTC {hour:02d}:00 in [{window[0]:02d}-{window[1]:02d}))"

    return False, f"outside_allowed_hours (UTC {hour:02d}:00)"


# ---------------------------------------------------------------------------
# 4.  Entry quality score
# ---------------------------------------------------------------------------

def compute_entry_quality_score(token: dict) -> Dict:
    """
    Compute a composite quality score (0–100) from multiple factors.

    Returns dict with:
        score       – float 0-100
        breakdown   – dict of component scores
        passed      – bool (score >= threshold)
        reason      – human-readable explanation
    """
    weights = _cfg("quality_weights", {
        "liquidity": 0.25,
        "volume_24h": 0.20,
        "momentum": 0.20,
        "rsi_position": 0.15,
        "volume_spike": 0.10,
        "holder_distribution": 0.10,
    })

    breakdown: Dict[str, float] = {}

    # --- Liquidity (0-100) ---
    liq_usd = float(token.get("liquidity") or 0)
    liq_excellent = _cfg_float("quality_liq_excellent", 500_000)
    liq_floor = _cfg_float("quality_liq_floor", 100_000)
    if liq_usd >= liq_excellent:
        breakdown["liquidity"] = 100.0
    elif liq_usd >= liq_floor:
        breakdown["liquidity"] = 40 + 60 * ((liq_usd - liq_floor) / (liq_excellent - liq_floor))
    else:
        breakdown["liquidity"] = max(0, 40 * (liq_usd / liq_floor)) if liq_floor > 0 else 0

    # --- Volume 24h (0-100) ---
    vol24h = float(token.get("volume24h") or 0)
    vol_excellent = _cfg_float("quality_vol_excellent", 500_000)
    vol_floor = _cfg_float("quality_vol_floor", 100_000)
    if vol24h >= vol_excellent:
        breakdown["volume_24h"] = 100.0
    elif vol24h >= vol_floor:
        breakdown["volume_24h"] = 40 + 60 * ((vol24h - vol_floor) / (vol_excellent - vol_floor))
    else:
        breakdown["volume_24h"] = max(0, 40 * (vol24h / vol_floor)) if vol_floor > 0 else 0

    # --- Momentum (0-100) from candle data ---
    lane_details = token.get("entry_lane_details", {})
    mom_1h = abs(lane_details.get("mom_1h", 0) or 0)
    long_mom = abs(lane_details.get("long_momentum", 0) or 0)
    best_mom = max(mom_1h, long_mom)
    mom_excellent = _cfg_float("quality_mom_excellent", 0.05)  # 5%
    mom_floor = _cfg_float("quality_mom_floor", 0.015)         # 1.5%
    if best_mom >= mom_excellent:
        breakdown["momentum"] = 100.0
    elif best_mom >= mom_floor:
        breakdown["momentum"] = 30 + 70 * ((best_mom - mom_floor) / (mom_excellent - mom_floor))
    else:
        breakdown["momentum"] = max(0, 30 * (best_mom / mom_floor)) if mom_floor > 0 else 0

    # --- RSI position (0-100) — best when RSI 50-60, penalise extremes ---
    rsi = token.get("rsi") or lane_details.get("rsi")
    if rsi is not None:
        rsi = float(rsi)
        if 50 <= rsi <= 60:
            breakdown["rsi_position"] = 100.0
        elif 45 <= rsi < 50 or 60 < rsi <= 65:
            breakdown["rsi_position"] = 70.0
        elif 40 <= rsi < 45 or 65 < rsi <= 70:
            breakdown["rsi_position"] = 40.0
        else:
            breakdown["rsi_position"] = 10.0
    else:
        breakdown["rsi_position"] = 30.0  # missing data penalty

    # --- Volume spike (0-100) — recent volume acceleration ---
    vol_change_15m = token.get("volumeChange15m")
    if vol_change_15m is not None:
        try:
            vc = float(vol_change_15m)
            if abs(vc) > 1:
                vc = vc / 100.0
            if vc >= 0.50:
                breakdown["volume_spike"] = 100.0
            elif vc >= 0.20:
                breakdown["volume_spike"] = 50 + 50 * ((vc - 0.20) / 0.30)
            elif vc > 0:
                breakdown["volume_spike"] = 50 * (vc / 0.20)
            else:
                breakdown["volume_spike"] = 0.0
        except (ValueError, TypeError):
            breakdown["volume_spike"] = 20.0
    else:
        breakdown["volume_spike"] = 20.0

    # --- Holder distribution (0-100) — lower top-10 concentration = better ---
    holder_pct = float(token.get("holder_concentration_pct", 100.0))
    if holder_pct <= 20:
        breakdown["holder_distribution"] = 100.0
    elif holder_pct <= 35:
        breakdown["holder_distribution"] = 60 + 40 * ((35 - holder_pct) / 15)
    elif holder_pct <= 50:
        breakdown["holder_distribution"] = 30 + 30 * ((50 - holder_pct) / 15)
    else:
        breakdown["holder_distribution"] = max(0, 30 * ((70 - holder_pct) / 20))

    # --- Weighted composite ---
    score = 0.0
    for factor, weight in weights.items():
        score += breakdown.get(factor, 0.0) * weight

    score = round(min(100.0, max(0.0, score)), 1)
    threshold = _cfg_float("min_quality_score", 55.0)
    passed = score >= threshold

    reason_parts = []
    if not passed:
        weakest = sorted(breakdown.items(), key=lambda x: x[1])[:2]
        reason_parts.append(f"score={score}<{threshold}")
        for k, v in weakest:
            reason_parts.append(f"{k}={v:.0f}")
        reason = "quality_too_low: " + ", ".join(reason_parts)
    else:
        reason = f"quality_ok: score={score}>={threshold}"

    return {
        "score": score,
        "breakdown": breakdown,
        "passed": passed,
        "reason": reason,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Public API — the single function to call before every trade
# ---------------------------------------------------------------------------

def trade_gate_check(token: dict) -> Tuple[bool, str, Dict]:
    """
    Run ALL gate checks for a proposed trade.

    Returns:
        (allowed, reason, diagnostics_dict)

    ``diagnostics_dict`` always contains the full breakdown so callers can
    log *why* a trade was taken or skipped.
    """
    if not _cfg_bool("enabled", True):
        return True, "trade_gate_disabled", {}

    diagnostics: Dict = {}

    # 1. Daily cap
    ok, msg = _check_daily_cap()
    diagnostics["daily_cap"] = msg
    if not ok:
        _log_gate_decision(token, False, msg, diagnostics)
        return False, msg, diagnostics

    # 2. Global cooldown
    ok, msg = _check_global_cooldown()
    diagnostics["global_cooldown"] = msg
    if not ok:
        _log_gate_decision(token, False, msg, diagnostics)
        return False, msg, diagnostics

    # 3. Time of day
    ok, msg = _check_time_of_day()
    diagnostics["time_of_day"] = msg
    if not ok:
        _log_gate_decision(token, False, msg, diagnostics)
        return False, msg, diagnostics

    # 4. Quality score
    quality = compute_entry_quality_score(token)
    diagnostics["quality"] = quality
    if not quality["passed"]:
        _log_gate_decision(token, False, quality["reason"], diagnostics)
        return False, quality["reason"], diagnostics

    _log_gate_decision(token, True, f"all_gates_passed (quality={quality['score']})", diagnostics)
    return True, "all_gates_passed", diagnostics


def record_trade_execution() -> None:
    """Call AFTER a trade is successfully executed to update counters."""
    _load_state()
    now = time.time()
    _state["last_trade_ts"] = now
    trades = _state.get("trades_24h", [])
    trades.append(now)
    _state["trades_24h"] = trades
    _prune_old_entries()
    _save_state()
    log_info("trade_gate.recorded", f"Trade recorded — {len(_state['trades_24h'])} trades in last 24h")


def get_gate_status() -> Dict:
    """Return current gate status for dashboards / diagnostics."""
    _load_state()
    _prune_old_entries()
    max_trades = _cfg_int("max_trades_per_day", 5)
    current = len(_state.get("trades_24h", []))
    cooldown_sec = _cfg_float("global_cooldown_seconds", 1800)
    last_ts = _state.get("last_trade_ts", 0)
    elapsed = time.time() - last_ts
    cooldown_remaining = max(0, cooldown_sec - elapsed)

    now_utc = datetime.now(timezone.utc)
    _, time_msg = _check_time_of_day()

    return {
        "trades_today": current,
        "max_trades_per_day": max_trades,
        "trades_remaining": max(0, max_trades - current),
        "cooldown_remaining_sec": int(cooldown_remaining),
        "current_utc_hour": now_utc.hour,
        "time_filter": time_msg,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_gate_decision(token: dict, allowed: bool, reason: str, diagnostics: Dict) -> None:
    symbol = token.get("symbol", "UNKNOWN")
    address = (token.get("address") or "")[:12]
    level_fn = log_info if allowed else log_warning
    action = "ALLOWED" if allowed else "BLOCKED"
    quality_score = diagnostics.get("quality", {}).get("score", "N/A")

    level_fn(
        "trade_gate.decision",
        f"{'✅' if allowed else '🚫'} Trade {action}: {symbol} ({address}...) — {reason} "
        f"[quality={quality_score}]",
        symbol=symbol,
        address=address,
        allowed=allowed,
        reason=reason,
        quality_score=quality_score,
        diagnostics=diagnostics,
    )
