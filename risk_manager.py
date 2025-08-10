# risk_manager.py
import json, os, time, yaml
from datetime import datetime, timezone

STATE_FILE = "risk_state.json"
POSITIONS_FILE = "open_positions.json"

# Load config (with safe defaults)
with open("config.yaml", "r") as f:
    _cfg = yaml.safe_load(f)

MAX_CONCURRENT_POS = int(_cfg.get("max_concurrent_positions", 5))
DAILY_LOSS_LIMIT_USD = float(_cfg.get("daily_loss_limit_usd", 50.0))  # stop for the day if realized loss exceeds this
MAX_LOSING_STREAK = int(_cfg.get("max_losing_streak", 3))             # pause after N consecutive losses
CIRCUIT_BREAK_MIN = int(_cfg.get("circuit_breaker_minutes", 60))      # pause window when triggered
PER_TRADE_MAX_USD  = float(_cfg.get("per_trade_max_usd", _cfg.get("trade_amount_usd", 5)))

def _today_utc():
    return datetime.utcnow().strftime("%Y-%m-%d")

def _now_ts():
    return int(time.time())

def _load_state():
    s = {
        "date": _today_utc(),
        "realized_pnl_usd": 0.0,
        "buys_today": 0,
        "sells_today": 0,
        "losing_streak": 0,
        "paused_until": 0,     # epoch seconds; 0 = not paused
        "daily_spend_usd": 0.0
    }
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
                s.update(data or {})
        except Exception:
            pass
    # reset if new day
    if s.get("date") != _today_utc():
        s.update({
            "date": _today_utc(),
            "realized_pnl_usd": 0.0,
            "buys_today": 0,
            "sells_today": 0,
            "losing_streak": 0,
            "paused_until": 0,
            "daily_spend_usd": 0.0
        })
    return s

def _save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=2)

def _open_positions_count():
    if not os.path.exists(POSITIONS_FILE):
        return 0
    try:
        with open(POSITIONS_FILE, "r") as f:
            data = json.load(f) or {}
            return len(data)
    except Exception:
        return 0

def allow_new_trade(trade_amount_usd: float):
    """
    Gatekeeper before any new buy.
    Returns (allowed: bool, reason: str)
    """
    s = _load_state()

    # paused by circuit breaker?
    if s.get("paused_until", 0) > _now_ts():
        return False, f"circuit_breaker_active_until_{s['paused_until']}"

    # daily loss limit hit?
    if s.get("realized_pnl_usd", 0.0) <= -abs(DAILY_LOSS_LIMIT_USD):
        # pause until UTC midnight
        tomorrow = datetime.utcnow().replace(hour=23, minute=59, second=59, microsecond=0).timestamp()
        s["paused_until"] = int(tomorrow)
        _save_state(s)
        return False, "daily_loss_limit_hit"

    # per-trade size guard
    if trade_amount_usd > PER_TRADE_MAX_USD:
        return False, f"trade_amount_exceeds_cap_{PER_TRADE_MAX_USD}"

    # concurrent positions guard
    if _open_positions_count() >= MAX_CONCURRENT_POS:
        return False, "max_concurrent_positions_reached"

    return True, "ok"

def register_buy(usd_size: float):
    s = _load_state()
    s["buys_today"] = int(s.get("buys_today", 0)) + 1
    s["daily_spend_usd"] = float(s.get("daily_spend_usd", 0.0)) + float(usd_size or 0.0)
    _save_state(s)

def register_sell(pnl_pct: float, usd_size: float):
    """
    Record realized PnL for risk counters.
    pnl_pct: e.g., +12.5 or -8.7 (percent)
    usd_size: original position size in USD (what you bought with)
    """
    s = _load_state()
    s["sells_today"] = int(s.get("sells_today", 0)) + 1

    # Convert percent to USD PnL
    try:
        pnl_usd = (float(pnl_pct) / 100.0) * float(usd_size or 0.0)
    except Exception:
        pnl_usd = 0.0

    s["realized_pnl_usd"] = float(s.get("realized_pnl_usd", 0.0)) + pnl_usd

    # update losing streak / circuit breaker
    if pnl_usd < 0:
        s["losing_streak"] = int(s.get("losing_streak", 0)) + 1
        if s["losing_streak"] >= MAX_LOSING_STREAK:
            s["paused_until"] = _now_ts() + CIRCUIT_BREAK_MIN * 60
    else:
        s["losing_streak"] = 0  # reset on win

    _save_state(s)

def status_summary():
    s = _load_state()
    return {
        "date": s["date"],
        "open_positions": _open_positions_count(),
        "buys_today": s["buys_today"],
        "sells_today": s["sells_today"],
        "realized_pnl_usd": round(s["realized_pnl_usd"], 2),
        "losing_streak": s["losing_streak"],
        "paused_until": s["paused_until"]
    }