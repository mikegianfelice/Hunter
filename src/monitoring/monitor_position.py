import os
import sys
import time
import json
import yaml
import csv
import signal
from datetime import datetime, timedelta
# Create an alias to avoid scoping issues in functions
_dt = datetime
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

# Add project root to path if not already there
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.execution.uniswap_executor import sell_token as sell_token_ethereum
from src.execution.base_executor import sell_token as sell_token_base
from src.execution.solana_executor import sell_token_solana
from src.utils.utils import fetch_token_price_usd
from src.monitoring.telegram_bot import send_telegram_message
from src.config.config_loader import get_config, get_config_float, get_config_int, get_config_bool
from src.utils.position_sync import (
    sync_all_open_positions,
    update_open_positions_from_wallet,
    create_position_key,
    resolve_token_address,
    is_native_gas_token,
)
from src.storage.positions import load_positions as load_positions_store, replace_positions
from src.storage.performance import load_performance_data, replace_performance_data
from src.storage.delist import load_delisted_state, save_delisted_state
from src.monitoring.structured_logger import log_info


class _SkipPartialTP(Exception):
    """Sentinel: skip the AI Partial TP block without running any logic."""
    pass

# Dynamic config loading
def get_monitor_config():
    """Get current configuration values dynamically"""
    return {
        'TAKE_PROFIT': get_config_float("take_profit", 0.12),
        'STOP_LOSS': get_config_float("stop_loss", 0.07),
        'TRAILING_STOP': get_config_float("trailing_stop_percent", 0),
        'USE_DYNAMIC_TP': get_config("use_dynamic_tp", False),
        'BASE_TP': get_config_float("base_take_profit", 0.12),
        'MAX_TP': get_config_float("max_take_profit", 0.20),
        'MIN_TP': get_config_float("min_take_profit", 0.08),
        'QUALITY_TP_BONUS': get_config_float("quality_tp_bonus", 0.03),
        'VOLUME_TP_BONUS': get_config_float("volume_tp_bonus", 0.02),
        'LIQUIDITY_TP_BONUS': get_config_float("liquidity_tp_bonus", 0.02),
        # UPGRADE #1: Momentum Decay Exit
        'ENABLE_MOMENTUM_DECAY_EXIT': get_config("enable_momentum_decay_exit", True),
        'momentum_decay_threshold': get_config_float("momentum_decay_threshold", 0.005),
        'momentum_decay_delta': get_config_float("momentum_decay_delta", 0.01),
        'momentum_decay_consecutive_checks': get_config_int("momentum_decay_consecutive_checks", 2),
        # UPGRADE #4: Structure Failure Exit
        'ENABLE_STRUCTURE_FAILURE_EXIT': get_config("enable_structure_failure_exit", True),
        'structure_failure_min_age_seconds': get_config_int("structure_failure_min_age_seconds", 300),
        'structure_break_threshold': get_config_float("structure_break_threshold", 0.02),
        'structure_lookback_candles': get_config_int("structure_lookback_candles", 20),
        'structure_recent_prices_window': get_config_int("structure_recent_prices_window", 10),
        'stop_loss_slippage_buffer': get_config_float("stop_loss_slippage_buffer", 0.15),
        # --- Soft exit profit floors ---
        'soft_exit_min_profit': get_config_float("soft_exit_min_profit", 0.05),
        'momentum_decay_min_profit': get_config_float("momentum_decay_min_profit", 0.05),
        'structure_failure_min_profit': get_config_float("structure_failure_min_profit", 0.05),
        # --- AI Partial TP gating ---
        'ENABLE_AI_PARTIAL_TP': get_config_bool("enable_ai_partial_tp_manager", False),
        'ai_partial_tp_min_profit': get_config_float("ai_partial_tp_min_profit", 0.10),
        # --- Price health failsafe ---
        'max_price_staleness_seconds': get_config_int("max_price_staleness_seconds", 45),
        'invalid_price_exit_enabled': get_config_bool("invalid_price_exit_enabled", True),
        'invalid_price_exit_grace_seconds': get_config_int("invalid_price_exit_grace_seconds", 120),
        'invalid_price_action': get_config("invalid_price_action", "defensive_exit"),
    }


# ---------------------------------------------------------------------------
# Soft-exit profit floor helper
# ---------------------------------------------------------------------------
def _soft_exit_gated(gain: float, exit_name: str, per_exit_key: str, config: dict) -> bool:
    """
    Return True if the soft exit is ALLOWED (gain exceeds its profit floor).
    Return False (gated) if gain is below the floor — logs once per call.

    Uses the *higher* of (global soft_exit_min_profit, per-exit min_profit).
    """
    global_floor = config.get('soft_exit_min_profit', 0.05)
    per_exit_floor = config.get(per_exit_key, global_floor)
    effective_floor = max(global_floor, per_exit_floor)

    if gain < effective_floor:
        log_info(
            "monitor.soft_exit_gated",
            f"🚫 {exit_name} GATED: gain={gain*100:.2f}% < floor={effective_floor*100:.1f}%",
            context={
                "exit_name": exit_name,
                "gain_pct": round(gain * 100, 2),
                "floor_pct": round(effective_floor * 100, 1),
                "global_floor": global_floor,
                "per_exit_floor": per_exit_floor,
            },
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Price-health per-position tracker (in-memory, ephemeral)
# ---------------------------------------------------------------------------
# Maps position_key -> {"last_valid_price": float, "last_valid_ts": float,
#                        "invalid_since": float | None}
_price_health: Dict[str, dict] = {}

# Use absolute paths based on project root
_project_root = Path(__file__).resolve().parents[2]
POSITIONS_FILE = _project_root / "data" / "open_positions.json"
LOG_FILE = _project_root / "data" / "trade_log.csv"
MONITOR_LOCK = _project_root / "data" / ".monitor_lock"
HEARTBEAT_FILE = _project_root / "data" / ".monitor_heartbeat"
DELISTED_TOKENS_FILE = _project_root / "data" / "delisted_tokens.json"

# === Global for cleanup ===
_running = True

# Rate limit: Telegram when we skip stop-loss check due to price (avoid spam every 15s)
_stop_loss_skip_telegram_last: Dict[str, float] = {}
_STOP_LOSS_SKIP_TELEGRAM_INTERVAL_SEC = 600  # 10 minutes per position


def _maybe_send_stop_loss_skip_telegram(
    position_key: str,
    symbol: str,
    token_address: str,
    chain_id: str,
    reason: str,
) -> None:
    """Send Telegram when we skip stop-loss check due to price (rate-limited)."""
    try:
        now = time.time()
        last = _stop_loss_skip_telegram_last.get(position_key, 0.0)
        if now - last < _STOP_LOSS_SKIP_TELEGRAM_INTERVAL_SEC:
            return
        _stop_loss_skip_telegram_last[position_key] = now
        send_telegram_message(
            f"⚠️ Stop-loss check skipped (price unavailable)\n"
            f"Token: {symbol} ({token_address[:8]}...{token_address[-8:]})\n"
            f"Chain: {chain_id.upper()}\n"
            f"Reason: {reason}\n"
            f"Position may be at a loss. Monitor manually. Will retry next cycle.",
            message_type="warning",
        )
    except Exception as e:
        print(f"⚠️ Failed to send stop-loss skip Telegram: {e}")

def _pid_is_alive(pid: int) -> bool:
    try:
        if pid <= 0:
            return False
        # On Unix, sending signal 0 just checks for existence / permissions
        os.kill(pid, 0)
        return True
    except Exception:
        return False

def _write_lock():
    data = {"pid": os.getpid(), "started_at": datetime.utcnow().isoformat()}
    MONITOR_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(MONITOR_LOCK, "w") as f:
        json.dump(data, f)
    print(f"🔒 Monitor lock acquired with PID {data['pid']}")

def _remove_lock():
    try:
        if MONITOR_LOCK.exists():
            MONITOR_LOCK.unlink()
            print("🧹 Monitor lock removed.")
    except Exception as e:
        print(f"⚠️ Failed to remove monitor lock: {e}")

def _ensure_singleton():
    """
    Make sure only one monitor runs.
    If a lock exists but its PID is dead, reclaim it.
    """
    if not MONITOR_LOCK.exists():
        _write_lock()
        return

    try:
        with open(MONITOR_LOCK, "r") as f:
            data = json.load(f) or {}
        pid = int(data.get("pid", -1))
    except Exception:
        # Corrupt lock; reclaim
        print("⚠️ Corrupt lock file; reclaiming.")
        _write_lock()
        return

    if _pid_is_alive(pid):
        print(f"👁️ Another monitor is already running (PID {pid}). Exiting.")
        raise SystemExit(0)
    else:
        print(f"🗑️ Found stale lock (PID {pid} not alive). Reclaiming.")
        _write_lock()

def _heartbeat():
    try:
        HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HEARTBEAT_FILE, "w") as f:
            f.write(datetime.utcnow().isoformat())
    except Exception:
        pass

def _signal_handler(signum, frame):
    global _running
    print(f"🛑 Received signal {signum}, shutting down monitor...")
    _running = False

# --- Attach signal handlers so we always clear the lock ---
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# === Position I/O ===
def load_positions(validate_balances: bool = False):
    """
    Load positions using the centralized positions store.
    If validate_balances is True, filters out positions with zero balance.
    """
    positions = load_positions_store()
    if not validate_balances:
        return positions

    validated_positions = {}
    for position_key, position_data in positions.items():
        if isinstance(position_data, dict):
            chain_id = position_data.get("chain_id", "ethereum").lower()
            token_address = resolve_token_address(position_key, position_data)
        else:
            chain_id = "ethereum"
            token_address = position_key

        balance = _check_token_balance_on_chain(token_address, chain_id)

        if balance == -1.0:
            validated_positions[position_key] = position_data
        elif balance <= 0.0 or balance < 0.000001:
            print(f"🚫 Filtering out position {position_key} - zero/dust balance detected (manually closed)")
        else:
            validated_positions[position_key] = position_data

    return validated_positions

def _verify_position_cleanup(position_key: str, token_address: str, chain_id: str) -> bool:
    """
    Verify that a position was successfully removed from all storage locations.
    
    Returns:
        True if position is confirmed removed, False otherwise
    """
    try:
        # Check if position still exists in open_positions.json
        positions = load_positions()
        if position_key in positions:
            print(f"⚠️ [VERIFY] Position {position_key} still exists in open_positions.json")
            return False
        
        # Check if position exists by address lookup
        from src.utils.position_sync import find_position_key_by_address
        found_key = find_position_key_by_address(token_address, chain_id)
        if found_key:
            print(f"⚠️ [VERIFY] Position still exists with key {found_key} (found by address lookup)")
            return False
        
        print(f"✅ [VERIFY] Position {position_key} confirmed removed from all storage locations")
        return True
    except Exception as e:
        print(f"⚠️ [VERIFY] Error verifying position cleanup: {e}")
        return False

def _cleanup_closed_position(position_key: str, token_address: str, chain_id: str) -> bool:
    """
    Comprehensive cleanup function to remove a position from ALL storage locations:
    1. open_positions.json (via remove_position)
    2. hunter_state.db (via remove_position)
    3. performance_data.json (mark trade as closed)
    4. Partial TP manager state (clear tracking state)
    5. Balance cache (invalidate stale cache)
    6. Price health cache (remove entry)
    
    Returns True if cleanup succeeded, False otherwise.
    """
    try:
        from src.storage.positions import remove_position as remove_position_from_db
        
        # Remove from open_positions.json and hunter_state.db (with address fallback)
        removed = remove_position_from_db(position_key, token_address)
        if removed:
            print(f"✅ Removed position {position_key} from open_positions.json and hunter_state.db")
        else:
            print(f"⚠️ Failed to remove position {position_key} - will verify and retry")
            
            # Verify position is actually gone
            positions = load_positions_store()
            if position_key in positions:
                # Still present, try finding by address
                from src.utils.position_sync import find_position_key_by_address
                found_key = find_position_key_by_address(token_address, chain_id)
                if found_key and found_key != position_key:
                    print(f"🔄 Found position with different key: {found_key}, removing...")
                    removed = remove_position_from_db(found_key, token_address)
                    if removed:
                        print(f"✅ Removed position {found_key} via address lookup")
                    else:
                        print(f"❌ Failed to remove position {found_key} even with address lookup")
                elif found_key == position_key:
                    # Same key, removal should have worked - check again
                    positions_after = load_positions_store()
                    if position_key not in positions_after:
                        removed = True
                        print(f"✅ Position {position_key} confirmed removed")
                    else:
                        print(f"❌ Position {position_key} still present after removal attempt")
            else:
                # Position is gone, removal succeeded
                removed = True
                print(f"✅ Position {position_key} confirmed removed (was not in positions)")
        
        if not removed:
            print(f"❌ CRITICAL: Failed to remove position {position_key} ({token_address[:8]}...{token_address[-8:]})")
            return False
        
        # Invalidate balance cache to prevent stale data
        try:
            from src.utils.balance_cache import invalidate_token_balance_cache
            invalidate_token_balance_cache(token_address, chain_id)
            print(f"✅ Invalidated balance cache for {token_address[:8]}...{token_address[-8:]}")
        except Exception as e:
            print(f"⚠️ Failed to invalidate balance cache: {e}")
        
        # Clear partial TP manager state for this position
        # IMPORTANT: Use the key format that partial TP manager expects: "chain_id:address"
        try:
            from src.ai.ai_partial_take_profit_manager import get_partial_tp_manager
            partial_tp_manager = get_partial_tp_manager()
            # Convert position_key to the format partial TP manager uses
            tp_position_key = f"{chain_id.lower()}:{token_address.lower()}"
            partial_tp_manager.clear_position_state(tp_position_key)
            print(f"✅ Cleared partial TP state for {tp_position_key}")
        except Exception as e:
            print(f"⚠️ Error clearing partial TP state: {e}")
            import traceback
            traceback.print_exc()
            # Don't fail cleanup if partial TP cleanup fails
        
        # Also try to find and mark as closed in performance_data.json
        # (We don't remove from performance_data to preserve historical records)
        try:
            perf_data = load_performance_data()
            trades = perf_data.get("trades", [])
            token_address_lower = token_address.lower()
            chain_id_lower = chain_id.lower()
            
            updated = False
            for trade in trades:
                if (trade.get("status") == "open" and 
                    trade.get("address", "").lower() == token_address_lower and
                    trade.get("chain", "").lower() == chain_id_lower):
                    trade["status"] = "closed"
                    if not trade.get("exit_time"):
                        trade["exit_time"] = datetime.now().isoformat()
                    updated = True
                    print(f"✅ Marked trade {trade.get('id', '?')} as closed in performance_data.json")
            
            if updated:
                replace_performance_data(perf_data)
        except Exception as e:
            print(f"⚠️ Error updating performance_data.json: {e}")
        
        # Clean up price health cache
        _price_health.pop(position_key, None)
        
        return removed
            
    except Exception as e:
        print(f"⚠️ Error in _cleanup_closed_position: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_positions(positions):
    """
    Save positions to both database and JSON file.
    This ensures positions are properly removed from the database when sells complete.
    """
    # Re-key entries to the canonical mint-only format before persisting.
    canonical_positions = {}
    for key, value in positions.items():
        if isinstance(value, dict):
            token_address = resolve_token_address(key, value)
            if is_native_gas_token(token_address, value.get("symbol"), value.get("chain_id")):
                continue
            canonical_key = create_position_key(token_address)
            canonical_positions[canonical_key] = value
        else:
            canonical_positions[key] = value
    
    # CRITICAL: Use replace_positions to update both database and JSON file
    # This ensures positions are properly removed from the database when sells complete
    try:
        replace_positions(canonical_positions)
        print(f"✅ Positions saved to database and JSON ({len(canonical_positions)} position(s))")
    except Exception as e:
        print(f"⚠️ Error saving positions to database: {e}")
        # Fallback to JSON-only save if database update fails
        POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(POSITIONS_FILE, "w") as f:
            json.dump(canonical_positions, f, indent=2)
        print(f"⚠️ Positions saved to JSON only (database update failed)")

def load_delisted_tokens():
    return load_delisted_state()


def save_delisted_tokens(delisted):
    save_delisted_state(delisted or {})

def _compute_mfe_mae(position_data: dict, entry_price: float, exit_price: float) -> tuple:
    """
    Compute Max Favorable Excursion (MFE) and Max Adverse Excursion (MAE) from position data.
    Returns (mfe_pct, mae_pct) as percentages.
    """
    if not isinstance(position_data, dict):
        return None, None
    
    peak_price = position_data.get("peak_price")
    low_price = position_data.get("low_price")
    
    # Initialize with entry price if not tracked
    if peak_price is None:
        peak_price = entry_price
    if low_price is None:
        low_price = entry_price
    
    # MFE: max gain from entry (peak - entry) / entry
    mfe_pct = ((peak_price - entry_price) / entry_price * 100) if entry_price > 0 else None
    
    # MAE: max loss from entry (entry - low) / entry (negative value)
    mae_pct = ((entry_price - low_price) / entry_price * 100) if entry_price > 0 else None
    
    return mfe_pct, mae_pct

def _get_holding_time_sec(position_data: dict) -> float:
    """Get holding time in seconds from position data."""
    if not isinstance(position_data, dict):
        return None
    
    entry_timestamp = position_data.get("timestamp")
    if not entry_timestamp:
        return None
    
    try:
        entry_time = datetime.fromisoformat(entry_timestamp.replace("Z", "+00:00"))
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=datetime.now().tzinfo)
        exit_time = datetime.now(entry_time.tzinfo)
        return (exit_time - entry_time).total_seconds()
    except Exception:
        return None

def _normalize_exit_reason(reason: str) -> str:
    """
    Normalize exit reason to canonical values.
    Returns one of: momentum_decay | structure_failure | stop_loss | take_profit | 
                    trailing_stop | volume_exit | delisting | technical_exit | manual | error
    """
    reason_lower = reason.lower()
    
    if "momentum_decay" in reason_lower or "momentum_below" in reason_lower:
        return "momentum_decay"
    elif "structure_failure" in reason_lower or "structure_break" in reason_lower:
        return "structure_failure"
    elif "stop_loss" in reason_lower or "stop-loss" in reason_lower:
        return "stop_loss"
    elif "take_profit" in reason_lower or "take-profit" in reason_lower:
        return "take_profit"
    elif "trailing_stop" in reason_lower or "trailing-stop" in reason_lower:
        return "trailing_stop"
    elif "volume_exit" in reason_lower or "volume_deterioration" in reason_lower:
        return "volume_exit"
    elif "delisted" in reason_lower or "delisting" in reason_lower:
        return "delisting"
    elif "technical_exit" in reason_lower:
        return "technical_exit"
    elif "manual" in reason_lower or "reconciliation" in reason_lower:
        return "manual"
    else:
        return "error"  # Unknown reason

def _log_trade_exit(token_address: str, symbol: str, entry_price: float, exit_price: float, 
                    reason: str, position_data: dict = None, tx_hash: str = None,
                    chain_id: str = None, pnl_usd: float = None):
    """
    Enhanced trade exit logging with canonical reason, MFE/MAE, and holding time.
    This is the canonical exit logging function - all exits should use this.
    Also records the trade close in token_trade_state for cooldown/blacklist.
    """
    # Compute MFE/MAE from position data
    mfe_pct, mae_pct = _compute_mfe_mae(position_data, entry_price, exit_price) if position_data else (None, None)
    
    # Get holding time
    holding_time_sec = _get_holding_time_sec(position_data) if position_data else None
    
    # Get entry/exit timestamps
    entry_time = position_data.get("timestamp") if position_data else None
    exit_time = datetime.now().isoformat()
    
    # Log with enhanced fields
    log_trade(token_address, entry_price, exit_price, reason,
              entry_time=entry_time, exit_time=exit_time, tx_hash=tx_hash,
              mfe_pct=mfe_pct, mae_pct=mae_pct, holding_time_sec=holding_time_sec)

    # --- Record trade close for cooldown/blacklist tracking ---
    try:
        _chain = chain_id or (position_data.get("chain_id") if position_data else None) or "solana"
        # Estimate PnL if not provided
        if pnl_usd is None and entry_price > 0:
            gain_pct = (exit_price - entry_price) / entry_price
            pos_size = (position_data.get("position_size_usd", 0) if position_data else 0) or 0
            pnl_usd = gain_pct * pos_size if pos_size else (exit_price - entry_price)
        from src.utils.token_trade_state import record_trade_close
        record_trade_close(token_address, _chain, pnl_usd or 0.0)
        log_info("monitor.trade_state_recorded",
                 f"📝 Trade state recorded: {symbol} chain={_chain} pnl=${pnl_usd or 0:.2f} reason={reason}",
                 context={"token": token_address, "chain_id": _chain,
                          "pnl_usd": round(pnl_usd or 0, 4), "reason": reason})
    except Exception as e:
        print(f"⚠️ Failed to record trade state for cooldown/blacklist: {e}")

def log_trade(token, entry_price, exit_price, reason="normal", 
              entry_time=None, exit_time=None, tx_hash=None, 
              mfe_pct=None, mae_pct=None, holding_time_sec=None):
    """
    Log trade exit with canonical exit reason and optional telemetry.
    
    Args:
        token: Token address/symbol
        entry_price: Entry price
        exit_price: Exit price
        reason: Exit reason (will be normalized)
        entry_time: ISO timestamp of entry (optional)
        exit_time: ISO timestamp of exit (optional)
        tx_hash: Transaction hash (optional)
        mfe_pct: Max favorable excursion % (optional)
        mae_pct: Max adverse excursion % (optional)
        holding_time_sec: Holding time in seconds (optional)
    """
    pnl_pct = ((exit_price - entry_price) / entry_price) * 100 if entry_price else 0.0
    canonical_reason = _normalize_exit_reason(reason)
    
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "token": token,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl_pct": round(pnl_pct, 2),
        "reason": reason,  # Keep original for backward compatibility
        "exit_reason": canonical_reason,  # Canonical reason
    }
    
    # Add optional fields if provided
    if entry_time:
        row["entry_time"] = entry_time
    if exit_time:
        row["exit_time"] = exit_time
    if tx_hash:
        row["tx_hash"] = tx_hash
    if mfe_pct is not None:
        row["mfe_pct"] = round(mfe_pct, 2)
    if mae_pct is not None:
        row["mae_pct"] = round(mae_pct, 2)
    if holding_time_sec is not None:
        row["holding_time_sec"] = round(holding_time_sec, 0)
    
    file_exists = LOG_FILE.exists()
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", newline="") as f:
            # Get all possible fieldnames (union of existing and new)
            all_fields = ["timestamp", "token", "entry_price", "exit_price", "pnl_pct", "reason", 
                         "exit_reason", "entry_time", "exit_time", "tx_hash", "mfe_pct", "mae_pct", "holding_time_sec"]
            writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction='ignore')
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        
        # Structured log for observability
        log_info("trade.exit",
                f"Trade exit: {canonical_reason} | Token: {token} | PnL: {pnl_pct:.2f}%",
                context={
                    "token": token,
                    "exit_reason": canonical_reason,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": pnl_pct,
                    "original_reason": reason,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "tx_hash": tx_hash,
                    "mfe_pct": mfe_pct,
                    "mae_pct": mae_pct,
                    "holding_time_sec": holding_time_sec,
                })
        print(f"📄 Trade logged: {row}")
    except Exception as e:
        print(f"⚠️ Failed to write trade log: {e}")

def _apply_trailing_stop(state: dict, addr: str, current_price: float) -> float:
    """
    Update peak price and compute dynamic stop if trailing is enabled.
    Returns the dynamic stop price (or None if not active).
    """
    config = get_monitor_config()
    if config['TRAILING_STOP'] <= 0:
        return None

    # track per-token peak
    peak_key = f"{addr}_peak"
    peak = state.get(peak_key)

    if peak is None or current_price > peak:
        state[peak_key] = current_price
        peak = current_price

    # trailing_stop is a % drop from peak (calculate always, not just when peak updates)
    trail_stop_price = peak * (1 - config['TRAILING_STOP'])
    return trail_stop_price

def _analyze_sell_fees(tx_hash: str, chain_id: str) -> dict:
    """Analyze sell transaction to extract fee data"""
    # Skip fee analysis for placeholder values (verified by balance check, etc.)
    if tx_hash in ["verified_by_balance", "assumed_success"]:
        return {}
    
    try:
        if chain_id.lower() == "solana":
            from src.utils.solana_transaction_analyzer import analyze_jupiter_transaction
            from src.config.secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS
            
            fee_data = analyze_jupiter_transaction(
                SOLANA_RPC_URL, 
                tx_hash, 
                SOLANA_WALLET_ADDRESS,
                is_buy=False
            )
            
            return {
                'exit_gas_fee_usd': fee_data.get('gas_fee_usd', 0),
                'actual_proceeds_usd': fee_data.get('actual_proceeds_usd', 0),
                'sell_tx_hash': tx_hash
            }
        elif chain_id.lower() in ["ethereum", "base", "arbitrum", "polygon"]:
            from src.utils.transaction_analyzer import analyze_sell_transaction
            from src.execution.uniswap_executor import w3
            
            fee_data = analyze_sell_transaction(w3, tx_hash)
            
            return {
                'exit_gas_fee_usd': fee_data.get('gas_fee_usd', 0),
                'actual_proceeds_usd': fee_data.get('actual_proceeds_usd', 0),
                'sell_tx_hash': tx_hash
            }
    except Exception as e:
        print(f"⚠️ Error analyzing sell transaction fees: {e}")
        return {}
    return {}

def _fetch_token_price_multi_chain(token_address: str) -> float:
    """
    Fetch token price based on chain type.
    For now, we'll try to detect Solana vs Ethereum tokens.
    """
    try:
        # Try Solana price first (if it looks like a Solana address)
        if len(token_address) == 44:  # Solana addresses are 44 chars
            try:
                from src.execution.solana_executor import get_token_price_usd
                price = get_token_price_usd(token_address)
                if price and price > 0:
                    print(f"🔗 Fetched Solana price for {token_address[:8]}...{token_address[-8:]}: ${price:.6f}")
                    return price
                else:
                    # Try Jupiter executor (different API/rate limit bucket than solana_executor)
                    # Status report uses this - align for consistency so stop loss triggers when status shows loss
                    print(f"⚠️ Solana price fetch returned 0, trying Jupiter executor fallback...")
                    try:
                        from src.execution.jupiter_executor import get_token_price_usd as jupiter_get_price
                        price = jupiter_get_price(token_address)
                        if price and price > 0:
                            print(f"✅ Jupiter fallback price: ${price:.6f}")
                            return price
                    except Exception as e2:
                        print(f"⚠️ Jupiter fallback failed: {e2}")
                    # Last resort: direct DexScreener call
                    try:
                        import requests
                        url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
                        response = requests.get(url, timeout=15)
                        if response.status_code == 200:
                            data = response.json()
                            pairs = data.get("pairs", [])
                            if pairs:
                                for pair in pairs:
                                    price = float(pair.get("priceUsd", 0))
                                    if price > 0:
                                        print(f"✅ Direct DexScreener fallback price: ${price:.6f}")
                                        return price
                    except Exception as e2:
                        print(f"⚠️ DexScreener fallback also failed: {e2}")
                    print(f"⚠️ Zero price returned for {token_address[:8]}...{token_address[-8:]}")
                    return 0.0
            except Exception as e:
                print(f"⚠️ Solana price fetch failed: {e}")
                return 0.0
        
        # Fallback to Ethereum price fetching (only for non-Solana addresses)
        price = fetch_token_price_usd(token_address)
        if price and price > 0:
            print(f"🔗 Fetched Ethereum price for {token_address[:8]}...{token_address[-8:]}: ${price:.6f}")
            return price
            
        return 0.0
    except Exception as e:
        print(f"⚠️ Price fetch failed for {token_address}: {e}")
        return 0.0

def _sell_token_multi_chain(token_address: str, chain_id: str, symbol: str = "?", amount_usd: Optional[float] = None) -> str:
    """
    Sell token using the appropriate executor based on chain
    
    Args:
        token_address: Token address to sell
        chain_id: Chain identifier (solana, ethereum, base)
        symbol: Token symbol for logging
        amount_usd: Optional USD amount to sell (for partial sells). If None, sells entire balance.
    
    Returns:
        Transaction hash or None if failed
    """
    print(f"🔍 [DEBUG] _sell_token_multi_chain called: token={token_address}, chain={chain_id}, symbol={symbol}, amount_usd={amount_usd}")
    try:
        if chain_id == "ethereum":
            print(f"🔄 Selling {symbol} on Ethereum...")
            # Ethereum executor doesn't support partial sells yet - sell full balance
            tx_hash, success = sell_token_ethereum(token_address)
            print(f"🔍 [DEBUG] Ethereum sell result: tx_hash={tx_hash}, success={success}")
        elif chain_id == "base":
            print(f"🔄 Selling {symbol} on Base...")
            # Get token balance for BASE
            from src.execution.base_executor import get_token_balance
            balance = get_token_balance(token_address)
            print(f"🔍 [DEBUG] Base balance check: {balance}")
            if balance > 0:
                # Base executor accepts token_amount parameter for partial sells
                sell_amount = balance if amount_usd is None else None  # TODO: Calculate token amount from USD
                tx_hash, success = sell_token_base(token_address, sell_amount or balance, symbol)
                print(f"🔍 [DEBUG] Base sell result: tx_hash={tx_hash}, success={success}")
            else:
                print(f"❌ [ERROR] No {symbol} balance to sell on Base: balance={balance}")
                return None
        elif chain_id == "solana":
            print(f"🔄 [SELL START] Selling {symbol} on Solana...")
            # For Solana, we need to get the balance first and convert to USD
            from src.execution.jupiter_lib import JupiterCustomLib
            from src.config.secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY
            
            print(f"🔍 [DEBUG] Initializing JupiterCustomLib...")
            lib = JupiterCustomLib(SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY)
            
            print(f"🔍 [DEBUG] Checking token balance for {token_address}...")
            balance = lib.get_token_balance(token_address)
            print(f"🔍 [DEBUG] Token balance check result: balance={balance}, type={type(balance)}")
            
            if amount_usd is None:
                # Full sell - calculate from balance
                if balance is None:
                    print(f"⚠️ [WARNING] Balance pre-check failed (likely RPC rate limit). Proceeding with executor-level balance read.")
                    amount_usd = 0.0  # Will be calculated in executor
                elif balance <= 0:
                    print(f"❌ [ERROR] No {symbol} balance to sell: balance={balance} (balance <= 0)")
                    return None
                else:
                    print(f"✅ [BALANCE OK] Token balance: {balance}")
                    
                    # Get current price to calculate USD value
                    print(f"🔍 [DEBUG] Fetching current price for USD calculation...")
                    current_price = _fetch_token_price_multi_chain(token_address)
                    print(f"🔍 [DEBUG] Current price: {current_price}")
                    
                    if current_price <= 0:
                        print(f"⚠️ [WARNING] Could not get current price for {symbol} (price={current_price}), using estimated value")
                        # Fallback: use a conservative estimate
                        current_price = 0.01  # Conservative estimate
                        print(f"🔍 [DEBUG] Using fallback price: {current_price}")
                    
                    # Calculate USD value of the token balance
                    amount_usd = balance * current_price
                    print(f"🔍 [DEBUG] Calculated amount_usd: {amount_usd} (balance={balance} * price={current_price})")
            else:
                # Partial sell - use provided amount_usd
                print(f"📊 [PARTIAL SELL] Selling ${amount_usd:.2f} worth of {symbol}")
                if balance is None or balance <= 0:
                    print(f"⚠️ [WARNING] Cannot verify balance for partial sell, proceeding anyway...")
                else:
                    current_price = _fetch_token_price_multi_chain(token_address)
                    if current_price > 0:
                        balance_value_usd = balance * current_price
                        if amount_usd > balance_value_usd:
                            print(f"⚠️ [WARNING] Requested amount ${amount_usd:.2f} exceeds balance value ${balance_value_usd:.2f}, selling full balance")
                            amount_usd = balance_value_usd
            
            # Try selling with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Get balance before sell attempt for verification fallback
                    from src.execution.jupiter_lib import JupiterCustomLib
                    from src.config.secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY
                    lib = JupiterCustomLib(SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY)
                    original_balance = lib.get_token_balance(token_address)
                    print(f"🔍 [DEBUG] Token balance before sell attempt: {original_balance}")
                    
                    print(f"🔍 [DEBUG] Calling sell_token_solana (attempt {attempt+1}/{max_retries}) with amount_usd={amount_usd}...")
                    tx_hash, success = sell_token_solana(token_address, amount_usd, symbol)
                    print(f"🔍 [DEBUG] sell_token_solana result: tx_hash={tx_hash}, success={success}, type(tx_hash)={type(tx_hash)}")
                    
                    # If success=True, treat as successful even if tx_hash is empty (unusual but possible)
                    if success:
                        if tx_hash:
                            print(f"✅ [SUCCESS] Sell successful on attempt {attempt+1}: {tx_hash}")
                            # Invalidate cache after successful sell
                            try:
                                from src.utils.balance_cache import invalidate_token_balance_cache
                                invalidate_token_balance_cache(token_address, chain_id)
                            except Exception as e:
                                print(f"⚠️ Failed to invalidate balance cache: {e}")
                            return tx_hash
                        else:
                            # Success=True but no tx_hash - verify with balance check
                            print(f"⚠️ [WARNING] Sell reported success but no tx_hash; verifying with balance check...")
                            time.sleep(5)  # Wait for transaction to propagate
                            check_balance = lib.get_token_balance(token_address)
                            if check_balance is not None and original_balance is not None:
                                if check_balance < original_balance * 0.9 or check_balance == 0 or check_balance < 0.000001:
                                    print(f"✅ [BALANCE VERIFIED] Balance check confirms sell succeeded (balance: {check_balance})")
                                    # Invalidate cache after successful sell
                                    try:
                                        from src.utils.balance_cache import invalidate_token_balance_cache
                                        invalidate_token_balance_cache(token_address, chain_id)
                                    except Exception as e:
                                        print(f"⚠️ Failed to invalidate balance cache: {e}")
                                    # Return a placeholder since we don't have tx_hash
                                    return "verified_by_balance"
                            # If balance check fails but success=True, still assume success
                            print(f"⚠️ [ASSUME SUCCESS] Success=True but can't verify; assuming success to avoid false negatives")
                            return "assumed_success"
                    elif tx_hash and not success:
                        # We have a transaction hash but success=False - verify on-chain
                        # This handles cases where RPC verification failed but transaction succeeded
                        print(f"⚠️ [VERIFY] Sell returned tx_hash but success=False; verifying on-chain: {tx_hash}")
                        try:
                            print(f"🔍 [VERIFY] Original token balance (before sell): {original_balance}")
                            
                            # Wait longer for transaction to propagate and be confirmed
                            print(f"⏳ Waiting 8 seconds for transaction to propagate...")
                            time.sleep(8)
                            
                            # Retry verification multiple times with exponential backoff
                            # Use more retries and longer waits since RPC verification can be slow
                            verified = None
                            for verify_attempt in range(5):
                                verified = lib.verify_transaction_success(tx_hash)
                                
                                if verified is True:
                                    print(f"✅ [VERIFIED] Transaction {tx_hash} verified successful on-chain (attempt {verify_attempt+1})")
                                    # Invalidate cache after successful sell
                                    try:
                                        from src.utils.balance_cache import invalidate_token_balance_cache
                                        invalidate_token_balance_cache(token_address, chain_id)
                                    except Exception as e:
                                        print(f"⚠️ Failed to invalidate balance cache: {e}")
                                    return tx_hash
                                elif verified is False:
                                    print(f"⚠️ [VERIFIED FAIL] Transaction {tx_hash} reported as failed on attempt {verify_attempt+1}")
                                    # Double-check with balance before assuming failure
                                    if verify_attempt >= 2:  # After multiple attempts, check balance
                                        print(f"🔍 [DOUBLE-CHECK] Verifying with balance check before assuming failure...")
                                        time.sleep(3)
                                        check_balance = lib.get_token_balance(token_address)
                                        if check_balance is not None and original_balance is not None:
                                            if check_balance < original_balance * 0.9 or check_balance == 0 or check_balance < 0.000001:
                                                print(f"✅ [BALANCE VERIFIED] Balance check shows sell succeeded despite verification failure report")
                                                return tx_hash
                                    # If we're on early attempts, continue retrying verification
                                    if verify_attempt < 2:
                                        wait_time = 4 * (verify_attempt + 1)  # 4s, 8s
                                        print(f"⏳ Verification reported failure, waiting {wait_time}s before retry...")
                                        time.sleep(wait_time)
                                        verified = None  # Reset to retry
                                        continue
                                    break  # Confirmed failure after multiple checks
                                else:
                                    # Can't verify yet - wait and retry
                                    if verify_attempt < 4:
                                        wait_time = 4 * (verify_attempt + 1)  # 4s, 8s, 12s, 16s
                                        print(f"⏳ Verification uncertain, waiting {wait_time}s before retry...")
                                        time.sleep(wait_time)
                            
                            # If verification is still uncertain, check wallet balance as fallback
                            if verified is None:
                                print(f"🔍 [FALLBACK] Verification uncertain after multiple attempts, checking wallet balance...")
                                time.sleep(5)  # Wait a bit more
                                new_balance = lib.get_token_balance(token_address)
                                print(f"🔍 [FALLBACK] New token balance: {new_balance}")
                                
                                if new_balance is not None and original_balance is not None:
                                    if new_balance < original_balance * 0.9:  # Balance decreased by at least 10%
                                        print(f"✅ [BALANCE CHECK] Token balance decreased from {original_balance} to {new_balance} - sell likely succeeded")
                                        # Invalidate cache after successful sell
                                        try:
                                            from src.utils.balance_cache import invalidate_token_balance_cache
                                            invalidate_token_balance_cache(token_address, chain_id)
                                        except Exception as e:
                                            print(f"⚠️ Failed to invalidate balance cache: {e}")
                                        return tx_hash
                                    elif new_balance == 0 or new_balance < 0.000001:
                                        print(f"✅ [BALANCE CHECK] Token balance is zero/dust - sell succeeded")
                                        # Invalidate cache after successful sell
                                        try:
                                            from src.utils.balance_cache import invalidate_token_balance_cache
                                            invalidate_token_balance_cache(token_address, chain_id)
                                        except Exception as e:
                                            print(f"⚠️ Failed to invalidate balance cache: {e}")
                                        return tx_hash
                                    else:
                                        print(f"⚠️ [BALANCE CHECK] Token balance unchanged ({original_balance} -> {new_balance}) - will assume success with tx_hash")
                                        # Even if balance unchanged, if we have a tx_hash and can't verify, assume success
                                        # (balance check might fail due to RPC issues, but transaction may have succeeded)
                                        print(f"⚠️ [UNCERTAIN] Cannot definitively verify but have tx_hash; assuming success to avoid false negatives")
                                        return tx_hash
                                else:
                                    print(f"⚠️ [BALANCE CHECK] Could not compare balances (original={original_balance}, new={new_balance})")
                            
                            # If we still can't verify but have a tx_hash, assume success
                            # (transaction was submitted, and RPC verification can be unreliable)
                            # Better to assume success than fail and potentially retry an already-successful transaction
                            if verified is None:
                                print(f"⚠️ [UNCERTAIN] Cannot verify transaction {tx_hash} after multiple attempts; assuming success (was submitted)")
                                return tx_hash
                            elif verified is False:
                                # Only assume failure if we've checked multiple times AND balance confirms it
                                print(f"❌ [VERIFIED FAIL] Transaction {tx_hash} confirmed as failed after multiple verification attempts")
                                # Continue to retry logic below - but this should be rare
                                
                        except Exception as verify_error:
                            print(f"⚠️ [VERIFY ERROR] Error verifying transaction {tx_hash}: {verify_error}")
                            import traceback
                            print(f"🔍 [DEBUG] Verification error traceback:\n{traceback.format_exc()}")
                            # If we have a tx_hash, assume success (transaction was submitted)
                            # Verification errors are often due to RPC issues, not actual transaction failure
                            print(f"⚠️ Assuming success for transaction {tx_hash} (was submitted, verification error)")
                            # Invalidate cache after successful sell
                            try:
                                from src.utils.balance_cache import invalidate_token_balance_cache
                                invalidate_token_balance_cache(token_address, chain_id)
                            except Exception as e:
                                print(f"⚠️ Failed to invalidate balance cache: {e}")
                            return tx_hash
                    
                    # No tx_hash or verified failure - retry
                    print(f"⚠️ [RETRY {attempt+1}/{max_retries}] Sell failed (success={success}, tx_hash={tx_hash}), retrying...")
                    if attempt < max_retries - 1:
                        wait_time = 2 * (attempt + 1)  # Exponential backoff: 2s, 4s, 6s
                        print(f"⏳ Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        print(f"❌ [FAILED] All {max_retries} sell attempts failed")
                        return None
                            
                except Exception as e:
                    print(f"❌ [EXCEPTION] Sell attempt {attempt+1} failed: {e}")
                    import traceback
                    print(f"🔍 [DEBUG] Exception traceback:\n{traceback.format_exc()}")
                    if attempt < max_retries - 1:
                        wait_time = 2 * (attempt + 1)
                        print(f"⏳ Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        print(f"❌ [FAILED] All {max_retries} sell attempts failed due to exceptions")
                        return None
            
            return None
        else:
            print(f"❌ [ERROR] Unsupported chain for selling: {chain_id}")
            return None
            
    except Exception as e:
        print(f"❌ [EXCEPTION] Error selling {symbol} on {chain_id}: {e}")
        import traceback
        print(f"🔍 [DEBUG] Exception traceback:\n{traceback.format_exc()}")
        return None

def _detect_delisted_token(token_address: str, consecutive_failures: int) -> bool:
    """
    Detect if a token is likely delisted based on consecutive price fetch failures
    """
    # Consider delisted after 5 consecutive failures (2.5 minutes of monitoring)
    return consecutive_failures >= 5

def _check_token_balance_on_chain(token_address: str, chain_id: str, use_cache: bool = True) -> float:
    """
    Check token balance on the specified chain with caching support.
    Returns balance amount (0.0 if balance is zero, -1.0 if check failed).
    
    Args:
        token_address: Token contract address
        chain_id: Chain identifier
        use_cache: If True, use cached balance if available and fresh
    """
    # Try cache first if enabled
    if use_cache:
        from src.utils.balance_cache import get_cached_token_balance, update_token_balance_cache
        from src.config.config_loader import get_config_bool
        
        if get_config_bool("balance_cache_enabled", True):
            cached_balance, is_valid = get_cached_token_balance(token_address, chain_id)
            if is_valid:
                return float(cached_balance)
    
    try:
        chain_lower = chain_id.lower()
        balance = None
        
        if chain_lower == "solana":
            from src.execution.jupiter_lib import JupiterCustomLib
            from src.config.secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY
            lib = JupiterCustomLib(SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY)
            balance = lib.get_token_balance(token_address)
            # None means check failed (error) - return -1.0 to indicate unknown state
            if balance is None:
                return -1.0
            balance = float(balance)
            
        elif chain_lower == "base":
            from src.execution.base_executor import get_token_balance
            balance = get_token_balance(token_address)
            balance = float(balance or 0.0)
            
        elif chain_lower == "ethereum":
            # Use web3 to check ERC20 token balance
            from web3 import Web3
            from src.config.secrets import INFURA_URL, WALLET_ADDRESS
            
            w3 = Web3(Web3.HTTPProvider(INFURA_URL))
            if not w3.is_connected():
                print(f"⚠️ Web3 not connected for balance check on Ethereum")
                return 0.0
            
            # ERC20 ABI minimal - just balanceOf and decimals
            erc20_abi = json.loads("""[
                {"constant":true,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"type":"function"},
                {"constant":true,"inputs":[{"name":"owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"type":"function"}
            ]""")
            
            wallet = Web3.to_checksum_address(WALLET_ADDRESS)
            token_addr = Web3.to_checksum_address(token_address)
            token_contract = w3.eth.contract(address=token_addr, abi=erc20_abi)
            
            balance_wei = token_contract.functions.balanceOf(wallet).call()
            decimals = token_contract.functions.decimals().call()
            balance = float(balance_wei) / (10 ** decimals)
            
        else:
            print(f"⚠️ Unsupported chain for balance check: {chain_id}")
            return 0.0
        
        # Cache successful balance check
        if balance is not None and use_cache:
            from src.utils.balance_cache import update_token_balance_cache
            from src.config.config_loader import get_config_bool
            if get_config_bool("balance_cache_enabled", True):
                update_token_balance_cache(token_address, chain_id, balance)
        
        return balance
            
    except Exception as e:
        # If balance check fails, assume we can't verify - keep position
        # (don't auto-remove on transient errors)
        print(f"⚠️ Balance check failed for {token_address} on {chain_id}: {e}")
        return -1.0  # Return -1 to indicate check failed (not zero balance)

def _prune_positions_with_zero_balance(positions: dict) -> Tuple[dict, list]:
    """
    Remove positions that no longer have wallet balance.
    Auto-reconciles after manual closes.
    
    Returns: (pruned_positions_dict, list_of_closed_token_addresses)
    """
    pruned = dict(positions)
    to_remove = []
    
    if not pruned:
        return pruned, []
    
    print("🔍 Auto-reconciling positions with on-chain balances...")
    
    for position_key, position_data in list(positions.items()):
        # Handle both old format (float) and new format (dict)
        if isinstance(position_data, dict):
            chain_id = position_data.get("chain_id", "ethereum").lower()
            symbol = position_data.get("symbol", "?")
            token_address = resolve_token_address(position_key, position_data)
        else:
            chain_id = "ethereum"
            symbol = "?"
            token_address = position_key  # Legacy format uses address as key
        
        try:
            balance = _check_token_balance_on_chain(token_address, chain_id)
            
            if balance == -1.0:
                # Check failed - keep position to be safe
                print(f"⏸️  Skipping balance check for {symbol} ({token_address[:8]}...{token_address[-8:]}) - check failed")
                continue
            elif balance <= 0.0 or balance < 0.000001:
                # Zero or dust balance - position was closed manually
                # Treat very small balances (< 0.000001) as dust/zero
                print(f"✅ Detected zero/dust balance ({balance:.8f}) for {symbol} ({token_address[:8]}...{token_address[-8:]} on {chain_id.upper()}) - removing from tracking (manually closed)")
                to_remove.append(position_key)
                pruned.pop(position_key, None)
            else:
                # Has balance - position still open
                print(f"✓ {symbol} ({token_address[:8]}...{token_address[-8:]}) still has balance: {balance:.6f}")
                
        except Exception as e:
            print(f"⚠️ Error checking balance for {position_key}: {e}")
            # On error, keep the position to be safe
    
    if to_remove:
        print(f"🧹 Auto-reconciled {len(to_remove)} manually closed position(s)")
        # Persist removals for manual closes
        save_positions(pruned)
        # Send Telegram notification
        try:
            closed_list = "\n".join([f"• {addr[:8]}...{addr[-8:]}" for addr in to_remove])
            send_telegram_message(
                f"✅ Auto-reconciled manually closed positions:\n{closed_list}\n"
                f"These positions were detected as closed and removed from tracking."
            )
        except Exception:
            pass  # Don't fail on Telegram errors
    
    return pruned, to_remove

def _find_open_trade_by_address(token_address: str, chain_id: str = None):
    """Find an open trade in performance tracker by token address"""
    try:
        from src.core.performance_tracker import performance_tracker
        open_trades = performance_tracker.get_open_trades()
        
        # Find trade matching address and optionally chain
        for trade in open_trades:
            if trade.get('address', '').lower() == token_address.lower():
                if chain_id is None or trade.get('chain', '').lower() == chain_id.lower():
                    return trade
        return None
    except Exception as e:
        print(f"⚠️ Error finding trade for {token_address}: {e}")
        return None

def _calculate_dynamic_take_profit_for_position(trade: Dict, config: Dict) -> float:
    """Calculate dynamic take profit for a specific position based on quality, volume, liquidity"""
    if not config.get('USE_DYNAMIC_TP', False):
        # Dynamic TP disabled, use static take profit
        return config['TAKE_PROFIT']
    
    # Start with base take profit
    tp = config.get('BASE_TP', 0.12)
    
    if not trade:
        # No trade data available, fallback to static
        return config['TAKE_PROFIT']
    
    # Get quality score (0-100 scale)
    quality_score = float(trade.get('quality_score', 0))
    volume_24h = float(trade.get('volume_24h', 0))
    liquidity = float(trade.get('liquidity', 0))
    
    # Apply bonuses based on token quality
    # Quality score is 0-100, so 0.7 means 70, etc.
    quality_percent = quality_score * 100 if quality_score <= 1.0 else quality_score
    
    # High quality tokens get bonus
    if quality_percent >= 65:  # Lowered from 70 to 65
        tp += config.get('QUALITY_TP_BONUS', 0.03)
        print(f"  📈 High quality token: +{config.get('QUALITY_TP_BONUS', 0.03)*100:.0f}% TP")
    
    # High volume tokens get bonus
    if volume_24h >= 2_000_000:  # Lowered from $10M to $2M
        tp += config.get('VOLUME_TP_BONUS', 0.02)
        print(f"  📊 High volume: +{config.get('VOLUME_TP_BONUS', 0.02)*100:.0f}% TP")
    
    # High liquidity tokens get bonus
    if liquidity >= 2_000_000:  # Lowered from $5M to $2M
        tp += config.get('LIQUIDITY_TP_BONUS', 0.02)
        print(f"  💧 High liquidity: +{config.get('LIQUIDITY_TP_BONUS', 0.02)*100:.0f}% TP")
    
    # Clamp to min/max
    tp = max(config.get('MIN_TP', 0.08), min(config.get('MAX_TP', 0.20), tp))
    
    print(f"  🎯 Dynamic TP: {tp*100:.0f}% (base: {config.get('BASE_TP', 0.12)*100:.0f}%)")
    return tp

def _check_technical_exit_signals(
    position: Dict[str, Any],
    current_price: float,
    gain: float,
    token_address: str,
    chain_id: str,
    config: Dict[str, Any]
) -> Optional[str]:
    """
    Check if technical indicators suggest exit signal
    
    Returns:
        Exit reason string if signal detected, None otherwise
    """
    from src.config.config_loader import get_config_bool, get_config_float
    
    if not get_config_bool("enable_technical_exit_signals", True):
        return None
    
    # Only check technical exits if position is profitable (unless configured otherwise)
    require_profit = get_config_bool("technical_exit_require_profit", True)
    min_profit = get_config_float("technical_exit_min_profit_for_signal", 0.02)
    
    if require_profit and gain < min_profit:
        return None  # Don't exit on technical signals if not profitable
    
    try:
        from src.utils.market_data_fetcher import MarketDataFetcher
        from src.utils.technical_indicators import TechnicalIndicators
        
        # Fetch candlestick data for technical analysis
        market_fetcher = MarketDataFetcher()
        candles = market_fetcher.get_candlestick_data(
            token_address,
            chain_id,
            hours=24,  # Get 24 hours of data
            force_fetch=False
        )
        
        if not candles or len(candles) < 20:  # Need minimum data
            return None
        
        # Calculate technical indicators
        tech_indicators = TechnicalIndicators()
        indicators = tech_indicators.calculate_all_indicators(candles, include_confidence=True)
        
        # Check data quality
        data_quality = indicators.get('data_quality', {})
        if data_quality.get('is_approximation', False):
            confidence = data_quality.get('confidence_score', 0.0)
            if confidence < 0.5:  # Low confidence - skip
                return None
        
        # 1. RSI Overbought Check
        rsi = indicators.get('rsi', 50)
        rsi_threshold = get_config_float("technical_exit_rsi_overbought", 70)
        if rsi > rsi_threshold and gain > 0:
            return f"rsi_overbought_{rsi:.1f}"
        
        # 2. MACD Bearish Crossover Check
        if get_config_bool("technical_exit_macd_bearish", True):
            macd_data = indicators.get('macd', {})
            if isinstance(macd_data, dict):
                macd_line = macd_data.get('macd', 0)
                macd_signal = macd_data.get('signal', 0)
                macd_histogram = macd_data.get('histogram', 0)
                
                # Bearish crossover: MACD line crosses below signal line
                if macd_line < macd_signal and macd_histogram < 0:
                    return f"macd_bearish_crossover"
        
        # 3. Bollinger Bands Upper Band Check
        bollinger_threshold = get_config_float("technical_exit_bollinger_upper", 0.85)
        bollinger = indicators.get('bollinger', {})
        if isinstance(bollinger, dict):
            bollinger_position = bollinger.get('position', 0.5)
            if bollinger_position >= bollinger_threshold and gain > 0:
                return f"bollinger_upper_band_{bollinger_position:.2f}"
        
        # 4. VWAP Below + Declining Volume Check
        if get_config_bool("technical_exit_vwap_below", True):
            vwap = indicators.get('vwap', {})
            if isinstance(vwap, dict):
                vwap_price = vwap.get('vwap', current_price)
                if current_price < vwap_price:
                    # Check if volume is declining
                    volume_data = indicators.get('volume_profile', {})
                    if isinstance(volume_data, dict):
                        volume_trend = volume_data.get('trend', 'neutral')
                        if volume_trend == 'declining' and gain > 0:
                            return f"vwap_below_declining_volume"
        
        return None
        
    except Exception as e:
        print(f"⚠️ Error checking technical exit signals: {e}")
        import traceback
        traceback.print_exc()
        return None

def _get_or_store_entry_volume(
    position_data: Dict[str, Any],
    token_address: str,
    chain_id: str
) -> Optional[float]:
    """
    Get entry volume from position data, or fetch and store if missing.
    Uses cached volume fetch to minimize API calls.
    
    Returns:
        Entry volume (24h average) in USD, or None if unavailable
    """
    # Check if entry volume already stored
    if isinstance(position_data, dict):
        entry_volume = position_data.get("entry_volume_24h_avg")
        if entry_volume is not None:
            return float(entry_volume)
        
        # If not stored, fetch current volume using cached method and store it
        try:
            from src.utils.market_data_fetcher import MarketDataFetcher
            market_fetcher = MarketDataFetcher()
            
            # Use cached volume fetch (will cache for 5 minutes)
            current_volume = market_fetcher.get_token_volume_cached(token_address, chain_id)
            if current_volume and current_volume > 0:
                # Store it for future reference
                position_data["entry_volume_24h_avg"] = float(current_volume)
                from src.storage.positions import upsert_position
                from src.utils.position_sync import create_position_key
                position_key = create_position_key(token_address)
                upsert_position(position_key, position_data)
                print(f"📊 Stored entry volume for {token_address[:8]}...: ${current_volume:,.0f}")
                return float(current_volume)
        except Exception as e:
            print(f"⚠️ Error fetching/storing entry volume: {e}")
            import traceback
            traceback.print_exc()
    
    return None

def _check_volume_deterioration(
    position_data: Dict[str, Any],
    current_price: float,
    gain: float,
    token_address: str,
    chain_id: str,
    config: Dict[str, Any]
) -> Optional[str]:
    """
    Check if volume has deteriorated significantly from entry.
    Uses cached volume fetch to minimize API calls.
    Requires confirmation (multiple consecutive checks) to avoid false exits.
    
    Returns:
        Exit reason string if volume deteriorated, None otherwise
    """
    from src.config.config_loader import get_config_bool, get_config_float
    
    if not get_config_bool("enable_volume_deterioration_exit", True):
        return None
    
    # Only check volume exits if position is profitable (unless configured otherwise)
    require_profit = get_config_bool("volume_deterioration_require_profit", True)
    min_profit = get_config_float("volume_deterioration_min_profit", 0.01)
    
    if require_profit and gain < min_profit:
        return None  # Don't exit on volume deterioration if not profitable
    
    try:
        # Get entry volume (fetch and store if missing)
        entry_volume_avg = _get_or_store_entry_volume(position_data, token_address, chain_id)
        if not entry_volume_avg or entry_volume_avg <= 0:
            return None  # Can't compare without entry volume
        
        # Get current volume using CACHED method (reduces API calls)
        from src.utils.market_data_fetcher import MarketDataFetcher
        market_fetcher = MarketDataFetcher()
        current_volume = market_fetcher.get_token_volume_cached(token_address, chain_id)
        
        if not current_volume or current_volume <= 0:
            return None  # Can't determine if deteriorated
        
        # Calculate volume drop percentage
        volume_drop_pct = (entry_volume_avg - current_volume) / entry_volume_avg
        threshold = get_config_float("volume_deterioration_threshold", 0.50)
        required_confirmations = int(get_config_float("volume_deterioration_confirmations", 2))
        
        # Track consecutive low-volume checks (requires confirmation)
        if isinstance(position_data, dict):
            low_volume_count = position_data.get("low_volume_count", 0)
            
            if volume_drop_pct >= threshold:
                # Volume has dropped significantly - increment counter
                low_volume_count += 1
                position_data["low_volume_count"] = low_volume_count
                
                # Save updated position data
                from src.storage.positions import upsert_position
                from src.utils.position_sync import create_position_key
                position_key = create_position_key(token_address)
                upsert_position(position_key, position_data)
                
                print(f"📉 Volume deteriorated: {(volume_drop_pct*100):.1f}% (entry: ${entry_volume_avg:,.0f}, current: ${current_volume:,.0f}) - confirmation {low_volume_count}/{required_confirmations}")
                
                # Require multiple consecutive checks before triggering exit
                if low_volume_count >= required_confirmations:
                    # Reset counter before returning exit signal
                    position_data["low_volume_count"] = 0
                    upsert_position(position_key, position_data)
                    drop_pct_display = volume_drop_pct * 100
                    return f"volume_deterioration_{drop_pct_display:.1f}%"
            else:
                # Volume recovered or not deteriorated - reset counter
                if position_data.get("low_volume_count", 0) > 0:
                    position_data["low_volume_count"] = 0
                    from src.storage.positions import upsert_position
                    from src.utils.position_sync import create_position_key
                    position_key = create_position_key(token_address)
                    upsert_position(position_key, position_data)
        
        return None
        
    except Exception as e:
        print(f"⚠️ Error checking volume deterioration: {e}")
        import traceback
        traceback.print_exc()
        return None

def _sync_only_positions_with_balance():
    """
    Sync positions from performance_data.json, but ONLY if they actually have on-chain balances.
    This prevents manually closed positions from being re-added.
    """
    try:
        from src.utils.position_sync import create_position_key

        perf_data = load_performance_data()
        open_positions = load_positions(validate_balances=False)

        trades = perf_data.get("trades", [])
        open_trades = [t for t in trades if t.get("status") == "open"]

        synced_count = 0
        for trade in open_trades:
            address = trade.get("address", "")
            if not address:
                continue

            chain = trade.get("chain", "ethereum").lower()
            symbol = trade.get("symbol", "?")
            trade_id = trade.get("id", "")

            position_key = create_position_key(address)

            if position_key in open_positions:
                continue

            balance = _check_token_balance_on_chain(address, chain)

            if balance == -1.0:
                print(f"⏸️  Skipping sync for {symbol} ({address[:8]}...{address[-8:]}) - balance check failed")
                continue
            elif balance <= 0.0 or balance < 0.000001:
                print(f"🚫 Skipping sync for {symbol} ({address[:8]}...{address[-8:]}) - zero/dust balance ({balance:.8f}) detected (manually closed)")
                _close_trade_record(trade)
                continue
            else:
                print(f"✓ {symbol} ({address[:8]}...{address[-8:]}) has balance {balance:.6f} - syncing")
                from src.utils.position_sync import sync_position_from_performance_data_with_key
                entry_price = float(trade.get("entry_price", 0))
                position_size_usd = trade.get("position_size_usd", 0.0)
                entry_time = trade.get("entry_time", "")
                if sync_position_from_performance_data_with_key(
                    position_key,
                    address,
                    symbol,
                    chain,
                    entry_price,
                    position_size_usd,
                    trade_id,
                    entry_time,
                ):
                    synced_count += 1

        if any(t.get("status") == "manual_close" for t in open_trades):
            perf_data["last_updated"] = datetime.now().isoformat()
            replace_performance_data(perf_data)

        if synced_count > 0:
            print(f"✅ Synced {synced_count} position(s) with verified balances from performance dataset")
    except Exception as e:
        print(f"⚠️ Error in smart position sync: {e}")

def _close_trade_record(trade: Dict[str, Any]) -> None:
    trade['status'] = 'manual_close'
    trade['exit_time'] = datetime.now().isoformat()
    trade['exit_price'] = 0.0
    trade['pnl_usd'] = 0.0
    trade['pnl_percent'] = 0.0


def monitor_all_positions():
    config = get_monitor_config()
    
    # Use timestamp-based reconciliation instead of cycle count (persists across restarts)
    # This ensures reconciliation runs regularly even if monitor restarts frequently
    from src.config.config_loader import get_config_int
    import time
    
    # Get reconciliation interval from config (default: every 5 cycles = ~75 seconds)
    reconciliation_interval_cycles = get_config_int("position_reconciliation.scan_interval_cycles", 5)
    
    # Track last reconciliation time (persists across function calls)
    _last_reconciliation_time = getattr(monitor_all_positions, '_last_reconciliation_time', 0.0)
    current_time = time.time()
    
    # Calculate cycle duration (15 seconds per cycle based on sleep interval)
    cycle_duration_seconds = 15
    reconciliation_interval_seconds = reconciliation_interval_cycles * cycle_duration_seconds
    
    # Run reconciliation if enough time has passed OR if this is the first run
    time_since_last_reconciliation = current_time - _last_reconciliation_time
    validate_balances = (time_since_last_reconciliation >= reconciliation_interval_seconds) or (_last_reconciliation_time == 0.0)
    
    # Update last reconciliation time if we're running it this cycle
    if validate_balances:
        monitor_all_positions._last_reconciliation_time = current_time
    
    # Also track cycle count for logging
    _monitor_cycle_count = getattr(monitor_all_positions, '_cycle_count', 0) + 1
    monitor_all_positions._cycle_count = _monitor_cycle_count
    
    # Load existing positions (with selective balance validation)
    # CRITICAL: For pruning we must pass the FULL positions dict from store - _prune does its
    # own balance checks. Passing load_positions(validate_balances=True) would pre-filter
    # zero-balance positions, so _prune would never find any to remove and never persist.
    if validate_balances:
        time_since_str = f"{time_since_last_reconciliation:.0f}s" if _last_reconciliation_time > 0 else "first run"
        print(f"🔍 [CYCLE {_monitor_cycle_count}] Validating balances this cycle (last run: {time_since_str} ago, interval: {reconciliation_interval_seconds}s)...")
    
    positions = load_positions_store()  # Always load full store for pruning
    reconciled_closed = []  # Track manually closed positions
    
    # Auto-reconcile FIRST: Remove positions with zero on-chain balance (manual closes)
    # This prevents manually closed positions from being re-added
    # Only prune when validating balances (every N cycles) to reduce RPC calls
    if positions and validate_balances:
        # Save original positions data before pruning (needed for performance tracker updates)
        original_positions_before_prune = dict(positions)
        
        # Auto-reconcile: Remove positions with zero on-chain balance (manual closes)
        positions, reconciled_closed = _prune_positions_with_zero_balance(positions)
        
        # Update performance tracker for manually closed positions BEFORE syncing
        if reconciled_closed:
            try:
                from src.core.performance_tracker import performance_tracker
                for position_key in reconciled_closed:
                    # Find the position data to get chain_id and entry_price before it was removed
                    old_position_data = original_positions_before_prune.get(position_key)
                    if old_position_data:
                        if isinstance(old_position_data, dict):
                            chain_id = old_position_data.get("chain_id", "ethereum")
                            entry_price = float(old_position_data.get("entry_price", 0))
                            # Extract actual token address
                            token_address = resolve_token_address(position_key, old_position_data)
                        else:
                            chain_id = "ethereum"
                            entry_price = float(old_position_data) if old_position_data else 0
                            token_address = position_key
                        
                        # Find and close the trade using trade_id if available
                        if isinstance(old_position_data, dict) and old_position_data.get("trade_id"):
                            trade_id = old_position_data.get("trade_id")
                            open_trades = performance_tracker.get_open_trades()
                            trade = next((t for t in open_trades if t.get("id") == trade_id), None)
                        else:
                            # Fallback to finding by address
                            trade = _find_open_trade_by_address(token_address, chain_id)
                        if trade:
                            # Try to get current price if possible, otherwise use entry price (0 PnL)
                            # Use original_entry_price if available for partial TP positions
                            original_entry_price = old_position_data.get("original_entry_price", entry_price) if isinstance(old_position_data, dict) else entry_price
                            current_price = _fetch_token_price_multi_chain(token_address) or original_entry_price
                            position_size = trade.get('position_size_usd', 0)
                            gain = ((current_price - original_entry_price) / original_entry_price) if original_entry_price > 0 else 0
                            pnl_usd = gain * position_size
                            performance_tracker.log_trade_exit(trade['id'], current_price, pnl_usd, "manual_close")
                            
                            # Get symbol for logging and notification
                            symbol = trade.get('symbol') or (old_position_data.get('symbol') if isinstance(old_position_data, dict) else '?')
                            
                            # Log to trade_log.csv
                            log_trade(token_address, original_entry_price, current_price, "manual_close_detected")
                            
                            # Send detailed Telegram notification
                            send_telegram_message(
                                f"✅ Position Closed (Detected)\n"
                                f"Token: {symbol} ({token_address})\n"
                                f"Chain: {chain_id.upper()}\n"
                                f"Entry: ${original_entry_price:.6f}\n"
                                f"Exit: ${current_price:.6f} ({gain * 100:.2f}%)\n"
                                f"Reason: Already sold (detected by balance check)"
                            )
                            
                            print(f"📊 Updated performance tracker for manually closed position: {symbol}")
            except Exception as e:
                print(f"⚠️ Failed to update performance tracker for reconciled positions: {e}")
    
    # Do not write or sync open_positions.json from monitor; only read current positions
    positions = load_positions()
    if not positions:
        print("📭 No open positions to monitor.")
        return
    

    # Load delisting tracking
    delisted_tokens = load_delisted_tokens()
    failure_counts = delisted_tokens.get("failure_counts", {})
    
    updated_positions = dict(positions)  # shallow copy
    closed_positions = list(reconciled_closed)  # Include reconciled positions
    # ephemeral state for trailing stop peaks
    trail_state = {}

    for position_key, position_data in list(positions.items()):
        # Handle both old format (float) and new format (dict)
        if isinstance(position_data, dict):
            entry_price = float(position_data.get("entry_price", 0))
            chain_id = position_data.get("chain_id", "ethereum").lower()
            symbol = position_data.get("symbol", "?")
            # Extract actual token address from position data (for composite keys)
            token_address = resolve_token_address(position_key, position_data)
            trade_id = position_data.get("trade_id")
        else:
            # Legacy format - assume Ethereum
            entry_price = float(position_data)
            chain_id = "ethereum"
            symbol = "?"
            token_address = position_key
            trade_id = None
            
        if entry_price <= 0:
            print(f"⚠️ Invalid entry price for {position_key}: {entry_price}")
            continue

        trade_id_str = f" [Trade: {trade_id}]" if trade_id else ""
        # Use _dt alias to avoid scoping issues
        timestamp = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] 🔍 Monitoring token: {symbol} ({token_address}) on {chain_id.upper()}{trade_id_str}")
        print(f"[{timestamp}] 🎯 Entry price: ${entry_price:.6f}")

        # Check max position duration (auto-close stale positions)
        if isinstance(position_data, dict) and position_data.get("timestamp"):
            try:
                from src.config.config_loader import get_config_bool, get_config_float
                
                if get_config_bool("enable_max_position_duration", True):
                    max_duration_hours = get_config_float("max_position_duration_hours", 72)
                    entry_timestamp_str = position_data.get("timestamp")
                    
                    if entry_timestamp_str:
                        entry_time = _dt.fromisoformat(entry_timestamp_str.replace("Z", "+00:00"))
                        if entry_time.tzinfo is None:
                            entry_time = entry_time.replace(tzinfo=_dt.now().tzinfo)
                        current_time = _dt.now(entry_time.tzinfo)
                        duration = current_time - entry_time
                        duration_hours = duration.total_seconds() / 3600
                        
                        if duration_hours >= max_duration_hours:
                            print(f"⏰ Position age: {duration_hours:.1f} hours (max: {max_duration_hours}h)")
                            print(f"🔄 Auto-closing stale position to free capital for new opportunities...")
                            
                            # Sell the token
                            try:
                                tx_hash = _sell_token_multi_chain(token_address, chain_id, symbol)
                                if tx_hash:
                                    print(f"✅ Sold stale position: {tx_hash}")
                                    _cleanup_closed_position(position_key, token_address, chain_id)
                                    closed_positions.append(position_key)
                                    
                                    # Update performance tracker
                                    if trade_id:
                                        try:
                                            current_price = _fetch_token_price_multi_chain(token_address) or entry_price
                                            position_size = position_data.get("position_size_usd", 0)
                                            pnl_percent = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                                            pnl_usd = (pnl_percent / 100) * position_size
                                            performance_tracker.log_trade_exit(
                                                trade_id,
                                                current_price,
                                                pnl_usd,
                                                "max_duration_close",
                                                sell_tx_hash=tx_hash
                                            )
                                        except Exception as e:
                                            print(f"⚠️ Failed to update performance tracker: {e}")
                                    continue
                                else:
                                    print(f"⚠️ Failed to sell stale position, will retry next cycle")
                            except Exception as e:
                                print(f"⚠️ Error selling stale position: {e}")
                                # Continue monitoring, will retry next cycle
                        else:
                            print(f"⏰ Position age: {duration_hours:.1f} hours (max: {max_duration_hours}h)")
            except Exception as e:
                print(f"⚠️ Error checking position duration: {e}")

        # Fetch current price using multi-chain function
        current_price = _fetch_token_price_multi_chain(token_address)

        # --- Price health tracking ---
        import math as _math
        _ph = _price_health.setdefault(position_key, {
            "last_valid_price": 0.0,
            "last_valid_ts": 0.0,
            "invalid_since": None,
        })
        _price_is_valid = (
            current_price is not None
            and isinstance(current_price, (int, float))
            and not _math.isnan(current_price)
            and current_price > 0
            and abs(current_price - 0.000001) > 1e-9  # reject sentinel fallback
        )
        _now_ts = time.time()
        if _price_is_valid:
            _ph["last_valid_price"] = current_price
            _ph["last_valid_ts"] = _now_ts
            _ph["invalid_since"] = None
        else:
            if _ph["invalid_since"] is None:
                _ph["invalid_since"] = _now_ts

        # Track price fetch failures
        if current_price == 0:
            # CRITICAL: For open positions, verify balance BEFORE incrementing failures
            # This prevents false failure accumulation when APIs have temporary issues
            if position_key in positions:
                print(f"🛡️ Position {symbol} is actively tracked - verifying balance to prevent false failures...")
                balance = _check_token_balance_on_chain(token_address, chain_id)
                
                if balance > 0:
                    # Token definitely exists - just a price API issue
                    # Try one more direct DexScreener call as emergency fallback
                    print(f"✅ Token has balance ({balance:.6f}) - price API issue, trying emergency fallback...")
                    try:
                        import requests
                        url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
                        response = requests.get(url, timeout=10)
                        if response.status_code == 200:
                            data = response.json()
                            pairs = data.get("pairs", [])
                            if pairs:
                                for pair in pairs:
                                    price = float(pair.get("priceUsd", 0))
                                    if price > 0:
                                        print(f"✅ Emergency DexScreener fallback successful: ${price:.6f}")
                                        current_price = price
                                        break
                    except Exception as e:
                        print(f"⚠️ Emergency fallback also failed: {e}")
                    
                    if current_price == 0:
                        print(f"⏳ Holding position (price unavailable but token exists in wallet)...")
                        _maybe_send_stop_loss_skip_telegram(
                            position_key, symbol, token_address, chain_id,
                            "price unavailable (holding; DexScreener fallback failed)",
                        )
                        continue
                elif balance == 0:
                    # Balance is zero - might be delisted or sold manually, track failure normally
                    print(f"⚠️ Zero balance detected for {symbol}, tracking price failure")
                else:  # balance == -1.0 (check failed)
                    # Can't verify - skip this check to avoid false negatives
                    print(f"⏸️ Balance check failed for {symbol}, skipping failure tracking")
                    _maybe_send_stop_loss_skip_telegram(
                        position_key, symbol, token_address, chain_id,
                        "balance check failed; skipping failure tracking",
                    )
                    continue
            
            failure_counts[token_address] = failure_counts.get(token_address, 0) + 1
            print(f"⚠️ Price fetch failure #{failure_counts[token_address]} for {token_address[:8]}...{token_address[-8:]}")
            
            # Check if token is likely delisted
            if _detect_delisted_token(token_address, failure_counts[token_address]):
                # CRITICAL SAFEGUARD: If this position is tracked in open_positions, never mark as delisted
                # This protects against false positives from balance check failures
                if position_key in positions:
                    print(f"🛡️ Position {symbol} is actively tracked in open positions - skipping delisting check to prevent false positive")
                    failure_counts[token_address] = max(0, failure_counts[token_address] - 2)
                    _maybe_send_stop_loss_skip_telegram(
                        position_key, symbol, token_address, chain_id,
                        "price unavailable; skipping delisting (tracked); stop-loss not checked",
                    )
                    continue
                
                # BEFORE marking as delisted, verify the token doesn't exist in wallet
                # If we have a balance, it's clearly not delisted - just a price API issue
                print(f"🔍 Verifying delisting status by checking on-chain balance...")
                balance = _check_token_balance_on_chain(token_address, chain_id)
                
                if balance == -1.0:
                    # Balance check failed - can't verify, so don't mark as delisted
                    print(f"⏸️  Cannot verify delisting: balance check failed. Keeping position active.")
                    # Reset failure count to give it more time
                    failure_counts[token_address] = max(0, failure_counts[token_address] - 2)
                    continue
                elif balance > 0:
                    # Token exists in wallet - definitely not delisted, just price API issue
                    print(f"✅ Token has balance ({balance:.6f}) - not delisted. Price API issue detected.")
                    # Reset failure count since we know the token exists
                    failure_counts[token_address] = 0
                    # Continue monitoring but skip PnL calculations since we don't have price
                    print(f"⏳ Holding position (price unavailable but token exists in wallet)...")
                    _maybe_send_stop_loss_skip_telegram(
                        position_key, symbol, token_address, chain_id,
                        "price unavailable (holding; balance verified, not delisted)",
                    )
                    continue
                else:
                    # Balance is 0 AND price fetch fails - likely delisted
                    print(f"🚨 TOKEN LIKELY DELISTED: {token_address[:8]}...{token_address[-8:]}")
                    print(f"💸 Investment lost: ${entry_price:.6f}")
                    
                    # Log as delisted trade
                    _log_trade_exit(token_address, symbol, entry_price, 0.0, "delisted",
                                  position_data=position_data if isinstance(position_data, dict) else None)
                    
                    # Update performance tracker with exit
                    trade = _find_open_trade_by_address(token_address, chain_id)
                    if trade:
                        # Position value lost (100% loss)
                        position_size = trade.get('position_size_usd', 0)
                        pnl_usd = -position_size  # Full loss
                        from src.core.performance_tracker import performance_tracker
                        performance_tracker.log_trade_exit(trade['id'], 0.0, pnl_usd, "delisted")
                        print(f"📊 Updated performance tracker for delisted token: {trade.get('symbol', '?')}")
                    
                    # Send Telegram alert
                    send_telegram_message(
                        f"🚨 TOKEN DELISTED - INVESTMENT LOST!\n"
                        f"Token: {token_address[:8]}...{token_address[-8:]}\n"
                        f"Entry: ${entry_price:.6f}\n"
                        f"Current: $0.00 (DELISTED)\n"
                        f"Loss: 100% (${entry_price:.6f})\n"
                        f"⚠️ Token no longer tradeable"
                    )
                    
                    # CRITICAL: Remove position from ALL storage locations (open_positions.json, hunter_state.db, performance_data.json)
                    _cleanup_closed_position(position_key, token_address, chain_id)
                    
                    # Also update performance tracker if we have a trade record
                    trade = _find_open_trade_by_address(token_address, chain_id)
                    if trade:
                        position_size = trade.get('position_size_usd', 0)
                        pnl_usd = -position_size  # 100% loss
                        from src.core.performance_tracker import performance_tracker
                        performance_tracker.log_trade_exit(trade['id'], 0.0, pnl_usd, "delisted")
                    
                    # Remove from active positions (use position_key)
                    closed_positions.append(position_key)
                    updated_positions.pop(position_key, None)
                    continue
        else:
            # Reset failure count on successful price fetch
            failure_counts[token_address] = 0

        if current_price is None or current_price == 0:
            print(f"⚠️ Could not fetch current price for {token_address}")
            # Still check if we can estimate loss from entry price (conservative approach)
            # If we can't get price, we can't determine stop loss, so skip this cycle
            # but log the issue for visibility
            print(f"⚠️ Skipping stop loss check for {symbol} - price unavailable")
            _maybe_send_stop_loss_skip_telegram(
                position_key, symbol, token_address, chain_id,
                "price unavailable",
            )
            continue

        # Check if price is fallback value (0.000001) - indicates API failure, not actual price
        # This prevents false stop-loss triggers when price APIs fail
        FALLBACK_PRICE = 0.000001
        if abs(current_price - FALLBACK_PRICE) < 1e-9:  # Use epsilon comparison for float
            print(f"⚠️ Price is fallback value ({FALLBACK_PRICE}) - all price APIs failed")
            print(f"⏳ Skipping stop-loss check for {symbol} to prevent false trigger")
            print(f"🔄 Will retry price fetch on next cycle...")
            _maybe_send_stop_loss_skip_telegram(
                position_key, symbol, token_address, chain_id,
                "price fallback (all APIs failed)",
            )
            # Fall through to price-health failsafe below (do NOT continue)

        # --- Price health failsafe ---
        # If price has been invalid for longer than grace period, trigger defensive exit.
        if not _price_is_valid:
            _grace = config.get('invalid_price_exit_grace_seconds', 120)
            _inv_since = _ph.get("invalid_since")
            _elapsed = (_now_ts - _inv_since) if _inv_since else 0
            _last_good = _ph.get("last_valid_price", 0)
            _last_good_age = _now_ts - _ph.get("last_valid_ts", 0)

            if _inv_since and _elapsed >= _grace and config.get('invalid_price_exit_enabled', True):
                _action = config.get('invalid_price_action', 'defensive_exit')
                log_info("monitor.price_health_failsafe",
                         f"🚨 PRICE_INVALID_FAILSAFE: {symbol} invalid for {_elapsed:.0f}s "
                         f"(grace={_grace}s, action={_action}, last_valid=${_last_good:.6f} {_last_good_age:.0f}s ago)",
                         context={"token": token_address, "chain_id": chain_id,
                                  "invalid_seconds": round(_elapsed), "grace": _grace,
                                  "action": _action, "last_valid_price": _last_good})
                if _action == "defensive_exit":
                    print(f"🚨 [PRICE_INVALID_FAILSAFE] Defensive exit for {symbol} — price invalid {_elapsed:.0f}s")
                    tx = _sell_token_multi_chain(token_address, chain_id, symbol)
                    if tx:
                        _exit_price = _last_good if _last_good > 0 else entry_price
                        _log_trade_exit(token_address, symbol, entry_price, _exit_price,
                                        "PRICE_INVALID_FAILSAFE",
                                        position_data=position_data if isinstance(position_data, dict) else None,
                                        tx_hash=tx, chain_id=chain_id)
                        trade = _find_open_trade_by_address(token_address, chain_id)
                        if trade:
                            _gain_est = (_exit_price - entry_price) / entry_price if entry_price > 0 else 0
                            _pnl_est = _gain_est * trade.get('position_size_usd', 0)
                            from src.core.performance_tracker import performance_tracker
                            performance_tracker.log_trade_exit(trade['id'], _exit_price, _pnl_est, "PRICE_INVALID_FAILSAFE")
                        _cleanup_closed_position(position_key, token_address, chain_id)
                        send_telegram_message(
                            f"🚨 PRICE_INVALID_FAILSAFE!\n"
                            f"Token: {symbol} ({token_address[:8]}...)\n"
                            f"Chain: {chain_id.upper()}\n"
                            f"Price invalid for {_elapsed:.0f}s\n"
                            f"Last valid: ${_last_good:.6f}\n"
                            f"TX: {tx}"
                        )
                        closed_positions.append(position_key)
                        updated_positions.pop(position_key, None)
                    else:
                        print(f"❌ [PRICE_INVALID_FAILSAFE] Sell failed for {symbol}, will retry next cycle")
                else:
                    # pause_monitoring mode — just log, skip all TP/SL checks
                    print(f"⏸️ [PRICE_HEALTH] Pausing monitoring for {symbol} (price invalid {_elapsed:.0f}s)")
            continue  # skip remaining checks this cycle (price unreliable)

        timestamp = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] 📈 Current price: ${current_price:.6f}")
        
        # Calculate original_entry_price FIRST - used by peak/low defaults and gain calculation
        # (Bug fix: was referenced at lines 1842-1843 before assignment, causing crash and blocking stop-loss)
        original_entry_price = position_data.get("original_entry_price", entry_price) if isinstance(position_data, dict) else entry_price
        gain = (current_price - original_entry_price) / original_entry_price
        print(f"[{timestamp}] 📊 PnL: {gain * 100:.2f}% (entry: ${original_entry_price:.6f})")
        
        # UPGRADE #4: Update recent_prices array for structure failure detection (every cycle)
        # Also track MFE/MAE (max favorable/adverse excursion) for trade summary
        if isinstance(position_data, dict):
            recent_prices = position_data.get("recent_prices", [])
            if not recent_prices:
                # Initialize with entry price if empty (backward compatibility)
                entry_price_for_init = float(position_data.get("entry_price", current_price))
                recent_prices = [entry_price_for_init]
            
            # Add current price
            recent_prices.append(current_price)
            window_size = get_config_int("structure_recent_prices_window", 10)
            if len(recent_prices) > window_size:
                recent_prices = recent_prices[-window_size:]
            
            # Track peak and low prices for MFE/MAE calculation
            peak_price = position_data.get("peak_price", original_entry_price)
            low_price = position_data.get("low_price", original_entry_price)
            
            if current_price > peak_price:
                peak_price = current_price
                position_data["peak_price"] = peak_price
            if current_price < low_price:
                low_price = current_price
                position_data["low_price"] = low_price
            
            # Update position
            position_data["recent_prices"] = recent_prices
            updated_positions[position_key] = position_data
        
        # Show partial sell info if applicable
        if isinstance(position_data, dict) and position_data.get("partial_sell_taken"):
            partial_pct = position_data.get("partial_sell_pct", 0)
            remaining_pct = 1.0 - partial_pct
            print(f"[{timestamp}] 📊 Partial sell: {partial_pct*100:.0f}% sold, {remaining_pct*100:.0f}% remaining")

        # Calculate take profit threshold (static or dynamic based on config)
        trade = _find_open_trade_by_address(token_address, chain_id)
        take_profit_threshold = _calculate_dynamic_take_profit_for_position(trade, config)
        print(f"💰 TP Threshold: {take_profit_threshold*100:.2f}%")
        print(f"🔍 [DEBUG] Take profit check: gain={gain*100:.2f}%, threshold={take_profit_threshold*100:.2f}%, condition={gain >= take_profit_threshold}")

        # Trailing stop logic (optional)
        dyn_stop = _apply_trailing_stop(trail_state, token_address, current_price)
        if dyn_stop:
            print(f"🧵 Trailing stop @ ${dyn_stop:.6f} (peak-based)")

        # AI Partial Take-Profit Manager
        partial_tp_handled_stop_loss = False  # Flag to track if Partial TP Manager already handled stop loss
        _ai_ptp_enabled = config.get('ENABLE_AI_PARTIAL_TP', False)
        _ai_ptp_min_profit = config.get('ai_partial_tp_min_profit', 0.10)
        if not _ai_ptp_enabled:
            log_info("monitor.ai_partial_tp_disabled",
                     f"⏭️ AI Partial TP Manager disabled by config for {symbol}")
        elif gain < _ai_ptp_min_profit:
            log_info("monitor.ai_partial_tp_gated",
                     f"🚫 AI Partial TP GATED for {symbol}: gain={gain*100:.2f}% < floor={_ai_ptp_min_profit*100:.1f}%",
                     context={"gain_pct": round(gain*100,2), "floor_pct": round(_ai_ptp_min_profit*100,1), "symbol": symbol})
        try:
            if not _ai_ptp_enabled or gain < _ai_ptp_min_profit:
                raise _SkipPartialTP()  # jump to except → no-op
            from src.ai.ai_partial_take_profit_manager import get_partial_tp_manager
            partial_tp_manager = get_partial_tp_manager()
            
            # Prepare position dict for partial TP manager
            position_dict = {
                "address": token_address,
                "token_address": token_address,
                "symbol": symbol,
                "entry_price": entry_price,
                "position_size_usd": position_data.get("position_size_usd", 0) if isinstance(position_data, dict) else 0,
                "chain_id": chain_id
            }
            
            # Get volatility score (simplified - can be enhanced)
            volatility_score = abs(gain) * 2.0  # Simple volatility proxy
            volatility_score = min(1.0, volatility_score)  # Cap at 1.0
            
            # Evaluate partial TP actions
            partial_tp_actions = partial_tp_manager.evaluate_and_manage(
                position_dict,
                current_price,
                volatility_score
            )
            
            # Execute partial TP actions
            for action in partial_tp_actions:
                if action.type == "sell" and action.size_pct > 0:
                    # Partial sell
                    sell_amount_usd = (position_data.get("position_size_usd", 0) if isinstance(position_data, dict) else 0) * action.size_pct
                    print(f"📊 [PARTIAL TP] Executing partial sell: {action.size_pct*100:.0f}% ({action.reason})")
                    
                    # For partial sells, we need to sell a percentage of the position
                    # This is a simplified implementation - full implementation would need
                    # to track remaining position size and handle partial sells properly
                    if action.size_pct >= 1.0:
                        # Full sell - use existing logic
                        # CRITICAL: Check balance BEFORE attempting to sell - if already sold, clean up position
                        print(f"🔍 [PARTIAL TP PRE-SELL CHECK] Verifying token balance before sell...")
                        pre_sell_balance = _check_token_balance_on_chain(token_address, chain_id)
                        
                        if pre_sell_balance == -1.0:
                            # Balance check failed - proceed with sell attempt (can't verify)
                            print(f"⚠️ [PARTIAL TP PRE-SELL CHECK] Balance check failed, proceeding with sell attempt...")
                            tx = _sell_token_multi_chain(token_address, chain_id, symbol)
                        elif pre_sell_balance <= 0.0 or pre_sell_balance < 0.000001:
                            # Token already sold (zero/dust balance) - clean up position immediately
                            print(f"✅ [PARTIAL TP PRE-SELL CHECK] Token balance is zero/dust ({pre_sell_balance:.8f}) - position already sold, cleaning up...")
                            tx = None  # No sell needed, already sold
                            
                            # Log the trade exit
                            _log_trade_exit(token_address, symbol, original_entry_price, current_price, action.reason,
                                          position_data=position_data if isinstance(position_data, dict) else None)
                            
                            # Update performance tracker
                            if trade:
                                position_size = trade.get('position_size_usd', 0)
                                pnl_usd = gain * position_size
                                from src.core.performance_tracker import performance_tracker
                                performance_tracker.log_trade_exit(trade['id'], current_price, pnl_usd, action.reason)
                                print(f"📊 Updated performance tracker for {action.reason}: {trade.get('symbol', '?')}")
                            
                            # Clean up position
                            cleanup_success = _cleanup_closed_position(position_key, token_address, chain_id)
                            if not cleanup_success:
                                cleanup_success = _verify_position_cleanup(position_key, token_address, chain_id)
                            
                            if cleanup_success:
                                # Re-fetch current price right before notification
                                fresh_current_price = _fetch_token_price_multi_chain(token_address) or current_price
                                if fresh_current_price != current_price:
                                    print(f"📊 [PRICE UPDATE] Re-fetched price: ${current_price:.6f} -> ${fresh_current_price:.6f}")
                                    current_price = fresh_current_price
                                    gain = (current_price - original_entry_price) / original_entry_price
                                
                                # Send notification
                                tx_display = "Already sold (detected by balance check)"
                                send_telegram_message(
                                    f"🛑 Stop-loss triggered (already sold)!\n"
                                    f"Token: {symbol} ({token_address})\n"
                                    f"Chain: {chain_id.upper()}\n"
                                    f"Entry: ${original_entry_price:.6f}\n"
                                    f"Now: ${current_price:.6f} ({gain * 100:.2f}%)\n"
                                    f"TX: {tx_display}"
                                )
                                
                                closed_positions.append(position_key)
                                updated_positions.pop(position_key, None)
                            else:
                                print(f"❌ [PARTIAL TP] Failed to verify position cleanup - notification not sent")
                                # Still mark as closed to prevent retry loops
                                closed_positions.append(position_key)
                                updated_positions.pop(position_key, None)
                            continue  # Skip to next position
                        else:
                            # Token still in wallet - proceed with sell attempt
                            print(f"✅ [PARTIAL TP PRE-SELL CHECK] Token balance confirmed: {pre_sell_balance:.6f} - proceeding with sell...")
                            tx = _sell_token_multi_chain(token_address, chain_id, symbol)
                        if tx:
                            # Check if this is a hard stop loss action - only set flag if sell succeeded
                            if "hard_stop_loss" in action.reason.lower():
                                partial_tp_handled_stop_loss = True
                            
                            log_trade(token_address, entry_price, current_price, action.reason)
                            if trade:
                                position_size = trade.get('position_size_usd', 0)
                                pnl_usd = gain * position_size
                                from src.core.performance_tracker import performance_tracker
                                performance_tracker.log_trade_exit(trade['id'], current_price, pnl_usd, action.reason)
                            
                            # CRITICAL: Remove position from ALL storage locations (open_positions.json, hunter_state.db, performance_data.json)
                            cleanup_success = _cleanup_closed_position(position_key, token_address, chain_id)
                            
                            # Verify cleanup succeeded before sending notification
                            if not cleanup_success:
                                print(f"⚠️ [STOP-LOSS] Cleanup failed, verifying position removal...")
                                cleanup_success = _verify_position_cleanup(position_key, token_address, chain_id)
                            
                            if cleanup_success:
                                # Re-fetch current price right before notification to ensure accuracy
                                fresh_current_price = _fetch_token_price_multi_chain(token_address) or current_price
                                if fresh_current_price != current_price:
                                    print(f"📊 [PRICE UPDATE] Re-fetched price: ${current_price:.6f} -> ${fresh_current_price:.6f}")
                                    current_price = fresh_current_price
                                    # Recalculate gain with fresh price
                                    gain = (current_price - original_entry_price) / original_entry_price
                                
                                # Send Telegram notification
                                tx_display = tx if tx and tx not in ["verified_by_balance", "assumed_success"] else "Verified by balance check"
                                send_telegram_message(
                                    f"🛑 Stop-loss triggered!\n"
                                    f"Token: {symbol} ({token_address})\n"
                                    f"Chain: {chain_id.upper()}\n"
                                    f"Entry: ${original_entry_price:.6f}\n"
                                    f"Now: ${current_price:.6f} ({gain * 100:.2f}%)\n"
                                    f"TX: {tx_display}"
                                )
                                
                                closed_positions.append(position_key)
                                updated_positions.pop(position_key, None)
                            else:
                                print(f"❌ [STOP-LOSS] Failed to verify position cleanup - notification not sent")
                                # Still mark as closed to prevent retry loops
                                closed_positions.append(position_key)
                                updated_positions.pop(position_key, None)
                            continue
                        else:
                            # CRITICAL FIX: Sell failed - do NOT set partial_tp_handled_stop_loss flag
                            # This allows the main stop loss check to catch it as a fallback
                            print(f"❌ [PARTIAL TP] Hard stop loss sell FAILED for {symbol} ({action.reason})")
                            print(f"⚠️ [PARTIAL TP] Sell returned None/False - will fall through to main stop loss check")
                            print(f"📊 [PARTIAL TP] Current PnL: {gain * 100:.2f}%, Entry: ${entry_price:.6f}, Current: ${current_price:.6f}")
                            log_info("partial_tp.hard_stop_failed",
                                    symbol=symbol,
                                    pnl_pct=gain,
                                    entry_price=entry_price,
                                    current_price=current_price,
                                    reason=action.reason)
                            # Do NOT set partial_tp_handled_stop_loss = True here
                            # Do NOT continue - let it fall through to main stop loss check
                            # The main stop loss check will handle the retry
                    else:
                        # Partial sell - execute the sell and update position
                        print(f"📊 [PARTIAL TP] Executing partial sell: {action.size_pct*100:.0f}% of position ({action.reason})")
                        
                        # Calculate the USD amount to sell
                        original_position_size = position_data.get("position_size_usd", 0) if isinstance(position_data, dict) else 0
                        if original_position_size <= 0:
                            print(f"⚠️ [PARTIAL TP] Cannot determine position size, skipping partial sell")
                            continue
                        
                        # Execute partial sell
                        tx = _sell_token_multi_chain(token_address, chain_id, symbol, sell_amount_usd)
                        
                        if tx:
                            print(f"✅ [PARTIAL TP] Partial sell successful: {action.size_pct*100:.0f}% sold (TX: {tx})")
                            
                            # Calculate remaining position size
                            remaining_size_pct = 1.0 - action.size_pct
                            remaining_position_size_usd = original_position_size * remaining_size_pct
                            
                            # Check if this is selling the remaining position (trailing stop after partial TP)
                            # This happens when: 1) action is from trailing stop, OR 2) position already had partial sell
                            is_remaining_sell = (
                                "trailing_stop" in action.reason.lower() or
                                (isinstance(position_data, dict) and position_data.get("partial_sell_taken", False))
                            )
                            
                            # Calculate realized profit from partial sell
                            sold_usd = original_position_size * action.size_pct
                            realized_pnl_usd = gain * sold_usd
                            
                            # Determine log reason
                            if is_remaining_sell:
                                log_reason = f"remaining_sell_{action.size_pct:.0%}"
                            else:
                                log_reason = f"partial_tp_{action.size_pct:.0%}"
                            
                            # Log partial sell to trade log - ALWAYS log when transaction succeeds
                            log_trade(token_address, entry_price, current_price, log_reason)
                            
                            if is_remaining_sell:
                                # This is the final sell - close the position
                                # Re-fetch current price right before notification
                                fresh_current_price = _fetch_token_price_multi_chain(token_address) or current_price
                                if fresh_current_price != current_price:
                                    print(f"📊 [PRICE UPDATE] Re-fetched price: ${current_price:.6f} -> ${fresh_current_price:.6f}")
                                    current_price = fresh_current_price
                                    gain = (current_price - original_entry_price) / original_entry_price
                                    realized_pnl_usd = gain * sold_usd
                                
                                # Update performance tracker and close position
                                if trade:
                                    position_size = trade.get('position_size_usd', 0)
                                    pnl_usd = gain * position_size
                                    from src.core.performance_tracker import performance_tracker
                                    performance_tracker.log_trade_exit(trade['id'], current_price, pnl_usd, log_reason)
                                
                                cleanup_success = _cleanup_closed_position(position_key, token_address, chain_id)
                                if not cleanup_success:
                                    cleanup_success = _verify_position_cleanup(position_key, token_address, chain_id)
                                
                                if cleanup_success:
                                    send_telegram_message(
                                        f"✅ Position Fully Closed (Remaining Sell)!\n"
                                        f"Token: {symbol} ({token_address[:8]}...{token_address[-8:]})\n"
                                        f"Chain: {chain_id.upper()}\n"
                                        f"Entry: ${original_entry_price:.6f}\n"
                                        f"Exit: ${current_price:.6f} (+{gain * 100:.2f}%)\n"
                                        f"Final Sell: {action.size_pct*100:.0f}% (${sold_usd:.2f} USD)\n"
                                        f"Realized PnL: ${realized_pnl_usd:.2f}\n"
                                        f"TX: {tx}",
                                        message_type="trade"
                                    )
                                
                                closed_positions.append(position_key)
                                updated_positions.pop(position_key, None)
                                continue  # Move to next position
                            else:
                                # Partial TP - keep position open
                                # Re-fetch current price right before notification
                                fresh_current_price = _fetch_token_price_multi_chain(token_address) or current_price
                                if fresh_current_price != current_price:
                                    print(f"📊 [PRICE UPDATE] Re-fetched price: ${current_price:.6f} -> ${fresh_current_price:.6f}")
                                    current_price = fresh_current_price
                                    gain = (current_price - original_entry_price) / original_entry_price
                                    realized_pnl_usd = gain * sold_usd
                                
                                send_telegram_message(
                                    f"💵 Partial Take-Profit Executed!\n"
                                    f"Token: {symbol} ({token_address[:8]}...{token_address[-8:]})\n"
                                    f"Chain: {chain_id.upper()}\n"
                                    f"Entry: ${original_entry_price:.6f}\n"
                                    f"Exit: ${current_price:.6f} (+{gain * 100:.2f}%)\n"
                                    f"Sold: {action.size_pct*100:.0f}% (${sold_usd:.2f} USD)\n"
                                    f"Remaining: {remaining_size_pct*100:.0f}% (${remaining_position_size_usd:.2f} USD)\n"
                                    f"Realized PnL: ${realized_pnl_usd:.2f}\n"
                                    f"TX: {tx}",
                                    message_type="trade"
                                )
                                
                                # Update position data with remaining size
                                if isinstance(position_data, dict):
                                    # Store original entry price if not already stored
                                    if "original_entry_price" not in position_data:
                                        position_data["original_entry_price"] = entry_price
                                    
                                    # Update position size to remaining amount
                                    position_data["position_size_usd"] = remaining_position_size_usd
                                    position_data["partial_sell_taken"] = True
                                    position_data["partial_sell_pct"] = action.size_pct
                                    position_data["partial_sell_tx"] = tx
                                    position_data["partial_sell_price"] = current_price
                                    position_data["partial_sell_time"] = _dt.now().isoformat()
                        else:
                            # Transaction verification failed, but sell may have succeeded on-chain
                            # Log a warning and attempt to verify via balance check as fallback
                            print(f"⚠️ [PARTIAL TP] Sell verification failed (tx=None), checking balance as fallback...")
                            
                            # Wait a bit for transaction to propagate
                            import time
                            time.sleep(5)
                            
                            # Check if balance decreased (indicating successful sell)
                            try:
                                from src.execution.jupiter_lib import JupiterCustomLib
                                from src.config.secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY
                                if chain_id == "solana":
                                    lib = JupiterCustomLib(SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY)
                                    balance_after = lib.get_token_balance(token_address)
                                    
                                    # Get original balance from position data if available
                                    original_balance = None
                                    if isinstance(position_data, dict):
                                        # Try to estimate original balance from position size and price
                                        if entry_price > 0:
                                            original_balance_estimate = original_position_size / entry_price
                                            # Check if balance decreased significantly (at least 10% for partial sell)
                                            if balance_after is not None:
                                                expected_balance_after = original_balance_estimate * (1.0 - action.size_pct)
                                                
                                                # Check if this is selling the remaining position
                                                is_remaining_sell = (
                                                    "trailing_stop" in action.reason.lower() or
                                                    (isinstance(position_data, dict) and position_data.get("partial_sell_taken", False))
                                                )
                                                
                                                # For remaining sells, be more strict - balance should be near zero
                                                if is_remaining_sell:
                                                    # For remaining sells, balance should be essentially zero (dust only)
                                                    DUST_THRESHOLD = 0.000001  # Very small dust threshold
                                                    if balance_after <= DUST_THRESHOLD:
                                                        print(f"✅ [PARTIAL TP] Remaining sell confirmed - balance is zero/dust ({balance_after:.8f})")
                                                        balance_check_passed = True
                                                    else:
                                                        # Balance still exists - sell may have failed or been partial
                                                        print(f"⚠️ [PARTIAL TP] Remaining sell verification failed - balance still exists: {balance_after:.8f} (expected: ~0)")
                                                        balance_check_passed = False
                                                        
                                                        # Track failed attempt to prevent infinite retries
                                                        if isinstance(position_data, dict):
                                                            failed_remaining_sell_attempts = position_data.get("failed_remaining_sell_attempts", 0) + 1
                                                            position_data["failed_remaining_sell_attempts"] = failed_remaining_sell_attempts
                                                            position_data["last_remaining_sell_attempt"] = _dt.now().isoformat()
                                                            
                                                            # If we've tried multiple times, clear partial TP state to prevent loops
                                                            if failed_remaining_sell_attempts >= 3:
                                                                print(f"⚠️ [PARTIAL TP] Multiple failed remaining sell attempts ({failed_remaining_sell_attempts}) - clearing partial TP state to prevent loops")
                                                                position_data["partial_sell_taken"] = False
                                                                position_data.pop("failed_remaining_sell_attempts", None)
                                                                position_data.pop("last_remaining_sell_attempt", None)
                                                                # Clear partial TP manager state
                                                                try:
                                                                    from src.ai.ai_partial_take_profit_manager import get_partial_tp_manager
                                                                    partial_tp_manager = get_partial_tp_manager()
                                                                    # Use the same key format as partial TP manager
                                                                    tp_position_key = f"{chain_id.lower()}:{token_address.lower()}"
                                                                    partial_tp_manager.clear_position_state(tp_position_key)
                                                                    print(f"✅ Cleared partial TP state for {tp_position_key}")
                                                                except Exception as e:
                                                                    print(f"⚠️ Error clearing partial TP state: {e}")
                                                                    import traceback
                                                                    traceback.print_exc()
                                                                from src.storage.positions import upsert_position
                                                                upsert_position(position_key, position_data)
                                                else:
                                                    # For partial sells (not remaining), use tolerance check
                                                    # Allow some tolerance for price changes and rounding
                                                    if abs(balance_after - expected_balance_after) / max(expected_balance_after, 0.000001) < 0.2:
                                                        print(f"✅ [PARTIAL TP] Balance check suggests partial sell succeeded (balance: {balance_after:.8f}, expected: {expected_balance_after:.8f})")
                                                        balance_check_passed = True
                                                    else:
                                                        print(f"⚠️ [PARTIAL TP] Balance check failed - balance mismatch (balance: {balance_after:.8f}, expected: {expected_balance_after:.8f})")
                                                        balance_check_passed = False
                                                
                                                if balance_check_passed:
                                                    # Determine log reason
                                                    if is_remaining_sell:
                                                        log_reason = f"remaining_sell_{action.size_pct:.0%}_unverified"
                                                    else:
                                                        log_reason = f"partial_tp_{action.size_pct:.0%}_unverified"
                                                    
                                                    # Log the trade even though we don't have tx hash
                                                    log_trade(token_address, entry_price, current_price, log_reason)
                                                    
                                                    # Update position data
                                                    remaining_size_pct = 1.0 - action.size_pct
                                                    remaining_position_size_usd = original_position_size * remaining_size_pct
                                                    
                                                    # Calculate realized profit from partial sell
                                                    sold_usd = original_position_size * action.size_pct
                                                    realized_pnl_usd = gain * sold_usd
                                                    
                                                    if is_remaining_sell:
                                                        # This is the final sell - close the position
                                                        # Re-fetch current price right before notification
                                                        fresh_current_price = _fetch_token_price_multi_chain(token_address) or current_price
                                                        if fresh_current_price != current_price:
                                                            print(f"📊 [PRICE UPDATE] Re-fetched price: ${current_price:.6f} -> ${fresh_current_price:.6f}")
                                                            current_price = fresh_current_price
                                                            gain = (current_price - original_entry_price) / original_entry_price
                                                            realized_pnl_usd = gain * sold_usd
                                                        
                                                        send_telegram_message(
                                                            f"✅ Position Fully Closed (Remaining Sell - Unverified)!\n"
                                                            f"Token: {symbol} ({token_address[:8]}...{token_address[-8:]})\n"
                                                            f"Chain: {chain_id.upper()}\n"
                                                            f"Entry: ${original_entry_price:.6f}\n"
                                                            f"Exit: ${current_price:.6f} (+{gain * 100:.2f}%)\n"
                                                            f"Final Sell: {action.size_pct*100:.0f}% (${sold_usd:.2f} USD)\n"
                                                            f"Realized PnL: ${realized_pnl_usd:.2f}\n"
                                                            f"TX: Unverified (balance check suggests success)",
                                                            message_type="trade"
                                                        )
                                                        
                                                        # Update performance tracker and close position
                                                        trade = _find_open_trade_by_address(token_address, chain_id)
                                                        if trade:
                                                            position_size = trade.get('position_size_usd', 0)
                                                            pnl_usd = gain * position_size
                                                            from src.core.performance_tracker import performance_tracker
                                                            performance_tracker.log_trade_exit(trade['id'], current_price, pnl_usd, log_reason)
                                                        
                                                        # Clear partial TP state BEFORE cleanup (using correct key format)
                                                        try:
                                                            from src.ai.ai_partial_take_profit_manager import get_partial_tp_manager
                                                            partial_tp_manager = get_partial_tp_manager()
                                                            # Use the same key format as partial TP manager expects
                                                            tp_position_key = f"{chain_id.lower()}:{token_address.lower()}"
                                                            partial_tp_manager.clear_position_state(tp_position_key)
                                                            print(f"✅ Cleared partial TP state for {tp_position_key}")
                                                        except Exception as e:
                                                            print(f"⚠️ Error clearing partial TP state: {e}")
                                                            import traceback
                                                            traceback.print_exc()
                                                        
                                                        _cleanup_closed_position(position_key, token_address, chain_id)
                                                        closed_positions.append(position_key)
                                                        updated_positions.pop(position_key, None)
                                                        print(f"✅ [PARTIAL TP] Position closed after remaining sell")
                                                        continue  # Move to next position
                                                    else:
                                                        # Partial TP - keep position open
                                                        # Re-fetch current price right before notification
                                                        fresh_current_price = _fetch_token_price_multi_chain(token_address) or current_price
                                                        if fresh_current_price != current_price:
                                                            print(f"📊 [PRICE UPDATE] Re-fetched price: ${current_price:.6f} -> ${fresh_current_price:.6f}")
                                                            current_price = fresh_current_price
                                                            gain = (current_price - original_entry_price) / original_entry_price
                                                            realized_pnl_usd = gain * sold_usd
                                                        
                                                        send_telegram_message(
                                                            f"💵 Partial Take-Profit Executed (Unverified)!\n"
                                                            f"Token: {symbol} ({token_address[:8]}...{token_address[-8:]})\n"
                                                            f"Chain: {chain_id.upper()}\n"
                                                            f"Entry: ${original_entry_price:.6f}\n"
                                                            f"Exit: ${current_price:.6f} (+{gain * 100:.2f}%)\n"
                                                            f"Sold: {action.size_pct*100:.0f}% (${sold_usd:.2f} USD)\n"
                                                            f"Remaining: {remaining_size_pct*100:.0f}% (${remaining_position_size_usd:.2f} USD)\n"
                                                            f"Realized PnL: ${realized_pnl_usd:.2f}\n"
                                                            f"TX: Unverified (balance check suggests success)",
                                                            message_type="trade"
                                                        )
                                                        
                                                        if isinstance(position_data, dict):
                                                            if "original_entry_price" not in position_data:
                                                                position_data["original_entry_price"] = entry_price
                                                            position_data["position_size_usd"] = remaining_position_size_usd
                                                            position_data["partial_sell_taken"] = True
                                                            position_data["partial_sell_pct"] = action.size_pct
                                                            position_data["partial_sell_tx"] = None  # No tx hash available
                                                            position_data["partial_sell_price"] = current_price
                                                            position_data["partial_sell_time"] = _dt.now().isoformat()
                                                            # Clear any failed attempt counters
                                                            position_data.pop("failed_remaining_sell_attempts", None)
                                                            position_data.pop("last_remaining_sell_attempt", None)
                                                            updated_positions[position_key] = position_data
                                                            from src.storage.positions import upsert_position
                                                            upsert_position(position_key, position_data)
                                                        print(f"✅ [PARTIAL TP] Trade logged as unverified - transaction may have succeeded but verification failed")
                                                        continue
                                                else:
                                                    # Balance check failed
                                                    print(f"❌ [PARTIAL TP] Balance check failed - sell may not have succeeded")
                                                    if is_remaining_sell:
                                                        # For remaining sells, don't retry immediately - wait for next cycle
                                                        print(f"⚠️ [PARTIAL TP] Remaining sell verification failed - will retry on next cycle if conditions still met")
                                                    else:
                                                        print(f"⚠️ [PARTIAL TP] Partial sell verification failed - will retry on next cycle")
                                            else:
                                                print(f"⚠️ [PARTIAL TP] Could not get balance after sell")
                                        else:
                                            print(f"⚠️ [PARTIAL TP] Invalid entry price, cannot verify balance")
                                    else:
                                        print(f"⚠️ [PARTIAL TP] Position data not in expected format")
                            except Exception as balance_check_error:
                                print(f"⚠️ [PARTIAL TP] Balance check also failed: {balance_check_error}")
                            
                            print(f"❌ [PARTIAL TP] Partial sell failed or could not be verified - will retry on next cycle")
                            log_info("partial_tp.failed",
                                    symbol=symbol,
                                    size_pct=action.size_pct,
                                    reason=action.reason)
                
                elif action.type == "move_stop" and action.new_stop_price:
                    # Update trailing stop
                    print(f"📊 [PARTIAL TP] Moving stop to ${action.new_stop_price:.6f} ({action.reason})")
                    trail_state[f"{token_address}_peak"] = current_price
                    trail_state[f"{token_address}_stop"] = action.new_stop_price
        except _SkipPartialTP:
            pass  # AI partial TP disabled or gated by profit floor
        except Exception as e:
            print(f"⚠️ Partial TP manager error: {e}")
            import traceback
            traceback.print_exc()

        # Technical Indicator-Based Exit Signals
        # Gate: skip entirely if disabled OR if gain below profit floor
        technical_exit_reason = None
        if get_config_bool("enable_technical_exit_signals", False) and \
                _soft_exit_gated(gain, "technical_exit", "technical_exit_min_profit_for_signal", config):
            technical_exit_reason = _check_technical_exit_signals(
                position_dict if isinstance(position_data, dict) else {},
                current_price,
                gain,
                token_address,
                chain_id,
                config
            )
        
        if technical_exit_reason:
            print(f"📊 [TECHNICAL EXIT] Signal detected: {technical_exit_reason}")
            print(f"💰 Current gain: {gain*100:.2f}%")
            
            # Check balance before selling
            print(f"🔍 [PRE-SELL CHECK] Verifying token balance before technical exit sell...")
            pre_sell_balance = _check_token_balance_on_chain(token_address, chain_id)
            
            if pre_sell_balance == -1.0:
                print(f"⚠️ [PRE-SELL CHECK] Balance check failed, proceeding with sell attempt...")
            elif pre_sell_balance <= 0.0 or pre_sell_balance < 0.000001:
                print(f"✅ [PRE-SELL CHECK] Token already sold, cleaning up...")
                _log_trade_exit(token_address, symbol, entry_price, current_price, f"technical_exit_{technical_exit_reason}",
                              position_data=position_data if isinstance(position_data, dict) else None, tx_hash=tx)
                if trade:
                    position_size = trade.get('position_size_usd', 0)
                    pnl_usd = gain * position_size
                    from src.core.performance_tracker import performance_tracker
                    performance_tracker.log_trade_exit(trade['id'], current_price, pnl_usd, f"technical_exit_{technical_exit_reason}")
                _cleanup_closed_position(position_key, token_address, chain_id)
                closed_positions.append(position_key)
                updated_positions.pop(position_key, None)
                continue
            else:
                print(f"✅ [PRE-SELL CHECK] Token balance confirmed: {pre_sell_balance:.6f} - proceeding with sell...")
            
            # Execute sell
            tx = _sell_token_multi_chain(token_address, chain_id, symbol)
            if tx:
                # Wait for transaction to propagate, then verify with Helius
                print(f"⏳ Waiting 30 seconds for transaction to propagate before verification...")
                time.sleep(30)
                
                # Run Helius reconciliation to verify sell (for Solana)
                sell_verified = True
                if chain_id.lower() == "solana":
                    try:
                        from src.utils.helius_client import HeliusClient
                        from src.config.secrets import HELIUS_API_KEY, SOLANA_WALLET_ADDRESS
                        
                        if HELIUS_API_KEY and SOLANA_WALLET_ADDRESS:
                            print(f"🔍 Verifying sell with Helius reconciliation...")
                            client = HeliusClient(HELIUS_API_KEY)
                            balances = client.get_address_balances(SOLANA_WALLET_ADDRESS)
                            tokens = balances.get("tokens", [])
                            
                            # Check if token is still in wallet
                            token_still_present = False
                            for token in tokens:
                                mint = (token.get("mint") or "").lower()
                                if mint == token_address.lower():
                                    amount_raw = token.get("amount") or 0
                                    decimals = token.get("decimals") or 0
                                    amount = float(amount_raw) / (10 ** decimals) if decimals else float(amount_raw)
                                    if amount > 1e-9:  # Still has balance
                                        token_still_present = True
                                        print(f"❌ Sell verification failed: {symbol} still in wallet ({amount:.6f} tokens)")
                                        break
                            
                            if token_still_present:
                                print(f"🔄 Sell failed - token still in wallet. Will retry on next cycle.")
                                sell_verified = False
                            else:
                                print(f"✅ Sell verified: {symbol} no longer in wallet")
                    except Exception as e:
                        print(f"⚠️ Helius verification error (proceeding anyway): {e}")
                        import traceback
                        traceback.print_exc()
                
                if not sell_verified:
                    # Don't proceed with cleanup, mark for retry
                    print(f"❌ [TECHNICAL EXIT] Sell verification failed, will retry on next cycle")
                    continue
                
                _log_trade_exit(token_address, symbol, entry_price, current_price, f"technical_exit_{technical_exit_reason}",
                              position_data=position_data if isinstance(position_data, dict) else None, tx_hash=tx)
                if trade:
                    position_size = trade.get('position_size_usd', 0)
                    pnl_usd = gain * position_size
                    from src.core.performance_tracker import performance_tracker
                    performance_tracker.log_trade_exit(trade['id'], current_price, pnl_usd, f"technical_exit_{technical_exit_reason}")
                
                # Cleanup with verification
                cleanup_success = _cleanup_closed_position(position_key, token_address, chain_id)
                if not cleanup_success:
                    print(f"⚠️ Cleanup failed for {symbol}, retrying with address lookup...")
                    from src.utils.position_sync import find_position_key_by_address
                    found_key = find_position_key_by_address(token_address, chain_id)
                    if found_key:
                        cleanup_success = _cleanup_closed_position(found_key, token_address, chain_id)
                    
                    if not cleanup_success:
                        print(f"❌ CRITICAL: Failed to remove {symbol} from open_positions.json after sell!")
                
                send_telegram_message(
                    f"📊 Technical Exit Signal Triggered!\n"
                    f"Token: {symbol} ({token_address[:8]}...{token_address[-8:]})\n"
                    f"Chain: {chain_id.upper()}\n"
                    f"Signal: {technical_exit_reason}\n"
                    f"Entry: ${entry_price:.6f}\n"
                    f"Exit: ${current_price:.6f} (+{gain * 100:.2f}%)\n"
                    f"TX: {tx}"
                )
                closed_positions.append(position_key)
                updated_positions.pop(position_key, None)
                continue
            else:
                print(f"❌ [TECHNICAL EXIT] Sell failed, will retry on next cycle")

        # Volume Deterioration Check (soft exit — gated by profit floor)
        volume_exit_reason = None
        if _soft_exit_gated(gain, "volume_deterioration", "volume_deterioration_min_profit", config):
            volume_exit_reason = _check_volume_deterioration(
                position_data if isinstance(position_data, dict) else {},
                current_price,
                gain,
                token_address,
                chain_id,
                config
            )
        
        if volume_exit_reason:
            print(f"📉 [VOLUME EXIT] Volume deteriorated: {volume_exit_reason}")
            print(f"💰 Current gain: {gain*100:.2f}%")
            
            # Check balance before selling
            print(f"🔍 [PRE-SELL CHECK] Verifying token balance before volume exit sell...")
            pre_sell_balance = _check_token_balance_on_chain(token_address, chain_id)
            
            if pre_sell_balance == -1.0:
                print(f"⚠️ [PRE-SELL CHECK] Balance check failed, proceeding with sell attempt...")
            elif pre_sell_balance <= 0.0 or pre_sell_balance < 0.000001:
                print(f"✅ [PRE-SELL CHECK] Token already sold, cleaning up...")
                _log_trade_exit(token_address, symbol, entry_price, current_price, f"volume_exit_{volume_exit_reason}",
                              position_data=position_data if isinstance(position_data, dict) else None, tx_hash=tx)
                if trade:
                    position_size = trade.get('position_size_usd', 0)
                    pnl_usd = gain * position_size
                    from src.core.performance_tracker import performance_tracker
                    performance_tracker.log_trade_exit(trade['id'], current_price, pnl_usd, f"volume_exit_{volume_exit_reason}")
                _cleanup_closed_position(position_key, token_address, chain_id)
                closed_positions.append(position_key)
                updated_positions.pop(position_key, None)
                continue
            else:
                print(f"✅ [PRE-SELL CHECK] Token balance confirmed: {pre_sell_balance:.6f} - proceeding with sell...")
            
            # Execute sell
            tx = _sell_token_multi_chain(token_address, chain_id, symbol)
            if tx:
                _log_trade_exit(token_address, symbol, entry_price, current_price, f"volume_exit_{volume_exit_reason}",
                              position_data=position_data if isinstance(position_data, dict) else None, tx_hash=tx)
                if trade:
                    position_size = trade.get('position_size_usd', 0)
                    pnl_usd = gain * position_size
                    from src.core.performance_tracker import performance_tracker
                    performance_tracker.log_trade_exit(trade['id'], current_price, pnl_usd, f"volume_exit_{volume_exit_reason}")
                
                _cleanup_closed_position(position_key, token_address, chain_id)
                
                send_telegram_message(
                    f"📉 Volume Deterioration Exit!\n"
                    f"Token: {symbol} ({token_address[:8]}...{token_address[-8:]})\n"
                    f"Chain: {chain_id.upper()}\n"
                    f"Reason: {volume_exit_reason}\n"
                    f"Entry: ${entry_price:.6f}\n"
                    f"Exit: ${current_price:.6f} (+{gain * 100:.2f}%)\n"
                    f"TX: {tx}"
                )
                closed_positions.append(position_key)
                updated_positions.pop(position_key, None)
                continue
            else:
                print(f"❌ [VOLUME EXIT] Sell failed, will retry on next cycle")

        # Take-profit (full position)
        if gain >= take_profit_threshold:
            print(f"💰 [TAKE PROFIT] Take-profit hit! Gain: {gain*100:.2f}% >= Threshold: {take_profit_threshold*100:.2f}%")
            
            # Check balance BEFORE attempting to sell - if already sold, clean up position
            print(f"🔍 [PRE-SELL CHECK] Verifying token balance before take-profit sell...")
            pre_sell_balance = _check_token_balance_on_chain(token_address, chain_id)
            
            if pre_sell_balance == -1.0:
                # Balance check failed - proceed with sell attempt (can't verify)
                print(f"⚠️ [PRE-SELL CHECK] Balance check failed, proceeding with sell attempt...")
            elif pre_sell_balance <= 0.0 or pre_sell_balance < 0.000001:
                # Token already sold (zero/dust balance) - clean up position
                print(f"✅ [PRE-SELL CHECK] Token balance is zero/dust ({pre_sell_balance:.8f}) - position already sold, cleaning up...")
                
                # Log the trade exit
                _log_trade_exit(token_address, symbol, entry_price, current_price, "take_profit",
                              position_data=position_data if isinstance(position_data, dict) else None, tx_hash=tx)
                
                # Update performance tracker
                trade = _find_open_trade_by_address(token_address, chain_id)
                if trade:
                    position_size = trade.get('position_size_usd', 0)
                    pnl_usd = gain * position_size
                    from src.core.performance_tracker import performance_tracker
                    performance_tracker.log_trade_exit(trade['id'], current_price, pnl_usd, "take_profit")
                    print(f"📊 Updated performance tracker for take profit: {trade.get('symbol', '?')}")
                
                # Clean up position
                _cleanup_closed_position(position_key, token_address, chain_id)
                
                # Send notification
                send_telegram_message(
                    f"💰 Take-profit triggered (already sold)!\n"
                    f"Token: {symbol} ({token_address})\n"
                    f"Chain: {chain_id.upper()}\n"
                    f"Entry: ${entry_price:.6f}\n"
                    f"Exit: ${current_price:.6f} (+{gain * 100:.2f}%)\n"
                    f"TX: Already sold (detected by balance check)"
                )
                closed_positions.append(position_key)
                updated_positions.pop(position_key, None)
                continue  # Skip to next position
            else:
                # Token still in wallet - proceed with sell attempt
                print(f"✅ [PRE-SELL CHECK] Token balance confirmed: {pre_sell_balance:.6f} - proceeding with sell...")
            
            print(f"🔍 [DEBUG] Calling _sell_token_multi_chain for {symbol} ({token_address}) on {chain_id}")
            tx = _sell_token_multi_chain(token_address, chain_id, symbol)
            print(f"🔍 [DEBUG] Sell result: tx={tx}, type={type(tx)}")
            
            if tx:  # Only proceed if sell succeeded
                # Wait for transaction to propagate, then verify with Helius
                print(f"⏳ Waiting 30 seconds for transaction to propagate before verification...")
                time.sleep(30)
                
                # Run Helius reconciliation to verify sell (for Solana)
                sell_verified = True
                if chain_id.lower() == "solana":
                    try:
                        from src.utils.helius_client import HeliusClient
                        from src.config.secrets import HELIUS_API_KEY, SOLANA_WALLET_ADDRESS
                        
                        if HELIUS_API_KEY and SOLANA_WALLET_ADDRESS:
                            print(f"🔍 Verifying sell with Helius reconciliation...")
                            client = HeliusClient(HELIUS_API_KEY)
                            balances = client.get_address_balances(SOLANA_WALLET_ADDRESS)
                            tokens = balances.get("tokens", [])
                            
                            # Check if token is still in wallet
                            token_still_present = False
                            for token in tokens:
                                mint = (token.get("mint") or "").lower()
                                if mint == token_address.lower():
                                    amount_raw = token.get("amount") or 0
                                    decimals = token.get("decimals") or 0
                                    amount = float(amount_raw) / (10 ** decimals) if decimals else float(amount_raw)
                                    if amount > 1e-9:  # Still has balance
                                        token_still_present = True
                                        print(f"❌ Sell verification failed: {symbol} still in wallet ({amount:.6f} tokens)")
                                        break
                            
                            if token_still_present:
                                print(f"🔄 Sell failed - token still in wallet. Will retry on next cycle.")
                                sell_verified = False
                            else:
                                print(f"✅ Sell verified: {symbol} no longer in wallet")
                    except Exception as e:
                        print(f"⚠️ Helius verification error (proceeding anyway): {e}")
                        import traceback
                        traceback.print_exc()
                
                if not sell_verified:
                    # Don't proceed with cleanup, mark for retry
                    print(f"❌ [TAKE PROFIT] Sell verification failed, will retry on next cycle")
                    continue
                
                _log_trade_exit(token_address, symbol, entry_price, current_price, "take_profit",
                              position_data=position_data if isinstance(position_data, dict) else None, tx_hash=tx)
                
                # Update performance tracker with exit
                if trade:
                    position_size = trade.get('position_size_usd', 0)
                    pnl_usd = gain * position_size  # gain is already a ratio
                    
                    # Analyze sell transaction for fees
                    sell_fee_data = _analyze_sell_fees(tx, chain_id)
                    
                    # Calculate fee-adjusted PnL
                    if sell_fee_data and sell_fee_data.get('exit_gas_fee_usd') is not None:
                        buy_gas = trade.get('entry_gas_fee_usd', 0) or 0
                        sell_gas = sell_fee_data.get('exit_gas_fee_usd', 0)
                        total_fees = buy_gas + sell_gas
                        sell_fee_data['total_fees_usd'] = total_fees
                    
                    from src.core.performance_tracker import performance_tracker
                    performance_tracker.log_trade_exit(trade['id'], current_price, pnl_usd, "take_profit",
                                                       additional_data=sell_fee_data)
                    print(f"📊 Updated performance tracker for take profit: {trade.get('symbol', '?')}")
                
                # CRITICAL: Remove position from ALL storage locations (open_positions.json, hunter_state.db, performance_data.json)
                cleanup_success = _cleanup_closed_position(position_key, token_address, chain_id)
                if not cleanup_success:
                    print(f"⚠️ Cleanup failed for {symbol}, retrying with address lookup...")
                    from src.utils.position_sync import find_position_key_by_address
                    found_key = find_position_key_by_address(token_address, chain_id)
                    if found_key:
                        cleanup_success = _cleanup_closed_position(found_key, token_address, chain_id)
                    
                    if not cleanup_success:
                        print(f"❌ CRITICAL: Failed to remove {symbol} from open_positions.json after sell!")
                
                # Verify cleanup succeeded before sending notification
                if not cleanup_success:
                    cleanup_success = _verify_position_cleanup(position_key, token_address, chain_id)
                
                if cleanup_success:
                    # Re-fetch current price right before notification to ensure accuracy
                    fresh_current_price = _fetch_token_price_multi_chain(token_address) or current_price
                    if fresh_current_price != current_price:
                        print(f"📊 [PRICE UPDATE] Re-fetched price: ${current_price:.6f} -> ${fresh_current_price:.6f}")
                        current_price = fresh_current_price
                        gain = (current_price - original_entry_price) / original_entry_price
                    
                    tx_display = tx if tx and tx not in ["verified_by_balance", "assumed_success"] else "Verified by balance check"
                    send_telegram_message(
                        f"💰 Take-profit triggered!\n"
                        f"Token: {symbol} ({token_address})\n"
                        f"Chain: {chain_id.upper()}\n"
                        f"Entry: ${original_entry_price:.6f}\n"
                        f"Now: ${current_price:.6f} (+{gain * 100:.2f}%)\n"
                        f"TX: {tx_display}"
                    )
                else:
                    print(f"❌ [TAKE-PROFIT] Failed to verify position cleanup - notification not sent")
                
                closed_positions.append(position_key)
                updated_positions.pop(position_key, None)
            else:  # Sell failed - but verify with balance check first to avoid false negatives
                # Double-check if sell actually succeeded by checking balance
                # This prevents false "SELL FAILED" messages when verification is delayed
                balance_check_passed = False
                if chain_id == "solana":
                    try:
                        from src.execution.jupiter_lib import JupiterCustomLib
                        from src.config.secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY
                        lib = JupiterCustomLib(SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY)
                        time.sleep(3)  # Wait a bit for transaction to propagate
                        current_balance = lib.get_token_balance(token_address)
                        
                        # If balance is zero or very small, sell likely succeeded
                        if current_balance is not None:
                            if current_balance == 0 or current_balance < 0.000001:
                                print(f"✅ [BALANCE CHECK] Token balance is zero/dust - sell actually succeeded despite verification failure")
                                balance_check_passed = True
                    except Exception as balance_check_error:
                        print(f"⚠️ [BALANCE CHECK] Error checking balance: {balance_check_error}")
                
                if balance_check_passed:
                    # Sell actually succeeded - log it and remove position
                    print(f"✅ [RECOVERY] Sell succeeded (verified by balance check), logging trade...")
                    log_trade(token_address, original_entry_price, current_price, "take_profit")
                    
                    # Update performance tracker
                    if trade:
                        position_size = trade.get('position_size_usd', 0)
                        pnl_usd = gain * position_size
                        
                        from src.core.performance_tracker import performance_tracker
                        performance_tracker.log_trade_exit(trade['id'], current_price, pnl_usd, "take_profit")
                    
                    # Clean up position
                    cleanup_success = _cleanup_closed_position(position_key, token_address, chain_id)
                    if not cleanup_success:
                        cleanup_success = _verify_position_cleanup(position_key, token_address, chain_id)
                    
                    if cleanup_success:
                        # Re-fetch current price right before notification
                        fresh_current_price = _fetch_token_price_multi_chain(token_address) or current_price
                        if fresh_current_price != current_price:
                            print(f"📊 [PRICE UPDATE] Re-fetched price: ${current_price:.6f} -> ${fresh_current_price:.6f}")
                            current_price = fresh_current_price
                            gain = (current_price - original_entry_price) / original_entry_price
                        
                        send_telegram_message(
                            f"💰 Take-profit triggered!\n"
                            f"Token: {symbol} ({token_address})\n"
                            f"Chain: {chain_id.upper()}\n"
                            f"Entry: ${original_entry_price:.6f}\n"
                            f"Now: ${current_price:.6f} (+{gain * 100:.2f}%)\n"
                            f"TX: Verified by balance check"
                        )
                    else:
                        print(f"❌ [TAKE-PROFIT] Failed to verify position cleanup - notification not sent")
                    
                    closed_positions.append(position_key)
                    updated_positions.pop(position_key, None)
                else:
                    # Sell actually failed - send error message
                    # Re-fetch current price right before notification
                    fresh_current_price = _fetch_token_price_multi_chain(token_address) or current_price
                    if fresh_current_price != current_price:
                        print(f"📊 [PRICE UPDATE] Re-fetched price: ${current_price:.6f} -> ${fresh_current_price:.6f}")
                        current_price = fresh_current_price
                        gain = (current_price - original_entry_price) / original_entry_price
                    
                    send_telegram_message(
                        f"⚠️ Take-profit triggered but SELL FAILED!\n"
                        f"Token: {symbol} ({token_address})\n"
                        f"Chain: {chain_id.upper()}\n"
                        f"Entry: ${original_entry_price:.6f}\n"
                        f"Now: ${current_price:.6f} (+{gain * 100:.2f}%)\n"
                        f"Will retry on next check..."
                    )
                    # Don't remove position - keep monitoring to retry
            continue  # move to next token

        # UPGRADE #1: Momentum Decay Exit (strategy integrity - before stop-loss)
        # Soft exit — gated by profit floor
        momentum_decay_exit_reason = None
        if config.get('ENABLE_MOMENTUM_DECAY_EXIT', True) and isinstance(position_data, dict) \
                and _soft_exit_gated(gain, "momentum_decay", "momentum_decay_min_profit", config):
            try:
                entry_momentum_score = position_data.get("entry_momentum_score")
                entry_momentum_source = position_data.get("entry_momentum_source")
                
                # Skip if entry momentum not available (old positions or entry-time failure)
                if entry_momentum_score is None:
                    # Old position without momentum data - skip decay check (backward compatibility)
                    pass
                else:
                    # Get current momentum using same logic as entry
                    from src.core.strategy import _calculate_momentum_score
                    from src.config.config_loader import get_config_values
                    
                    # Build token dict for momentum calculation
                    current_token = {
                        "address": token_address,
                        "symbol": symbol,
                        "priceUsd": current_price,
                        "priceChange5m": None,  # Will fetch from DexScreener if needed
                        "priceChange1h": None,
                        "priceChange24h": None,
                        "candles_validated": False,
                        "candles_15m": None,
                        "technical_indicators": {},
                    }
                    
                    # Try to get current momentum from DexScreener
                    try:
                        import requests
                        dexscreener_url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
                        response = requests.get(dexscreener_url, timeout=10)
                        if response.status_code == 200:
                            data = response.json()
                            pairs = data.get("pairs", [])
                            if pairs:
                                pair = pairs[0]  # Use first pair
                                price_change = pair.get("priceChange", {})
                                current_token["priceChange5m"] = price_change.get("m5")
                                current_token["priceChange1h"] = price_change.get("h1")
                                current_token["priceChange24h"] = price_change.get("h24")
                    except Exception:
                        pass  # Continue without DexScreener data
                    
                    current_config = get_config_values()
                    current_momentum_score, current_source, _ = _calculate_momentum_score(current_token, current_config)
                    
                    if current_momentum_score is not None:
                        momentum_decay_threshold = config.get('momentum_decay_threshold', 0.005)
                        momentum_decay_delta = config.get('momentum_decay_delta', 0.01)
                        momentum_decay_consecutive = config.get('momentum_decay_consecutive_checks', 2)
                        
                        momentum_decay_amount = entry_momentum_score - current_momentum_score
                        decay_checks = position_data.get("momentum_decay_checks", 0)
                        last_decay_log = position_data.get("momentum_decay_last_log_check", 0)
                        current_time_ts = time.time()
                        
                        # Rate-limited momentum decay telemetry (log every 5 checks or when state changes)
                        should_log_telemetry = (
                            (decay_checks == 0 and momentum_decay_amount > momentum_decay_delta) or  # Decay started
                            (decay_checks > 0 and momentum_decay_amount <= momentum_decay_delta) or  # Decay recovered
                            (current_time_ts - last_decay_log > 300) or  # 5 minutes since last log
                            (decay_checks % 5 == 0 and decay_checks > 0)  # Every 5 consecutive checks
                        )
                        
                        if should_log_telemetry:
                            log_info("momentum_decay.check",
                                    f"Momentum decay check: entry={entry_momentum_score:.4f}, current={current_momentum_score:.4f}, decay={momentum_decay_amount:.4f}, consecutive={decay_checks}",
                                    context={
                                        "token": token_address,
                                        "symbol": symbol,
                                        "entry_momentum_score": entry_momentum_score,
                                        "current_momentum_score": current_momentum_score,
                                        "entry_momentum_source": entry_momentum_source,
                                        "current_momentum_source": current_source,
                                        "decay_delta": momentum_decay_amount,
                                        "decay_consecutive_count": decay_checks,
                                        "momentum_decay_threshold": momentum_decay_threshold,
                                        "momentum_decay_delta": momentum_decay_delta,
                                    })
                            position_data["momentum_decay_last_log_check"] = current_time_ts
                            updated_positions[position_key] = position_data
                        
                        # Check absolute threshold
                        if current_momentum_score < momentum_decay_threshold:
                            momentum_decay_exit_reason = f"momentum_below_threshold_{current_momentum_score:.4f}_vs_{momentum_decay_threshold:.4f}"
                        
                        # Check decay delta
                        if momentum_decay_amount > momentum_decay_delta:
                            # Track consecutive decay checks
                            decay_checks = position_data.get("momentum_decay_checks", 0) + 1
                            position_data["momentum_decay_checks"] = decay_checks
                            updated_positions[position_key] = position_data
                            
                            if decay_checks >= momentum_decay_consecutive:
                                momentum_decay_exit_reason = f"momentum_decay_{momentum_decay_amount:.4f}_over_{momentum_decay_consecutive}_checks"
                        else:
                            # Reset decay counter if momentum recovered
                            if position_data.get("momentum_decay_checks", 0) > 0:
                                position_data["momentum_decay_checks"] = 0
                                updated_positions[position_key] = position_data
                        
                        if momentum_decay_exit_reason:
                            # Log momentum decay exit trigger
                            trigger_reason = "below_threshold" if current_momentum_score < momentum_decay_threshold else "decay_delta_consecutive"
                            log_info("momentum_decay.trigger",
                                    f"Momentum decay exit triggered: {trigger_reason}",
                                    context={
                                        "token": token_address,
                                        "symbol": symbol,
                                        "entry_momentum_score": entry_momentum_score,
                                        "current_momentum_score": current_momentum_score,
                                        "decay_delta": momentum_decay_amount,
                                        "decay_consecutive_count": decay_checks,
                                        "trigger_reason": trigger_reason,
                                    })
                            print(f"\n{'='*60}")
                            print(f"📉 MOMENTUM DECAY EXIT TRIGGERED!")
                            print(f"Token: {symbol} ({token_address[:8]}...{token_address[-8:]})")
                            print(f"Entry Momentum: {entry_momentum_score:.4f} ({entry_momentum_source})")
                            print(f"Current Momentum: {current_momentum_score:.4f} ({current_source})")
                            print(f"Decay: {momentum_decay_amount:.4f}")
                            print(f"Reason: {momentum_decay_exit_reason}")
                            
                            # Execute sell
                            tx = _sell_token_multi_chain(token_address, chain_id, symbol)
                            if tx:
                                _log_trade_exit(token_address, symbol, original_entry_price, current_price, 
                                              f"momentum_decay_{momentum_decay_exit_reason}", 
                                              position_data=position_data, tx_hash=tx)
                                if trade:
                                    position_size = trade.get('position_size_usd', 0)
                                    pnl_usd = gain * position_size
                                    from src.core.performance_tracker import performance_tracker
                                    performance_tracker.log_trade_exit(trade['id'], current_price, pnl_usd, f"momentum_decay_{momentum_decay_exit_reason}")
                                
                                _cleanup_closed_position(position_key, token_address, chain_id)
                                send_telegram_message(
                                    f"📉 Momentum Decay Exit!\n"
                                    f"Token: {symbol} ({token_address[:8]}...{token_address[-8:]})\n"
                                    f"Entry Momentum: {entry_momentum_score:.4f}\n"
                                    f"Current Momentum: {current_momentum_score:.4f}\n"
                                    f"Decay: {momentum_decay_amount:.4f}\n"
                                    f"Entry: ${original_entry_price:.6f}\n"
                                    f"Exit: ${current_price:.6f} ({gain * 100:.2f}%)\n"
                                    f"TX: {tx}"
                                )
                                closed_positions.append(position_key)
                                updated_positions.pop(position_key, None)
                                continue
            except Exception as e:
                print(f"⚠️ Momentum decay exit check error: {e}")
                import traceback
                traceback.print_exc()

        # UPGRADE #4: Structure Failure Exit (strategy integrity - before stop-loss)
        # Soft exit — gated by profit floor
        structure_failure_exit_reason = None
        if config.get('ENABLE_STRUCTURE_FAILURE_EXIT', True) and isinstance(position_data, dict) \
                and _soft_exit_gated(gain, "structure_failure", "structure_failure_min_profit", config):
            try:
                # Check minimum position age
                position_timestamp = position_data.get("timestamp")
                if position_timestamp:
                    try:
                        entry_time = _dt.fromisoformat(position_timestamp.replace('Z', '+00:00'))
                        position_age_seconds = (_dt.now(entry_time.tzinfo) - entry_time).total_seconds()
                        min_age = config.get('structure_failure_min_age_seconds', 300)
                        
                        if position_age_seconds >= min_age:
                            structure_break_threshold = config.get('structure_break_threshold', 0.02)
                            
                            # Try candles first (preferred method)
                            # Try to get candles from stored position data
                            candles = None
                            if isinstance(position_data, dict):
                                candles = position_data.get('candles_15m')
                            
                            if candles and len(candles) >= 4:
                                # Use candles to detect swing low break
                                lookback = config.get('structure_lookback_candles', 20)
                                lookback_candles = candles[-lookback:] if len(candles) >= lookback else candles
                                
                                # Find swing low (minimum low in lookback window)
                                swing_low = min(float(c.get('low', current_price)) for c in lookback_candles)
                                
                                # Check if current price broke below swing low
                                break_pct = (swing_low - current_price) / swing_low if swing_low > 0 else 0
                                if break_pct >= structure_break_threshold:
                                    structure_failure_exit_reason = f"structure_break_candle_swing_low_{swing_low:.6f}_break_{break_pct:.4f}"
                            else:
                                # Fallback: use recent prices array (already updated above)
                                recent_prices = position_data.get("recent_prices", [])
                                
                                if len(recent_prices) >= 3:  # Need at least 3 prices
                                    # Find support level (minimum price in window, excluding current)
                                    support_level = min(recent_prices[:-1]) if len(recent_prices) > 1 else min(recent_prices)
                                    
                                    # Check if current price broke below support
                                    if current_price < support_level:
                                        break_pct = (support_level - current_price) / support_level if support_level > 0 else 0
                                        if break_pct >= structure_break_threshold:
                                            structure_failure_exit_reason = f"structure_break_price_support_{support_level:.6f}_break_{break_pct:.4f}"
                            
                            if structure_failure_exit_reason:
                                print(f"\n{'='*60}")
                                print(f"🏗️ STRUCTURE FAILURE EXIT TRIGGERED!")
                                print(f"Token: {symbol} ({token_address[:8]}...{token_address[-8:]})")
                                print(f"Position Age: {position_age_seconds:.0f}s")
                                print(f"Current Price: ${current_price:.6f}")
                                print(f"Reason: {structure_failure_exit_reason}")
                                
                                # Execute sell
                                tx = _sell_token_multi_chain(token_address, chain_id, symbol)
                                if tx:
                                    _log_trade_exit(token_address, symbol, original_entry_price, current_price,
                                                  f"structure_failure_{structure_failure_exit_reason}",
                                                  position_data=position_data, tx_hash=tx)
                                    if trade:
                                        position_size = trade.get('position_size_usd', 0)
                                        pnl_usd = gain * position_size
                                        from src.core.performance_tracker import performance_tracker
                                        performance_tracker.log_trade_exit(trade['id'], current_price, pnl_usd, f"structure_failure_{structure_failure_exit_reason}")
                                    
                                    _cleanup_closed_position(position_key, token_address, chain_id)
                                    send_telegram_message(
                                        f"🏗️ Structure Failure Exit!\n"
                                        f"Token: {symbol} ({token_address[:8]}...{token_address[-8:]})\n"
                                        f"Reason: {structure_failure_exit_reason}\n"
                                        f"Entry: ${original_entry_price:.6f}\n"
                                        f"Exit: ${current_price:.6f} ({gain * 100:.2f}%)\n"
                                        f"TX: {tx}"
                                    )
                                    closed_positions.append(position_key)
                                    updated_positions.pop(position_key, None)
                                    continue
                    except Exception as e:
                        print(f"⚠️ Structure failure exit check error: {e}")
            except Exception as e:
                print(f"⚠️ Structure failure exit check error: {e}")
                import traceback
                traceback.print_exc()

        # Hard stop-loss (only if not already handled by Partial TP Manager)
        # FALLBACK: If Partial TP Manager attempted but failed, this will catch it
        # Calculate effective stop loss with slippage buffer to account for execution slippage
        stop_loss_slippage_buffer = config.get('stop_loss_slippage_buffer', 0.15)
        effective_stop_loss = config['STOP_LOSS'] * (1 + stop_loss_slippage_buffer)
        # Trigger stop loss earlier to account for slippage
        if not partial_tp_handled_stop_loss and gain <= -effective_stop_loss:
            print(f"\n{'='*60}")
            print(f"🛑 STOP-LOSS TRIGGERED!")
            print(f"Token: {symbol} ({token_address[:8]}...{token_address[-8:]})")
            print(f"Chain: {chain_id.upper()}")
            print(f"Entry Price: ${original_entry_price:.6f}")
            print(f"Current Price: ${current_price:.6f}")
            print(f"Gain/Loss: {gain * 100:.2f}%")
            print(f"Stop Loss Threshold: {config['STOP_LOSS'] * 100:.2f}%")
            print(f"Effective Stop Loss (with slippage buffer): {effective_stop_loss * 100:.2f}%")
            
            # Check balance BEFORE attempting to sell - if already sold, clean up position
            print(f"🔍 [PRE-SELL CHECK] Verifying token balance before stop-loss sell...")
            pre_sell_balance = _check_token_balance_on_chain(token_address, chain_id)
            
            if pre_sell_balance == -1.0:
                # Balance check failed - proceed with sell attempt (can't verify)
                print(f"⚠️ [PRE-SELL CHECK] Balance check failed, proceeding with sell attempt...")
            elif pre_sell_balance <= 0.0 or pre_sell_balance < 0.000001:
                # Token already sold (zero/dust balance) - clean up position
                print(f"✅ [PRE-SELL CHECK] Token balance is zero/dust ({pre_sell_balance:.8f}) - position already sold, cleaning up...")
                
                # Log the trade exit
                _log_trade_exit(token_address, symbol, original_entry_price, current_price, "stop_loss",
                              position_data=position_data if isinstance(position_data, dict) else None)
                
                # Update performance tracker
                trade = _find_open_trade_by_address(token_address, chain_id)
                if trade:
                    position_size = trade.get('position_size_usd', 0)
                    pnl_usd = gain * position_size
                    from src.core.performance_tracker import performance_tracker
                    performance_tracker.log_trade_exit(trade['id'], current_price, pnl_usd, "stop_loss")
                    print(f"📊 Updated performance tracker for stop loss: {trade.get('symbol', '?')}")
                
                # Clean up position
                cleanup_success = _cleanup_closed_position(position_key, token_address, chain_id)
                if not cleanup_success:
                    cleanup_success = _verify_position_cleanup(position_key, token_address, chain_id)
                
                if cleanup_success:
                    # Re-fetch current price right before notification
                    fresh_current_price = _fetch_token_price_multi_chain(token_address) or current_price
                    if fresh_current_price != current_price:
                        print(f"📊 [PRICE UPDATE] Re-fetched price: ${current_price:.6f} -> ${fresh_current_price:.6f}")
                        current_price = fresh_current_price
                        gain = (current_price - original_entry_price) / original_entry_price
                    
                    # Send notification
                    send_telegram_message(
                        f"🛑 Stop-loss triggered (already sold)!\n"
                        f"Token: {symbol} ({token_address})\n"
                        f"Chain: {chain_id.upper()}\n"
                        f"Entry: ${original_entry_price:.6f}\n"
                        f"Exit: ${current_price:.6f} ({gain * 100:.2f}%)\n"
                        f"TX: Already sold (detected by balance check)"
                    )
                else:
                    print(f"❌ [STOP-LOSS] Failed to verify position cleanup - notification not sent")
                
                closed_positions.append(position_key)
                updated_positions.pop(position_key, None)
                continue  # Skip to next position
            else:
                # Token still in wallet - proceed with sell attempt
                print(f"✅ [PRE-SELL CHECK] Token balance confirmed: {pre_sell_balance:.6f} - proceeding with sell...")
            
            print(f"Attempting to sell...")
            print(f"{'='*60}\n")
            tx = _sell_token_multi_chain(token_address, chain_id, symbol)
            
            if tx:  # Only proceed if sell succeeded
                # Wait for transaction to propagate, then verify with Helius
                print(f"⏳ Waiting 30 seconds for transaction to propagate before verification...")
                time.sleep(30)
                
                # Run Helius reconciliation to verify sell (for Solana)
                sell_verified = True
                if chain_id.lower() == "solana":
                    try:
                        from src.utils.helius_client import HeliusClient
                        from src.config.secrets import HELIUS_API_KEY, SOLANA_WALLET_ADDRESS
                        
                        if HELIUS_API_KEY and SOLANA_WALLET_ADDRESS:
                            print(f"🔍 Verifying sell with Helius reconciliation...")
                            client = HeliusClient(HELIUS_API_KEY)
                            balances = client.get_address_balances(SOLANA_WALLET_ADDRESS)
                            tokens = balances.get("tokens", [])
                            
                            # Check if token is still in wallet
                            token_still_present = False
                            for token in tokens:
                                mint = (token.get("mint") or "").lower()
                                if mint == token_address.lower():
                                    amount_raw = token.get("amount") or 0
                                    decimals = token.get("decimals") or 0
                                    amount = float(amount_raw) / (10 ** decimals) if decimals else float(amount_raw)
                                    if amount > 1e-9:  # Still has balance
                                        token_still_present = True
                                        print(f"❌ Sell verification failed: {symbol} still in wallet ({amount:.6f} tokens)")
                                        break
                            
                            if token_still_present:
                                print(f"🔄 Sell failed - token still in wallet. Will retry on next cycle.")
                                sell_verified = False
                            else:
                                print(f"✅ Sell verified: {symbol} no longer in wallet")
                                
                                # Run full Helius reconciliation immediately after verified sell
                                try:
                                    from src.core.helius_reconciliation import reconcile_positions_and_pnl
                                    print(f"🔄 Running immediate reconciliation after sell...")
                                    recon_summary = reconcile_positions_and_pnl(limit=50)  # Smaller limit for post-sell
                                    print(f"✅ Post-sell reconciliation complete: closed={recon_summary.get('open_positions_closed', 0)}, verified={recon_summary.get('open_positions_verified', 0)}")
                                except Exception as recon_e:
                                    print(f"⚠️ Post-sell reconciliation error (non-critical): {recon_e}")
                    except Exception as e:
                        print(f"⚠️ Helius verification error (proceeding anyway): {e}")
                        import traceback
                        traceback.print_exc()
                
                if not sell_verified:
                    # Don't proceed with cleanup, mark for retry
                    print(f"❌ [STOP LOSS] Sell verification failed, will retry on next cycle")
                    continue
                
                log_trade(token_address, original_entry_price, current_price, "stop_loss")
                
                # Update performance tracker with exit
                trade = _find_open_trade_by_address(token_address, chain_id)
                if trade:
                    position_size = trade.get('position_size_usd', 0)
                    pnl_usd = gain * position_size  # gain is negative for stop loss
                    
                    # Analyze sell transaction for fees
                    sell_fee_data = _analyze_sell_fees(tx, chain_id)
                    
                    # Calculate fee-adjusted PnL
                    if sell_fee_data and sell_fee_data.get('exit_gas_fee_usd') is not None:
                        buy_gas = trade.get('entry_gas_fee_usd', 0) or 0
                        sell_gas = sell_fee_data.get('exit_gas_fee_usd', 0)
                        total_fees = buy_gas + sell_gas
                        sell_fee_data['total_fees_usd'] = total_fees
                    
                    from src.core.performance_tracker import performance_tracker
                    performance_tracker.log_trade_exit(trade['id'], current_price, pnl_usd, "stop_loss",
                                                       additional_data=sell_fee_data)
                    print(f"📊 Updated performance tracker for stop loss: {trade.get('symbol', '?')}")
                
                # CRITICAL: Remove position from ALL storage locations (open_positions.json, hunter_state.db, performance_data.json)
                cleanup_success = _cleanup_closed_position(position_key, token_address, chain_id)
                if not cleanup_success:
                    print(f"⚠️ Cleanup failed for {symbol}, retrying with address lookup...")
                    from src.utils.position_sync import find_position_key_by_address
                    found_key = find_position_key_by_address(token_address, chain_id)
                    if found_key:
                        cleanup_success = _cleanup_closed_position(found_key, token_address, chain_id)
                    
                    if not cleanup_success:
                        print(f"❌ CRITICAL: Failed to remove {symbol} from open_positions.json after sell!")
                
                # Verify cleanup succeeded before sending notification
                if not cleanup_success:
                    cleanup_success = _verify_position_cleanup(position_key, token_address, chain_id)
                
                if cleanup_success:
                    # Re-fetch current price right before notification to ensure accuracy
                    fresh_current_price = _fetch_token_price_multi_chain(token_address) or current_price
                    if fresh_current_price != current_price:
                        print(f"📊 [PRICE UPDATE] Re-fetched price: ${current_price:.6f} -> ${fresh_current_price:.6f}")
                        current_price = fresh_current_price
                        gain = (current_price - original_entry_price) / original_entry_price
                    
                    tx_display = tx if tx and tx not in ["verified_by_balance", "assumed_success"] else "Verified by balance check"
                    send_telegram_message(
                        f"🛑 Stop-loss triggered!\n"
                        f"Token: {symbol} ({token_address})\n"
                        f"Chain: {chain_id.upper()}\n"
                        f"Entry: ${original_entry_price:.6f}\n"
                        f"Now: ${current_price:.6f} ({gain * 100:.2f}%)\n"
                        f"TX: {tx_display}"
                    )
                else:
                    print(f"❌ [STOP-LOSS] Failed to verify position cleanup - notification not sent")
                
                closed_positions.append(position_key)
                updated_positions.pop(position_key, None)
            else:  # Sell failed - but verify with balance check first to avoid false negatives
                # Double-check if sell actually succeeded by checking balance
                balance_check_passed = False
                if chain_id == "solana":
                    try:
                        from src.execution.jupiter_lib import JupiterCustomLib
                        from src.config.secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY
                        lib = JupiterCustomLib(SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY)
                        time.sleep(3)  # Wait a bit for transaction to propagate
                        current_balance = lib.get_token_balance(token_address)
                        
                        # If balance is zero or very small, sell likely succeeded
                        if current_balance is not None:
                            if current_balance == 0 or current_balance < 0.000001:
                                print(f"✅ [BALANCE CHECK] Token balance is zero/dust - sell actually succeeded despite verification failure")
                                balance_check_passed = True
                    except Exception as balance_check_error:
                        print(f"⚠️ [BALANCE CHECK] Error checking balance: {balance_check_error}")
                
                if balance_check_passed:
                    # Sell actually succeeded - log it and remove position
                    print(f"✅ [RECOVERY] Stop-loss sell succeeded (verified by balance check), logging trade...")
                    log_trade(token_address, original_entry_price, current_price, "stop_loss")
                    
                    # Update performance tracker
                    trade = _find_open_trade_by_address(token_address, chain_id)
                    if trade:
                        position_size = trade.get('position_size_usd', 0)
                        pnl_usd = gain * position_size
                        
                        from src.core.performance_tracker import performance_tracker
                        performance_tracker.log_trade_exit(trade['id'], current_price, pnl_usd, "stop_loss")
                    
                    # Clean up position
                    cleanup_success = _cleanup_closed_position(position_key, token_address, chain_id)
                    if not cleanup_success:
                        cleanup_success = _verify_position_cleanup(position_key, token_address, chain_id)
                    
                    if cleanup_success:
                        # Re-fetch current price right before notification
                        fresh_current_price = _fetch_token_price_multi_chain(token_address) or current_price
                        if fresh_current_price != current_price:
                            print(f"📊 [PRICE UPDATE] Re-fetched price: ${current_price:.6f} -> ${fresh_current_price:.6f}")
                            current_price = fresh_current_price
                            gain = (current_price - original_entry_price) / original_entry_price
                        
                        send_telegram_message(
                            f"🛑 Stop-loss triggered!\n"
                            f"Token: {symbol} ({token_address})\n"
                            f"Chain: {chain_id.upper()}\n"
                            f"Entry: ${original_entry_price:.6f}\n"
                            f"Now: ${current_price:.6f} ({gain * 100:.2f}%)\n"
                            f"TX: Verified by balance check"
                        )
                    else:
                        print(f"❌ [STOP-LOSS] Failed to verify position cleanup - notification not sent")
                    
                    closed_positions.append(position_key)
                    updated_positions.pop(position_key, None)
                else:
                    # Sell actually failed
                    # Re-fetch current price right before notification
                    fresh_current_price = _fetch_token_price_multi_chain(token_address) or current_price
                    if fresh_current_price != current_price:
                        print(f"📊 [PRICE UPDATE] Re-fetched price: ${current_price:.6f} -> ${fresh_current_price:.6f}")
                        current_price = fresh_current_price
                        gain = (current_price - original_entry_price) / original_entry_price
                    
                    print(f"\n{'='*60}")
                    print(f"❌ CRITICAL: STOP-LOSS TRIGGERED BUT SELL FAILED!")
                    print(f"Token: {symbol} ({token_address[:8]}...{token_address[-8:]})")
                    print(f"Chain: {chain_id.upper()}")
                    print(f"Entry Price: ${original_entry_price:.6f}")
                    print(f"Current Price: ${current_price:.6f}")
                    print(f"Loss: {gain * 100:.2f}%")
                    print(f"Position will remain open and retry on next check")
                    print(f"{'='*60}\n")
                    send_telegram_message(
                        f"⚠️ Stop-loss triggered but SELL FAILED!\n"
                        f"Token: {symbol} ({token_address})\n"
                        f"Chain: {chain_id.upper()}\n"
                        f"Entry: ${original_entry_price:.6f}\n"
                        f"Now: ${current_price:.6f} ({gain * 100:.2f}%)\n"
                        f"Will retry on next check..."
                    )
                    # Don't remove position - keep monitoring to retry
            continue

        # Trailing stop (if enabled and price fell below dynamic level)
        if dyn_stop and current_price <= dyn_stop:
            print("🧵 Trailing stop-loss hit!")
            
            # Check balance BEFORE attempting to sell - if already sold, clean up position
            print(f"🔍 [PRE-SELL CHECK] Verifying token balance before trailing stop-loss sell...")
            pre_sell_balance = _check_token_balance_on_chain(token_address, chain_id)
            
            if pre_sell_balance == -1.0:
                # Balance check failed - proceed with sell attempt (can't verify)
                print(f"⚠️ [PRE-SELL CHECK] Balance check failed, proceeding with sell attempt...")
            elif pre_sell_balance <= 0.0 or pre_sell_balance < 0.000001:
                # Token already sold (zero/dust balance) - clean up position
                print(f"✅ [PRE-SELL CHECK] Token balance is zero/dust ({pre_sell_balance:.8f}) - position already sold, cleaning up...")
                
                # Log the trade exit
                _log_trade_exit(token_address, symbol, entry_price, current_price, "trailing_stop",
                              position_data=position_data if isinstance(position_data, dict) else None, tx_hash=tx)
                
                # Update performance tracker
                trade = _find_open_trade_by_address(token_address, chain_id)
                if trade:
                    position_size = trade.get('position_size_usd', 0)
                    pnl_usd = gain * position_size
                    from src.core.performance_tracker import performance_tracker
                    performance_tracker.log_trade_exit(trade['id'], current_price, pnl_usd, "trailing_stop")
                    print(f"📊 Updated performance tracker for trailing stop: {trade.get('symbol', '?')}")
                
                # Clean up position
                _cleanup_closed_position(position_key, token_address, chain_id)
                
                # Send notification
                send_telegram_message(
                    f"🧵 Trailing stop-loss triggered (already sold)!\n"
                    f"Token: {symbol} ({token_address})\n"
                    f"Chain: {chain_id.upper()}\n"
                    f"Entry: ${entry_price:.6f}\n"
                    f"Exit: ${current_price:.6f}\n"
                    f"TX: Already sold (detected by balance check)"
                )
                closed_positions.append(position_key)
                updated_positions.pop(position_key, None)
                continue  # Skip to next position
            else:
                # Token still in wallet - proceed with sell attempt
                print(f"✅ [PRE-SELL CHECK] Token balance confirmed: {pre_sell_balance:.6f} - proceeding with sell...")
            
            print("Selling...")
            tx = _sell_token_multi_chain(token_address, chain_id, symbol)
            
            if tx:  # Only proceed if sell succeeded
                # Wait for transaction to propagate, then verify with Helius
                print(f"⏳ Waiting 30 seconds for transaction to propagate before verification...")
                time.sleep(30)
                
                # Run Helius reconciliation to verify sell (for Solana)
                sell_verified = True
                if chain_id.lower() == "solana":
                    try:
                        from src.utils.helius_client import HeliusClient
                        from src.config.secrets import HELIUS_API_KEY, SOLANA_WALLET_ADDRESS
                        
                        if HELIUS_API_KEY and SOLANA_WALLET_ADDRESS:
                            print(f"🔍 Verifying sell with Helius reconciliation...")
                            client = HeliusClient(HELIUS_API_KEY)
                            balances = client.get_address_balances(SOLANA_WALLET_ADDRESS)
                            tokens = balances.get("tokens", [])
                            
                            # Check if token is still in wallet
                            token_still_present = False
                            for token in tokens:
                                mint = (token.get("mint") or "").lower()
                                if mint == token_address.lower():
                                    amount_raw = token.get("amount") or 0
                                    decimals = token.get("decimals") or 0
                                    amount = float(amount_raw) / (10 ** decimals) if decimals else float(amount_raw)
                                    if amount > 1e-9:  # Still has balance
                                        token_still_present = True
                                        print(f"❌ Sell verification failed: {symbol} still in wallet ({amount:.6f} tokens)")
                                        break
                            
                            if token_still_present:
                                print(f"🔄 Sell failed - token still in wallet. Will retry on next cycle.")
                                sell_verified = False
                            else:
                                print(f"✅ Sell verified: {symbol} no longer in wallet")
                    except Exception as e:
                        print(f"⚠️ Helius verification error (proceeding anyway): {e}")
                        import traceback
                        traceback.print_exc()
                
                if not sell_verified:
                    # Don't proceed with cleanup, mark for retry
                    print(f"❌ [TRAILING STOP] Sell verification failed, will retry on next cycle")
                    continue
                
                _log_trade_exit(token_address, symbol, entry_price, current_price, "trailing_stop",
                              position_data=position_data if isinstance(position_data, dict) else None, tx_hash=tx)
                
                # Update performance tracker with exit
                trade = _find_open_trade_by_address(token_address, chain_id)
                if trade:
                    position_size = trade.get('position_size_usd', 0)
                    pnl_usd = gain * position_size
                    
                    # Analyze sell transaction for fees
                    sell_fee_data = _analyze_sell_fees(tx, chain_id)
                    
                    # Calculate fee-adjusted PnL
                    if sell_fee_data and sell_fee_data.get('exit_gas_fee_usd') is not None:
                        buy_gas = trade.get('entry_gas_fee_usd', 0) or 0
                        sell_gas = sell_fee_data.get('exit_gas_fee_usd', 0)
                        total_fees = buy_gas + sell_gas
                        sell_fee_data['total_fees_usd'] = total_fees
                    
                    from src.core.performance_tracker import performance_tracker
                    performance_tracker.log_trade_exit(trade['id'], current_price, pnl_usd, "trailing_stop",
                                                       additional_data=sell_fee_data)
                    print(f"📊 Updated performance tracker for trailing stop: {trade.get('symbol', '?')}")
                
                # CRITICAL: Remove position from ALL storage locations (open_positions.json, hunter_state.db, performance_data.json)
                cleanup_success = _cleanup_closed_position(position_key, token_address, chain_id)
                if not cleanup_success:
                    print(f"⚠️ Cleanup failed for {symbol}, retrying with address lookup...")
                    from src.utils.position_sync import find_position_key_by_address
                    found_key = find_position_key_by_address(token_address, chain_id)
                    if found_key:
                        cleanup_success = _cleanup_closed_position(found_key, token_address, chain_id)
                    
                    if not cleanup_success:
                        print(f"❌ CRITICAL: Failed to remove {symbol} from open_positions.json after sell!")
                
                tx_display = tx if tx and tx not in ["verified_by_balance", "assumed_success"] else "Verified by balance check"
                send_telegram_message(
                    f"🧵 Trailing stop-loss triggered!\n"
                    f"Token: {symbol} ({token_address})\n"
                    f"Chain: {chain_id.upper()}\n"
                    f"Entry: ${entry_price:.6f}\n"
                    f"Now: ${current_price:.6f}\n"
                    f"TX: {tx_display}"
                )
                closed_positions.append(position_key)
                updated_positions.pop(position_key, None)
            else:  # Sell failed - but verify with balance check first to avoid false negatives
                # Double-check if sell actually succeeded by checking balance
                balance_check_passed = False
                if chain_id == "solana":
                    try:
                        from src.execution.jupiter_lib import JupiterCustomLib
                        from src.config.secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY
                        lib = JupiterCustomLib(SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY)
                        time.sleep(3)  # Wait a bit for transaction to propagate
                        current_balance = lib.get_token_balance(token_address)
                        
                        # If balance is zero or very small, sell likely succeeded
                        if current_balance is not None:
                            if current_balance == 0 or current_balance < 0.000001:
                                print(f"✅ [BALANCE CHECK] Token balance is zero/dust - sell actually succeeded despite verification failure")
                                balance_check_passed = True
                    except Exception as balance_check_error:
                        print(f"⚠️ [BALANCE CHECK] Error checking balance: {balance_check_error}")
                
                if balance_check_passed:
                    # Sell actually succeeded - log it and remove position
                    print(f"✅ [RECOVERY] Trailing stop sell succeeded (verified by balance check), logging trade...")
                    _log_trade_exit(token_address, symbol, entry_price, current_price, "trailing_stop",
                              position_data=position_data if isinstance(position_data, dict) else None, tx_hash=tx)
                    
                    # Update performance tracker
                    trade = _find_open_trade_by_address(token_address, chain_id)
                    if trade:
                        position_size = trade.get('position_size_usd', 0)
                        pnl_usd = gain * position_size
                        
                        from src.core.performance_tracker import performance_tracker
                        performance_tracker.log_trade_exit(trade['id'], current_price, pnl_usd, "trailing_stop")
                    
                    # Clean up position
                    _cleanup_closed_position(position_key, token_address, chain_id)
                    
                    send_telegram_message(
                        f"🧵 Trailing stop-loss triggered!\n"
                        f"Token: {symbol} ({token_address})\n"
                        f"Chain: {chain_id.upper()}\n"
                        f"Entry: ${entry_price:.6f}\n"
                        f"Now: ${current_price:.6f}\n"
                        f"TX: Verified by balance check"
                    )
                    closed_positions.append(position_key)
                    updated_positions.pop(position_key, None)
                else:
                    # Sell actually failed
                    send_telegram_message(
                        f"⚠️ Trailing stop-loss triggered but SELL FAILED!\n"
                        f"Token: {symbol} ({token_address})\n"
                        f"Chain: {chain_id.upper()}\n"
                        f"Entry: ${entry_price:.6f}\n"
                        f"Now: ${current_price:.6f}\n"
                        f"Will retry on next check..."
                    )
                    # Don't remove position - keep monitoring to retry
        else:
            print("⏳ Holding position...")

    # Save failure counts
    delisted_tokens["failure_counts"] = failure_counts
    save_delisted_tokens(delisted_tokens)
    
    # Save updated positions (with closed positions removed)
    # This ensures positions are properly removed when sells succeed
    if closed_positions:
        print(f"💾 Saving positions after closing {len(closed_positions)} position(s)...")
        save_positions(updated_positions)
        print(f"✅ Positions saved. Removed: {len(closed_positions)} position(s)")

    if closed_positions and not updated_positions:
        closed_list = "\n".join([f"• {addr}" for addr in closed_positions])
        send_telegram_message(
            f"✅ All positions closed.\nTokens:\n{closed_list}\nBot is now idle."
        )

def _main_loop():
    global _running
    startup_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}")
    print(f"🚀 POSITION MONITOR STARTING - {startup_time}")
    print(f"{'='*60}\n")
    
    _ensure_singleton()
    
    # Monitor should not modify open_positions.json; skip wallet reconciliation and sync
    
    cycle_count = 0
    try:
        while _running:
            _heartbeat()
            
            # Monitor should not write to open_positions.json; skip periodic reconciliation/sync
            cycle_count += 1
            # (no-op)
            
            cycle_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{cycle_start}] 🔄 Starting monitoring cycle #{cycle_count}")
            try:
                monitor_all_positions()
                cycle_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{cycle_end}] ✅ Monitoring cycle complete, sleeping 15s...\n")
            except Exception as e:
                cycle_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n[{cycle_end}] ❌ ERROR in monitoring cycle: {e}")
                import traceback
                print(f"[{cycle_end}] Traceback:\n{traceback.format_exc()}")
                print(f"[{cycle_end}] ⚠️ Continuing to next cycle...\n")
                # Don't re-raise - allow loop to continue
            time.sleep(15)  # poll interval (15 seconds)
    finally:
        # Always remove lock on exit
        _remove_lock()

if __name__ == "__main__":
    try:
        _main_loop()
    except KeyboardInterrupt:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ Monitor interrupted by user")
        _running = False
        _remove_lock()
    except Exception as e:
        error_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{error_time}] ❌ CRITICAL ERROR in position monitor!")
        print(f"[{error_time}] Error: {e}")
        import traceback
        print(f"[{error_time}] Traceback:\n{traceback.format_exc()}")
        _running = False
        _remove_lock()
        # Don't re-raise - allow process to exit cleanly
        # The error handling in _main_loop should catch most issues now
        sys.exit(1)