import os
import time
import json
import yaml
import csv
import signal
from datetime import datetime

from uniswap_executor import sell_token
from utils import fetch_token_price_usd
from telegram_bot import send_telegram_message

# === Config / files ===
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f) or {}

TAKE_PROFIT = float(config.get("take_profit", 0.5))           # 50%
STOP_LOSS = float(config.get("stop_loss", 0.25))               # 25%
TRAILING_STOP = float(config.get("trailing_stop_percent", 0))  # e.g., 0.10 = 10% (0 to disable)

POSITIONS_FILE = "open_positions.json"
LOG_FILE = "trade_log.csv"
MONITOR_LOCK = ".monitor_lock"
HEARTBEAT_FILE = ".monitor_heartbeat"

# === Global for cleanup ===
_running = True

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
    with open(MONITOR_LOCK, "w") as f:
        json.dump(data, f)
    print(f"🔒 Monitor lock acquired with PID {data['pid']}")

def _remove_lock():
    try:
        if os.path.exists(MONITOR_LOCK):
            os.remove(MONITOR_LOCK)
            print("🧹 Monitor lock removed.")
    except Exception as e:
        print(f"⚠️ Failed to remove monitor lock: {e}")

def _ensure_singleton():
    """
    Make sure only one monitor runs.
    If a lock exists but its PID is dead, reclaim it.
    """
    if not os.path.exists(MONITOR_LOCK):
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
def load_positions():
    if not os.path.exists(POSITIONS_FILE):
        return {}
    with open(POSITIONS_FILE, "r") as f:
        try:
            return json.load(f) or {}
        except Exception:
            return {}

def save_positions(positions):
    with open(POSITIONS_FILE, "w") as f:
        json.dump(positions, f, indent=2)

def log_trade(token, entry_price, exit_price):
    pnl_pct = ((exit_price - entry_price) / entry_price) * 100 if entry_price else 0.0
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "token": token,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl_pct": round(pnl_pct, 2)
    }
    file_exists = os.path.isfile(LOG_FILE)
    try:
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"📄 Trade logged: {row}")
    except Exception as e:
        print(f"⚠️ Failed to write trade log: {e}")

def _apply_trailing_stop(state: dict, addr: str, current_price: float) -> float:
    """
    Update peak price and compute dynamic stop if trailing is enabled.
    Returns the dynamic stop price (or None if not active).
    """
    if TRAILING_STOP <= 0:
        return None

    # track per-token peak
    peak_key = f"{addr}_peak"
    peak = state.get(peak_key)

    if peak is None or current_price > peak:
        state[peak_key] = current_price
        peak = current_price

    # trailing_stop is a % drop from peak
    trail_stop_price = peak * (1 - TRAILING_STOP)
    return trail_stop_price

def monitor_all_positions():
    positions = load_positions()
    if not positions:
        print("📭 No open positions to monitor.")
        return

    updated_positions = dict(positions)  # shallow copy
    closed_positions = []
    # ephemeral state for trailing stop peaks
    trail_state = {}

    for token_address, entry_price_raw in list(positions.items()):
        try:
            entry_price = float(entry_price_raw)
        except Exception:
            print(f"⚠️ Invalid entry price for {token_address}: {entry_price_raw}")
            continue

        print(f"\n🔍 Monitoring token: {token_address}")
        print(f"🎯 Entry price: ${entry_price:.6f}")

        # Fetch current price
        try:
            current_price = fetch_token_price_usd(token_address)
        except Exception as e:
            print(f"⚠️ Price fetch failed for {token_address}: {e}")
            continue

        if current_price is None:
            print(f"⚠️ Could not fetch current price for {token_address}")
            continue

        print(f"📈 Current price: ${current_price:.6f}")
        gain = (current_price - entry_price) / entry_price
        print(f"📊 PnL: {gain * 100:.2f}%")

        # Trailing stop logic (optional)
        dyn_stop = _apply_trailing_stop(trail_state, token_address, current_price)
        if dyn_stop:
            print(f"🧵 Trailing stop @ ${dyn_stop:.6f} (peak-based)")

        # Take-profit
        if gain >= TAKE_PROFIT:
            print("💰 Take-profit hit! Selling...")
            tx = sell_token(token_address)
            log_trade(token_address, entry_price, current_price)
            send_telegram_message(
                f"💰 Take-profit triggered!\n"
                f"Token: {token_address}\n"
                f"Entry: ${entry_price:.6f}\n"
                f"Now: ${current_price:.6f} (+{gain * 100:.2f}%)\n"
                f"TX: {tx or 'SIMULATED'}"
            )
            closed_positions.append(token_address)
            updated_positions.pop(token_address, None)
            continue  # move to next token

        # Hard stop-loss
        if gain <= -STOP_LOSS:
            print("🛑 Stop-loss hit! Selling...")
            tx = sell_token(token_address)
            log_trade(token_address, entry_price, current_price)
            send_telegram_message(
                f"🛑 Stop-loss triggered!\n"
                f"Token: {token_address}\n"
                f"Entry: ${entry_price:.6f}\n"
                f"Now: ${current_price:.6f} ({gain * 100:.2f}%)\n"
                f"TX: {tx or 'SIMULATED'}"
            )
            closed_positions.append(token_address)
            updated_positions.pop(token_address, None)
            continue

        # Trailing stop (if enabled and price fell below dynamic level)
        if dyn_stop and current_price <= dyn_stop:
            print("🧵 Trailing stop-loss hit! Selling...")
            tx = sell_token(token_address)
            log_trade(token_address, entry_price, current_price)
            send_telegram_message(
                f"🧵 Trailing stop-loss triggered!\n"
                f"Token: {token_address}\n"
                f"Entry: ${entry_price:.6f}\n"
                f"Now: ${current_price:.6f}\n"
                f"TX: {tx or 'SIMULATED'}"
            )
            closed_positions.append(token_address)
            updated_positions.pop(token_address, None)
        else:
            print("⏳ Holding position...")

    save_positions(updated_positions)

    if closed_positions and not updated_positions:
        closed_list = "\n".join([f"• {addr}" for addr in closed_positions])
        send_telegram_message(
            f"✅ All positions closed.\nTokens:\n{closed_list}\nBot is now idle."
        )

def _main_loop():
    global _running
    _ensure_singleton()
    try:
        while _running:
            _heartbeat()
            monitor_all_positions()
            time.sleep(30)  # poll interval
    finally:
        # Always remove lock on exit
        _remove_lock()

if __name__ == "__main__":
    _main_loop()