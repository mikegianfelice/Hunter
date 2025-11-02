import os
import sys
import time
import json
import yaml
import csv
import signal
from datetime import datetime
from pathlib import Path

# Add project root to path if not already there
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.execution.uniswap_executor import sell_token as sell_token_ethereum
from src.execution.base_executor import sell_token as sell_token_base
from src.execution.solana_executor import sell_token_solana
from src.utils.utils import fetch_token_price_usd
from src.monitoring.telegram_bot import send_telegram_message
from src.config.config_loader import get_config, get_config_float

# Dynamic config loading
def get_monitor_config():
    """Get current configuration values dynamically"""
    return {
        'TAKE_PROFIT': get_config_float("take_profit", 0.5),
        'STOP_LOSS': get_config_float("stop_loss", 0.25),
        'TRAILING_STOP': get_config_float("trailing_stop_percent", 0)
    }

# Use absolute paths based on project root
_project_root = Path(__file__).resolve().parents[2]
POSITIONS_FILE = _project_root / "data" / "open_positions.json"
LOG_FILE = _project_root / "data" / "trade_log.csv"
MONITOR_LOCK = _project_root / "data" / ".monitor_lock"
HEARTBEAT_FILE = _project_root / "data" / ".monitor_heartbeat"
DELISTED_TOKENS_FILE = _project_root / "data" / "delisted_tokens.json"

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
def load_positions():
    if not POSITIONS_FILE.exists():
        return {}
    with open(POSITIONS_FILE, "r") as f:
        try:
            return json.load(f) or {}
        except Exception:
            return {}

def save_positions(positions):
    POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(POSITIONS_FILE, "w") as f:
        json.dump(positions, f, indent=2)

def load_delisted_tokens():
    if not DELISTED_TOKENS_FILE.exists():
        return {}
    with open(DELISTED_TOKENS_FILE, "r") as f:
        try:
            return json.load(f) or {}
        except Exception:
            return {}

def save_delisted_tokens(delisted):
    DELISTED_TOKENS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DELISTED_TOKENS_FILE, "w") as f:
        json.dump(delisted, f, indent=2)

def log_trade(token, entry_price, exit_price, reason="normal"):
    pnl_pct = ((exit_price - entry_price) / entry_price) * 100 if entry_price else 0.0
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "token": token,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl_pct": round(pnl_pct, 2),
        "reason": reason
    }
    file_exists = LOG_FILE.exists()
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
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
    config = get_monitor_config()
    if config['TRAILING_STOP'] <= 0:
        return None

    # track per-token peak
    peak_key = f"{addr}_peak"
    peak = state.get(peak_key)

    if peak is None or current_price > peak:
        state[peak_key] = current_price
        peak = current_price

            # trailing_stop is a % drop from peak
        trail_stop_price = peak * (1 - config['TRAILING_STOP'])
    return trail_stop_price

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
                    print(f"⚠️ Zero price returned for {token_address[:8]}...{token_address[-8:]}")
                    return 0.0
            except Exception as e:
                print(f"⚠️ Solana price fetch failed: {e}")
                return 0.0
        
        # Fallback to Ethereum price fetching
        price = fetch_token_price_usd(token_address)
        if price and price > 0:
            print(f"🔗 Fetched Ethereum price for {token_address[:8]}...{token_address[-8:]}: ${price:.6f}")
            return price
            
        return 0.0
    except Exception as e:
        print(f"⚠️ Price fetch failed for {token_address}: {e}")
        return 0.0

def _sell_token_multi_chain(token_address: str, chain_id: str, symbol: str = "?") -> str:
    """
    Sell token using the appropriate executor based on chain
    """
    try:
        if chain_id == "ethereum":
            print(f"🔄 Selling {symbol} on Ethereum...")
            tx_hash, success = sell_token_ethereum(token_address)
        elif chain_id == "base":
            print(f"🔄 Selling {symbol} on Base...")
            # Get token balance for BASE
            from src.execution.base_executor import get_token_balance
            balance = get_token_balance(token_address)
            if balance > 0:
                tx_hash, success = sell_token_base(token_address, balance, symbol)
            else:
                print(f"❌ No {symbol} balance to sell")
                return None
        elif chain_id == "solana":
            print(f"🔄 Selling {symbol} on Solana...")
            # For Solana, we need to get the balance first and convert to USD
            from src.execution.jupiter_lib import JupiterExecutor
            executor = JupiterExecutor()
            balance = executor.get_token_balance(token_address)
            if balance > 0:
                # Get current price to calculate USD value
                current_price = _fetch_token_price_multi_chain(token_address)
                if current_price > 0:
                    # Calculate USD value of the token balance
                    amount_usd = balance * current_price
                    tx_hash, success = sell_token_solana(token_address, amount_usd, symbol)
                else:
                    print(f"⚠️ Could not get current price for {symbol}, using estimated value")
                    # Fallback: use a conservative estimate
                    amount_usd = balance * 0.01  # Conservative estimate
                    tx_hash, success = sell_token_solana(token_address, amount_usd, symbol)
            else:
                print(f"❌ No {symbol} balance to sell")
                return None
        else:
            print(f"❌ Unsupported chain for selling: {chain_id}")
            return None
            
        if success:
            print(f"✅ {symbol} sold successfully: {tx_hash}")
            return tx_hash
        else:
            print(f"❌ Failed to sell {symbol}")
            return None
            
    except Exception as e:
        print(f"❌ Error selling {symbol} on {chain_id}: {e}")
        return None

def _detect_delisted_token(token_address: str, consecutive_failures: int) -> bool:
    """
    Detect if a token is likely delisted based on consecutive price fetch failures
    """
    # Consider delisted after 5 consecutive failures (2.5 minutes of monitoring)
    return consecutive_failures >= 5

def monitor_all_positions():
    config = get_monitor_config()
    positions = load_positions()
    if not positions:
        print("📭 No open positions to monitor.")
        return

    # Load delisting tracking
    delisted_tokens = load_delisted_tokens()
    failure_counts = delisted_tokens.get("failure_counts", {})
    
    updated_positions = dict(positions)  # shallow copy
    closed_positions = []
    # ephemeral state for trailing stop peaks
    trail_state = {}

    for token_address, position_data in list(positions.items()):
        # Handle both old format (float) and new format (dict)
        if isinstance(position_data, dict):
            entry_price = float(position_data.get("entry_price", 0))
            chain_id = position_data.get("chain_id", "ethereum").lower()
            symbol = position_data.get("symbol", "?")
        else:
            # Legacy format - assume Ethereum
            entry_price = float(position_data)
            chain_id = "ethereum"
            symbol = "?"
            
        if entry_price <= 0:
            print(f"⚠️ Invalid entry price for {token_address}: {entry_price}")
            continue

        print(f"\n🔍 Monitoring token: {symbol} ({token_address}) on {chain_id.upper()}")
        print(f"🎯 Entry price: ${entry_price:.6f}")

        # Fetch current price using multi-chain function
        current_price = _fetch_token_price_multi_chain(token_address)

        # Track price fetch failures
        if current_price == 0:
            failure_counts[token_address] = failure_counts.get(token_address, 0) + 1
            print(f"⚠️ Price fetch failure #{failure_counts[token_address]} for {token_address[:8]}...{token_address[-8:]}")
            
            # Check if token is likely delisted
            if _detect_delisted_token(token_address, failure_counts[token_address]):
                print(f"🚨 TOKEN LIKELY DELISTED: {token_address[:8]}...{token_address[-8:]}")
                print(f"💸 Investment lost: ${entry_price:.6f}")
                
                # Log as delisted trade
                log_trade(token_address, entry_price, 0.0, "delisted")
                
                # Send Telegram alert
                send_telegram_message(
                    f"🚨 TOKEN DELISTED - INVESTMENT LOST!\n"
                    f"Token: {token_address[:8]}...{token_address[-8:]}\n"
                    f"Entry: ${entry_price:.6f}\n"
                    f"Current: $0.00 (DELISTED)\n"
                    f"Loss: 100% (${entry_price:.6f})\n"
                    f"⚠️ Token no longer tradeable"
                )
                
                # Remove from active positions
                closed_positions.append(token_address)
                updated_positions.pop(token_address, None)
                continue
        else:
            # Reset failure count on successful price fetch
            failure_counts[token_address] = 0

        if current_price is None or current_price == 0:
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
        if gain >= config['TAKE_PROFIT']:
            print("💰 Take-profit hit! Selling...")
            tx = _sell_token_multi_chain(token_address, chain_id, symbol)
            log_trade(token_address, entry_price, current_price, "take_profit")
            send_telegram_message(
                f"💰 Take-profit triggered!\n"
                f"Token: {symbol} ({token_address})\n"
                f"Chain: {chain_id.upper()}\n"
                f"Entry: ${entry_price:.6f}\n"
                f"Now: ${current_price:.6f} (+{gain * 100:.2f}%)\n"
                f"TX: {tx or 'N/A'}"
            )
            closed_positions.append(token_address)
            updated_positions.pop(token_address, None)
            continue  # move to next token

        # Hard stop-loss
        if gain <= -config['STOP_LOSS']:
            print("🛑 Stop-loss hit! Selling...")
            tx = _sell_token_multi_chain(token_address, chain_id, symbol)
            log_trade(token_address, entry_price, current_price, "stop_loss")
            send_telegram_message(
                f"🛑 Stop-loss triggered!\n"
                f"Token: {symbol} ({token_address})\n"
                f"Chain: {chain_id.upper()}\n"
                f"Entry: ${entry_price:.6f}\n"
                f"Now: ${current_price:.6f} ({gain * 100:.2f}%)\n"
                f"TX: {tx or 'N/A'}"
            )
            closed_positions.append(token_address)
            updated_positions.pop(token_address, None)
            continue

        # Trailing stop (if enabled and price fell below dynamic level)
        if dyn_stop and current_price <= dyn_stop:
            print("🧵 Trailing stop-loss hit! Selling...")
            tx = _sell_token_multi_chain(token_address, chain_id, symbol)
            log_trade(token_address, entry_price, current_price, "trailing_stop")
            send_telegram_message(
                f"🧵 Trailing stop-loss triggered!\n"
                f"Token: {symbol} ({token_address})\n"
                f"Chain: {chain_id.upper()}\n"
                f"Entry: ${entry_price:.6f}\n"
                f"Now: ${current_price:.6f}\n"
                f"TX: {tx or 'N/A'}"
            )
            closed_positions.append(token_address)
            updated_positions.pop(token_address, None)
        else:
            print("⏳ Holding position...")

    # Save updated positions and failure counts
    save_positions(updated_positions)
    delisted_tokens["failure_counts"] = failure_counts
    save_delisted_tokens(delisted_tokens)

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