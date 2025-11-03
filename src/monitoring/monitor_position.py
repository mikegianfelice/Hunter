import os
import sys
import time
import json
import yaml
import csv
import signal
from datetime import datetime
from pathlib import Path
from typing import Tuple

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
from src.utils.position_sync import sync_all_open_positions

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
            from src.execution.jupiter_lib import JupiterCustomLib
            from src.config.secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY
            
            lib = JupiterCustomLib(SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY)
            balance = lib.get_token_balance(token_address)
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

def _check_token_balance_on_chain(token_address: str, chain_id: str) -> float:
    """
    Check token balance on the specified chain.
    Returns balance amount (0.0 if balance is zero or check fails).
    """
    try:
        chain_lower = chain_id.lower()
        
        if chain_lower == "solana":
            from src.execution.jupiter_lib import JupiterCustomLib
            from src.config.secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY
            lib = JupiterCustomLib(SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY)
            balance = lib.get_token_balance(token_address)
            return float(balance or 0.0)
            
        elif chain_lower == "base":
            from src.execution.base_executor import get_token_balance
            balance = get_token_balance(token_address)
            return float(balance or 0.0)
            
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
            return balance
            
        else:
            print(f"⚠️ Unsupported chain for balance check: {chain_id}")
            return 0.0
            
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
    
    for token_address, position_data in list(positions.items()):
        # Handle both old format (float) and new format (dict)
        if isinstance(position_data, dict):
            chain_id = position_data.get("chain_id", "ethereum").lower()
            symbol = position_data.get("symbol", "?")
        else:
            chain_id = "ethereum"
            symbol = "?"
        
        try:
            balance = _check_token_balance_on_chain(token_address, chain_id)
            
            if balance == -1.0:
                # Check failed - keep position to be safe
                print(f"⏸️  Skipping balance check for {symbol} ({token_address[:8]}...{token_address[-8:]}) - check failed")
                continue
            elif balance <= 0.0:
                # Zero balance - position was closed manually
                print(f"✅ Detected zero balance for {symbol} ({token_address[:8]}...{token_address[-8:]} on {chain_id.upper()}) - removing from tracking")
                to_remove.append(token_address)
                pruned.pop(token_address, None)
            else:
                # Has balance - position still open
                print(f"✓ {symbol} ({token_address[:8]}...{token_address[-8:]}) still has balance: {balance:.6f}")
                
        except Exception as e:
            print(f"⚠️ Error checking balance for {token_address}: {e}")
            # On error, keep the position to be safe
    
    if to_remove:
        print(f"🧹 Auto-reconciled {len(to_remove)} manually closed position(s)")
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

def monitor_all_positions():
    config = get_monitor_config()
    
    # FIRST: Sync positions from performance_data.json to ensure nothing is missing
    # This ensures positions are tracked even if initial logging failed
    try:
        sync_all_open_positions()
    except Exception as e:
        print(f"⚠️ Failed to sync positions: {e}")
    
    positions = load_positions()
    if not positions:
        print("📭 No open positions to monitor.")
        return

    # Save original positions data before pruning (needed for performance tracker updates)
    original_positions_before_prune = dict(positions)

    # Auto-reconcile: Remove positions with zero on-chain balance (manual closes)
    positions, reconciled_closed = _prune_positions_with_zero_balance(positions)
    
    # Update performance tracker for manually closed positions
    if reconciled_closed:
        try:
            from src.core.performance_tracker import performance_tracker
            for token_address in reconciled_closed:
                # Find the position data to get chain_id and entry_price before it was removed
                old_position_data = original_positions_before_prune.get(token_address)
                if old_position_data:
                    if isinstance(old_position_data, dict):
                        chain_id = old_position_data.get("chain_id", "ethereum")
                        entry_price = float(old_position_data.get("entry_price", 0))
                    else:
                        chain_id = "ethereum"
                        entry_price = float(old_position_data) if old_position_data else 0
                    
                    # Find and close the trade
                    trade = _find_open_trade_by_address(token_address, chain_id)
                    if trade:
                        # Try to get current price if possible, otherwise use entry price (0 PnL)
                        current_price = _fetch_token_price_multi_chain(token_address) or entry_price
                        position_size = trade.get('position_size_usd', 0)
                        gain = ((current_price - entry_price) / entry_price) if entry_price > 0 else 0
                        pnl_usd = gain * position_size
                        performance_tracker.log_trade_exit(trade['id'], current_price, pnl_usd, "manual_close")
                        print(f"📊 Updated performance tracker for manually closed position: {trade.get('symbol', '?')}")
        except Exception as e:
            print(f"⚠️ Failed to update performance tracker for reconciled positions: {e}")
    
    if not positions:
        print("📭 No open positions after auto-reconciliation.")
        return

    # Load delisting tracking
    delisted_tokens = load_delisted_tokens()
    failure_counts = delisted_tokens.get("failure_counts", {})
    
    updated_positions = dict(positions)  # shallow copy
    closed_positions = list(reconciled_closed)  # Include reconciled positions
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
            
            # Update performance tracker with exit
            trade = _find_open_trade_by_address(token_address, chain_id)
            if trade:
                position_size = trade.get('position_size_usd', 0)
                pnl_usd = gain * position_size  # gain is already a ratio
                from src.core.performance_tracker import performance_tracker
                performance_tracker.log_trade_exit(trade['id'], current_price, pnl_usd, "take_profit")
                print(f"📊 Updated performance tracker for take profit: {trade.get('symbol', '?')}")
            
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
            
            # Update performance tracker with exit
            trade = _find_open_trade_by_address(token_address, chain_id)
            if trade:
                position_size = trade.get('position_size_usd', 0)
                pnl_usd = gain * position_size  # gain is negative for stop loss
                from src.core.performance_tracker import performance_tracker
                performance_tracker.log_trade_exit(trade['id'], current_price, pnl_usd, "stop_loss")
                print(f"📊 Updated performance tracker for stop loss: {trade.get('symbol', '?')}")
            
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
            
            # Update performance tracker with exit
            trade = _find_open_trade_by_address(token_address, chain_id)
            if trade:
                position_size = trade.get('position_size_usd', 0)
                pnl_usd = gain * position_size
                from src.core.performance_tracker import performance_tracker
                performance_tracker.log_trade_exit(trade['id'], current_price, pnl_usd, "trailing_stop")
                print(f"📊 Updated performance tracker for trailing stop: {trade.get('symbol', '?')}")
            
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
    
    # Sync positions on startup to catch any missed positions
    try:
        sync_all_open_positions()
        print("✅ Initial position sync completed")
    except Exception as e:
        print(f"⚠️ Failed to sync positions on startup: {e}")
    
    cycle_count = 0
    try:
        while _running:
            _heartbeat()
            
            # Periodically sync positions (every 10 cycles = ~5 minutes)
            # This ensures positions stay in sync even if performance_data is updated elsewhere
            cycle_count += 1
            if cycle_count % 10 == 0:
                try:
                    sync_all_open_positions()
                except Exception as e:
                    print(f"⚠️ Periodic position sync failed: {e}")
            
            monitor_all_positions()
            time.sleep(30)  # poll interval
    finally:
        # Always remove lock on exit
        _remove_lock()

if __name__ == "__main__":
    _main_loop()