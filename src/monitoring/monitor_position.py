import os
import sys
import time
import json
import yaml
import csv
import signal
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict

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
        'TAKE_PROFIT': get_config_float("take_profit", 0.12),
        'STOP_LOSS': get_config_float("stop_loss", 0.07),
        'TRAILING_STOP': get_config_float("trailing_stop_percent", 0),
        'USE_DYNAMIC_TP': get_config("use_dynamic_tp", False),
        'BASE_TP': get_config_float("base_take_profit", 0.12),
        'MAX_TP': get_config_float("max_take_profit", 0.20),
        'MIN_TP': get_config_float("min_take_profit", 0.08),
        'QUALITY_TP_BONUS': get_config_float("quality_tp_bonus", 0.03),
        'VOLUME_TP_BONUS': get_config_float("volume_tp_bonus", 0.02),
        'LIQUIDITY_TP_BONUS': get_config_float("liquidity_tp_bonus", 0.02)
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
def load_positions(validate_balances: bool = False):
    """
    Load positions from open_positions.json.
    If validate_balances is True, filters out positions with zero balance.
    """
    if not POSITIONS_FILE.exists():
        return {}
    with open(POSITIONS_FILE, "r") as f:
        try:
            positions = json.load(f) or {}
        except Exception:
            return {}
    
    if not validate_balances:
        return positions
    
    # Validate balances and filter out positions with zero balance
    validated_positions = {}
    for position_key, position_data in positions.items():
        # Handle both old format (float) and new format (dict)
        if isinstance(position_data, dict):
            chain_id = position_data.get("chain_id", "ethereum").lower()
            # Extract actual token address from position data (for composite keys)
            token_address = position_data.get("address", position_key)
            # If position_key is composite (address_tradeid), extract just the address part
            if "_" in position_key and not position_data.get("address"):
                token_address = position_key.split("_")[0]
        else:
            chain_id = "ethereum"
            token_address = position_key  # Legacy format uses address as key
        
        # Check wallet balance
        balance = _check_token_balance_on_chain(token_address, chain_id)
        
        if balance == -1.0:
            # Balance check failed - keep position to be safe
            validated_positions[position_key] = position_data
        elif balance <= 0.0 or balance < 0.000001:
            # Zero or dust balance - position was manually closed, skip it
            print(f"🚫 Filtering out position {position_key} - zero/dust balance detected (manually closed)")
        else:
            # Has balance - position is still open
            validated_positions[position_key] = position_data
    
    # If positions were filtered, save the cleaned list
    if len(validated_positions) < len(positions):
        print(f"🧹 Filtered {len(positions) - len(validated_positions)} position(s) with zero balance")
        save_positions(validated_positions)
    
    return validated_positions

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

def _analyze_sell_fees(tx_hash: str, chain_id: str) -> dict:
    """Analyze sell transaction to extract fee data"""
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
            # None means check failed (error) - return -1.0 to indicate unknown state
            if balance is None:
                return -1.0
            return float(balance)
            
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
    
    for position_key, position_data in list(positions.items()):
        # Handle both old format (float) and new format (dict)
        if isinstance(position_data, dict):
            chain_id = position_data.get("chain_id", "ethereum").lower()
            symbol = position_data.get("symbol", "?")
            # Extract actual token address from position data (for composite keys)
            # If address field exists, use it; otherwise fallback to position_key
            token_address = position_data.get("address", position_key)
            # If position_key is composite (address_tradeid), extract just the address part
            if "_" in position_key and not position_data.get("address"):
                token_address = position_key.split("_")[0]
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
    if quality_percent >= 70:
        tp += config.get('QUALITY_TP_BONUS', 0.03)
        print(f"  📈 High quality token: +{config.get('QUALITY_TP_BONUS', 0.03)*100:.0f}% TP")
    
    # High volume tokens get bonus
    if volume_24h >= 10_000_000:  # $10M+ volume
        tp += config.get('VOLUME_TP_BONUS', 0.02)
        print(f"  📊 High volume: +{config.get('VOLUME_TP_BONUS', 0.02)*100:.0f}% TP")
    
    # High liquidity tokens get bonus
    if liquidity >= 5_000_000:  # $5M+ liquidity
        tp += config.get('LIQUIDITY_TP_BONUS', 0.02)
        print(f"  💧 High liquidity: +{config.get('LIQUIDITY_TP_BONUS', 0.02)*100:.0f}% TP")
    
    # Clamp to min/max
    tp = max(config.get('MIN_TP', 0.08), min(config.get('MAX_TP', 0.20), tp))
    
    print(f"  🎯 Dynamic TP: {tp*100:.0f}% (base: {config.get('BASE_TP', 0.12)*100:.0f}%)")
    return tp

def _sync_only_positions_with_balance():
    """
    Sync positions from performance_data.json, but ONLY if they actually have on-chain balances.
    This prevents manually closed positions from being re-added.
    """
    try:
        from src.utils.position_sync import PERFORMANCE_DATA_FILE, OPEN_POSITIONS_FILE
        
        if not PERFORMANCE_DATA_FILE.exists():
            return
        
        # Load performance data
        with open(PERFORMANCE_DATA_FILE, "r") as f:
            perf_data = json.load(f)
        
        # Load existing open positions
        if OPEN_POSITIONS_FILE.exists():
            with open(OPEN_POSITIONS_FILE, "r") as f:
                open_positions = json.load(f) or {}
        else:
            open_positions = {}
        
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
            
            # Check if position already exists in open_positions
            # Create composite key same as position_sync.py
            if trade_id:
                position_key = f"{address}_{trade_id}"
            else:
                entry_time = trade.get("entry_time", "")
                position_key = f"{address}_{entry_time.replace(':', '-')}"
            
            # Skip if already tracked
            if position_key in open_positions:
                continue
            
            # CRITICAL: Check balance BEFORE syncing
            balance = _check_token_balance_on_chain(address, chain)
            
            if balance == -1.0:
                # Balance check failed - skip to be safe (don't add if we can't verify)
                print(f"⏸️  Skipping sync for {symbol} ({address[:8]}...{address[-8:]}) - balance check failed")
                continue
            elif balance <= 0.0 or balance < 0.000001:
                # Zero or dust balance - position was manually closed, don't sync it
                print(f"🚫 Skipping sync for {symbol} ({address[:8]}...{address[-8:]}) - zero/dust balance ({balance:.8f}) detected (manually closed)")
                # Mark as closed in performance_data to prevent future syncs
                trade['status'] = 'manual_close'
                trade['exit_time'] = datetime.now().isoformat()
                trade['exit_price'] = 0.0
                trade['pnl_usd'] = 0.0
                trade['pnl_percent'] = 0.0
                continue
            else:
                # Has balance - safe to sync
                print(f"✓ {symbol} ({address[:8]}...{address[-8:]}) has balance {balance:.6f} - syncing")
                from src.utils.position_sync import sync_position_from_performance_data_with_key
                entry_price = float(trade.get("entry_price", 0))
                position_size_usd = trade.get("position_size_usd", 0.0)
                if sync_position_from_performance_data_with_key(
                    position_key, address, symbol, chain, entry_price, position_size_usd, trade_id
                ):
                    synced_count += 1
        
        # Save updated performance_data if any trades were marked as closed
        if any(t.get("status") == "manual_close" for t in open_trades):
            with open(PERFORMANCE_DATA_FILE, "w") as f:
                json.dump(perf_data, f, indent=2)
        
        if synced_count > 0:
            print(f"✅ Synced {synced_count} position(s) with verified balances from performance_data.json")
    except Exception as e:
        print(f"⚠️ Error in smart position sync: {e}")

def monitor_all_positions():
    config = get_monitor_config()
    
    # Load existing positions first (with balance validation to filter out manually closed positions)
    positions = load_positions(validate_balances=True)
    reconciled_closed = []  # Track manually closed positions
    
    # Auto-reconcile FIRST: Remove positions with zero on-chain balance (manual closes)
    # This prevents manually closed positions from being re-added
    if positions:
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
                            token_address = old_position_data.get("address", position_key)
                            if "_" in position_key and not token_address:
                                token_address = position_key.split("_")[0]
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
                            current_price = _fetch_token_price_multi_chain(token_address) or entry_price
                            position_size = trade.get('position_size_usd', 0)
                            gain = ((current_price - entry_price) / entry_price) if entry_price > 0 else 0
                            pnl_usd = gain * position_size
                            performance_tracker.log_trade_exit(trade['id'], current_price, pnl_usd, "manual_close")
                            print(f"📊 Updated performance tracker for manually closed position: {trade.get('symbol', '?')}")
            except Exception as e:
                print(f"⚠️ Failed to update performance tracker for reconciled positions: {e}")
    
    # NOW sync positions from performance_data.json, but ONLY for trades that still have balances
    # This ensures positions are tracked even if initial logging failed, but doesn't re-add manually closed ones
    try:
        _sync_only_positions_with_balance()
    except Exception as e:
        print(f"⚠️ Failed to sync positions: {e}")
    
    # Reload positions after sync (in case new ones were added)
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
            token_address = position_data.get("address", position_key)
            # If position_key is composite (address_tradeid), extract just the address part
            if "_" in position_key and not position_data.get("address"):
                token_address = position_key.split("_")[0]
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
        print(f"\n🔍 Monitoring token: {symbol} ({token_address}) on {chain_id.upper()}{trade_id_str}")
        print(f"🎯 Entry price: ${entry_price:.6f}")

        # Fetch current price using multi-chain function
        current_price = _fetch_token_price_multi_chain(token_address)

        # Track price fetch failures
        if current_price == 0:
            # CRITICAL: For open positions, verify balance BEFORE incrementing failures
            # This prevents false failure accumulation when APIs have temporary issues
            if position_key in positions:
                print(f"🛡️ Position {symbol} is actively tracked - verifying balance to prevent false failures...")
                balance = _check_token_balance_on_chain(token_address, chain_id)
                
                if balance > 0:
                    # Token definitely exists - just a price API issue, don't track failure
                    print(f"✅ Token has balance ({balance:.6f}) - price API issue, not tracking failure")
                    print(f"⏳ Holding position (price unavailable but token exists in wallet)...")
                    continue
                elif balance == 0:
                    # Balance is zero - might be delisted or sold manually, track failure normally
                    print(f"⚠️ Zero balance detected for {symbol}, tracking price failure")
                else:  # balance == -1.0 (check failed)
                    # Can't verify - skip this check to avoid false negatives
                    print(f"⏸️ Balance check failed for {symbol}, skipping failure tracking")
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
                    continue
                else:
                    # Balance is 0 AND price fetch fails - likely delisted
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
                    
                    # Remove from active positions (use position_key)
                    closed_positions.append(position_key)
                    updated_positions.pop(position_key, None)
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

        # Calculate take profit threshold (static or dynamic based on config)
        trade = _find_open_trade_by_address(token_address, chain_id)
        take_profit_threshold = _calculate_dynamic_take_profit_for_position(trade, config)
        print(f"💰 TP Threshold: {take_profit_threshold*100:.2f}%")

        # Trailing stop logic (optional)
        dyn_stop = _apply_trailing_stop(trail_state, token_address, current_price)
        if dyn_stop:
            print(f"🧵 Trailing stop @ ${dyn_stop:.6f} (peak-based)")

        # Take-profit
        if gain >= take_profit_threshold:
            print("💰 Take-profit hit! Selling...")
            tx = _sell_token_multi_chain(token_address, chain_id, symbol)
            
            if tx:  # Only proceed if sell succeeded
                log_trade(token_address, entry_price, current_price, "take_profit")
                
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
                
                send_telegram_message(
                    f"💰 Take-profit triggered!\n"
                    f"Token: {symbol} ({token_address})\n"
                    f"Chain: {chain_id.upper()}\n"
                    f"Entry: ${entry_price:.6f}\n"
                    f"Now: ${current_price:.6f} (+{gain * 100:.2f}%)\n"
                    f"TX: {tx}"
                )
                closed_positions.append(position_key)
                updated_positions.pop(position_key, None)
            else:  # Sell failed
                send_telegram_message(
                    f"⚠️ Take-profit triggered but SELL FAILED!\n"
                    f"Token: {symbol} ({token_address})\n"
                    f"Chain: {chain_id.upper()}\n"
                    f"Entry: ${entry_price:.6f}\n"
                    f"Now: ${current_price:.6f} (+{gain * 100:.2f}%)\n"
                    f"Will retry on next check..."
                )
                # Don't remove position - keep monitoring to retry
            continue  # move to next token

        # Hard stop-loss
        if gain <= -config['STOP_LOSS']:
            print("🛑 Stop-loss hit! Selling...")
            tx = _sell_token_multi_chain(token_address, chain_id, symbol)
            
            if tx:  # Only proceed if sell succeeded
                log_trade(token_address, entry_price, current_price, "stop_loss")
                
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
                
                send_telegram_message(
                    f"🛑 Stop-loss triggered!\n"
                    f"Token: {symbol} ({token_address})\n"
                    f"Chain: {chain_id.upper()}\n"
                    f"Entry: ${entry_price:.6f}\n"
                    f"Now: ${current_price:.6f} ({gain * 100:.2f}%)\n"
                    f"TX: {tx}"
                )
                closed_positions.append(position_key)
                updated_positions.pop(position_key, None)
            else:  # Sell failed
                send_telegram_message(
                    f"⚠️ Stop-loss triggered but SELL FAILED!\n"
                    f"Token: {symbol} ({token_address})\n"
                    f"Chain: {chain_id.upper()}\n"
                    f"Entry: ${entry_price:.6f}\n"
                    f"Now: ${current_price:.6f} ({gain * 100:.2f}%)\n"
                    f"Will retry on next check..."
                )
                # Don't remove position - keep monitoring to retry
            continue

        # Trailing stop (if enabled and price fell below dynamic level)
        if dyn_stop and current_price <= dyn_stop:
            print("🧵 Trailing stop-loss hit! Selling...")
            tx = _sell_token_multi_chain(token_address, chain_id, symbol)
            
            if tx:  # Only proceed if sell succeeded
                log_trade(token_address, entry_price, current_price, "trailing_stop")
                
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
                
                send_telegram_message(
                    f"🧵 Trailing stop-loss triggered!\n"
                    f"Token: {symbol} ({token_address})\n"
                    f"Chain: {chain_id.upper()}\n"
                    f"Entry: ${entry_price:.6f}\n"
                    f"Now: ${current_price:.6f}\n"
                    f"TX: {tx}"
                )
                closed_positions.append(position_key)
                updated_positions.pop(position_key, None)
            else:  # Sell failed
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
    # Use balance-verified sync to prevent manually closed positions from being re-added
    try:
        _sync_only_positions_with_balance()
        print("✅ Initial position sync completed (balance-verified)")
    except Exception as e:
        print(f"⚠️ Failed to sync positions on startup: {e}")
    
    cycle_count = 0
    try:
        while _running:
            _heartbeat()
            
            # Periodically sync positions (every 10 cycles = ~5 minutes)
            # This ensures positions stay in sync even if performance_data is updated elsewhere
            # Use balance-verified sync to prevent manually closed positions from being re-added
            cycle_count += 1
            if cycle_count % 10 == 0:
                try:
                    _sync_only_positions_with_balance()
                except Exception as e:
                    print(f"⚠️ Periodic position sync failed: {e}")
            
            monitor_all_positions()
            time.sleep(30)  # poll interval
    finally:
        # Always remove lock on exit
        _remove_lock()

if __name__ == "__main__":
    _main_loop()