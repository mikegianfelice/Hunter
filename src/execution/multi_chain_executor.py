import yaml
import time
import sys
import subprocess
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from web3 import Web3

from src.config.secrets import INFURA_URL, WALLET_ADDRESS, SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY
from src.utils.utils import get_eth_price_usd
from src.utils.position_sync import create_position_key, is_native_gas_token
from src.monitoring.telegram_bot import send_telegram_message
from src.config.config_loader import get_config, get_config_bool, get_config_float
from src.monitoring.logger import log_event
from src.core.advanced_trading import advanced_trading
from src.utils.address_utils import validate_chain_address_match, normalize_evm_address, detect_chain_from_address
from src.storage.positions import upsert_position

# Dynamic config loading
def get_multi_chain_config():
    """Get current configuration values dynamically"""
    return {
        'TEST_MODE': get_config_bool("test_mode", True),
        'TRADE_AMOUNT_USD_DEFAULT': get_config_float("trade_amount_usd", 5),
        'SLIPPAGE': get_config_float("slippage", 0.02)
    }

# Chain-specific configurations
CHAIN_CONFIGS = {
    "ethereum": {
        "rpc_url": INFURA_URL,
        "native_token": "ETH",
        "wrapped_token": "0xC02aaA39b223FE8D0A0E5C4F27eAD9083C756Cc2",  # WETH
        "dex": "uniswap",
        "router": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
        "factory": "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f",
        "executor_module": "uniswap_executor"
    },
    "solana": {
        "rpc_url": SOLANA_RPC_URL,
        "native_token": "SOL",
        "wrapped_token": "So11111111111111111111111111111111111111112",  # Wrapped SOL
        "dex": "raydium",
        "executor_module": "raydium_executor"
    },
    "base": {
        "rpc_url": os.getenv("BASE_RPC_URL", "https://mainnet.base.org"),
        "native_token": "ETH",
        "wrapped_token": "0x4200000000000000000000000000000000000006",  # WETH on Base
        "dex": "uniswap",
        "router": "0x4752ba5DBc23f44D87826276BF6Fd6b1C372aD24",
        "factory": "0x8909Dc15e40173Ff4699343b6eB8132c65e18eC6",
        "executor_module": "base_executor"
    },
    "polygon": {
        "rpc_url": "https://polygon-rpc.com",
        "native_token": "MATIC",
        "wrapped_token": "0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270",  # WMATIC
        "dex": "uniswap",
        "router": "0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff",
        "factory": "0x5757371414417b8C6CAad45bAeF941aBc7d3Ab32",
        "executor_module": "uniswap_executor"
    },
    "bsc": {
        "rpc_url": "https://bsc-dataseed.binance.org",
        "native_token": "BNB",
        "wrapped_token": "0xbb4CdB9CBd36B01bD1cBaEF60aF814a3f6F0Ee75",  # WBNB
        "dex": "pancakeswap",
        "router": "0x10ED43C718714eb63d5aA57B78B54704E256024E",
        "factory": "0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73",
        "executor_module": "pancakeswap_executor"
    },
    "arbitrum": {
        "rpc_url": "https://arb1.arbitrum.io/rpc",
        "native_token": "ETH",
        "wrapped_token": "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",  # WETH on Arbitrum
        "dex": "uniswap",
        "router": "0xE592427A0AEce92De3Edee1F18E0157C05861564",
        "factory": "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f",
        "executor_module": "uniswap_executor"
    },
    "optimism": {
        "rpc_url": "https://mainnet.optimism.io",
        "native_token": "ETH",
        "wrapped_token": "0x4200000000000000000000000000000000000006",  # WETH on Optimism
        "dex": "uniswap",
        "router": "0xE592427A0AEce92De3Edee1F18E0157C05861564",
        "factory": "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f",
        "executor_module": "uniswap_executor"
    },
    "pulsechain": {
        "rpc_url": "https://rpc.pulsechain.com",
        "native_token": "PLS",
        "wrapped_token": "0xA1077a294dDE1B09bB078844df40758a5D0f9a27",  # WPLS
        "dex": "pulsex",
        "router": "0x165C3410fC91EF562C50559f7d2289fEbed552d9",
        "factory": "0x1715a3E4A142d8b5e4F161f3D6C0D5C4e4fD4F6e",
        "executor_module": "pulsex_executor"
    }
}

# Housekeeping
MONITOR_SCRIPT = Path(__file__).resolve().parents[2] / "src" / "monitoring" / "monitor_position.py"
WATCHDOG_SCRIPT = Path(__file__).resolve().parents[2] / "src" / "monitoring" / "monitor_watchdog.py"


def _validate_and_update_position_size(token: dict, chain_id: str, token_address: str = None):
    """
    Validate position_size_usd against actual wallet balance and update if there's a discrepancy.
    This fixes the issue where a buy fails and retries with smaller amount, but position was
    logged with the original intended amount.
    """
    try:
        # Get token address from token dict if not provided
        if not token_address:
            token_address = token.get("address") or token.get("token_address")
            if not token_address:
                log_event("trading.position_validation_skipped", level="WARNING",
                         symbol=token.get("symbol", "?"), reason="Token address not found")
                return
        
        # Get actual wallet balance
        balance = _get_token_balance_for_validation(token_address, chain_id)
        if balance is None or balance <= 0:
            log_event("trading.position_validation_skipped", level="WARNING", 
                     symbol=token.get("symbol", "?"), reason="Could not get wallet balance")
            return
        
        # Get current price - fallback to entry price if current price unavailable
        current_price = _get_token_price_for_validation(token_address, chain_id)
        if current_price <= 0:
            # Use entry price as fallback (should be close to buy price)
            current_price = float(token.get("priceUsd") or token.get("entry_price") or 0.0)
            if current_price <= 0:
                log_event("trading.position_validation_skipped", level="WARNING",
                         symbol=token.get("symbol", "?"), reason="Could not get price (current or entry)")
                return
        
        # Calculate actual position value
        actual_position_size_usd = balance * current_price
        
        # Get logged position size
        logged_position_size_usd = float(token.get("position_size_usd", 0))
        
        if logged_position_size_usd <= 0:
            log_event("trading.position_validation_skipped", level="WARNING",
                     symbol=token.get("symbol", "?"), reason="Logged position size is zero")
            return
        
        # Check if there's a significant discrepancy (>10%)
        discrepancy_ratio = abs(actual_position_size_usd - logged_position_size_usd) / logged_position_size_usd
        if discrepancy_ratio > 0.1:  # More than 10% difference
            log_event("trading.position_size_discrepancy", level="WARNING",
                     symbol=token.get("symbol", "?"),
                     logged_amount=logged_position_size_usd,
                     actual_amount=actual_position_size_usd,
                     discrepancy_pct=discrepancy_ratio * 100)
            
            # Update the position with actual value
            position_key = create_position_key(token_address)
            from src.storage.positions import load_positions, upsert_position
            positions = load_positions()
            if position_key in positions:
                position_data = positions[position_key]
                position_data["position_size_usd"] = actual_position_size_usd
                upsert_position(position_key, position_data)
                
                # Also update performance tracker if we have trade_id
                if token.get("trade_id"):
                    try:
                        from src.core.performance_tracker import performance_tracker
                        for trade in performance_tracker.trades:
                            if trade.get('id') == token.get("trade_id"):
                                trade['position_size_usd'] = actual_position_size_usd
                                trade['entry_amount_usd_actual'] = actual_position_size_usd
                                performance_tracker.save_data()
                                break
                    except Exception as e:
                        log_event("trading.position_validation_perf_update_error", level="WARNING", error=str(e))
                
                print(f"✅ Updated position size for {token.get('symbol', '?')}: ${logged_position_size_usd:.2f} -> ${actual_position_size_usd:.2f}")
                log_event("trading.position_size_updated",
                         symbol=token.get("symbol", "?"),
                         old_amount=logged_position_size_usd,
                         new_amount=actual_position_size_usd)
        else:
            log_event("trading.position_size_validated",
                     symbol=token.get("symbol", "?"),
                     logged_amount=logged_position_size_usd,
                     actual_amount=actual_position_size_usd)
    
    except Exception as e:
        log_event("trading.position_validation_error", level="WARNING", error=str(e))
        # Don't raise - validation failure shouldn't break position logging


def _get_token_balance_for_validation(token_address: str, chain_id: str) -> Optional[float]:
    """Get token balance for validation purposes"""
    try:
        chain_lower = chain_id.lower()
        
        if chain_lower == "solana":
            from src.execution.jupiter_lib import JupiterCustomLib
            from src.config.secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY
            lib = JupiterCustomLib(SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY)
            balance = lib.get_token_balance(token_address)
            return float(balance) if balance is not None else None
            
        elif chain_lower in ["base", "ethereum"]:
            from src.execution.base_executor import get_token_balance
            balance = get_token_balance(token_address)
            return float(balance) if balance is not None else None
        
        return None
    except Exception:
        return None


def _get_token_price_for_validation(token_address: str, chain_id: str) -> float:
    """Get token price for validation purposes"""
    try:
        chain_lower = chain_id.lower()
        
        if chain_lower == "solana":
            from src.execution.solana_executor import get_token_price_usd
            price = get_token_price_usd(token_address)
            return float(price) if price and price > 0 else 0.0
        elif chain_lower in ["base", "ethereum"]:
            # For EVM chains, try to get price from token data or use a price API
            # For now, return 0.0 if we can't get it easily
            # This could be enhanced to use price APIs
            return 0.0
        
        return 0.0
    except Exception:
        return 0.0


def _log_position(token: dict, *, trade_id: Optional[str] = None, entry_time: Optional[str] = None):
    addr = token["address"]
    entry = float(token.get("priceUsd") or 0.0)
    chain_id = token.get("chainId", "ethereum").lower()
    timestamp = entry_time or datetime.now().isoformat()
    if is_native_gas_token(addr, token.get("symbol"), chain_id):
        try:
            print(
                f"⛽️ Skipping native gas token {token.get('symbol','?')} ({addr}) on "
                f"{chain_id.upper()} - not logging to open_positions"
            )
        except BrokenPipeError:
            pass
        return
    position_key = create_position_key(addr)

    position_data = {
        "entry_price": entry,
        "chain_id": chain_id,
        "symbol": token.get("symbol", "?"),
        "timestamp": timestamp,
        "address": addr,
    }

    if "position_size_usd" in token:
        position_data["position_size_usd"] = float(token["position_size_usd"])

    if trade_id:
        position_data["trade_id"] = trade_id

    upsert_position(position_key, position_data)
    try:
        print(
            f"📝 Logged position: {token.get('symbol','?')} ({addr}) on {chain_id.upper()} @ ${entry:.6f}"
        )
    except BrokenPipeError:
        pass

def _launch_monitor_detached(use_watchdog: bool = True):
    """
    Launch the position monitor, optionally with watchdog for automatic restart.
    
    Args:
        use_watchdog: If True, launch the watchdog which will manage the monitor.
                     If False, launch the monitor directly (legacy behavior).
    """
    if use_watchdog:
        # Launch watchdog instead - it will manage the monitor
        watchdog_script = WATCHDOG_SCRIPT if isinstance(WATCHDOG_SCRIPT, Path) else Path(WATCHDOG_SCRIPT)
        
        if not watchdog_script.exists():
            try:
                print(f"⚠️ monitor_watchdog.py not found at {watchdog_script}, falling back to direct launch")
                use_watchdog = False
            except BrokenPipeError:
                pass
        
        if use_watchdog:
            try:
                project_root = watchdog_script.parents[2]  # Go up from src/monitoring to project root
                env = os.environ.copy()
                if 'PYTHONPATH' in env:
                    env['PYTHONPATH'] = f"{project_root}:{env['PYTHONPATH']}"
                else:
                    env['PYTHONPATH'] = str(project_root)
                
                log_dir = project_root / "logs"
                log_dir.mkdir(exist_ok=True)
                watchdog_log_file = log_dir / "monitor_watchdog.log"
                
                log_file = open(watchdog_log_file, "a", buffering=1)
                
                subprocess.Popen(
                    [sys.executable, str(watchdog_script)],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=str(project_root),
                    start_new_session=True
                )
                
                try:
                    print(f"👁️ Started monitor_watchdog.py (will manage position monitor)")
                except BrokenPipeError:
                    pass
            except Exception as e:
                try:
                    print(f"⚠️ Could not launch monitor_watchdog.py: {e}")
                except BrokenPipeError:
                    pass
            return

    # Fallback to direct monitor launch (legacy behavior)
    script = MONITOR_SCRIPT if isinstance(MONITOR_SCRIPT, Path) else Path(MONITOR_SCRIPT)
    
    if not script.exists():
        try:
            print(f"⚠️ monitor_position.py not found at {script}")
            print(f"⚠️ Current working directory: {os.getcwd()}")
        except BrokenPipeError:
            pass
        return
    
    try:
        # Add the project root to PYTHONPATH so monitor can import modules correctly
        project_root = script.parents[2]  # Go up from src/monitoring to project root
        env = os.environ.copy()
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{project_root}:{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = str(project_root)
        
        # Redirect output to log file instead of /dev/null so we can see what's happening
        log_dir = project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        monitor_log_file = log_dir / "position_monitor.log"
        
        # Open log file in append mode with line buffering for real-time visibility
        log_file = open(monitor_log_file, "a", buffering=1)  # Line buffering for real-time output
        
        subprocess.Popen(
            [sys.executable, str(script)],
            stdout=log_file,
            stderr=subprocess.STDOUT,  # Redirect stderr to same file
            env=env,
            cwd=str(project_root),  # Set working directory to project root
            start_new_session=True  # Ensure process runs independently
        )
        # Note: We don't close log_file here - it will be closed when the process exits
        try:
            print(f"👁️ Started monitor_position.py at {script} via {sys.executable}")
        except BrokenPipeError:
            pass
    except Exception as e:
        try:
            print(f"⚠️ Could not launch monitor_position.py: {e}")
            import traceback
            print(f"⚠️ Error details: {traceback.format_exc()}")
        except BrokenPipeError:
            pass

def execute_trade(token: dict, trade_amount_usd: float = None):
    """
    Multi-chain trade execution with advanced features:
    1. Enhanced preflight checks
    2. Order splitting for large trades
    3. Dynamic slippage calculation
    4. ExactOut trades for sketchy tokens
    5. Route restrictions and preferences
    6. Delegate to appropriate executor
    7. Log position and launch monitor if successful
    
    Returns: (tx_hash_hex_or_sim, success_bool)
    """
    config = get_multi_chain_config()

    symbol = token.get("symbol", "?")
    token_address = token["address"]
    chain_id = token.get("chainId", "ethereum").lower()

    # Enhanced chain/address validation before any executor is selected
    try:
        is_valid, corrected_chain, error_message = validate_chain_address_match(token_address, chain_id)
        
        if not is_valid:
            error_msg = f"Chain/address validation failed for {symbol}: {error_message}"
            print(f"❌ {error_msg}")
            log_event("trade.invalid_chain_address", level="ERROR", log_type="chain_validation", 
                     symbol=symbol, token_address=token_address, chain=chain_id, error=error_message)
            return None, False
        
        # Update chain if it was corrected
        if corrected_chain != chain_id:
            print(f"🔧 Correcting chain for {symbol}: {chain_id} → {corrected_chain} (by address format)")
            log_event("trade.chain_corrected", log_type="chain_validation", 
                     symbol=symbol, token_address=token_address, from_chain=chain_id, to_chain=corrected_chain)
            chain_id = corrected_chain
        
        # Normalize EVM addresses
        if detect_chain_from_address(token_address) == "evm":
            token_address = normalize_evm_address(token_address)
            
    except Exception as e:
        error_msg = f"Address validation failed for {symbol}: {str(e)}"
        print(f"❌ {error_msg}")
        log_event("trade.address_validation_error", level="ERROR", symbol=symbol, token_address=token_address, error=str(e))
        return None, False
    amount_usd = float(trade_amount_usd or config['TRADE_AMOUNT_USD_DEFAULT'])
    
    log_event("trade.start", symbol=symbol, token_address=token_address, chain=chain_id, amount_usd=amount_usd)
    
    # Enhanced preflight check with timeout
    try:
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Preflight check timed out")
        
        # Set 30 second timeout for preflight check
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        
        try:
            preflight_passed, reason = advanced_trading.enhanced_preflight_check(token, amount_usd)
            signal.alarm(0)  # Cancel the alarm
            if not preflight_passed:
                error_msg = f"Preflight check failed: {reason}"
                print(f"❌ {error_msg}")
                log_event("trade.preflight_failed", level="WARNING", symbol=symbol, reason=reason)
                return None, False
        except TimeoutError:
            signal.alarm(0)  # Cancel the alarm
            error_msg = "Preflight check timed out after 30 seconds"
            print(f"⚠️ {error_msg}")
            log_event("trade.preflight_timeout", level="WARNING", symbol=symbol, timeout=30)
            # Continue with trade if preflight times out
        except Exception as e:
            signal.alarm(0)  # Cancel the alarm
            error_msg = f"Preflight check error: {str(e)}"
            print(f"⚠️ {error_msg}")
            log_event("trade.preflight_error", level="WARNING", symbol=symbol, error=str(e))
            # Continue with trade if preflight fails
    except Exception as e:
        error_msg = f"Preflight setup error: {str(e)}"
        print(f"⚠️ {error_msg}")
        log_event("trade.preflight_setup_error", level="WARNING", symbol=symbol, error=str(e))
        # Continue with trade if preflight setup fails
    
    # Calculate order slices
    slices = advanced_trading.calculate_order_slices(amount_usd, token)
    
    # Calculate dynamic slippage
    base_slippage = config['SLIPPAGE']
    dynamic_slippage = advanced_trading.calculate_dynamic_slippage(token, base_slippage)
    
    # Determine if ExactOut should be used
    use_exactout = advanced_trading.should_use_exactout(token)
    
    # Get route preferences
    route_preferences = advanced_trading.get_route_preferences(token)
    
    log_event(
        "trade.config",
        slices=len(slices),
        total_amount_usd=round(amount_usd, 2),
        slippage=round(dynamic_slippage, 6),
        exactout=bool(use_exactout),
        route_preferences=route_preferences,
    )
    
    try:
        # Get chain configuration with error handling
        try:
            chain_config = get_chain_config(chain_id)
            if not chain_config:
                error_msg = f"No configuration found for chain: {chain_id}"
                print(f"❌ {error_msg}")
                log_event("trade.chain_config_error", level="ERROR", symbol=symbol, chain=chain_id, error=error_msg)
                return None, False
        except Exception as e:
            error_msg = f"Failed to get chain configuration for {chain_id}: {str(e)}"
            print(f"❌ {error_msg}")
            log_event("trade.chain_config_exception", level="ERROR", symbol=symbol, chain=chain_id, error=str(e))
            return None, False
        
        # Execute trades for each slice
        successful_txs = []
        total_successful_amount = 0
        
        for i, slice_amount in enumerate(slices):
            if slice_amount <= 0:
                continue
                
            log_event("trade.slice.start", index=i+1, slices=len(slices), amount_usd=round(slice_amount, 2))
            
            # Real trading mode (test_mode is false in config)
            if chain_id == "ethereum":
                # Use existing uniswap executor for Ethereum (MetaMask)
                from src.execution.uniswap_executor import buy_token
                tx_hash, ok = buy_token(token_address, slice_amount, symbol)
            elif chain_id == "base":
                # Use BASE executor for Base chain
                from src.execution.base_executor import buy_token
                tx_hash, ok = buy_token(token_address, slice_amount, symbol)
            elif chain_id == "solana":
                # Try Jupiter first, then fallback to Raydium if Jupiter fails
                from src.execution.jupiter_executor import buy_token_solana
                from src.execution.raydium_executor import execute_raydium_fallback_trade

                # Check if this is a volatile token (like BONK) that should use Raydium first
                volatile_tokens = [
                    "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",  # BONK
                ]

                if token_address in volatile_tokens:
                    log_event("trade.raydium.preferred", index=i+1)
                    try:
                        raydium_ok, raydium_tx = execute_raydium_fallback_trade(token_address, symbol, slice_amount)

                        if raydium_ok:
                            log_event("trade.slice.success", index=i+1, tx_hash=raydium_tx)
                            tx_hash = raydium_tx
                            ok = True
                        else:
                            log_event("trade.raydium.failed_try_jupiter", index=i+1)
                            tx_hash, ok = buy_token_solana(
                                token_address,
                                slice_amount,
                                symbol,
                                test_mode=False,  # LIVE TRADING ENABLED
                                slippage=dynamic_slippage,
                                route_preferences=route_preferences,
                                use_exactout=use_exactout,
                            )
                    except Exception as e:
                        log_event("trade.raydium.error_try_jupiter", level="WARNING", index=i+1, error=str(e))
                        tx_hash, ok = buy_token_solana(
                            token_address,
                            slice_amount,
                            symbol,
                            test_mode=False,  # LIVE TRADING ENABLED
                            slippage=dynamic_slippage,
                            route_preferences=route_preferences,
                            use_exactout=use_exactout,
                        )
                else:
                    log_event("trade.jupiter.attempt", index=i+1)
                    try:
                        tx_hash, ok = buy_token_solana(
                            token_address,
                            slice_amount,
                            symbol,
                            test_mode=False,  # LIVE TRADING ENABLED
                            slippage=dynamic_slippage,
                            route_preferences=route_preferences,
                            use_exactout=use_exactout,
                        )

                        if not ok:
                            log_event("trade.jupiter.failed_try_raydium", index=i+1)
                            try:
                                raydium_ok, raydium_tx = execute_raydium_fallback_trade(
                                    token_address, symbol, slice_amount
                                )

                                if raydium_ok:
                                    log_event("trade.slice.success", index=i+1, tx_hash=raydium_tx)
                                    tx_hash = raydium_tx
                                    ok = True
                            except Exception as e:
                                log_event("trade.raydium.fallback_error", level="WARNING", index=i+1, error=str(e))
                    except Exception as e:
                        log_event("trade.jupiter.error", level="WARNING", index=i+1, error=str(e))
                        ok = False
                        tx_hash = None
            
            # Verify fill using AI Fill Verifier (for Solana trades)
            fill_result = None
            tokens_received = None
            if ok and tx_hash and chain_id == "solana":
                try:
                    from src.ai.ai_fill_verifier import get_fill_verifier
                    from src.execution.jupiter_executor import JupiterCustomExecutor
                    from src.execution.raydium_executor import RaydiumExecutor
                    
                    fill_verifier = get_fill_verifier()
                    executor = JupiterCustomExecutor()
                    
                    # Get tokens received from transaction
                    tokens_received = None
                    amount_usd_actual = slice_amount  # Default assumption
                    
                    try:
                        # Try to get actual tokens received from balance check
                        time.sleep(2)  # Wait for transaction to settle
                        balance_before = executor.get_token_raw_balance(token_address)
                        if balance_before is not None:
                            time.sleep(1)
                            balance_after = executor.get_token_raw_balance(token_address)
                            if balance_after is not None and balance_after > balance_before:
                                tokens_received = balance_after - balance_before
                                # Estimate USD value
                                price = executor.get_token_price_usd(token_address)
                                if price and price > 0:
                                    amount_usd_actual = tokens_received * price
                    except Exception as e:
                        log_event("trade.fill_check_error", level="WARNING", error=str(e))
                    
                    # Verify fill
                    intent = {
                        "token_address": token_address,
                        "symbol": symbol,
                        "amount_usd": slice_amount
                    }
                    initial_attempt = {
                        "tx_hash": tx_hash,
                        "tokens_received": tokens_received,
                        "amount_usd_actual": amount_usd_actual,
                        "route": "jupiter"
                    }
                    executors = {
                        "jupiter": executor,
                        "raydium": RaydiumExecutor() if chain_id == "solana" else None
                    }
                    
                    fill_result = fill_verifier.verify_and_finalize_entry(
                        intent, initial_attempt, executors, chain_id
                    )
                    
                    if fill_result.status == "accepted" or fill_result.status == "rerouted":
                        # ENHANCED: Ensure we have actual token receipt confirmation
                        if fill_result.tokens_received is None or fill_result.tokens_received <= 0:
                            # Try to verify balance one more time
                            try:
                                time.sleep(2)  # Wait for transaction to settle
                                balance_after = executor.get_token_raw_balance(token_address)
                                if balance_after is not None:
                                    balance_before = executor.get_token_raw_balance(token_address)  # This should be cached
                                    if balance_after > (balance_before or 0):
                                        fill_result.tokens_received = balance_after - (balance_before or 0)
                                        # Recalculate USD amount
                                        price = executor.get_token_price_usd(token_address)
                                        if price and price > 0:
                                            fill_result.amount_usd_actual = fill_result.tokens_received * price
                            except Exception as e:
                                log_event("trade.balance_verification_error", level="WARNING", error=str(e))
                        
                        # Only proceed if we have confirmed tokens
                        if fill_result.tokens_received and fill_result.tokens_received > 0:
                            successful_txs.append(fill_result.tx_hash or tx_hash)
                            total_successful_amount += fill_result.amount_usd_actual or slice_amount
                            log_event("trade.slice.success", index=i+1, tx_hash=fill_result.tx_hash or tx_hash, fill_verified=True, tokens_received=fill_result.tokens_received)
                        else:
                            # Fill verification failed - no tokens received
                            log_event("trade.slice.no_tokens_received", level="WARNING", index=i+1, reason="No tokens received despite tx hash")
                            ok = False
                            tx_hash = None
                            fill_result = None
                    else:
                        # Fill verification failed - abort this slice
                        log_event("trade.slice.fill_verification_failed", level="WARNING", index=i+1, reason=fill_result.error_message)
                        ok = False
                        tx_hash = None
                        fill_result = None
                except Exception as e:
                    log_event("trade.fill_verifier_error", level="WARNING", error=str(e))
                    # On error, proceed with original result only if we have tokens_received
                    if ok and tx_hash:
                        if tokens_received and tokens_received > 0:
                            successful_txs.append(tx_hash)
                            total_successful_amount += slice_amount
                            log_event("trade.slice.success", index=i+1, tx_hash=tx_hash)
                        else:
                            log_event("trade.slice.no_verification", level="WARNING", index=i+1, reason="Cannot verify tokens received")
                            ok = False
                            tx_hash = None
            
            # Track successful slices (for non-Solana or if fill verifier not used)
            elif ok and tx_hash:
                successful_txs.append(tx_hash)
                total_successful_amount += slice_amount
                log_event("trade.slice.success", index=i+1, tx_hash=tx_hash)
            
            if not ok or not tx_hash:
                log_event("trade.slice.failure", level="WARNING", index=i+1)
                
                # For ExactOut trades, continue with remaining slices even if some fail
                if use_exactout:
                    log_event("trade.slice.continue_exactout", index=i+1)
                    continue
                else:
                    # For regular trades, stop on first failure
                    log_event("trade.slice.stop_on_failure", level="WARNING", index=i+1)
                    break
            
            # Add delay between slices to avoid overwhelming the network
            if i < len(slices) - 1:
                time.sleep(2)
        
        # Execute trades for each slice
        if successful_txs:
            log_event(
                "trade.end",
                successful_slices=len(successful_txs),
                total_slices=len(slices),
                total_filled_usd=round(total_successful_amount, 2),
                requested_usd=round(amount_usd, 2),
                transactions=successful_txs,
            )
            
            # Prepare token payloads for downstream logging
            token_for_logging = dict(token)
            if "chainId" not in token_for_logging:
                token_for_logging["chainId"] = chain_id
            token_for_logging["position_size_usd"] = total_successful_amount or amount_usd

            trade_id = None
            fee_data = {}
            
            # CRITICAL FIX: Analyze transactions BEFORE logging to get actual execution data
            # This ensures entry_amount_usd_actual is set correctly
            try:
                # Analyze the first successful transaction to get fee data
                primary_tx_hash = successful_txs[0] if successful_txs else None
                
                if primary_tx_hash:
                    # Import transaction analyzer based on chain
                    if chain_id == "solana":
                        from src.utils.solana_transaction_analyzer import analyze_jupiter_transaction
                        from src.config.secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS
                        
                        # Retry logic for transaction analysis (up to 3 attempts)
                        max_retries = 3
                        retry_delay = 2  # seconds
                        
                        for attempt in range(max_retries):
                            try:
                                # Wait a bit for transaction to settle on-chain
                                if attempt > 0:
                                    time.sleep(retry_delay * attempt)
                                
                                analyzed_data = analyze_jupiter_transaction(
                                    SOLANA_RPC_URL,
                                    primary_tx_hash,
                                    SOLANA_WALLET_ADDRESS,
                                    is_buy=True,
                                    quoted_output_amount=None
                                )
                                
                                # Check if we got valid data
                                if analyzed_data.get('actual_cost_usd', 0) > 0 or analyzed_data.get('tokens_received', 0) > 0:
                                    fee_data = {
                                        'entry_gas_fee_usd': analyzed_data.get('gas_fee_usd', 0),
                                        'entry_amount_usd_actual': analyzed_data.get('actual_cost_usd', 0),
                                        'entry_tokens_received': analyzed_data.get('tokens_received'),
                                        'buy_tx_hash': primary_tx_hash,
                                        'buy_slippage_actual': analyzed_data.get('actual_slippage')
                                    }
                                    log_event("trading.transaction_analysis_success", 
                                            symbol=symbol, 
                                            attempt=attempt+1,
                                            actual_cost=fee_data.get('entry_amount_usd_actual', 0),
                                            tokens_received=fee_data.get('entry_tokens_received'))
                                    break
                                else:
                                    log_event("trading.transaction_analysis_empty", 
                                            level="WARNING",
                                            symbol=symbol, 
                                            attempt=attempt+1)
                            except Exception as e:
                                log_event("trading.transaction_analysis_error", 
                                        level="WARNING",
                                        symbol=symbol, 
                                        attempt=attempt+1,
                                        error=str(e))
                                if attempt == max_retries - 1:
                                    # Last attempt failed - log warning but continue
                                    log_event("trading.transaction_analysis_failed", 
                                            level="ERROR",
                                            symbol=symbol,
                                            error=str(e))
                    
                    elif chain_id in ["ethereum", "base", "arbitrum", "polygon"]:
                        from src.utils.transaction_analyzer import analyze_buy_transaction
                        from src.execution.uniswap_executor import w3
                        
                        # Retry logic for transaction analysis
                        max_retries = 3
                        retry_delay = 2
                        
                        for attempt in range(max_retries):
                            try:
                                if attempt > 0:
                                    time.sleep(retry_delay * attempt)
                                
                                analyzed_data = analyze_buy_transaction(w3, primary_tx_hash)
                                
                                if analyzed_data.get('actual_cost_usd', 0) > 0:
                                    fee_data = {
                                        'entry_gas_fee_usd': analyzed_data.get('gas_fee_usd', 0),
                                        'entry_amount_usd_actual': analyzed_data.get('actual_cost_usd', 0),
                                        'buy_tx_hash': primary_tx_hash
                                    }
                                    log_event("trading.transaction_analysis_success", 
                                            symbol=symbol, 
                                            attempt=attempt+1,
                                            actual_cost=fee_data.get('entry_amount_usd_actual', 0))
                                    break
                                else:
                                    log_event("trading.transaction_analysis_empty", 
                                            level="WARNING",
                                            symbol=symbol, 
                                            attempt=attempt+1)
                            except Exception as e:
                                log_event("trading.transaction_analysis_error", 
                                        level="WARNING",
                                        symbol=symbol, 
                                        attempt=attempt+1,
                                        error=str(e))
                                if attempt == max_retries - 1:
                                    log_event("trading.transaction_analysis_failed", 
                                            level="ERROR",
                                            symbol=symbol,
                                            error=str(e))
                
                # Get current window_score for tracking (if available)
                try:
                    from src.ai.ai_time_window_scheduler import get_time_window_scheduler
                    scheduler = get_time_window_scheduler()
                    # Get current window score without triggering recalculation
                    current_window_score = getattr(scheduler, 'current_window_score', None)
                    if current_window_score is not None:
                        fee_data['window_score'] = current_window_score
                except Exception as e:
                    log_event("trading.window_score_fetch_error", level="WARNING", error=str(e))
                
                # Verify we have actual execution data before logging
                entry_amount_actual = fee_data.get('entry_amount_usd_actual', 0) or 0
                tokens_received = fee_data.get('entry_tokens_received')
                
                # Only log trade if we have confirmed execution
                if entry_amount_actual > 0 or (tokens_received is not None and tokens_received > 0):
                    # Log trade entry to performance tracker with fee data
                    try:
                        from src.core.performance_tracker import performance_tracker

                        quality_score = float(token.get("quality_score", 0.0))
                        trade_id = performance_tracker.log_trade_entry(
                            token_for_logging,
                            total_successful_amount or amount_usd,
                            quality_score,
                            additional_data=fee_data  # Pass fee_data here (includes window_score)
                        )
                        log_event("trading.performance_logged", symbol=symbol, 
                                entry_amount_actual=entry_amount_actual,
                                tokens_received=tokens_received)
                    except Exception as e:
                        log_event(
                            "trading.performance_log_error",
                            level="WARNING",
                            symbol=symbol,
                            error=str(e),
                        )
                else:
                    # No confirmed execution - don't log trade
                    log_event("trading.trade_not_logged", 
                            level="WARNING",
                            symbol=symbol,
                            reason="No confirmed execution data (entry_amount_usd_actual=0 and no tokens_received)")
            except Exception as e:
                log_event(
                    "trading.transaction_analysis_critical_error",
                    level="ERROR",
                    symbol=symbol,
                    error=str(e),
                )

            # Log position for monitoring (use canonical key schema)
            # ONLY log if we have verified that tokens were actually received
            try:
                # Check if we have verification data indicating successful trade
                should_log_position = False
                verification_reason = None
                
                # Option 1: Check performance tracker data for actual tokens received
                if trade_id:
                    try:
                        from src.core.performance_tracker import performance_tracker
                        # Find the trade we just logged
                        for trade in performance_tracker.trades:
                            if trade.get('id') == trade_id:
                                entry_amount = trade.get('entry_amount_usd_actual', 0) or 0
                                tokens_received = trade.get('entry_tokens_received')
                                # Only log if we have actual tokens or confirmed USD amount
                                if entry_amount > 0 or (tokens_received is not None and tokens_received > 0):
                                    should_log_position = True
                                    verification_reason = f"performance_tracker: entry_amount={entry_amount}, tokens={tokens_received}"
                                break
                    except Exception as e:
                        log_event("trading.position_verification_error", level="WARNING", error=str(e))
                
                # Option 2: Check if we have tokens_received from balance check (for non-Solana or fallback)
                if not should_log_position and total_successful_amount > 0:
                    # If we have successful amount, assume trade succeeded (for non-Solana chains)
                    # This is a fallback for chains without fill verification
                    if chain_id != "solana":
                        should_log_position = True
                        verification_reason = f"successful_amount: {total_successful_amount}"
                    else:
                        # For Solana, we need more verification
                        log_event("trading.position_verification_needed", level="WARNING", symbol=symbol, reason="Solana trade needs token verification")
                
                if should_log_position:
                    # Ensure token_for_logging has priceUsd for _log_position
                    if "priceUsd" not in token_for_logging:
                        # Try to get price from token data
                        token_for_logging["priceUsd"] = token.get("priceUsd") or token.get("entry_price") or token.get("price") or 0.0
                    
                    # Log position using the helper function (creates proper position_key and position_data)
                    _log_position(token_for_logging, trade_id=trade_id)
                    log_event("trading.position_logged", symbol=symbol, reason=verification_reason)
                    
                    # CRITICAL: Validate position size against actual wallet balance
                    # This fixes the issue where a buy fails and retries with smaller amount,
                    # but the position was logged with the original intended amount
                    # Wait a brief moment for blockchain state to update
                    import time
                    time.sleep(1)
                    try:
                        _validate_and_update_position_size(token_for_logging, chain_id, token_address)
                    except Exception as e:
                        log_event("trading.position_validation_error", level="WARNING", symbol=symbol, error=str(e))
                        # Don't fail the position logging if validation fails
                else:
                    log_event(
                        "trading.position_not_logged",
                        level="WARNING",
                        symbol=symbol,
                        reason="No verified tokens received - trade may have failed"
                    )
                    # Mark trade as failed in performance tracker if we can
                    if trade_id:
                        try:
                            from src.core.performance_tracker import performance_tracker
                            for trade in performance_tracker.trades:
                                if trade.get('id') == trade_id:
                                    if (trade.get('entry_amount_usd_actual', 0) or 0) == 0:
                                        trade['status'] = 'manual_close'
                                        trade['exit_time'] = trade.get('entry_time')
                                        trade['exit_price'] = trade.get('entry_price', 0)
                                        trade['pnl_usd'] = 0.0
                                        trade['pnl_percent'] = 0.0
                                        performance_tracker.save_data()
                                        log_event("trading.trade_marked_failed", trade_id=trade_id)
                                    break
                        except Exception as e:
                            log_event("trading.mark_failed_error", level="WARNING", error=str(e))
            except Exception as e:
                log_event(
                    "trading.position_log_error",
                    level="WARNING",
                    symbol=symbol,
                    error=str(e),
                )

            _launch_monitor_detached()
            
            return successful_txs[0], True  # Return first transaction hash
        else:
            log_event("trade.end", level="ERROR", successful_slices=0, total_slices=len(slices))
            return None, False
        
    except Exception as e:
        log_event("trade.error", level="ERROR", chain=chain_id, error=str(e))
        return None, False

def get_supported_chains():
    """Return list of supported chains"""
    return list(CHAIN_CONFIGS.keys())

def is_chain_supported(chain_id: str) -> bool:
    """Check if a chain is supported"""
    return chain_id.lower() in CHAIN_CONFIGS
