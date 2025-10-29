import yaml
import json
import time
import sys
import subprocess
import os
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Dict, Any
from web3 import Web3

from secrets import INFURA_URL, WALLET_ADDRESS, SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY
from utils import get_eth_price_usd
from telegram_bot import send_telegram_message
from config_loader import get_config, get_config_bool, get_config_float
from logger import log_event
from advanced_trading import advanced_trading
from address_utils import validate_chain_address_match, normalize_evm_address, detect_chain_from_address

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
POSITIONS_FILE = "open_positions.json"
MONITOR_SCRIPT = "monitor_position.py"
PRICE_MEMORY_FILE = "price_memory.json"

def get_chain_config(chain_id: str) -> Dict[str, Any]:
    """Get configuration for a specific chain"""
    chain_id_lower = chain_id.lower()
    if chain_id_lower not in CHAIN_CONFIGS:
        raise ValueError(f"Unsupported chain: {chain_id}")
    return CHAIN_CONFIGS[chain_id_lower]

def _ensure_positions_file():
    p = Path(POSITIONS_FILE)
    if not p.exists():
        p.write_text("{}")

@contextmanager
def _atomic_write_json(path: Path):
    tmp_path = Path(str(path) + ".tmp")
    try:
        with open(tmp_path, "w") as f:
            yield f
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

def _log_position(token: dict):
    _ensure_positions_file()
    try:
        data = json.loads(Path(POSITIONS_FILE).read_text() or "{}")
    except Exception:
        data = {}
    addr = token["address"]
    entry = float(token.get("priceUsd") or 0.0)
    chain_id = token.get("chainId", "ethereum").lower()
    
    # Store position with chain information
    data[addr] = {
        "entry_price": entry,
        "chain_id": chain_id,
        "symbol": token.get("symbol", "?"),
        "timestamp": datetime.now().isoformat()
    }
    
    # Atomic write to prevent corruption in concurrent environments
    target = Path(POSITIONS_FILE)
    with _atomic_write_json(target) as f:
        f.write(json.dumps(data, indent=2))
    try:
        print(f"📝 Logged position: {token.get('symbol','?')} ({addr}) on {chain_id.upper()} @ ${entry:.6f}")
    except BrokenPipeError:
        pass

def _launch_monitor_detached():
    script = Path(MONITOR_SCRIPT).resolve()
    if not script.exists():
        try:
            print(f"⚠️ {MONITOR_SCRIPT} not found at {script}")
        except BrokenPipeError:
            pass
        return
    try:
        subprocess.Popen([sys.executable, str(script)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            print(f"👁️ Started {MONITOR_SCRIPT} via {sys.executable}")
        except BrokenPipeError:
            pass
    except Exception as e:
        try:
            print(f"⚠️ Could not launch {MONITOR_SCRIPT}: {e}")
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
                from uniswap_executor import buy_token
                tx_hash, ok = buy_token(token_address, slice_amount, symbol)
            elif chain_id == "base":
                # Use BASE executor for Base chain
                from base_executor import buy_token
                tx_hash, ok = buy_token(token_address, slice_amount, symbol)
            elif chain_id == "solana":
                # Try Jupiter first, then fallback to Raydium if Jupiter fails
                from jupiter_executor import buy_token_solana
                from raydium_executor import execute_raydium_fallback_trade

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
            
            # Track successful slices
            if ok and tx_hash:
                successful_txs.append(tx_hash)
                total_successful_amount += slice_amount
                log_event("trade.slice.success", index=i+1, tx_hash=tx_hash)
            else:
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
            
            # Log position for monitoring (use total successful amount)
            _log_position(token)
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
