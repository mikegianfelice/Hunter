import yaml
import json
import time
import sys
import subprocess
import os
from pathlib import Path
from typing import Optional, Dict, Any
from web3 import Web3

from secrets import INFURA_URL, WALLET_ADDRESS, SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY
from utils import get_eth_price_usd
from telegram_bot import send_telegram_message
from config_loader import get_config, get_config_bool, get_config_float

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
        "executor_module": "uniswap_executor"
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

def _log_position(token: dict):
    _ensure_positions_file()
    try:
        data = json.loads(Path(POSITIONS_FILE).read_text() or "{}")
    except Exception:
        data = {}
    addr = token["address"]
    entry = float(token.get("priceUsd") or 0.0)
    data[addr] = entry
    Path(POSITIONS_FILE).write_text(json.dumps(data, indent=2))
    print(f"📝 Logged position: {token.get('symbol','?')} ({addr}) @ ${entry:.6f}")

def _launch_monitor_detached():
    script = Path(MONITOR_SCRIPT).resolve()
    if not script.exists():
        print(f"⚠️ {MONITOR_SCRIPT} not found at {script}")
        return
    try:
        subprocess.Popen([sys.executable, str(script)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"👁️ Started {MONITOR_SCRIPT} via {sys.executable}")
    except Exception as e:
        print(f"⚠️ Could not launch {MONITOR_SCRIPT}: {e}")

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
    amount_usd = float(trade_amount_usd or config['TRADE_AMOUNT_USD_DEFAULT'])
    
    print(f"🚀 Executing advanced trade on {chain_id.upper()}: {symbol} ({token_address})")
    
    # Import advanced trading engine
    from advanced_trading import advanced_trading
    
    # Enhanced preflight check
    try:
        preflight_passed, reason = advanced_trading.enhanced_preflight_check(token, amount_usd)
        if not preflight_passed:
            print(f"❌ Preflight check failed: {reason}")
            return None, False
    except Exception as e:
        print(f"⚠️ Preflight check error: {e}")
        # Continue with trade if preflight fails
    
    # Calculate order slices
    slices = advanced_trading.calculate_order_slices(amount_usd, token)
    
    # Calculate dynamic slippage
    base_slippage = config['SLIPPAGE']
    dynamic_slippage = advanced_trading.calculate_dynamic_slippage(token, base_slippage)
    
    # Determine if ExactOut should be used
    use_exactout = advanced_trading.should_use_exactout(token)
    
    # Get route preferences
    route_preferences = advanced_trading.get_route_preferences(token)
    
    print(f"📊 Trade configuration:")
    print(f"   - Slices: {len(slices)} (${amount_usd:.2f} total)")
    print(f"   - Slippage: {dynamic_slippage*100:.2f}% (dynamic)")
    print(f"   - ExactOut: {'Yes' if use_exactout else 'No'}")
    print(f"   - Route preferences: {route_preferences}")
    
    try:
        # Get chain configuration
        chain_config = get_chain_config(chain_id)
        
        # Execute trades for each slice
        successful_txs = []
        total_successful_amount = 0
        
        for i, slice_amount in enumerate(slices):
            if slice_amount <= 0:
                continue
                
            print(f"🔄 Executing slice {i+1}/{len(slices)}: ${slice_amount:.2f}")
            
            # Real trading mode (test_mode is false in config)
            if chain_id == "ethereum":
                # Use existing uniswap executor for Ethereum (MetaMask)
                from uniswap_executor import buy_token
                tx_hash, ok = buy_token(token_address, slice_amount, symbol)
            elif chain_id == "solana":
                # Try Jupiter first, then fallback to Raydium if Jupiter fails
                from jupiter_executor import buy_token_solana
                from raydium_executor import execute_raydium_fallback_trade
            
            # Check if this is a volatile token (like BONK) that should use Raydium first
            volatile_tokens = [
                "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",  # BONK
                "7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr",  # PEPE
                "EPeUFDgHRxs9xxEPVaL6kfGQvCon7jmAWKVUHuux1Tpz",  # JITO
            ]
            
            if token_address in volatile_tokens:
                print(f"🔥 Volatile token detected - using Raydium first")
                raydium_ok, raydium_tx = execute_raydium_fallback_trade(token_address, symbol, slice_amount)
                
                if raydium_ok:
                    print(f"✅ Raydium slice {i+1} successful")
                    tx_hash = raydium_tx
                    ok = True
                else:
                                    print(f"🔄 Raydium failed, trying Jupiter...")
                tx_hash, ok = buy_token_solana(token_address, slice_amount, symbol, test_mode=False, 
                                              slippage=dynamic_slippage, route_preferences=route_preferences, 
                                              use_exactout=use_exactout)
            else:
                print(f"🔄 Attempting Jupiter trade for slice {i+1}...")
                tx_hash, ok = buy_token_solana(token_address, slice_amount, symbol, test_mode=False, 
                                              slippage=dynamic_slippage, route_preferences=route_preferences, 
                                              use_exactout=use_exactout)
                
                if not ok:
                    print(f"🔄 Jupiter failed, trying Raydium...")
                    raydium_ok, raydium_tx = execute_raydium_fallback_trade(token_address, symbol, slice_amount)
                    
                    if raydium_ok:
                        print(f"✅ Raydium fallback successful for slice {i+1}")
                        tx_hash = raydium_tx
                        ok = True
            
            # Track successful slices
            if ok and tx_hash:
                successful_txs.append(tx_hash)
                total_successful_amount += slice_amount
                print(f"✅ Slice {i+1} executed successfully: {tx_hash}")
            else:
                print(f"❌ Slice {i+1} failed")
                
                # For ExactOut trades, continue with remaining slices even if some fail
                if use_exactout:
                    print(f"🔄 Continuing with remaining slices (ExactOut mode)")
                    continue
                else:
                    # For regular trades, stop on first failure
                    print(f"❌ Stopping execution due to slice failure")
                    break
            
            # Add delay between slices to avoid overwhelming the network
            if i < len(slices) - 1:
                time.sleep(2)
        
        # Execute trades for each slice
        if successful_txs:
            print(f"✅ Trade execution completed:")
            print(f"   - Successful slices: {len(successful_txs)}/{len(slices)}")
            print(f"   - Total amount: ${total_successful_amount:.2f}/{amount_usd:.2f}")
            print(f"   - Transactions: {successful_txs}")
            
            # Log position for monitoring (use total successful amount)
            _log_position(token)
            _launch_monitor_detached()
            
            return successful_txs[0], True  # Return first transaction hash
        else:
            print(f"❌ All slices failed")
            return None, False
        return tx_hash, True
        
    except Exception as e:
        print(f"❌ Error executing trade on {chain_id}: {e}")
        return None, False

def get_supported_chains():
    """Return list of supported chains"""
    return list(CHAIN_CONFIGS.keys())

def is_chain_supported(chain_id: str) -> bool:
    """Check if a chain is supported"""
    return chain_id.lower() in CHAIN_CONFIGS
