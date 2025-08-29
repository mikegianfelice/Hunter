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
    Multi-chain trade execution:
    1. Determine the chain from token data
    2. Get chain-specific configuration
    3. Delegate to appropriate executor
    4. Log position and launch monitor if successful
    
    Returns: (tx_hash_hex_or_sim, success_bool)
    """
    config = get_multi_chain_config()
    symbol = token.get("symbol", "?")
    token_address = token["address"]
    chain_id = token.get("chainId", "ethereum").lower()
    amount_usd = float(trade_amount_usd or config['TRADE_AMOUNT_USD_DEFAULT'])
    
    print(f"🚀 Executing trade on {chain_id.upper()}: {symbol} ({token_address})")
    
    try:
        # Get chain configuration
        chain_config = get_chain_config(chain_id)
        
        # Real trading mode (test_mode is false in config)
        if chain_id == "ethereum":
            # Use existing uniswap executor for Ethereum (MetaMask)
            from uniswap_executor import buy_token
            tx_hash, ok = buy_token(token_address, amount_usd, symbol)
        elif chain_id == "solana":
            # Use Jupiter executor for Solana trading
            from jupiter_executor import buy_token_solana
            tx_hash, ok = buy_token_solana(token_address, amount_usd, symbol, test_mode=False)
        else:
            # For unsupported chains, skip
            print(f"❌ Chain {chain_id.upper()} not supported - only Ethereum and Solana enabled")
            return None, False
            
        if not ok:
            print(f"❌ Trade failed for {symbol} on {chain_id}")
            return None, False
            
        # Log position and launch monitor
        _log_position(token)
        _launch_monitor_detached()
        
        print(f"✅ Trade executed successfully on {chain_id.upper()}: {symbol}")
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
