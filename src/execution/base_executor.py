# base_executor.py
from web3 import Web3
import json
import yaml
import time
import os
import sys
from pathlib import Path

from src.config.secrets import BASE_RPC_URL, WALLET_ADDRESS, PRIVATE_KEY
from gas import suggest_fees
from utils import get_eth_price_usd  # robust ETH/USD (Graph -> on-chain V2)
from config_loader import get_config, get_config_bool, get_config_float, get_config_int
from src.monitoring.logger import log_event

# Dynamic config loading
def get_base_config():
    """Get current configuration values dynamically"""
    return {
        'TEST_MODE': get_config_bool("test_mode", True),
        'SLIPPAGE': get_config_float("slippage", 0.02),
        'USE_SUPPORTING_FEE': get_config_bool("use_supporting_fee_swap", True),
        'REQUOTE_DELAY_SECONDS': get_config_int("requote_delay_seconds", 10),
        'REQUOTE_SLIPPAGE_BUFFER': get_config_float("requote_slippage_buffer", 0.005),
        'GAS_CFG': {
            "gas_blocks": get_config("gas_blocks"),
            "gas_reward_percentile": get_config("gas_reward_percentile"),
            "gas_basefee_headroom": get_config("gas_basefee_headroom"),
            "gas_priority_min_gwei": get_config("gas_priority_min_gwei"),
            "gas_priority_max_gwei": get_config("gas_priority_max_gwei"),
            "gas_ceiling_gwei": get_config("gas_ceiling_gwei"),
            "gas_multiplier": get_config("gas_multiplier"),
            "gas_extra_priority_gwei": get_config("gas_extra_priority_gwei"),
        }
    }

# === Web3 / Router setup ===
if not BASE_RPC_URL:
    raise RuntimeError("BASE_RPC_URL missing. Put it in your .env and secrets.py.")
if not WALLET_ADDRESS or not PRIVATE_KEY:
    raise RuntimeError("WALLET_ADDRESS / PRIVATE_KEY missing from .env.")

w3 = Web3(Web3.HTTPProvider(BASE_RPC_URL))
if not w3.is_connected():
    raise RuntimeError("Web3 is not connected to BASE. Check your RPC URL / network.")

WALLET = Web3.to_checksum_address(WALLET_ADDRESS)

# BASE chain configuration
BASE_CHAIN_ID = 8453
BASE_WETH = Web3.to_checksum_address("0x4200000000000000000000000000000000000006")  # WETH on Base

# Uniswap V3 router on BASE
ROUTER_ADDR = Web3.to_checksum_address("0x4752ba5DBc23f44D87826276BF6Fd6b1C372aD24")  # Uniswap V3 Router on Base

# Uniswap V3 Router ABI (simplified for swapExactETHForTokens)
ROUTER_ABI = json.loads("""
[
  {
    "inputs": [
      {
        "components": [
          {"internalType": "address", "name": "tokenIn", "type": "address"},
          {"internalType": "address", "name": "tokenOut", "type": "address"},
          {"internalType": "uint24", "name": "fee", "type": "uint24"},
          {"internalType": "address", "name": "recipient", "type": "address"},
          {"internalType": "uint256", "name": "deadline", "type": "uint256"},
          {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
          {"internalType": "uint256", "name": "amountOutMinimum", "type": "uint256"},
          {"internalType": "uint160", "name": "sqrtPriceLimitX96", "type": "uint160"}
        ],
        "internalType": "struct ISwapRouter.ExactInputSingleParams",
        "name": "params",
        "type": "tuple"
      }
    ],
    "name": "exactInputSingle",
    "outputs": [{"internalType": "uint256", "name": "amountOut", "type": "uint256"}],
    "stateMutability": "payable",
    "type": "function"
  },
  {
    "inputs": [
      {
        "components": [
          {"internalType": "address", "name": "tokenIn", "type": "address"},
          {"internalType": "address", "name": "tokenOut", "type": "address"},
          {"internalType": "uint24", "name": "fee", "type": "uint24"},
          {"internalType": "address", "name": "recipient", "type": "address"},
          {"internalType": "uint256", "name": "deadline", "type": "uint256"},
          {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
          {"internalType": "uint256", "name": "amountOutMinimum", "type": "uint256"},
          {"internalType": "uint160", "name": "sqrtPriceLimitX96", "type": "uint160"}
        ],
        "internalType": "struct ISwapRouter.ExactInputSingleParams",
        "name": "params",
        "type": "tuple"
      }
    ],
    "name": "exactInputSingleSupportingFeeOnTransferTokens",
    "outputs": [{"internalType": "uint256", "name": "amountOut", "type": "uint256"}],
    "stateMutability": "payable",
    "type": "function"
  }
]
""")

router = w3.eth.contract(address=ROUTER_ADDR, abi=ROUTER_ABI)

# Minimal ERC20 ABI for token operations
ERC20_MIN_ABI = json.loads("""
[
  {"constant":true,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"type":"function"},
  {"constant":true,"inputs":[{"name":"owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"type":"function"},
  {"constant":true,"inputs":[{"name":"owner","type":"address"},{"name":"spender","type":"address"}],"name":"allowance","outputs":[{"name":"","type":"uint256"}],"type":"function"},
  {"constant":false,"inputs":[{"name":"spender","type":"address"},{"name":"amount","type":"uint256"}],"name":"approve","outputs":[{"name":"","type":"bool"}],"type":"function"}
]
""")

# === Helpers ===
def _next_nonce():
    return w3.eth.get_transaction_count(WALLET)

def _apply_eip1559(tx_dict: dict) -> dict:
    """
    Attach EIP-1559 gas fields using on-chain fee history, with config guardrails.
    """
    config = get_base_config()
    max_fee, max_prio = suggest_fees(w3, config['GAS_CFG'])
    out = dict(tx_dict)
    out["maxFeePerGas"] = int(max_fee)
    out["maxPriorityFeePerGas"] = int(max_prio)
    out["type"] = 2
    return out

def _erc20(token_addr):
    return w3.eth.contract(address=Web3.to_checksum_address(token_addr), abi=ERC20_MIN_ABI)

def _quote_v3_out(amount_in_wei: int, token_out: str, fee: int = 3000) -> int:
    """
    Quote expected token out using Uniswap V3 router.
    Returns raw token amount in smallest units.
    """
    try:
        # For V3, we call the router contract to get a quote (estimate swap output)
        # This uses the actual router contract call to get real quotes
        params = {
            'tokenIn': BASE_WETH,
            'tokenOut': Web3.to_checksum_address(token_out),
            'fee': fee,
            'recipient': WALLET,
            'deadline': int(time.time()) + 600,
            'amountIn': amount_in_wei,
            'amountOutMinimum': 0,
            'sqrtPriceLimitX96': 0
        }
        
        # Estimate the swap to get the quote
        result = router.functions.exactInputSingle(params).call({'value': amount_in_wei})
        return int(result)
    except Exception as e:
        print(f"⚠️ Quote failed: {e}")
        return 0

# === BUY with re-quote protection ===
def buy_token(token_address: str, usd_amount: float, symbol: str = "?") -> tuple[str, bool]:
    """
    Buy `token_address` spending `usd_amount` USD worth of ETH on BASE.
    - Uses V3 quote to compute amountOutMin with slippage
    - If delay between quote and send > REQUOTE_DELAY_SECONDS, re-quotes and
      recomputes minOut with an extra REQUOTE_SLIPPAGE_BUFFER
    - Uses EIP-1559 fees
    Returns: (tx_hash_hex or None, success_bool)
    """
    config = get_base_config()
    token_address = Web3.to_checksum_address(token_address)
    deadline = int(time.time()) + 600

    # Sizing: USD -> ETH
    eth_usd = get_eth_price_usd()
    if not eth_usd or eth_usd <= 0:
        print("❌ Could not fetch ETH/USD price for sizing.")
        return None, False
    eth_amount = float(usd_amount) / float(eth_usd)
    value_wei = w3.to_wei(eth_amount, "ether")

    base_tx = {
        "from": WALLET,
        "value": int(value_wei),
        "nonce": _next_nonce(),
        "chainId": BASE_CHAIN_ID,
    }

    # Initial quote - try different fee tiers
    quoted_out = 0
    fee_tiers = [3000, 500, 10000]  # 0.3%, 0.05%, 1%
    
    for fee in fee_tiers:
        try:
            quoted_out = _quote_v3_out(value_wei, token_address, fee)
            if quoted_out > 0:
                print(f"✅ Quote successful with {fee/10000}% fee tier")
                break
        except Exception as e:
            print(f"⚠️ Quote failed with {fee/10000}% fee: {e}")
            continue
    
    if quoted_out <= 0:
        print(f"❌ All quote attempts failed for {symbol}")
        return None, False

    amount_out_min = int(quoted_out * (1.0 - config['SLIPPAGE']))
    if amount_out_min <= 0:
        print(f"❌ Computed minOut <= 0 for {symbol}; aborting.")
        return None, False

    log_event("base.buy.quote", symbol=symbol, eth_in=round(eth_amount, 6), out_raw=int(quoted_out), slippage=round(config['SLIPPAGE'], 6), min_out=int(amount_out_min))

    # Build swap parameters
    params = {
        'tokenIn': BASE_WETH,
        'tokenOut': token_address,
        'fee': 3000,  # 0.3% fee tier
        'recipient': WALLET,
        'deadline': deadline,
        'amountIn': value_wei,
        'amountOutMinimum': amount_out_min,
        'sqrtPriceLimitX96': 0
    }

    # Build transaction
    if config['USE_SUPPORTING_FEE']:
        fn = router.functions.exactInputSingleSupportingFeeOnTransferTokens(params)
    else:
        fn = router.functions.exactInputSingle(params)

    tx = fn.build_transaction(base_tx)
    tx = _apply_eip1559(tx)

    # Estimate gas + buffer
    try:
        est = w3.eth.estimate_gas(tx)
        tx["gas"] = int(est * 1.2)
    except Exception:
        tx["gas"] = 300000

    # Re-quote protection: if we spent too long, re-quote and rebuild minOut
    t_quote = time.time()
    def _maybe_requote_and_adjust(tx_dict):
        elapsed = time.time() - t_quote
        if elapsed <= config['REQUOTE_DELAY_SECONDS']:
            return tx_dict  # still fresh

        print(f"⏱️ Quote stale ({elapsed:.1f}s). Re-quoting before send…")
        try:
            re_quoted_out = _quote_v3_out(value_wei, token_address, 3000)
        except Exception as e:
            print(f"⚠️ Re-quote failed ({e}); sending with original minOut.")
            return tx_dict

        if re_quoted_out <= 0:
            print("⚠️ Re-quote returned 0; sending with original minOut.")
            return tx_dict

        # Expand slippage slightly on re-quote to reduce false failures
        total_slip = config['SLIPPAGE'] + max(0.0, config['REQUOTE_SLIPPAGE_BUFFER'])
        new_min_out = int(re_quoted_out * (1.0 - total_slip))
        if new_min_out <= 0:
            print("⚠️ Re-quote minOut <= 0; keeping original minOut.")
            return tx_dict

        print(f"🔁 Re-quoted out: {re_quoted_out} → new minOut {new_min_out} (slip {total_slip*100:.2f}%)")
        
        # rebuild function with new minOut
        params['amountOutMinimum'] = new_min_out
        if config['USE_SUPPORTING_FEE']:
            fn2 = router.functions.exactInputSingleSupportingFeeOnTransferTokens(params)
        else:
            fn2 = router.functions.exactInputSingle(params)

        tx2 = fn2.build_transaction(base_tx)
        tx2 = _apply_eip1559(tx2)
        tx2["gas"] = tx_dict["gas"]  # keep same gas estimate
        return tx2

    # Final transaction with potential re-quote
    final_tx = _maybe_requote_and_adjust(tx)

    # Send transaction
    try:
        if config['TEST_MODE']:
            # In test mode, still build and validate transaction but don't send
            # This allows testing with real market data and quotes
            log_event("base.buy.test_mode", symbol=symbol, note="Transaction validated but not sent")
            # Return None to indicate no actual transaction was sent
            return None, True
        
        signed = w3.eth.account.sign_transaction(final_tx, PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
        log_event("base.buy.sent", symbol=symbol, tx_hash=tx_hash.hex())
        return tx_hash.hex(), True
    except Exception as e:
        log_event("base.buy.error", level="ERROR", symbol=symbol, error=str(e))
        return None, False

# === SELL functionality ===
def sell_token(token_address: str, token_amount: float, symbol: str = "?") -> tuple[str, bool]:
    """
    Sell `token_amount` of `token_address` for ETH on BASE.
    Returns: (tx_hash_hex or None, success_bool)
    """
    config = get_base_config()
    token_address = Web3.to_checksum_address(token_address)
    
    # Get token contract
    token_contract = _erc20(token_address)
    
    # Get token balance
    try:
        balance = token_contract.functions.balanceOf(WALLET).call()
        decimals = token_contract.functions.decimals().call()
        token_amount_wei = int(token_amount * (10 ** decimals))
        
        if balance < token_amount_wei:
            print(f"❌ Insufficient {symbol} balance: {balance} < {token_amount_wei}")
            return None, False
    except Exception as e:
        print(f"❌ Failed to get {symbol} balance: {e}")
        return None, False

    # Approve router to spend tokens
    if not config['TEST_MODE']:
        try:
            allowance = token_contract.functions.allowance(WALLET, ROUTER_ADDR).call()
            if allowance < token_amount_wei:
                print(f"🔐 Approving {symbol} spend...")
                approve_tx = token_contract.functions.approve(
                    ROUTER_ADDR, token_amount_wei
                ).build_transaction({
                    "from": WALLET,
                    "nonce": _next_nonce(),
                    "chainId": BASE_CHAIN_ID,
                })
                approve_tx = _apply_eip1559(approve_tx)
                approve_tx["gas"] = 100000
                
                signed = w3.eth.account.sign_transaction(approve_tx, PRIVATE_KEY)
                approve_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
                print(f"✅ {symbol} approval: {approve_hash.hex()}")
                
                # Wait for approval to be mined
                w3.eth.wait_for_transaction_receipt(approve_hash, timeout=60)
        except Exception as e:
            print(f"❌ {symbol} approval failed: {e}")
            return None, False

    # Build sell transaction
    deadline = int(time.time()) + 600
    
    params = {
        'tokenIn': token_address,
        'tokenOut': BASE_WETH,
        'fee': 3000,  # 0.3% fee tier
        'recipient': WALLET,
        'deadline': deadline,
        'amountIn': token_amount_wei,
        'amountOutMinimum': 0,  # Will be set after quote
        'sqrtPriceLimitX96': 0
    }

    # Quote the sell
    try:
        quoted_out = router.functions.exactInputSingle(params).call()
        amount_out_min = int(quoted_out * (1.0 - config['SLIPPAGE']))
        params['amountOutMinimum'] = amount_out_min
        
        print(f"🧮 Sell quote for {symbol}: in {token_amount_wei} → out {quoted_out} ETH wei")
        print(f"🎯 Slippage {config['SLIPPAGE']*100:.2f}% → minOut {amount_out_min}")
    except Exception as e:
        print(f"❌ Sell quote failed for {symbol}: {e}")
        return None, False

    # Build transaction
    if config['USE_SUPPORTING_FEE']:
        fn = router.functions.exactInputSingleSupportingFeeOnTransferTokens(params)
    else:
        fn = router.functions.exactInputSingle(params)

    tx = fn.build_transaction({
        "from": WALLET,
        "nonce": _next_nonce(),
        "chainId": BASE_CHAIN_ID,
    })
    tx = _apply_eip1559(tx)

    # Estimate gas
    try:
        est = w3.eth.estimate_gas(tx)
        tx["gas"] = int(est * 1.2)
    except Exception:
        tx["gas"] = 300000

    # Re-quote protection similar to buy path
    t_quote = time.time()
    def _maybe_requote_and_adjust(tx_dict):
        elapsed = time.time() - t_quote
        if elapsed <= config['REQUOTE_DELAY_SECONDS']:
            return tx_dict
        print(f"⏱️ Sell quote stale ({elapsed:.1f}s). Re-quoting before send…")
        try:
            re_quoted_out = router.functions.exactInputSingle(params).call()
        except Exception as e:
            print(f"⚠️ Sell re-quote failed ({e}); sending with original minOut.")
            return tx_dict
        if re_quoted_out <= 0:
            print("⚠️ Sell re-quote returned 0; sending with original minOut.")
            return tx_dict
        total_slip = config['SLIPPAGE'] + max(0.0, config['REQUOTE_SLIPPAGE_BUFFER'])
        new_min_out = int(re_quoted_out * (1.0 - total_slip))
        if new_min_out <= 0:
            print("⚠️ Sell re-quote minOut <= 0; keeping original minOut.")
            return tx_dict
        params['amountOutMinimum'] = new_min_out
        if config['USE_SUPPORTING_FEE']:
            fn2 = router.functions.exactInputSingleSupportingFeeOnTransferTokens(params)
        else:
            fn2 = router.functions.exactInputSingle(params)
        tx2 = fn2.build_transaction({
            "from": WALLET,
            "nonce": tx_dict.get("nonce", _next_nonce()),
            "chainId": BASE_CHAIN_ID,
        })
        tx2 = _apply_eip1559(tx2)
        tx2["gas"] = tx_dict.get("gas", 300000)
        return tx2

    # Send transaction
    try:
        if config['TEST_MODE']:
            # In test mode, still build and validate transaction but don't send
            # This allows testing with real market data and quotes
            log_event("base.sell.test_mode", symbol=symbol, note="Transaction validated but not sent")
            # Return None to indicate no actual transaction was sent
            return None, True
        
        final_tx = _maybe_requote_and_adjust(tx)
        signed = w3.eth.account.sign_transaction(final_tx, PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
        log_event("base.sell.sent", symbol=symbol, tx_hash=tx_hash.hex())
        return tx_hash.hex(), True
    except Exception as e:
        log_event("base.sell.error", level="ERROR", symbol=symbol, error=str(e))
        return None, False

# === Utility functions ===
def get_base_balance() -> float:
    """Get ETH balance on BASE chain"""
    try:
        balance_wei = w3.eth.get_balance(WALLET)
        balance_eth = w3.from_wei(balance_wei, "ether")
        return float(balance_eth)
    except Exception as e:
        print(f"❌ Failed to get BASE balance: {e}")
        return 0.0

def get_token_balance(token_address: str) -> float:
    """Get token balance on BASE chain"""
    try:
        token_contract = _erc20(token_address)
        balance_wei = token_contract.functions.balanceOf(WALLET).call()
        decimals = token_contract.functions.decimals().call()
        balance = balance_wei / (10 ** decimals)
        return float(balance)
    except Exception as e:
        print(f"❌ Failed to get token balance: {e}")
        return 0.0
