# base_executor.py
from web3 import Web3
import json
import yaml
import time
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

from src.config.secrets import BASE_RPC_URL, WALLET_ADDRESS, PRIVATE_KEY
from src.utils.gas import suggest_fees
from src.utils.utils import get_eth_price_usd  # robust ETH/USD (Graph -> on-chain V2)
from src.config.config_loader import get_config, get_config_bool, get_config_float, get_config_int
from src.monitoring.logger import log_event
from src.monitoring.metrics import record_trade_attempt, record_trade_success, record_trade_failure
from src.execution.execution_policy import (
    enforce_slippage_limit,
    gas_ceiling_for_chain,
    priority_fee_limit_for_chain,
    apply_gas_guardrails,
    should_simulate_transaction,
    simulate_evm_transaction,
)
from src.utils.idempotency import (
    build_trade_intent,
    register_trade_intent,
    mark_trade_intent_pending,
    mark_trade_intent_completed,
    mark_trade_intent_failed,
)

# Dynamic config loading
def get_base_config():
    """Get current configuration values dynamically"""
    slippage_raw = get_config_float("slippage", 0.02)
    slippage = enforce_slippage_limit(
        "base",
        slippage_raw,
        context={"config_key": "slippage"},
    )
    gas_ceiling_config = float(get_config("gas_ceiling_gwei"))
    gas_ceiling_policy = gas_ceiling_for_chain("base")
    priority_ceiling_config = float(get_config("gas_priority_max_gwei"))
    priority_ceiling_policy = priority_fee_limit_for_chain("base")
    gas_ceiling = min(gas_ceiling_config, gas_ceiling_policy)
    priority_ceiling = min(priority_ceiling_config, priority_ceiling_policy)

    return {
        'TEST_MODE': get_config_bool("test_mode", True),
        'SLIPPAGE': slippage,
        'USE_SUPPORTING_FEE': get_config_bool("use_supporting_fee_swap", True),
        'REQUOTE_DELAY_SECONDS': get_config_int("requote_delay_seconds", 10),
        'REQUOTE_SLIPPAGE_BUFFER': get_config_float("requote_slippage_buffer", 0.005),
        'GAS_CFG': {
            "gas_blocks": get_config("gas_blocks"),
            "gas_reward_percentile": get_config("gas_reward_percentile"),
            "gas_basefee_headroom": get_config("gas_basefee_headroom"),
            "gas_priority_min_gwei": get_config("gas_priority_min_gwei"),
            "gas_priority_max_gwei": priority_ceiling,
            "gas_ceiling_gwei": gas_ceiling,
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

# Logging helper
def _log(level: str, event: str, message: str, **context):
    log_event(event, level=level, message=message, **context)

# === Helpers ===
def _next_nonce():
    return w3.eth.get_transaction_count(WALLET)

def _apply_eip1559(tx_dict: dict) -> dict:
    """
    Attach EIP-1559 gas fields using on-chain fee history, with config guardrails.
    """
    config = get_base_config()
    max_fee, max_prio = suggest_fees(w3, config['GAS_CFG'])
    max_fee, max_prio = apply_gas_guardrails("base", w3, max_fee, max_prio)
    out = dict(tx_dict)
    out["maxFeePerGas"] = int(max_fee)
    out["maxPriorityFeePerGas"] = int(max_prio)
    out["type"] = 2
    return out

def _estimate_gas(tx: dict, fallback: int) -> dict:
    try:
        est = w3.eth.estimate_gas(tx)
        tx["gas"] = int(est * 1.2)
    except Exception:
        tx["gas"] = fallback
    return tx

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
        _log("WARNING", "base.quote.error", "Quote call failed", error=str(e))
        return 0

# === BUY with re-quote protection ===
def buy_token(token_address: str, usd_amount: float, symbol: str = "?") -> Tuple[Optional[str], bool]:
    """
    Buy `token_address` spending `usd_amount` USD worth of ETH on BASE.
    Returns (tx_hash_hex or None, success_bool).
    """
    trade_started = time.time()
    record_trade_attempt("base", "buy")
    config = get_base_config()

    # Build idempotent trade intent prior to heavy lifting
    intent = build_trade_intent(
        chain="base",
        side="buy",
        token_address=token_address,
        symbol=symbol,
        quantity=float(usd_amount),
        metadata={"slippage": config['SLIPPAGE']},
    )
    registered, _ = register_trade_intent(intent)
    if not registered:
        record_trade_failure("base", "buy", "duplicate_intent")
        return None, False
    mark_trade_intent_pending(intent.intent_id)

    def abort(reason: str, level: str, message: str, **ctx):
        mark_trade_intent_failed(intent.intent_id, reason)
        latency_ms = int((time.time() - trade_started) * 1000)
        record_trade_failure("base", "buy", reason, latency_ms=latency_ms)
        _log(level, f"base.buy.{reason}", message, symbol=symbol, **ctx)
        return None, False

    def succeed(tx_hash: Optional[str], slippage_used: float):
        mark_trade_intent_completed(intent.intent_id, tx_hash=tx_hash)
        latency_ms = int((time.time() - trade_started) * 1000)
        record_trade_success(
            "base",
            "buy",
            latency_ms=latency_ms,
            slippage_bps=slippage_used * 10000.0,
        )
        _log("INFO", "base.buy.completed", "Buy transaction completed", symbol=symbol, tx_hash=tx_hash)
        return tx_hash, True

    try:
        token_address_checksum = Web3.to_checksum_address(token_address)
    except Exception as exc:
        return abort("address_invalid", "ERROR", "Invalid token address", error=str(exc))

    deadline = int(time.time()) + 600

    eth_usd = get_eth_price_usd()
    if not eth_usd or eth_usd <= 0:
        return abort("eth_price_unavailable", "ERROR", "Could not fetch ETH/USD price for sizing.")

    eth_amount = float(usd_amount) / float(eth_usd)
    value_wei = w3.to_wei(eth_amount, "ether")

    base_tx = {
        "from": WALLET,
        "value": int(value_wei),
        "nonce": _next_nonce(),
        "chainId": BASE_CHAIN_ID,
    }

    quoted_out = 0
    fee_tiers = [3000, 500, 10000]  # 0.3%, 0.05%, 1%
    for fee in fee_tiers:
        try:
            quoted_out = _quote_v3_out(value_wei, token_address_checksum, fee)
            if quoted_out > 0:
                _log(
                    "INFO",
                    "base.buy.quote_success",
                    "Quote successful",
                    symbol=symbol,
                    fee_tier_bps=fee,
                    raw_out=int(quoted_out),
                )
                break
        except Exception as exc:
            _log(
                "WARNING",
                "base.buy.quote_retry",
                "Quote attempt failed; trying next fee tier",
                symbol=symbol,
                fee_tier_bps=fee,
                error=str(exc),
            )
            continue

    if quoted_out <= 0:
        return abort("quote_failed", "ERROR", "All quote attempts failed", fee_tiers=fee_tiers)

    amount_out_min = int(quoted_out * (1.0 - config['SLIPPAGE']))
    if amount_out_min <= 0:
        return abort("min_out_invalid", "ERROR", "Computed minimum output is non-positive", quoted_out=int(quoted_out))

    log_event(
        "base.buy.quote",
        symbol=symbol,
        eth_in=round(eth_amount, 6),
        out_raw=int(quoted_out),
        slippage=round(config['SLIPPAGE'], 6),
        min_out=int(amount_out_min),
    )

    params = {
        'tokenIn': BASE_WETH,
        'tokenOut': token_address_checksum,
        'fee': 3000,
        'recipient': WALLET,
        'deadline': deadline,
        'amountIn': value_wei,
        'amountOutMinimum': amount_out_min,
        'sqrtPriceLimitX96': 0
    }

    if config['USE_SUPPORTING_FEE']:
        fn = router.functions.exactInputSingleSupportingFeeOnTransferTokens(params)
    else:
        fn = router.functions.exactInputSingle(params)

    tx = fn.build_transaction(base_tx)
    tx = _apply_eip1559(tx)

    try:
        est = w3.eth.estimate_gas(tx)
        tx["gas"] = int(est * 1.2)
    except Exception as exc:
        _log("WARNING", "base.buy.gas_estimate_failed", "Gas estimate failed; using fallback", error=str(exc))
        tx["gas"] = 300000

    t_quote = time.time()

    def _maybe_requote_and_adjust(tx_dict):
        elapsed = time.time() - t_quote
        if elapsed <= config['REQUOTE_DELAY_SECONDS']:
            return tx_dict

        _log(
            "INFO",
            "base.buy.requote",
            "Quote stale; attempting re-quote before send",
            elapsed_seconds=round(elapsed, 2),
            symbol=symbol,
        )

        try:
            re_quoted_out = _quote_v3_out(value_wei, token_address_checksum, 3000)
        except Exception as exc:
            _log(
                "WARNING",
                "base.buy.requote_failed",
                "Re-quote failed; sending original transaction",
                error=str(exc),
                symbol=symbol,
            )
            return tx_dict

        if re_quoted_out <= 0:
            _log(
                "WARNING",
                "base.buy.requote_zero",
                "Re-quote returned zero; keeping original minOut",
                symbol=symbol,
            )
            return tx_dict

        total_slip = config['SLIPPAGE'] + max(0.0, config['REQUOTE_SLIPPAGE_BUFFER'])
        new_min_out = int(re_quoted_out * (1.0 - total_slip))
        if new_min_out <= 0:
            _log(
                "WARNING",
                "base.buy.requote_invalid_min_out",
                "Re-quote produced invalid minOut; keeping original",
                symbol=symbol,
                re_quoted_out=int(re_quoted_out),
            )
            return tx_dict

        _log(
            "INFO",
            "base.buy.requote_applied",
            "Re-quote applied with expanded slippage buffer",
            symbol=symbol,
            re_quoted_out=int(re_quoted_out),
            new_min_out=int(new_min_out),
            total_slippage_percent=total_slip * 100.0,
        )
        params['amountOutMinimum'] = new_min_out
        if config['USE_SUPPORTING_FEE']:
            fn2 = router.functions.exactInputSingleSupportingFeeOnTransferTokens(params)
        else:
            fn2 = router.functions.exactInputSingle(params)

        tx2 = fn2.build_transaction(base_tx)
        tx2 = _apply_eip1559(tx2)
        tx2["gas"] = tx_dict["gas"]
        return tx2

    final_tx = _maybe_requote_and_adjust(tx)

    if should_simulate_transaction("base"):
        simulated, sim_error = simulate_evm_transaction("base", w3, final_tx)
        if not simulated:
            return abort("simulation_failed", "ERROR", "Transaction simulation failed", error=sim_error)

    if config['TEST_MODE']:
        log_event("base.buy.test_mode", symbol=symbol, note="Transaction validated but not sent")
        return succeed(None, config['SLIPPAGE'])

    try:
        signed = w3.eth.account.sign_transaction(final_tx, PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
        return succeed(tx_hash.hex(), config['SLIPPAGE'])
    except Exception as exc:
        return abort("send_failed", "ERROR", "Buy transaction failed to send", error=str(exc))

# === SELL functionality ===
def sell_token(token_address: str, token_amount: Optional[float] = None, symbol: str = "?") -> Tuple[Optional[str], bool]:
    """
    Sell `token_amount` (in human units) of `token_address` for ETH on BASE.
    If token_amount is None, sell the entire balance.
    Returns: (tx_hash_hex or None, success_bool)
    """
    trade_started = time.time()
    record_trade_attempt("base", "sell")
    config = get_base_config()

    intent = build_trade_intent(
        chain="base",
        side="sell",
        token_address=token_address,
        symbol=symbol,
        quantity=float(token_amount) if token_amount is not None else -1.0,
        metadata={"slippage": config["SLIPPAGE"]},
    )
    registered, existing = register_trade_intent(intent)
    if not registered:
        record_trade_failure("base", "sell", "duplicate_intent")
        _log("INFO", "base.sell.duplicate", "Duplicate sell intent detected", token=token_address, existing_intent=existing)
        return None, False
    mark_trade_intent_pending(intent.intent_id)

    def abort(reason: str, level: str, message: str, **ctx):
        mark_trade_intent_failed(intent.intent_id, reason)
        latency_ms = int((time.time() - trade_started) * 1000)
        record_trade_failure("base", "sell", reason, latency_ms=latency_ms)
        _log(level, f"base.sell.{reason}", message, symbol=symbol, **ctx)
        return None, False

    def succeed(tx_hash: Optional[str]):
        mark_trade_intent_completed(intent.intent_id, tx_hash=tx_hash)
        latency_ms = int((time.time() - trade_started) * 1000)
        record_trade_success(
            "base",
            "sell",
            latency_ms=latency_ms,
            slippage_bps=config["SLIPPAGE"] * 10000.0,
        )
        _log("INFO", "base.sell.completed", "Sell transaction completed", symbol=symbol, tx_hash=tx_hash)
        return tx_hash, True

    try:
        token_address_checksum = Web3.to_checksum_address(token_address)
    except Exception as exc:
        return abort("address_invalid", "ERROR", "Invalid token address", error=str(exc))

    token_contract = _erc20(token_address_checksum)

    try:
        decimals = token_contract.functions.decimals().call()
    except Exception:
        decimals = 9

    try:
        balance_wei = token_contract.functions.balanceOf(WALLET).call()
    except Exception as exc:
        return abort("balance_fetch_failed", "ERROR", "Failed to fetch token balance", error=str(exc))

    if token_amount is None:
        token_amount_wei = int(balance_wei)
    else:
        token_amount_wei = int(float(token_amount) * (10 ** decimals))

    if token_amount_wei <= 0:
        return abort("amount_invalid", "ERROR", "Sell amount must be positive", token_amount=token_amount)

    if balance_wei < token_amount_wei:
        return abort(
            "insufficient_balance",
            "ERROR",
            "Insufficient token balance for sell",
            balance=int(balance_wei),
            required=int(token_amount_wei),
        )

    if not config['TEST_MODE']:
        try:
            allowance = token_contract.functions.allowance(WALLET, ROUTER_ADDR).call()
            if allowance < token_amount_wei:
                _log("INFO", "base.sell.approval", "Approving router spend", symbol=symbol)
                approve_tx = token_contract.functions.approve(ROUTER_ADDR, token_amount_wei).build_transaction({
                    "from": WALLET,
                    "nonce": _next_nonce(),
                    "chainId": BASE_CHAIN_ID,
                })
                approve_tx = _apply_eip1559(approve_tx)
                approve_tx = _estimate_gas(approve_tx, fallback=120000)
                signed = w3.eth.account.sign_transaction(approve_tx, PRIVATE_KEY)
                approve_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
                w3.eth.wait_for_transaction_receipt(approve_hash, timeout=60)
        except Exception as exc:
            return abort("approval_failed", "ERROR", "Approval transaction failed", error=str(exc))

    deadline = int(time.time()) + 600
    params = {
        'tokenIn': token_address_checksum,
        'tokenOut': BASE_WETH,
        'fee': 3000,
        'recipient': WALLET,
        'deadline': deadline,
        'amountIn': token_amount_wei,
        'amountOutMinimum': 0,
        'sqrtPriceLimitX96': 0
    }

    try:
        quoted_out = router.functions.exactInputSingle(params).call()
    except Exception as exc:
        return abort("quote_failed", "ERROR", "Sell quote failed", error=str(exc))

    amount_out_min = int(quoted_out * (1.0 - config['SLIPPAGE']))
    if amount_out_min <= 0:
        return abort("min_out_invalid", "ERROR", "Sell minimum output invalid", quoted_out=int(quoted_out))

    log_event(
        "base.sell.quote",
        symbol=symbol,
        amount_in=int(token_amount_wei),
        out_raw=int(quoted_out),
        slippage=round(config['SLIPPAGE'], 6),
        min_out=int(amount_out_min),
    )

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
    tx = _estimate_gas(tx, fallback=300000)

    if should_simulate_transaction("base"):
        simulated, sim_error = simulate_evm_transaction("base", w3, tx)
        if not simulated:
            return abort("simulation_failed", "ERROR", "Sell simulation failed", error=sim_error)

    if config['TEST_MODE']:
        log_event("base.sell.test_mode", symbol=symbol, note="Transaction validated but not sent")
        return succeed(None)

    try:
        signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
        return succeed(tx_hash.hex())
    except Exception as exc:
        return abort("send_failed", "ERROR", "Sell transaction failed to send", error=str(exc))

# === Utility functions ===
def get_base_balance() -> float:
    """Get ETH balance on BASE chain"""
    try:
        balance_wei = w3.eth.get_balance(WALLET)
        balance_eth = w3.from_wei(balance_wei, "ether")
        return float(balance_eth)
    except Exception as e:
        _log("ERROR", "base.balance.failed", "Failed to fetch BASE balance", error=str(e))
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
        _log("ERROR", "base.token_balance.failed", "Failed to fetch token balance", token_address=token_address, error=str(e))
        return 0.0
