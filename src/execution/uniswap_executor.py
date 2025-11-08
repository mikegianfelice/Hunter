from __future__ import annotations

import json
import os
import time
from typing import Optional, Tuple

from web3 import Web3

from src.config.config_loader import get_config, get_config_bool, get_config_float, get_config_int
from src.config.secrets import INFURA_URL, WALLET_ADDRESS, PRIVATE_KEY
from src.execution.execution_policy import (
    enforce_slippage_limit,
    gas_ceiling_for_chain,
    priority_fee_limit_for_chain,
    apply_gas_guardrails,
    should_simulate_transaction,
    simulate_evm_transaction,
)
from src.monitoring.logger import log_event
from src.monitoring.metrics import (
    record_trade_attempt,
    record_trade_failure,
    record_trade_success,
)
from src.utils.gas import suggest_fees
from src.utils.idempotency import (
    build_trade_intent,
    mark_trade_intent_completed,
    mark_trade_intent_failed,
    mark_trade_intent_pending,
    register_trade_intent,
)
from src.utils.utils import get_eth_price_usd

CHAIN_NAME = "ethereum"
CHAIN_ID = 1


def _log(level: str, event: str, message: str, **context) -> None:
    log_event(event, level=level, message=message, **context)


def get_uniswap_config() -> dict:
    """Load configuration and apply execution guardrails."""
    raw_slippage = get_config_float("slippage", 0.02)
    slippage = enforce_slippage_limit(
        CHAIN_NAME,
        raw_slippage,
        context={"config_key": "slippage"},
    )
    gas_ceiling_cfg = float(get_config("gas_ceiling_gwei"))
    gas_ceiling_policy = gas_ceiling_for_chain(CHAIN_NAME)
    priority_ceiling_cfg = float(get_config("gas_priority_max_gwei"))
    priority_ceiling_policy = priority_fee_limit_for_chain(CHAIN_NAME)

    return {
        "TEST_MODE": get_config_bool("test_mode", True),
        "SLIPPAGE": slippage,
        "USE_SUPPORTING_FEE": get_config_bool("use_supporting_fee_swap", True),
        "REQUOTE_DELAY_SECONDS": get_config_int("requote_delay_seconds", 10),
        "REQUOTE_SLIPPAGE_BUFFER": get_config_float("requote_slippage_buffer", 0.005),
        "GAS_CFG": {
            "gas_blocks": get_config("gas_blocks"),
            "gas_reward_percentile": get_config("gas_reward_percentile"),
            "gas_basefee_headroom": get_config("gas_basefee_headroom"),
            "gas_priority_min_gwei": get_config("gas_priority_min_gwei"),
            "gas_priority_max_gwei": min(priority_ceiling_cfg, priority_ceiling_policy),
            "gas_ceiling_gwei": min(gas_ceiling_cfg, gas_ceiling_policy),
            "gas_multiplier": get_config("gas_multiplier"),
            "gas_extra_priority_gwei": get_config("gas_extra_priority_gwei"),
        },
    }


if not INFURA_URL:
    raise RuntimeError("INFURA_URL missing. Add it to your environment and secrets.")
if not WALLET_ADDRESS or not PRIVATE_KEY:
    raise RuntimeError("WALLET_ADDRESS or PRIVATE_KEY missing from configuration.")

w3 = Web3(Web3.HTTPProvider(INFURA_URL))
if not w3.is_connected():
    raise RuntimeError("Web3 cannot connect to the configured RPC endpoint.")

WALLET = Web3.to_checksum_address(WALLET_ADDRESS)
ROUTER_ADDR = Web3.to_checksum_address("0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D")
ABI_PATHS = ["uniswap_router_abi.json", os.path.join("data", "uniswap_router_abi.json")]
ABI_PATH = next((path for path in ABI_PATHS if os.path.exists(path)), None)
if not ABI_PATH:
    raise FileNotFoundError("uniswap_router_abi.json not found in project root or data/ directory.")
with open(ABI_PATH, "r", encoding="utf-8") as fh:
    ROUTER_ABI = json.load(fh)
router = w3.eth.contract(address=ROUTER_ADDR, abi=ROUTER_ABI)

WETH = Web3.to_checksum_address("0xC02aaA39b223FE8D0A0E5C4F27eAD9083C756Cc2")
ERC20_MIN_ABI = json.loads(
    """
    [
      {"constant":true,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"type":"function"},
      {"constant":true,"inputs":[{"name":"owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"type":"function"},
      {"constant":true,"inputs":[{"name":"owner","type":"address"},{"name":"spender","type":"address"}],"name":"allowance","outputs":[{"name":"","type":"uint256"}],"type":"function"},
      {"constant":false,"inputs":[{"name":"spender","type":"address"},{"name":"amount","type":"uint256"}],"name":"approve","outputs":[{"name":"","type":"bool"}],"type":"function"}
    ]
    """
)


def _next_nonce() -> int:
    return w3.eth.get_transaction_count(WALLET)


def _apply_eip1559(tx_dict: dict) -> dict:
    config = get_uniswap_config()
    max_fee, max_prio = suggest_fees(w3, config["GAS_CFG"])
    adjusted_max_fee, adjusted_priority = apply_gas_guardrails(
        CHAIN_NAME,
        w3,
        max_fee,
        max_prio,
    )
    tx = dict(tx_dict)
    tx["maxFeePerGas"] = int(adjusted_max_fee)
    tx["maxPriorityFeePerGas"] = int(adjusted_priority)
    tx["type"] = 2
    return tx


def _erc20(token_addr: str):
    return w3.eth.contract(address=Web3.to_checksum_address(token_addr), abi=ERC20_MIN_ABI)


def _quote_v2_out(amount_in_wei: int, path: list[str]) -> int:
    amounts = router.functions.getAmountsOut(int(amount_in_wei), path).call()
    return int(amounts[-1])


def _estimate_gas(tx: dict, fallback: int) -> dict:
    try:
        est = w3.eth.estimate_gas(tx)
        tx["gas"] = int(est * 1.2)
    except Exception:
        tx["gas"] = fallback
    return tx


def buy_token(token_address: str, usd_amount: float, symbol: str = "?") -> Tuple[Optional[str], bool]:
    trade_started = time.time()
    record_trade_attempt(CHAIN_NAME, "buy")
    config = get_uniswap_config()

    intent = build_trade_intent(
        chain=CHAIN_NAME,
        side="buy",
        token_address=token_address,
        symbol=symbol,
        quantity=float(usd_amount),
        metadata={"slippage": config["SLIPPAGE"]},
    )
    registered, _ = register_trade_intent(intent)
    if not registered:
        record_trade_failure(CHAIN_NAME, "buy", "duplicate_intent")
        return None, False
    mark_trade_intent_pending(intent.intent_id)

    def abort(reason: str, level: str, message: str, **ctx):
        mark_trade_intent_failed(intent.intent_id, reason)
        latency_ms = int((time.time() - trade_started) * 1000)
        record_trade_failure(CHAIN_NAME, "buy", reason, latency_ms=latency_ms)
        _log(level, f"eth.buy.{reason}", message, symbol=symbol, **ctx)
        return None, False

    def succeed(tx_hash: Optional[str]):
        mark_trade_intent_completed(intent.intent_id, tx_hash=tx_hash)
        latency_ms = int((time.time() - trade_started) * 1000)
        record_trade_success(
            CHAIN_NAME,
            "buy",
            latency_ms=latency_ms,
            slippage_bps=config["SLIPPAGE"] * 10000.0,
        )
        _log("INFO", "eth.buy.completed", "Buy transaction completed", symbol=symbol, tx_hash=tx_hash)
        return tx_hash, True

    try:
        token_checksum = Web3.to_checksum_address(token_address)
    except Exception as exc:
        return abort("address_invalid", "ERROR", "Invalid token address", error=str(exc))

    path = [WETH, token_checksum]
    deadline = int(time.time()) + 600

    eth_usd = get_eth_price_usd()
    if not eth_usd or eth_usd <= 0:
        return abort("eth_price_unavailable", "ERROR", "Could not fetch ETH/USD price.")

    eth_amount = float(usd_amount) / float(eth_usd)
    value_wei = w3.to_wei(eth_amount, "ether")

    base_tx = {
        "from": WALLET,
        "value": int(value_wei),
        "nonce": _next_nonce(),
        "chainId": CHAIN_ID,
    }

    try:
        quoted_out = _quote_v2_out(value_wei, path)
    except Exception as exc:
        return abort("quote_failed", "ERROR", "Quote failed", error=str(exc))

    if quoted_out <= 0:
        return abort("quote_zero", "ERROR", "Quote returned zero output", raw_out=int(quoted_out))

    amount_out_min = int(quoted_out * (1.0 - config["SLIPPAGE"]))
    if amount_out_min <= 0:
        return abort("min_out_invalid", "ERROR", "Computed minimum output is non-positive", quoted_out=int(quoted_out))

    log_event(
        "eth.buy.quote",
        symbol=symbol,
        eth_in=round(eth_amount, 6),
        out_raw=int(quoted_out),
        slippage=round(config["SLIPPAGE"], 6),
        min_out=int(amount_out_min)
    )

    if config["USE_SUPPORTING_FEE"]:
        fn = router.functions.swapExactETHForTokensSupportingFeeOnTransferTokens(
            amount_out_min, path, WALLET, deadline
        )
    else:
        fn = router.functions.swapExactETHForTokens(
            amount_out_min, path, WALLET, deadline
        )

    tx = fn.build_transaction(base_tx)
    tx = _apply_eip1559(tx)
    tx = _estimate_gas(tx, fallback=300000)

    t_quote = time.time()

    def _maybe_requote(tx_dict: dict) -> dict:
        elapsed = time.time() - t_quote
        if elapsed <= config["REQUOTE_DELAY_SECONDS"]:
            return tx_dict

        _log(
            "INFO",
            "eth.buy.requote",
            "Quote stale; attempting re-quote before send",
            symbol=symbol,
            elapsed_seconds=round(elapsed, 2),
        )

        try:
            re_quoted_out = _quote_v2_out(value_wei, path)
        except Exception as exc_inner:
            _log(
                "WARNING",
                "eth.buy.requote_failed",
                "Re-quote failed; using original transaction",
                symbol=symbol,
                error=str(exc_inner),
            )
            return tx_dict

        if re_quoted_out <= 0:
            _log("WARNING", "eth.buy.requote_zero", "Re-quote returned zero; keeping original", symbol=symbol)
            return tx_dict

        total_slip = config["SLIPPAGE"] + max(0.0, config["REQUOTE_SLIPPAGE_BUFFER"])
        new_min_out = int(re_quoted_out * (1.0 - total_slip))
        if new_min_out <= 0:
            _log(
                "WARNING",
                "eth.buy.requote_invalid_min_out",
                "Re-quote produced invalid minOut; keeping original",
                symbol=symbol,
            )
            return tx_dict

        _log(
            "INFO",
            "eth.buy.requote_applied",
            "Re-quote applied with expanded slippage buffer",
            symbol=symbol,
            re_quoted_out=int(re_quoted_out),
            new_min_out=int(new_min_out),
            total_slippage_percent=total_slip * 100.0,
        )

        if config["USE_SUPPORTING_FEE"]:
            fn_new = router.functions.swapExactETHForTokensSupportingFeeOnTransferTokens(
                new_min_out, path, WALLET, deadline
            )
        else:
            fn_new = router.functions.swapExactETHForTokens(
                new_min_out, path, WALLET, deadline
            )

        rebuilt = fn_new.build_transaction(base_tx)
        rebuilt = _apply_eip1559(rebuilt)
        rebuilt["gas"] = tx_dict.get("gas", 300000)
        return rebuilt

    final_tx = _maybe_requote(tx)

    if should_simulate_transaction(CHAIN_NAME):
        simulated, sim_error = simulate_evm_transaction(CHAIN_NAME, w3, final_tx)
        if not simulated:
            return abort("simulation_failed", "ERROR", "Transaction simulation failed", error=sim_error)

    if config["TEST_MODE"]:
        log_event("eth.buy.test_mode", symbol=symbol, note="Transaction validated but not sent")
        return succeed(None)

    try:
        signed = w3.eth.account.sign_transaction(final_tx, PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
        return succeed(tx_hash.hex())
    except Exception as exc:
        return abort("send_failed", "ERROR", "Buy transaction failed to send", error=str(exc))


def sell_token(token_address: str, token_amount: Optional[float] = None, symbol: str = "?") -> Tuple[Optional[str], bool]:
    """
    Sell `token_amount` (in human units) of `token_address` for ETH.
    If token_amount is None, sell the full balance.
    Returns: (tx_hash_hex or None, success_bool)
    """
    trade_started = time.time()
    record_trade_attempt(CHAIN_NAME, "sell")
    config = get_uniswap_config()

    intent = build_trade_intent(
        chain=CHAIN_NAME,
        side="sell",
        token_address=token_address,
        symbol=symbol,
        quantity=float(token_amount) if token_amount is not None else -1.0,
        metadata={"slippage": config["SLIPPAGE"]},
    )
    registered, existing = register_trade_intent(intent)
    if not registered:
        record_trade_failure(CHAIN_NAME, "sell", "duplicate_intent")
        _log("INFO", "eth.sell.duplicate", "Duplicate sell intent detected", token=token_address, existing_intent=existing)
        return None, False
    mark_trade_intent_pending(intent.intent_id)

    def abort(reason: str, level: str, message: str, **ctx):
        mark_trade_intent_failed(intent.intent_id, reason)
        latency_ms = int((time.time() - trade_started) * 1000)
        record_trade_failure(CHAIN_NAME, "sell", reason, latency_ms=latency_ms)
        _log(level, f"eth.sell.{reason}", message, symbol=symbol, **ctx)
        return None, False

    def succeed(tx_hash: Optional[str]):
        mark_trade_intent_completed(intent.intent_id, tx_hash=tx_hash)
        latency_ms = int((time.time() - trade_started) * 1000)
        record_trade_success(
            CHAIN_NAME,
            "sell",
            latency_ms=latency_ms,
            slippage_bps=config["SLIPPAGE"] * 10000.0,
        )
        _log("INFO", "eth.sell.completed", "Sell transaction completed", symbol=symbol, tx_hash=tx_hash)
        return tx_hash, True

    try:
        token_checksum = Web3.to_checksum_address(token_address)
    except Exception as exc:
        return abort("address_invalid", "ERROR", "Invalid token address", error=str(exc))

    token_contract = _erc20(token_checksum)

    try:
        decimals = token_contract.functions.decimals().call()
    except Exception:
        decimals = 18

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
            "Insufficient token balance",
            balance=int(balance_wei),
            required=int(token_amount_wei),
        )

    if not config["TEST_MODE"]:
        try:
            allowance = token_contract.functions.allowance(WALLET, ROUTER_ADDR).call()
            if allowance < token_amount_wei:
                _log("INFO", "eth.sell.approval", "Approving router spend", symbol=symbol)
                approve_tx = token_contract.functions.approve(ROUTER_ADDR, token_amount_wei).build_transaction({
                    "from": WALLET,
                    "nonce": _next_nonce(),
                    "chainId": CHAIN_ID,
                })
                approve_tx = _apply_eip1559(approve_tx)
                approve_tx = _estimate_gas(approve_tx, fallback=120000)
                signed = w3.eth.account.sign_transaction(approve_tx, PRIVATE_KEY)
                approve_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
                w3.eth.wait_for_transaction_receipt(approve_hash)
        except Exception as exc:
            return abort("approval_failed", "ERROR", "Approval transaction failed", error=str(exc))

    path = [token_checksum, WETH]
    deadline = int(time.time()) + 600

    try:
        quoted_out = _quote_v2_out(token_amount_wei, path)
    except Exception as exc:
        return abort("quote_failed", "ERROR", "Sell quote failed", error=str(exc))

    if quoted_out <= 0:
        return abort("quote_zero", "ERROR", "Sell quote returned zero output", raw_out=int(quoted_out))

    amount_out_min = int(quoted_out * (1.0 - config["SLIPPAGE"]))
    if amount_out_min <= 0:
        return abort("min_out_invalid", "ERROR", "Sell minimum output invalid", quoted_out=int(quoted_out))

    log_event(
        "eth.sell.quote",
        symbol=symbol,
        amount_in=int(token_amount_wei),
        out_raw=int(quoted_out),
        slippage=round(config["SLIPPAGE"], 6),
        min_out=int(amount_out_min),
    )

    swap_fn = router.functions.swapExactTokensForETHSupportingFeeOnTransferTokens(
        token_amount_wei,
        amount_out_min,
        path,
        WALLET,
        deadline,
    )
    tx = swap_fn.build_transaction({"from": WALLET, "nonce": _next_nonce(), "chainId": CHAIN_ID})
    tx = _apply_eip1559(tx)
    tx = _estimate_gas(tx, fallback=300000)

    if should_simulate_transaction(CHAIN_NAME):
        simulated, sim_error = simulate_evm_transaction(CHAIN_NAME, w3, tx)
        if not simulated:
            return abort("simulation_failed", "ERROR", "Sell simulation failed", error=sim_error)

    if config["TEST_MODE"]:
        log_event("eth.sell.test_mode", symbol=symbol, note="Transaction validated but not sent")
        return succeed(None)

    try:
        signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
        return succeed(tx_hash.hex())
    except Exception as exc:
        return abort("send_failed", "ERROR", "Sell transaction failed to send", error=str(exc))


def get_eth_balance() -> float:
    try:
        balance_wei = w3.eth.get_balance(WALLET)
        return float(w3.from_wei(balance_wei, "ether"))
    except Exception as exc:
        _log("WARNING", "eth.balance.failed", "Failed to fetch ETH balance", error=str(exc))
        return 0.0


def get_token_balance(token_address: str) -> float:
    try:
        token_contract = _erc20(token_address)
        decimals = token_contract.functions.decimals().call()
        balance = token_contract.functions.balanceOf(WALLET).call()
        return balance / (10 ** int(decimals))
    except Exception as exc:
        _log(
            "WARNING",
            "eth.token_balance.failed",
            "Failed to fetch token balance",
            token_address=token_address,
            error=str(exc),
        )
        return 0.0