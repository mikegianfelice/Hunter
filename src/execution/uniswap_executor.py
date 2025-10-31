# uniswap_executor.py
from web3 import Web3
import json
import yaml
import time
import os
import sys
from pathlib import Path

from src.config.secrets import INFURA_URL, WALLET_ADDRESS, PRIVATE_KEY
from src.utils.gas import suggest_fees
from src.utils.utils import get_eth_price_usd  # robust ETH/USD (Graph -> on-chain V2)
from src.config.config_loader import get_config, get_config_bool, get_config_float, get_config_int
from src.monitoring.logger import log_event

# Dynamic config loading
def get_uniswap_config():
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
if not INFURA_URL:
    raise RuntimeError("INFURA_URL missing. Put it in your .env and secrets.py.")
if not WALLET_ADDRESS or not PRIVATE_KEY:
    raise RuntimeError("WALLET_ADDRESS / PRIVATE_KEY missing from .env.")

w3 = Web3(Web3.HTTPProvider(INFURA_URL))
if not w3.is_connected():
    raise RuntimeError("Web3 is not connected. Check your RPC URL / network.")

WALLET = Web3.to_checksum_address(WALLET_ADDRESS)

# Uniswap V2 router (0x7a25...)
ROUTER_ADDR = Web3.to_checksum_address("0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D")
ABI_PATH = "uniswap_router_abi.json"
if not os.path.exists(ABI_PATH):
    raise FileNotFoundError("uniswap_router_abi.json not found. Add the V2 Router ABI JSON file.")
with open(ABI_PATH, "r") as f:
    ROUTER_ABI = json.load(f)
router = w3.eth.contract(address=ROUTER_ADDR, abi=ROUTER_ABI)

WETH = Web3.to_checksum_address("0xC02aaA39b223FE8D0A0E5C4F27eAD9083C756Cc2")

# Minimal ERC20 ABI for sell path
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
    config = get_uniswap_config()
    max_fee, max_prio = suggest_fees(w3, config['GAS_CFG'])
    out = dict(tx_dict)
    out["maxFeePerGas"] = int(max_fee)
    out["maxPriorityFeePerGas"] = int(max_prio)
    out["type"] = 2
    return out

def _erc20(token_addr):
    return w3.eth.contract(address=Web3.to_checksum_address(token_addr), abi=ERC20_MIN_ABI)

def _quote_v2_out(amount_in_wei: int, path: list) -> int:
    """
    Quote expected token out using Uniswap V2 router getAmountsOut.
    Returns raw token amount in smallest units.
    """
    amounts = router.functions.getAmountsOut(int(amount_in_wei), path).call()
    return int(amounts[-1])

# === BUY with re-quote protection ===
def buy_token(token_address: str, usd_amount: float, symbol: str = "?") -> tuple[str, bool]:
    """
    Buy `token_address` spending `usd_amount` USD worth of ETH.
    - Uses V2 quote to compute amountOutMin with slippage
    - If delay between quote and send > REQUOTE_DELAY_SECONDS, re-quotes and
      recomputes minOut with an extra REQUOTE_SLIPPAGE_BUFFER
    - Uses EIP-1559 fees
    Returns: (tx_hash_hex or None, success_bool)
    """
    config = get_uniswap_config()
    token_address = Web3.to_checksum_address(token_address)
    path = [WETH, token_address]
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
        "chainId": 1,
    }

    # Initial quote
    try:
        quoted_out = _quote_v2_out(value_wei, path)
    except Exception as e:
        print(f"❌ Quote failed for {symbol}: {e}")
        return None, False

    if quoted_out <= 0:
        print(f"❌ Bad quote (0 out) for {symbol}; aborting.")
        return None, False

    amount_out_min = int(quoted_out * (1.0 - config['SLIPPAGE']))
    if amount_out_min <= 0:
        print(f"❌ Computed minOut <= 0 for {symbol}; aborting.")
        return None, False

    log_event("eth.buy.quote", symbol=symbol, eth_in=round(eth_amount, 6), out_raw=int(quoted_out), slippage=round(config['SLIPPAGE'], 6), min_out=int(amount_out_min))

    # Build with initial minOut
    if config['USE_SUPPORTING_FEE']:
        fn = router.functions.swapExactETHForTokensSupportingFeeOnTransferTokens(
            amount_out_min, path, WALLET, deadline
        )
    else:
        fn = router.functions.swapExactETHForTokens(
            amount_out_min, path, WALLET, deadline
        )

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
            re_quoted_out = _quote_v2_out(value_wei, path)
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
        if config['USE_SUPPORTING_FEE']:
            fn2 = router.functions.swapExactETHForTokensSupportingFeeOnTransferTokens(
                new_min_out, path, WALLET, deadline
            )
        else:
            fn2 = router.functions.swapExactETHForTokens(
                new_min_out, path, WALLET, deadline
            )

        new_tx = fn2.build_transaction({
            "from": WALLET,
            "value": int(value_wei),
            "nonce": tx_dict["nonce"],   # keep same nonce
            "chainId": 1,
        })
        new_tx = _apply_eip1559(new_tx)
        # reuse gas (or re-estimate if you want)
        new_tx["gas"] = tx_dict.get("gas", 300000)
        return new_tx

    # LIVE send (with possible re-quote)
    tx = _maybe_requote_and_adjust(tx)
    try:
        if config['TEST_MODE']:
            # In test mode, still build and validate transaction but don't send
            # This allows testing with real market data and quotes
            log_event("eth.buy.test_mode", symbol=symbol, eth_in=round(eth_amount, 6), note="Transaction validated but not sent")
            # Return None to indicate no actual transaction was sent
            return None, True
        
        signed = w3.eth.account.sign_transaction(tx, private_key=PRIVATE_KEY)
        sent = w3.eth.send_raw_transaction(signed.rawTransaction)
        tx_hash_hex = sent.hex()
        log_event("eth.buy.sent", symbol=symbol, tx_hash=tx_hash_hex)
        # You can wait for receipt if desired:
        # w3.eth.wait_for_transaction_receipt(sent)
        return tx_hash_hex, True
    except Exception as e:
        log_event("eth.buy.error", level="ERROR", symbol=symbol, error=str(e))
        return None, False

# === SELL path (unchanged semantics) ===
def sell_token(token_address: str):
    """
    Sell *all* of a given ERC-20 token for ETH using Uniswap V2:
      - Approves router if needed
      - Uses swapExactTokensForETHSupportingFeeOnTransferTokens to handle taxed tokens
    Respects TEST_MODE. Returns tx hash (hex str) or None.
    """
    config = get_uniswap_config()
    token = _erc20(token_address)
    token_addr = Web3.to_checksum_address(token_address)

    try:
        token.functions.decimals().call()
    except Exception:
        pass  # not strictly needed below

    try:
        balance = token.functions.balanceOf(WALLET).call()
    except Exception as e:
        print(f"❌ Failed to read token balance: {e}")
        return None

    if balance == 0:
        print("ℹ️ No token balance to sell.")
        return None

    # Check allowance; approve if needed
    try:
        allowance = token.functions.allowance(WALLET, ROUTER_ADDR).call()
    except Exception as e:
        print(f"❌ Failed to read allowance: {e}")
        return None

    nonce = _next_nonce()

    if allowance < balance:
        print("🔐 Approving router to spend tokens...")
        try:
            base_tx = {"from": WALLET, "nonce": nonce, "chainId": 1}
            tx = token.functions.approve(ROUTER_ADDR, balance).build_transaction(base_tx)
            tx = _apply_eip1559(tx)

            # estimate gas if possible
            try:
                gas_est = w3.eth.estimate_gas(tx)
                tx["gas"] = int(gas_est * 1.2)
            except Exception:
                tx["gas"] = 120000

            if config['TEST_MODE']:
                print("🚫 [TEST MODE] Approval built, not sent.")
            else:
                signed = w3.eth.account.sign_transaction(tx, private_key=PRIVATE_KEY)
                tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
                print(f"✅ Approval sent: {tx_hash.hex()}")
                w3.eth.wait_for_transaction_receipt(tx_hash)
            nonce += 1
        except Exception as e:
            print(f"❌ Approve failed: {e}")
            return None

    # Now swap all tokens for ETH (supporting fee-on-transfer tokens)
    deadline = int(time.time()) + 600
    path = [token_addr, WETH]

    try:
        base_tx = {"from": WALLET, "nonce": nonce, "chainId": 1}
        tx = router.functions.swapExactTokensForETHSupportingFeeOnTransferTokens(
            balance,
            0,         # minOut can be enhanced with quote if desired
            path,
            WALLET,
            deadline
        ).build_transaction(base_tx)

        tx = _apply_eip1559(tx)

        try:
            gas_est = w3.eth.estimate_gas(tx)
            tx["gas"] = int(gas_est * 1.2)
        except Exception:
            tx["gas"] = 300000

        if config['TEST_MODE']:
            # In test mode, still build and validate transaction but don't send
            # This allows testing with real market data and quotes
            log_event("eth.sell.test_mode", symbol=symbol, note="Transaction validated but not sent")
            # Return None to indicate no actual transaction was sent
            return None

        signed = w3.eth.account.sign_transaction(tx, private_key=PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
        log_event("eth.sell.sent", symbol=symbol, tx_hash=tx_hash.hex())
        w3.eth.wait_for_transaction_receipt(tx_hash)
        log_event("eth.sell.confirmed", symbol=symbol, tx_hash=tx_hash.hex())
        return tx_hash.hex()

    except Exception as e:
        log_event("eth.sell.error", level="ERROR", symbol=symbol, error=str(e))
        return None