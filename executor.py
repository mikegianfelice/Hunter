from web3 import Web3
import yaml
import json
import time
import sys
import subprocess
from pathlib import Path

from secrets import INFURA_URL, WALLET_ADDRESS, PRIVATE_KEY
from gas import suggest_fees
from utils import get_eth_price_usd  # robust ETH/USD (Graph -> on-chain V2)
from telegram_bot import send_telegram_message

# --- Config ---
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f) or {}

TEST_MODE = bool(CONFIG.get("test_mode", True))
SLIPPAGE = float(CONFIG.get("slippage", 0.02))                      # e.g., 0.02 = 2%
TRADE_AMOUNT_USD_DEFAULT = float(CONFIG.get("trade_amount_usd", 5))
USE_SUPPORTING_FEE = bool(CONFIG.get("use_supporting_fee_swap", True))
PRICE_IMPACT_MAX = float(CONFIG.get("price_impact_max_pct", 0.15))  # abort if quote >15% worse than EMA
PRICE_EMA_ALPHA = float(CONFIG.get("price_ema_alpha", 0.30))        # EMA smoothing factor
PUMP_GUARD_MAX = float(CONFIG.get("pump_guard_max_pct", 0.25))      # abort if quote >25% better than EMA

# Optional gas knobs
GAS_CFG = {
    "gas_blocks": CONFIG.get("gas_blocks"),
    "gas_reward_percentile": CONFIG.get("gas_reward_percentile"),
    "gas_basefee_headroom": CONFIG.get("gas_basefee_headroom"),
    "gas_priority_min_gwei": CONFIG.get("gas_priority_min_gwei"),
    "gas_priority_max_gwei": CONFIG.get("gas_priority_max_gwei"),
    "gas_ceiling_gwei": CONFIG.get("gas_ceiling_gwei"),
    "gas_multiplier": CONFIG.get("gas_multiplier"),
    "gas_extra_priority_gwei": CONFIG.get("gas_extra_priority_gwei"),
}

# --- Web3 / Router ---
if not INFURA_URL:
    raise RuntimeError("INFURA_URL missing in .env / secrets.py")
if not WALLET_ADDRESS or not PRIVATE_KEY:
    raise RuntimeError("WALLET_ADDRESS / PRIVATE_KEY missing in .env / secrets.py")

w3 = Web3(Web3.HTTPProvider(INFURA_URL))
if not w3.is_connected():
    raise RuntimeError("Web3 not connected — check RPC URL")

WALLET = Web3.to_checksum_address(WALLET_ADDRESS)

ROUTER_ADDR = Web3.to_checksum_address("0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D")  # Uniswap V2
ABI_PATH = "uniswap_router_abi.json"
with open(ABI_PATH, "r") as f:
    ROUTER_ABI = json.load(f)
router = w3.eth.contract(address=ROUTER_ADDR, abi=ROUTER_ABI)

WETH = Web3.to_checksum_address("0xC02aaA39b223FE8D0A0E5C4F27eAD9083C756Cc2")

# --- Files ---
POSITIONS_FILE = "open_positions.json"
MONITOR_SCRIPT = "monitor_position.py"
PRICE_MEMORY_FILE = "price_memory.json"   # stores tokens_per_wei + ema per token

# ===== Helpers =====
def _next_nonce():
    return w3.eth.get_transaction_count(WALLET)

def _apply_eip1559(tx: dict) -> dict:
    max_fee, max_prio = suggest_fees(w3, GAS_CFG)
    tx = dict(tx)
    tx["maxFeePerGas"] = int(max_fee)
    tx["maxPriorityFeePerGas"] = int(max_prio)
    tx["type"] = 2
    return tx

def _ensure_positions_file():
    p = Path(POSITIONS_FILE)
    if not p.exists():
        p.write_text("{}")

def _log_position(token):
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

def calculate_trade_size(usd_amount: float, eth_price_usd: float) -> float:
    """Return ETH amount to spend for a given USD size."""
    return float(usd_amount) / float(eth_price_usd)

def _quote_v2_out(amount_in_wei: int, path: list[int]) -> int:
    """Quote expected token out using Uniswap V2 router getAmountsOut."""
    amounts = router.functions.getAmountsOut(int(amount_in_wei), path).call()
    return int(amounts[-1])

# ---- Price memory (EMA) ----
def _load_price_memory():
    p = Path(PRICE_MEMORY_FILE)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text() or "{}")
    except Exception:
        return {}

def _save_price_memory(mem: dict):
    try:
        Path(PRICE_MEMORY_FILE).write_text(json.dumps(mem, indent=2))
    except Exception as e:
        print(f"⚠️ Failed to write {PRICE_MEMORY_FILE}: {e}")

def _update_price_memory(addr: str, curr_tpw: float):
    """
    Update both last tokens_per_wei and an EMA (exponential moving average).
    EMA := alpha * curr + (1 - alpha) * prev_ema
    """
    addr = addr.lower()
    mem = _load_price_memory()
    rec = mem.get(addr, {})
    prev_ema = rec.get("ema_tokens_per_wei")
    if prev_ema is None or prev_ema <= 0:
        new_ema = curr_tpw
    else:
        a = max(0.0, min(1.0, PRICE_EMA_ALPHA))
        new_ema = a * curr_tpw + (1 - a) * float(prev_ema)

    mem[addr] = {
        "tokens_per_wei": float(curr_tpw),
        "ema_tokens_per_wei": float(new_ema),
        "alpha": float(PRICE_EMA_ALPHA),
        "ts": int(time.time())
    }
    _save_price_memory(mem)
    return new_ema

def _get_prev_metrics(addr: str):
    mem = _load_price_memory()
    rec = mem.get(addr.lower())
    if not rec:
        return None, None
    return float(rec.get("tokens_per_wei") or 0) or None, float(rec.get("ema_tokens_per_wei") or 0) or None

# ===== Entry point =====
def execute_trade(token: dict, trade_amount_usd: float = None):
    """
    BUY token with:
      1) USD sizing -> ETH
      2) V2 quote (getAmountsOut)
      3) amountOutMin = quotedOut * (1 - SLIPPAGE)
      4) Guard vs EMA:
         - Abort if current quote is > PRICE_IMPACT_MAX worse than EMA (adverse impact)
         - Abort if current quote is > PUMP_GUARD_MAX better than EMA (too-good pump)
      5) Submit swap (EIP-1559)
    Returns: (tx_hash_hex, success)
    """
    symbol = token.get("symbol", "?")
    token_address = Web3.to_checksum_address(token["address"])
    amount_usd = float(trade_amount_usd or TRADE_AMOUNT_USD_DEFAULT)

    # 1) Robust ETH/USD sizing
    eth_usd = get_eth_price_usd()
    if not eth_usd or eth_usd <= 0:
        print("❌ Failed to compute ETH amount: Could not fetch ETH/USD price for sizing.")
        return None, False

    eth_amount = calculate_trade_size(amount_usd, eth_usd)  # in ETH
    value_wei = w3.to_wei(eth_amount, "ether")

    path = [WETH, token_address]
    deadline = int(time.time()) + 600

    # 2) Quote exact output on V2
    try:
        quoted_out = _quote_v2_out(value_wei, path)
    except Exception as e:
        print(f"❌ Quote failed for {symbol}: {e}")
        return None, False

    if quoted_out <= 0:
        print(f"❌ Bad quote (0 out) for {symbol}; aborting.")
        return None, False

    # tokens per wei
    curr_tpw = quoted_out / float(value_wei)

    # ---- 4) Guards vs EMA ----
    last_tpw, prev_ema = _get_prev_metrics(token_address)
    if prev_ema and prev_ema > 0:
        # Adverse move: fewer tokens per wei (price got worse)
        adverse = (prev_ema - curr_tpw) / prev_ema
        # Pump move: many more tokens per wei (suspiciously better / bait)
        favorable = (curr_tpw - prev_ema) / prev_ema

        if adverse > PRICE_IMPACT_MAX:
            pct = adverse * 100.0
            print(f"🛑 Price impact guard (EMA): ema={prev_ema:.12f}, now={curr_tpw:.12f}, "
                  f"worse by {pct:.2f}% > {PRICE_IMPACT_MAX*100:.2f}% — aborting buy.")
            try:
                send_telegram_message(
                    f"🛑 *Buy Blocked: Price Impact (EMA)*\n"
                    f"Token: `{symbol}` `{token_address}`\n"
                    f"Worse by: *{pct:.2f}%* (limit {PRICE_IMPACT_MAX*100:.2f}%)\n"
                    f"EMA tpw: `{prev_ema:.12f}`\n"
                    f"Curr tpw: `{curr_tpw:.12f}`\n"
                    f"α: {PRICE_EMA_ALPHA:.2f}"
                )
            except Exception as e:
                print(f"⚠️ Telegram alert failed: {e}")
            return None, False

        if favorable > PUMP_GUARD_MAX:
            pct = favorable * 100.0
            print(f"🛑 Pump guard: ema={prev_ema:.12f}, now={curr_tpw:.12f}, "
                  f"better by {pct:.2f}% > {PUMP_GUARD_MAX*100:.2f}% — aborting buy.")
            try:
                send_telegram_message(
                    f"🛑 *Buy Blocked: Pump Guard*\n"
                    f"Token: `{symbol}` `{token_address}`\n"
                    f"Better by: *{pct:.2f}%* (limit {PUMP_GUARD_MAX*100:.2f}%)\n"
                    f"EMA tpw: `{prev_ema:.12f}`\n"
                    f"Curr tpw: `{curr_tpw:.12f}`\n"
                    f"α: {PRICE_EMA_ALPHA:.2f}"
                )
            except Exception as e:
                print(f"⚠️ Telegram alert failed: {e}")
            return None, False

        print(f"🛡️ EMA guards OK: adverse {(adverse*100):.2f}% | favorable {(favorable*100):.2f}%")

    else:
        print("ℹ️ No EMA history; skipping guards for first observation.")

    # Update EMA memory (after checks, before send so next loop has it)
    new_ema = _update_price_memory(token_address, curr_tpw)
    print(f"📈 Updated EMA tpw: {new_ema:.12f} (α={PRICE_EMA_ALPHA:.2f})")

    # 3) Compute amountOutMin by slippage
    amount_out_min = int(quoted_out * (1.0 - SLIPPAGE))
    if amount_out_min <= 0:
        print(f"❌ Computed minOut <= 0 for {symbol}; aborting.")
        return None, False

    print(f"🧮 Quote for {symbol}: in {eth_amount:.6f} ETH → out {quoted_out} (raw units)")
    print(f"🎯 Slippage {SLIPPAGE*100:.2f}% → minOut {amount_out_min} (raw units)")

    base_tx = {
        "from": WALLET,
        "value": int(value_wei),
        "nonce": _next_nonce(),
        "chainId": 1,
    }

    try:
        # Build the swap with enforced minOut
        if USE_SUPPORTING_FEE:
            fn = router.functions.swapExactETHForTokensSupportingFeeOnTransferTokens(
                amount_out_min, path, WALLET, deadline
            )
        else:
            fn = router.functions.swapExactETHForTokens(
                amount_out_min, path, WALLET, deadline
            )

        tx = fn.build_transaction(base_tx)
        tx = _apply_eip1559(tx)

        # Estimate gas and add buffer
        try:
            est = w3.eth.estimate_gas(tx)
            tx["gas"] = int(est * 1.2)
        except Exception:
            tx["gas"] = 300000

        if TEST_MODE:
            print(f"🚫 [TEST MODE] Simulated buy of {symbol} for {eth_amount:.6f} ETH")
            _log_position(token)
            _launch_monitor_detached()
            return "0xSIMULATED_TX", True

        # LIVE: sign + send
        signed = w3.eth.account.sign_transaction(tx, private_key=PRIVATE_KEY)
        sent = w3.eth.send_raw_transaction(signed.rawTransaction)
        tx_hash_hex = sent.hex()
        print(f"📥 Buy sent: {tx_hash_hex}")

        # Optional: w3.eth.wait_for_transaction_receipt(sent)

        _log_position(token)
        _launch_monitor_detached()

        return tx_hash_hex, True

    except Exception as e:
        print(f"❌ Buy failed for {symbol}: {e}")
        return None, False