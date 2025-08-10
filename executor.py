from web3 import Web3
import yaml
import json
import time
import sys
import subprocess
from pathlib import Path
from typing import Optional

from secrets import INFURA_URL, WALLET_ADDRESS  # PRIVATE_KEY not needed here
from utils import get_eth_price_usd            # robust ETH/USD (Graph -> on-chain V2)
from telegram_bot import send_telegram_message
from uniswap_executor import buy_token         # executes swap w/ re-quote protection

# --- Config ---
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f) or {}

TEST_MODE = bool(CONFIG.get("test_mode", True))
TRADE_AMOUNT_USD_DEFAULT = float(CONFIG.get("trade_amount_usd", 5))
SLIPPAGE = float(CONFIG.get("slippage", 0.02))

# Guards (EMA + pump)
PRICE_IMPACT_MAX = float(CONFIG.get("price_impact_max_pct", 0.15))  # abort if quote >15% worse than EMA
PUMP_GUARD_MAX   = float(CONFIG.get("pump_guard_max_pct", 0.25))    # abort if quote >25% better than EMA
PRICE_EMA_ALPHA  = float(CONFIG.get("price_ema_alpha", 0.30))        # EMA smoothing

# Liquidity / size sanity
MAX_SIZE_PI      = float(CONFIG.get("max_size_price_impact_pct", 0.20))  # size impact limit, e.g., 20%
MIN_RESERVE_WETH = float(CONFIG.get("min_pair_reserve_weth", 5.0))       # require ≥ this WETH in pool

# Housekeeping
POSITIONS_FILE   = "open_positions.json"
MONITOR_SCRIPT   = "monitor_position.py"
PRICE_MEMORY_FILE= "price_memory.json"

# --- Web3 / Router / Factory ---
if not INFURA_URL:
    raise RuntimeError("INFURA_URL missing in .env / secrets.py")
if not WALLET_ADDRESS:
    raise RuntimeError("WALLET_ADDRESS missing in .env / secrets.py")

w3 = Web3(Web3.HTTPProvider(INFURA_URL))
if not w3.is_connected():
    raise RuntimeError("Web3 not connected — check RPC URL")

WALLET = Web3.to_checksum_address(WALLET_ADDRESS)

# Uniswap V2 router + ABI
ROUTER_ADDR = Web3.to_checksum_address("0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D")
with open("uniswap_router_abi.json", "r") as f:
    ROUTER_ABI = json.load(f)
router = w3.eth.contract(address=ROUTER_ADDR, abi=ROUTER_ABI)

# Uniswap V2 factory + minimal ABIs for reserves
V2_FACTORY = Web3.to_checksum_address("0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f")
V2_FACTORY_ABI = json.loads("""
[
  {"constant":true,"inputs":[{"name":"tokenA","type":"address"},{"name":"tokenB","type":"address"}],
   "name":"getPair","outputs":[{"name":"pair","type":"address"}],"type":"function"}
]
""")
V2_PAIR_ABI = json.loads("""
[
  {"constant":true,"inputs":[],"name":"getReserves",
   "outputs":[{"name":"_reserve0","type":"uint112"},{"name":"_reserve1","type":"uint112"},{"name":"_blockTimestampLast","type":"uint32"}],"type":"function"},
  {"constant":true,"inputs":[],"name":"token0","outputs":[{"name":"","type":"address"}],"type":"function"},
  {"constant":true,"inputs":[],"name":"token1","outputs":[{"name":"","type":"address"}],"type":"function"}
]
""")
factory = w3.eth.contract(address=V2_FACTORY, abi=V2_FACTORY_ABI)

WETH = Web3.to_checksum_address("0xC02aaA39b223FE8D0A0E5C4F27eAD9083C756Cc2")


# ===== Helpers =====
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

def _quote_v2_out(amount_in_wei: int, path: list) -> int:
    """Uniswap V2 router getAmountsOut for exact input; returns raw token amount."""
    amounts = router.functions.getAmountsOut(int(amount_in_wei), path).call()
    return int(amounts[-1])

def _get_pair_address(token_a: str, token_b: str) -> Optional[str]:
    try:
        addr = factory.functions.getPair(Web3.to_checksum_address(token_a),
                                         Web3.to_checksum_address(token_b)).call()
        if int(addr, 16) == 0:
            return None
        return Web3.to_checksum_address(addr)
    except Exception:
        return None

def _get_weth_side_reserve(token_addr: str):
    """
    Returns (reserve_weth_eth, reserve_token_raw) or None.
    reserve_weth_eth is in ETH (not wei). reserve_token_raw is raw units.
    """
    pair_addr = _get_pair_address(WETH, token_addr)
    if not pair_addr:
        return None
    pair = w3.eth.contract(address=pair_addr, abi=V2_PAIR_ABI)
    try:
        r0, r1, _ = pair.functions.getReserves().call()
        t0 = pair.functions.token0().call()
        t1 = pair.functions.token1().call()
    except Exception:
        return None
    if Web3.to_checksum_address(t0) == WETH:
        reserve_weth_wei = int(r0); reserve_token_raw = int(r1)
    elif Web3.to_checksum_address(t1) == WETH:
        reserve_weth_wei = int(r1); reserve_token_raw = int(r0)
    else:
        return None
    return reserve_weth_wei / 1e18, reserve_token_raw

# ---- Price memory (EMA) ----
def _load_price_memory():
    p = Path(PRICE_MEMORY_FILE)
    if not p.exists(): return {}
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


# ===== Entry point used by main.py =====
def execute_trade(token: dict, trade_amount_usd: float = None):
    """
    Pre-trade guards + execution:
      1) Size in USD -> ETH
      2) Liquidity/size sanity (reserve WETH + approx size impact)
      3) V2 quote and EMA guards (price-impact + pump)
      4) Update EMA memory
      5) Delegate actual buy (with re-quote protection) to uniswap_executor.buy_token
      6) If success, log position + launch monitor

    Returns: (tx_hash_hex_or_sim, success_bool)
    """
    symbol = token.get("symbol", "?")
    token_address = Web3.to_checksum_address(token["address"])
    amount_usd = float(trade_amount_usd or TRADE_AMOUNT_USD_DEFAULT)

    # 1) USD -> ETH sizing
    eth_usd = get_eth_price_usd()
    if not eth_usd or eth_usd <= 0:
        print("❌ Failed to compute ETH amount: Could not fetch ETH/USD price for sizing.")
        return None, False
    eth_amount = float(amount_usd) / float(eth_usd)
    value_wei = w3.to_wei(eth_amount, "ether")

    # 2) Liquidity/size sanity
    reserves = _get_weth_side_reserve(token_address)
    if not reserves:
        print("🛑 No V2 pair found or reserves unreadable — aborting buy.")
        return None, False

    reserve_weth_eth, _reserve_token_raw = reserves
    if reserve_weth_eth < MIN_RESERVE_WETH:
        print(f"🛑 Reserve too low: WETH side {reserve_weth_eth:.4f} < {MIN_RESERVE_WETH} — aborting buy.")
        try:
            send_telegram_message(
                f"🛑 *Buy Blocked: Low Reserves*\n"
                f"Token: `{symbol}` `{token_address}`\n"
                f"WETH reserve: {reserve_weth_eth:.4f} < {MIN_RESERVE_WETH}"
            )
        except Exception:
            pass
        return None, False

    approx_impact = eth_amount / reserve_weth_eth if reserve_weth_eth > 0 else 1.0
    if approx_impact > MAX_SIZE_PI:
        print(f"🛑 Size price impact too high: {approx_impact*100:.2f}% > {MAX_SIZE_PI*100:.2f}% — aborting buy.")
        try:
            send_telegram_message(
                f"🛑 *Buy Blocked: Size Impact*\n"
                f"Token: `{symbol}` `{token_address}`\n"
                f"Impact: *{approx_impact*100:.2f}%* (limit {MAX_SIZE_PI*100:.2f}%)\n"
                f"WETH reserve: {reserve_weth_eth:.4f} WETH\n"
                f"Size: {eth_amount:.6f} ETH"
            )
        except Exception:
            pass
        return None, False

    # 3) Quote + EMA guards
    path = [WETH, token_address]
    try:
        quoted_out = _quote_v2_out(value_wei, path)
    except Exception as e:
        print(f"❌ Quote failed for {symbol}: {e}")
        return None, False
    if quoted_out <= 0:
        print(f"❌ Bad quote (0 out) for {symbol}; aborting.")
        return None, False

    curr_tpw = quoted_out / float(value_wei)
    last_tpw, prev_ema = _get_prev_metrics(token_address)

    if prev_ema and prev_ema > 0:
        adverse   = (prev_ema - curr_tpw) / prev_ema   # worse than EMA
        favorable = (curr_tpw - prev_ema) / prev_ema   # better than EMA

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
            except Exception:
                pass
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
            except Exception:
                pass
            return None, False

        print(f"🛡️ EMA guards OK: adverse {(adverse*100):.2f}% | favorable {(favorable*100):.2f}%")
    else:
        print("ℹ️ No EMA history; skipping EMA guards on first observation.")

    # 4) Update EMA memory
    new_ema = _update_price_memory(token_address, curr_tpw)
    print(f"📈 Updated EMA tpw: {new_ema:.12f} (α={PRICE_EMA_ALPHA:.2f})")

    # 5) Execute swap via uniswap_executor (handles re-quote & minOut & EIP-1559)
    tx_hash, ok = buy_token(token_address, amount_usd, symbol)
    if not ok:
        return None, False

    # 6) Log position + start monitor
    _log_position(token)
    _launch_monitor_detached()

    return tx_hash, True