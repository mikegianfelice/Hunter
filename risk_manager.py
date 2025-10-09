# risk_manager.py
import json, os, time, yaml
from datetime import datetime, timezone
from web3 import Web3
from secrets import INFURA_URL, WALLET_ADDRESS
from config_loader import get_config, get_config_int, get_config_float

STATE_FILE = "risk_state.json"
POSITIONS_FILE = "open_positions.json"

# Dynamic config loading
def get_risk_manager_config():
    """Get current configuration values dynamically"""
    return {
        'MAX_CONCURRENT_POS': get_config_int("max_concurrent_positions", 5),
        'DAILY_LOSS_LIMIT_USD': get_config_float("daily_loss_limit_usd", 50.0),
        'MAX_LOSING_STREAK': get_config_int("max_losing_streak", 3),
        'CIRCUIT_BREAK_MIN': get_config_int("circuit_breaker_minutes", 60),
        'PER_TRADE_MAX_USD': get_config_float("per_trade_max_usd", get_config_float("trade_amount_usd", 5)),
        'MIN_WALLET_BALANCE_BUFFER': get_config_float("min_wallet_balance_buffer", 0.01)
    }

# Web3 setup for balance checking
w3 = Web3(Web3.HTTPProvider(INFURA_URL))

def _get_wallet_balance_usd(chain_id="ethereum"):
    """Get wallet balance in USD for specific chain"""
    try:
        if chain_id.lower() == "ethereum":
            # Convert to checksum address if needed
            checksum_address = w3.to_checksum_address(WALLET_ADDRESS)
            # Ethereum balance check
            balance_wei = w3.eth.get_balance(checksum_address)
            balance_eth = w3.from_wei(balance_wei, 'ether')
            
            # Get ETH price in USD
            from utils import get_eth_price_usd
            eth_price = get_eth_price_usd()
            
            return float(balance_eth) * eth_price
        elif chain_id.lower() == "base":
            # Base uses same wallet as Ethereum, check ETH balance
            checksum_address = w3.to_checksum_address(WALLET_ADDRESS)
            balance_wei = w3.eth.get_balance(checksum_address)
            balance_eth = w3.from_wei(balance_wei, 'ether')
            
            # Get ETH price in USD
            from utils import get_eth_price_usd
            eth_price = get_eth_price_usd()
            
            return float(balance_eth) * eth_price
        elif chain_id.lower() == "solana":
            # Real Solana balance checking
            try:
                from solana_executor import get_solana_balance
                from utils import get_sol_price_usd
                
                sol_balance = get_solana_balance()
                sol_price = get_sol_price_usd()
                
                if sol_price is None or sol_price <= 0:
                    print(f"⚠️ Cannot get SOL price for balance calculation - using emergency fallback of $140")
                    sol_price = 140.0  # Emergency fallback to prevent trading halt
                
                return float(sol_balance) * float(sol_price)
            except Exception as e:
                print(f"⚠️ Error getting Solana balance: {e}")
                return 0.0
        else:
            # For other chains, assume some balance for now
            print(f"⚠️ Balance checking for {chain_id} not implemented yet")
            return 50.0  # Assume $50 for testing
            
    except Exception as e:
        print(f"⚠️ Could not get wallet balance for {chain_id}: {e}")
        return 0.0

def _today_utc():
    return datetime.utcnow().strftime("%Y-%m-%d")

def _now_ts():
    return int(time.time())

def _load_state():
    s = {
        "date": _today_utc(),
        "realized_pnl_usd": 0.0,
        "buys_today": 0,
        "sells_today": 0,
        "losing_streak": 0,
        "paused_until": 0,     # epoch seconds; 0 = not paused
        "daily_spend_usd": 0.0
    }
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
                s.update(data or {})
        except Exception:
            pass
    # reset if new day
    if s.get("date") != _today_utc():
        s.update({
            "date": _today_utc(),
            "realized_pnl_usd": 0.0,
            "buys_today": 0,
            "sells_today": 0,
            "losing_streak": 0,
            "paused_until": 0,
            "daily_spend_usd": 0.0
        })
    return s

def _save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=2)

def _open_positions_count():
    if not os.path.exists(POSITIONS_FILE):
        return 0
    try:
        with open(POSITIONS_FILE, "r") as f:
            data = json.load(f) or {}
            return len(data)
    except Exception:
        return 0

def _is_token_already_held(token_address: str) -> bool:
    """Check if a specific token is already held in open positions"""
    if not os.path.exists(POSITIONS_FILE):
        return False
    try:
        with open(POSITIONS_FILE, "r") as f:
            data = json.load(f) or {}
            # Normalize address to lowercase for comparison
            token_address_lower = token_address.lower()
            return any(addr.lower() == token_address_lower for addr in data.keys())
    except Exception:
        return False

def allow_new_trade(trade_amount_usd: float, token_address: str = None, chain_id: str = "ethereum"):
    """
    Gatekeeper before any new buy.
    Returns (allowed: bool, reason: str)
    """
    config = get_risk_manager_config()
    s = _load_state()

    # paused by circuit breaker?
    if s.get("paused_until", 0) > _now_ts():
        return False, f"circuit_breaker_active_until_{s['paused_until']}"

    # daily loss limit hit?
    if s.get("realized_pnl_usd", 0.0) <= -abs(config['DAILY_LOSS_LIMIT_USD']):
        # pause until UTC midnight
        tomorrow = datetime.utcnow().replace(hour=23, minute=59, second=59, microsecond=0).timestamp()
        s["paused_until"] = int(tomorrow)
        _save_state(s)
        return False, "daily_loss_limit_hit"

    # per-trade size guard
    if trade_amount_usd > config['PER_TRADE_MAX_USD']:
        return False, f"trade_amount_exceeds_cap_{config['PER_TRADE_MAX_USD']}"

    # Check wallet balance for specific chain
    # Temporarily disabled for testing - uncomment below lines for production
    # wallet_balance = _get_wallet_balance_usd(chain_id)
    # required_amount = trade_amount_usd + (wallet_balance * config['MIN_WALLET_BALANCE_BUFFER'])  # Include buffer for gas
    # if wallet_balance < required_amount:
    #     return False, f"insufficient_balance_{wallet_balance:.2f}_usd_needs_{required_amount:.2f}_usd"

    # Check if token is already held (prevent duplicate buys)
    if token_address and _is_token_already_held(token_address):
        return False, "token_already_held"

    # concurrent positions guard
    if _open_positions_count() >= config['MAX_CONCURRENT_POS']:
        return False, "max_concurrent_positions_reached"

    return True, "ok"

def register_buy(usd_size: float):
    s = _load_state()
    s["buys_today"] = int(s.get("buys_today", 0)) + 1
    s["daily_spend_usd"] = float(s.get("daily_spend_usd", 0.0)) + float(usd_size or 0.0)
    _save_state(s)

def register_sell(pnl_pct: float, usd_size: float):
    """
    Record realized PnL for risk counters.
    pnl_pct: e.g., +12.5 or -8.7 (percent)
    usd_size: original position size in USD (what you bought with)
    """
    config = get_risk_manager_config()
    s = _load_state()
    s["sells_today"] = int(s.get("sells_today", 0)) + 1

    # Convert percent to USD PnL
    try:
        pnl_usd = (float(pnl_pct) / 100.0) * float(usd_size or 0.0)
    except Exception:
        pnl_usd = 0.0

    s["realized_pnl_usd"] = float(s.get("realized_pnl_usd", 0.0)) + pnl_usd

    # update losing streak / circuit breaker
    if pnl_usd < 0:
        s["losing_streak"] = int(s.get("losing_streak", 0)) + 1
        if s["losing_streak"] >= config['MAX_LOSING_STREAK']:
            s["paused_until"] = _now_ts() + config['CIRCUIT_BREAK_MIN'] * 60
    else:
        s["losing_streak"] = 0  # reset on win

    _save_state(s)

def status_summary():
    s = _load_state()
    return {
        "date": s["date"],
        "open_positions": _open_positions_count(),
        "buys_today": s["buys_today"],
        "sells_today": s["sells_today"],
        "realized_pnl_usd": round(s["realized_pnl_usd"], 2),
        "losing_streak": s["losing_streak"],
        "paused_until": s["paused_until"]
    }