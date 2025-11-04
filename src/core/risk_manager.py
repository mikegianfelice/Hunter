# risk_manager.py
import json, os, time, yaml
from datetime import datetime, timezone
from web3 import Web3
from src.config.secrets import INFURA_URL, WALLET_ADDRESS
from src.config.config_loader import get_config, get_config_int, get_config_float

STATE_FILE = "data/risk_state.json"
POSITIONS_FILE = "data/open_positions.json"

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

def get_wallet_tier(wallet_balance_usd: float) -> dict:
    """Resolve wallet tier from config.yaml:wallet_tiers based on total USD balance."""
    tiers = get_config('wallet_tiers', {}) or {}
    if not isinstance(tiers, dict) or not tiers:
        return {
            'tier_name': 'unknown',
            'description': 'No tiers configured',
            'tier_config': {
                'max_total_exposure_usd': 100.0,
                'max_position_size_usd': 10.0
            }
        }

    # Find matching tier
    selected_name, selected_cfg = None, None
    try:
        wallet_balance_usd = float(wallet_balance_usd)
        for name, cfg in tiers.items():
            min_b = float(cfg.get('min_balance', 0))
            max_b = float(cfg.get('max_balance', float('inf')))
            if min_b <= wallet_balance_usd <= max_b:
                selected_name, selected_cfg = name, cfg
                break
    except Exception as e:
        print(f"⚠️ Error parsing wallet tiers: {e}")

    # Fallback to closest boundary if not found
    if not selected_cfg:
        try:
            items = sorted(tiers.items(), key=lambda kv: float(kv[1].get('min_balance', 0)))
            if items:
                if wallet_balance_usd < float(items[0][1].get('min_balance', 0)):
                    selected_name, selected_cfg = items[0]
                else:
                    selected_name, selected_cfg = items[-1]
        except Exception:
            pass

    # Final fallback if still nothing found
    if not selected_cfg:
        return {
            'tier_name': 'unknown',
            'description': 'Default tier',
            'tier_config': {
                'max_total_exposure_usd': 100.0,
                'max_position_size_usd': 10.0
            }
        }

    return {
        'tier_name': selected_name or 'unknown',
        'description': selected_cfg.get('description', ''),
        'tier_config': selected_cfg,
    }

def get_tier_based_risk_limits(wallet_balance_usd: float = None):
    """Get risk limits based on wallet tier using combined balance from all chains"""
    try:
        # If no balance provided, get combined balance from all chains
        if wallet_balance_usd is None:
            wallet_balance_usd = _get_combined_wallet_balance_usd()
        
        tier_info = get_wallet_tier(wallet_balance_usd)
        tier_config = tier_info['tier_config']
        tier_name = tier_info['tier_name']
        
        # Calculate tier-based limits
        max_total_exposure = tier_config.get('max_total_exposure_usd', 100.0)
        max_position_size = tier_config.get('max_position_size_usd', 10.0)
        
        # Scale daily loss limit based on wallet size (5% of wallet)
        daily_loss_limit = wallet_balance_usd * 0.05
        
        # Scale concurrent positions based on wallet size
        if wallet_balance_usd < 1000:
            max_concurrent_pos = 3
        elif wallet_balance_usd < 5000:
            max_concurrent_pos = 5
        elif wallet_balance_usd < 20000:
            max_concurrent_pos = 8
        else:
            max_concurrent_pos = 10
        
        return {
            'MAX_CONCURRENT_POS': max_concurrent_pos,
            'DAILY_LOSS_LIMIT_USD': daily_loss_limit,
            'MAX_LOSING_STREAK': 3,  # Keep consistent
            'CIRCUIT_BREAK_MIN': 60,  # Keep consistent
            'PER_TRADE_MAX_USD': max_position_size,
            'MAX_TOTAL_EXPOSURE_USD': max_total_exposure,
            'MIN_WALLET_BALANCE_BUFFER': 0.05,  # 5% buffer
            'TIER_NAME': tier_name,
            'TIER_DESCRIPTION': tier_info['description']
        }
        
    except Exception as e:
        print(f"⚠️ Tier-based risk limits failed: {e}, using defaults")
        return get_risk_manager_config()

# Web3 setup for balance checking
w3 = Web3(Web3.HTTPProvider(INFURA_URL))

def _get_wallet_balance_usd(chain_id="ethereum"):
    """
    Get wallet balance in USD for specific chain
    
    Returns:
        float: Wallet balance in USD
        None: If balance check failed after retries (rate limit or RPC error)
    """
    try:
        if chain_id.lower() == "ethereum":
            # Convert to checksum address if needed
            checksum_address = w3.to_checksum_address(WALLET_ADDRESS)
            # Ethereum balance check
            balance_wei = w3.eth.get_balance(checksum_address)
            balance_eth = w3.from_wei(balance_wei, 'ether')
            
            # Get ETH price in USD
            from src.utils.utils import get_eth_price_usd
            eth_price = get_eth_price_usd()
            
            if eth_price is None or eth_price <= 0:
                print(f"⚠️ Could not get ETH price for balance calculation - using emergency fallback of $3000")
                eth_price = 3000.0  # Emergency fallback to prevent trading halt
            
            return float(balance_eth) * eth_price
        elif chain_id.lower() == "base":
            # Base uses same wallet as Ethereum, check ETH balance
            checksum_address = w3.to_checksum_address(WALLET_ADDRESS)
            balance_wei = w3.eth.get_balance(checksum_address)
            balance_eth = w3.from_wei(balance_wei, 'ether')
            
            # Get ETH price in USD
            from src.utils.utils import get_eth_price_usd
            eth_price = get_eth_price_usd()
            
            if eth_price is None or eth_price <= 0:
                print(f"⚠️ Could not get ETH price for balance calculation - using emergency fallback of $3000")
                eth_price = 3000.0  # Emergency fallback to prevent trading halt
            
            return float(balance_eth) * eth_price
        elif chain_id.lower() == "solana":
            # Real Solana balance checking
            try:
                from src.config.config_loader import get_config
                
                # Check if base currency is USDC or SOL
                base_currency = get_config("solana_base_currency", "USDC")
                
                if base_currency.upper() == "USDC":
                    # Get USDC balance when using USDC as base currency
                    # Add retry logic at this level in case of rate limits
                    max_balance_retries = 3
                    retry_delay = 2.0
                    
                    for balance_attempt in range(max_balance_retries):
                        try:
                            from src.execution.jupiter_executor import JupiterCustomExecutor
                            executor = JupiterCustomExecutor()
                            usdc_balance = executor.get_usdc_balance()
                            
                            if usdc_balance is not None:
                                # Success - USDC is 1:1 with USD, so return directly
                                print(f"✅ USDC balance check successful: ${usdc_balance:.2f}")
                                return float(usdc_balance)
                            
                            # Balance check failed (likely rate limit)
                            if balance_attempt < max_balance_retries - 1:
                                wait_time = retry_delay * (2 ** balance_attempt)
                                print(f"⚠️ USDC balance check failed (attempt {balance_attempt + 1}/{max_balance_retries}), retrying in {wait_time:.1f}s...")
                                time.sleep(wait_time)
                            else:
                                # All retries exhausted - return None to indicate check failure
                                print(f"⚠️ Failed to get USDC balance after {max_balance_retries} attempts - cannot verify balance")
                                return None
                                
                        except Exception as e:
                            if balance_attempt < max_balance_retries - 1:
                                wait_time = retry_delay * (2 ** balance_attempt)
                                print(f"⚠️ Error getting USDC balance (attempt {balance_attempt + 1}/{max_balance_retries}): {e}, retrying in {wait_time:.1f}s...")
                                time.sleep(wait_time)
                            else:
                                print(f"⚠️ Error getting USDC balance after {max_balance_retries} attempts: {e}")
                                return None
                    
                    # Should not reach here, but just in case
                    return None
                else:
                    # Fallback to SOL balance checking
                    from src.execution.solana_executor import get_solana_balance
                    from src.utils.utils import get_sol_price_usd
                    
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
            # For other chains, return 0.0 instead of assuming a test balance
            # Balance checking should be implemented for each chain separately
            print(f"⚠️ Balance checking for {chain_id} not implemented yet - returning 0.0")
            return 0.0  # Return 0 instead of simulated test balance
            
    except Exception as e:
        print(f"⚠️ Could not get wallet balance for {chain_id}: {e}")
        return 0.0


def _get_combined_wallet_balance_usd():
    """Get combined wallet balance in USD from all supported chains (Ethereum + Solana)"""
    try:
        total_balance = 0.0
        
        # Get Ethereum balance
        try:
            eth_balance = _get_wallet_balance_usd("ethereum")
            total_balance += eth_balance
            print(f"💰 Ethereum balance: ${eth_balance:.2f}")
        except Exception as e:
            print(f"⚠️ Failed to get Ethereum balance: {e}")
        
        # Get Solana balance
        try:
            sol_balance = _get_wallet_balance_usd("solana")
            total_balance += sol_balance
            print(f"💰 Solana balance: ${sol_balance:.2f}")
        except Exception as e:
            print(f"⚠️ Failed to get Solana balance: {e}")
        
        print(f"💰 Combined wallet balance: ${total_balance:.2f}")
        return total_balance
        
    except Exception as e:
        print(f"⚠️ Error getting combined wallet balance: {e}")
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
    os.makedirs('data', exist_ok=True)
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

def _get_current_exposure_usd() -> float:
    """Calculate current total exposure in USD across all open positions"""
    try:
        total_exposure = 0.0
        
        # Use performance_data.json as source of truth (has accurate position_size_usd)
        perf_file = "data/performance_data.json"
        if os.path.exists(perf_file):
            try:
                with open(perf_file, 'r') as f:
                    perf_data = json.load(f)
                    trades = perf_data.get("trades", [])
                    
                    # Track unique token addresses to avoid duplicates
                    seen_addresses = set()
                    
                    for trade in trades:
                        if trade.get("status") == "open":
                            address = trade.get("address", "").lower()
                            
                            # Skip duplicates - only count each token address once
                            # (take the most recent one if there are duplicates)
                            if address in seen_addresses:
                                continue
                            
                            seen_addresses.add(address)
                            position_size = trade.get("position_size_usd", 0.0)
                            if position_size > 0:
                                total_exposure += position_size
            except Exception as e:
                print(f"⚠️ Failed to read performance_data.json for exposure: {e}")
        
        # Fallback to open_positions.json if performance_data doesn't exist or has no data
        if total_exposure == 0.0 and os.path.exists(POSITIONS_FILE):
            try:
                with open(POSITIONS_FILE, 'r') as f:
                    data = json.load(f)
                    for token_address, position_data in data.items():
                        position_size = position_data.get('position_size_usd', 0.0)
                        if position_size > 0:
                            total_exposure += position_size
            except Exception as e:
                print(f"⚠️ Failed to read open_positions.json for exposure: {e}")
        
        return total_exposure
            
    except Exception as e:
        print(f"⚠️ Failed to calculate current exposure: {e}")
        return 0.0

def allow_new_trade(trade_amount_usd: float, token_address: str = None, chain_id: str = "ethereum"):
    """
    Gatekeeper before any new buy.
    Returns (allowed: bool, reason: str)
    """
    # Get combined wallet balance from all chains for tier-based limits
    combined_wallet_balance = _get_combined_wallet_balance_usd()
    config = get_tier_based_risk_limits(combined_wallet_balance)
    s = _load_state()
    
    # Log tier information
    tier_name = config.get('TIER_NAME', 'unknown')
    tier_description = config.get('TIER_DESCRIPTION', 'Unknown Tier')
    print(f"🎯 Risk Manager - Tier: {tier_name} ({tier_description}), Combined Balance: ${combined_wallet_balance:.2f}")

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

    # Check wallet balance for specific chain (still need chain-specific balance for gas)
    chain_wallet_balance = _get_wallet_balance_usd(chain_id)
    
    # Handle balance check failure (None indicates rate limit or RPC error after retries)
    if chain_wallet_balance is None:
        return False, "balance_check_failed_rate_limit_or_rpc_error_cannot_verify_balance"
    
    required_amount = trade_amount_usd + (chain_wallet_balance * config['MIN_WALLET_BALANCE_BUFFER'])  # Include buffer for gas
    if chain_wallet_balance < required_amount:
        return False, f"insufficient_balance_{chain_wallet_balance:.2f}_usd_needs_{required_amount:.2f}_usd"

    # Check if token is already held (prevent duplicate buys)
    if token_address and _is_token_already_held(token_address):
        return False, "token_already_held"

    # concurrent positions guard
    if _open_positions_count() >= config['MAX_CONCURRENT_POS']:
        return False, "max_concurrent_positions_reached"
    
    # total exposure guard (tier-based) - uses combined balance for tier calculation
    max_total_exposure = config.get('MAX_TOTAL_EXPOSURE_USD', 100.0)
    current_exposure = _get_current_exposure_usd()
    if current_exposure + trade_amount_usd > max_total_exposure:
        return False, f"total_exposure_limit_exceeded_{current_exposure:.2f}_{max_total_exposure:.2f}"

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