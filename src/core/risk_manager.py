# risk_manager.py
import os
import time
from datetime import datetime, timezone
from web3 import Web3

from src.config.secrets import INFURA_URL, WALLET_ADDRESS
from src.config.config_loader import get_config, get_config_int, get_config_float
from src.storage.positions import load_positions as load_positions_store
from src.storage.risk import (
    load_risk_state as load_risk_state_store,
    save_risk_state as save_risk_state_store,
    load_balance_cache as load_balance_cache_store,
    save_balance_cache as save_balance_cache_store,
)
from src.utils.position_sync import resolve_token_address

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
        # Support percentage-based exposure for dynamic scaling
        max_total_exposure_percent = tier_config.get('max_total_exposure_percent', None)
        if max_total_exposure_percent is not None:
            # Calculate exposure as percentage of wallet balance
            max_total_exposure = wallet_balance_usd * max_total_exposure_percent
        else:
            # Fall back to fixed dollar amount
            max_total_exposure = tier_config.get('max_total_exposure_usd', 100.0)
        
        # Support percentage-based position sizing for dynamic scaling
        max_position_size_percent = tier_config.get('max_position_size_percent', None)
        if max_position_size_percent is not None:
            # Calculate max position size as percentage of wallet balance
            max_position_size = wallet_balance_usd * max_position_size_percent
        else:
            # Fall back to fixed dollar amount
            max_position_size = tier_config.get('max_position_size_usd', 10.0)
        
        # Support percentage-based base position size for dynamic scaling
        base_position_size_percent = tier_config.get('base_position_size_percent', None)
        if base_position_size_percent is not None:
            # Calculate base position size as percentage of wallet balance
            base_position_size = wallet_balance_usd * base_position_size_percent
        else:
            # Fall back to fixed dollar amount
            base_position_size = tier_config.get('base_position_size_usd', 5.0)
        
        # Scale daily loss limit based on wallet size (5% of wallet)
        daily_loss_limit = wallet_balance_usd * 0.05
        
        # Scale concurrent positions based on wallet size, but respect config.yaml override
        config_max_concurrent = get_config_int("max_concurrent_positions", None)
        if config_max_concurrent is not None:
            # Use config value if explicitly set
            max_concurrent_pos = config_max_concurrent
        elif wallet_balance_usd < 1000:
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
            'BASE_POSITION_SIZE_USD': base_position_size,
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

def _get_wallet_balance_usd(chain_id="ethereum", use_cache_fallback=True):
    """
    Get wallet balance in USD for specific chain
    
    Args:
        chain_id: Chain identifier (ethereum, base, solana)
        use_cache_fallback: If True, use cached balance as fallback when RPC fails
    
    Returns:
        float: Wallet balance in USD
        None: If balance check failed after retries and no valid cache available
    """
    try:
        if chain_id.lower() == "ethereum":
            # Convert to checksum address if needed
            checksum_address = w3.to_checksum_address(WALLET_ADDRESS)
            # Ethereum balance check
            try:
                balance_wei = w3.eth.get_balance(checksum_address)
                balance_eth = w3.from_wei(balance_wei, 'ether')
                
                # Get ETH price in USD
                from src.utils.utils import get_eth_price_usd
                eth_price = get_eth_price_usd()
                
                if eth_price is None or eth_price <= 0:
                    print(f"⚠️ Could not get ETH price for balance calculation - using emergency fallback of $3000")
                    eth_price = 3000.0  # Emergency fallback to prevent trading halt
                
                balance_usd = float(balance_eth) * eth_price
                # Cache the successful balance
                _update_balance_cache(chain_id, balance_usd)
                return balance_usd
            except Exception as e:
                print(f"⚠️ Error getting Ethereum balance: {e}")
                if use_cache_fallback:
                    cached_balance, is_valid = _get_cached_balance(chain_id)
                    if is_valid:
                        print(f"✅ Using cached Ethereum balance (${cached_balance:.2f})")
                        return cached_balance
                return None
        elif chain_id.lower() == "base":
            # Base uses same wallet as Ethereum, check ETH balance
            checksum_address = w3.to_checksum_address(WALLET_ADDRESS)
            try:
                balance_wei = w3.eth.get_balance(checksum_address)
                balance_eth = w3.from_wei(balance_wei, 'ether')
                
                # Get ETH price in USD
                from src.utils.utils import get_eth_price_usd
                eth_price = get_eth_price_usd()
                
                if eth_price is None or eth_price <= 0:
                    print(f"⚠️ Could not get ETH price for balance calculation - using emergency fallback of $3000")
                    eth_price = 3000.0  # Emergency fallback to prevent trading halt
                
                balance_usd = float(balance_eth) * eth_price
                # Cache the successful balance
                _update_balance_cache(chain_id, balance_usd)
                return balance_usd
            except Exception as e:
                print(f"⚠️ Error getting Base balance: {e}")
                if use_cache_fallback:
                    cached_balance, is_valid = _get_cached_balance(chain_id)
                    if is_valid:
                        print(f"✅ Using cached Base balance (${cached_balance:.2f})")
                        return cached_balance
                return None
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
                                # Cache the successful balance
                                _update_balance_cache(chain_id, usdc_balance)
                                return float(usdc_balance)
                            
                            # Balance check failed (likely rate limit)
                            if balance_attempt < max_balance_retries - 1:
                                wait_time = retry_delay * (2 ** balance_attempt)
                                print(f"⚠️ USDC balance check failed (attempt {balance_attempt + 1}/{max_balance_retries}), retrying in {wait_time:.1f}s...")
                                time.sleep(wait_time)
                            else:
                                # All retries exhausted - try cache fallback
                                print(f"⚠️ Failed to get USDC balance after {max_balance_retries} attempts")
                                if use_cache_fallback:
                                    cached_balance, is_valid = _get_cached_balance(chain_id)
                                    if is_valid:
                                        print(f"✅ Using cached balance (${cached_balance:.2f}) - cache is fresh")
                                        return cached_balance
                                    elif cached_balance is not None:
                                        print(f"⚠️ Cache exists but is stale - rejecting trade for safety")
                                        return None
                                    else:
                                        print(f"⚠️ No cache available - cannot verify balance")
                                        return None
                                else:
                                    print(f"⚠️ Cannot verify balance (cache fallback disabled)")
                                    return None
                                
                        except Exception as e:
                            if balance_attempt < max_balance_retries - 1:
                                wait_time = retry_delay * (2 ** balance_attempt)
                                print(f"⚠️ Error getting USDC balance (attempt {balance_attempt + 1}/{max_balance_retries}): {e}, retrying in {wait_time:.1f}s...")
                                time.sleep(wait_time)
                            else:
                                print(f"⚠️ Error getting USDC balance after {max_balance_retries} attempts: {e}")
                                # Try cache fallback
                                if use_cache_fallback:
                                    cached_balance, is_valid = _get_cached_balance(chain_id)
                                    if is_valid:
                                        print(f"✅ Using cached balance (${cached_balance:.2f}) - cache is fresh")
                                        return cached_balance
                                    elif cached_balance is not None:
                                        print(f"⚠️ Cache exists but is stale - rejecting trade for safety")
                                        return None
                                    else:
                                        print(f"⚠️ No cache available - cannot verify balance")
                                        return None
                                else:
                                    return None
                    
                    # Should not reach here, but just in case
                    return None
                else:
                    # Fallback to SOL balance checking
                    from src.execution.solana_executor import get_solana_balance
                    from src.utils.utils import get_sol_price_usd
                    
                    try:
                        sol_balance = get_solana_balance()
                        sol_price = get_sol_price_usd()
                        
                        if sol_price is None or sol_price <= 0:
                            print(f"⚠️ Cannot get SOL price for balance calculation - using emergency fallback of $140")
                            sol_price = 140.0  # Emergency fallback to prevent trading halt
                        
                        balance_usd = float(sol_balance) * float(sol_price)
                        # Cache the successful balance
                        _update_balance_cache(chain_id, balance_usd)
                        return balance_usd
                    except Exception as e:
                        print(f"⚠️ Error getting SOL balance: {e}")
                        if use_cache_fallback:
                            cached_balance, is_valid = _get_cached_balance(chain_id)
                            if is_valid:
                                print(f"✅ Using cached SOL balance (${cached_balance:.2f})")
                                return cached_balance
                        return None
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
    """Get combined wallet balance in USD from all supported chains"""
    try:
        total_balance = 0.0
        
        # Get supported chains from config
        supported_chains = get_config("supported_chains", ["solana", "base"])
        if not isinstance(supported_chains, list):
            supported_chains = ["solana", "base"]  # Fallback
        
        # Only check balances for supported chains
        for chain_id in supported_chains:
            try:
                chain_balance = _get_wallet_balance_usd(chain_id)
                if chain_balance is not None:
                    total_balance += chain_balance
                    chain_name = chain_id.capitalize()
                    print(f"💰 {chain_name} balance: ${chain_balance:.2f}")
                else:
                    chain_name = chain_id.capitalize()
                    print(f"⚠️ {chain_name} balance unavailable (None returned)")
            except Exception as e:
                chain_name = chain_id.capitalize()
                print(f"⚠️ Failed to get {chain_name} balance: {e}")
        
        print(f"💰 Combined wallet balance: ${total_balance:.2f}")
        return total_balance
        
    except Exception as e:
        print(f"⚠️ Error getting combined wallet balance: {e}")
        return 0.0

def _today_utc():
    return datetime.utcnow().strftime("%Y-%m-%d")

def _now_ts():
    return int(time.time())

def _load_balance_cache():
    """Load balance cache from persistent storage"""
    return load_balance_cache_store().copy()


def _save_balance_cache(cache_data):
    """Persist balance cache via storage layer"""
    try:
        save_balance_cache_store(cache_data or {})
    except Exception as e:
        print(f"⚠️ Failed to save balance cache: {e}")

def _get_cached_balance(chain_id: str, cache_ttl_seconds: int = None) -> tuple:
    """
    Get cached balance for a chain if it exists and is fresh
    
    Returns:
        tuple: (balance: float or None, is_valid: bool)
    """
    # Get cache TTL from config if not provided
    if cache_ttl_seconds is None:
        cache_ttl_seconds = get_config_int("balance_cache_ttl_seconds", 300)
    
    cache = _load_balance_cache()
    chain_key = chain_id.lower()
    
    if chain_key not in cache:
        return None, False
    
    cached_entry = cache[chain_key]
    cached_balance = cached_entry.get("balance")
    cached_timestamp = cached_entry.get("timestamp", 0)
    
    if cached_balance is None:
        return None, False
    
    # Check if cache is still valid (within TTL)
    age_seconds = _now_ts() - cached_timestamp
    if age_seconds > cache_ttl_seconds:
        return cached_balance, False  # Cache exists but is stale
    
    return cached_balance, True  # Cache is valid

def _update_balance_cache(chain_id: str, balance: float):
    """Update balance cache with new successful balance check"""
    cache = _load_balance_cache()
    chain_key = chain_id.lower()
    cache[chain_key] = {
        "balance": float(balance),
        "timestamp": _now_ts()
    }
    _save_balance_cache(cache)

def _load_state():
    base_state = {
        "date": _today_utc(),
        "realized_pnl_usd": 0.0,
        "buys_today": 0,
        "sells_today": 0,
        "losing_streak": 0,
        "paused_until": 0,
        "daily_spend_usd": 0.0,
    }
    try:
        stored = load_risk_state_store() or {}
        if isinstance(stored, dict):
            base_state.update(stored)
    except Exception as e:
        print(f"⚠️ Failed to load risk state from storage: {e}")

    if base_state.get("date") != _today_utc():
        base_state.update(
            {
                "date": _today_utc(),
                "realized_pnl_usd": 0.0,
                "buys_today": 0,
                "sells_today": 0,
                "losing_streak": 0,
                "paused_until": 0,
                "daily_spend_usd": 0.0,
            }
        )
    return base_state


def _save_state(state):
    try:
        save_risk_state_store(state or {})
    except Exception as e:
        print(f"⚠️ Failed to persist risk state: {e}")

def _open_positions_count():
    """
    Get current count of open positions.
    This function always loads fresh data from the store to ensure accuracy
    after positions are closed/sold.
    """
    try:
        positions = load_positions_store()
        count = len(positions)
        # Always log position count when checking (helps debug race conditions)
        position_keys = list(positions.keys())[:5]  # Show first 5 for debugging
        print(f"🔍 Position count check: {count} positions (keys: {position_keys}...)")
        return count
    except Exception as e:
        print(f"⚠️ Error loading positions for count: {e}")
        # Return 0 on error to allow trading (fail open)
        return 0


def _is_token_already_held(token_address: str) -> bool:
    """Check if a specific token is already held in open positions"""
    if not token_address:
        return False
    
    positions = load_positions_store()
    token_address_lower = token_address.lower().strip()
    
    # Debug logging (can be enabled via environment variable for troubleshooting)
    debug_enabled = os.getenv("DEBUG_TOKEN_HELD_CHECK", "false").lower() == "true"
    
    if debug_enabled:
        print(f"🔍 Checking if token {token_address_lower} is already held. Current positions: {list(positions.keys())}")
    
    for position_key, payload in positions.items():
        try:
            resolved = resolve_token_address(position_key, payload).lower().strip()
        except Exception as e:
            if debug_enabled:
                print(f"⚠️ Error resolving address for position_key {position_key}: {e}")
            resolved = position_key.lower().strip()
        
        if debug_enabled:
            print(f"  - Position key: {position_key}, Resolved address: {resolved}, Match: {resolved == token_address_lower}")
        
        if resolved == token_address_lower:
            if debug_enabled:
                print(f"✅ Token {token_address_lower} found in position {position_key}")
            else:
                # Always log when a match is found (even without debug mode) to help diagnose false positives
                print(f"⚠️ Token {token_address_lower} marked as already held (position key: {position_key}, resolved: {resolved})")
            return True
    
    if debug_enabled:
        print(f"❌ Token {token_address_lower} not found in any positions")
    return False


def _get_current_exposure_usd() -> float:
    """Calculate current total exposure in USD across all open positions"""
    try:
        total_exposure = 0.0
        positions = load_positions_store()
        for payload in positions.values():
            if isinstance(payload, dict):
                total_exposure += float(payload.get("position_size_usd", 0.0) or 0.0)
        return total_exposure
    except Exception as e:
        print(f"⚠️ Failed to calculate exposure from positions store: {e}")
        return 0.0

def allow_new_trade(trade_amount_usd: float, token_address: str = None, chain_id: str = "ethereum", 
                    recommended_position_size: float = None, signal: str = None):
    """
    Gatekeeper before any new buy.
    Returns (allowed: bool, reason: str, is_add_to_position: bool, additional_amount: float)
    
    Args:
        trade_amount_usd: Amount to trade in USD
        token_address: Token address to check
        chain_id: Chain ID
        recommended_position_size: AI-recommended total position size (optional)
        signal: Trading signal ('buy', 'sell', 'hold') (optional)
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
        return False, f"circuit_breaker_active_until_{s['paused_until']}", False, 0.0
    
    # daily loss limit hit?
    if s.get("realized_pnl_usd", 0.0) <= -abs(config['DAILY_LOSS_LIMIT_USD']):
        # pause until UTC midnight
        tomorrow = datetime.utcnow().replace(hour=23, minute=59, second=59, microsecond=0).timestamp()
        s["paused_until"] = int(tomorrow)
        _save_state(s)
        return False, "daily_loss_limit_hit", False, 0.0

    # per-trade size guard
    if trade_amount_usd > config['PER_TRADE_MAX_USD']:
        return False, f"trade_amount_exceeds_cap_{config['PER_TRADE_MAX_USD']}", False, 0.0

    # Check wallet balance for specific chain (still need chain-specific balance for gas)
    chain_wallet_balance = _get_wallet_balance_usd(chain_id, use_cache_fallback=True)
    
    # Handle balance check failure (None indicates rate limit or RPC error after retries and no valid cache)
    if chain_wallet_balance is None:
        # Check if we have a stale cache (exists but expired)
        cached_balance, is_valid = _get_cached_balance(chain_id)
        if cached_balance is not None and not is_valid:
            return False, "balance_check_failed_cache_stale_requires_fresh_balance_check", False, 0.0
        return False, "balance_check_failed_rate_limit_or_rpc_error_cannot_verify_balance", False, 0.0
    
    # When using cached balance, apply a conservative buffer for safety
    # Check if we're using cached balance (by checking if cache is valid)
    cached_balance, cache_is_valid = _get_cached_balance(chain_id)
    if cache_is_valid and abs(cached_balance - chain_wallet_balance) < 0.01:
        # We're using cached balance - apply conservative reduction from config
        conservative_buffer = get_config_float("balance_cache_conservative_buffer", 0.1)
        conservative_balance = chain_wallet_balance * (1.0 - conservative_buffer)
        buffer_pct = conservative_buffer * 100
        print(f"⚠️ Using cached balance with {buffer_pct:.0f}% conservative buffer: ${conservative_balance:.2f} (from ${chain_wallet_balance:.2f})")
        chain_wallet_balance = conservative_balance
    
    # Check if token is already held - NEW LOGIC: Allow adding if conditions met
    if token_address and _is_token_already_held(token_address):
        # Check if we should allow adding to existing position
        if recommended_position_size is not None and signal == "buy":
            positions = load_positions_store()
            token_address_lower = token_address.lower().strip()
            
            # Find existing position
            existing_position = None
            existing_position_key = None
            for position_key, payload in positions.items():
                try:
                    resolved = resolve_token_address(position_key, payload).lower().strip()
                    if resolved == token_address_lower:
                        existing_position = payload if isinstance(payload, dict) else {}
                        existing_position_key = position_key
                        break
                except Exception:
                    continue
            
            if existing_position:
                current_position_size = float(existing_position.get("position_size_usd", 0.0) or 0.0)
                
                # Check if recommended size is higher than current position
                if recommended_position_size > current_position_size:
                    additional_amount = recommended_position_size - current_position_size
                    
                    # Check if we have enough balance for the additional amount
                    required_amount = additional_amount + (chain_wallet_balance * config['MIN_WALLET_BALANCE_BUFFER'])
                    if chain_wallet_balance >= required_amount:
                        # Check total exposure limit (use current exposure + additional amount)
                        max_total_exposure = config.get('MAX_TOTAL_EXPOSURE_USD', 100.0)
                        current_exposure = _get_current_exposure_usd()
                        if current_exposure + additional_amount <= max_total_exposure:
                            print(f"✅ Adding to existing position: current=${current_position_size:.2f}, "
                                  f"recommended=${recommended_position_size:.2f}, additional=${additional_amount:.2f}")
                            return True, "ok_add_to_position", True, additional_amount
                        else:
                            return False, f"total_exposure_limit_exceeded_{current_exposure:.2f}_{max_total_exposure:.2f}", False, 0.0
                    else:
                        return False, f"insufficient_balance_for_addition_{chain_wallet_balance:.2f}_usd_needs_{required_amount:.2f}_usd", False, 0.0
        
        # Default: block duplicate buys
        return False, "token_already_held", False, 0.0

    # For new positions, check balance
    required_amount = trade_amount_usd + (chain_wallet_balance * config['MIN_WALLET_BALANCE_BUFFER'])  # Include buffer for gas
    if chain_wallet_balance < required_amount:
        return False, f"insufficient_balance_{chain_wallet_balance:.2f}_usd_needs_{required_amount:.2f}_usd", False, 0.0

    # concurrent positions guard
    # CRITICAL: Always get fresh position count to ensure we see positions that were just closed
    open_count = _open_positions_count()
    max_concurrent = config['MAX_CONCURRENT_POS']
    
    if open_count >= max_concurrent:
        # Log detailed info to help debug why trades are blocked
        print(f"🚫 Trade blocked: {open_count} open positions >= max {max_concurrent}")
        try:
            positions = load_positions_store()
            if positions:
                position_keys = list(positions.keys())
                print(f"📊 All current positions ({len(position_keys)}): {position_keys}")
                # Also show position details
                for key, payload in list(positions.items())[:5]:
                    symbol = payload.get('symbol', 'N/A') if isinstance(payload, dict) else 'N/A'
                    print(f"   - {key[:20]}... ({symbol})")
        except Exception as e:
            print(f"⚠️ Error loading position details: {e}")
        return False, f"max_concurrent_positions_reached_{open_count}_{max_concurrent}", False, 0.0
    
    # total exposure guard (tier-based) - uses combined balance for tier calculation
    max_total_exposure = config.get('MAX_TOTAL_EXPOSURE_USD', 100.0)
    current_exposure = _get_current_exposure_usd()
    if current_exposure + trade_amount_usd > max_total_exposure:
        return False, f"total_exposure_limit_exceeded_{current_exposure:.2f}_{max_total_exposure:.2f}", False, 0.0

    return True, "ok", False, trade_amount_usd

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