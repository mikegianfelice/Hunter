# strategy.py
import json
import os
import time
from pathlib import Path
from typing import Optional
from contextlib import contextmanager
import yaml
import requests
from src.config.config_loader import get_config, get_config_bool, get_config_float, get_config_int, get_config_values
from src.monitoring.structured_logger import log_info, log_warning, log_error
from src.storage.price_memory import load_price_memory, save_price_memory
from src.storage.delist import add_delisted_token, load_delisted_state


def _emit(level: str, event: str, message: str, **context):
    ctx = context or None
    if level == "info":
        log_info(event, message, context=ctx)
    elif level == "warning":
        log_warning(event, message, context=ctx)
    elif level == "error":
        log_error(event, message, context=ctx)


def _log_trace(message: str, level: Optional[str] = None, event: str = "strategy.trace", **context):
    resolved_level = level
    lowered = message.strip()
    if resolved_level is None:
        if lowered.startswith("❌") or lowered.startswith("🚨"):
            resolved_level = "error"
        elif lowered.startswith("⚠️"):
            resolved_level = "warning"
        else:
            resolved_level = "info"
    _emit(resolved_level, event, message, **context)

@contextmanager
def _atomic_write_json(path: Path):
    tmp_path = Path(str(path) + ".tmp")
    try:
        with open(tmp_path, "w") as f:
            yield f
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

def _now() -> int:
    return int(time.time())

def _load_price_mem() -> dict:
    mem = load_price_memory()
    return _prune_price_mem(mem or {})


def _save_price_mem(mem: dict):
    try:
        save_price_memory(mem or {})
    except Exception:
        pass

def _prune_price_mem(mem: dict) -> dict:
    now_ts = _now()
    config = get_config_values()
    pruned = {addr: info for addr, info in mem.items()
              if now_ts - int(info.get("ts", 0)) <= config['PRICE_MEM_PRUNE_SECS']}
    removed = len(mem) - len(pruned)
    if removed > 0:
        _save_price_mem(pruned)
        _emit(
            "info",
            "strategy.price_memory.prune",
            "Pruned old entries from price memory",
            removed=removed,
        )
    return pruned

def prune_price_memory() -> int:
    mem = load_price_memory()
    before = len(mem)
    pruned = _prune_price_mem(mem)
    return max(0, before - len(pruned))

def _pct_change(curr: float, prev: float) -> float:
    if prev <= 0:
        return 0.0
    return (curr - prev) / prev

def _calculate_momentum_score(token: dict, config: dict) -> tuple:
    """
    Calculate momentum score from available sources (reusable for entry and exit).
    Returns (momentum_score: float, source: str, momentum_data: dict) tuple.
    momentum_score: Normalized score (0-1) or raw momentum value
    source: "candle", "token_data", or None
    momentum_data: Dict with 5m/1h/24h values for tracking decay
    """
    momentum_data = {
        'momentum_5m': None,
        'momentum_1h': None,
        'momentum_24h': None,
    }
    
    # Try candle-based momentum first (most accurate)
    if token.get('candles_validated') and token.get('candles_15m'):
        candles = token['candles_15m']
        candle_momentum = token.get('candle_momentum')
        
        if candle_momentum is not None and len(candles) >= 4:
            # Extract individual timeframe momentum if available
            tech_indicators = token.get('technical_indicators', {})
            momentum_data['momentum_5m'] = token.get('priceChange5m')
            momentum_data['momentum_1h'] = token.get('priceChange1h')
            momentum_data['momentum_24h'] = token.get('priceChange24h')
            
            return candle_momentum, "candle", momentum_data
    
    # Fallback to token data momentum
    momentum_24h = token.get("momentum_24h") or token.get("priceChange24h")
    momentum_1h = token.get("momentum_1h") or token.get("priceChange1h")
    
    def to_decimal(mom_val):
        if mom_val is None or mom_val == "" or str(mom_val).lower() == "none" or str(mom_val).strip() == "":
            return None
        try:
            val = float(mom_val)
            return val / 100.0 if abs(val) > 1 else val
        except (ValueError, TypeError):
            return None
    
    mom_24h = to_decimal(momentum_24h)
    mom_1h = to_decimal(momentum_1h)
    mom_5m = to_decimal(token.get("priceChange5m"))
    
    momentum_data['momentum_5m'] = mom_5m
    momentum_data['momentum_1h'] = mom_1h
    momentum_data['momentum_24h'] = mom_24h
    
    if mom_1h is not None:
        return mom_1h, "token_data", momentum_data
    elif mom_24h is not None:
        return mom_24h, "token_data", momentum_data
    
    return None, None, momentum_data

def _add_to_delisted_tokens(address: str, symbol: str, reason: str):
    """
    Add a token to the delisted_tokens.json file with smart verification
    """
    try:
        # Import the smart verification function
        from smart_blacklist_cleaner import add_to_delisted_tokens_smart
        
        # Use smart verification to only add if actually delisted
        success = add_to_delisted_tokens_smart(address, symbol, reason)
        
        if not success:
            _emit(
                "info",
                "strategy.delisting.skip",
                "Token appears active; skipping delisted entry",
                symbol=symbol,
                address=address,
            )
            
    except Exception as e:
        _emit(
            "warning",
            "strategy.delisting.error",
            "Failed smart verification when adding token to delisted list",
            symbol=symbol,
            address=address,
            error=str(e),
        )
        # Fallback to storage-based method if smart verification fails
        try:
            added = add_delisted_token(address, symbol=symbol, reason=reason)
            if added:
                _emit(
                    "warning",
                    "strategy.delisting.added",
                    "Token added to delisted tokens",
                    symbol=symbol,
                    address=address,
                    reason=reason,
                )
            else:
                _emit(
                    "info",
                    "strategy.delisting.already_present",
                    "Token already present in delisted list",
                    symbol=symbol,
                    address=address,
                )
        except Exception as fallback_error:
            _emit(
                "error",
                "strategy.delisting.failure",
                "Failed to add token to delisted list using smart and fallback methods",
                symbol=symbol,
                address=address,
                error=str(fallback_error),
            )

def _check_token_delisted(token: dict) -> bool:
    """
    Pre-buy check to detect if a token is likely delisted or inactive.
    Returns True if token appears to be delisted/inactive.
    Automatically adds delisted tokens to delisted_tokens.json.
    """
    config = get_config_values()
    
    # Check if pre-buy delisting check is enabled
    if not config['ENABLE_PRE_BUY_DELISTING_CHECK']:
        _emit(
            "info",
            "strategy.delisting.disabled",
            "Pre-buy delisting check disabled; token allowed",
            symbol=token.get("symbol", "UNKNOWN"),
            address=token.get("address"),
        )
        return False
    
    address = token.get("address", "")
    chain_id = token.get("chainId", "ethereum").lower()
    symbol = token.get("symbol", "")
    
    if not address:
        return True  # No address = likely invalid
    
    # Check if token is already in our delisted tokens list
    try:
        state = load_delisted_state()
        delisted_tokens = state.get("delisted_tokens", [])
        if address.lower() in delisted_tokens:
            _emit(
                "warning",
                "strategy.prebuy.already_delisted",
                "Token found in delisted list during pre-buy check",
                symbol=symbol,
                address=address,
            )
            return True
    except Exception as e:
        _emit(
            "warning",
            "strategy.prebuy.delisted_read_error",
            "Failed to read delisted tokens state",
            error=str(e),
        )
        # Don't fail the check if we can't read the file
    
    # Check for Solana tokens (43-44 character addresses)
    if (len(address) in [43, 44]) and chain_id == "solana":
        return _check_solana_token_delisted(token)
    
    # For Ethereum tokens, try to get current price
    elif chain_id == "ethereum":
        return _check_ethereum_token_delisted(token)
    
    # For other chains, skip the check
    _emit(
        "info",
        "strategy.prebuy.chain_skipped",
        "Pre-buy delisting check skipped for chain",
        chain=chain_id,
        symbol=symbol,
    )
    return False

def _check_solana_token_delisted(token: dict) -> bool:
    """
    Enhanced Solana token delisting check with better fallback logic.
    More lenient when DexScreener shows good data and when APIs are down.
    
    IMPROVED: Now trusts DexScreener data first, only uses API as confirmation.
    Emergency bypass for tokens with excellent metrics.
    
    Returns True if token appears to be delisted/inactive.
    """
    config = get_config_values()
    address = token.get("address", "")
    symbol = token.get("symbol", "")
    volume_24h = float(token.get("volume24h", 0))
    liquidity = float(token.get("liquidity", 0))
    price_usd = float(token.get("priceUsd", 0))
    
    # EMERGENCY BYPASS: If token has excellent metrics from DexScreener, trust it completely
    # This prevents API failures from blocking obviously good tokens
    if volume_24h > 50000 and liquidity > 100000 and price_usd > 0:
        _emit(
            "info",
            "strategy.prebuy.solana.metrics_excellent",
            "Solana token bypass due to excellent DexScreener metrics",
            symbol=symbol,
            volume_24h=volume_24h,
            liquidity_usd=liquidity,
            price_usd=price_usd,
        )
        return False  # NOT delisted - trade it!
    
    # STRONG BYPASS: Very good metrics
    if volume_24h > 25000 and liquidity > 50000 and price_usd > 0:
        _emit(
            "info",
            "strategy.prebuy.solana.metrics_very_good",
            "Solana token bypass due to strong DexScreener metrics",
            symbol=symbol,
            volume_24h=volume_24h,
            liquidity_usd=liquidity,
            price_usd=price_usd,
        )
        return False  # NOT delisted
    
    # Adjust thresholds based on sensitivity setting
    if config['PRE_BUY_CHECK_SENSITIVITY'] == "lenient":
        vol_threshold = 25
        liq_threshold = 100
        poor_vol_threshold = 5
        poor_liq_threshold = 25
    elif config['PRE_BUY_CHECK_SENSITIVITY'] == "strict":
        vol_threshold = 1000
        liq_threshold = 5000
        poor_vol_threshold = 50
        poor_liq_threshold = 200
    else:  # moderate (default)
        vol_threshold = 500
        liq_threshold = 1000
        poor_vol_threshold = 10
        poor_liq_threshold = 50
    
    # If token has good volume and liquidity from DexScreener, trust it
    if volume_24h > vol_threshold and liquidity > liq_threshold:
        _emit(
            "info",
            "strategy.prebuy.solana.metrics_good",
            "Solana token trusted based on DexScreener metrics",
            symbol=symbol,
            volume_24h=volume_24h,
            liquidity_usd=liquidity,
        )
        return False
    elif volume_24h > vol_threshold/2 and liquidity > liq_threshold/2:
        _emit(
            "info",
            "strategy.prebuy.solana.metrics_moderate",
            "Solana token trusted with moderate DexScreener metrics",
            symbol=symbol,
            volume_24h=volume_24h,
            liquidity_usd=liquidity,
        )
        return False
    elif volume_24h > vol_threshold/4 and liquidity > liq_threshold/4:
        _emit(
            "info",
            "strategy.prebuy.solana.metrics_acceptable",
            "Solana token acceptable metrics; trusting DexScreener",
            symbol=symbol,
            volume_24h=volume_24h,
            liquidity_usd=liquidity,
        )
        return False
    
    # In lenient mode, trust DexScreener data more when APIs fail
    if config['PRE_BUY_CHECK_SENSITIVITY'] == "lenient":
        # If we have any reasonable volume/liquidity from DexScreener, trust it
        if volume_24h > 10 or liquidity > 50:
            _emit(
                "info",
                "strategy.prebuy.solana.lenient_allow",
                "Lenient mode trusting Solana token metrics",
                symbol=symbol,
                volume_24h=volume_24h,
                liquidity_usd=liquidity,
            )
            return False
    
    # CRITICAL: If we have valid price from DexScreener, trust it over API checks
    # This prevents false delisting when APIs are rate-limited or down
    if price_usd > 0.0000001:  # Very low but non-zero price threshold
        # If DexScreener shows valid price AND decent volume/liquidity, trust it completely
        if volume_24h > poor_vol_threshold or liquidity > poor_liq_threshold:
            _emit(
                "info",
                "strategy.prebuy.solana.price_trusted",
                "DexScreener price trusted for Solana token",
                symbol=symbol,
                price_usd=price_usd,
                volume_24h=volume_24h,
                liquidity_usd=liquidity,
            )
            return False
        else:
            _emit(
                "info",
                "strategy.prebuy.solana.price_valid_low_metrics",
                "DexScreener price valid despite low metrics",
                symbol=symbol,
                price_usd=price_usd,
                volume_24h=volume_24h,
                liquidity_usd=liquidity,
            )
            return False
    
    # Only do API price verification if DexScreener price is missing/zero
    # Try to get current price from multiple sources to verify if token is actually delisted
    # IMPORTANT: Only mark as delisted if we can CONFIRM it's delisted, not on API failures
    try:
        from src.execution.solana_executor import SimpleSolanaExecutor
        executor = SimpleSolanaExecutor()
        current_price = executor.get_token_price_usd(address)
        
        # Note: executor may return 0.000001 as a fallback, not actual zero
        if current_price > 0.00001:  # Higher threshold to account for fallback value
            _emit(
                "info",
                "strategy.prebuy.solana.price_verified",
                "Solana token price verified via executor",
                symbol=symbol,
                current_price=current_price,
            )
            return False
        elif current_price <= 0.00001:
            # Only mark as delisted if we have very poor metrics AND API also shows zero/fallback
            if volume_24h < poor_vol_threshold and liquidity < poor_liq_threshold:
                _emit(
                    "warning",
                    "strategy.prebuy.solana.price_zero_low_metrics",
                    "Zero price and poor metrics; marking Solana token as delisted",
                    symbol=symbol,
                    current_price=current_price,
                    volume_24h=volume_24h,
                    liquidity_usd=liquidity,
                )
                _add_to_delisted_tokens(address, symbol, f"Zero price and low metrics (vol: ${volume_24h:.0f}, liq: ${liquidity:.0f})")
                return True
            else:
                _emit(
                    "warning",
                    "strategy.prebuy.solana.price_zero_but_metrics_ok",
                    "Zero price returned but metrics decent; skipping without blacklist",
                    symbol=symbol,
                    current_price=current_price,
                    volume_24h=volume_24h,
                    liquidity_usd=liquidity,
                )
                return True
    except Exception as e:
        _emit(
            "warning",
            "strategy.prebuy.solana.price_verification_failed",
            "Solana price verification failed; allowing token",
            symbol=symbol,
            error=str(e),
        )
        # Graceful degradation: When APIs fail, trust DexScreener data
        # This prevents false rejections during API outages
        # CRITICAL: Never blacklist on API failures!
        if volume_24h > 5000 or liquidity > 10000:
            _emit(
                "info",
                "strategy.prebuy.solana.api_failed_but_metrics_good",
                "Solana API failure but metrics strong; allowing token",
                symbol=symbol,
                volume_24h=volume_24h,
                liquidity_usd=liquidity,
            )
            return False  # NOT delisted - API just failed
        elif volume_24h > 1000 or liquidity > 5000:
            _emit(
                "info",
                "strategy.prebuy.solana.api_failed_metrics_reasonable",
                "Solana API failure but metrics reasonable; allowing token",
                symbol=symbol,
                volume_24h=volume_24h,
                liquidity_usd=liquidity,
            )
            return False  # NOT delisted - API just failed
        else:
            _emit(
                "warning",
                "strategy.prebuy.solana.api_failed_metrics_poor",
                "Solana API failure and weak metrics; skipping without blacklist",
                symbol=symbol,
                volume_24h=volume_24h,
                liquidity_usd=liquidity,
            )
            return True  # Skip this cycle but don't blacklist
    
    # If we're unsure, be conservative but don't blacklist
    _emit(
        "warning",
        "strategy.prebuy.solana.uncertain_metrics",
        "Solana token metrics uncertain; skipping without blacklist",
        symbol=symbol,
        volume_24h=volume_24h,
        liquidity_usd=liquidity,
    )
    return True

def _check_ethereum_token_delisted(token: dict) -> bool:
    """
    Enhanced Ethereum token delisting check with better error handling.
    More lenient when APIs are down but DexScreener shows good data.
    
    IMPROVED: Now trusts DexScreener data first, emergency bypass for excellent metrics.
    """
    config = get_config_values()
    address = token.get("address", "")
    symbol = token.get("symbol", "")
    volume_24h = float(token.get("volume24h", 0))
    liquidity = float(token.get("liquidity", 0))
    price_usd = float(token.get("priceUsd", 0))
    
    # EMERGENCY BYPASS: If token has excellent metrics from DexScreener, trust it completely
    if volume_24h > 100000 and liquidity > 200000 and price_usd > 0:
        _emit(
            "info",
            "strategy.prebuy.evm.metrics_excellent",
            "Ethereum token bypass due to excellent metrics",
            symbol=symbol,
            volume_24h=volume_24h,
            liquidity_usd=liquidity,
            price_usd=price_usd,
        )
        return False  # NOT delisted
    
    # STRONG BYPASS: Very good metrics
    if volume_24h > 50000 and liquidity > 100000 and price_usd > 0:
        _emit(
            "info",
            "strategy.prebuy.evm.metrics_very_good",
            "Ethereum token bypass due to strong metrics",
            symbol=symbol,
            volume_24h=volume_24h,
            liquidity_usd=liquidity,
            price_usd=price_usd,
        )
        return False  # NOT delisted
    
    # Adjust thresholds based on sensitivity setting
    if config['PRE_BUY_CHECK_SENSITIVITY'] == "lenient":
        vol_threshold = 100
        liq_threshold = 500
        poor_vol_threshold = 10
        poor_liq_threshold = 50
    elif config['PRE_BUY_CHECK_SENSITIVITY'] == "strict":
        vol_threshold = 2000
        liq_threshold = 10000
        poor_vol_threshold = 100
        poor_liq_threshold = 500
    else:  # moderate (default)
        vol_threshold = 1000
        liq_threshold = 5000
        poor_vol_threshold = 50
        poor_liq_threshold = 200
    
    # If DexScreener shows good data, trust it even if price APIs fail
    if volume_24h > vol_threshold and liquidity > liq_threshold and price_usd > 0:
        _emit(
            "info",
            "strategy.prebuy.evm.metrics_good",
            "Ethereum token trusted with strong metrics",
            symbol=symbol,
            volume_24h=volume_24h,
            liquidity_usd=liquidity,
        )
        return False
    elif volume_24h > vol_threshold/2 and liquidity > liq_threshold/2 and price_usd > 0:
        _emit(
            "info",
            "strategy.prebuy.evm.metrics_moderate",
            "Ethereum token trusted with moderate metrics",
            symbol=symbol,
            volume_24h=volume_24h,
            liquidity_usd=liquidity,
        )
        return False
    elif volume_24h > vol_threshold/4 and liquidity > liq_threshold/4 and price_usd > 0:
        _emit(
            "info",
            "strategy.prebuy.evm.metrics_acceptable",
            "Ethereum token accepted with limited metrics",
            symbol=symbol,
            volume_24h=volume_24h,
            liquidity_usd=liquidity,
        )
        return False
    
    # In lenient mode, trust DexScreener data more when APIs fail
    if config['PRE_BUY_CHECK_SENSITIVITY'] == "lenient":
        # If we have any reasonable volume/liquidity from DexScreener, trust it
        if (volume_24h > 50 or liquidity > 200) and price_usd > 0:
            _emit(
                "info",
                "strategy.prebuy.evm.lenient_allow",
                "Lenient mode trusting Ethereum token metrics",
                symbol=symbol,
                volume_24h=volume_24h,
                liquidity_usd=liquidity,
            )
            return False
    
    # Try multiple price sources with fallbacks
    current_price = _get_ethereum_token_price_with_fallbacks(address, symbol)
    
    if current_price is None:
        # API FAILED - Never blacklist on API failures!
        _emit(
            "warning",
            "strategy.prebuy.evm.api_all_failed",
            "All Ethereum price sources failed; allowing token",
            symbol=symbol,
        )
        
        # Trust DexScreener data when APIs fail
        if volume_24h > 10000 or liquidity > 20000:
            _emit(
                "info",
                "strategy.prebuy.evm.api_failed_metrics_good",
                "Ethereum API failure but strong metrics; allowing token",
                symbol=symbol,
                volume_24h=volume_24h,
                liquidity_usd=liquidity,
            )
            return False  # NOT delisted - API just failed
        elif volume_24h > 5000 or liquidity > 10000:
            _emit(
                "info",
                "strategy.prebuy.evm.api_failed_metrics_reasonable",
                "Ethereum API failure but metrics reasonable; allowing token",
                symbol=symbol,
                volume_24h=volume_24h,
                liquidity_usd=liquidity,
            )
            return False  # NOT delisted - API just failed
        elif volume_24h > 1000 or liquidity > 5000:
            _emit(
                "info",
                "strategy.prebuy.evm.api_failed_metrics_moderate",
                "Ethereum API failure but acceptable metrics; allowing token",
                symbol=symbol,
                volume_24h=volume_24h,
                liquidity_usd=liquidity,
            )
            return False  # NOT delisted - API just failed
        elif volume_24h < poor_vol_threshold and liquidity < poor_liq_threshold:
            # Only skip (not blacklist) if metrics are very poor AND API failed
            _emit(
                "warning",
                "strategy.prebuy.evm.api_failed_metrics_poor",
                "Ethereum API failure and poor metrics; skipping without blacklist",
                symbol=symbol,
                volume_24h=volume_24h,
                liquidity_usd=liquidity,
            )
            return True  # Skip this cycle but don't blacklist
        else:
            _emit(
                "warning",
                "strategy.prebuy.evm.api_failed_generic",
                "Ethereum API failure; skipping without blacklist",
                symbol=symbol,
                volume_24h=volume_24h,
                liquidity_usd=liquidity,
            )
            return True  # Skip this cycle but don't blacklist
    
    if current_price == 0:
        # Only mark as delisted if we also have poor metrics
        if volume_24h < poor_vol_threshold and liquidity < poor_liq_threshold:
            _emit(
                "warning",
                "strategy.prebuy.evm.price_zero_poor_metrics",
                "Zero price and poor metrics; marking Ethereum token delisted",
                symbol=symbol,
                current_price=current_price,
                volume_24h=volume_24h,
                liquidity_usd=liquidity,
            )
            _add_to_delisted_tokens(address, symbol, "Zero price detected")
            return True
        else:
            _emit(
                "warning",
                "strategy.prebuy.evm.price_zero_but_metrics_ok",
                "Zero price but metrics acceptable; skipping without blacklist",
                symbol=symbol,
                current_price=current_price,
                volume_24h=volume_24h,
                liquidity_usd=liquidity,
            )
            return True
        
    # Check if price is extremely low (potential delisting)
    if current_price < 0.0000001:
        # Only mark as delisted if we also have poor metrics
        if volume_24h < poor_vol_threshold and liquidity < poor_liq_threshold:
            _emit(
                "warning",
                "strategy.prebuy.evm.price_low_poor_metrics",
                "Low price and poor metrics; marking Ethereum token delisted",
                symbol=symbol,
                current_price=current_price,
                volume_24h=volume_24h,
                liquidity_usd=liquidity,
            )
            _add_to_delisted_tokens(address, symbol, f"Low price: ${current_price}")
            return True
        else:
            _emit(
                "warning",
                "strategy.prebuy.evm.price_low_but_metrics_ok",
                "Low price but metrics acceptable; skipping without blacklist",
                symbol=symbol,
                current_price=current_price,
                volume_24h=volume_24h,
                liquidity_usd=liquidity,
            )
            return True
        
    _emit(
        "info",
        "strategy.prebuy.evm.price_verified",
        "Ethereum token price verified",
        symbol=symbol,
        current_price=current_price,
    )
    return False

def _get_ethereum_token_price_with_fallbacks(address: str, symbol: str) -> float:
    """
    Get Ethereum token price with multiple fallback mechanisms.
    Returns None if all sources fail.
    """
    # Primary: Uniswap Graph API
    try:
        from src.utils.utils import fetch_token_price_usd
        price = fetch_token_price_usd(address)
        if price and price > 0:
            return price
    except Exception as e:
        _emit(
            "warning",
            "strategy.price_fetch.primary_failed",
            "Primary price source failed",
            symbol=symbol,
            address=address,
            error=str(e),
        )
    
    # Fallback 1: Try DexScreener API
    try:
        url = f"https://api.dexscreener.com/latest/dex/tokens/{address}"
        response = requests.get(url, timeout=PRE_BUY_CHECK_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            pairs = data.get("pairs", [])
            if pairs:
                # Get the first pair with USDC or USDT
                for pair in pairs:
                    quote_token = pair.get("quoteToken", {}).get("symbol", "").upper()
                    if quote_token in ["USDC", "USDT"]:
                        price = float(pair.get("priceUsd", 0))
                        if price > 0:
                            _emit(
                                "info",
                                "strategy.price_fetch.dexscreener_primary_quote",
                                "DexScreener fallback price obtained from stable pair",
                                symbol=symbol,
                                address=address,
                                price_usd=price,
                            )
                            return price
                # If no USDC/USDT pair, use any pair with price
                for pair in pairs:
                    price = float(pair.get("priceUsd", 0))
                    if price > 0:
                        _emit(
                            "info",
                            "strategy.price_fetch.dexscreener_secondary_quote",
                            "DexScreener fallback price obtained from non-stable pair",
                            symbol=symbol,
                            address=address,
                            price_usd=price,
                        )
                        return price
    except Exception as e:
        _emit(
            "warning",
            "strategy.price_fetch.dexscreener_failed",
            "DexScreener fallback failed",
            symbol=symbol,
            address=address,
            error=str(e),
        )
    
    # Fallback 2: Try CoinGecko API (if we have the token ID)
    try:
        # This would require a mapping of addresses to CoinGecko IDs
        # For now, we'll skip this fallback
        pass
    except Exception as e:
        _emit(
            "warning",
            "strategy.price_fetch.coingecko_failed",
            "CoinGecko fallback failed",
            symbol=symbol,
            address=address,
            error=str(e),
        )
    
    _emit(
        "warning",
        "strategy.price_fetch.all_failed",
        "All price sources failed",
        symbol=symbol,
        address=address,
    )
    return None

def _get_token_liquidity(token_address: str) -> float:
    """Get token liquidity from DexScreener"""
    try:
        url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            pairs = data.get("pairs", [])
            if pairs:
                # Get the first pair with liquidity
                for pair in pairs:
                    liquidity = float(pair.get("liquidity", {}).get("usd", 0))
                    if liquidity > 0:
                        return liquidity
    except Exception as e:
        _emit(
            "warning",
            "strategy.liquidity_fetch.failed",
            "Failed to fetch token liquidity",
            token_address=token_address,
            error=str(e),
        )
    return 0.0

def _check_jupiter_tradeable(token_address: str, symbol: str) -> bool:
    """
    Pre-check if token is tradeable on Jupiter before doing expensive evaluations.
    Returns True if token is tradeable on Jupiter.
    """
    try:
        # Validate Solana address format (base58: 32–44 chars for 32-byte keys)
        addr_len = len(token_address)
        if addr_len < 32 or addr_len > 44:
            _emit(
                "error",
                "strategy.jupiter.invalid_address_length",
                "Jupiter pre-check failed due to invalid address length",
                symbol=symbol,
                token_address=token_address,
                length=addr_len,
            )
            return False
        
        # Try a small quote to see if Jupiter supports this token
        # NOTE: Jupiter API endpoint has changed to api.jup.ag  
        url = "https://api.jup.ag/v6/quote"
        params = {
            "inputMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
            "outputMint": token_address,
            "amount": "1000000",  # 1 USDC test amount
            "slippageBps": 100,  # 1% slippage
            "onlyDirectRoutes": "false",
            "asLegacyTransaction": "false"
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("data"):
                _emit(
                    "info",
                    "strategy.jupiter.tradeable",
                    "Jupiter pre-check indicates token is tradeable",
                    symbol=symbol,
                    token_address=token_address,
                )
                return True
            else:
                error_msg = data.get('error', 'Unknown error')
                _emit(
                    "error",
                    "strategy.jupiter.not_tradeable",
                    "Jupiter pre-check indicates token not tradeable",
                    symbol=symbol,
                    token_address=token_address,
                    error=error_msg,
                )
                return False
        elif response.status_code == 400:
            try:
                error_data = response.json()
                error_msg = error_data.get('error', 'Bad Request')
                if "not tradable" in error_msg.lower() or "not tradeable" in error_msg.lower():
                    _emit(
                        "error",
                        "strategy.jupiter.not_tradeable_400",
                        "Jupiter pre-check indicates token not tradeable (400)",
                        symbol=symbol,
                        token_address=token_address,
                        error=error_msg,
                    )
                    return False
                elif "cannot be parsed" in error_msg.lower() or "invalid" in error_msg.lower():
                    _emit(
                        "warning",
                        "strategy.jupiter.validation_issue",
                        "Jupiter pre-check had address validation issue; allowing token",
                        symbol=symbol,
                        token_address=token_address,
                        error=error_msg,
                    )
                    return True
                else:
                    _emit(
                        "warning",
                        "strategy.jupiter.generic_400",
                        "Jupiter pre-check 400 error; allowing token",
                        symbol=symbol,
                        token_address=token_address,
                        error=error_msg,
                    )
                    return True
            except:
                _emit(
                    "warning",
                    "strategy.jupiter.400_parse_failed",
                    "Jupiter pre-check 400 error could not be parsed; allowing token",
                    symbol=symbol,
                    token_address=token_address,
                )
                return True
        else:
            _emit(
                "warning",
                "strategy.jupiter.unexpected_status",
                "Jupiter pre-check unexpected HTTP status; allowing token",
                symbol=symbol,
                token_address=token_address,
                status=response.status_code,
            )
            return True
            
    except Exception as e:
        _emit(
            "warning",
            "strategy.jupiter.exception",
            "Jupiter pre-check failed due to exception; allowing token",
            symbol=symbol,
            token_address=token_address,
            error=str(e),
        )
        return True


# ============================================================================
# TWO-LANE ENTRY SYSTEM
# ============================================================================
# Early Scout: Lower thresholds + stricter safety + smaller size (catch early breakouts)
# Confirm/Add: Standard thresholds (current behavior for confirmed moves)

# Cooldown persistence file path
_COOLDOWN_FILE = Path("data/early_scout_cooldowns.json")
_COOLDOWNS_LOADED = False
_EARLY_SCOUT_COOLDOWNS: dict = {}

# Cooldown retention period (7 days in seconds)
_COOLDOWN_RETENTION_SECONDS = 7 * 24 * 60 * 60


def _prune_old_cooldowns():
    """Remove cooldown entries older than 7 days."""
    global _EARLY_SCOUT_COOLDOWNS
    now = time.time()
    cutoff = now - _COOLDOWN_RETENTION_SECONDS
    
    original_count = len(_EARLY_SCOUT_COOLDOWNS)
    _EARLY_SCOUT_COOLDOWNS = {
        k: v for k, v in _EARLY_SCOUT_COOLDOWNS.items()
        if v > cutoff
    }
    pruned_count = original_count - len(_EARLY_SCOUT_COOLDOWNS)
    
    if pruned_count > 0:
        _log_trace(
            f"🗑️ Pruned {pruned_count} cooldown entries older than 7 days",
            level="info",
            event="strategy.cooldown.pruned",
            pruned_count=pruned_count,
            remaining_count=len(_EARLY_SCOUT_COOLDOWNS)
        )
    
    return pruned_count


def _load_cooldowns_from_disk():
    """Load cooldowns from persistent storage on first access."""
    global _COOLDOWNS_LOADED, _EARLY_SCOUT_COOLDOWNS
    if _COOLDOWNS_LOADED:
        return
    
    try:
        if _COOLDOWN_FILE.exists():
            with open(_COOLDOWN_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    _EARLY_SCOUT_COOLDOWNS = data
                    _log_trace(
                        f"📁 Loaded {len(_EARLY_SCOUT_COOLDOWNS)} cooldown entries from disk",
                        level="info",
                        event="strategy.cooldown.loaded",
                        count=len(_EARLY_SCOUT_COOLDOWNS)
                    )
                    # Prune old entries on load
                    _prune_old_cooldowns()
    except Exception as e:
        _log_trace(
            f"⚠️ Failed to load cooldowns from disk: {e}",
            level="warning",
            event="strategy.cooldown.load_error",
            error=str(e)
        )
    _COOLDOWNS_LOADED = True


def _save_cooldowns_to_disk():
    """Persist cooldowns to disk with atomic write (prunes old entries first)."""
    global _EARLY_SCOUT_COOLDOWNS
    
    try:
        # Prune old entries before saving
        _prune_old_cooldowns()
        
        # Ensure data directory exists
        _COOLDOWN_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Atomic write: write to temp file then replace
        tmp_path = Path(str(_COOLDOWN_FILE) + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(_EARLY_SCOUT_COOLDOWNS, f, indent=2)
        os.replace(tmp_path, _COOLDOWN_FILE)
    except Exception as e:
        _log_trace(
            f"⚠️ Failed to save cooldowns to disk: {e}",
            level="warning",
            event="strategy.cooldown.save_error",
            error=str(e)
        )


def _get_lane_config(lane_name: str) -> dict:
    """Get configuration for a specific entry lane."""
    lane_config = get_config(f"trading.entry_lanes.{lane_name}", {})
    return lane_config if isinstance(lane_config, dict) else {}


def _is_early_scout_on_cooldown(address: str) -> bool:
    """Check if a token is on early_scout cooldown (prevents duplicate buys)."""
    _load_cooldowns_from_disk()  # Ensure loaded
    
    address_lower = address.lower()
    cooldown_key = f"{address_lower}:early_scout:buy"
    
    if cooldown_key not in _EARLY_SCOUT_COOLDOWNS:
        return False
    
    cooldown_minutes = get_config_float("trading.entry_lanes.early_scout.cooldown_minutes", 30.0)
    cooldown_secs = cooldown_minutes * 60
    last_buy_ts = _EARLY_SCOUT_COOLDOWNS[cooldown_key]
    
    return (time.time() - last_buy_ts) < cooldown_secs


def _record_early_scout_buy(address: str):
    """Record a successful early_scout buy to start cooldown (persisted)."""
    _load_cooldowns_from_disk()  # Ensure loaded
    
    address_lower = address.lower()
    cooldown_key = f"{address_lower}:early_scout:buy"
    _EARLY_SCOUT_COOLDOWNS[cooldown_key] = time.time()
    
    # Persist to disk
    _save_cooldowns_to_disk()


def _calculate_momentum_15m(candles: list) -> float:
    """
    Calculate 15-minute momentum (1 candle on 15m timeframe).
    Returns momentum as a decimal (e.g., 0.02 = 2%).
    """
    if not candles or len(candles) < 1:
        return 0.0
    
    last_candle = candles[-1]
    open_price = last_candle.get('open', 0) or last_candle.get('close', 0)
    close_price = last_candle.get('close', 0)
    
    if open_price <= 0:
        return 0.0
    
    return (close_price - open_price) / open_price


def _calculate_momentum_30m(candles: list) -> float:
    """
    Calculate 30-minute momentum (2 candles on 15m timeframe).
    Returns momentum as a decimal (e.g., 0.02 = 2%).
    """
    if not candles or len(candles) < 2:
        return 0.0
    
    recent_candles = candles[-2:]
    first_price = recent_candles[0].get('open', 0) or recent_candles[0].get('close', 0)
    last_price = recent_candles[-1].get('close', 0)
    
    if first_price <= 0:
        return 0.0
    
    return (last_price - first_price) / first_price


def _calculate_momentum_1h(candles: list) -> float:
    """
    Calculate 1-hour momentum (4 candles on 15m timeframe).
    Returns momentum as a decimal (e.g., 0.02 = 2%).
    """
    if not candles or len(candles) < 4:
        return 0.0
    
    recent_candles = candles[-4:]
    first_price = recent_candles[0].get('close', 0)
    last_price = recent_candles[-1].get('close', 0)
    
    if first_price <= 0:
        return 0.0
    
    return (last_price - first_price) / first_price


def _calculate_short_momentum(candles: list, num_candles: int = 4) -> float:
    """
    Calculate short-term momentum using the last N candles (default 4 = ~1 hour at 15m).
    Returns momentum as a decimal (e.g., 0.02 = 2%).
    DEPRECATED: Use _calculate_momentum_1h() instead.
    """
    return _calculate_momentum_1h(candles) if num_candles == 4 else _calculate_momentum_30m(candles) if num_candles == 2 else _calculate_momentum_15m(candles)


def _calculate_long_momentum(candles: list) -> float:
    """
    Calculate long-term momentum using all available candles (3-4h+).
    Returns momentum as a decimal (e.g., 0.02 = 2%).
    """
    if not candles or len(candles) < 4:
        return 0.0
    
    first_price = candles[0].get('close', 0)
    last_price = candles[-1].get('close', 0)
    
    if first_price <= 0:
        return 0.0
    
    return (last_price - first_price) / first_price


def _is_last_candle_green(candles: list) -> tuple:
    """
    Check if the last candle is green (close > open).
    
    Returns:
        (is_green: bool, mode: str)
        - mode: "open_close" if using open vs close
        - mode: "close_close_proxy" if using close[-1] vs close[-2]
        - mode: "unavailable" if cannot determine
    """
    if not candles:
        return (False, "unavailable")
    
    last_candle = candles[-1]
    open_price = last_candle.get('open', 0)
    close_price = last_candle.get('close', 0)
    
    # Prefer open vs close if available
    if open_price and open_price > 0:
        is_green = close_price > open_price
        return (is_green, "open_close")
    
    # Fallback: close-to-close comparison with previous candle
    if len(candles) >= 2:
        prev_close = candles[-2].get('close', 0)
        if prev_close > 0:
            is_green = close_price > prev_close
            return (is_green, "close_close_proxy")
    
    return (False, "unavailable")


def _check_candle_freshness(candles: list, max_stale_minutes: int = 20) -> tuple:
    """
    Check if candles are fresh (last candle within max_stale_minutes) and consecutive.
    
    Returns:
        (is_valid: bool, fail_reason: str or None, details: dict)
    """
    if not candles or len(candles) < 2:
        return (False, "insufficient_candles", {"candles_count": len(candles) if candles else 0})
    
    now = time.time()
    last_candle = candles[-1]
    
    # Check timestamp freshness
    last_ts = last_candle.get('timestamp') or last_candle.get('time') or last_candle.get('t')
    
    if last_ts:
        # Handle various timestamp formats
        try:
            if isinstance(last_ts, str):
                # ISO format or numeric string
                if 'T' in last_ts or '-' in last_ts:
                    from datetime import datetime
                    dt = datetime.fromisoformat(last_ts.replace('Z', '+00:00'))
                    last_ts_unix = dt.timestamp()
                else:
                    last_ts_unix = float(last_ts)
            elif isinstance(last_ts, (int, float)):
                last_ts_unix = float(last_ts)
                # Handle milliseconds
                if last_ts_unix > 1e12:
                    last_ts_unix = last_ts_unix / 1000
            else:
                last_ts_unix = None
            
            if last_ts_unix:
                age_seconds = now - last_ts_unix
                age_minutes = age_seconds / 60
                
                if age_minutes > max_stale_minutes:
                    return (False, "stale_candles", {
                        "last_candle_age_minutes": round(age_minutes, 1),
                        "max_stale_minutes": max_stale_minutes
                    })
        except Exception:
            # Timestamp parsing failed - continue without freshness check
            pass
    
    # Check consecutiveness of last 2 candles
    if len(candles) >= 2:
        last_candle = candles[-1]
        prev_candle = candles[-2]
        
        last_ts = last_candle.get('timestamp') or last_candle.get('time') or last_candle.get('t')
        prev_ts = prev_candle.get('timestamp') or prev_candle.get('time') or prev_candle.get('t')
        
        if last_ts and prev_ts:
            try:
                # Parse timestamps
                def parse_ts(ts):
                    if isinstance(ts, str):
                        if 'T' in ts or '-' in ts:
                            from datetime import datetime
                            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                            return dt.timestamp()
                        else:
                            val = float(ts)
                            return val / 1000 if val > 1e12 else val
                    elif isinstance(ts, (int, float)):
                        val = float(ts)
                        return val / 1000 if val > 1e12 else val
                    return None
                
                last_ts_unix = parse_ts(last_ts)
                prev_ts_unix = parse_ts(prev_ts)
                
                if last_ts_unix and prev_ts_unix:
                    gap_minutes = (last_ts_unix - prev_ts_unix) / 60
                    
                    # 15m candles should be ~15 min apart (allow 10-20 min for tolerance)
                    if gap_minutes < 10 or gap_minutes > 25:
                        return (False, "non_consecutive_candles", {
                            "gap_minutes": round(gap_minutes, 1),
                            "expected_gap": "15"
                        })
            except Exception:
                # Timestamp parsing failed - skip consecutiveness check
                pass
    
    return (True, None, {})


def _check_vwap_band(price: float, vwap_value: float, vwap_rule: str, band_bps_below: float = 80,
                     candles: list = None, require_green_if_below: bool = True,
                     mom_30m: float = None) -> tuple:
    """
    Check VWAP condition based on rule type with falling-knife protection.
    
    Args:
        price: Current price
        vwap_value: VWAP value
        vwap_rule: "required_above", "band", or "off"
        band_bps_below: Basis points below VWAP allowed for "band" mode
        candles: Optional candle data for green candle check
        require_green_if_below: If True and price < VWAP in band mode, require last candle green
        mom_30m: 30-minute momentum (required positive if price < VWAP in band mode)
    
    Returns:
        (passes: bool, reason: str, price_vs_vwap_pct: float)
    """
    if not vwap_value or vwap_value <= 0:
        return (False, "vwap_unavailable", 0.0)
    
    price_vs_vwap_pct = ((price - vwap_value) / vwap_value)
    
    if vwap_rule == "off":
        return (True, "vwap_off", price_vs_vwap_pct)
    
    if vwap_rule == "required_above":
        if price >= vwap_value:
            return (True, "above_vwap", price_vs_vwap_pct)
        else:
            return (False, "below_vwap", price_vs_vwap_pct)
    
    if vwap_rule == "band":
        # Allow price within band below VWAP
        band_pct = band_bps_below / 10000.0  # Convert bps to decimal (80 bps = 0.008)
        min_allowed_pct = -band_pct  # e.g., -0.008 = 0.8% below VWAP
        
        if price_vs_vwap_pct >= min_allowed_pct:
            # Within band - but if below VWAP, apply falling-knife protection
            if price < vwap_value:
                # FALLING-KNIFE PROTECTION: require positive momentum AND green candle
                if require_green_if_below and candles:
                    is_green, green_mode = _is_last_candle_green(candles)
                    if not is_green:
                        return (False, f"below_vwap_red_candle_{green_mode}", price_vs_vwap_pct)
                if mom_30m is not None and mom_30m <= 0:
                    return (False, "below_vwap_neg_momentum", price_vs_vwap_pct)
            return (True, "within_vwap_band", price_vs_vwap_pct)
        else:
            return (False, "below_vwap_band", price_vs_vwap_pct)
    
    # Default: require above VWAP
    return (price >= vwap_value, "default_above", price_vs_vwap_pct)


def _check_add_to_position_allowed(token: dict, confirm_cfg: dict) -> tuple:
    """
    Check if adding to an existing position is allowed based on PnL window and cap.
    
    Uses actual current exposure to enforce max_total_position_multiplier.
    
    Returns: (allowed: bool, add_multiplier: float, reason: str)
    """
    if not confirm_cfg.get("enable_adds_if_held", True):
        return (False, 0.0, "adds_disabled")
    
    # Check if token is already held
    try:
        from src.storage.positions import load_positions as load_positions_store
        from src.utils.position_sync import create_position_key
        
        address = (token.get("address") or "").lower()
        positions = load_positions_store()
        position_key = create_position_key(address)
        existing_position = positions.get(position_key)
        
        if not existing_position:
            # Not held - this is a new entry, not an add
            return (True, confirm_cfg.get("position_size_multiplier", 1.0), "new_position")
        
        # Check if position has unrealized PnL data
        entry_price = float(existing_position.get("entry_price", 0) or 0)
        current_price = float(token.get("priceUsd") or 0)
        
        if entry_price <= 0 or current_price <= 0:
            return (False, 0.0, "price_unavailable")
        
        unrealized_pnl_pct = (current_price - entry_price) / entry_price
        
        # Get PnL window from config
        pnl_window = confirm_cfg.get("add_only_if_unrealized_pnl_between", [-0.01, 0.06])
        min_pnl = pnl_window[0] if len(pnl_window) > 0 else -0.01
        max_pnl = pnl_window[1] if len(pnl_window) > 1 else 0.06
        
        if unrealized_pnl_pct < min_pnl:
            return (False, 0.0, f"unrealized_pnl_too_low_{unrealized_pnl_pct*100:.1f}%")
        if unrealized_pnl_pct > max_pnl:
            return (False, 0.0, f"unrealized_pnl_too_high_{unrealized_pnl_pct*100:.1f}%")
        
        # =====================================================================
        # CHECK ACTUAL CURRENT EXPOSURE VS CAP
        # =====================================================================
        add_mult = confirm_cfg.get("add_position_size_multiplier", 0.5)
        max_total_mult = confirm_cfg.get("max_total_position_multiplier", 1.25)
        
        # Get current position size and the base tier size for comparison
        current_size_usd = float(existing_position.get("position_size_usd", 0) or 0)
        
        # Get base tier size from config (or use a reference)
        # We need the original tier size that the multipliers are relative to
        base_size_usd = float(existing_position.get("original_tier_size_usd", 0) or 0)
        
        # Fallback: estimate from the trading config
        if base_size_usd <= 0:
            from src.utils.config_loader import get_config
            tier_config = get_config("trading.tiers.micro", {})
            base_size_usd = tier_config.get("max_size_usd", 100)
        
        if base_size_usd > 0:
            # Calculate current position multiplier based on actual exposure
            current_position_mult = current_size_usd / base_size_usd
            
            # Check if already at or over cap
            if current_position_mult >= max_total_mult:
                return (False, 0.0, f"already_at_cap({current_position_mult:.2f}x>={max_total_mult:.2f}x)")
            
            # Calculate remaining room
            remaining_mult = max_total_mult - current_position_mult
            
            # If remaining room is less than requested add, either:
            # - Reduce add to fit (partial add)
            # - Or reject if partial adds not supported (we'll reject for safety)
            if remaining_mult < add_mult:
                # Check if partial adds are allowed (default: no, for simplicity)
                allow_partial = confirm_cfg.get("allow_partial_adds", False)
                
                if allow_partial and remaining_mult > 0.1:  # At least 0.1x room
                    adjusted_mult = remaining_mult
                    return (True, adjusted_mult, f"partial_add({adjusted_mult:.2f}x,cap_room)")
                else:
                    return (False, 0.0, f"insufficient_cap_room({remaining_mult:.2f}x<{add_mult:.2f}x)")
            
            return (True, add_mult, f"add_allowed_pnl_{unrealized_pnl_pct*100:.1f}%_mult_{current_position_mult:.2f}x")
        else:
            # Cannot determine base size - allow add with warning
            _log_trace(
                f"⚠️ Cannot determine base tier size for cap check",
                level="warning",
                event="strategy.lane.add_cap_check_failed",
                address=address[:8],
                current_size_usd=current_size_usd
            )
            return (True, add_mult, f"add_allowed_pnl_{unrealized_pnl_pct*100:.1f}%_nocap")
        
    except Exception as e:
        _log_trace(
            f"⚠️ Error checking add-to-position: {e}",
            level="warning",
            event="strategy.lane.add_check_error",
            error=str(e)
        )
        return (False, 0.0, f"add_check_error_{str(e)[:20]}")


def evaluate_entry_lane(token: dict) -> dict:
    """
    Evaluate which entry lane (if any) a token qualifies for.
    
    Returns dict with:
        - lane: "early_scout", "confirm_add", or None
        - position_size_multiplier: float
        - reason: str (why lane was selected or rejected)
        - details: dict (metrics used for decision)
        - early_fail_reasons: list[str] (all failing gates for early_scout)
        - confirm_fail_reasons: list[str] (all failing gates for confirm_add)
    """
    address = (token.get("address") or "").lower()
    price = float(token.get("priceUsd") or 0.0)
    vol24h = float(token.get("volume24h") or 0.0)
    liq_usd = float(token.get("liquidity") or 0.0)
    is_trusted = bool(token.get("is_trusted", False))
    chain_id = token.get("chainId", "ethereum").lower()
    symbol = token.get("symbol", "UNKNOWN")
    
    # Get candles and technical indicators
    candles = token.get('candles_15m', [])
    tech_indicators = token.get('technical_indicators', {})
    rsi = token.get('rsi') or (tech_indicators.get('rsi') if isinstance(tech_indicators, dict) else None)
    vwap_dict = tech_indicators.get('vwap') or token.get('vwap')
    vwap_value = None
    if isinstance(vwap_dict, dict):
        vwap_value = vwap_dict.get('vwap')
    elif isinstance(vwap_dict, (int, float)):
        vwap_value = float(vwap_dict)
    
    # Get volume change data
    volume_change_1h = token.get("volumeChange1h")
    volume_change_15m = token.get("volumeChange15m")
    
    # Get accel_score if computed by trading engine
    accel_score = token.get("_accel_score")
    
    # Calculate momentum windows (15m, 30m, 1h, long)
    mom_15m = _calculate_momentum_15m(candles)
    mom_30m = _calculate_momentum_30m(candles)
    mom_1h = _calculate_momentum_1h(candles)
    long_momentum = _calculate_long_momentum(candles)
    
    # Check green candle with mode tracking
    last_candle_green, green_mode = _is_last_candle_green(candles)
    
    details = {
        "vol24h": vol24h,
        "liq_usd": liq_usd,
        "price": price,
        "rsi": rsi,
        "vwap": vwap_value,
        "mom_15m": round(mom_15m, 5) if mom_15m else 0,
        "mom_30m": round(mom_30m, 5) if mom_30m else 0,
        "mom_1h": round(mom_1h, 5) if mom_1h else 0,
        "long_momentum": round(long_momentum, 5) if long_momentum else 0,
        "last_candle_green": last_candle_green,
        "last_candle_green_mode": green_mode,
        "volume_change_1h": volume_change_1h,
        "volume_change_15m": volume_change_15m,
        "candles_count": len(candles),
        "accel_score": round(accel_score, 4) if accel_score else None,
    }
    
    # Collect fail reasons for each lane
    early_fail_reasons = []
    confirm_fail_reasons = []
    
    # =========================================================================
    # CONFIRM_ADD LANE EVALUATION
    # =========================================================================
    confirm_cfg = _get_lane_config("confirm_add")
    confirm_passed = False
    confirm_multiplier = 1.0
    
    if confirm_cfg.get("enabled", True):
        min_liq = confirm_cfg.get("min_liquidity_usd", 250000)
        min_vol = confirm_cfg.get("min_volume_24h_usd", 250000)
        min_momentum = confirm_cfg.get("min_momentum_pct_long", 0.015)
        rsi_max = confirm_cfg.get("rsi_max", 75)
        vwap_rule = confirm_cfg.get("vwap_rule", "required_above")
        
        # Check thresholds and collect all fail reasons
        if liq_usd < min_liq:
            confirm_fail_reasons.append(f"liq<{min_liq/1000:.0f}k")
        if vol24h < min_vol:
            confirm_fail_reasons.append(f"vol24h<{min_vol/1000:.0f}k")
        if rsi is None:
            confirm_fail_reasons.append("rsi_unavailable")
        elif rsi > rsi_max:
            confirm_fail_reasons.append(f"rsi>{rsi_max}")
        
        vwap_pass, vwap_reason, vwap_pct = _check_vwap_band(price, vwap_value, vwap_rule)
        if not vwap_pass:
            confirm_fail_reasons.append(f"vwap_{vwap_reason}")
        
        if long_momentum < min_momentum:
            confirm_fail_reasons.append(f"long_mom<{min_momentum*100:.1f}%")
        
        # If all checks pass
        if not confirm_fail_reasons:
            # Check add-to-position logic
            add_allowed, add_mult, add_reason = _check_add_to_position_allowed(token, confirm_cfg)
            if add_allowed:
                confirm_passed = True
                confirm_multiplier = add_mult
                details["add_reason"] = add_reason
            else:
                confirm_fail_reasons.append(add_reason)
    else:
        confirm_fail_reasons.append("disabled")
    
    if confirm_passed:
        _log_trace(
            f"✅ CONFIRM_ADD lane selected for {symbol}: liq=${liq_usd:,.0f}, vol=${vol24h:,.0f}, "
            f"long_mom={long_momentum*100:.2f}%, RSI={rsi:.1f if rsi else 0}, mult={confirm_multiplier:.2f}x",
            level="info",
            event="strategy.lane.confirm_add_selected",
            symbol=symbol,
            address=address[:8],
            **details
        )
        return {
            "lane": "confirm_add",
            "position_size_multiplier": confirm_multiplier,
            "reason": "passes_confirm_add",
            "details": details,
            "early_fail_reasons": early_fail_reasons,
            "confirm_fail_reasons": []
        }
    
    # =========================================================================
    # EARLY_SCOUT LANE EVALUATION
    # =========================================================================
    scout_cfg = _get_lane_config("early_scout")
    
    if scout_cfg.get("enabled", True):
        min_liq = scout_cfg.get("min_liquidity_usd", 80000)
        min_vol = scout_cfg.get("min_volume_24h_usd", 120000)
        min_mom_15m = scout_cfg.get("min_momentum_pct_15m", 0.002)
        min_mom_30m = scout_cfg.get("min_momentum_pct_30m", 0.006)
        min_mom_1h = scout_cfg.get("min_momentum_pct_1h", 0.008)
        require_mom_30m_positive = scout_cfg.get("require_mom_30m_positive", True)
        rsi_min = scout_cfg.get("rsi_min", 45)
        rsi_max = scout_cfg.get("rsi_max", 65)
        vwap_rule = scout_cfg.get("vwap_rule", "band")
        vwap_band_bps = scout_cfg.get("vwap_band_bps_below", 80)
        min_vol_change_15m = scout_cfg.get("min_volume_change_15m", 0.30)
        require_green_if_below_vwap = scout_cfg.get("require_last_candle_green_if_below_vwap", True)
        candle_max_stale_minutes = scout_cfg.get("candle_max_stale_minutes", 20)
        require_consecutive_candles = scout_cfg.get("require_consecutive_candles", True)
        
        # Check cooldown first
        if _is_early_scout_on_cooldown(address):
            early_fail_reasons.append("cooldown_active")
            _log_trace(
                f"⏳ EARLY_SCOUT cooldown active for {symbol}",
                level="info",
                event="strategy.lane.early_scout_cooldown",
                symbol=symbol,
                address=address[:8]
            )
        else:
            # Check candle freshness and consecutiveness (critical for early_scout)
            candles_valid, candle_fail_reason, candle_details = _check_candle_freshness(
                candles, max_stale_minutes=candle_max_stale_minutes
            )
            if candle_details:
                details.update(candle_details)
            details["candles_valid"] = candles_valid
            
            if not candles_valid:
                early_fail_reasons.append(candle_fail_reason)
            
            # Check liquidity/volume
            if liq_usd < min_liq:
                early_fail_reasons.append(f"liq<{min_liq/1000:.0f}k")
            if vol24h < min_vol:
                early_fail_reasons.append(f"vol24h<{min_vol/1000:.0f}k")
            
            # Check RSI band
            if rsi is None:
                early_fail_reasons.append("rsi_unavailable")
            elif rsi < rsi_min:
                early_fail_reasons.append(f"rsi<{rsi_min}")
            elif rsi > rsi_max:
                early_fail_reasons.append(f"rsi>{rsi_max}")
            
            # Check VWAP with falling-knife protection
            vwap_pass, vwap_reason, vwap_pct = _check_vwap_band(
                price, vwap_value, vwap_rule, vwap_band_bps,
                candles=candles,
                require_green_if_below=require_green_if_below_vwap,
                mom_30m=mom_30m if require_mom_30m_positive else None
            )
            if not vwap_pass:
                early_fail_reasons.append(f"vwap_{vwap_reason}")
            
            # Check momentum hierarchy (30m is primary, 15m/1h for confirmation)
            if require_mom_30m_positive and mom_30m <= 0:
                early_fail_reasons.append(f"mom_30m<=0({mom_30m*100:.2f}%)")
            elif mom_30m < min_mom_30m:
                early_fail_reasons.append(f"mom_30m<{min_mom_30m*100:.1f}%({mom_30m*100:.2f}%)")
            
            # Require at least one of: mom_1h meets threshold OR mom_15m > 0 (stability check)
            if mom_1h < min_mom_1h and mom_15m <= 0:
                early_fail_reasons.append(f"stability_fail(mom_1h<{min_mom_1h*100:.1f}%,mom_15m<=0)")
            
            # Check 15m volume acceleration (if available)
            if volume_change_15m is not None:
                try:
                    vol_change_pct = float(volume_change_15m) / 100.0 if abs(float(volume_change_15m)) > 1 else float(volume_change_15m)
                    if vol_change_pct < min_vol_change_15m:
                        early_fail_reasons.append(f"vol15m<{min_vol_change_15m*100:.0f}%({vol_change_pct*100:.1f}%)")
                except (ValueError, TypeError):
                    pass  # Graceful fallback
        
        # If all checks pass
        if not early_fail_reasons:
            _log_trace(
                f"🔍 EARLY_SCOUT lane selected for {symbol}: liq=${liq_usd:,.0f}, vol=${vol24h:,.0f}, "
                f"mom_30m={mom_30m*100:.2f}%, mom_1h={mom_1h*100:.2f}%, RSI={rsi:.1f if rsi else 0}, "
                f"last_green={last_candle_green}",
                level="info",
                event="strategy.lane.early_scout_selected",
                symbol=symbol,
                address=address[:8],
                **details
            )
            return {
                "lane": "early_scout",
                "position_size_multiplier": scout_cfg.get("position_size_multiplier", 0.25),
                "reason": "passes_early_scout",
                "details": details,
                "early_fail_reasons": [],
                "confirm_fail_reasons": confirm_fail_reasons
            }
    else:
        early_fail_reasons.append("disabled")
    
    # =========================================================================
    # NO LANE QUALIFIES - LOG STRUCTURED FAIL REASONS
    # =========================================================================
    early_fail_str = ",".join(early_fail_reasons) if early_fail_reasons else "disabled"
    confirm_fail_str = ",".join(confirm_fail_reasons) if confirm_fail_reasons else "disabled"
    
    _log_trace(
        f"❌ No entry lane for {symbol} | early_fail=\"{early_fail_str}\" confirm_fail=\"{confirm_fail_str}\"",
        level="info",
        event="strategy.lane.no_lane",
        symbol=symbol,
        address=address[:8] if address else "N/A",
        early_fail=early_fail_str,
        confirm_fail=confirm_fail_str,
        **details
    )
    
    return {
        "lane": None,
        "position_size_multiplier": 0.0,
        "reason": "no_lane_qualifies",
        "details": details,
        "early_fail_reasons": early_fail_reasons,
        "confirm_fail_reasons": confirm_fail_reasons
    }


def check_buy_signal(token: dict) -> bool:
    config = get_config_values()
    raw_address = (token.get("address") or "").strip()
    address = raw_address.lower()
    price   = float(token.get("priceUsd") or 0.0)
    vol24h  = float(token.get("volume24h") or 0.0)
    liq_usd = float(token.get("liquidity") or 0.0)
    is_trusted = bool(token.get("is_trusted", False))
    chain_id = token.get("chainId", "ethereum").lower()

    if not address or price <= config['MIN_PRICE_USD']:
        _log_trace(
            "📉 No address or price too low; skipping buy signal.",
            level="info",
            event="strategy.buy.skip_low_price",
            address=address,
            price=price,
        )
        return False

    # CRITICAL: Require validated candles before trade entry (required for technical checks)
    # Candles must be fetched and validated to ensure VWAP and other technical indicators are available
    if not token.get('candles_validated') or not token.get('candles_15m'):
        _log_trace(
            f"❌ Candles not validated - blocking trade entry (candles_validated={token.get('candles_validated')}, candles_15m={'present' if token.get('candles_15m') else 'missing'})",
            level="info",
            event="strategy.buy.candles_not_validated",
            symbol=token.get("symbol"),
            candles_validated=token.get('candles_validated'),
            has_candles_15m=bool(token.get('candles_15m')),
        )
        return False

    # Jupiter tradeability pre-check for Solana tokens (PRE-TRADE SAFETY CHECK)
    # This prevents attempting to buy tokens that cannot be traded on Jupiter
    if chain_id == "solana" and not is_trusted and config.get('ENABLE_JUPITER_PRE_CHECK', True):
        jupiter_result = _check_jupiter_tradeable(address, token.get("symbol", "UNKNOWN"))
        if not jupiter_result:
            if config.get('JUPITER_PRE_CHECK_STRICT', False):
                _log_trace(
                    "❌ Token not tradeable on Jupiter; skipping buy signal.",
                    level="error",
                    event="strategy.buy.jupiter_not_tradeable",
                    symbol=token.get("symbol", "UNKNOWN"),
                    address=address,
                )
                return False
            else:
                # Instead of rejecting, allow the token to proceed and let the actual swap attempt determine tradeability
                # Many new tokens are not immediately available on Jupiter but may be tradeable on other DEXs
                _log_trace(
                    "⚠️ Jupiter pre-check failed - allowing token to proceed (will attempt actual swap)",
                    level="warning",
                    event="strategy.buy.jupiter_failed_allow",
                    symbol=token.get("symbol", "UNKNOWN"),
                    address=address,
                )
                # Don't return False - let it continue to the actual swap attempt

    # Pre-buy delisting check (DISABLED - causes too many false positives)
    # The actual trade attempt will determine if token is tradeable
    # Other safety checks (cooldown, blacklist, risk manager) provide sufficient protection
    # if config.get('ENABLE_PRE_BUY_DELISTING_CHECK', False):
    #     if _check_token_delisted(token):
    #         print("🚨 Token appears delisted/inactive; skipping buy signal.")
    #         return False
    # else:
    _log_trace(
        "🔓 Pre-buy delisting check disabled - relying on trade execution and other safety checks",
        level="info",
        event="strategy.buy.delisting_disabled",
    )

    # WETH is handled specially in executor.py - skip here
    if address == "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2":  # WETH
        _log_trace(
            "🔓 WETH detected - will be handled in executor",
            level="info",
            event="strategy.buy.weth_special_case",
        )
        return True

    # Order-flow defense for Solana tokens (safety check before lane evaluation)
    # Helius API requires canonical base58 address (case-sensitive); do not pass lowercased address.
    if chain_id == "solana" and config.get('ENABLE_ORDER_FLOW_DEFENSE', True):
        from src.utils.order_flow_defense_solana import evaluate_order_flow_solana
        
        order_flow_result = evaluate_order_flow_solana(raw_address)
        
        if not order_flow_result.get("pass", False):
            reasons = order_flow_result.get("reasons", [])
            _log_trace(
                f"🚫 Order-flow defense blocked: {', '.join(reasons)}",
                level="warning",
                event="strategy.buy.order_flow_blocked",
                symbol=token.get("symbol"),
                address=address,
                metrics=order_flow_result.get("metrics", {})
            )
            return False

    # Update price memory
    mem = load_price_memory()
    now_ts = _now()
    mem[address] = {"price": price, "ts": now_ts}
    save_price_memory(mem)

    # =========================================================================
    # TWO-LANE ENTRY EVALUATION
    # =========================================================================
    # Evaluates token against both early_scout and confirm_add lanes.
    # Each lane has different thresholds for liquidity, volume, VWAP, RSI, momentum.
    # Returns lane info with position_size_multiplier for sizing.
    # =========================================================================
    
    lane_result = evaluate_entry_lane(token)
    selected_lane = lane_result.get("lane")
    position_multiplier = lane_result.get("position_size_multiplier", 1.0)
    lane_reason = lane_result.get("reason", "unknown")
    lane_details = lane_result.get("details", {})
    
    if selected_lane is None:
        # No lane qualifies - log and reject
        _log_trace(
            f"❌ No entry lane qualifies for {token.get('symbol', 'UNKNOWN')}: {lane_reason}",
            level="info",
            event="strategy.buy.no_lane",
            symbol=token.get("symbol"),
            reason=lane_reason,
            **lane_details
        )
        return False
    
    # Store lane information in token dict for position sizing and tracking
    token['entry_lane'] = selected_lane
    token['entry_lane_multiplier'] = position_multiplier
    token['entry_lane_details'] = lane_details
    
    # Log successful lane selection
    _log_trace(
        f"✅ Entry lane '{selected_lane}' selected for {token.get('symbol', 'UNKNOWN')} "
        f"(size_mult={position_multiplier:.2f}x, liq=${lane_details.get('liq_usd', 0):,.0f}, "
        f"short_mom={lane_details.get('short_momentum', 0)*100:.2f}%, long_mom={lane_details.get('long_momentum', 0)*100:.2f}%)",
        level="info",
        event="strategy.buy.lane_selected",
        symbol=token.get("symbol"),
        lane=selected_lane,
        position_multiplier=position_multiplier,
        **lane_details
    )
    
    # Record early_scout buy for cooldown tracking
    if selected_lane == "early_scout":
        _record_early_scout_buy(address)
    
    return True

def get_dynamic_take_profit(token: dict) -> float:
    config = get_config_values()
    tp = config['BASE_TP']
    vol24h = float(token.get("volume24h") or 0.0)
    sent_score = float(token.get("sent_score") or 0.0)
    mentions   = int(token.get("sent_mentions") or 0)

    if sent_score >= 75 or mentions >= 10:
        tp += 0.15
    elif sent_score <= 50 and mentions < 3:
        tp -= 0.10

    if vol24h >= 200_000:
        tp += 0.10
    elif vol24h < 20_000:
        tp -= 0.10

    tp = max(config['TP_MIN'], min(config['TP_MAX'], tp))
    _log_trace(
        f"🎯 Dynamic TP computed: {tp*100:.0f}% (base {config['BASE_TP']*100:.0f}%)",
        level="info",
        event="strategy.take_profit.dynamic",
        symbol=token.get("symbol"),
        base_tp=config['BASE_TP'],
        computed_tp=tp,
    )
    return tp