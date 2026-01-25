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
    source: "candle", "external", "token_data", or None
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
    
    # Try external momentum (DexScreener)
    if config.get('ENABLE_EXTERNAL_MOMENTUM', True):
        ext_momentum, ext_source = _get_external_momentum(token, config)
        if ext_momentum is not None:
            # Extract individual timeframes
            def to_decimal(pct_val):
                if pct_val is None or pct_val == "" or str(pct_val).lower() == "none" or str(pct_val).strip() == "":
                    return None
                try:
                    val = float(pct_val)
                    return val / 100.0 if abs(val) > 1 else val
                except (ValueError, TypeError):
                    return None
            
            momentum_data['momentum_5m'] = to_decimal(token.get('priceChange5m'))
            momentum_data['momentum_1h'] = to_decimal(token.get('priceChange1h'))
            momentum_data['momentum_24h'] = to_decimal(token.get('priceChange24h'))
            
            return ext_momentum, "external", momentum_data
    
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

def _get_external_momentum(token: dict, config: dict):
    """
    Get momentum from external historical data (DexScreener price changes).
    Returns (momentum_value, source_description) tuple.
    Returns (None, None) if no external data available.
    """
    if not config.get('ENABLE_EXTERNAL_MOMENTUM', True):
        return None, None
    
    # Get price change data from token dict (DexScreener provides these as percentages)
    # Note: DexScreener returns priceChange as percentages (e.g., 5.5 means 5.5%)
    price_change_5m = token.get("priceChange5m")
    price_change_1h = token.get("priceChange1h")
    price_change_24h = token.get("priceChange24h")
    
    # Convert from percentage to decimal (5.5% -> 0.055)
    def to_decimal(pct_val):
        if pct_val is None or pct_val == "" or str(pct_val).lower() == "none" or str(pct_val).strip() == "":
            return None
        try:
            return float(pct_val) / 100.0
        except (ValueError, TypeError):
            return None
    
    pc_5m = to_decimal(price_change_5m)
    pc_1h = to_decimal(price_change_1h)
    pc_24h = to_decimal(price_change_24h)
    
    primary_timeframe = config.get('EXTERNAL_MOMENTUM_PRIMARY_TIMEFRAME', 'h1')
    use_multi = config.get('USE_MULTI_TIMEFRAME_MOMENTUM', True)
    require_alignment = config.get('REQUIRE_MOMENTUM_ALIGNMENT', True)
    require_24h_positive = config.get('REQUIRE_POSITIVE_24H_MOMENTUM', True)
    
    # Check momentum alignment requirement (5m and 1h both positive)
    if require_alignment and pc_5m is not None and pc_1h is not None:
        if pc_5m <= 0 or pc_1h <= 0:
            # Timeframes not aligned - reject
            return None, None
    
    # Tiered 24h momentum requirement with override logic
    min_24h_momentum = config.get('MIN_24H_MOMENTUM_PCT', 0.03)  # 3% default
    min_1h_momentum = config.get('MIN_1H_MOMENTUM_PCT', 0.015)  # 1.5% default
    allow_override = config.get('ALLOW_NEGATIVE_24H_OVERRIDE', True)
    min_1h_for_override = config.get('MIN_1H_MOMENTUM_FOR_OVERRIDE', 0.04)  # 4% default
    min_5m_for_override = config.get('MIN_5M_MOMENTUM_FOR_OVERRIDE', 0.05)  # 5% default
    
    # Check 1h momentum minimum
    if pc_1h is not None:
        if pc_1h < min_1h_momentum:
            # 1h momentum below minimum - reject
            return None, None
    
    # Check 24h momentum with tiered logic
    if require_24h_positive and pc_24h is not None:
        if pc_24h < 0:
            # 24h momentum negative - check for override
            if allow_override and pc_1h is not None and pc_5m is not None:
                # Allow override if recent momentum is very strong
                if pc_1h >= min_1h_for_override and pc_5m >= min_5m_for_override:
                    # Strong recent momentum overrides negative 24h
                    pass  # Continue with momentum calculation
                else:
                    # Not strong enough for override - reject
                    return None, None
            else:
                # No override allowed or missing data - reject
                return None, None
        elif pc_24h < min_24h_momentum:
            # 24h momentum positive but below minimum - check for override
            if allow_override and pc_1h is not None and pc_5m is not None:
                # Allow override if recent momentum is very strong
                if pc_1h >= min_1h_for_override and pc_5m >= min_5m_for_override:
                    # Strong recent momentum overrides low 24h
                    pass  # Continue with momentum calculation
                else:
                    # Not strong enough for override - reject
                    return None, None
            else:
                # No override allowed or missing data - reject
                return None, None
    
    if use_multi and (pc_5m is not None or pc_1h is not None or pc_24h is not None):
        # Multi-timeframe weighted average
        weights = {
            'm5': config.get('EXTERNAL_MOMENTUM_M5_WEIGHT', 0.3),
            'h1': config.get('EXTERNAL_MOMENTUM_H1_WEIGHT', 0.5),
            'h24': config.get('EXTERNAL_MOMENTUM_H24_WEIGHT', 0.2)
        }
        
        total_weight = 0
        weighted_sum = 0
        sources = []
        
        if pc_5m is not None:
            weighted_sum += pc_5m * weights['m5']
            total_weight += weights['m5']
            sources.append(f"5m:{pc_5m*100:.2f}%")
        
        if pc_1h is not None:
            weighted_sum += pc_1h * weights['h1']
            total_weight += weights['h1']
            sources.append(f"1h:{pc_1h*100:.2f}%")
        
        if pc_24h is not None:
            weighted_sum += pc_24h * weights['h24']
            total_weight += weights['h24']
            sources.append(f"24h:{pc_24h*100:.2f}%")
        
        if total_weight > 0:
            momentum = weighted_sum / total_weight  # Normalize by actual available weights
            source_desc = f"external (weighted: {', '.join(sources)})"
            return momentum, source_desc
    
    # Single timeframe fallback
    if primary_timeframe == 'h1' and pc_1h is not None:
        return pc_1h, "external (1h)"
    elif primary_timeframe == 'm5' and pc_5m is not None:
        return pc_5m, "external (5m)"
    elif primary_timeframe == 'h24' and pc_24h is not None:
        return pc_24h, "external (24h)"
    
    # Try fallback timeframes
    fallback_timeframe = config.get('EXTERNAL_MOMENTUM_FALLBACK_TIMEFRAME', 'm5')
    if fallback_timeframe == 'm5' and pc_5m is not None:
        return pc_5m, "external (5m fallback)"
    elif fallback_timeframe == 'h1' and pc_1h is not None:
        return pc_1h, "external (1h fallback)"
    elif fallback_timeframe == 'h24' and pc_24h is not None:
        return pc_24h, "external (24h fallback)"
    
    # Try any available
    if pc_1h is not None:
        return pc_1h, "external (1h any)"
    if pc_5m is not None:
        return pc_5m, "external (5m any)"
    if pc_24h is not None:
        return pc_24h, "external (24h any)"
    
    return None, None

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
        # Validate Solana address format
        if len(token_address) != 44:
            _emit(
                "error",
                "strategy.jupiter.invalid_address_length",
                "Jupiter pre-check failed due to invalid address length",
                symbol=symbol,
                token_address=token_address,
                length=len(token_address),
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

def check_buy_signal(token: dict) -> bool:
    config = get_config_values()
    address = (token.get("address") or "").lower()
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

    # For trusted tokens, require milder depth floors
    min_vol = config['MIN_VOL_24H_BUY'] if not is_trusted else max(2000.0, config['MIN_VOL_24H_BUY'] * 0.5)
    min_liq = config['MIN_LIQ_USD_BUY'] if not is_trusted else max(2000.0, config['MIN_LIQ_USD_BUY'] * 0.5)
    
    # For multi-chain tokens, use lower requirements but not too aggressive
    if chain_id != "ethereum":
        # Previously: 20% / 30% of ETH thresholds (too lenient)
        # Now: require 75% of ETH thresholds with sensible floors
        min_vol = max(50_000.0, min_vol * 0.75)
        min_liq = max(75_000.0, min_liq * 0.75)

    if vol24h < min_vol or liq_usd < min_liq:
        _log_trace(
            f"🪫 Fails market depth: vol ${vol24h:,.0f} (need ≥ {min_vol:,.0f}), liq ${liq_usd:,.0f} (need ≥ {min_liq:,.0f})",
            level="info",
            event="strategy.buy.market_depth_fail",
            volume_24h=vol24h,
            required_volume=min_vol,
            liquidity_usd=liq_usd,
            required_liquidity=min_liq,
            symbol=token.get("symbol"),
        )
        return False

    # Order-flow defense for Solana tokens (after liquidity/volume, before momentum)
    if chain_id == "solana" and config.get('ENABLE_ORDER_FLOW_DEFENSE', True):
        from src.utils.order_flow_defense_solana import evaluate_order_flow_solana
        
        order_flow_result = evaluate_order_flow_solana(address)
        
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

    mem = load_price_memory()
    entry = mem.get(address)
    now_ts = _now()
    mem[address] = {"price": price, "ts": now_ts}
    save_price_memory(mem)

    # Trusted tokens: slightly easier momentum threshold
    momentum_need = config['MIN_MOMENTUM_PCT'] if not is_trusted else max(0.003, config['MIN_MOMENTUM_PCT'] * 0.5)  # e.g. 0.3%
    
    # Multi-chain tokens: require real momentum (increased threshold for quality)
    if chain_id != "ethereum":
        # Previously: momentum_need * 0.5 (≈0.15% threshold), too loose.
        # Now: require at least 0.2-0.225% momentum for better entry quality.
        multichain_momentum_multiplier = config.get('multichain_momentum_multiplier', 0.75)  # 75% of ETH threshold
        multichain_momentum_min = config.get('multichain_momentum_min', 0.002)  # Minimum 0.2%
        momentum_need = max(multichain_momentum_min, momentum_need * multichain_momentum_multiplier)
        _log_trace(
            f"🔓 Multi-chain momentum threshold: {momentum_need*100:.4f}%",
            level="info",
            event="strategy.buy.multichain_momentum_threshold",
            symbol=token.get("symbol"),
            chain=chain_id,
            momentum_threshold=momentum_need,
        )

    # WETH is handled specially in executor.py - skip here
    if address == "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2":  # WETH
        _log_trace(
            "🔓 WETH detected - will be handled in executor",
            level="info",
            event="strategy.buy.weth_special_case",
        )
        return True

    # RSI filter - MANDATORY: avoid overbought entries (required for all trades)
    rsi_threshold = config.get('RSI_OVERBOUGHT_THRESHOLD', 70)
    # Try to get RSI from token dict (if available from technical indicators)
    rsi = token.get("rsi")
    if rsi is None:
        # Try to get from technical_indicators nested dict
        tech_indicators = token.get("technical_indicators", {})
        if isinstance(tech_indicators, dict):
            rsi = tech_indicators.get("rsi")
    
    # MANDATORY: Block if RSI unavailable (should be calculated from validated candles)
    if rsi is None:
        _log_trace(
            f"❌ RSI filter blocked: RSI unavailable (required but not calculated from candles)",
            level="info",
            event="strategy.buy.rsi_unavailable_blocked",
            symbol=token.get("symbol"),
        )
        return False
    
    # Block if overbought
    if rsi > rsi_threshold:
        _log_trace(
            f"❌ Token overbought (RSI: {rsi:.1f} > {rsi_threshold})",
            level="info",
            event="strategy.buy.rsi_overbought",
            symbol=token.get("symbol"),
            rsi=rsi,
            threshold=rsi_threshold,
        )
        return False

    # Volume momentum check - MANDATORY: require increasing volume (required for all trades)
    min_volume_change = config.get('MIN_VOLUME_CHANGE_1H', 0.1)
    # Try to get volume change data (from DexScreener)
    volume_change_1h = token.get("volumeChange1h")
    
    # MANDATORY: Block if volume change data unavailable
    if volume_change_1h is None:
        _log_trace(
            f"❌ Volume momentum check blocked: volume change data unavailable (required but not available from DexScreener)",
            level="info",
            event="strategy.buy.volume_momentum_unavailable_blocked",
            symbol=token.get("symbol"),
        )
        return False
    
    # Parse and validate volume change
    try:
        volume_change_pct = float(volume_change_1h) / 100.0 if float(volume_change_1h) > 1 else float(volume_change_1h)
        if volume_change_pct < min_volume_change:
            _log_trace(
                f"❌ Volume not increasing (1h change: {volume_change_pct*100:.1f}% < {min_volume_change*100:.1f}%)",
                level="info",
                event="strategy.buy.volume_momentum_fail",
                symbol=token.get("symbol"),
                volume_change=volume_change_pct,
                required_change=min_volume_change,
            )
            return False
    except (ValueError, TypeError) as e:
        # MANDATORY: Block if volume change data is unparseable
        _log_trace(
            f"❌ Volume momentum check blocked: volume change data unparseable (required): {e}",
            level="info",
            event="strategy.buy.volume_momentum_parse_error_blocked",
            symbol=token.get("symbol"),
            error=str(e),
        )
        return False

    # VWAP Entry Filter (UPGRADE #2: Strength filter)
    if config.get('ENABLE_VWAP_ENTRY_FILTER', True):
        vwap_entry_required = config.get('VWAP_ENTRY_REQUIRED', True)
        max_vwap_extension_pct = config.get('MAX_VWAP_EXTENSION_PCT', 0.05)
        vwap_pullback_tolerance_pct = config.get('VWAP_PULLBACK_TOLERANCE_PCT', 0.02)
        
        # Try to get VWAP from technical indicators or token dict
        tech_indicators = token.get('technical_indicators', {})
        vwap_dict = tech_indicators.get('vwap') or token.get('vwap')
        vwap_value = None
        
        # Handle VWAP dict format (from TechnicalIndicators.calculate_vwap)
        if isinstance(vwap_dict, dict):
            vwap_value = vwap_dict.get('vwap')
        elif isinstance(vwap_dict, (int, float)):
            vwap_value = float(vwap_dict)
        
        if vwap_value and vwap_value > 0:
            # VWAP available - apply filter
            price_vs_vwap_pct = ((price - vwap_value) / vwap_value) if vwap_value > 0 else 0.0
            
            if vwap_entry_required and price < vwap_value:
                # Hard block: price below VWAP
                _log_trace(
                    f"❌ VWAP entry filter blocked: price ${price:.6f} < VWAP ${vwap_value:.6f} ({price_vs_vwap_pct*100:.2f}% below)",
                    level="info",
                    event="strategy.buy.vwap_below_blocked",
                    symbol=token.get("symbol"),
                    price=price,
                    vwap=vwap_value,
                    price_vs_vwap_pct=price_vs_vwap_pct,
                )
                return False
            
            # Check for extended price (too far above VWAP)
            if price_vs_vwap_pct > max_vwap_extension_pct:
                # Price extended above VWAP - require extra momentum confirmation
                _log_trace(
                    f"⚠️ Price extended above VWAP: {price_vs_vwap_pct*100:.2f}% (threshold: {max_vwap_extension_pct*100:.2f}%) - requiring extra momentum confirmation",
                    level="warning",
                    event="strategy.buy.vwap_extended",
                    symbol=token.get("symbol"),
                    price_vs_vwap_pct=price_vs_vwap_pct,
                    max_extension=max_vwap_extension_pct,
                )
                # Continue but momentum threshold will be checked below
            elif abs(price_vs_vwap_pct) <= vwap_pullback_tolerance_pct:
                # Price near VWAP (pullback) - preferred entry zone
                _log_trace(
                    f"✅ VWAP pullback entry: price ${price:.6f} near VWAP ${vwap_value:.6f} ({abs(price_vs_vwap_pct)*100:.2f}% distance)",
                    level="info",
                    event="strategy.buy.vwap_pullback",
                    symbol=token.get("symbol"),
                    price=price,
                    vwap=vwap_value,
                    distance_pct=abs(price_vs_vwap_pct),
                )
        else:
            # VWAP unavailable - block trade if VWAP entry is required
            if vwap_entry_required:
                _log_trace(
                    f"❌ VWAP entry filter blocked: VWAP unavailable (required but not calculated from candles)",
                    level="info",
                    event="strategy.buy.vwap_unavailable_blocked",
                    symbol=token.get("symbol"),
                )
                return False
            else:
                # VWAP not required - log warning but allow trade
                _log_trace(
                    "⚠️ VWAP unavailable - proceeding without VWAP filter (not required)",
                    level="warning",
                    event="strategy.buy.vwap_unavailable",
                    symbol=token.get("symbol"),
                )

    # Use candle-based momentum if available (most accurate - computed from validated 15m candles)
    # This takes priority over external momentum since it's computed from real on-chain data
    if token.get('candles_validated') and token.get('candles_15m'):
        candles = token['candles_15m']
        candle_momentum = token.get('candle_momentum')
        
        if candle_momentum is not None and len(candles) >= 4:  # Need at least 1 hour of data
            # Get VWAP from technical indicators if available
            tech_indicators = token.get('technical_indicators', {})
            vwap = tech_indicators.get('vwap') or token.get('vwap')
            
            # Format VWAP safely to avoid format string errors
            vwap_str = f"{vwap:.8f}" if isinstance(vwap, (int, float)) and vwap is not None else "N/A"
            
            _log_trace(
                f"📈 Candle-based momentum: {candle_momentum*100:.4f}% (need ≥ {momentum_need*100:.4f}%), VWAP={vwap_str}",
                level="info",
                event="strategy.buy.candle_momentum",
                symbol=token.get("symbol"),
                momentum=candle_momentum,
                vwap=vwap,
                required_momentum=momentum_need,
                candles_count=len(candles),
            )
            
            # Use candle momentum if it meets threshold
            if candle_momentum >= momentum_need:
                _log_trace(
                    "✅ Candle-based momentum buy signal → TRUE",
                    level="info",
                    event="strategy.buy.candle_momentum_pass",
                    symbol=token.get("symbol"),
                    momentum=candle_momentum,
                    vwap=vwap,
                )
                return True
            else:
                _log_trace(
                    f"❌ Candle-based momentum insufficient ({candle_momentum*100:.4f}% < {momentum_need*100:.4f}%).",
                    level="info",
                    event="strategy.buy.candle_momentum_fail",
                    symbol=token.get("symbol"),
                    momentum=candle_momentum,
                    required_momentum=momentum_need,
                )
                return False
        else:
            _log_trace(
                f"⚠️ Candle momentum unavailable (momentum={candle_momentum}, candles={len(candles) if candles else 0}), falling back to external momentum",
                level="warning",
                event="strategy.buy.candle_momentum_unavailable",
                symbol=token.get("symbol"),
                candle_momentum=candle_momentum,
                candles_count=len(candles) if candles else 0,
            )
            # Fall through to external momentum below
    
    # Try external momentum (DexScreener price change data)
    # Only check external momentum if it's enabled
    if config.get('ENABLE_EXTERNAL_MOMENTUM', True):
        ext_momentum, ext_source = _get_external_momentum(token, config)
        if ext_momentum is not None:
            # Check momentum acceleration (5m momentum must exceed 1h by threshold)
            min_acceleration = config.get('MIN_MOMENTUM_ACCELERATION', 0.002)
            price_change_5m = token.get("priceChange5m")
            price_change_1h = token.get("priceChange1h")
            
            # Convert to decimal if needed
            def to_decimal(pct_val):
                if pct_val is None or pct_val == "" or str(pct_val).lower() == "none" or str(pct_val).strip() == "":
                    return None
                try:
                    val = float(pct_val)
                    return val / 100.0 if val > 1 else val
                except (ValueError, TypeError):
                    return None
            
            pc_5m = to_decimal(price_change_5m)
            pc_1h = to_decimal(price_change_1h)
            
            # Check momentum acceleration
            if pc_5m is not None and pc_1h is not None:
                momentum_acceleration = pc_5m - pc_1h
                if momentum_acceleration < min_acceleration:
                    _log_trace(
                        f"❌ Momentum not accelerating (5m-1h: {momentum_acceleration*100:.2f}% < {min_acceleration*100:.2f}%)",
                        level="info",
                        event="strategy.buy.momentum_acceleration_fail",
                        symbol=token.get("symbol"),
                        acceleration=momentum_acceleration,
                        required_acceleration=min_acceleration,
                    )
                    return False
            
            # NEW: Velocity check - require 5m momentum ≥ 2.5-3% (fast moves only)
            min_5m_velocity = config.get('MIN_MOMENTUM_5M_VELOCITY', 0.025)  # 2.5% minimum
            if pc_5m is not None:
                if pc_5m < min_5m_velocity:
                    _log_trace(
                        f"❌ 5m momentum too weak for velocity check: {pc_5m*100:.2f}% < {min_5m_velocity*100:.2f}% (slow move, likely noise)",
                        level="info",
                        event="strategy.buy.momentum_velocity_fail",
                        symbol=token.get("symbol"),
                        momentum_5m=pc_5m,
                        required_velocity=min_5m_velocity,
                    )
                    return False
                else:
                    _log_trace(
                        f"✅ 5m momentum velocity check passed: {pc_5m*100:.2f}% ≥ {min_5m_velocity*100:.2f}% (fast move)",
                        level="info",
                        event="strategy.buy.momentum_velocity_pass",
                        symbol=token.get("symbol"),
                        momentum_5m=pc_5m,
                        required_velocity=min_5m_velocity,
                    )
            
            _log_trace(
                f"📈 Momentum from {ext_source}: {ext_momentum*100:.4f}% (need ≥ {momentum_need*100:.4f}%)",
                level="info",
                event="strategy.buy.external_momentum",
                source=ext_source,
                momentum=ext_momentum,
                required_momentum=momentum_need,
                symbol=token.get("symbol"),
            )
            if ext_momentum >= momentum_need:
                _log_trace(
                    "✅ External momentum buy signal → TRUE",
                    level="info",
                    event="strategy.buy.external_momentum_pass",
                    symbol=token.get("symbol"),
                )
                return True
            else:
                _log_trace(
                    "❌ External momentum insufficient.",
                    level="error",
                    event="strategy.buy.external_momentum_fail",
                    symbol=token.get("symbol"),
                )
                return False

        # External momentum enabled but unavailable - try using momentum from token dict
        _log_trace(
            "⚠️ External momentum unavailable; trying momentum from token data.",
            level="info",
            event="strategy.buy.no_external_momentum",
            symbol=token.get("symbol"),
        )
        # Fall through to token momentum check below
    else:
        # External momentum disabled - use momentum from token dict
        _log_trace(
            "⏭️ External momentum disabled; using momentum from token data.",
            level="info",
            event="strategy.buy.external_momentum_disabled",
            symbol=token.get("symbol"),
        )
    
    # Try using momentum data from token dict (from AI recommendations)
    # Check for momentum_24h/momentum_1h or priceChange24h/priceChange1h
    momentum_24h = token.get("momentum_24h") or token.get("priceChange24h")
    momentum_1h = token.get("momentum_1h") or token.get("priceChange1h")
    
    # Convert to decimal if needed (DexScreener provides as percentages)
    def to_decimal(mom_val):
        if mom_val is None or mom_val == "" or str(mom_val).lower() == "none" or str(mom_val).strip() == "":
            return None
        try:
            val = float(mom_val)
            # If value > 1, assume it's a percentage (e.g., 3.04 = 3.04%), convert to decimal
            # If value < 1, assume it's already decimal (e.g., 0.0304 = 3.04%)
            return val / 100.0 if abs(val) > 1 else val
        except (ValueError, TypeError):
            return None
    
    mom_24h = to_decimal(momentum_24h)
    mom_1h = to_decimal(momentum_1h)
    
    if mom_1h is not None:
        # Use 1h momentum as primary (most relevant for entry timing)
        _log_trace(
            f"📈 Momentum from token data (1h): {mom_1h*100:.4f}% (need ≥ {momentum_need*100:.4f}%)",
            level="info",
            event="strategy.buy.token_momentum",
            momentum_1h=mom_1h,
            momentum_24h=mom_24h,
            required_momentum=momentum_need,
            symbol=token.get("symbol"),
        )
        if mom_1h >= momentum_need:
            _log_trace(
                "✅ Token momentum buy signal → TRUE",
                level="info",
                event="strategy.buy.token_momentum_pass",
                symbol=token.get("symbol"),
            )
            return True
        else:
            _log_trace(
                f"❌ Token momentum insufficient ({mom_1h*100:.4f}% < {momentum_need*100:.4f}%).",
                level="info",
                event="strategy.buy.token_momentum_fail",
                symbol=token.get("symbol"),
            )
            return False
    elif mom_24h is not None:
        # Fallback to 24h momentum if 1h not available
        _log_trace(
            f"📈 Momentum from token data (24h): {mom_24h*100:.4f}% (need ≥ {momentum_need*100:.4f}%)",
            level="info",
            event="strategy.buy.token_momentum_24h",
            momentum_24h=mom_24h,
            required_momentum=momentum_need,
            symbol=token.get("symbol"),
        )
        if mom_24h >= momentum_need:
            _log_trace(
                "✅ Token momentum (24h) buy signal → TRUE",
                level="info",
                event="strategy.buy.token_momentum_24h_pass",
                symbol=token.get("symbol"),
            )
            return True
        else:
            _log_trace(
                f"❌ Token momentum (24h) insufficient ({mom_24h*100:.4f}% < {momentum_need*100:.4f}%).",
                level="info",
                event="strategy.buy.token_momentum_24h_fail",
                symbol=token.get("symbol"),
            )
            return False
    
    # No momentum available from any source
    _log_trace(
        "❌ No buy signal (no momentum confirmation from external or token data).",
        level="error",
        event="strategy.buy.no_signal",
        symbol=token.get("symbol"),
    )
    return False

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