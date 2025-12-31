"""
Solana Order-Flow Defense Module (Zero API Calls)

Uses indexed swap events from database to analyze order flow.
No RPC calls required - all data comes from local SQLite database.

This module blocks trades when order flow indicates:
- Single-wallet pumps
- Wash trading
- Fake momentum spikes
- Whale manipulation
"""

import time
import statistics
from typing import Dict, List, Optional
from collections import defaultdict

from src.config.config_loader import (
    get_config, get_config_bool, get_config_float, get_config_int
)
from src.storage.swap_events import get_swap_events
from src.monitoring.structured_logger import log_info, log_warning, log_error
from src.utils.advanced_cache import cache_get, cache_set

# Common quote tokens
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDT_MINT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
WSOL_MINT = "So11111111111111111111111111111111111111112"

# Default parameters (from spec)
DEFAULT_LOOKBACK_SECONDS = 120
DEFAULT_BUY_DOM_MIN = 0.55
DEFAULT_TOP_BUYER_MAX = 0.20
DEFAULT_MIN_UNIQUE_BUYERS = 12
DEFAULT_MAX_BUY_MULT_MEDIAN = 4.0
DEFAULT_MIN_BUY_VOLUME_USD = 15000
DEFAULT_MIN_TRADES = 25


def get_sol_price_usd() -> float:
    """Get current SOL price in USD"""
    try:
        from src.utils.market_data_fetcher import MarketDataFetcher
        fetcher = MarketDataFetcher()
        price = fetcher.get_token_price("So11111111111111111111111111111111111111112")
        if price and price > 0:
            return float(price)
    except Exception:
        pass
    return 150.0  # Conservative fallback


def _determine_side(swap: Dict, target_token: str) -> str:
    """
    Determine if swap is BUY or SELL based on token mints
    
    BUY: receiving target_token (quote -> base)
    SELL: sending target_token (base -> quote)
    """
    base_mint = swap.get("base_mint", "").lower() if swap.get("base_mint") else ""
    quote_mint = swap.get("quote_mint", "").lower() if swap.get("quote_mint") else ""
    target_token_lower = target_token.lower()
    
    # If base_mint is target token, this is a SELL
    if base_mint == target_token_lower:
        return "SELL"
    # If quote_mint is target token, this is a BUY (unusual but possible)
    elif quote_mint == target_token_lower:
        return "BUY"
    # Default: check amount direction
    # If amount_out > amount_in, likely a BUY (receiving more tokens)
    else:
        amount_in = swap.get("amount_in", 0) or 0
        amount_out = swap.get("amount_out", 0) or 0
        # Heuristic: if we're tracking base_mint and amount_out is significant, it's a BUY
        if base_mint == target_token_lower and amount_out > 0:
            return "BUY"
        return "SELL"


def calculate_trade_size_usd(
    swap: Dict,
    target_token: str,
    is_buy: bool
) -> float:
    """
    Calculate trade size in USD from swap data
    
    Args:
        swap: Swap event dict from database
        target_token: Token we're analyzing
        is_buy: True if this is a buy, False if sell
    """
    # Use volume_usd if available (most accurate)
    volume_usd = swap.get("volume_usd")
    if volume_usd and volume_usd > 0:
        return float(volume_usd)
    
    # Fallback: calculate from amounts
    amount_in = swap.get("amount_in", 0) or 0
    amount_out = swap.get("amount_out", 0) or 0
    base_mint = swap.get("base_mint", "").lower() if swap.get("base_mint") else ""
    quote_mint = swap.get("quote_mint", "").lower() if swap.get("quote_mint") else ""
    target_token_lower = target_token.lower()
    
    if is_buy:
        # Buying: we're receiving target_token, paying quote token
        if quote_mint in [USDC_MINT.lower(), USDT_MINT.lower()]:
            # Paying USDC/USDT - use amount_in
            return amount_in / 1e6  # USDC/USDT have 6 decimals
        elif quote_mint == WSOL_MINT.lower():
            # Paying SOL - convert to USD
            sol_price = get_sol_price_usd()
            return (amount_in / 1e9) * sol_price
        else:
            # Unknown quote - use price_usd if available
            price_usd = swap.get("price_usd", 0) or 0
            if price_usd > 0 and amount_out > 0:
                return amount_out * price_usd
    else:
        # Selling: we're sending target_token, receiving quote token
        if quote_mint in [USDC_MINT.lower(), USDT_MINT.lower()]:
            # Receiving USDC/USDT - use amount_out
            return amount_out / 1e6
        elif quote_mint == WSOL_MINT.lower():
            # Receiving SOL - convert to USD
            sol_price = get_sol_price_usd()
            return (amount_out / 1e9) * sol_price
        else:
            # Unknown quote - use price_usd if available
            price_usd = swap.get("price_usd", 0) or 0
            if price_usd > 0 and amount_in > 0:
                return amount_in * price_usd
    
    return 0.0


def fetch_swaps_from_indexer(
    token_mint: str,
    lookback_seconds: int,
    max_txs: int
) -> List[Dict]:
    """
    Fetch swaps from indexed database (ZERO API CALLS!)
    
    Args:
        token_mint: Token mint address
        lookback_seconds: How far back to look
        max_txs: Maximum swaps to return
    
    Returns:
        List of swap dicts with order-flow analysis fields
    """
    end_time = time.time()
    start_time = end_time - lookback_seconds
    
    # Query database - no API calls!
    swaps = get_swap_events(
        token_address=token_mint,
        start_time=start_time,
        end_time=end_time,
        limit=max_txs
    )
    
    if not swaps:
        return []
    
    # Convert to order-flow format
    result = []
    for swap in swaps:
        side = _determine_side(swap, token_mint)
        is_buy = (side == "BUY")
        trade_size_usd = calculate_trade_size_usd(swap, token_mint, is_buy)
        
        if trade_size_usd <= 0:
            continue  # Skip swaps with invalid trade size
        
        result.append({
            "signature": swap["tx_signature"],
            "signer_wallet": swap.get("signer_wallet", ""),
            "timestamp": swap["block_time"],
            "token_in_mint": swap.get("base_mint") if not is_buy else swap.get("quote_mint"),
            "token_out_mint": swap.get("quote_mint") if not is_buy else swap.get("base_mint"),
            "amount_in": swap.get("amount_in", 0) or 0,
            "amount_out": swap.get("amount_out", 0) or 0,
            "side": side,
            "trade_size_usd": trade_size_usd
        })
    
    return result


def evaluate_order_flow_solana(
    token_mint: str,
    raydium_pool: Optional[str] = None,
    now_ts: Optional[int] = None
) -> Dict:
    """
    Evaluate order flow for a Solana token (ZERO API CALLS)
    
    Uses indexed swap events from local database.
    
    Args:
        token_mint: Token mint address
        raydium_pool: Optional pool address (for future optimization)
        now_ts: Optional timestamp (defaults to now)
    
    Returns:
        {
            "pass": bool,
            "metrics": {
                "buy_dominance": float,
                "top_buyer_share": float,
                "unique_buyers": int,
                "total_buy_usd": float,
                "total_sell_usd": float,
                "num_buys": int,
                "num_sells": int,
                "largest_buy_usd": float,
                "median_buy_usd": float,
                "largest_vs_median": float
            },
            "reasons": [str]
        }
    """
    # Check cache first
    cache_key = f"order_flow_{token_mint}"
    cache_ttl = get_config_int("order_flow_cache_seconds", 30)
    cached = cache_get(cache_key)
    if cached:
        return cached
    
    # Get config parameters
    lookback_seconds = get_config_int("order_flow_lookback_seconds", DEFAULT_LOOKBACK_SECONDS)
    buy_dom_min = get_config_float("order_flow_buy_dom_min", DEFAULT_BUY_DOM_MIN)
    top_buyer_max = get_config_float("order_flow_top_buyer_max", DEFAULT_TOP_BUYER_MAX)
    min_unique_buyers = get_config_int("order_flow_min_unique_buyers", DEFAULT_MIN_UNIQUE_BUYERS)
    max_buy_mult_median = get_config_float("order_flow_max_buy_mult_median", DEFAULT_MAX_BUY_MULT_MEDIAN)
    min_buy_volume_usd = get_config_float("order_flow_min_buy_volume_usd", DEFAULT_MIN_BUY_VOLUME_USD)
    min_trades = get_config_int("order_flow_min_trades", DEFAULT_MIN_TRADES)
    max_txs_scanned = get_config_int("order_flow_max_txs_scanned", 200)
    fail_open = get_config_bool("order_flow_fail_open", False)
    
    now_ts = now_ts or int(time.time())
    
    # Fetch swaps from database (ZERO API CALLS!)
    swaps = fetch_swaps_from_indexer(token_mint, lookback_seconds, max_txs_scanned)
    
    if not swaps:
        if fail_open:
            result = {
                "pass": True,
                "metrics": {},
                "reasons": ["No swap data available (fail-open mode)"]
            }
        else:
            result = {
                "pass": False,
                "metrics": {},
                "reasons": ["No swap data available"]
            }
        cache_set(cache_key, result, ttl=cache_ttl)
        return result
    
    # Aggregate trades
    buy_volume = 0.0
    sell_volume = 0.0
    wallet_buy_volume = defaultdict(float)
    buy_sizes = []
    
    for swap in swaps:
        side = swap.get("side")
        signer = swap.get("signer_wallet", "")
        trade_size = swap.get("trade_size_usd", 0)
        
        if not signer or trade_size <= 0:
            continue
        
        if side == "BUY":
            buy_volume += trade_size
            wallet_buy_volume[signer] += trade_size
            buy_sizes.append(trade_size)
        else:  # SELL
            sell_volume += trade_size
    
    # Calculate metrics
    total_volume = buy_volume + sell_volume
    buy_dominance = buy_volume / total_volume if total_volume > 0 else 0.0
    
    top_buyer_share = 0.0
    if buy_volume > 0 and wallet_buy_volume:
        top_buyer_volume = max(wallet_buy_volume.values())
        top_buyer_share = top_buyer_volume / buy_volume
    
    unique_buyers = len(wallet_buy_volume)
    
    largest_buy = max(buy_sizes) if buy_sizes else 0.0
    median_buy = statistics.median(buy_sizes) if len(buy_sizes) >= 2 else (buy_sizes[0] if buy_sizes else 0.0)
    largest_vs_median = largest_buy / median_buy if median_buy > 0 else 0.0
    
    num_buys = len(buy_sizes)
    num_sells = len(swaps) - num_buys
    
    metrics = {
        "buy_dominance": buy_dominance,
        "top_buyer_share": top_buyer_share,
        "unique_buyers": unique_buyers,
        "total_buy_usd": buy_volume,
        "total_sell_usd": sell_volume,
        "num_buys": num_buys,
        "num_sells": num_sells,
        "largest_buy_usd": largest_buy,
        "median_buy_usd": median_buy,
        "largest_vs_median": largest_vs_median
    }
    
    # Apply filter rules
    reasons = []
    pass_all = True
    
    if buy_dominance < buy_dom_min:
        reasons.append(f"Buy dominance {buy_dominance:.2%} < {buy_dom_min:.2%}")
        pass_all = False
    
    if top_buyer_share > top_buyer_max:
        reasons.append(f"Top buyer share {top_buyer_share:.2%} > {top_buyer_max:.2%}")
        pass_all = False
    
    if unique_buyers < min_unique_buyers:
        reasons.append(f"Unique buyers {unique_buyers} < {min_unique_buyers}")
        pass_all = False
    
    if largest_vs_median > max_buy_mult_median:
        reasons.append(f"Largest buy {largest_vs_median:.1f}x median > {max_buy_mult_median:.1f}x")
        pass_all = False
    
    if buy_volume < min_buy_volume_usd:
        reasons.append(f"Buy volume ${buy_volume:,.0f} < ${min_buy_volume_usd:,.0f}")
        pass_all = False
    
    if (num_buys + num_sells) < min_trades:
        reasons.append(f"Total trades {num_buys + num_sells} < {min_trades}")
        pass_all = False
    
    result = {
        "pass": pass_all,
        "metrics": metrics,
        "reasons": reasons if not pass_all else ["All checks passed"]
    }
    
    # Cache result
    cache_set(cache_key, result, ttl=cache_ttl)
    
    # Log evaluation (JSON format as specified)
    log_info(
        "order_flow.evaluated",
        f"Order flow evaluation: {'PASS' if pass_all else 'FAIL'}",
        {
            "token": token_mint[:8],
            "pass": pass_all,
            "reasons": reasons if not pass_all else [],
            "buy_dominance": f"{buy_dominance:.2%}",
            "top_buyer_share": f"{top_buyer_share:.2%}",
            "unique_buyers": unique_buyers,
            "total_buy_usd": f"${buy_volume:,.0f}",
            "num_buys": num_buys,
            "num_sells": num_sells,
            "largest_vs_median": f"{largest_vs_median:.1f}x"
        }
    )
    
    return result
