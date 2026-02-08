#!/usr/bin/env python3
"""
Enhanced Async Trading Loop - Phase 3
Advanced async processing with connection pooling, batch processing, and performance optimization
"""

import asyncio
import aiohttp
import time
import json
import random
import math
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import logging

# Add src to path for imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.monitoring.structured_logger import log_info, log_error, log_warning, log_trade, log_performance
from src.config.config_validator import get_validated_config
from src.config.config_loader import get_config_bool, get_config_int, get_config_float, get_config
from src.monitoring.performance_monitor import record_trade_metrics, start_trading_session, end_trading_session
from src.core.centralized_risk_manager import assess_trade_risk, update_trade_result, is_circuit_breaker_active
from src.ai.ai_circuit_breaker import circuit_breaker_manager, check_ai_module_health
from src.monitoring.telegram_bot import send_periodic_status_report, send_telegram_message
from src.execution.multi_chain_executor import _launch_monitor_detached
from src.ai.ai_market_regime_detector import ai_market_regime_detector

logger = logging.getLogger(__name__)

@dataclass
class TradingMetrics:
    """Trading performance metrics"""
    total_trades: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    total_pnl: float = 0.0
    avg_execution_time: float = 0.0
    success_rate: float = 0.0
    trades_per_hour: float = 0.0
    health_score: int = 100

@dataclass
class ConnectionPool:
    """HTTP connection pool configuration"""
    max_connections: int = 100
    max_connections_per_host: int = 30
    keepalive_timeout: int = 30
    enable_compression: bool = True
    timeout: int = 30

class EnhancedAsyncTradingEngine:
    """
    Enhanced async trading engine with advanced features:
    - Connection pooling and reuse
    - Batch processing with configurable sizes
    - Real-time performance monitoring
    - Circuit breaker integration
    - Memory-efficient token processing
    - Parallel AI analysis
    """
    
    def __init__(self, config: Any = None):
        self.config = config or get_validated_config()
        self.connection_pool = ConnectionPool()
        self.metrics = TradingMetrics()
        self.session: Optional[aiohttp.ClientSession] = None
        self.token_cache: Dict[str, Any] = {}
        self.cache_ttl = 300  # 5 minutes
        # Make batch size configurable, default to 10 for better efficiency
        self.batch_size = get_config_int("trading.batch_size", 10)
        # Make max_concurrent_trades configurable, default to max_concurrent_positions or 3
        max_positions = get_config_int("max_concurrent_positions", 6)
        self.max_concurrent_trades = get_config_int("max_concurrent_trades", min(max_positions, 5))
        self.performance_window = deque(maxlen=100)  # Last 100 trades
        
        # Rate limiting
        self.rate_limiter = asyncio.Semaphore(self.max_concurrent_trades)
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # AI module health tracking
        self.ai_health_cache = {}
        self.ai_health_ttl = 60  # 1 minute
        
        # Cached AI engine instance for better performance
        self.ai_engine = None
        self._ai_engine_lock = asyncio.Lock()
        
        # DEX executor circuit breaker state
        self._dex_consecutive_failures = 0
        self._dex_circuit_breaker_until = 0.0  # timestamp when circuit breaker expires
        self._dex_last_alert_time = 0.0  # avoid alert spam
        self._dex_failure_threshold = get_config_int("dex_circuit_breaker.failure_threshold", 3)
        self._dex_cooldown_minutes = get_config_int("dex_circuit_breaker.cooldown_minutes", 15)
        self._dex_retry_attempts = get_config_int("dex_circuit_breaker.retry_attempts", 2)
        self._dex_retry_delay_seconds = get_config_float("dex_circuit_breaker.retry_delay_seconds", 3.0)
        
        # In-memory duplicate buy guard (belt-and-suspenders alongside open_positions.json check)
        # Prevents buying the same token twice even if position file save is delayed/fails
        self._recently_bought_tokens: Dict[str, float] = {}  # address -> timestamp
    
    def _safe_float(self, value, default=0.0) -> float:
        """Safely convert value to float, handling None, strings, and invalid types"""
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        return default
    
    def _get_token_sort_key(self, token: Dict) -> Tuple[float, float]:
        """
        Extract sort key (liquidity + volume) from token dict.
        Supports both normalized (flat) and nested (DexScreener) formats.
        Safe against None/strings by using safe-float conversion.
        Returns: (combined_score, volume) for sorting
        """
        # Try normalized format first (most common after early filtering)
        volume = self._safe_float(token.get("volume24h"))
        liquidity = self._safe_float(token.get("liquidity"))
        
        # Fallback to nested format if normalized fields are missing/zero
        if volume == 0 and liquidity == 0:
            # Try nested DexScreener format
            volume_data = token.get("volume")
            if isinstance(volume_data, dict):
                volume = self._safe_float(volume_data.get("h24"))
            else:
                volume = self._safe_float(volume_data)
            
            liquidity_data = token.get("liquidity")
            if isinstance(liquidity_data, dict):
                liquidity = self._safe_float(liquidity_data.get("usd"))
            else:
                liquidity = self._safe_float(liquidity_data)
        
        # Combined score: liquidity + volume (both in USD)
        combined_score = liquidity + volume
        return (combined_score, volume)
    
    def _calculate_accel_score(self, token: Dict) -> float:
        """
        Calculate acceleration score for a token to prioritize short-term movers.
        
        Score = w1 * priceChange15m + w2 * volumeChange15m + w3 * txnsPerMin + w4 * liquidityNorm
        
        Uses DexScreener fields when available; graceful fallback when data is missing.
        Applies input clamping to prevent score gaming.
        Returns a normalized score (higher = more acceleration).
        """
        import math
        
        # Get config
        accel_cfg = get_config("trading.accel_scoring", {})
        weights = accel_cfg.get("weights", {})
        w_price = weights.get("price_change_15m", 0.35)
        w_volume = weights.get("volume_change_15m", 0.30)
        w_txns = weights.get("txns_per_min", 0.20)
        w_liq = weights.get("liquidity_normalized", 0.15)
        
        # Get clamping config
        clamp_price_min = accel_cfg.get("clamp_price_change_15m_min", -0.50)
        clamp_price_max = accel_cfg.get("clamp_price_change_15m_max", 2.00)
        clamp_vol_min = accel_cfg.get("clamp_volume_change_15m_min", 0.00)
        clamp_vol_max = accel_cfg.get("clamp_volume_change_15m_max", 5.00)
        use_log_liq = accel_cfg.get("use_log_liquidity", True)
        max_txns_per_min = accel_cfg.get("max_txns_per_min", 10.0)
        
        # Get raw values
        price_change_15m = self._safe_float(token.get("priceChange15m"))
        price_change_5m = self._safe_float(token.get("priceChange5m"))
        volume_change_15m = self._safe_float(token.get("volumeChange15m"))
        volume_change_5m = self._safe_float(token.get("volumeChange5m"))
        liquidity = self._safe_float(token.get("liquidity"))
        volume_24h = self._safe_float(token.get("volume24h"))
        
        # Try to get transaction data (buys + sells as proxy for txns/min)
        txns_5m = token.get("txns5m", {})
        txns_15m = token.get("txns15m", {})
        buys_5m = self._safe_float(txns_5m.get("buys") if isinstance(txns_5m, dict) else 0)
        sells_5m = self._safe_float(txns_5m.get("sells") if isinstance(txns_5m, dict) else 0)
        buys_15m = self._safe_float(txns_15m.get("buys") if isinstance(txns_15m, dict) else 0)
        sells_15m = self._safe_float(txns_15m.get("sells") if isinstance(txns_15m, dict) else 0)
        
        # =====================================================================
        # 1. Price acceleration with clamping
        # =====================================================================
        if price_change_15m != 0:
            price_accel = price_change_15m / 100.0  # Convert to decimal
        elif price_change_5m != 0:
            price_accel = (price_change_5m * 3) / 100.0  # Estimate 15m from 5m
        else:
            price_accel = 0.0
        
        # CLAMP: Apply configurable bounds to prevent gaming
        price_accel = max(clamp_price_min, min(clamp_price_max, price_accel))
        
        # Shift to positive scale based on clamp range
        price_range = clamp_price_max - clamp_price_min
        price_score = (price_accel - clamp_price_min) / price_range if price_range > 0 else 0.5
        
        # =====================================================================
        # 2. Volume acceleration with clamping
        # =====================================================================
        if volume_change_15m != 0:
            vol_accel = volume_change_15m / 100.0
        elif volume_change_5m != 0:
            vol_accel = (volume_change_5m * 3) / 100.0
        else:
            vol_accel = 0.0
        
        # CLAMP: Apply configurable bounds (ignore negative volume change)
        vol_accel = max(clamp_vol_min, min(clamp_vol_max, vol_accel))
        
        # Shift to positive scale
        vol_range = clamp_vol_max - clamp_vol_min
        vol_score = (vol_accel - clamp_vol_min) / vol_range if vol_range > 0 else 0.0
        
        # =====================================================================
        # 3. Transaction rate proxy with capping
        # =====================================================================
        txns_per_min = 0.0
        txns_available = False
        
        if buys_5m + sells_5m > 0:
            txns_per_min = (buys_5m + sells_5m) / 5.0  # 5 minutes
            txns_available = True
        elif buys_15m + sells_15m > 0:
            txns_per_min = (buys_15m + sells_15m) / 15.0  # 15 minutes
            txns_available = True
        
        # CLAMP: Cap to prevent wash trading boost
        txns_per_min = min(max_txns_per_min, txns_per_min)
        
        # Normalize: scale to 0-1 based on max
        txns_score = txns_per_min / max_txns_per_min if max_txns_per_min > 0 else 0.0
        
        # If txns unavailable, reduce weight contribution
        effective_w_txns = w_txns if txns_available else w_txns * 0.3
        
        # =====================================================================
        # 4. Liquidity normalization (log scale if enabled)
        # =====================================================================
        if liquidity > 0:
            if use_log_liq:
                # Log scale: $10k=1.0, $100k=2.0, $1M=3.0, $10M=4.0
                liq_log = math.log10(max(liquidity, 1))
                # Normalize: log10($60k)≈4.78, log10($10M)≈7.0
                # Scale so $60k baseline gives ~0.3, $1M gives ~0.7, $10M gives 1.0
                liq_score = min(1.0, max(0.0, (liq_log - 4.0) / 3.0))
            else:
                # Linear normalization (old behavior)
                liq_log = math.log10(liquidity / 30000.0 + 1)
                liq_score = min(1.0, liq_log / 2.5)
        else:
            liq_score = 0.0
        
        # =====================================================================
        # Calculate weighted score (adjust for missing txns)
        # =====================================================================
        total_weight = w_price + w_volume + effective_w_txns + w_liq
        
        accel_score = (
            w_price * price_score +
            w_volume * vol_score +
            effective_w_txns * txns_score +
            w_liq * liq_score
        ) / total_weight if total_weight > 0 else 0.0
        
        # Store components in token for logging/debugging
        token['_accel_score'] = accel_score
        token['_accel_components'] = {
            'price_score': round(price_score, 4),
            'vol_score': round(vol_score, 4),
            'txns_score': round(txns_score, 4),
            'liq_score': round(liq_score, 4),
            'price_change_15m': price_change_15m,
            'price_change_15m_clamped': round(price_accel * 100, 2),
            'volume_change_15m': volume_change_15m,
            'volume_change_15m_clamped': round(vol_accel * 100, 2),
            'txns_per_min': round(txns_per_min, 2),
            'txns_available': txns_available,
            'liquidity': liquidity,
            'use_log_liq': use_log_liq,
        }
        
        return accel_score
    
    def _select_tokens_for_candles(self, tokens: List[Dict], max_tokens: int) -> List[Dict]:
        """
        Select tokens for candle fetching using acceleration-based prioritization.
        
        Split:
        - accel_slots_pct (70%): Highest acceleration score (requires higher liquidity floor)
        - stability_slots_pct (30%): Highest liquidity+volume (stability)
        
        Tokens must meet min_safety_liquidity for stability, accel_candidate_min_liquidity_usd for accel.
        """
        if not tokens:
            return []
        
        # Get config
        accel_cfg = get_config("trading.accel_scoring", {})
        accel_enabled = accel_cfg.get("enabled", True)
        accel_pct = accel_cfg.get("accel_slots_pct", 0.70)
        stability_pct = accel_cfg.get("stability_slots_pct", 0.30)
        min_safety_liq = accel_cfg.get("min_safety_liquidity", 30000)
        accel_min_liq = accel_cfg.get("accel_candidate_min_liquidity_usd", 60000)  # Higher floor for accel
        min_vol_15m_usd = accel_cfg.get("min_volume_15m_usd", 10000)  # Absolute volume guardrail for accel
        
        # Filter by minimum safety liquidity (for stability candidates)
        safe_tokens = [
            t for t in tokens
            if self._safe_float(t.get("liquidity")) >= min_safety_liq
        ]
        
        if not safe_tokens:
            log_info("trading.candle_selection.no_safe_tokens",
                    f"No tokens meet minimum safety liquidity ${min_safety_liq:,.0f}",
                    min_safety_liq=min_safety_liq)
            return []
        
        if not accel_enabled:
            # Fall back to old behavior: sort by liquidity + volume
            safe_tokens.sort(key=self._get_token_sort_key, reverse=True)
            selected = safe_tokens[:max_tokens]
            log_info("trading.candle_selection.fallback",
                    f"Accel scoring disabled, using liq+vol sort: {len(selected)} tokens",
                    selected_count=len(selected))
            return selected
        
        # Filter accel candidates by higher liquidity floor AND minimum 15m volume
        # Calculate volume_15m_usd from volume24h and volumeChange15m if not directly available
        def get_volume_15m_usd(t: Dict) -> float:
            """Estimate 15-minute USD volume for accel candidate filtering."""
            # Try direct volume15m field first
            vol_15m = self._safe_float(t.get("volume15m"))
            if vol_15m > 0:
                return vol_15m
            
            # Estimate from 24h volume (assuming uniform distribution: vol24h / 96 periods)
            vol_24h = self._safe_float(t.get("volume24h"))
            if vol_24h > 0:
                base_15m = vol_24h / 96.0  # 96 15-min periods in 24h
                
                # Adjust by volume change if available
                vol_change_15m = self._safe_float(t.get("volumeChange15m"))
                if vol_change_15m != 0:
                    # volumeChange15m is percentage, e.g., 50 means +50%
                    change_mult = 1.0 + (vol_change_15m / 100.0)
                    return base_15m * max(0.1, change_mult)  # Floor at 10% of base
                
                return base_15m
            
            return 0.0
        
        accel_candidates = [
            t for t in safe_tokens
            if (self._safe_float(t.get("liquidity")) >= accel_min_liq and
                get_volume_15m_usd(t) >= min_vol_15m_usd)
        ]
        
        # Track how many were filtered out by volume
        liq_only_candidates = [
            t for t in safe_tokens
            if self._safe_float(t.get("liquidity")) >= accel_min_liq
        ]
        vol_filtered_count = len(liq_only_candidates) - len(accel_candidates)
        
        # Calculate acceleration scores for accel candidates only
        for token in accel_candidates:
            self._calculate_accel_score(token)
        
        # Split slots
        accel_slots = int(max_tokens * accel_pct)
        stability_slots = max_tokens - accel_slots
        
        # Sort accel candidates by acceleration score (descending)
        accel_sorted = sorted(accel_candidates, key=lambda t: t.get('_accel_score', 0), reverse=True)
        
        # Sort all safe tokens by liquidity + volume (descending) for stability
        stability_sorted = sorted(safe_tokens, key=self._get_token_sort_key, reverse=True)
        
        # Select accel tokens (may be fewer than accel_slots if not enough candidates)
        accel_selected = accel_sorted[:accel_slots]
        actual_accel_count = len(accel_selected)
        
        # For stability slots, exclude tokens already selected by accel
        accel_addresses = {t.get("address", "").lower() for t in accel_selected}
        stability_candidates = [
            t for t in stability_sorted
            if t.get("address", "").lower() not in accel_addresses
        ]
        
        # Fill remaining slots with stability (including any unfilled accel slots)
        remaining_slots = max_tokens - actual_accel_count
        stability_selected = stability_candidates[:remaining_slots]
        
        # Combine
        selected = accel_selected + stability_selected
        
        # Log selection details
        log_info("trading.candle_selection.accel_priority",
                f"Selected {len(selected)} tokens for candles: "
                f"{actual_accel_count} by acceleration (target {accel_slots}, min_liq ${accel_min_liq/1000:.0f}k, min_vol15m ${min_vol_15m_usd/1000:.0f}k), "
                f"{len(stability_selected)} by stability, "
                f"{vol_filtered_count} filtered by low 15m volume",
                total_selected=len(selected),
                accel_count=actual_accel_count,
                accel_candidates_available=len(accel_candidates),
                vol_filtered_count=vol_filtered_count,
                stability_count=len(stability_selected),
                accel_slots_target=accel_slots,
                accel_min_liquidity=accel_min_liq,
                min_volume_15m_usd=min_vol_15m_usd)
        
        # Log top acceleration tokens
        if accel_selected:
            top_accel = accel_selected[:3]
            for i, t in enumerate(top_accel):
                components = t.get('_accel_components', {})
                log_info("trading.candle_selection.top_accel",
                        f"Top accel #{i+1}: {t.get('symbol', 'UNKNOWN')} "
                        f"(score={t.get('_accel_score', 0):.3f}, "
                        f"price15m={components.get('price_change_15m', 0):.1f}%, "
                        f"vol15m={components.get('volume_change_15m', 0):.1f}%, "
                        f"txns/min={components.get('txns_per_min', 0):.1f})",
                        symbol=t.get('symbol'),
                        accel_score=t.get('_accel_score', 0),
                        **components)
        
        return selected
        
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(
            limit=self.connection_pool.max_connections,
            limit_per_host=self.connection_pool.max_connections_per_host,
            keepalive_timeout=self.connection_pool.keepalive_timeout,
            enable_cleanup_closed=True,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        
        timeout = aiohttp.ClientTimeout(total=self.connection_pool.timeout)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'Hunter-Trading-Bot/3.0',
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip, deflate' if self.connection_pool.enable_compression else 'identity'
            }
        )
        
        log_info("trading.engine_init", "Enhanced async trading engine initialized", 
                {"max_connections": self.connection_pool.max_connections,
                "batch_size": self.batch_size})
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            log_info("trading.engine_close", "Async trading engine session closed")
    
    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    async def _get_cached_tokens(self, chain: str, cache_key: str) -> Optional[List[Dict]]:
        """Get tokens from cache if still valid"""
        if cache_key in self.token_cache:
            cached_data, timestamp = self.token_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                log_info("trading.cache", f"Using cached tokens for {chain}", {"count": len(cached_data)})
                return cached_data
        return None
    
    async def _cache_tokens(self, chain: str, cache_key: str, tokens: List[Dict]):
        """Cache tokens with timestamp"""
        self.token_cache[cache_key] = (tokens, time.time())
        log_info("trading.cache", f"Cached {len(tokens)} tokens for {chain}")
    
    async def _fetch_real_trending_tokens(self, chain: str, limit: int) -> List[Dict]:
        """Fetch real trending tokens from DexScreener API"""
        try:
            import aiohttp
            from datetime import datetime
            
            # Map chain names to DexScreener chain IDs
            chain_mapping = {
                "ethereum": "ethereum",
                "solana": "solana", 
                "base": "base",
                "arbitrum": "arbitrum",
                "polygon": "polygon",
                "bsc": "bsc"
            }
            
            dex_chain = chain_mapping.get(chain.lower(), chain.lower())
            
            # Get configurable DexScreener queries from config
            # Use config_loader for nested access with dot notation
            queries_by_chain = get_config("token_discovery.dexscreener_queries", {}) or {}
            query_terms = queries_by_chain.get(dex_chain) if isinstance(queries_by_chain, dict) else None
            
            # Fallback to default queries if config not found
            if not query_terms:
                if dex_chain == "ethereum":
                    query_terms = ["uniswap", "trending", "top", "volume", "liquidity", "ethereum", "eth", "weth"]
                elif dex_chain == "solana":
                    query_terms = ["bonk", "raydium", "jupiter", "orca"]
                else:
                    query_terms = [dex_chain, "trending"]
            
            # Build URLs from query terms
            trending_urls = [
                f"https://api.dexscreener.com/latest/dex/search/?q={term}"
                for term in query_terms
            ]
            
            # Get configurable prefilter thresholds
            min_vol = get_config_float("token_discovery.prefilter_min_volume_24h", 5000)
            min_liq = get_config_float("token_discovery.prefilter_min_liquidity_usd", 20000)
            
            all_tokens = []
            seen_pairs = set()
            
            # Debug counters for query stats
            query_stats = []
            
            async with aiohttp.ClientSession() as session:
                for url in trending_urls:
                    query_start_count = len(all_tokens)
                    pairs_returned = 0
                    pairs_after_chain_filter = 0
                    pairs_after_dedupe = 0
                    pairs_after_prefilter = 0
                    
                    try:
                        async with session.get(url, timeout=10) as response:
                            if response.status == 200:
                                data = await response.json()
                                pairs = data.get("pairs", []) if data and data.get("pairs") else []
                                pairs_returned = len(pairs)
                                log_info("trading.api_debug", f"Found {len(pairs)} pairs from {url}")
                                
                                for pair in pairs:
                                    # Extract token information first
                                    base_token = pair.get("baseToken", {})
                                    quote_token = pair.get("quoteToken", {})
                                    
                                    # Filter by chain and ensure we have valid data
                                    pair_chain = pair.get("chainId", "").lower()
                                    symbol = base_token.get("symbol", "")
                                    if (pair_chain == dex_chain or 
                                        pair_chain == chain.lower() or
                                        (dex_chain == "ethereum" and pair_chain in ["eth", "ethereum"]) or
                                        (dex_chain == "solana" and pair_chain == "solana") or
                                        (dex_chain == "base" and pair_chain == "base") or
                                        (dex_chain == "arbitrum" and pair_chain == "arbitrum") or
                                        (dex_chain == "polygon" and pair_chain == "polygon") or
                                        (dex_chain == "bsc" and pair_chain in ["bsc", "bsc-bnb"])):
                                        pairs_after_chain_filter += 1
                                        log_info("trading.filter", f"Processing {symbol} on {pair_chain} (target: {dex_chain})")
                                        
                                        # Skip if no valid token data
                                        if not base_token.get("address") or not base_token.get("symbol"):
                                            continue
                                            
                                        # Skip stablecoins and wrapped tokens
                                        symbol = base_token.get("symbol", "").upper()
                                        if any(stable in symbol for stable in ["USDT", "USDC", "DAI", "BUSD", "TUSD", "FRAX", "LUSD"]):
                                            continue
                                        if any(wrapped in symbol for wrapped in ["WETH", "WBTC", "WMATIC", "WBNB"]):
                                            continue
                                            
                                        # Skip native tokens (ETH, SOL, etc.) as they're not tradeable tokens
                                        if symbol in ["ETH", "SOL", "BTC", "BNB", "MATIC", "AVAX"]:
                                            log_info("trading.filter", f"Skipping native token: {symbol}")
                                            continue
                                            
                                        pair_id = pair.get("pairAddress") or base_token.get("address")
                                        if not pair_id:
                                            continue
                                        if pair_id in seen_pairs:
                                            continue
                                        pairs_after_dedupe += 1
                                        
                                        # Calculate metrics
                                        price_usd = float(pair.get("priceUsd", 0))
                                        volume_24h = float(pair.get("volume", {}).get("h24", 0))
                                        volume_1h = float(pair.get("volume", {}).get("h1", 0))
                                        liquidity_usd = float(pair.get("liquidity", {}).get("usd", 0))
                                        price_change_24h = float(pair.get("priceChange", {}).get("h24", 0))
                                        price_change_1h = float(pair.get("priceChange", {}).get("h1", 0))
                                        
                                        # Calculate volume change 1h as decimal (0.1 = 10% increase)
                                        # Compare current hour volume to 24h average hourly rate
                                        # Formula: (volume.h1 * 24) / volume.h24 - 1
                                        # This gives decimal change vs 24h average (positive = increasing, negative = decreasing)
                                        volume_change_1h = None
                                        if volume_24h > 0 and volume_1h > 0:
                                            # Calculate decimal change: e.g., 0.1 = 10% increase, -0.5 = 50% decrease
                                            volume_change_1h = (volume_1h * 24) / volume_24h - 1
                                        elif volume_24h > 0:
                                            # If h1 volume is 0 but h24 exists, it's a -1.0 change (100% decrease)
                                            volume_change_1h = -1.0
                                        
                                        # Skip tokens with poor metrics (using configurable thresholds)
                                        if price_usd <= 0 or volume_24h < min_vol or liquidity_usd < min_liq:
                                            continue
                                        pairs_after_prefilter += 1
                                            
                                        token = {
                                            "symbol": base_token.get("symbol", ""),
                                            "address": base_token.get("address", ""),
                                            "chain": chain,
                                            "priceUsd": price_usd,
                                            "volume24h": volume_24h,
                                            "liquidity": liquidity_usd,
                                            "marketCap": float(pair.get("marketCap", 0)),
                                            "priceChange24h": price_change_24h,
                                            "priceChange1h": price_change_1h,
                                            "volumeChange1h": volume_change_1h,  # Percentage change vs 24h average
                                            "holders": int(pair.get("holders", 0)),
                                            "transactions24h": int(pair.get("txns", {}).get("h24", {}).get("buys", 0)) + 
                                                             int(pair.get("txns", {}).get("h24", {}).get("sells", 0)),
                                            "dex": pair.get("dexId", ""),
                                            "pair_address": pair.get("pairAddress", ""),
                                            "timestamp": datetime.now().isoformat()
                                        }
                                        
                                        all_tokens.append(token)
                                        seen_pairs.add(pair_id)
                                        
                                        if len(all_tokens) >= limit * 2:  # Get more than needed for filtering
                                            break
                    
                        # Record query stats (successful query)
                        query_end_count = len(all_tokens)
                        tokens_kept_this_query = query_end_count - query_start_count
                        query_stats.append({
                            "url": url,
                            "chain": chain,
                            "pairs_returned": pairs_returned,
                            "pairs_after_chain_filter": pairs_after_chain_filter,
                            "pairs_after_dedupe": pairs_after_dedupe,
                            "pairs_after_prefilter": pairs_after_prefilter,
                            "tokens_kept": tokens_kept_this_query,
                            "tokens_total_so_far": query_end_count
                        })
                                            
                    except Exception as e:
                        log_error("trading.api_error", f"Error fetching from {url}: {e}")
                        # Record failed query stats
                        query_stats.append({
                            "url": url,
                            "chain": chain,
                            "error": str(e),
                            "pairs_returned": 0,
                            "tokens_kept": 0
                        })
                        # Debug: print response for troubleshooting
                        try:
                            async with session.get(url, timeout=5) as debug_response:
                                debug_data = await debug_response.text()
                                log_error("trading.api_debug", f"API response: {debug_data[:200]}...")
                        except:
                            pass
                        continue
                    
                    if len(all_tokens) >= limit * 2:
                        break
            
            # Log query stats for debugging
            for stat in query_stats:
                if "error" in stat:
                    log_info("trading.discovery_query_stats", f"Query failed: {stat['url']}", {
                        "chain": stat["chain"],
                        "error": stat["error"]
                    })
                else:
                    log_info("trading.discovery_query_stats", f"DexScreener query stats for {stat['url']}", {
                        "chain": stat["chain"],
                        "pairs_returned": stat["pairs_returned"],
                        "pairs_after_chain_filter": stat["pairs_after_chain_filter"],
                        "pairs_after_dedupe": stat["pairs_after_dedupe"],
                        "pairs_after_prefilter": stat["pairs_after_prefilter"],
                        "tokens_kept": stat["tokens_kept"],
                        "tokens_total_so_far": stat["tokens_total_so_far"]
                    })
                        
            # Sort by volume and take the top tokens
            all_tokens.sort(key=lambda x: x["volume24h"], reverse=True)
            return all_tokens[:limit]
            
        except Exception as e:
            log_error("trading.fetch_error", f"Error fetching real tokens: {e}")
            return []
    
    async def _fetch_tracked_tokens(self, chain: str) -> List[Dict]:
        """
        Fetch token data for minute_price_tracker TRACKED_TOKENS.
        Always include these in selection so price tracking stays continuous.
        Returns tokens in same format as _fetch_real_trending_tokens, with _is_tracked_token=True.
        """
        try:
            from src.storage.minute_price_tracker import TRACKED_TOKENS
        except ImportError:
            log_warning("trading.tracked_tokens", "minute_price_tracker not available, skipping tracked tokens")
            return []
        
        if not TRACKED_TOKENS:
            return []
        
        # All current tracked tokens are Solana
        if chain.lower() != "solana":
            return []
        
        tracked_list = []
        async with aiohttp.ClientSession() as session:
            for name, address in TRACKED_TOKENS.items():
                if not address:
                    continue
                try:
                    url = f"https://api.dexscreener.com/latest/dex/tokens/{address}"
                    async with session.get(url, timeout=10) as resp:
                        if resp.status != 200:
                            raise ValueError(f"HTTP {resp.status}")
                        data = await resp.json()
                        pairs = data.get("pairs") or []
                        # Prefer Solana pairs
                        sol_pairs = [p for p in pairs if (p.get("chainId") or "").lower() == "solana"]
                        pair = (sol_pairs or pairs)[0] if (sol_pairs or pairs) else None
                        if not pair:
                            raise ValueError("no pairs")
                        base = pair.get("baseToken", {}) or {}
                        price_usd = float(pair.get("priceUsd") or 0)
                        vol = pair.get("volume") or {}
                        liq = pair.get("liquidity") or {}
                        volume_24h = float(vol.get("h24") or 0)
                        volume_1h = float(vol.get("h1") or 0)
                        liquidity_usd = float(liq.get("usd") or 0)
                        pc = pair.get("priceChange") or {}
                        price_change_24h = float(pc.get("h24") or 0)
                        price_change_1h = float(pc.get("h1") or 0)
                        volume_change_1h = None
                        if volume_24h > 0 and volume_1h > 0:
                            volume_change_1h = (volume_1h * 24) / volume_24h - 1
                        elif volume_24h > 0:
                            volume_change_1h = -1.0
                        token = {
                            "symbol": base.get("symbol") or name,
                            "address": base.get("address") or address,
                            "chain": chain,
                            "chainId": chain,
                            "priceUsd": price_usd,
                            "volume24h": volume_24h,
                            "liquidity": liquidity_usd,
                            "marketCap": float(pair.get("marketCap") or 0),
                            "priceChange24h": price_change_24h,
                            "priceChange1h": price_change_1h,
                            "volumeChange1h": volume_change_1h,
                            "dex": pair.get("dexId", ""),
                            "pair_address": pair.get("pairAddress", ""),
                            "timestamp": datetime.now().isoformat(),
                            "_is_tracked_token": True,
                        }
                        tracked_list.append(token)
                except Exception as e:
                    # Minimal token dict so we still include it (filters bypassed for tracked)
                    token = {
                        "symbol": name,
                        "address": address,
                        "chain": chain,
                        "chainId": chain,
                        "priceUsd": 0.0,
                        "volume24h": 1.0,
                        "liquidity": 1.0,
                        "marketCap": 0,
                        "priceChange24h": 0,
                        "priceChange1h": 0,
                        "volumeChange1h": None,
                        "dex": "",
                        "pair_address": "",
                        "timestamp": datetime.now().isoformat(),
                        "_is_tracked_token": True,
                    }
                    tracked_list.append(token)
                    log_warning("trading.tracked_tokens.fallback",
                               f"Tracked token {name} API fetch failed, using minimal dict: {e}",
                               symbol=name, address=address[:16])
                await asyncio.sleep(0.15)
        
        if tracked_list:
            log_info("trading.tracked_tokens.fetched", f"Fetched {len(tracked_list)} tracked tokens for {chain}",
                     count=len(tracked_list), chain=chain)
        return tracked_list
    
    def _categorize_error(self, error_msg: str, token: Dict) -> Tuple[str, str]:
        """
        Categorize error type for proper handling
        
        Returns:
            (error_type, token_address) tuple
            error_type: "gate", "token", or "systemic"
        """
        error_lower = error_msg.lower()
        token_address = token.get("address", "").lower()
        
        # Gate failures - protective mechanisms that prevent trading
        gate_indicators = [
            "time window blocked",
            "window_score",
            "risk gate blocked",
            "circuit breaker active",
            "risk assessment failed"
        ]
        if any(indicator in error_lower for indicator in gate_indicators):
            return ("gate", token_address)
        
        # Token-specific failures - issues with this specific token
        token_indicators = [
            "jupiter execution failed",
            "jupiter execution exception",
            "not tradeable",
            "not tradable",
            "token not found",
            "insufficient liquidity",
            "pool not found",
            "invalid token",
            "token delisted",
            "raydium execution failed",
            "uniswap execution failed"
        ]
        if any(indicator in error_lower for indicator in token_indicators):
            return ("token", token_address)
        
        # Systemic failures - network, wallet, RPC issues
        systemic_indicators = [
            "network error",
            "connection error",
            "timeout",
            "rpc error",
            "wallet error",
            "insufficient funds",
            "gas estimation failed",
            "transaction failed",
            "unknown error"
        ]
        if any(indicator in error_lower for indicator in systemic_indicators):
            return ("systemic", token_address)
        
        # Default to systemic for unknown errors (fail safe)
        return ("systemic", token_address)
    
    async def _execute_real_trade(self, token: Dict, position_size: float, chain: str) -> Dict[str, Any]:
        """
        Execute real trade using chain-specific executors with DEX circuit breaker protection.
        """
        symbol = token.get("symbol", "")
        
        # DEX circuit breaker check - block trades if executors are persistently failing
        now = time.time()
        if now < self._dex_circuit_breaker_until:
            remaining = int(self._dex_circuit_breaker_until - now)
            log_warning("trading.dex_circuit_breaker_active",
                       f"DEX circuit breaker active for {symbol} - {remaining}s remaining "
                       f"({self._dex_consecutive_failures} consecutive failures)",
                       symbol=symbol, remaining_seconds=remaining,
                       consecutive_failures=self._dex_consecutive_failures)
            return {
                "success": False,
                "error": f"DEX circuit breaker active ({remaining}s remaining after {self._dex_consecutive_failures} consecutive failures)",
                "error_type": "gate",  # Mark as gate so it doesn't count as a trade attempt
                "profit_loss": 0,
                "tx_hash": ""
            }
        
        # Execute the trade
        result = await self._execute_real_trade_internal(token, position_size, chain)
        
        # Track DEX executor health
        if result.get("success", False):
            if self._dex_consecutive_failures > 0:
                log_info("trading.dex_recovered",
                        f"DEX executor recovered after {self._dex_consecutive_failures} consecutive failures",
                        consecutive_failures=self._dex_consecutive_failures)
                # Send recovery alert if we previously alerted about failures
                if self._dex_consecutive_failures >= self._dex_failure_threshold:
                    send_telegram_message(
                        f"✅ DEX Executor Recovered\n"
                        f"Successfully executed trade for {symbol} after "
                        f"{self._dex_consecutive_failures} consecutive failures.",
                        message_type="dex_health"
                    )
            self._dex_consecutive_failures = 0
        elif result.get("error_type") != "gate":
            # Only count actual execution failures, not gate blocks
            self._dex_consecutive_failures += 1
            log_warning("trading.dex_failure_tracked",
                       f"DEX failure #{self._dex_consecutive_failures} for {symbol}: {result.get('error', 'unknown')}",
                       symbol=symbol, consecutive_failures=self._dex_consecutive_failures,
                       error=result.get('error', 'unknown'))
            
            # Trigger circuit breaker after threshold
            if self._dex_consecutive_failures >= self._dex_failure_threshold:
                cooldown_seconds = self._dex_cooldown_minutes * 60
                self._dex_circuit_breaker_until = time.time() + cooldown_seconds
                
                log_error("trading.dex_circuit_breaker_triggered",
                         f"DEX circuit breaker TRIGGERED after {self._dex_consecutive_failures} consecutive failures. "
                         f"Pausing DEX execution for {self._dex_cooldown_minutes} minutes.",
                         consecutive_failures=self._dex_consecutive_failures,
                         cooldown_minutes=self._dex_cooldown_minutes)
                
                # Send Telegram alert (rate-limited: max once per cooldown period)
                if time.time() - self._dex_last_alert_time > cooldown_seconds:
                    self._dex_last_alert_time = time.time()
                    send_telegram_message(
                        f"🚨 DEX Circuit Breaker Triggered\n\n"
                        f"• Consecutive failures: {self._dex_consecutive_failures}\n"
                        f"• Last error: {result.get('error', 'unknown')[:100]}\n"
                        f"• Cooldown: {self._dex_cooldown_minutes} minutes\n"
                        f"• Resumes at: {datetime.fromtimestamp(self._dex_circuit_breaker_until).strftime('%H:%M:%S')}\n\n"
                        f"⚠️ Check Jupiter/Raydium executor health and RPC connectivity.",
                        message_type="dex_health"
                    )
        
        return result
    
    async def _execute_real_trade_internal(self, token: Dict, position_size: float, chain: str) -> Dict[str, Any]:
        """Execute real trade using DEX integrations with retry logic"""
        try:
            symbol = token.get("symbol", "")
            address = token.get("address", "")
            retry_attempts = self._dex_retry_attempts
            retry_delay = self._dex_retry_delay_seconds
            last_error = "unknown"
            
            # Import the appropriate executor based on chain
            if chain.lower() == "solana":
                from .jupiter_executor import buy_token_solana
                
                # Retry loop: try Jupiter then Raydium, with configurable retries
                for attempt in range(retry_attempts):
                    if attempt > 0:
                        log_info("trading.dex_retry", 
                                f"Retry attempt {attempt + 1}/{retry_attempts} for {symbol} "
                                f"(waiting {retry_delay}s)",
                                symbol=symbol, attempt=attempt + 1, max_attempts=retry_attempts)
                        await asyncio.sleep(retry_delay * attempt)  # Progressive backoff
                    
                    # Try Jupiter first
                    try:
                        log_info("trading.jupiter", f"Executing Jupiter trade for {symbol} on Solana" +
                                (f" (attempt {attempt + 1})" if attempt > 0 else ""))
                        result = buy_token_solana(address, position_size, symbol, test_mode=False)
                        
                        # Handle both 2-value (legacy) and 3-value return formats
                        if len(result) == 3:
                            tx_hash, success, quoted_output_amount = result
                        elif len(result) == 2:
                            tx_hash, success = result
                            quoted_output_amount = None
                            log_info("trading.jupiter_legacy_return", f"Jupiter returned 2 values for {symbol}, assuming no quoted amount")
                        else:
                            log_error("trading.jupiter_error", f"Jupiter returned unexpected number of values for {symbol}: {len(result)}")
                            raise ValueError(f"Unexpected return format from buy_token_solana: {len(result)} values")
                        
                        if success and tx_hash:
                            # Analyze transaction to get actual execution details including slippage
                            buy_fee_data = await self._analyze_buy_transaction(tx_hash, chain, "jupiter", quoted_output_amount)
                            
                            # At entry time, P&L is 0.0 - will be calculated when position is closed
                            profit_loss = 0.0  # No profit/loss until position is closed
                            
                            trade_result = {
                                "success": True,
                                "profit_loss": profit_loss,
                                "tx_hash": tx_hash,
                                "dex": "jupiter",
                                "fee_data": buy_fee_data
                            }
                            
                            # Include actual slippage in trade result for metrics
                            if 'actual_slippage' in buy_fee_data:
                                trade_result['slippage'] = buy_fee_data['actual_slippage']
                            
                            return trade_result
                        elif not success:
                            # Log detailed error for Jupiter failures
                            error_msg = tx_hash if isinstance(tx_hash, str) and not tx_hash.startswith("0x") and tx_hash else "Jupiter returned unsuccessful"
                            last_error = f"Jupiter: {error_msg}"
                            log_error("trading.jupiter_error", f"Jupiter execution failed for {symbol}: {error_msg}")
                    except ValueError as ve:
                        last_error = f"Jupiter ValueError: {ve}"
                        log_error("trading.jupiter_error", f"Jupiter execution value error for {symbol}: {ve}")
                    except Exception as e:
                        error_str = str(e)
                        last_error = f"Jupiter exception: {error_str[:200]}"
                        log_error("trading.jupiter_error", f"Jupiter execution exception for {symbol}: {error_str}")
                        
                        if "not enough values to unpack" in error_str.lower():
                            log_error("trading.jupiter_unpack_error", 
                                    f"Jupiter return value unpacking error for {symbol}. "
                                    f"This may indicate the transaction actually succeeded but return format was unexpected. "
                                    f"Please verify transaction on-chain manually.")
                    
                    # Try Raydium fallback
                    try:
                        from .raydium_executor import RaydiumExecutor
                        
                        log_info("trading.raydium", f"Executing Raydium trade for {symbol} on Solana" +
                                (f" (attempt {attempt + 1})" if attempt > 0 else ""))
                        raydium = RaydiumExecutor()
                        success, tx_hash = raydium.execute_trade(address, position_size, is_buy=True)
                        if success and tx_hash:
                            buy_fee_data = await self._analyze_buy_transaction(tx_hash, chain, "raydium", None)
                            
                            profit_loss = 0.0  # No profit/loss until position is closed
                            return {
                                "success": True,
                                "profit_loss": profit_loss,
                                "tx_hash": tx_hash,
                                "dex": "raydium",
                                "fee_data": buy_fee_data
                            }
                        else:
                            last_error = f"Raydium: trade unsuccessful"
                    except Exception as e:
                        last_error = f"Raydium: {str(e)[:200]}"
                        log_error("trading.raydium_error", f"Raydium execution failed: {e}")
                
                # All retry attempts exhausted for Solana
                log_error("trading.dex_all_retries_exhausted",
                         f"All {retry_attempts} DEX retry attempts exhausted for {symbol}. Last error: {last_error}",
                         symbol=symbol, retry_attempts=retry_attempts, last_error=last_error)
                    
            elif chain.lower() in ["ethereum", "base", "arbitrum", "polygon"]:
                from .uniswap_executor import buy_token
                
                # Execute Uniswap trade
                try:
                    log_info("trading.uniswap", f"Executing Uniswap trade for {symbol} on {chain}")
                    tx_hash, success = buy_token(address, position_size, symbol)
                    if success and tx_hash:
                        # Analyze transaction to get actual execution details
                        # Note: Uniswap executor doesn't return quoted amount yet
                        buy_fee_data = await self._analyze_buy_transaction(tx_hash, chain, "uniswap", None)
                        
                        # At entry time, P&L is 0.0 - will be calculated when position is closed
                        profit_loss = 0.0  # No profit/loss until position is closed
                        return {
                            "success": True,
                            "profit_loss": profit_loss,
                            "tx_hash": tx_hash,
                            "dex": "uniswap",
                            "fee_data": buy_fee_data
                        }
                except Exception as e:
                    log_error("trading.uniswap_error", f"Uniswap execution failed: {e}")
            
            # If all executors failed
            return {
                "success": False,
                "error": f"No working DEX executor found for {chain}",
                "profit_loss": 0,
                "tx_hash": ""
            }
            
        except Exception as e:
            log_error("trading.real_execution_error", f"Real trade execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "profit_loss": 0,
                "tx_hash": ""
            }
    
    async def fetch_trending_tokens_async(self, chain: str, limit: int = 20) -> List[Dict]:
        """Enhanced token fetching with caching and error handling"""
        cache_key = f"{chain}_trending_{limit}"
        
        # Check cache first
        cached_tokens = await self._get_cached_tokens(chain, cache_key)
        if cached_tokens:
            return cached_tokens
        
        await self._rate_limit()
        
        try:
            log_info("trading.fetch", f"Fetching real trending tokens for {chain} (limit: {limit})")
            
            # Fetch real data from DexScreener API
            tokens = await self._fetch_real_trending_tokens(chain, limit)
            
            # Cache the results
            await self._cache_tokens(chain, cache_key, tokens)
            
            log_info("trading.fetch", f"Fetched {len(tokens)} trending tokens for {chain}")
            return tokens
            
        except Exception as e:
            log_error("trading.fetch_error", f"Error fetching tokens for {chain}: {e}")
            return []
    
    async def _check_ai_module_health(self) -> Dict[str, Any]:
        """Check AI module health with caching"""
        current_time = time.time()
        
        # Check cache first
        if 'ai_health' in self.ai_health_cache:
            cached_health, timestamp = self.ai_health_cache['ai_health']
            if current_time - timestamp < self.ai_health_ttl:
                return cached_health
        
        # Get fresh health data
        try:
            health = check_ai_module_health()
            self.ai_health_cache['ai_health'] = (health, current_time)
            return health
        except Exception as e:
            log_error("trading.ai_health_error", f"Error checking AI health: {e}")
            return {"overall_healthy": False, "unhealthy_modules": ["unknown"]}
    
    async def _get_ai_engine(self):
        """Get or initialize cached AI engine instance"""
        if self.ai_engine is None:
            async with self._ai_engine_lock:
                # Double-check after acquiring lock
                if self.ai_engine is None:
                    from src.ai.ai_integration_engine import AIIntegrationEngine
                    self.ai_engine = AIIntegrationEngine()
                    await self.ai_engine.initialize()
                    log_info("trading.ai_engine_cached", "AI engine initialized and cached")
        return self.ai_engine
    
    async def _analyze_token_ai(self, token: Dict, regime_data: Optional[Dict[str, Any]] = None) -> Dict:
        """Perform AI analysis on a single token"""
        symbol = token.get("symbol", "UNKNOWN")
        
        try:
            # Use cached AI integration engine for analysis
            ai_engine = await self._get_ai_engine()
            
            # Perform real AI analysis
            ai_result = await ai_engine.analyze_token(token, regime_data=regime_data)
            
            # Extract results from AI analysis
            # AIAnalysisResult uses overall_score, not quality_score, and attributes are Dict objects
            sentiment = ai_result.sentiment_analysis or {}
            prediction = ai_result.prediction_analysis or {}
            risk = ai_result.risk_assessment or {}
            market = ai_result.market_analysis or {}
            technical = ai_result.technical_analysis or {}
            execution = ai_result.execution_analysis or {}
            recommendations = ai_result.recommendations or {}
            
            analysis = {
                "quality_score": round(ai_result.overall_score, 3),  # Use overall_score instead of quality_score
                "sentiment_analysis": {
                    "category": sentiment.get("category", "neutral"),
                    "score": round(sentiment.get("score", 0.5), 3),
                    "confidence": round(sentiment.get("confidence", 0.5), 3)
                },
                "price_prediction": {
                    "success_probability": round(prediction.get("price_movement_probability", 0.5), 3),
                    "confidence_level": round(prediction.get("confidence", 0.5), 3),
                    "expected_return": round(prediction.get("expected_return", 0.1), 3),
                    "risk_score": round(risk.get("risk_score", 0.3), 3)
                },
                "market_analysis": {
                    "trend": market.get("market_trend", "neutral"),
                    "volatility": "high" if abs(token.get("priceChange24h", 0)) > 0.1 else "low",
                    "liquidity_score": round(market.get("liquidity_score", 0.5), 3),
                    "volume_score": round(market.get("volume_score", 0.5), 3)
                },
                "trading_recommendation": {
                    "action": recommendations.get("action", "hold"),
                    "confidence": round(recommendations.get("confidence", ai_result.confidence), 3),
                    "position_size": recommendations.get("position_size", 10),
                    "take_profit": round(recommendations.get("take_profit", 0.15), 3),
                    "stop_loss": round(recommendations.get("stop_loss", 0.08), 3)
                },
                "risk_assessment": {  # Include risk_assessment at top level for proper extraction
                    "risk_score": round(risk.get("risk_score", 0.5), 3),
                    "risk_level": risk.get("risk_level", "medium"),
                    "risk_factors": risk.get("risk_factors", []),
                    "confidence": round(risk.get("confidence", 0.5), 3)
                },
                "risk_factors": risk.get("risk_factors", []),
                "technical_analysis": {
                    "technical_score": round(technical.get("technical_score", 0.5), 3),
                    "trend": technical.get("trend", "neutral"),
                    "signals": technical.get("signals", [])
                },
                "execution_analysis": {
                    "execution_score": round(execution.get("execution_score", 0.5), 3),
                    "recommended_slippage": round(execution.get("recommended_slippage", 0.05), 3),
                    "optimal_timing": execution.get("optimal_timing", "wait")
                },
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            log_error("trading.ai_analysis_error", f"AI analysis failed for {symbol}: {e}")
            return {
                "quality_score": 0,
                "trading_recommendation": {"action": "skip", "confidence": 0.0},
                "error": str(e)
            }
    
    def _filter_tokens_early(self, tokens: List[Dict]) -> List[Dict]:
        """
        Early filtering: Apply volume/liquidity filters BEFORE expensive AI analysis.
        This saves significant API calls and processing time.
        Tracked tokens (minute_price_tracker) always pass and are included.
        
        Returns only tokens that pass basic quality thresholds.
        """
        min_volume = get_config_float("min_volume_24h_for_buy", 200000)
        min_liquidity = get_config_float("min_liquidity_usd_for_buy", 200000)
        min_price = get_config_float("min_price_usd", 0.000001)
        
        filtered_tokens: List[Dict] = []
        filtered_count = 0
        tracked_bypassed = 0
        
        for token in tokens:
            if token.get("_is_tracked_token", False):
                filtered_tokens.append(token)
                tracked_bypassed += 1
                continue
            
            volume_24h = float(token.get("volume24h", 0))
            liquidity = float(token.get("liquidity", 0))
            price_usd = float(token.get("priceUsd", 0))
            
            # Apply early filters - reject tokens that don't meet basic requirements
            if price_usd < min_price:
                filtered_count += 1
                continue
            
            if volume_24h < min_volume:
                filtered_count += 1
                continue
            
            if liquidity < min_liquidity:
                filtered_count += 1
                continue
            
            # Token passed early filters
            filtered_tokens.append(token)
        
        if tracked_bypassed > 0:
            log_info("trading.tracked_tokens.bypassed_filters",
                     f"{tracked_bypassed} tracked tokens bypassed early filters",
                     tracked_bypassed=tracked_bypassed)
        if filtered_count > 0:
            log_info("trading.early_filter", 
                    f"Early filtering: {filtered_count} tokens filtered out before AI analysis "
                    f"({len(filtered_tokens)}/{len(tokens)} passed)",
                    {"filtered_out": filtered_count, "passed": len(filtered_tokens), "total": len(tokens)})
        
        return filtered_tokens
    
    async def _analyze_buy_transaction(self, tx_hash: str, chain: str, dex: str, quoted_output_amount: Optional[int] = None) -> Dict[str, Any]:
        """Analyze buy transaction to extract fee data and actual slippage"""
        try:
            if chain.lower() == "solana":
                from src.utils.solana_transaction_analyzer import analyze_jupiter_transaction
                from src.config.secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS
                
                fee_data = analyze_jupiter_transaction(
                    SOLANA_RPC_URL, 
                    tx_hash, 
                    SOLANA_WALLET_ADDRESS,
                    is_buy=True,
                    quoted_output_amount=quoted_output_amount
                )
                
                result = {
                    'entry_gas_fee_usd': fee_data.get('gas_fee_usd', 0),
                    'entry_amount_usd_actual': fee_data.get('actual_cost_usd', 0),
                    'entry_tokens_received': fee_data.get('tokens_received'),
                    'buy_tx_hash': tx_hash
                }
                
                # Include actual slippage if calculated
                if 'actual_slippage' in fee_data:
                    result['actual_slippage'] = fee_data['actual_slippage']
                
                return result
            elif chain.lower() in ["ethereum", "base", "arbitrum", "polygon"]:
                from src.utils.transaction_analyzer import analyze_buy_transaction
                from src.execution.uniswap_executor import w3
                
                fee_data = analyze_buy_transaction(w3, tx_hash)
                
                return {
                    'entry_gas_fee_usd': fee_data.get('gas_fee_usd', 0),
                    'entry_amount_usd_actual': fee_data.get('actual_cost_usd', 0),
                    'entry_tokens_received': fee_data.get('tokens_received'),
                    'buy_tx_hash': tx_hash
                }
        except Exception as e:
            log_error("trading.fee_analysis_error", f"Error analyzing buy transaction {tx_hash}: {e}")
            return {}
        return {}
    
    async def _analyze_sell_transaction(self, tx_hash: str, chain: str, dex: str) -> Dict[str, Any]:
        """Analyze sell transaction to extract fee data"""
        try:
            if chain.lower() == "solana":
                from src.utils.solana_transaction_analyzer import analyze_jupiter_transaction
                from src.config.secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS
                
                fee_data = analyze_jupiter_transaction(
                    SOLANA_RPC_URL, 
                    tx_hash, 
                    SOLANA_WALLET_ADDRESS,
                    is_buy=False
                )
                
                return {
                    'exit_gas_fee_usd': fee_data.get('gas_fee_usd', 0),
                    'actual_proceeds_usd': fee_data.get('actual_proceeds_usd', 0),
                    'sell_tx_hash': tx_hash
                }
            elif chain.lower() in ["ethereum", "base", "arbitrum", "polygon"]:
                from src.utils.transaction_analyzer import analyze_sell_transaction
                from src.execution.uniswap_executor import w3
                
                fee_data = analyze_sell_transaction(w3, tx_hash)
                
                return {
                    'exit_gas_fee_usd': fee_data.get('gas_fee_usd', 0),
                    'actual_proceeds_usd': fee_data.get('actual_proceeds_usd', 0),
                    'sell_tx_hash': tx_hash
                }
        except Exception as e:
            log_error("trading.fee_analysis_error", f"Error analyzing sell transaction {tx_hash}: {e}")
            return {}
        return {}
    
    async def _process_token_batch(self, batch: List[Dict], regime_data: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """Process a batch of tokens with parallel AI analysis"""
        log_info("trading.batch", f"Processing batch of {len(batch)} tokens")
        
        # Add tokens to swap indexer for background indexing (they passed quality filters)
        try:
            from src.config.config_loader import get_config
            if get_config("swap_indexer.enabled", False):
                from src.indexing.swap_indexer import get_indexer
                indexer = get_indexer()
                if indexer.running:
                    added_count = 0
                    for token in batch:
                        address = token.get('address')
                        chain_id = token.get('chainId', '').lower()
                        if address and chain_id == 'solana':
                            indexer.add_token(address)
                            added_count += 1
                    if added_count > 0:
                        log_info("trading.indexer", f"Added {added_count} Solana tokens to swap indexer for background indexing")
        except Exception as e:
            log_error("trading.indexer", f"Failed to add tokens to indexer: {e}")
        
        # Check market regime before processing batch
        enable_regime_controls = get_config_bool("enable_regime_trading_controls", False)
        if enable_regime_controls:
            should_trade, reason = ai_market_regime_detector.should_trade_in_current_regime()
            
            if not should_trade:
                log_info("trading.regime_pause", f"Trading paused due to market regime: {reason}")
                return []  # Skip this batch
            
            # Get regime data for quality adjustments if not provided
            if regime_data is None:
                regime_data = ai_market_regime_detector.detect_market_regime()
        
        # Update price_memory for all tokens (builds price history even when analysis is skipped)
        try:
            from src.storage.price_memory import load_price_memory, save_price_memory
            import time
            mem = load_price_memory()
            now_ts = int(time.time())
            updated_count = 0
            
            for token in batch:
                address = (token.get("address") or "").lower()
                price = float(token.get("priceUsd") or 0.0)
                
                if address and price > 0:
                    mem[address] = {"price": price, "ts": now_ts}
                    updated_count += 1
            
            if updated_count > 0:
                save_price_memory(mem)
                log_info("trading.price_memory_update",
                        f"Updated price_memory for {updated_count}/{len(batch)} tokens",
                        updated_count=updated_count,
                        batch_size=len(batch))
        except Exception as e:
            log_error("trading.price_memory_error", 
                     f"Error updating price_memory: {e}")
            # Continue even if price_memory update fails (non-critical)
        
        # OPTIMIZATION: Skip analysis if at max positions (saves API calls and compute)
        try:
            from src.core.risk_manager import _open_positions_count
            from src.config.config_loader import get_config_int
            open_count = _open_positions_count()
            max_concurrent = get_config_int("max_concurrent_positions", 5)
            
            if open_count >= max_concurrent:
                log_info("trading.max_positions_skip_analysis",
                        f"Skipping analysis: {open_count} open positions >= max {max_concurrent}",
                        open_count=open_count,
                        max_positions=max_concurrent)
                # Return empty results - no analysis performed
                return []
        except Exception as e:
            log_error("trading.max_positions_check_error", 
                     f"Error checking max positions before analysis: {e}")
            # Continue with analysis if check fails (fail-safe)
        
        # OPTIMIZATION: Filter out tokens already held before AI analysis
        # This saves API calls for AI analysis and candle fetching
        original_batch_size = len(batch)
        try:
            from src.core.risk_manager import _is_token_already_held
            filtered_batch = []
            skipped_already_held = 0
            
            for token in batch:
                token_address = (token.get("address") or "").lower()
                if token_address and _is_token_already_held(token_address):
                    log_info("trading.token_already_held.skip_analysis",
                            f"Skipping AI analysis for {token.get('symbol', 'UNKNOWN')} - already held",
                            symbol=token.get("symbol"),
                            token_address=token_address[:8] + "..." if len(token_address) > 8 else token_address,
                            chain_id=token.get("chainId", "unknown"))
                    skipped_already_held += 1
                    continue
                filtered_batch.append(token)
            
            if skipped_already_held > 0:
                log_info("trading.batch.filtered",
                        f"Filtered {skipped_already_held} tokens already held from batch of {original_batch_size}",
                        skipped_count=skipped_already_held,
                        original_batch_size=original_batch_size,
                        remaining_batch_size=len(filtered_batch))
            
            batch = filtered_batch
            
            # If all tokens filtered out, return early
            if not batch:
                log_info("trading.batch.all_filtered",
                        f"All {original_batch_size} tokens in batch already held - skipping analysis",
                        original_batch_size=original_batch_size)
                return []
        except Exception as e:
            log_error("trading.token_held_filter_error",
                     f"Error filtering tokens already held: {e}",
                     error=str(e),
                     error_type=type(e).__name__)
            # Continue with original batch on error (fail-safe)
        
        # Create tasks for parallel AI analysis
        analysis_tasks = [self._analyze_token_ai(token, regime_data) for token in batch]
        analyses = await asyncio.gather(*analysis_tasks, return_exceptions=True)
        
        results = []
        for i, (token, analysis) in enumerate(zip(batch, analyses)):
            if isinstance(analysis, Exception):
                log_error("trading.analysis_failed", f"Analysis failed for {token.get('symbol', 'UNKNOWN')}: {analysis}")
                continue
            
            # Combine token data with AI analysis
            enhanced_token = {**token, "ai_analysis": analysis}
            
            # Apply trading recommendation
            recommendation = analysis.get("trading_recommendation", {})
            action = recommendation.get("action", "unknown")
            confidence = recommendation.get("confidence", 0)
            
            # Extract additional AI signals for stricter gating
            price_pred = analysis.get("price_prediction", {})
            success_prob = price_pred.get("success_probability", 0.5)
            
            # Get risk_score from risk_assessment, not price_prediction
            risk_assessment = analysis.get("risk_assessment", {})
            risk_score = risk_assessment.get("risk_score", 0.5)
            
            # Normalize quality_score to 0-1 scale (support legacy 0-100 scale)
            quality_score = analysis.get("quality_score", 0)
            if quality_score > 1.0:
                quality_score = quality_score / 100.0
            quality_score = max(0.0, min(1.0, quality_score))
            
            # Base quality threshold
            base_min_quality = get_config_float("min_quality_score", 65) / 100.0  # Convert to 0-1 scale
            
            # Apply regime adjustments if enabled
            if enable_regime_controls and regime_data:
                regime = regime_data.get("regime", "unknown")
                if regime == "high_volatility":
                    adjustment = get_config_float("regime_high_volatility_quality_adjustment", 25) / 100.0
                    base_min_quality += adjustment
                    log_info("trading.regime_adjustment",
                            f"High volatility regime: quality threshold adjusted to {base_min_quality*100:.1f}%",
                            regime=regime,
                            adjustment=adjustment*100)
                elif regime == "bear_market":
                    adjustment = get_config_float("regime_bear_quality_adjustment", 10) / 100.0
                    base_min_quality += adjustment
                    log_info("trading.regime_adjustment",
                            f"Bear market regime: quality threshold adjusted to {base_min_quality*100:.1f}%",
                            regime=regime,
                            adjustment=adjustment*100)
            
            # Hard AI thresholds for trade approval
            MIN_QUALITY_SCORE = base_min_quality
            MIN_SUCCESS_PROB = 0.60     # 60%
            MAX_RISK_SCORE = 0.50       # 50%
            
            passes_ai_filters = (
                quality_score >= MIN_QUALITY_SCORE
                and success_prob >= MIN_SUCCESS_PROB
                and risk_score <= MAX_RISK_SCORE
            )
            
            # Volume momentum check - part of AI filters (uses DexScreener volumeChange1h, no API call)
            # Check BEFORE price momentum - filters tokens with decreasing volume early
            if passes_ai_filters and get_config_bool("enable_volume_momentum_check", True):
                from src.config.config_loader import get_config_values
                config = get_config_values()
                min_volume_change = config.get("MIN_VOLUME_CHANGE_1H", 0.05)
                volume_change_1h = token.get("volumeChange1h")
                if volume_change_1h is None:
                    passes_ai_filters = False
                else:
                    try:
                        volume_change_pct = float(volume_change_1h) / 100.0 if abs(float(volume_change_1h)) > 1 else float(volume_change_1h)
                        if volume_change_pct < min_volume_change:
                            passes_ai_filters = False
                    except (ValueError, TypeError):
                        passes_ai_filters = False
            
            # Override action to "skip" if AI filters fail (makes logs clearer)
            # This ensures action in logs reflects actual execution status
            original_action = action
            if not passes_ai_filters and action in ["buy", "weak_buy", "strong_buy"]:
                action = "skip"
            
            # Log recommendation details for debugging (includes volume momentum for filter visibility)
            momentum_24h = token.get("priceChange24h", 0)
            momentum_1h = token.get("priceChange1h", 0)
            volume_24h = float(token.get("volume24h", 0))
            liquidity = float(token.get("liquidity", 0))
            volume_change_1h = token.get("volumeChange1h")
            # Compute volume momentum for logging (same logic as passes_ai_filters check)
            volume_momentum_pass = True
            volume_change_pct = None
            if volume_change_1h is not None:
                try:
                    from src.config.config_loader import get_config_values
                    _cfg = get_config_values()
                    min_vol = _cfg.get("MIN_VOLUME_CHANGE_1H", 0.05)
                    volume_change_pct = float(volume_change_1h) / 100.0 if abs(float(volume_change_1h)) > 1 else float(volume_change_1h)
                    volume_momentum_pass = volume_change_pct >= min_vol
                except (ValueError, TypeError):
                    volume_momentum_pass = False
            else:
                volume_momentum_pass = False
            log_info("trading.recommendation_check",
                    f"Token {token.get('symbol', 'UNKNOWN')}: action={action} (original={original_action}), passes_ai_filters={passes_ai_filters}",
                    symbol=token.get("symbol"),
                    action=action,
                    original_action=original_action,
                    confidence=round(confidence, 3),
                    quality_score=round(quality_score, 3),
                    success_prob=round(success_prob, 3),
                    risk_score=round(risk_score, 3),
                    momentum_24h=round(momentum_24h, 3),
                    momentum_1h=round(momentum_1h, 3),
                    volume_24h=round(volume_24h, 0),
                    liquidity=round(liquidity, 0),
                    volumeChange1h=volume_change_1h,
                    volume_change_pct=round(volume_change_pct * 100, 2) if volume_change_pct is not None else None,
                    volume_momentum_pass=volume_momentum_pass,
                    passes_ai_filters=passes_ai_filters)
            
            # Allow both "buy" and "weak_buy" actions, with different confidence thresholds
            # weak_buy uses lower confidence threshold (0.6) and will get smaller position size
            is_weak_buy = action == "weak_buy"
            is_regular_buy = action == "buy"
            confidence_threshold = 0.6 if is_weak_buy else 0.7
            
            if (is_regular_buy or is_weak_buy) and confidence > confidence_threshold and passes_ai_filters:
                # CRITICAL: Perform checks in order to save API calls:
                # 1. Hyperactivity check (no API call)
                # 2. Holder concentration check (cached RPC calls - blocks before expensive candle fetch)
                # 3. Fetch and validate 15-minute candles (expensive Helius API call)
                # 4. Buy signal check (uses candles)
                try:
                    from src.utils.market_data_fetcher import market_data_fetcher
                    from src.utils.technical_indicators import TechnicalIndicators
                    
                    # Keep original address case for validation and RPC calls (Solana addresses are case-sensitive)
                    token_address = token.get("address", "")
                    chain_id = token.get("chainId", token.get("chain", "solana")).lower()
                    
                    # Check if token is marked as hyperactive (should be blocked from trading)
                    # Use lowercase for lookup since _hyperactive_skip_tokens stores addresses in lowercase
                    from src.utils.market_data_fetcher import market_data_fetcher
                    if token_address and token_address.lower() in market_data_fetcher._hyperactive_skip_tokens:
                        log_error("trading.hyperactivity_blocked",
                                f"Token {token.get('symbol', 'UNKNOWN')} blocked for hyperactivity",
                                symbol=token.get("symbol"),
                                token_address=token_address[:8] + "..." if len(token_address) > 8 else token_address,
                                chain_id=chain_id)
                        enhanced_token["approved_for_trading"] = False
                        enhanced_token["rejection_reason"] = "hyperactivity_blocked"
                        continue  # Skip to next token
                    
                    # HARD BLOCK: Check holder concentration BEFORE candle fetch to save API calls
                    # This prevents tokens exceeding threshold from even attempting expensive candle fetches
                    if not token_address:
                        log_error("trading.holder_concentration.missing_address",
                                f"Token {token.get('symbol', 'UNKNOWN')} blocked: missing token address for holder concentration check",
                                symbol=token.get("symbol"))
                        enhanced_token["approved_for_trading"] = False
                        enhanced_token["rejection_reason"] = "missing_token_address"
                        continue  # Skip to next token
                    
                    # Validate address format before attempting RPC calls
                    if chain_id == "solana":
                        from src.utils.address_utils import is_solana_address
                        if not is_solana_address(token_address):
                            log_error("trading.holder_concentration.invalid_address",
                                    f"Token {token.get('symbol', 'UNKNOWN')} blocked: invalid Solana address format",
                                    symbol=token.get("symbol"),
                                    token_address=token_address[:8] + "..." if len(token_address) > 8 else token_address)
                            enhanced_token["approved_for_trading"] = False
                            enhanced_token["rejection_reason"] = "invalid_address_format"
                            continue  # Skip to next token
                    else:
                        # For EVM chains, validate address format
                        from src.utils.address_utils import validate_chain_address_match
                        is_valid, corrected_chain, error_message = validate_chain_address_match(token_address, chain_id)
                        if not is_valid:
                            log_error("trading.holder_concentration.invalid_address",
                                    f"Token {token.get('symbol', 'UNKNOWN')} blocked: invalid address format for {chain_id}",
                                    symbol=token.get("symbol"),
                                    token_address=token_address[:8] + "..." if len(token_address) > 8 else token_address,
                                    error=error_message)
                            enhanced_token["approved_for_trading"] = False
                            enhanced_token["rejection_reason"] = f"invalid_address_format_{chain_id}"
                            continue  # Skip to next token
                    
                    try:
                        from src.utils.holder_concentration_checker import check_holder_concentration
                        symbol = token.get("symbol", "UNKNOWN")
                        
                        if get_config_bool("enable_holder_concentration_check", True):
                            holder_check = check_holder_concentration(token_address, chain_id)
                            
                            if holder_check and not holder_check.get("error"):
                                holder_concentration_pct = holder_check.get("top_10_percentage", 100.0)
                                threshold = get_config_float("holder_concentration_threshold", 50.0)
                                
                                if holder_concentration_pct >= threshold:
                                    # HARD BLOCK: Exceeds threshold, reject immediately (saves candle fetch API call)
                                    log_error("trading.holder_concentration_blocked",
                                            f"Token {symbol} BLOCKED: holder concentration {holder_concentration_pct:.2f}% >= threshold {threshold:.2f}%",
                                            symbol=symbol,
                                            holder_concentration_pct=holder_concentration_pct,
                                            threshold=threshold)
                                    enhanced_token["approved_for_trading"] = False
                                    enhanced_token["holder_concentration_pct"] = holder_concentration_pct
                                    enhanced_token["rejection_reason"] = f"holder_concentration_exceeded_{holder_concentration_pct:.2f}_{threshold:.2f}"
                                    continue  # Skip to next token (don't fetch candles)
                                else:
                                    # Passes threshold, store for ranking later
                                    enhanced_token["holder_concentration_pct"] = holder_concentration_pct
                                    log_info("trading.holder_concentration",
                                            f"Holder concentration for {symbol}: {holder_concentration_pct:.2f}% (below threshold {threshold:.2f}%)",
                                            symbol=symbol,
                                            holder_concentration_pct=holder_concentration_pct,
                                            threshold=threshold)
                            elif get_config_bool("holder_concentration_fail_closed", True):
                                # Fail-closed: block if check fails (including account_not_found - we can't verify token safety without data)
                                error_type = holder_check.get("error_type", "unknown") if holder_check else "unknown"
                                error_msg = holder_check.get("error") if holder_check else "check_returned_none"
                                log_error("trading.holder_concentration_blocked",
                                        f"Token {symbol} BLOCKED: holder concentration check failed (fail-closed mode)",
                                        symbol=symbol,
                                        error=error_msg,
                                        error_type=error_type)
                                enhanced_token["approved_for_trading"] = False
                                enhanced_token["rejection_reason"] = f"holder_concentration_check_failed_{error_type}"
                                enhanced_token["holder_concentration_pct"] = 100.0
                                continue  # Skip to next token (don't fetch candles)
                            else:
                                # Fail-open: allow but mark as unknown
                                enhanced_token["holder_concentration_pct"] = 100.0
                                log_info("trading.holder_concentration",
                                        f"Holder concentration check failed for {symbol} (fail-open mode), allowing trade",
                                        symbol=symbol)
                        else:
                            # Check disabled, allow trade
                            enhanced_token["holder_concentration_pct"] = 100.0
                    except Exception as e:
                        # On exception, check fail-closed setting
                        if get_config_bool("holder_concentration_fail_closed", True):
                            log_error("trading.holder_concentration_blocked",
                                    f"Token {token.get('symbol', 'UNKNOWN')} BLOCKED: holder concentration check exception (fail-closed mode): {e}",
                                    symbol=token.get("symbol"),
                                    error=str(e))
                            enhanced_token["approved_for_trading"] = False
                            enhanced_token["rejection_reason"] = f"holder_concentration_exception_{str(e)}"
                            enhanced_token["holder_concentration_pct"] = 100.0
                            continue  # Skip to next token (don't fetch candles)
                        else:
                            # Fail-open: allow trade on exception
                            enhanced_token["holder_concentration_pct"] = 100.0
                            log_error("trading.holder_concentration_error",
                                    f"Holder concentration check exception for {token.get('symbol', 'UNKNOWN')} (fail-open mode): {e}")
                    
                    # Volume momentum already checked in passes_ai_filters (trading.recommendation_check)
                    
                    # ALWAYS fetch and validate candles before trade entry (required for technical checks)
                    # This ensures VWAP and other technical indicators are available
                    log_info("trading.candle_validation.start",
                            f"Fetching 15-minute candles for {token.get('symbol', 'UNKNOWN')}",
                            symbol=token.get("symbol"),
                            token_address=token_address[:8] + "...",
                            chain_id=chain_id)
                    
                    # Fetch 15-minute candles (6 hours = 24 candles)
                    # CRITICAL FIX: Run synchronous blocking function in thread pool to prevent event loop blocking
                    # This prevents all concurrent async tasks from hanging when candle fetch blocks
                    candles_15m = await asyncio.to_thread(
                        market_data_fetcher.get_candlestick_data,
                        token_address=token_address,
                        chain_id=chain_id,
                        hours=6,  # 6 hours = 24 candles (4 per hour)
                        force_fetch=False  # Use cache if available
                    )
                    
                    if not candles_15m:
                        log_error("trading.candle_validation.no_candles",
                                f"Token {token.get('symbol', 'UNKNOWN')} blocked: no candles returned",
                                symbol=token.get("symbol"),
                                token_address=token_address[:8] + "...",
                                chain_id=chain_id)
                        enhanced_token["approved_for_trading"] = False
                        enhanced_token["rejection_reason"] = "no_candles_returned"
                    else:
                        # Quality validation already happened in market_data_fetcher
                        # If candles exist, they passed lenient or strict validation
                        # However, enforce absolute minimum candle count as a safety check
                        from src.config.config_loader import get_config_int
                        absolute_min_candles = get_config_int('helius_15m_candle_policy.absolute_min_candles', 12)
                        
                        if len(candles_15m) < absolute_min_candles:
                            log_error("trading.candle_validation.insufficient_candles",
                                    f"Token {token.get('symbol', 'UNKNOWN')} blocked: insufficient candles ({len(candles_15m)} < {absolute_min_candles} minimum)",
                                    symbol=token.get("symbol"),
                                    candles_count=len(candles_15m),
                                    minimum_required=absolute_min_candles,
                                    token_address=token_address[:8] + "...")
                            enhanced_token["approved_for_trading"] = False
                            enhanced_token["rejection_reason"] = f"insufficient_candles_{len(candles_15m)}_min_{absolute_min_candles}"
                        elif len(candles_15m) < 16:
                            logger.warning(
                                f"Token {token.get('symbol', 'UNKNOWN')} has {len(candles_15m)} candles (<16 ideal), "
                                f"but passed quality validation - continuing"
                            )
                        
                        # Only proceed with OHLC validation if token wasn't explicitly rejected
                        # Check if approved_for_trading was explicitly set to False (rejection)
                        # If key doesn't exist or is True, continue with validation
                        if enhanced_token.get("approved_for_trading") is False:
                            continue
                        
                        # Validate candle quality (check OHLC integrity)
                        invalid_count = 0
                        for idx, c in enumerate(candles_15m):
                            high = c.get('high', 0)
                            low = c.get('low', 0)
                            close = c.get('close', 0)
                            open_price = c.get('open', 0)
                            
                            if high < low or close < low or close > high or open_price < low or open_price > high:
                                invalid_count += 1
                                if invalid_count <= 3:  # Log first 3 invalid candles
                                    log_error("trading.candle_validation.invalid_ohlc_detail",
                                            f"Invalid OHLC at candle {idx}: high={high:.8f}, low={low:.8f}, open={open_price:.8f}, close={close:.8f}",
                                            symbol=token.get("symbol"),
                                            candle_index=idx,
                                            high=high,
                                            low=low,
                                            open=open_price,
                                            close=close)
                        
                        if invalid_count > 0:
                            log_error("trading.candle_validation.invalid_ohlc",
                                    f"Token {token.get('symbol', 'UNKNOWN')} blocked: {invalid_count} invalid OHLC candles",
                                    symbol=token.get("symbol"),
                                    invalid_candles=invalid_count,
                                    total_candles=len(candles_15m),
                                    token_address=token_address[:8] + "...")
                            enhanced_token["approved_for_trading"] = False
                            enhanced_token["rejection_reason"] = f"invalid_ohlc_{invalid_count}"
                        else:
                            # Calculate technical indicators from candles
                            try:
                                tech_indicators = TechnicalIndicators()
                                indicators = tech_indicators.calculate_all_indicators(candles_15m, include_confidence=True)
                                
                                # Extract VWAP and other indicators
                                vwap_dict = indicators.get('vwap', None)
                                rsi = indicators.get('rsi', None)
                                
                                # Extract actual VWAP value (it's a dict with 'vwap' key)
                                vwap_value = None
                                if vwap_dict:
                                    if isinstance(vwap_dict, dict):
                                        vwap_value = vwap_dict.get('vwap')
                                    else:
                                        vwap_value = vwap_dict  # Fallback if it's already a float
                                
                                # Calculate momentum from candles (first vs last candle)
                                first_price = candles_15m[0].get('close', 0)
                                last_price = candles_15m[-1].get('close', 0)
                                candle_momentum = None
                                if first_price > 0:
                                    candle_momentum = (last_price - first_price) / first_price
                                
                                # Store in token dict for check_buy_signal to use
                                token['candles_15m'] = candles_15m
                                token['candles_validated'] = True
                                token['candle_momentum'] = candle_momentum
                                
                                if vwap_dict:
                                    token['vwap'] = vwap_dict  # Store full dict for other uses
                                if rsi:
                                    token['rsi'] = rsi
                                if indicators:
                                    token['technical_indicators'] = indicators
                                
                                # Format values safely before using in f-string to avoid format specifier errors
                                vwap_str = f"{vwap_value:.8f}" if vwap_value is not None else "N/A"
                                momentum_str = f"{candle_momentum*100:.4f}%" if candle_momentum is not None else "N/A"
                                
                                log_info("trading.candle_validation.passed",
                                        f"Token {token.get('symbol', 'UNKNOWN')} passed candle validation: {len(candles_15m)} candles, VWAP={vwap_str}, momentum={momentum_str}",
                                        symbol=token.get("symbol"),
                                        candles_count=len(candles_15m),
                                        vwap=vwap_value,  # Store the numeric value for logging
                                        candle_momentum=candle_momentum,
                                        rsi=rsi,
                                        token_address=token_address[:8] + "...")
                                
                                # Set approved_for_trading to True after successful candle validation
                                # This allows check_buy_signal to run
                                enhanced_token["approved_for_trading"] = True
                                
                            except Exception as indicator_error:
                                log_error("trading.candle_validation.indicator_error",
                                        f"Error calculating indicators for {token.get('symbol', 'UNKNOWN')}: {indicator_error}",
                                        symbol=token.get("symbol"),
                                        error=str(indicator_error),
                                        error_type=type(indicator_error).__name__,
                                        token_address=token_address[:8] + "...")
                                enhanced_token["approved_for_trading"] = False
                                enhanced_token["rejection_reason"] = f"indicator_error_{str(indicator_error)}"
                
                except Exception as candle_error:
                    log_error("trading.candle_validation.error",
                             f"Error fetching/validating candles for {token.get('symbol', 'UNKNOWN')}: {candle_error}",
                             symbol=token.get("symbol"),
                             error=str(candle_error),
                             error_type=type(candle_error).__name__,
                             token_address=token.get("address", "")[:8] + "..." if token.get("address") else "N/A")
                    enhanced_token["approved_for_trading"] = False
                    enhanced_token["rejection_reason"] = f"candle_fetch_error_{type(candle_error).__name__}"
                
                # Only proceed with buy signal check if candles validated successfully
                if not enhanced_token.get("approved_for_trading", False):
                    # Already blocked above, skip check_buy_signal
                    continue
                
                # CRITICAL: Check buy signal with momentum requirements BEFORE approval
                from src.core.strategy import check_buy_signal
                try:
                    log_info("trading.debug.check_buy_signal_start",
                            f"🔍 Calling check_buy_signal for {token.get('symbol', 'UNKNOWN')}",
                            symbol=token.get("symbol"),
                            token_address=token.get("address", "")[:8] + "..." if token.get("address") else None,
                            volume_24h=token.get("volume24h"),
                            liquidity=token.get("liquidity"),
                            price=token.get("priceUsd"),
                            chain_id=token.get("chainId", "unknown"))
                    buy_signal_passed = check_buy_signal(token)
                    log_info("trading.debug.check_buy_signal_result",
                            f"🔍 check_buy_signal returned: {buy_signal_passed} for {token.get('symbol', 'UNKNOWN')}",
                            symbol=token.get("symbol"),
                            buy_signal_passed=buy_signal_passed)
                except Exception as e:
                    log_error("trading.check_buy_signal_error",
                            f"Exception in check_buy_signal for {token.get('symbol', 'UNKNOWN')}: {e}",
                            symbol=token.get("symbol"),
                            error=str(e),
                            error_type=type(e).__name__,
                            token_address=token.get("address", "")[:8] + "..." if token.get("address") else None)
                    buy_signal_passed = False
                
                if not buy_signal_passed:
                    log_info("trading.buy_signal_blocked",
                            f"Token {token.get('symbol', 'UNKNOWN')} blocked by check_buy_signal (momentum/strategy requirements not met)",
                            symbol=token.get("symbol"),
                            action=action,
                            confidence=confidence,
                            passes_ai_filters=passes_ai_filters)
                    enhanced_token["approved_for_trading"] = False
                    enhanced_token["rejection_reason"] = "buy_signal_failed"
                else:
                    enhanced_token["approved_for_trading"] = True
                
                # Only proceed with position sizing if still approved after all checks
                if enhanced_token.get("approved_for_trading", False):
                    # Get tier-based limits FIRST
                    try:
                        from src.core.risk_manager import get_tier_based_risk_limits
                        tier_limits = get_tier_based_risk_limits()
                        tier_base_size = tier_limits.get("BASE_POSITION_SIZE_USD", 5.0)
                        tier_max_size = tier_limits.get("PER_TRADE_MAX_USD", 10.0)  # Fallback to reasonable default
                    except Exception as e:
                        log_error("trading.tier_limits_error", 
                                f"Error getting tier limits: {e}")
                        tier_base_size = 5.0
                        tier_max_size = 25.0
                    
                    # Get AI-recommended position size
                    ai_recommended_size = recommendation.get("position_size", tier_base_size)
                    
                    # Scale AI recommendation to work within tier range (base to max)
                    # Old AI system: 5.0 base, recommendations were 10.0 (2x), 15.0 (3x), 20.0 (4x)
                    # New system: Map AI recommendations to tier range proportionally
                    OLD_AI_BASE = 5.0
                    OLD_AI_MAX = 20.0  # Strong buy was 4x base
                    tier_range = tier_max_size - tier_base_size
                    
                    if ai_recommended_size > 0:
                        # Map old AI range (5.0-20.0) to tier range (base-max)
                        # Formula: tier_base + (ai_size - old_base) / (old_max - old_base) * tier_range
                        if ai_recommended_size >= OLD_AI_MAX:
                            # Strong buy: use tier max
                            scaled_size = tier_max_size
                        elif ai_recommended_size <= OLD_AI_BASE:
                            # Base or below: use tier base
                            scaled_size = tier_base_size
                        else:
                            # Scale proportionally within tier range
                            old_range = OLD_AI_MAX - OLD_AI_BASE
                            proportion = (ai_recommended_size - OLD_AI_BASE) / old_range
                            scaled_size = tier_base_size + (proportion * tier_range)
                    else:
                        scaled_size = tier_base_size
                    
                    # OPTIMIZATION #3: Quality-based position sizing multiplier
                    # Higher quality tokens get larger position sizes (within tier limits)
                    # FIX: Use 0-1 scale (0.91 = 91%, 0.81 = 81%, etc.)
                    quality_multiplier = 1.0
                    if quality_score >= 0.91:  # 91% = 0.91
                        quality_multiplier = 2.0  # Excellent quality: 100% larger
                    elif quality_score >= 0.81:  # 81% = 0.81
                        quality_multiplier = 1.5  # Very high quality: 50% larger
                    elif quality_score >= 0.71:  # 71% = 0.71
                        quality_multiplier = 1.2  # High quality: 20% larger
                    # Quality 0.65-0.70: 1.0x (base size)
                    
                    # Apply quality multiplier to scaled size
                    quality_adjusted_size = scaled_size * quality_multiplier
                    
                    # UPGRADE #3: Regime-based position sizing throttle
                    regime_throttle_enabled = get_config_bool("regime_sideways_throttle", True)
                    if regime_throttle_enabled and regime_data:
                        regime = regime_data.get("regime", "unknown")
                        if regime in ['sideways_market', 'recovery_market']:
                            regime_multiplier = get_config_float("regime_throttle_position_multiplier", 0.7)
                            size_before_regime = quality_adjusted_size
                            quality_adjusted_size = quality_adjusted_size * regime_multiplier
                            log_info("trading.regime_throttle",
                                    f"Regime throttle applied for {token.get('symbol', 'UNKNOWN')}: "
                                    f"${size_before_regime:.2f} × {regime_multiplier:.2f} ({regime} regime) = ${quality_adjusted_size:.2f}",
                                    symbol=token.get("symbol"),
                                    regime=regime,
                                    regime_multiplier=regime_multiplier,
                                    size_before=round(size_before_regime, 3),
                                    size_after=round(quality_adjusted_size, 3))
                    
                    # Apply weak_buy position size reduction (50% of regular buy size)
                    # Store action type for position sizing adjustment
                    action_type = action  # This is either "buy" or "weak_buy" at this point
                    if action_type == "weak_buy":
                        # Reduce position size to 50% for weak_buy signals
                        weak_buy_multiplier = 0.5
                        size_before_weak_buy = quality_adjusted_size
                        quality_adjusted_size = quality_adjusted_size * weak_buy_multiplier
                        log_info("trading.weak_buy_sizing",
                                f"Weak buy position size reduction for {token.get('symbol', 'UNKNOWN')}: "
                                f"${size_before_weak_buy:.2f} × {weak_buy_multiplier:.2f} = ${quality_adjusted_size:.2f}",
                                symbol=token.get("symbol"),
                                weak_buy_multiplier=weak_buy_multiplier,
                                size_before_reduction=round(size_before_weak_buy, 3),
                                adjusted_size=round(quality_adjusted_size, 3))
                    
                    # Apply entry lane multiplier (early_scout = 0.25x, confirm_add = 1.0x)
                    # This is applied after quality/regime adjustments but before tier bounds
                    entry_lane = token.get('entry_lane')
                    entry_lane_multiplier = token.get('entry_lane_multiplier', 1.0)
                    if entry_lane and entry_lane_multiplier != 1.0:
                        size_before_lane = quality_adjusted_size
                        quality_adjusted_size = quality_adjusted_size * entry_lane_multiplier
                        log_info("trading.entry_lane_sizing",
                                f"Entry lane '{entry_lane}' position size adjustment for {token.get('symbol', 'UNKNOWN')}: "
                                f"${size_before_lane:.2f} × {entry_lane_multiplier:.2f} = ${quality_adjusted_size:.2f}",
                                symbol=token.get("symbol"),
                                entry_lane=entry_lane,
                                entry_lane_multiplier=entry_lane_multiplier,
                                size_before_lane=round(size_before_lane, 3),
                                adjusted_size=round(quality_adjusted_size, 3))
                    
                    # Ensure position size is within tier bounds (safety check)
                    final_position_size = max(tier_base_size, min(quality_adjusted_size, tier_max_size))
                    
                    # Log quality-based adjustment if it made a difference
                    if abs(final_position_size - scaled_size) > 0.01:
                        log_info("trading.quality_sizing",
                                f"Quality-based sizing for {token.get('symbol', 'UNKNOWN')}: "
                                f"base=${scaled_size:.2f} × {quality_multiplier:.2f} (quality={quality_score:.1f}) = ${final_position_size:.2f}",
                                symbol=token.get("symbol"),
                                quality_score=round(quality_score, 3),
                                quality_multiplier=round(quality_multiplier, 3),
                                base_size=round(scaled_size, 3),
                                final_size=round(final_position_size, 3))
                    
                    # Log the scaling
                    if abs(final_position_size - ai_recommended_size) > 0.01:
                        log_info("trading.position_size_scaled",
                                f"Position size scaled for {token.get('symbol', 'UNKNOWN')}: "
                                f"AI: ${ai_recommended_size:.2f} → Tier-scaled: ${scaled_size:.2f} → Final: ${final_position_size:.2f} "
                                f"(tier base: ${tier_base_size:.2f}, tier max: ${tier_max_size:.2f})",
                                symbol=token.get("symbol"),
                                ai_recommended=ai_recommended_size,
                                scaled=scaled_size,
                                final=final_position_size,
                                tier_base=tier_base_size,
                                tier_max=tier_max_size)
                    
                    enhanced_token["recommended_position_size"] = final_position_size
                    
                    enhanced_token["recommended_tp"] = recommendation.get("take_profit", 0.15)
            else:
                enhanced_token["approved_for_trading"] = False
            
            results.append(enhanced_token)
        
        log_info("trading.batch", f"Batch processing complete: {len(results)} tokens analyzed")
        return results
    
    async def _execute_trade_async(self, token: Dict) -> Dict[str, Any]:
        """Execute a single trade asynchronously"""
        symbol = token.get("symbol", "UNKNOWN")
        address = token.get("address", "")
        # Default to first supported chain if chain is not specified
        supported_chains = getattr(self.config.chains, 'supported_chains', ['solana', 'base'])
        default_chain = supported_chains[0] if supported_chains else "solana"
        chain = token.get("chain", default_chain)
        # Get tier base size as fallback (position sizing is tier-based, not fixed)
        try:
            from src.core.risk_manager import get_tier_based_risk_limits
            tier_limits = get_tier_based_risk_limits()
            tier_base_fallback = tier_limits.get("BASE_POSITION_SIZE_USD", 5.0)
        except Exception:
            tier_base_fallback = 5.0
        position_size = token.get("recommended_position_size", tier_base_fallback)
        take_profit = token.get("recommended_tp", 0.15)
        
        # Safety cap: Ensure position size is within tier-based bounds
        # (This is a safety measure in case position size wasn't adjusted earlier)
        try:
            from src.core.risk_manager import get_tier_based_risk_limits
            tier_limits = get_tier_based_risk_limits()
            tier_base_size = tier_limits.get("BASE_POSITION_SIZE_USD", 5.0)
            tier_max_size = tier_limits.get("PER_TRADE_MAX_USD", 10.0)  # Fallback to reasonable default
            
            # Ensure position size is within tier bounds
            if position_size < tier_base_size:
                log_info("trading.position_size_below_base",
                        f"Position size below tier base for {symbol}: ${position_size:.2f} → ${tier_base_size:.2f}",
                        symbol=symbol,
                        original_size=position_size,
                        adjusted_size=tier_base_size)
                position_size = tier_base_size
            elif position_size > tier_max_size:
                log_info("trading.position_size_above_max",
                        f"Position size above tier max for {symbol}: ${position_size:.2f} → ${tier_max_size:.2f}",
                        symbol=symbol,
                        original_size=position_size,
                        adjusted_size=tier_max_size)
                position_size = tier_max_size
        except Exception as e:
            log_error("trading.position_size_safety_cap_error", 
                    f"Error applying tier bounds for {symbol}: {e}")
            # Continue with original position size if tier check fails
        
        # DEBUG: Log that we're entering the trade execution
        log_info("trading.debug", f"🔍 DEBUG: Entering _execute_trade_async for {symbol} on {chain}")
        
        # Get action signal from AI analysis
        action = token.get("ai_analysis", {}).get("recommendations", {}).get("action", "hold")
        
        async with self.rate_limiter:
            start_time = time.time()
            
            try:
                log_info("trading.execute", f"Executing trade for {symbol} on {chain}")
                
                # Check if we're adding to an existing position
                is_adding_to_position = False
                additional_amount = position_size
                current_position_size = 0.0
                existing_entry_price = 0.0
                
                # In-memory duplicate buy guard (fast check before file I/O)
                # Prevents buying same token twice if position save was delayed
                address_lower = (address or "").lower().strip()
                if address_lower and address_lower in self._recently_bought_tokens:
                    bought_at = self._recently_bought_tokens[address_lower]
                    elapsed = time.time() - bought_at
                    # Guard active for 10 minutes (enough time for position to be saved to file)
                    if elapsed < 600:
                        log_warning("trading.duplicate_buy_guard",
                                   f"In-memory duplicate guard blocked {symbol} - "
                                   f"already bought {elapsed:.0f}s ago",
                                   symbol=symbol, elapsed_seconds=elapsed)
                        return {
                            "success": False,
                            "symbol": symbol,
                            "error": f"Duplicate buy guard: {symbol} bought {elapsed:.0f}s ago",
                            "error_type": "gate",
                            "chain": chain
                        }
                    else:
                        # Expired - remove from guard
                        del self._recently_bought_tokens[address_lower]
                
                # Pre-trade wallet/limits gate - check balance and position limits
                try:
                    from src.core.risk_manager import allow_new_trade
                    from src.storage.positions import load_positions as load_positions_store
                    from src.utils.position_sync import resolve_token_address, create_position_key
                    
                    # Check if token is already held and we should add to it
                    allowed, reason, is_add_to_pos, add_amount = allow_new_trade(
                        position_size, 
                        token_address=address, 
                        chain_id=chain,
                        recommended_position_size=position_size,
                        signal=action
                    )
                    
                    if is_add_to_pos:
                        is_adding_to_position = True
                        additional_amount = add_amount
                        
                        # Get current position details
                        positions = load_positions_store()
                        position_key = create_position_key(address)
                        existing_position = positions.get(position_key, {})
                        current_position_size = float(existing_position.get("position_size_usd", 0.0) or 0.0)
                        existing_entry_price = float(existing_position.get("entry_price", 0.0) or 0.0)
                        
                        log_info("trading.add_to_position",
                                f"Adding to existing position for {symbol}: "
                                f"current=${current_position_size:.2f}, additional=${additional_amount:.2f}, "
                                f"target=${position_size:.2f}",
                                symbol=symbol,
                                current_size=current_position_size,
                                additional=additional_amount,
                                target_size=position_size)
                    elif not allowed:
                        log_error("trading.risk_gate_blocked",
                                  symbol=symbol, chain=chain, amount_usd=position_size, reason=reason)
                        return {
                            "success": False,
                            "symbol": symbol,
                            "error": f"Risk gate blocked: {reason}",
                            "error_type": "gate",  # Mark as gate failure
                            "chain": chain
                        }
                except Exception as e:
                    log_error("trading.risk_gate_error", f"Risk gate error: {e}")
                    # BLOCK trade if risk gate check fails - fail-safe approach
                    return {
                        "success": False,
                        "symbol": symbol,
                        "error": f"Risk gate check failed: {e}",
                        "error_type": "gate",  # Mark as gate failure
                        "chain": chain
                    }
                
                # Risk assessment
                trade_amount = additional_amount if is_adding_to_position else position_size
                log_info("trading.risk_assessment_start", f"Starting risk assessment for {symbol}")
                risk_result = await assess_trade_risk(token, trade_amount)
                log_info("trading.risk_assessment_complete", 
                        f"Risk assessment complete for {symbol}: approved={risk_result.approved}, "
                        f"risk_score={risk_result.overall_risk_score:.2f}, reason={risk_result.reason}")
                if not risk_result.approved:
                    log_error("trading.risk_assessment_blocked",
                             f"Risk assessment blocked trade for {symbol}: {risk_result.reason}",
                             symbol=symbol,
                             risk_score=risk_result.overall_risk_score,
                             reason=risk_result.reason,
                             risk_level=risk_result.risk_level.value if hasattr(risk_result.risk_level, 'value') else str(risk_result.risk_level))
                    return {
                        "success": False,
                        "error": f"Risk assessment failed: {risk_result.reason}",
                        "error_type": "gate",  # Mark as gate failure
                        "symbol": symbol
                    }
                
                # Execute real trade using DEX integrations
                trade_amount = additional_amount if is_adding_to_position else position_size
                log_info("trading.trade_execution_start", 
                        f"Starting trade execution for {symbol} on {chain} "
                        f"({'adding to position' if is_adding_to_position else 'new position'})")
                trade_result = await self._execute_real_trade(token, trade_amount, chain)
                
                # Log the trade result for debugging
                log_info("trading.debug", f"Trade result for {symbol}: {trade_result}")
                
                # Handle gate failures early - don't count them as trade attempts
                if not trade_result.get("success", False) and trade_result.get("error_type") == "gate":
                    error_msg = trade_result.get("error", "Gate failure")
                    log_info("trading.gate_blocked", f"🚫 Gate blocked trade for {symbol}: {error_msg} (not counting as failure)")
                    return {
                        "success": False,
                        "symbol": symbol,
                        "error": error_msg,
                        "error_type": "gate",
                        "chain": chain
                    }
                
                if trade_result.get("success", False):
                    # Successful real trade
                    profit_loss = trade_result.get("profit_loss", 0)
                    tx_hash = trade_result.get("tx_hash", "")
                    execution_time = (time.time() - start_time) * 1000
                    
                    # Register in in-memory duplicate guard IMMEDIATELY (before any I/O)
                    if address_lower and not is_adding_to_position:
                        self._recently_bought_tokens[address_lower] = time.time()
                    
                    log_trade("buy", symbol, trade_amount, True, profit_loss, execution_time)
                    log_info("trading.success", f"✅ Real trade successful: {symbol} - PnL: ${profit_loss:.2f} - TX: {tx_hash}")
                    
                    # Add token to swap indexer for continuous tracking
                    try:
                        from src.config.config_loader import get_config
                        if get_config("swap_indexer.enabled", False):
                            from src.indexing.swap_indexer import get_indexer
                            indexer = get_indexer()
                            if address:
                                indexer.add_token(address)
                                log_info("trading.indexer", f"Added {symbol} to swap indexer")
                    except Exception as e:
                        log_error("trading.indexer", f"Failed to add token to indexer: {e}")
                    
                    # Register buy with risk manager
                    try:
                        from src.core.risk_manager import register_buy
                        register_buy(trade_amount)
                    except Exception as e:
                        log_error("trading.register_buy_error", f"Failed to register buy: {e}")
                    
                    # Update position if adding to existing position
                    if is_adding_to_position:
                        try:
                            from src.storage.positions import upsert_position, load_positions as load_positions_store
                            from src.utils.position_sync import create_position_key
                            
                            # Get current price for weighted average calculation
                            current_price = float(token.get("priceUsd") or 0.0)
                            if current_price <= 0:
                                # Fallback: try to fetch price from trade result or use entry price
                                current_price = float(trade_result.get("price", 0.0) or 0.0)
                                if current_price <= 0:
                                    # Use existing entry price as fallback
                                    current_price = existing_entry_price
                            
                            # Calculate weighted average entry price
                            # Formula: (old_size * old_price + new_size * new_price) / (old_size + new_size)
                            new_position_size = current_position_size + additional_amount
                            if new_position_size > 0 and current_price > 0:
                                weighted_entry_price = (
                                    (current_position_size * existing_entry_price) + 
                                    (additional_amount * current_price)
                                ) / new_position_size
                            else:
                                weighted_entry_price = existing_entry_price if existing_entry_price > 0 else current_price
                            
                            # Update position
                            position_key = create_position_key(address)
                            positions = load_positions_store()
                            existing_position = positions.get(position_key, {})
                            
                            # Update position data
                            existing_position["position_size_usd"] = new_position_size
                            existing_position["entry_price"] = weighted_entry_price
                            existing_position["last_added_at"] = datetime.now().isoformat()
                            existing_position["additions_count"] = existing_position.get("additions_count", 0) + 1
                            
                            # Preserve other fields
                            if "symbol" not in existing_position:
                                existing_position["symbol"] = symbol
                            if "chain_id" not in existing_position:
                                existing_position["chain_id"] = chain
                            if "address" not in existing_position:
                                existing_position["address"] = address
                            
                            upsert_position(position_key, existing_position)
                            
                            log_info("trading.position_updated",
                                    f"Updated position for {symbol}: "
                                    f"size=${new_position_size:.2f} (was ${current_position_size:.2f}), "
                                    f"entry_price=${weighted_entry_price:.6f} (was ${existing_entry_price:.6f})",
                                    symbol=symbol,
                                    old_size=current_position_size,
                                    new_size=new_position_size,
                                    old_entry_price=existing_entry_price,
                                    new_entry_price=weighted_entry_price)
                            
                            # Update position_size for logging below
                            position_size = new_position_size
                        except Exception as e:
                            log_error("trading.position_update_error", 
                                    f"Failed to update position after adding: {e}")
                            # Don't fail the trade if position update fails
                    
                    # Log trade entry to performance tracker for status reports and analytics
                    # CRITICAL FIX: Verify execution succeeded before logging
                    trade_id = None
                    try:
                        from src.core.performance_tracker import performance_tracker
                        quality_score = token.get('ai_analysis', {}).get('quality_score', 0.0)
                        # Prepare token data with all required fields for performance tracker
                        pt_token = {
                            "address": address,
                            "symbol": symbol,
                            "chainId": chain,
                            "priceUsd": float(token.get("priceUsd") or 0.0),
                            "volume24h": float(token.get("volume24h", 0)),
                            "liquidity": float(token.get("liquidity", 0))
                        }
                        # Include fee data if available
                        fee_data = trade_result.get('fee_data', {})
                        
                        # Include window_score if available from trade result or token
                        window_score = trade_result.get('window_score') or token.get('_window_score')
                        if window_score is not None:
                            fee_data['window_score'] = window_score
                        
                        # Verify we have actual execution data before logging
                        entry_amount_actual = fee_data.get('entry_amount_usd_actual', 0) or 0
                        tokens_received = fee_data.get('entry_tokens_received')
                        
                        # Only log if we have confirmed execution
                        # For adding to position, don't create a new trade entry (position already exists)
                        if entry_amount_actual > 0 or (tokens_received is not None and tokens_received > 0):
                            if not is_adding_to_position:
                                trade_id = performance_tracker.log_trade_entry(pt_token, trade_amount, quality_score, additional_data=fee_data)
                                log_info("trading.performance_logged", 
                                       f"✅ Trade entry logged to performance tracker for {symbol} "
                                       f"(entry_amount_usd_actual=${entry_amount_actual:.2f}, tokens_received={tokens_received})")
                            else:
                                log_info("trading.position_added",
                                       f"✅ Added to existing position for {symbol} "
                                       f"(additional_amount=${entry_amount_actual:.2f}, tokens_received={tokens_received})")
                        else:
                            # Transaction analysis may have failed - retry analysis
                            log_info("trading.execution_unverified", 
                                      f"⚠️ Trade execution for {symbol} not verified - retrying transaction analysis")
                            
                            # Retry transaction analysis
                            if tx_hash:
                                try:
                                    retry_fee_data = await self._analyze_buy_transaction(tx_hash, chain, trade_result.get('dex', 'jupiter'), None)
                                    entry_amount_actual = retry_fee_data.get('entry_amount_usd_actual', 0) or 0
                                    tokens_received = retry_fee_data.get('entry_tokens_received')
                                    
                                    if entry_amount_actual > 0 or (tokens_received is not None and tokens_received > 0):
                                        # Merge retry data with original fee_data
                                        fee_data.update(retry_fee_data)
                                        if not is_adding_to_position:
                                            trade_id = performance_tracker.log_trade_entry(pt_token, trade_amount, quality_score, additional_data=fee_data)
                                            log_info("trading.performance_logged_retry", 
                                                   f"✅ Trade entry logged after retry for {symbol} "
                                                   f"(entry_amount_usd_actual=${entry_amount_actual:.2f})")
                                        else:
                                            log_info("trading.position_added_retry",
                                                   f"✅ Added to existing position after retry for {symbol} "
                                                   f"(additional_amount=${entry_amount_actual:.2f})")
                                    else:
                                        log_error("trading.execution_failed", 
                                                f"❌ Trade execution for {symbol} failed - no tokens received. TX: {tx_hash}")
                                except Exception as retry_error:
                                    log_error("trading.retry_analysis_failed", 
                                            f"Failed to retry transaction analysis for {symbol}: {retry_error}")
                    except Exception as e:
                        log_error("trading.performance_log_error", f"Failed to log trade entry for {symbol}: {e}")
                    
                    # Log position to open_positions.json and launch monitor
                    # CRITICAL: This must succeed - retry if needed
                    position_logged = False
                    max_retries = 3
                    # Only log position if it's a new position (not adding to existing)
                    # When adding to position, it's already updated above via upsert_position
                    if not is_adding_to_position:
                        position_token = {
                            "address": address,
                            "priceUsd": float(token.get("priceUsd") or 0.0),
                            "chainId": chain,
                            "symbol": symbol,
                            "position_size_usd": position_size,  # Intended size
                            "intended_position_size_usd": position_size,  # Track intended for partial fill detection
                        }
                        # UPGRADE #1: Pass momentum and VWAP data for entry snapshot storage
                        # Copy momentum/VWAP data from token dict if available
                        if token.get('candle_momentum') is not None:
                            position_token['candle_momentum'] = token.get('candle_momentum')
                        if token.get('candles_15m'):
                            position_token['candles_15m'] = token.get('candles_15m')
                            position_token['candles_validated'] = token.get('candles_validated', False)
                        if token.get('technical_indicators'):
                            position_token['technical_indicators'] = token.get('technical_indicators')
                        if token.get('vwap'):
                            position_token['vwap'] = token.get('vwap')
                        # Copy DexScreener momentum data
                        for key in ['priceChange5m', 'priceChange1h', 'priceChange24h', 'momentum_5m', 'momentum_1h', 'momentum_24h']:
                            if key in token:
                                position_token[key] = token[key]
                        try:
                            from src.execution.multi_chain_executor import _log_position, _launch_monitor_detached
                            for attempt in range(max_retries):
                                try:
                                    _log_position(position_token, trade_id=trade_id)
                                    _launch_monitor_detached()
                                    position_logged = True
                                    log_info("trading.position_logged", f"✅ Position logged for {symbol} on {chain}")
                                    break  # Success - exit retry loop
                                except Exception as e:
                                    log_error(
                                        "trading.position_log_error",
                                        f"Failed to log position for {symbol} (attempt {attempt + 1}/{max_retries}): {e}"
                                    )
                                    if attempt < max_retries - 1:
                                        time.sleep(0.5)  # Brief delay before retry
                        except Exception as import_error:
                            log_error(
                                "trading.position_log_error",
                                f"Failed to import position logger for {symbol}: {import_error}"
                            )
                    else:
                        # Position already updated via upsert_position above, mark as logged
                        position_logged = True
                        # Launch monitor to ensure it's running
                        try:
                            from src.execution.multi_chain_executor import _launch_monitor_detached
                            _launch_monitor_detached()
                        except Exception as e:
                            log_error("trading.monitor_launch_error", f"Failed to launch monitor: {e}")
                    
                    # If position logging still failed after retries, we'll sync from performance_data
                    # after it's logged there (see below in performance tracker section)
                    position_needs_sync = (not position_logged) or (trade_id is None)
                    
                    # Record metrics
                    record_trade_metrics(
                        symbol=symbol,
                        chain=chain,
                        amount_usd=trade_amount,
                        success=True,
                        execution_time_ms=execution_time,
                        profit_loss_usd=profit_loss,
                        quality_score=token.get('ai_analysis', {}).get('quality_score', 0.5),
                        risk_score=risk_result.overall_risk_score
                    )
                    update_trade_result(True, profit_loss)
                    
                    return {
                        "success": True,
                        "symbol": symbol,
                        "position_size": position_size,
                        "profit_loss": profit_loss,
                        "execution_time": execution_time,
                        "tx_hash": tx_hash,
                        "chain": chain,
                        "is_add_to_position": is_adding_to_position,
                        "trade_amount": trade_amount
                    }
                else:
                    # Failed real trade
                    error_msg = trade_result.get("error", "Unknown error")
                    error_type = trade_result.get("error_type") or self._categorize_error(error_msg, token)[0]
                    token_address = token.get("address", "")
                    execution_time = (time.time() - start_time) * 1000
                    
                    log_trade("buy", symbol, position_size, False, 0.0, execution_time, error_msg)
                    log_error("trading.trade_failed", f"❌ Real trade failed: {symbol} - {error_msg} (type: {error_type})")
                    
                    # Record metrics
                    record_trade_metrics(
                        symbol=symbol,
                        chain=chain,
                        amount_usd=position_size,
                        success=False,
                        execution_time_ms=execution_time,
                        profit_loss_usd=0,
                        quality_score=token.get('ai_analysis', {}).get('quality_score', 0.5),
                        risk_score=risk_result.overall_risk_score,
                        error_message=error_msg
                    )
                    # Pass error type and token address for proper handling
                    update_trade_result(False, 0, error_type=error_type, token_address=token_address)
                    
                    return {
                        "success": False,
                        "symbol": symbol,
                        "error": error_msg,
                        "error_type": error_type,
                        "execution_time": execution_time,
                        "chain": chain
                    }
                    
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                error_msg = str(e)
                error_type, token_address = self._categorize_error(error_msg, token)
                log_error("trading.execution_error", f"Trade execution error for {symbol}: {e} (type: {error_type})")
                
                # Record error metrics
                record_trade_metrics(
                    symbol=symbol,
                    chain=chain,
                    amount_usd=position_size,
                    success=False,
                    execution_time_ms=execution_time,
                    profit_loss_usd=0,
                    quality_score=0,
                    risk_score=1.0,
                    error_message=error_msg
                )
                # Pass error type and token address for proper handling
                update_trade_result(False, 0, error_type=error_type, token_address=token_address)
                
                return {
                    "success": False,
                    "symbol": symbol,
                    "error": error_msg,
                    "error_type": error_type,
                    "execution_time": execution_time,
                    "chain": chain
                }
    
    def _get_recent_fill_rate(self) -> float:
        """Get recent fill success rate from execution history with time-based weighting"""
        if len(self.performance_window) == 0:
            return 0.85  # Default assumption
        
        current_time = time.time()
        recent = list(self.performance_window)[-20:]  # Last 20 trades
        
        if not recent:
            return 0.85
        
        # Time-based weighting: more recent trades count more
        # Half-life of 15 minutes (900 seconds) - trades older than 15 min have <50% weight
        total_weight = 0.0
        weighted_successes = 0.0
        
        for trade in recent:
            trade_time = trade.get("timestamp", current_time)
            age_seconds = current_time - trade_time
            
            # Exponential decay: weight = e^(-age / half_life)
            # For 15 min half-life, weight drops to 50% after 15 min, 25% after 30 min
            half_life_seconds = 900  # 15 minutes
            weight = math.exp(-age_seconds / half_life_seconds)
            
            if trade.get("success", False):
                weighted_successes += weight
            total_weight += weight
        
        if total_weight > 0:
            return weighted_successes / total_weight
        return 0.85
    
    def _get_recent_avg_slippage(self) -> float:
        """Get recent average slippage with time-based weighting"""
        if len(self.performance_window) == 0:
            return 0.02  # Default 2%
        
        current_time = time.time()
        recent = [t for t in list(self.performance_window)[-20:] if t.get("slippage") is not None]
        
        if not recent:
            return 0.02
        
        # Time-based weighting: more recent trades count more
        half_life_seconds = 900  # 15 minutes
        total_weight = 0.0
        weighted_slippage = 0.0
        
        for trade in recent:
            trade_time = trade.get("timestamp", current_time)
            age_seconds = current_time - trade_time
            weight = math.exp(-age_seconds / half_life_seconds)
            
            weighted_slippage += trade.get("slippage", 0.02) * weight
            total_weight += weight
        
        if total_weight > 0:
            return weighted_slippage / total_weight
        return 0.02
    
    def _calculate_adaptive_wait_time(self, cycle_result: Dict[str, Any]) -> int:
        """
        Calculate adaptive wait time between cycles based on market conditions.
        
        Shorter waits when:
        - Many tokens approved (active market)
        - Trades executed (opportunities found)
        - High volatility detected
        
        Longer waits when:
        - No tokens approved (quiet market)
        - No trades executed (no opportunities)
        - Low activity
        """
        base_wait = get_config_int("trading.cycle_wait_base_seconds", 300)  # Default 5 minutes
        min_wait = get_config_int("trading.cycle_wait_min_seconds", 120)  # Minimum 2 minutes
        max_wait = get_config_int("trading.cycle_wait_max_seconds", 600)  # Maximum 10 minutes
        
        tokens_approved = cycle_result.get("tokens_approved", 0)
        trades_executed = cycle_result.get("trades_executed", 0)
        tokens_filtered_early = cycle_result.get("tokens_filtered_early", 0)
        
        # Calculate wait time adjustment
        wait_time = base_wait
        
        # Reduce wait if many tokens approved (active market)
        if tokens_approved >= 5:
            wait_time = max(min_wait, wait_time - 120)  # Reduce by 2 minutes
        elif tokens_approved >= 3:
            wait_time = max(min_wait, wait_time - 60)  # Reduce by 1 minute
        
        # Reduce wait if trades were executed (opportunities found)
        if trades_executed > 0:
            wait_time = max(min_wait, wait_time - 60)  # Reduce by 1 minute
        
        # Increase wait if no tokens approved and many filtered early (quiet market)
        if tokens_approved == 0 and tokens_filtered_early > 20:
            wait_time = min(max_wait, wait_time + 120)  # Increase by 2 minutes
        
        # Ensure wait time is within bounds
        wait_time = max(min_wait, min(wait_time, max_wait))
        
        return int(wait_time)
    
    def _get_recent_avg_latency(self) -> float:
        """Get recent average execution latency in ms with time-based weighting"""
        if len(self.performance_window) == 0:
            return 2000.0  # Default 2s
        
        current_time = time.time()
        recent = [t for t in list(self.performance_window)[-20:] if t.get("latency_ms") is not None]
        
        if not recent:
            return 2000.0
        
        # Time-based weighting: more recent trades count more
        half_life_seconds = 900  # 15 minutes
        total_weight = 0.0
        weighted_latency = 0.0
        
        for trade in recent:
            trade_time = trade.get("timestamp", current_time)
            age_seconds = current_time - trade_time
            weight = math.exp(-age_seconds / half_life_seconds)
            
            weighted_latency += trade.get("latency_ms", 2000.0) * weight
            total_weight += weight
        
        if total_weight > 0:
            return weighted_latency / total_weight
        return 2000.0
    
    async def _update_metrics(self, trade_result: Dict[str, Any]):
        """Update trading metrics with new trade result"""
        self.metrics.total_trades += 1
        
        if trade_result.get("success", False):
            self.metrics.successful_trades += 1
            self.metrics.total_pnl += trade_result.get("profit_loss", 0)
        else:
            self.metrics.failed_trades += 1
            self.metrics.total_pnl += trade_result.get("loss", 0)
        
        # Update averages
        if self.metrics.total_trades > 0:
            self.metrics.success_rate = self.metrics.successful_trades / self.metrics.total_trades
            self.metrics.avg_execution_time = (
                (self.metrics.avg_execution_time * (self.metrics.total_trades - 1) + 
                 trade_result.get("execution_time", 0)) / self.metrics.total_trades
            )
        
        # Calculate trades per hour (last hour)
        current_time = time.time()
        one_hour_ago = current_time - 3600
        recent_trades = [t for t in self.performance_window if t.get("timestamp", 0) > one_hour_ago]
        self.metrics.trades_per_hour = len(recent_trades)
        
        # Calculate health score
        self.metrics.health_score = min(100, max(0, 
            int(100 * self.metrics.success_rate - 
                (self.metrics.failed_trades * 5) + 
                (self.metrics.trades_per_hour * 2))
        ))
        
        # Add to performance window
        trade_result["timestamp"] = current_time
        self.performance_window.append(trade_result)
        
        # Record execution metrics in time window scheduler for better window score calculation
        # CRITICAL: Skip gate failures - they're protective mechanisms, not execution failures
        # Recording them creates a feedback loop that keeps window score low
        error_type = trade_result.get("error_type")
        if error_type != "gate":
            try:
                from src.ai.ai_time_window_scheduler import get_time_window_scheduler
                scheduler = get_time_window_scheduler()

                # Extract execution metrics from trade result
                success = trade_result.get("success", False)
                slippage = trade_result.get("slippage")  # May be None if not available
                execution_time = trade_result.get("execution_time", 0)  # Already in ms

                # Record execution in scheduler's execution_history
                # Only record actual execution attempts (successful or real failures, not gate blocks)
                scheduler.record_execution(
                    success=success,
                    slippage=slippage,
                    latency_ms=execution_time if execution_time > 0 else None
                )
            except ModuleNotFoundError:
                # Time window scheduler module has been removed; safely skip recording
                # This prevents noisy errors like:
                # \"No module named 'src.ai.ai_time_window_scheduler'\"
                pass
            except Exception as e:
                # Don't fail metrics update if scheduler recording fails for any other reason
                log_error("trading.scheduler_record_error", f"Failed to record execution in scheduler: {e}")
        else:
            # Gate failures are logged but not recorded as execution failures
            log_info("trading.gate_not_recorded", 
                    f"Gate failure not recorded in scheduler (error_type=gate): {trade_result.get('error', 'Unknown')}")
    
    async def _process_partial_fill_retries(self) -> List[Dict[str, Any]]:
        """
        Process partial fill retries for existing positions
        
        Returns:
            List of retry trade results
        """
        try:
            from .partial_fill_retry_manager import get_partial_fill_retry_manager
            
            retry_manager = get_partial_fill_retry_manager()
            candidates = retry_manager.detect_partial_fills()
            
            if not candidates:
                return []
            
            # Sort by priority if configured
            if retry_manager.retry_priority == "high":
                candidates.sort(key=lambda x: x["unfilled_amount"], reverse=True)
            elif retry_manager.retry_priority == "low":
                candidates.sort(key=lambda x: x["unfilled_amount"])
            
            retry_results = []
            
            for candidate in candidates:
                try:
                    # Prepare token for retry
                    retry_token = retry_manager.prepare_retry_token(candidate)
                    position_key = candidate["position_key"]
                    
                    log_info("partial_fill_retry.attempting",
                            f"Attempting retry for {retry_token.get('symbol', '?')}: "
                            f"${candidate['unfilled_amount']:.2f} unfilled",
                            {
                                "symbol": retry_token.get("symbol"),
                                "unfilled_amount": candidate["unfilled_amount"],
                                "retry_attempt": retry_token.get("retry_attempt", 1)
                            })
                    
                    # Execute retry trade
                    trade_result = await self._execute_trade_async(retry_token)
                    
                    # Mark retry attempt
                    retry_manager.mark_retry_attempted(position_key, trade_result.get("success", False))
                    
                    if trade_result.get("success", False):
                        # Update position with new total size
                        retry_filled = trade_result.get("position_size", candidate["unfilled_amount"])
                        new_total = candidate["actual_size"] + retry_filled
                        retry_manager.update_position_after_retry(position_key, retry_filled, new_total)
                        
                        log_info("partial_fill_retry.success",
                                f"✅ Retry successful for {retry_token.get('symbol', '?')}: "
                                f"filled ${retry_filled:.2f}, new total: ${new_total:.2f}")
                    else:
                        log_info("partial_fill_retry.failed",
                                f"❌ Retry failed for {retry_token.get('symbol', '?')}: "
                                f"{trade_result.get('error', 'Unknown error')}")
                    
                    retry_results.append(trade_result)
                    
                    # Rate limit between retries
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    log_error("partial_fill_retry.execution_error",
                             f"Error executing retry for {candidate.get('position_key', 'unknown')}: {e}")
                    continue
            
            return retry_results
            
        except Exception as e:
            log_error("partial_fill_retry.process_error", f"Error processing partial fill retries: {e}")
            return []
    
    async def run_enhanced_trading_cycle(self) -> Dict[str, Any]:
        """Run a single enhanced trading cycle"""
        cycle_start = time.time()
        log_info("trading.cycle_start", "🚀 Starting enhanced async trading cycle")
        
        # CRITICAL: Force refresh position count at start of each cycle
        # This ensures we have accurate counts after sells complete
        try:
            from src.core.risk_manager import _open_positions_count
            current_positions = _open_positions_count()
            log_info("trading.position_count", f"📊 Current open positions: {current_positions}")
        except Exception as e:
            log_error("trading.position_count_error", f"Error checking position count: {e}")
        
        # Position reconciliation runs in watchdog (every 5 min) - no need to run here
        # Check AI module health
        ai_health = await self._check_ai_module_health()
        if not ai_health.get("overall_healthy", False):
            log_error("trading.ai_unhealthy", f"AI modules unhealthy: {ai_health.get('unhealthy_modules', [])}")
            return {"success": False, "error": "AI modules unhealthy"}
        
        # Check circuit breaker
        if is_circuit_breaker_active():
            log_info("trading.circuit_breaker", "⏸️ Circuit breaker active - skipping cycle")
            return {"success": False, "error": "Circuit breaker active"}
        
        # Capture market regime once per cycle for downstream consumers
        regime_data: Optional[Dict[str, Any]] = None
        try:
            regime_data = ai_market_regime_detector.detect_market_regime()
            log_info(
                "trading.regime_snapshot",
                "📊 Captured market regime for cycle",
                {
                    "regime": regime_data.get("regime") if isinstance(regime_data, dict) else None,
                    "confidence": regime_data.get("confidence") if isinstance(regime_data, dict) else None,
                },
            )
        except Exception as exc:
            log_error("trading.regime_error", f"Failed to capture market regime: {exc}")
            regime_data = None
        
        # Fetch tokens from all supported chains
        # Get configurable discovery settings
        per_chain_limit = get_config_int("token_discovery.tokens_per_chain", 30)
        max_total = get_config_int("token_discovery.max_tokens_total", 80)
        
        all_tokens = []
        fetch_tasks = []
        
        for chain in self.config.chains.supported_chains:
            task = self.fetch_trending_tokens_async(chain, limit=per_chain_limit)
            fetch_tasks.append(task)
        
        try:
            chain_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            
            for i, result in enumerate(chain_results):
                if isinstance(result, Exception):
                    log_error("trading.chain_fetch_error", f"Error fetching tokens for chain {i}: {result}")
                    continue
                all_tokens.extend(result)
            
            # Cap total tokens to prevent excessive processing
            if len(all_tokens) > max_total:
                all_tokens = all_tokens[:max_total]
                log_info("trading.fetch", f"📊 Capped tokens to {max_total} (fetched {len(chain_results) * per_chain_limit} total)")
            
            log_info("trading.fetch", f"📊 Fetched {len(all_tokens)} tokens across {len(self.config.chains.supported_chains)} chains")
            
            # Always include minute_price_tracker tokens in selection
            tracked_all: List[Dict] = []
            for ch in self.config.chains.supported_chains:
                tracked = await self._fetch_tracked_tokens(ch)
                tracked_all.extend(tracked)
            if tracked_all:
                seen = {(t.get("address") or "").lower() for t in all_tokens}
                added = 0
                for t in tracked_all:
                    addr = (t.get("address") or "").lower()
                    if not addr:
                        continue
                    if addr in seen:
                        for i, tok in enumerate(all_tokens):
                            if (tok.get("address") or "").lower() == addr:
                                all_tokens[i] = t
                                break
                    else:
                        all_tokens.append(t)
                        seen.add(addr)
                        added += 1
                log_info("trading.tracked_tokens.injected",
                         f"Injected {len(tracked_all)} tracked tokens into selection ({added} new)",
                         total_tracked=len(tracked_all), added=added)
            
        except Exception as e:
            log_error("trading.token_fetch_error", f"Error in token fetching: {e}")
            return {"success": False, "error": f"Token fetching failed: {e}"}
        
        if not all_tokens:
            log_info("trading.no_tokens", "😴 No tokens found this cycle")
            return {"success": True, "tokens_processed": 0}
        
        # OPTIMIZATION #1: Early filtering - apply volume/liquidity filters BEFORE AI analysis
        log_info("trading.early_filter_start", f"🔍 Applying early filters to {len(all_tokens)} tokens before AI analysis")
        filtered_tokens = self._filter_tokens_early(all_tokens)
        log_info("trading.early_filter_complete", 
                f"✅ Early filtering complete: {len(filtered_tokens)}/{len(all_tokens)} tokens passed basic thresholds")
        
        if not filtered_tokens:
            log_info("trading.no_filtered_tokens", "😴 No tokens passed early filters this cycle")
            return {"success": True, "tokens_processed": 0, "tokens_filtered_early": len(all_tokens)}
        
        # Apply hard limiter for candle fetching with acceleration-based prioritization
        # Use accel_scoring config if available, otherwise fall back to legacy config
        accel_cfg = get_config("trading.accel_scoring", {})
        max_tokens_for_candles = accel_cfg.get("max_tokens_per_cycle_for_candles") or get_config_int('helius_15m_candle_policy.max_tokens_per_cycle_for_candles', 15)
        
        tracked_tokens_list = [t for t in filtered_tokens if t.get("_is_tracked_token", False)]
        regular_tokens = [t for t in filtered_tokens if not t.get("_is_tracked_token", False)]
        original_regular_count = len(regular_tokens)
        
        # Use acceleration-based selection for regular tokens
        if len(regular_tokens) > max_tokens_for_candles:
            regular_tokens = self._select_tokens_for_candles(regular_tokens, max_tokens_for_candles)
            log_info("trading.candle_selection.applied",
                    f"📊 Applied acceleration-based selection: {len(regular_tokens)}/{original_regular_count} regular tokens selected",
                    selected=len(regular_tokens),
                    original=original_regular_count,
                    max_allowed=max_tokens_for_candles)
        
        filtered_tokens = tracked_tokens_list + regular_tokens
        if tracked_tokens_list:
            log_info("trading.tracked_tokens.included",
                     f"📊 Including {len(tracked_tokens_list)} tracked tokens + {len(regular_tokens)} regular "
                     f"(max regular={max_tokens_for_candles})",
                     tracked=len(tracked_tokens_list), regular=len(regular_tokens))
        if original_regular_count > max_tokens_for_candles:
            log_info("trading.candle_limiter",
                    f"📊 Limited tokens for candle fetching: {len(filtered_tokens)} tokens "
                    f"({len(tracked_tokens_list)} tracked + {len(regular_tokens)} regular, max regular={max_tokens_for_candles})")
        
        # Process tokens in batches (only tokens that passed early filters AND limiter)
        approved_tokens = []
        batch_results = []
        
        for i in range(0, len(filtered_tokens), self.batch_size):
            batch = filtered_tokens[i:i + self.batch_size]
            batch_result = await self._process_token_batch(batch, regime_data)
            batch_results.extend(batch_result)
            
            # Filter approved tokens
            approved = [token for token in batch_result if token.get("approved_for_trading", False)]
            approved_tokens.extend(approved)
            
            log_info("trading.batch", f"Batch {i//self.batch_size + 1}: {len(approved)}/{len(batch)} tokens approved")
        
        log_info("trading.approval", f"✅ Total approved tokens: {len(approved_tokens)}/{len(all_tokens)}")
        
        # Process partial fill retries BEFORE new trades (if priority is high)
        # Otherwise process after new trades
        retry_results = []
        try:
            from .partial_fill_retry_manager import get_partial_fill_retry_manager
            retry_manager = get_partial_fill_retry_manager()
            
            if retry_manager.retry_priority == "high":
                retry_results = await self._process_partial_fill_retries()
        except Exception as e:
            log_error("trading.retry_error", f"Error processing retries: {e}")
        
        # Execute trades for approved tokens (limit concurrent trades)
        trade_results = []
        if approved_tokens:
            # Initial sort by AI confidence, then by quality_score
            approved_tokens.sort(
                key=lambda x: (
                    -x.get("ai_analysis", {}).get("trading_recommendation", {}).get("confidence", 0),
                    -x.get("ai_analysis", {}).get("quality_score", 0)
                )
            )
            
            # Check holder concentration for top 20 candidates to prioritize better distribution
            top_candidates_count = min(len(approved_tokens), 20)
            log_info("trading.holder_concentration_check", 
                    f"Checking holder concentration for top {top_candidates_count} candidates")
            
            for token in approved_tokens[:top_candidates_count]:
                if "holder_concentration_pct" not in token:
                    try:
                        from src.utils.holder_concentration_checker import check_holder_concentration
                        
                        token_address = token.get("address", "")
                        chain_id = token.get("chainId", token.get("chain", "solana")).lower()
                        symbol = token.get("symbol", "UNKNOWN")
                        
                        if token_address and get_config_bool("enable_holder_concentration_check", True):
                            holder_check = check_holder_concentration(token_address, chain_id)
                            if holder_check and not holder_check.get("error"):
                                holder_concentration_pct = holder_check.get("top_10_percentage", 100.0)
                                token["holder_concentration_pct"] = holder_concentration_pct
                                log_info("trading.holder_concentration",
                                        f"Holder concentration for {symbol}: {holder_concentration_pct:.2f}%",
                                        symbol=symbol,
                                        holder_concentration_pct=holder_concentration_pct)
                            else:
                                # If check failed, use high value (100%) so it ranks lower
                                token["holder_concentration_pct"] = 100.0
                                log_info("trading.holder_concentration",
                                        f"Holder concentration check failed for {symbol}, defaulting to 100%",
                                        symbol=symbol)
                        else:
                            token["holder_concentration_pct"] = 100.0
                    except Exception as e:
                        log_error("trading.holder_concentration_error",
                                f"Error checking holder concentration for {token.get('symbol', 'UNKNOWN')}: {e}")
                        # Use high value on error so it ranks lower
                        token["holder_concentration_pct"] = 100.0
            
            # Set default holder_concentration_pct for tokens not checked (ranks lower)
            for token in approved_tokens[top_candidates_count:]:
                if "holder_concentration_pct" not in token:
                    token["holder_concentration_pct"] = 100.0
            
            # Re-sort with holder concentration included (lower concentration = better distribution = higher rank)
            approved_tokens.sort(
                key=lambda x: (
                    -x.get("ai_analysis", {}).get("trading_recommendation", {}).get("confidence", 0),
                    -x.get("ai_analysis", {}).get("quality_score", 0),
                    x.get("holder_concentration_pct", 100.0)  # Lower is better (better distribution)
                )
            )
            
            log_info("trading.sorting_complete",
                    f"Re-sorted {len(approved_tokens)} approved tokens with holder concentration ranking")
            
            # Execute all approved tokens (ranked by confidence, quality, and holder concentration)
            # Best tokens execute first, but all approved tokens will attempt execution
            tokens_to_trade = approved_tokens
            
            log_info("trading.execute", f"🎯 Executing trades for {len(tokens_to_trade)} approved tokens (ranked best to worst)")
            
            # Execute trades in parallel (rate limiter controls concurrent executions)
            trade_tasks = [self._execute_trade_async(token) for token in tokens_to_trade]
            trade_results = await asyncio.gather(*trade_tasks, return_exceptions=True)
            
            # Update metrics for each trade
            for result in trade_results:
                if isinstance(result, Exception):
                    log_error("trading.trade_execution_error", f"Trade execution error: {result}")
                    continue
                await self._update_metrics(result)
        
        # Process partial fill retries AFTER new trades (if priority is low/normal)
        if not retry_results:
            try:
                retry_results = await self._process_partial_fill_retries()
            except Exception as e:
                log_error("trading.retry_error", f"Error processing retries: {e}")
        
        # Calculate cycle metrics
        cycle_time = time.time() - cycle_start
        successful_trades = len([r for r in trade_results if isinstance(r, dict) and r.get("success", False)])
        
        # CRITICAL: Add wallet balance and risk metrics
        wallet_metrics = {}
        try:
            from src.core.risk_manager import _get_combined_wallet_balance_usd, _get_wallet_balance_usd, get_tier_based_risk_limits, _get_current_exposure_usd, _load_state
            combined_balance = _get_combined_wallet_balance_usd()
            risk_config = get_tier_based_risk_limits(combined_balance)
            current_exposure = _get_current_exposure_usd()
            risk_state = _load_state()
            
            wallet_metrics = {
                "combined_balance_usd": combined_balance,
                "risk_tier": risk_config.get('TIER_NAME', 'unknown'),
                "current_exposure_usd": current_exposure,
                "max_exposure_usd": risk_config.get('MAX_TOTAL_EXPOSURE_USD', 0),
                "exposure_utilization_pct": (current_exposure / risk_config.get('MAX_TOTAL_EXPOSURE_USD', 1)) * 100 if risk_config.get('MAX_TOTAL_EXPOSURE_USD', 0) > 0 else 0,
                "daily_pnl_usd": risk_state.get("realized_pnl_usd", 0.0),
                "daily_loss_limit_usd": risk_config.get('DAILY_LOSS_LIMIT_USD', 0),
                "loss_limit_remaining_usd": risk_config.get('DAILY_LOSS_LIMIT_USD', 0) + risk_state.get("realized_pnl_usd", 0.0),  # Positive = remaining buffer
            }
            
            # Per-chain balances
            chain_balances = {}
            for chain in self.config.chains.supported_chains:
                try:
                    chain_balance = _get_wallet_balance_usd(chain, use_cache_fallback=True)
                    if chain_balance is not None:
                        chain_balances[chain] = chain_balance
                except Exception:
                    pass
            if chain_balances:
                wallet_metrics["chain_balances"] = chain_balances
        except Exception as e:
            log_error("trading.wallet_metrics_error", f"Error collecting wallet metrics: {e}")
        
        # CRITICAL: Add API rate limit status
        api_metrics = {}
        try:
            from src.utils.api_tracker import get_tracker
            api_tracker = get_tracker()
            api_metrics = {
                "helius_remaining": api_tracker.get_remaining("helius", 300000),
                "coingecko_remaining": api_tracker.get_remaining("coingecko", 300),
                "coincap_remaining": api_tracker.get_remaining("coincap", 130),
            }
        except Exception as e:
            log_error("trading.api_metrics_error", f"Error collecting API metrics: {e}")
        
        # CRITICAL: Add price memory stats
        price_memory_stats = {}
        try:
            from src.storage.price_memory import load_price_memory
            price_mem = load_price_memory()
            if price_mem:
                timestamps = [entry.get("ts", 0) for entry in price_mem.values() if isinstance(entry, dict)]
                if timestamps:
                    now = int(time.time())
                    oldest_age = now - min(timestamps) if timestamps else 0
                    newest_age = now - max(timestamps) if timestamps else 0
                    price_memory_stats = {
                        "tokens_tracked": len(price_mem),
                        "oldest_entry_hours": oldest_age / 3600,
                        "newest_entry_hours": newest_age / 3600,
                    }
        except Exception as e:
            log_error("trading.price_memory_stats_error", f"Error collecting price memory stats: {e}")
        
        # CRITICAL: Add position summary
        position_summary = {}
        try:
            from src.storage.positions import load_positions as load_positions_store
            positions = load_positions_store()
            if positions:
                position_sizes = []
                for pos_data in positions.values():
                    if isinstance(pos_data, dict):
                        size = float(pos_data.get("position_size_usd", 0) or 0)
                        if size > 0:
                            position_sizes.append(size)
                
                if position_sizes:
                    position_summary = {
                        "total_positions": len(positions),
                        "avg_position_size_usd": sum(position_sizes) / len(position_sizes),
                        "largest_position_usd": max(position_sizes),
                        "total_exposure_usd": sum(position_sizes),
                    }
        except Exception as e:
            log_error("trading.position_summary_error", f"Error collecting position summary: {e}")
        
        cycle_summary = {
            "success": True,
            "cycle_time": cycle_time,
            "tokens_fetched": len(all_tokens),
            "tokens_filtered_early": len(all_tokens) - len(filtered_tokens) if 'filtered_tokens' in locals() else 0,
            "tokens_analyzed": len(batch_results),
            "tokens_approved": len(approved_tokens),
            "trades_executed": len(trade_results),
            "trades_successful": successful_trades,
            "retries_attempted": len(retry_results),
            "retries_successful": len([r for r in retry_results if isinstance(r, dict) and r.get("success", False)]),
            "success_rate": successful_trades / len(trade_results) if trade_results else 0,
            "wallet_metrics": wallet_metrics,
            "api_metrics": api_metrics,
            "price_memory_stats": price_memory_stats,
            "position_summary": position_summary,
            "metrics": {
                "total_trades": self.metrics.total_trades,
                "success_rate": self.metrics.success_rate,
                "total_pnl": self.metrics.total_pnl,
                "avg_execution_time": self.metrics.avg_execution_time,
                "trades_per_hour": self.metrics.trades_per_hour,
                "health_score": self.metrics.health_score
            }
        }
        
        log_info("trading.cycle_complete", "📊 Trading cycle complete", cycle_summary)
        return cycle_summary

async def run_enhanced_async_trading():
    """Main function to run enhanced async trading"""
    log_info("trading.start", "🌱 Starting Enhanced Async Trading Bot - Phase 3")
    
    # Start trading session
    start_trading_session()
    
    # Ensure position monitor is running for stop-loss/take-profit exits
    try:
        _launch_monitor_detached()
        log_info("trading.monitor_launched", "✅ Position monitor launched")
    except Exception as e:
        log_error("trading.monitor_launch_error", f"Failed to launch monitor: {e}")
    
    try:
        async with EnhancedAsyncTradingEngine() as engine:
            cycle_count = 0
            
            while True:
                cycle_count += 1
                log_info("trading.cycle", f"🔄 Starting trading cycle #{cycle_count}")
                
                try:
                    result = await engine.run_enhanced_trading_cycle()
                    
                    if not result.get("success", False):
                        error_msg = result.get('error', 'Unknown error')
                        log_error("trading.cycle_failed", f"Cycle #{cycle_count} failed: {error_msg}")
                        
                        # CRITICAL: Check if failure is due to max positions - log current count
                        if "max_concurrent_positions" in str(error_msg).lower():
                            try:
                                from src.core.risk_manager import _open_positions_count
                                current_count = _open_positions_count()
                                log_info("trading.max_positions_check", 
                                        f"⚠️ Cycle blocked by max positions. Current count: {current_count}. "
                                        f"This should refresh after positions are closed.")
                            except Exception:
                                pass
                        
                        await asyncio.sleep(30)  # Wait before retry
                        continue  # Continue to next cycle - don't stop
                    
                    # Send periodic status report (throttled to every 1 hour)
                    try:
                        send_periodic_status_report()
                    except Exception as e:
                        log_error("trading.status_report", f"Periodic status report failed: {e}")
                    
                    # CRITICAL: Log position count after cycle to verify positions were properly closed
                    try:
                        from src.core.risk_manager import _open_positions_count
                        positions_after = _open_positions_count()
                        log_info("trading.position_count_after", 
                                f"📊 Open positions after cycle: {positions_after} "
                                f"(tokens_processed: {result.get('tokens_processed', 0)}, "
                                f"trades_executed: {result.get('trades_executed', 0)})")
                    except Exception as e:
                        log_error("trading.position_count_error_after", f"Error checking position count after cycle: {e}")
                    
                    # OPTIMIZATION #2: Adaptive cycle wait time based on market conditions
                    wait_time = engine._calculate_adaptive_wait_time(result)
                    wait_minutes = wait_time / 60
                    log_info("trading.wait", 
                            f"⏰ Waiting {wait_minutes:.1f} minutes before next cycle "
                            f"(adaptive: {result.get('tokens_approved', 0)} approved, "
                            f"{result.get('trades_executed', 0)} executed)",
                            {"wait_seconds": wait_time, "wait_minutes": wait_minutes})
                    await asyncio.sleep(wait_time)
                    
                except KeyboardInterrupt:
                    # Allow keyboard interrupt to stop the loop
                    raise
                except Exception as e:
                    log_error("trading.cycle_error", f"Error in trading cycle #{cycle_count}: {e}")
                    import traceback
                    log_error("trading.cycle_error_trace", f"Traceback: {traceback.format_exc()}")
                    # CRITICAL: Ensure we continue the loop even after exceptions
                    # Don't let exceptions stop the trading loop - this is why bot stops after sells
                    log_info("trading.cycle_recover", f"🔄 Recovering from error, continuing to next cycle...")
                    await asyncio.sleep(60)  # Wait before retry
                    continue  # Explicitly continue to next cycle - this ensures loop never stops
                    
    except KeyboardInterrupt:
        log_info("trading.stop", "🛑 Enhanced async trading stopped by user")
    except Exception as e:
        log_error("trading.fatal_error", f"Fatal error in enhanced async trading: {e}")
    finally:
        end_trading_session()
        log_info("trading.shutdown", "👋 Enhanced async trading shutdown complete")

if __name__ == "__main__":
    asyncio.run(run_enhanced_async_trading())
