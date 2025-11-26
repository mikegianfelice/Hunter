#!/usr/bin/env python3
"""
Enhanced Async Trading Loop - Phase 3
Advanced async processing with connection pooling, batch processing, and performance optimization
"""

import asyncio
import aiohttp
import aiofiles
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

from src.monitoring.structured_logger import log_info, log_error, log_trade, log_performance
from src.config.config_validator import get_validated_config
from src.config.config_loader import get_config_bool, get_config_int
from src.monitoring.performance_monitor import record_trade_metrics, start_trading_session, end_trading_session
from src.core.centralized_risk_manager import assess_trade_risk, update_trade_result, is_circuit_breaker_active
from src.ai.ai_circuit_breaker import circuit_breaker_manager, check_ai_module_health
from src.monitoring.telegram_bot import send_periodic_status_report
from src.execution.multi_chain_executor import _launch_monitor_detached
from src.core.helius_reconciliation import reconcile_positions_and_pnl
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
        self.batch_size = 5
        self.max_concurrent_trades = 3
        self.performance_window = deque(maxlen=100)  # Last 100 trades
        self.helius_reconciliation_enabled = get_config_bool("enable_helius_reconciliation", True)
        self.helius_reconciliation_interval = max(1, get_config_int("helius_reconciliation_interval_minutes", 10))
        self.helius_reconciliation_limit = max(1, get_config_int("helius_reconciliation_tx_limit", 200))
        self._helius_last_reconciliation = 0.0
        self._helius_disabled_reason: Optional[str] = None
        
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
    
    async def _maybe_run_helius_reconciliation(self) -> None:
        """Periodically reconcile Solana positions via Helius to keep exposure accurate."""
        if not self.helius_reconciliation_enabled:
            return
        
        interval_seconds = self.helius_reconciliation_interval * 60
        now = time.time()
        if now - self._helius_last_reconciliation < interval_seconds:
            return
        
        self._helius_last_reconciliation = now
        
        try:
            summary = await asyncio.to_thread(
                reconcile_positions_and_pnl,
                limit=self.helius_reconciliation_limit,
            )
        except Exception as e:
            log_error("trading.helius_reconcile_error", f"Helius reconciliation failed: {e}")
            return
        
        if not summary.get("enabled", False):
            reason = summary.get("reason", "unknown reason")
            log_info(
                "trading.helius_reconcile_disabled",
                f"Helius reconciliation disabled: {reason}",
            )
            self.helius_reconciliation_enabled = False
            self._helius_disabled_reason = reason
            return
        
        log_info(
            "trading.helius_reconcile",
            "🔄 Helius reconciliation executed",
            {
                "open_positions_closed": summary.get("open_positions_closed", 0),
                "open_positions_verified": summary.get("open_positions_verified", 0),
                "trades_updated": summary.get("trades_updated", 0),
                "issues": summary.get("issues", []),
            },
        )
    
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
            
            # DexScreener trending API endpoints - use specific chain searches
            if dex_chain == "ethereum":
                trending_urls = [
                    # Broader nets for ERC-20 discovery on Ethereum
                    "https://api.dexscreener.com/latest/dex/search/?q=uniswap",
                    "https://api.dexscreener.com/latest/dex/search/?q=trending",
                    "https://api.dexscreener.com/latest/dex/search/?q=top",
                    "https://api.dexscreener.com/latest/dex/search/?q=volume",
                    "https://api.dexscreener.com/latest/dex/search/?q=liquidity",
                    # Keep originals as supplemental (though they mostly return ETH/WETH pairs)
                    "https://api.dexscreener.com/latest/dex/search/?q=ethereum",
                    "https://api.dexscreener.com/latest/dex/search/?q=eth",
                    "https://api.dexscreener.com/latest/dex/search/?q=weth"
                ]
            elif dex_chain == "solana":
                trending_urls = [
                    "https://api.dexscreener.com/latest/dex/search/?q=bonk",
                    "https://api.dexscreener.com/latest/dex/search/?q=raydium",
                    "https://api.dexscreener.com/latest/dex/search/?q=jupiter",
                    "https://api.dexscreener.com/latest/dex/search/?q=orca"
                ]
            else:
                trending_urls = [
                    f"https://api.dexscreener.com/latest/dex/search/?q={dex_chain}",
                    f"https://api.dexscreener.com/latest/dex/search/?q=trending"
                ]
            
            all_tokens = []
            seen_pairs = set()
            
            async with aiohttp.ClientSession() as session:
                for url in trending_urls:
                    try:
                        async with session.get(url, timeout=10) as response:
                            if response.status == 200:
                                data = await response.json()
                                pairs = data.get("pairs", []) if data and data.get("pairs") else []
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
                                        
                                        # Calculate metrics
                                        price_usd = float(pair.get("priceUsd", 0))
                                        volume_24h = float(pair.get("volume", {}).get("h24", 0))
                                        liquidity_usd = float(pair.get("liquidity", {}).get("usd", 0))
                                        price_change_24h = float(pair.get("priceChange", {}).get("h24", 0))
                                        
                                        # Skip tokens with poor metrics
                                        if price_usd <= 0 or volume_24h < 1000 or liquidity_usd < 5000:
                                            continue
                                            
                                        token = {
                                            "symbol": base_token.get("symbol", ""),
                                            "address": base_token.get("address", ""),
                                            "chain": chain,
                                            "priceUsd": price_usd,
                                            "volume24h": volume_24h,
                                            "liquidity": liquidity_usd,
                                            "marketCap": float(pair.get("marketCap", 0)),
                                            "priceChange24h": price_change_24h,
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
                                            
                    except Exception as e:
                        log_error("trading.api_error", f"Error fetching from {url}: {e}")
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
                        
            # Sort by volume and take the top tokens
            all_tokens.sort(key=lambda x: x["volume24h"], reverse=True)
            return all_tokens[:limit]
            
        except Exception as e:
            log_error("trading.fetch_error", f"Error fetching real tokens: {e}")
            return []
    
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
        Execute real trade with time window scheduler gate
        """
        # Check time window scheduler before executing
        try:
            from src.ai.ai_time_window_scheduler import get_time_window_scheduler
            scheduler = get_time_window_scheduler()
            
            # Get recent execution metrics
            recent_metrics = {
                "fill_success_rate": self._get_recent_fill_rate(),
                "avg_slippage": self._get_recent_avg_slippage(),
                "avg_latency_ms": self._get_recent_avg_latency()
            }
            
            # Get market quality metrics
            market_metrics = {
                "volatility": abs(float(token.get("priceChange24h", 0)) / 100.0) if token.get("priceChange24h") else 0.5,
                "liquidity_score": min(1.0, float(token.get("liquidity", 0)) / 1_000_000.0) if token.get("liquidity") else 0.5
            }
            
            # Check if we should trade now
            window_decision = scheduler.should_trade_now(recent_metrics, market_metrics)
            
            if not window_decision.should_trade:
                log_info("trading.window_blocked",
                        symbol=token.get("symbol", "?"),
                        score=window_decision.score,
                        reason=window_decision.reason)
                return {
                    "success": False,
                    "error": f"Time window blocked: {window_decision.reason}",
                    "window_score": window_decision.score,
                    "error_type": "gate"  # Mark as gate failure
                }
        except Exception as e:
            log_error("trading.scheduler_error", error=str(e))
            # On error, proceed with trade (fail open)
        
        # Proceed with original trade execution
        return await self._execute_real_trade_internal(token, position_size, chain)
    
    async def _execute_real_trade_internal(self, token: Dict, position_size: float, chain: str) -> Dict[str, Any]:
        """Execute real trade using DEX integrations"""
        try:
            symbol = token.get("symbol", "")
            address = token.get("address", "")
            
            # Import the appropriate executor based on chain
            if chain.lower() == "solana":
                from .jupiter_executor import buy_token_solana
                
                # Try Jupiter first
                try:
                    log_info("trading.jupiter", f"Executing Jupiter trade for {symbol} on Solana")
                    tx_hash, success = buy_token_solana(address, position_size, symbol, test_mode=False)
                    if success and tx_hash:
                        # Analyze transaction to get actual execution details
                        buy_fee_data = await self._analyze_buy_transaction(tx_hash, chain, "jupiter")
                        
                        # At entry time, P&L is 0.0 - will be calculated when position is closed
                        # Real P&L = (exit_price - entry_price) / entry_price * position_size
                        profit_loss = 0.0  # No profit/loss until position is closed
                        return {
                            "success": True,
                            "profit_loss": profit_loss,
                            "tx_hash": tx_hash,
                            "dex": "jupiter",
                            "fee_data": buy_fee_data
                        }
                    elif not success:
                        # Log detailed error for Jupiter failures
                        error_msg = tx_hash if isinstance(tx_hash, str) and not tx_hash.startswith("0x") else "Jupiter returned unsuccessful"
                        log_error("trading.jupiter_error", f"Jupiter execution failed for {symbol}: {error_msg}")
                except Exception as e:
                    log_error("trading.jupiter_error", f"Jupiter execution exception for {symbol}: {e}")
                
                # Try Raydium fallback
                try:
                    from .raydium_executor import RaydiumExecutor
                    
                    log_info("trading.raydium", f"Executing Raydium trade for {symbol} on Solana")
                    raydium = RaydiumExecutor()
                    success, tx_hash = raydium.execute_trade(address, position_size, is_buy=True)
                    if success and tx_hash:
                        # Analyze transaction to get actual execution details
                        buy_fee_data = await self._analyze_buy_transaction(tx_hash, chain, "raydium")
                        
                        # At entry time, P&L is 0.0 - will be calculated when position is closed
                        profit_loss = 0.0  # No profit/loss until position is closed
                        return {
                            "success": True,
                            "profit_loss": profit_loss,
                            "tx_hash": tx_hash,
                            "dex": "raydium",
                            "fee_data": buy_fee_data
                        }
                except Exception as e:
                    log_error("trading.raydium_error", f"Raydium execution failed: {e}")
                    
            elif chain.lower() in ["ethereum", "base", "arbitrum", "polygon"]:
                from .uniswap_executor import buy_token
                
                # Execute Uniswap trade
                try:
                    log_info("trading.uniswap", f"Executing Uniswap trade for {symbol} on {chain}")
                    tx_hash, success = buy_token(address, position_size, symbol)
                    if success and tx_hash:
                        # Analyze transaction to get actual execution details
                        buy_fee_data = await self._analyze_buy_transaction(tx_hash, chain, "uniswap")
                        
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
                "quality_score": ai_result.overall_score,  # Use overall_score instead of quality_score
                "sentiment_analysis": {
                    "category": sentiment.get("category", "neutral"),
                    "score": sentiment.get("score", 0.5),
                    "confidence": sentiment.get("confidence", 0.5)
                },
                "price_prediction": {
                    "success_probability": prediction.get("price_movement_probability", 0.5),
                    "confidence_level": prediction.get("confidence", 0.5),
                    "expected_return": prediction.get("expected_return", 0.1),
                    "risk_score": risk.get("risk_score", 0.3)
                },
                "market_analysis": {
                    "trend": market.get("market_trend", "neutral"),
                    "volatility": "high" if abs(token.get("priceChange24h", 0)) > 0.1 else "low",
                    "liquidity_score": market.get("liquidity_score", 0.5),
                    "volume_score": market.get("volume_score", 0.5)
                },
                "trading_recommendation": {
                    "action": recommendations.get("action", "hold"),
                    "confidence": recommendations.get("confidence", ai_result.confidence),
                    "position_size": recommendations.get("position_size", 10),
                    "take_profit": recommendations.get("take_profit", 0.15),
                    "stop_loss": recommendations.get("stop_loss", 0.08)
                },
                "risk_factors": risk.get("risk_factors", []),
                "technical_analysis": {
                    "technical_score": technical.get("technical_score", 0.5),
                    "trend": technical.get("trend", "neutral"),
                    "signals": technical.get("signals", [])
                },
                "execution_analysis": {
                    "execution_score": execution.get("execution_score", 0.5),
                    "recommended_slippage": execution.get("recommended_slippage", 0.05),
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
    
    async def _analyze_buy_transaction(self, tx_hash: str, chain: str, dex: str) -> Dict[str, Any]:
        """Analyze buy transaction to extract fee data"""
        try:
            if chain.lower() == "solana":
                from src.utils.solana_transaction_analyzer import analyze_jupiter_transaction
                from src.config.secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS
                
                fee_data = analyze_jupiter_transaction(
                    SOLANA_RPC_URL, 
                    tx_hash, 
                    SOLANA_WALLET_ADDRESS,
                    is_buy=True
                )
                
                return {
                    'entry_gas_fee_usd': fee_data.get('gas_fee_usd', 0),
                    'entry_amount_usd_actual': fee_data.get('actual_cost_usd', 0),
                    'buy_tx_hash': tx_hash
                }
            elif chain.lower() in ["ethereum", "base", "arbitrum", "polygon"]:
                from src.utils.transaction_analyzer import analyze_buy_transaction
                from src.execution.uniswap_executor import w3
                
                fee_data = analyze_buy_transaction(w3, tx_hash)
                
                return {
                    'entry_gas_fee_usd': fee_data.get('gas_fee_usd', 0),
                    'entry_amount_usd_actual': fee_data.get('actual_cost_usd', 0),
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
            quality_score = analysis.get("quality_score", 0)
            
            # Log recommendation details for debugging
            log_info("trading.recommendation_check",
                    f"Token {token.get('symbol', 'UNKNOWN')}: action={action}, confidence={confidence:.2f}, quality_score={quality_score:.2f}",
                    symbol=token.get("symbol"),
                    action=action,
                    confidence=confidence,
                    quality_score=quality_score)
            
            if action == "buy" and confidence > 0.7:
                enhanced_token["approved_for_trading"] = True
                
                # Get AI-recommended position size
                ai_recommended_size = recommendation.get("position_size", 5)
                
                # Cap position size to tier-based maximum to prevent risk gate blocks
                try:
                    from src.core.risk_manager import get_tier_based_risk_limits
                    tier_limits = get_tier_based_risk_limits()
                    tier_max_size = tier_limits.get("PER_TRADE_MAX_USD", 25.0)
                    
                    # Cap the position size to the tier maximum
                    capped_size = min(ai_recommended_size, tier_max_size)
                    
                    if capped_size < ai_recommended_size:
                        log_info("trading.position_size_capped",
                                f"Position size capped for {token.get('symbol', 'UNKNOWN')}: "
                                f"${ai_recommended_size:.2f} → ${capped_size:.2f} (tier max: ${tier_max_size:.2f})",
                                symbol=token.get("symbol"),
                                ai_recommended=ai_recommended_size,
                                capped=capped_size,
                                tier_max=tier_max_size)
                    
                    enhanced_token["recommended_position_size"] = capped_size
                except Exception as e:
                    log_error("trading.position_size_cap_error", 
                            f"Error capping position size for {token.get('symbol', 'UNKNOWN')}: {e}")
                    # Fallback to AI recommendation if tier check fails
                    enhanced_token["recommended_position_size"] = ai_recommended_size
                
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
        position_size = token.get("recommended_position_size", 5)
        take_profit = token.get("recommended_tp", 0.15)
        
        # Safety cap: Ensure position size doesn't exceed tier-based maximum
        # (This is a safety measure in case position size wasn't capped earlier)
        try:
            from src.core.risk_manager import get_tier_based_risk_limits
            tier_limits = get_tier_based_risk_limits()
            tier_max_size = tier_limits.get("PER_TRADE_MAX_USD", 25.0)
            
            if position_size > tier_max_size:
                log_info("trading.position_size_safety_cap",
                        f"Safety cap applied for {symbol}: ${position_size:.2f} → ${tier_max_size:.2f}",
                        symbol=symbol,
                        original_size=position_size,
                        capped_size=tier_max_size)
                position_size = tier_max_size
        except Exception as e:
            log_error("trading.position_size_safety_cap_error", 
                    f"Error applying safety cap for {symbol}: {e}")
            # Continue with original position size if cap check fails
        
        # DEBUG: Log that we're entering the trade execution
        log_info("trading.debug", f"🔍 DEBUG: Entering _execute_trade_async for {symbol} on {chain}")
        
        async with self.rate_limiter:
            start_time = time.time()
            
            try:
                log_info("trading.execute", f"Executing trade for {symbol} on {chain}")
                
                # Pre-trade wallet/limits gate - check balance and position limits
                try:
                    from src.core.risk_manager import allow_new_trade
                    allowed, reason = allow_new_trade(position_size, token_address=address, chain_id=chain)
                    if not allowed:
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
                    # Continue if risk gate check fails (don't block on errors)
                
                # Risk assessment
                log_info("trading.risk_assessment_start", f"Starting risk assessment for {symbol}")
                risk_result = await assess_trade_risk(token, position_size)
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
                log_info("trading.trade_execution_start", f"Starting trade execution for {symbol} on {chain}")
                trade_result = await self._execute_real_trade(token, position_size, chain)
                
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
                    
                    log_trade("buy", symbol, position_size, True, profit_loss, execution_time)
                    log_info("trading.success", f"✅ Real trade successful: {symbol} - PnL: ${profit_loss:.2f} - TX: {tx_hash}")
                    
                    # Register buy with risk manager
                    try:
                        from src.core.risk_manager import register_buy
                        register_buy(position_size)
                    except Exception as e:
                        log_error("trading.register_buy_error", f"Failed to register buy: {e}")
                    
                    # Log trade entry to performance tracker for status reports and analytics
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
                        trade_id = performance_tracker.log_trade_entry(pt_token, position_size, quality_score, additional_data=fee_data)
                        log_info("trading.performance_logged", f"✅ Trade entry logged to performance tracker for {symbol}")
                    except Exception as e:
                        log_error("trading.performance_log_error", f"Failed to log trade entry for {symbol}: {e}")
                    
                    # Log position to open_positions.json and launch monitor
                    # CRITICAL: This must succeed - retry if needed
                    position_logged = False
                    max_retries = 3
                    position_token = {
                        "address": address,
                        "priceUsd": float(token.get("priceUsd") or 0.0),
                        "chainId": chain,
                        "symbol": symbol,
                        "position_size_usd": position_size,
                    }
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
                    
                    # If position logging still failed after retries, we'll sync from performance_data
                    # after it's logged there (see below in performance tracker section)
                    position_needs_sync = (not position_logged) or (trade_id is None)
                    
                    # Record metrics
                    record_trade_metrics(
                        symbol=symbol,
                        chain=chain,
                        amount_usd=position_size,
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
                        "chain": chain
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
        
        await self._maybe_run_helius_reconciliation()
        
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
        all_tokens = []
        fetch_tasks = []
        
        for chain in self.config.chains.supported_chains:
            task = self.fetch_trending_tokens_async(chain, limit=15)
            fetch_tasks.append(task)
        
        try:
            chain_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            
            for i, result in enumerate(chain_results):
                if isinstance(result, Exception):
                    log_error("trading.chain_fetch_error", f"Error fetching tokens for chain {i}: {result}")
                    continue
                all_tokens.extend(result)
            
            log_info("trading.fetch", f"📊 Fetched {len(all_tokens)} tokens across {len(self.config.chains.supported_chains)} chains")
            
        except Exception as e:
            log_error("trading.token_fetch_error", f"Error in token fetching: {e}")
            return {"success": False, "error": f"Token fetching failed: {e}"}
        
        if not all_tokens:
            log_info("trading.no_tokens", "😴 No tokens found this cycle")
            return {"success": True, "tokens_processed": 0}
        
        # Process tokens in batches
        approved_tokens = []
        batch_results = []
        
        for i in range(0, len(all_tokens), self.batch_size):
            batch = all_tokens[i:i + self.batch_size]
            batch_result = await self._process_token_batch(batch, regime_data)
            batch_results.extend(batch_result)
            
            # Filter approved tokens
            approved = [token for token in batch_result if token.get("approved_for_trading", False)]
            approved_tokens.extend(approved)
            
            log_info("trading.batch", f"Batch {i//self.batch_size + 1}: {len(approved)}/{len(batch)} tokens approved")
        
        log_info("trading.approval", f"✅ Total approved tokens: {len(approved_tokens)}/{len(all_tokens)}")
        
        # Execute trades for approved tokens (limit concurrent trades)
        trade_results = []
        if approved_tokens:
            # Sort by AI confidence for best execution order
            approved_tokens.sort(
                key=lambda x: x.get("ai_analysis", {}).get("trading_recommendation", {}).get("confidence", 0),
                reverse=True
            )
            
            # Limit to max concurrent trades
            tokens_to_trade = approved_tokens[:self.max_concurrent_trades]
            
            log_info("trading.execute", f"🎯 Executing trades for {len(tokens_to_trade)} tokens")
            
            # Execute trades in parallel
            trade_tasks = [self._execute_trade_async(token) for token in tokens_to_trade]
            trade_results = await asyncio.gather(*trade_tasks, return_exceptions=True)
            
            # Update metrics for each trade
            for result in trade_results:
                if isinstance(result, Exception):
                    log_error("trading.trade_execution_error", f"Trade execution error: {result}")
                    continue
                await self._update_metrics(result)
        
        # Calculate cycle metrics
        cycle_time = time.time() - cycle_start
        successful_trades = len([r for r in trade_results if isinstance(r, dict) and r.get("success", False)])
        
        cycle_summary = {
            "success": True,
            "cycle_time": cycle_time,
            "tokens_fetched": len(all_tokens),
            "tokens_analyzed": len(batch_results),
            "tokens_approved": len(approved_tokens),
            "trades_executed": len(trade_results),
            "trades_successful": successful_trades,
            "success_rate": successful_trades / len(trade_results) if trade_results else 0,
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
                    
                    # Wait between cycles
                    wait_time = 300  # 5 minutes
                    wait_minutes = wait_time / 60
                    log_info("trading.wait", f"⏰ Waiting {wait_minutes:.1f} minutes before next cycle...")
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
