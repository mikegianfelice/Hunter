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
from src.monitoring.performance_monitor import record_trade_metrics, start_trading_session, end_trading_session
from src.core.centralized_risk_manager import assess_trade_risk, update_trade_result, is_circuit_breaker_active
from src.ai.ai_circuit_breaker import circuit_breaker_manager, check_ai_module_health

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
        
        # Rate limiting
        self.rate_limiter = asyncio.Semaphore(self.max_concurrent_trades)
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # AI module health tracking
        self.ai_health_cache = {}
        self.ai_health_ttl = 60  # 1 minute
        
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
        
        log_info("Enhanced async trading engine initialized", 
                max_connections=self.connection_pool.max_connections,
                batch_size=self.batch_size)
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            log_info("Async trading engine session closed")
    
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
                log_info(f"Using cached tokens for {chain}", count=len(cached_data))
                return cached_data
        return None
    
    async def _cache_tokens(self, chain: str, cache_key: str, tokens: List[Dict]):
        """Cache tokens with timestamp"""
        self.token_cache[cache_key] = (tokens, time.time())
        log_info(f"Cached {len(tokens)} tokens for {chain}")
    
    async def fetch_trending_tokens_async(self, chain: str, limit: int = 20) -> List[Dict]:
        """Enhanced token fetching with caching and error handling"""
        cache_key = f"{chain}_trending_{limit}"
        
        # Check cache first
        cached_tokens = await self._get_cached_tokens(chain, cache_key)
        if cached_tokens:
            return cached_tokens
        
        await self._rate_limit()
        
        try:
            log_info(f"Fetching trending tokens for {chain} (limit: {limit})")
            
            # Simulate API call with realistic delay
            await asyncio.sleep(random.uniform(0.5, 1.5))
            
            # Generate realistic token data
            tokens = []
            for i in range(limit):
                token = {
                    "symbol": f"ENH{chain.upper()}{i:03d}",
                    "address": f"0x{random.randbytes(20).hex()}",
                    "chain": chain,
                    "priceUsd": round(random.uniform(0.001, 100.0), 6),
                    "volume24h": round(random.uniform(10000, 2000000), 2),
                    "liquidity": round(random.uniform(50000, 10000000), 2),
                    "marketCap": round(random.uniform(100000, 50000000), 2),
                    "priceChange24h": round(random.uniform(-0.5, 0.5), 4),
                    "holders": random.randint(100, 50000),
                    "transactions24h": random.randint(100, 10000),
                    "ai_enhanced_quality_score": round(random.uniform(60, 95), 1),
                    "ai_sentiment": {
                        "category": random.choice(["very_positive", "positive", "neutral", "negative"]),
                        "score": round(random.uniform(0.2, 0.9), 3),
                        "confidence": round(random.uniform(0.6, 0.95), 3)
                    },
                    "ai_prediction": {
                        "overall_success_probability": round(random.uniform(0.6, 0.9), 3),
                        "confidence_level": random.choice(["high", "medium", "low"]),
                        "expected_return": round(random.uniform(0.1, 0.3), 3),
                        "risk_score": round(random.uniform(0.1, 0.4), 3)
                    },
                    "practical_position_size": round(random.uniform(5, 25), 2),
                    "practical_tp": round(random.uniform(0.1, 0.2), 3),
                    "timestamp": datetime.now().isoformat()
                }
                tokens.append(token)
            
            # Cache the results
            await self._cache_tokens(chain, cache_key, tokens)
            
            log_info(f"Fetched {len(tokens)} trending tokens for {chain}")
            return tokens
            
        except Exception as e:
            log_error(f"Error fetching tokens for {chain}: {e}")
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
            log_error(f"Error checking AI health: {e}")
            return {"overall_healthy": False, "unhealthy_modules": ["unknown"]}
    
    async def _analyze_token_ai(self, token: Dict) -> Dict:
        """Perform AI analysis on a single token"""
        symbol = token.get("symbol", "UNKNOWN")
        
        try:
            # Simulate AI analysis with realistic processing time
            analysis_time = random.uniform(0.1, 0.3)
            await asyncio.sleep(analysis_time)
            
            # Enhanced AI analysis results
            analysis = {
                "quality_score": token.get("ai_enhanced_quality_score", 0),
                "sentiment_analysis": {
                    "category": token.get("ai_sentiment", {}).get("category", "neutral"),
                    "score": token.get("ai_sentiment", {}).get("score", 0.5),
                    "confidence": token.get("ai_sentiment", {}).get("confidence", 0.5)
                },
                "price_prediction": {
                    "success_probability": token.get("ai_prediction", {}).get("overall_success_probability", 0.5),
                    "confidence_level": token.get("ai_prediction", {}).get("confidence_level", "medium"),
                    "expected_return": token.get("ai_prediction", {}).get("expected_return", 0.1),
                    "risk_score": token.get("ai_prediction", {}).get("risk_score", 0.3)
                },
                "market_analysis": {
                    "volume_trend": "increasing" if random.random() > 0.5 else "decreasing",
                    "liquidity_health": "good" if token.get("liquidity", 0) > 100000 else "poor",
                    "holder_distribution": "concentrated" if random.random() > 0.7 else "distributed"
                },
                "trading_recommendation": {
                    "action": "buy" if token.get("ai_enhanced_quality_score", 0) > 70 else "hold",
                    "position_size": token.get("practical_position_size", 5),
                    "take_profit": token.get("practical_tp", 0.15),
                    "stop_loss": 0.08,
                    "confidence": random.uniform(0.6, 0.95)
                },
                "analysis_timestamp": datetime.now().isoformat(),
                "processing_time": analysis_time
            }
            
            return analysis
            
        except Exception as e:
            log_error(f"AI analysis failed for {symbol}: {e}")
            return {
                "quality_score": 0,
                "trading_recommendation": {"action": "skip", "confidence": 0.0},
                "error": str(e)
            }
    
    async def _process_token_batch(self, batch: List[Dict]) -> List[Dict]:
        """Process a batch of tokens with parallel AI analysis"""
        log_info(f"Processing batch of {len(batch)} tokens")
        
        # Create tasks for parallel AI analysis
        analysis_tasks = [self._analyze_token_ai(token) for token in batch]
        analyses = await asyncio.gather(*analysis_tasks, return_exceptions=True)
        
        results = []
        for i, (token, analysis) in enumerate(zip(batch, analyses)):
            if isinstance(analysis, Exception):
                log_error(f"Analysis failed for {token.get('symbol', 'UNKNOWN')}: {analysis}")
                continue
            
            # Combine token data with AI analysis
            enhanced_token = {**token, "ai_analysis": analysis}
            
            # Apply trading recommendation
            recommendation = analysis.get("trading_recommendation", {})
            if recommendation.get("action") == "buy" and recommendation.get("confidence", 0) > 0.7:
                enhanced_token["approved_for_trading"] = True
                enhanced_token["recommended_position_size"] = recommendation.get("position_size", 5)
                enhanced_token["recommended_tp"] = recommendation.get("take_profit", 0.15)
            else:
                enhanced_token["approved_for_trading"] = False
            
            results.append(enhanced_token)
        
        log_info(f"Batch processing complete: {len(results)} tokens analyzed")
        return results
    
    async def _execute_trade_async(self, token: Dict) -> Dict[str, Any]:
        """Execute a single trade asynchronously"""
        symbol = token.get("symbol", "UNKNOWN")
        address = token.get("address", "")
        chain = token.get("chain", "ethereum")
        position_size = token.get("recommended_position_size", 5)
        take_profit = token.get("recommended_tp", 0.15)
        
        async with self.rate_limiter:
            start_time = time.time()
            
            try:
                log_info(f"Executing trade for {symbol} on {chain}")
                
                # Risk assessment
                risk_result = await assess_trade_risk(token, position_size)
                if not risk_result.get('approved', False):
                    return {
                        "success": False,
                        "error": f"Risk assessment failed: {risk_result.get('reason', 'Unknown')}",
                        "symbol": symbol
                    }
                
                # Simulate trade execution
                execution_delay = random.uniform(0.5, 2.0)
                await asyncio.sleep(execution_delay)
                
                # Simulate success/failure based on AI confidence
                ai_confidence = token.get("ai_analysis", {}).get("trading_recommendation", {}).get("confidence", 0.5)
                success_probability = min(0.95, ai_confidence + 0.1)  # Boost success rate based on AI confidence
                
                if random.random() < success_probability:
                    # Successful trade
                    profit_loss = position_size * take_profit
                    execution_time = (time.time() - start_time) * 1000
                    
                    log_trade(symbol, address, chain, position_size, "buy", "success", take_profit)
                    log_info(f"✅ Trade successful: {symbol} - PnL: ${profit_loss:.2f}")
                    
                    # Record metrics
                    record_trade_metrics(True, execution_time, profit_loss)
                    update_trade_result(token, True, profit_loss)
                    
                    return {
                        "success": True,
                        "symbol": symbol,
                        "position_size": position_size,
                        "profit_loss": profit_loss,
                        "execution_time": execution_time,
                        "chain": chain
                    }
                else:
                    # Failed trade
                    loss = position_size * 0.08  # 8% stop loss
                    execution_time = (time.time() - start_time) * 1000
                    
                    log_trade(symbol, address, chain, position_size, "buy", "failure", -0.08)
                    log_error(f"❌ Trade failed: {symbol}")
                    
                    # Record metrics
                    record_trade_metrics(False, execution_time, -loss)
                    update_trade_result(token, False, -loss)
                    
                    return {
                        "success": False,
                        "symbol": symbol,
                        "error": "Trade execution failed",
                        "loss": loss,
                        "execution_time": execution_time,
                        "chain": chain
                    }
                    
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                log_error(f"Trade execution error for {symbol}: {e}")
                
                # Record error metrics
                record_trade_metrics(False, execution_time, 0)
                update_trade_result(token, False, 0)
                
                return {
                    "success": False,
                    "symbol": symbol,
                    "error": str(e),
                    "execution_time": execution_time,
                    "chain": chain
                }
    
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
        log_info("🚀 Starting enhanced async trading cycle")
        
        # Check AI module health
        ai_health = await self._check_ai_module_health()
        if not ai_health.get("overall_healthy", False):
            log_error(f"AI modules unhealthy: {ai_health.get('unhealthy_modules', [])}")
            return {"success": False, "error": "AI modules unhealthy"}
        
        # Check circuit breaker
        if is_circuit_breaker_active():
            log_info("⏸️ Circuit breaker active - skipping cycle")
            return {"success": False, "error": "Circuit breaker active"}
        
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
                    log_error(f"Error fetching tokens for chain {i}: {result}")
                    continue
                all_tokens.extend(result)
            
            log_info(f"📊 Fetched {len(all_tokens)} tokens across {len(self.config.chains.supported_chains)} chains")
            
        except Exception as e:
            log_error(f"Error in token fetching: {e}")
            return {"success": False, "error": f"Token fetching failed: {e}"}
        
        if not all_tokens:
            log_info("😴 No tokens found this cycle")
            return {"success": True, "tokens_processed": 0}
        
        # Process tokens in batches
        approved_tokens = []
        batch_results = []
        
        for i in range(0, len(all_tokens), self.batch_size):
            batch = all_tokens[i:i + self.batch_size]
            batch_result = await self._process_token_batch(batch)
            batch_results.extend(batch_result)
            
            # Filter approved tokens
            approved = [token for token in batch_result if token.get("approved_for_trading", False)]
            approved_tokens.extend(approved)
            
            log_info(f"Batch {i//self.batch_size + 1}: {len(approved)}/{len(batch)} tokens approved")
        
        log_info(f"✅ Total approved tokens: {len(approved_tokens)}/{len(all_tokens)}")
        
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
            
            log_info(f"🎯 Executing trades for {len(tokens_to_trade)} tokens")
            
            # Execute trades in parallel
            trade_tasks = [self._execute_trade_async(token) for token in tokens_to_trade]
            trade_results = await asyncio.gather(*trade_tasks, return_exceptions=True)
            
            # Update metrics for each trade
            for result in trade_results:
                if isinstance(result, Exception):
                    log_error(f"Trade execution error: {result}")
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
        
        log_info("📊 Trading cycle complete", **cycle_summary)
        return cycle_summary

async def run_enhanced_async_trading():
    """Main function to run enhanced async trading"""
    log_info("🌱 Starting Enhanced Async Trading Bot - Phase 3")
    
    # Start trading session
    start_trading_session()
    
    try:
        async with EnhancedAsyncTradingEngine() as engine:
            cycle_count = 0
            
            while True:
                cycle_count += 1
                log_info(f"🔄 Starting trading cycle #{cycle_count}")
                
                try:
                    result = await engine.run_enhanced_trading_cycle()
                    
                    if not result.get("success", False):
                        log_error(f"Cycle #{cycle_count} failed: {result.get('error', 'Unknown error')}")
                        await asyncio.sleep(30)  # Wait before retry
                        continue
                    
                    # Wait between cycles
                    wait_time = 300  # 5 minutes
                    log_info(f"⏰ Waiting {wait_time} seconds before next cycle...")
                    await asyncio.sleep(wait_time)
                    
                except Exception as e:
                    log_error(f"Error in trading cycle #{cycle_count}: {e}")
                    await asyncio.sleep(60)  # Wait before retry
                    
    except KeyboardInterrupt:
        log_info("🛑 Enhanced async trading stopped by user")
    except Exception as e:
        log_error(f"Fatal error in enhanced async trading: {e}")
    finally:
        end_trading_session()
        log_info("👋 Enhanced async trading shutdown complete")

if __name__ == "__main__":
    asyncio.run(run_enhanced_async_trading())
