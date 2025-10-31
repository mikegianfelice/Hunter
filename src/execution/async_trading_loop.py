#!/usr/bin/env python3
"""
Async Trading Loop for Trading Bot
Provides high-performance async trading with better resource utilization
"""

import asyncio
import aiohttp
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict

from src.ai.ai_circuit_breaker import circuit_breaker_manager
from src.monitoring.performance_monitor import performance_monitor, record_trade_metrics
from src.config.config_validator import get_validated_config

logger = logging.getLogger(__name__)

class AsyncTradingLoop:
    """
    High-performance async trading loop with circuit breaker protection
    """
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.config = get_validated_config()
        self.running = False
        self.trade_semaphore = asyncio.Semaphore(self.config.trading.max_concurrent_positions)
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(
                limit=100,  # Total connection pool size
                limit_per_host=30,  # Per-host connection limit
                ttl_dns_cache=300,  # DNS cache TTL
                use_dns_cache=True,
            ),
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def fetch_trending_tokens_async(self, chain: str, limit: int = 30) -> List[Dict[str, Any]]:
        """Fetch trending tokens asynchronously"""
        try:
            # Use circuit breaker for API calls
            result = circuit_breaker_manager.call_with_breaker(
                f"trending_tokens_{chain}",
                self._fetch_tokens_internal,
                chain, limit
            )
            return result
        except Exception as e:
            logger.error(f"Failed to fetch trending tokens for {chain}: {e}")
            return []
    
    async def _fetch_tokens_internal(self, chain: str, limit: int) -> List[Dict[str, Any]]:
        """Internal token fetching with proper async handling"""
        try:
            # Use real token scraper
            from src.utils.token_scraper import fetch_trending_tokens
            
            # Fetch tokens (this is synchronous but we can make it work in async context)
            import asyncio
            loop = asyncio.get_event_loop()
            tokens = await loop.run_in_executor(None, fetch_trending_tokens, limit)
            
            # Filter by chain
            filtered_tokens = [t for t in tokens if t.get('chainId', '').lower() == chain.lower()]
            
            return filtered_tokens[:limit]
            
        except Exception as e:
            logger.error(f"Error fetching tokens for {chain}: {e}")
            return []
    
    async def process_token_async(self, token: Dict[str, Any], market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single token asynchronously with circuit breaker protection"""
        async with self.trade_semaphore:
            try:
                symbol = token.get('symbol', 'UNKNOWN')
                start_time = time.time()
                
                # Use circuit breaker for AI analysis
                analysis_result = circuit_breaker_manager.call_with_breaker(
                    'ai_analysis',
                    self._analyze_token_internal,
                    token, market_data
                )
                
                execution_time = (time.time() - start_time) * 1000
                
                # Record performance metrics
                record_trade_metrics(
                    symbol=symbol,
                    chain=token.get('chain', 'unknown'),
                    amount_usd=token.get('amount_usd', 0),
                    success=analysis_result.get('success', False),
                    execution_time_ms=execution_time,
                    profit_loss_usd=analysis_result.get('profit_loss', 0),
                    quality_score=analysis_result.get('quality_score', 0),
                    risk_score=analysis_result.get('risk_score', 0),
                    error_message=analysis_result.get('error')
                )
                
                return analysis_result
                
            except Exception as e:
                logger.error(f"Error processing token {token.get('symbol', 'UNKNOWN')}: {e}")
                return None
    
    async def _analyze_token_internal(self, token: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Internal token analysis with real AI modules"""
        try:
            # Use real AI integration engine
            from src.ai.ai_integration_engine import analyze_token_ai
            
            # Prepare token data
            token_data = {
                "symbol": token.get('symbol', 'UNKNOWN'),
                "address": token.get('address', ''),
                "priceUsd": float(token.get('priceUsd', 0)),
                "volume24h": float(token.get('volume24h', 0)),
                "marketCap": float(token.get('marketCap', 0)),
                "liquidity": float(token.get('liquidity', 0)),
                "priceChange24h": float(token.get('priceChange24h', 0)),
                "chainId": token.get('chainId', 'unknown')
            }
            
            # Run AI analysis
            ai_result = await analyze_token_ai(token_data)
            
            # Determine if we should trade
            should_trade = (
                ai_result.overall_score > 0.7 and
                ai_result.confidence > 0.7 and
                ai_result.risk_assessment.get('risk_level', 'high') != 'high'
            )
            
            return {
                'success': True,
                'quality_score': ai_result.overall_score * 100,
                'risk_score': ai_result.risk_assessment.get('risk_score', 0.5),
                'profit_loss': 0.0,
                'should_trade': should_trade,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Error analyzing token: {e}")
            return {
                'success': False,
                'quality_score': 0.0,
                'risk_score': 1.0,
                'profit_loss': 0.0,
                'should_trade': False,
                'error': str(e)
            }
    
    async def execute_trade_async(self, token: Dict[str, Any], trade_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trade asynchronously with circuit breaker protection"""
        try:
            symbol = token.get('symbol', 'UNKNOWN')
            start_time = time.time()
            
            # Use circuit breaker for trade execution
            result = circuit_breaker_manager.call_with_breaker(
                'trade_execution',
                self._execute_trade_internal,
                token, trade_params
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            # Record trade metrics
            record_trade_metrics(
                symbol=symbol,
                chain=token.get('chain', 'unknown'),
                amount_usd=trade_params.get('amount_usd', 0),
                success=result.get('success', False),
                execution_time_ms=execution_time,
                profit_loss_usd=result.get('profit_loss', 0),
                quality_score=token.get('quality_score', 0),
                risk_score=token.get('risk_score', 0),
                error_message=result.get('error')
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing trade for {token.get('symbol', 'UNKNOWN')}: {e}")
            return {
                'success': False,
                'error': str(e),
                'profit_loss': 0.0
            }
    
    async def _execute_trade_internal(self, token: Dict[str, Any], trade_params: Dict[str, Any]) -> Dict[str, Any]:
        """Internal trade execution with real executors"""
        try:
            chain_id = token.get('chainId', '').lower()
            
            # Route to appropriate executor based on chain
            if chain_id == 'solana':
                from src.execution.solana_executor import execute_solana_trade
                tx_hash, success = await execute_solana_trade(
                    token.get('address'),
                    trade_params.get('amount_usd', 10),
                    trade_params.get('slippage', 0.1),
                    is_buy=True
                )
            elif chain_id == 'ethereum':
                from src.execution.uniswap_executor import execute_eth_trade
                tx_hash, success = await execute_eth_trade(
                    token.get('address'),
                    trade_params.get('amount_usd', 10),
                    trade_params.get('slippage', 0.1),
                    is_buy=True
                )
            elif chain_id == 'base':
                from src.execution.base_executor import execute_base_trade
                tx_hash, success = await execute_base_trade(
                    token.get('address'),
                    trade_params.get('amount_usd', 10),
                    trade_params.get('slippage', 0.1),
                    is_buy=True
                )
            else:
                return {
                    'success': False,
                    'transaction_hash': None,
                    'profit_loss': 0.0,
                    'error': f"Unsupported chain: {chain_id}"
                }
            
            return {
                'success': success,
                'transaction_hash': tx_hash,
                'profit_loss': 0.0,  # Will be calculated after sell
                'error': None if success else "Trade execution failed"
            }
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {
                'success': False,
                'transaction_hash': None,
                'profit_loss': 0.0,
                'error': str(e)
            }
    
    async def run_trading_cycle(self) -> Dict[str, Any]:
        """Run a single trading cycle asynchronously"""
        cycle_start = time.time()
        results = {
            'tokens_processed': 0,
            'trades_executed': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_profit_loss': 0.0,
            'cycle_duration': 0.0,
            'errors': []
        }
        
        try:
            # Fetch trending tokens for all supported chains
            supported_chains = self.config.chains.supported_chains
            fetch_tasks = [
                self.fetch_trending_tokens_async(chain, 10) 
                for chain in supported_chains
            ]
            
            all_tokens = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            
            # Flatten and filter results
            tokens = []
            for chain_tokens in all_tokens:
                if isinstance(chain_tokens, list):
                    tokens.extend(chain_tokens)
                elif isinstance(chain_tokens, Exception):
                    results['errors'].append(f"Failed to fetch tokens: {chain_tokens}")
            
            results['tokens_processed'] = len(tokens)
            
            if not tokens:
                logger.info("No tokens found in this cycle")
                return results
            
            # Process tokens in parallel batches
            batch_size = 5
            process_tasks = []
            
            for i in range(0, len(tokens), batch_size):
                batch = tokens[i:i+batch_size]
                batch_tasks = [
                    self.process_token_async(token, {}) 
                    for token in batch
                ]
                process_tasks.extend(batch_tasks)
            
            # Wait for all processing to complete
            processed_results = await asyncio.gather(*process_tasks, return_exceptions=True)
            
            # Filter successful results and execute trades
            trade_tasks = []
            for i, result in enumerate(processed_results):
                if isinstance(result, dict) and result.get('should_trade', False):
                    trade_params = {
                        'amount_usd': self.config.trading.trade_amount_usd,
                        'slippage': 0.1,
                        'take_profit': self.config.trading.take_profit,
                        'stop_loss': self.config.trading.stop_loss
                    }
                    trade_tasks.append(
                        self.execute_trade_async(tokens[i], trade_params)
                    )
            
            # Execute trades in parallel
            if trade_tasks:
                trade_results = await asyncio.gather(*trade_tasks, return_exceptions=True)
                
                for result in trade_results:
                    if isinstance(result, dict):
                        results['trades_executed'] += 1
                        if result.get('success', False):
                            results['successful_trades'] += 1
                            results['total_profit_loss'] += result.get('profit_loss', 0)
                        else:
                            results['failed_trades'] += 1
                    elif isinstance(result, Exception):
                        results['failed_trades'] += 1
                        results['errors'].append(f"Trade execution error: {result}")
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            results['errors'].append(str(e))
        
        finally:
            results['cycle_duration'] = time.time() - cycle_start
        
        return results
    
    async def run_continuous_loop(self, cycle_interval: int = 600):
        """Run continuous trading loop"""
        self.running = True
        logger.info(f"Starting continuous async trading loop (interval: {cycle_interval}s)")
        
        try:
            while self.running:
                cycle_start = time.time()
                
                # Run trading cycle
                results = await self.run_trading_cycle()
                
                # Log cycle results
                logger.info(
                    f"Trading cycle completed: "
                    f"{results['tokens_processed']} tokens, "
                    f"{results['trades_executed']} trades, "
                    f"{results['successful_trades']} successful, "
                    f"{results['cycle_duration']:.2f}s duration"
                )
                
                # Calculate sleep time
                cycle_duration = time.time() - cycle_start
                sleep_time = max(0, cycle_interval - cycle_duration)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    logger.warning(f"Cycle took {cycle_duration:.2f}s, longer than interval {cycle_interval}s")
        
        except KeyboardInterrupt:
            logger.info("Trading loop interrupted by user")
        except Exception as e:
            logger.error(f"Error in continuous trading loop: {e}")
        finally:
            self.running = False
            logger.info("Trading loop stopped")
    
    def stop(self):
        """Stop the trading loop"""
        self.running = False

# Convenience functions for easy integration
async def run_async_trading_loop(cycle_interval: int = 600):
    """Run the async trading loop"""
    async with AsyncTradingLoop() as trading_loop:
        await trading_loop.run_continuous_loop(cycle_interval)

async def run_single_cycle():
    """Run a single trading cycle"""
    async with AsyncTradingLoop() as trading_loop:
        return await trading_loop.run_trading_cycle()

# Integration with existing main.py
def start_async_trading():
    """Start async trading (can be called from main.py)"""
    try:
        asyncio.run(run_async_trading_loop())
    except KeyboardInterrupt:
        logger.info("Async trading stopped by user")
    except Exception as e:
        logger.error(f"Error starting async trading: {e}")

if __name__ == "__main__":
    start_async_trading()
