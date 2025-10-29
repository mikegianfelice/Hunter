#!/usr/bin/env python3
"""
Comprehensive Integration Tests for Phase 3 & 4
Tests all new features and integrations
"""

import asyncio
import pytest
import json
import time
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.monitoring.structured_logger import log_info, log_error
from src.config.config_validator import get_validated_config, validate_config
from src.execution.enhanced_async_trading import EnhancedAsyncTradingEngine
from src.utils.advanced_cache import get_cache, cache_set, cache_get, cache_stats
from src.monitoring.realtime_dashboard import RealTimeDashboard
from src.ai.ai_integration_engine import AIIntegrationEngine, analyze_token_ai
from src.analytics.backtesting_engine import BacktestEngine, MarketDataSimulator
from src.deployment.production_manager import ProductionManager, HealthChecker

class TestPhase3Features:
    """Test Phase 3: Performance & Scalability features"""
    
    @pytest.mark.asyncio
    async def test_enhanced_async_trading_engine(self):
        """Test enhanced async trading engine"""
        log_info("Testing Enhanced Async Trading Engine")
        
        config = get_validated_config()
        
        async with EnhancedAsyncTradingEngine(config) as engine:
            # Test token fetching
            tokens = await engine.fetch_trending_tokens_async("ethereum", limit=5)
            assert len(tokens) > 0
            assert all("symbol" in token for token in tokens)
            
            # Test batch processing
            batch = tokens[:3]
            processed = await engine._process_token_batch(batch)
            assert len(processed) == len(batch)
            assert all("ai_analysis" in token for token in processed)
            
            # Test metrics collection
            initial_metrics = engine.metrics
            assert initial_metrics.total_trades == 0
            
            log_info("✅ Enhanced async trading engine test passed")
    
    @pytest.mark.asyncio
    async def test_advanced_cache_system(self):
        """Test advanced caching system"""
        log_info("Testing Advanced Cache System")
        
        cache = await get_cache()
        
        # Test basic operations
        await cache_set("test_key", {"data": "test_value"}, ttl=60)
        cached_value = await cache_get("test_key")
        assert cached_value["data"] == "test_value"
        
        # Test cache statistics
        stats = await cache_stats()
        assert "memory_items" in stats
        assert "metrics" in stats
        
        # Test cache with factory function
        def factory():
            return {"generated": time.time()}
        
        value1 = await cache.get_or_set("factory_key", factory, ttl=60)
        value2 = await cache.get_or_set("factory_key", factory, ttl=60)
        assert value1 == value2  # Should be cached
        
        log_info("✅ Advanced cache system test passed")
    
    @pytest.mark.asyncio
    async def test_realtime_dashboard(self):
        """Test real-time dashboard"""
        log_info("Testing Real-time Dashboard")
        
        dashboard = RealTimeDashboard()
        
        # Test metrics collection
        trading_metrics = await dashboard.collect_trading_metrics()
        assert trading_metrics is not None
        assert hasattr(trading_metrics, 'timestamp')
        
        system_metrics = await dashboard.collect_system_metrics()
        assert system_metrics is not None
        assert hasattr(system_metrics, 'cpu_usage')
        
        ai_metrics = await dashboard.collect_ai_metrics()
        assert ai_metrics is not None
        assert hasattr(ai_metrics, 'overall_healthy')
        
        # Test alert checking
        await dashboard.check_alerts()
        # Should not raise exceptions
        
        log_info("✅ Real-time dashboard test passed")

class TestPhase4Features:
    """Test Phase 4: AI Integration & Advanced Analytics features"""
    
    @pytest.mark.asyncio
    async def test_ai_integration_engine(self):
        """Test AI integration engine"""
        log_info("Testing AI Integration Engine")
        
        engine = AIIntegrationEngine()
        await engine.initialize()
        
        # Test token analysis
        token_data = {
            "symbol": "TEST",
            "priceUsd": 1.0,
            "volume24h": 100000,
            "marketCap": 1000000,
            "priceChange24h": 0.05,
            "liquidity": 500000,
            "holders": 1000,
            "transactions24h": 100,
            "social_mentions": 50,
            "news_sentiment": 0.7,
            "technical_indicators": {
                "rsi": 60,
                "macd": 0.1,
                "moving_avg_20": 0.95
            },
            "on_chain_metrics": {
                "active_addresses": 500,
                "transaction_volume": 10000
            }
        }
        
        result = await engine.analyze_token(token_data)
        assert result.symbol == "TEST"
        assert 0 <= result.overall_score <= 1
        assert 0 <= result.confidence <= 1
        assert "recommendations" in result.recommendations
        
        log_info("✅ AI integration engine test passed")
    
    @pytest.mark.asyncio
    async def test_backtesting_engine(self):
        """Test backtesting engine"""
        log_info("Testing Backtesting Engine")
        
        # Create test data
        symbols = ["TEST1", "TEST2"]
        simulator = MarketDataSimulator("2024-01-01", "2024-01-07", symbols)
        market_data = simulator.generate_market_data()
        
        # Test market data generation
        assert len(market_data) == 2
        for symbol, df in market_data.items():
            assert len(df) > 0
            assert "price" in df.columns
            assert "volume_24h" in df.columns
        
        # Test backtest engine
        engine = BacktestEngine()
        strategy_params = {
            "score_threshold": 0.7,
            "confidence_threshold": 0.8,
            "max_position_pct": 0.1
        }
        
        result = engine.run_backtest(strategy_params, market_data)
        assert result.total_trades >= 0
        assert 0 <= result.win_rate <= 1
        assert result.start_date == "2024-01-01"
        assert result.end_date == "2024-01-07"
        
        log_info("✅ Backtesting engine test passed")
    
    @pytest.mark.asyncio
    async def test_production_manager(self):
        """Test production manager"""
        log_info("Testing Production Manager")
        
        manager = ProductionManager()
        
        # Test health checker
        health_checker = HealthChecker()
        await health_checker.register_check("test_check", lambda: {"status": "healthy", "message": "Test check passed"})
        
        health_checks = await health_checker.run_all_checks()
        assert len(health_checks) > 0
        
        # Test system status
        status = await manager.get_system_status()
        assert status.overall_health in ["healthy", "warning", "unhealthy"]
        assert len(status.health_checks) > 0
        assert status.uptime >= 0
        
        log_info("✅ Production manager test passed")

class TestIntegrationScenarios:
    """Test complete integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_complete_trading_cycle(self):
        """Test complete trading cycle with all systems"""
        log_info("Testing Complete Trading Cycle")
        
        # Initialize all systems
        config = get_validated_config()
        
        # Test enhanced trading engine
        async with EnhancedAsyncTradingEngine(config) as engine:
            # Run one trading cycle
            result = await engine.run_enhanced_trading_cycle()
            assert result["success"] is True
            assert "tokens_fetched" in result
            assert "trades_executed" in result
        
        # Test AI integration
        ai_engine = AIIntegrationEngine()
        await ai_engine.initialize()
        
        test_token = {
            "symbol": "INTEGRATION_TEST",
            "priceUsd": 1.0,
            "volume24h": 100000,
            "marketCap": 1000000,
            "priceChange24h": 0.05,
            "liquidity": 500000,
            "holders": 1000,
            "transactions24h": 100,
            "social_mentions": 50,
            "news_sentiment": 0.7,
            "technical_indicators": {"rsi": 60, "macd": 0.1},
            "on_chain_metrics": {"active_addresses": 500}
        }
        
        ai_result = await analyze_token_ai(test_token)
        assert ai_result.symbol == "INTEGRATION_TEST"
        
        # Test caching
        cache = await get_cache()
        await cache_set("integration_test", {"test": "data"}, ttl=60)
        cached = await cache_get("integration_test")
        assert cached["test"] == "data"
        
        log_info("✅ Complete trading cycle test passed")
    
    @pytest.mark.asyncio
    async def test_backtest_optimization_workflow(self):
        """Test backtesting and optimization workflow"""
        log_info("Testing Backtest Optimization Workflow")
        
        symbols = ["TEST1", "TEST2"]
        start_date = "2024-01-01"
        end_date = "2024-01-31"
        
        # Run backtest
        from src.analytics.backtesting_engine import run_comprehensive_backtest, optimize_strategy
        
        result = await run_comprehensive_backtest(symbols, start_date, end_date)
        assert result.total_trades >= 0
        assert result.win_rate >= 0
        
        # Run optimization
        best_params = await optimize_strategy(symbols, start_date, end_date)
        assert best_params is not None
        assert len(best_params) > 0
        
        log_info("✅ Backtest optimization workflow test passed")
    
    @pytest.mark.asyncio
    async def test_production_health_monitoring(self):
        """Test production health monitoring"""
        log_info("Testing Production Health Monitoring")
        
        manager = ProductionManager()
        
        # Test individual health checks
        health_checker = manager.health_checker
        
        # Test database health
        db_health = await health_checker.check_database_health()
        assert db_health["status"] in ["healthy", "warning", "unhealthy"]
        
        # Test AI modules health
        ai_health = await health_checker.check_ai_modules_health()
        assert ai_health["status"] in ["healthy", "warning", "unhealthy"]
        
        # Test system resources
        system_health = await health_checker.check_system_resources()
        assert system_health["status"] in ["healthy", "warning", "unhealthy"]
        
        # Test trading system health
        trading_health = await health_checker.check_trading_system_health()
        assert trading_health["status"] in ["healthy", "warning", "unhealthy"]
        
        log_info("✅ Production health monitoring test passed")

class TestPerformanceMetrics:
    """Test performance and scalability metrics"""
    
    @pytest.mark.asyncio
    async def test_async_performance(self):
        """Test async performance"""
        log_info("Testing Async Performance")
        
        config = get_validated_config()
        
        async with EnhancedAsyncTradingEngine(config) as engine:
            start_time = time.time()
            
            # Test parallel token fetching
            tasks = [
                engine.fetch_trending_tokens_async("ethereum", limit=5),
                engine.fetch_trending_tokens_async("solana", limit=5),
                engine.fetch_trending_tokens_async("base", limit=5)
            ]
            
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            # Should complete in reasonable time
            assert end_time - start_time < 10  # 10 seconds max
            
            # All should return results
            assert all(len(result) > 0 for result in results)
            
            log_info(f"✅ Async performance test passed ({end_time - start_time:.2f}s)")
    
    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Test cache performance"""
        log_info("Testing Cache Performance")
        
        cache = await get_cache()
        
        # Test cache write performance
        start_time = time.time()
        for i in range(100):
            await cache_set(f"perf_test_{i}", {"data": f"value_{i}"}, ttl=60)
        write_time = time.time() - start_time
        
        # Test cache read performance
        start_time = time.time()
        for i in range(100):
            await cache_get(f"perf_test_{i}")
        read_time = time.time() - start_time
        
        # Should be fast
        assert write_time < 5  # 5 seconds max for 100 writes
        assert read_time < 2   # 2 seconds max for 100 reads
        
        log_info(f"✅ Cache performance test passed (write: {write_time:.2f}s, read: {read_time:.2f}s)")

async def run_all_tests():
    """Run all integration tests"""
    log_info("🧪 Starting Comprehensive Phase 3 & 4 Integration Tests")
    
    test_classes = [
        TestPhase3Features,
        TestPhase4Features,
        TestIntegrationScenarios,
        TestPerformanceMetrics
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for test_class in test_classes:
        log_info(f"Running {test_class.__name__}")
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            test_name = f"{test_class.__name__}.{test_method}"
            
            try:
                log_info(f"  Running {test_name}")
                
                # Create test instance and run method
                test_instance = test_class()
                test_func = getattr(test_instance, test_method)
                
                if asyncio.iscoroutinefunction(test_func):
                    await test_func()
                else:
                    test_func()
                
                passed_tests += 1
                log_info(f"  ✅ {test_name} passed")
                
            except Exception as e:
                failed_tests += 1
                log_error(f"  ❌ {test_name} failed: {e}")
    
    # Summary
    log_info("📊 Test Summary:")
    log_info(f"  Total Tests: {total_tests}")
    log_info(f"  Passed: {passed_tests}")
    log_info(f"  Failed: {failed_tests}")
    log_info(f"  Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    if failed_tests == 0:
        log_info("🎉 All tests passed!")
        return True
    else:
        log_error(f"❌ {failed_tests} tests failed")
        return False

if __name__ == "__main__":
    # Run tests
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
