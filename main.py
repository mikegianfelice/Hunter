#!/usr/bin/env python3
"""
Enhanced Hunter Trading Bot - Phase 4 Complete
Production-ready trading bot with all Phase 3 & 4 features
"""

import asyncio
import os
import sys
import signal
import time
from datetime import datetime
from typing import Dict, Any, Optional
import logging

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.monitoring.structured_logger import log_info, log_error, start_logging_session, end_logging_session
from src.config.config_validator import get_validated_config, validate_config
from src.deployment.production_manager import ProductionManager
from src.execution.enhanced_async_trading import run_enhanced_async_trading
from src.monitoring.realtime_dashboard import start_realtime_dashboard
from src.analytics.backtesting_engine import run_comprehensive_backtest, optimize_strategy
from src.monitoring.telegram_bot import send_telegram_message
from src.utils.preflight_check import run_preflight_checks
from src.monitoring.metrics import init_metrics_server, record_preflight_failure
from src.monitoring.health_server import start_health_server, HealthServer

# Global variables for graceful shutdown
shutdown_event = asyncio.Event()
production_manager = None
health_server: Optional[HealthServer] = None


async def readiness_check() -> Dict[str, Any]:
    """
    Readiness probe used by the health server.
    """
    config_ok = validate_config()
    shutting_down = shutdown_event.is_set()
    production_active = bool(production_manager)
    latest_health = None

    if production_manager and getattr(production_manager, "status_history", None):
        latest_health = (
            production_manager.status_history[-1].overall_health
            if production_manager.status_history
            else None
        )

    ready = config_ok and not shutting_down

    return {
        "ready": ready,
        "config_valid": config_ok,
        "production_active": production_active,
        "shutdown_requested": shutting_down,
        "latest_health": latest_health,
    }

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    log_info("system.shutdown", f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_event.set()

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, signal_handler)

async def run_backtest_mode(symbols: list, start_date: str, end_date: str):
    """Run backtesting mode"""
    log_info("backtest.start", "🔬 Starting Backtesting Mode")
    log_info("backtest.config", f"Symbols: {', '.join(symbols)}")
    log_info("backtest.config", f"Period: {start_date} to {end_date}")
    
    try:
        # Run comprehensive backtest
        result = await run_comprehensive_backtest(symbols, start_date, end_date)
        
        # Display results
        log_info("backtest.results", "📊 Backtest Results:")
        log_info("backtest.results", f"  • Total Trades: {result.total_trades}")
        log_info("backtest.results", f"  • Win Rate: {result.win_rate:.2%}")
        log_info("backtest.results", f"  • Net PnL: ${result.net_pnl:.2f}")
        log_info("backtest.results", f"  • Max Drawdown: {result.max_drawdown:.2%}")
        log_info("backtest.results", f"  • Sharpe Ratio: {result.sharpe_ratio:.2f}")
        log_info("backtest.results", f"  • Profit Factor: {result.profit_factor:.2f}")
        
        # Optimize strategy
        log_info("backtest.optimization", "🔧 Optimizing Strategy...")
        best_params = await optimize_strategy(symbols, start_date, end_date)
        log_info("backtest.optimization", f"Best Parameters: {best_params}")
        
        return True
        
    except Exception as e:
        log_error("main.backtest", f"Backtest failed: {e}")
        return False

async def run_optimization_mode(symbols: list, start_date: str, end_date: str):
    """Run strategy optimization mode"""
    log_info("optimization.start", "⚡ Starting Strategy Optimization Mode")
    
    try:
        # Run optimization
        best_params = await optimize_strategy(symbols, start_date, end_date)
        
        log_info("optimization.results", "🎯 Optimization Results:")
        for param, value in best_params.items():
            log_info("optimization.results", f"  • {param}: {value}")
        
        return best_params
        
    except Exception as e:
        log_error("main.optimization", f"Optimization failed: {e}")
        return None

async def run_dashboard_mode(host: str = "localhost", port: int = 8765):
    """Run dashboard-only mode"""
    log_info("dashboard.start", "📊 Starting Dashboard Mode")
    log_info("dashboard.config", f"Dashboard URL: http://{host}:{port}")
    
    try:
        await start_realtime_dashboard(host, port)
        return True
    except Exception as e:
        log_error("main.dashboard", f"Dashboard failed: {e}")
        return False

async def run_production_mode():
    """Run full production mode"""
    log_info("production.start", "🚀 Starting Production Mode")
    
    global production_manager
    
    try:
        # Run preflight checks
        config = get_validated_config()
        supported_chains = getattr(config.chains, 'supported_chains', ['solana', 'ethereum', 'base'])
        preflight_result = await run_preflight_checks(chains=supported_chains)
        
        if not preflight_result["overall_ready"]:
            record_preflight_failure("production")
            log_error("main.preflight_failed", 
                     "❌ Preflight checks failed. Please fix errors before starting production mode.")
            return False
        
        # Initialize production manager
        production_manager = ProductionManager()
        
        # Start production system
        await production_manager.start_production_system()
        
        return True
        
    except Exception as e:
        log_error("main.production", f"Production mode failed: {e}")
        return False

async def run_enhanced_trading_mode():
    """Run enhanced async trading mode"""
    log_info("async_trading.start", "⚡ Starting Enhanced Async Trading Mode")
    
    try:
        # Run preflight checks
        config = get_validated_config()
        supported_chains = getattr(config.chains, 'supported_chains', ['solana', 'ethereum', 'base'])
        preflight_result = await run_preflight_checks(chains=supported_chains)
        
        if not preflight_result["overall_ready"]:
            record_preflight_failure("trading")
            log_error("main.preflight_failed", 
                     "❌ Preflight checks failed. Please fix errors before starting trading.")
            return False
        
        # Start price tracker
        from src.deployment.production_manager import ProductionManager
        pm = ProductionManager()
        price_tracker_task = asyncio.create_task(pm.run_price_tracker())
        
        # Start trading
        trading_task = asyncio.create_task(run_enhanced_async_trading())
        
        await asyncio.gather(price_tracker_task, trading_task, return_exceptions=True)
        return True
    except Exception as e:
        log_error("main.enhanced_trading", f"Enhanced trading failed: {e}")
        return False

def print_banner():
    """Print application banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    HUNTER TRADING BOT v4.0                  ║
║              AI-Enhanced Cryptocurrency Trading             ║
║                                                              ║
║  🚀 Phase 3: Performance & Scalability                     ║
║  🤖 Phase 4: AI Integration & Advanced Analytics           ║
║  📊 Real-time Monitoring & Production Ready                ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

def print_usage():
    """Print usage information"""
    usage = """
Usage: python main_enhanced.py [MODE] [OPTIONS]

Modes:
  production     Run full production system (default)
  trading        Run enhanced async trading only
  dashboard      Run real-time dashboard only
  backtest       Run backtesting mode
  optimize       Run strategy optimization
  health         Check system health

Options:
  --symbols SYMBOLS    Comma-separated list of symbols (for backtest/optimize)
  --start-date DATE    Start date for backtesting (YYYY-MM-DD)
  --end-date DATE      End date for backtesting (YYYY-MM-DD)
  --host HOST          Dashboard host (default: localhost)
  --port PORT          Dashboard port (default: 8765)

Examples:
  python main_enhanced.py production
  python main_enhanced.py trading
  python main_enhanced.py dashboard --host 0.0.0.0 --port 8080
  python main_enhanced.py backtest --symbols BTC,ETH,ADA --start-date 2024-01-01 --end-date 2024-12-31
  python main_enhanced.py optimize --symbols BTC,ETH --start-date 2024-01-01 --end-date 2024-06-30
  python main_enhanced.py health
"""
    print(usage)

async def check_system_health():
    """Check system health"""
    log_info("system.health", "🔍 Checking System Health")
    
    try:
        # Validate configuration
        if not validate_config():
            log_error("main.config_validation", "❌ Configuration validation failed")
            return False
        log_info("system.health", "✅ Configuration valid")
        
        # Check AI modules
        from src.ai.ai_circuit_breaker import check_ai_module_health
        ai_health = check_ai_module_health()
        if ai_health['overall_healthy']:
            log_info("system.health", "✅ AI modules healthy")
        else:
            log_info("system.health", f"⚠️ AI modules unhealthy: {ai_health['unhealthy_modules']}")
        
        # Check system resources
        import psutil
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        
        log_info("system.health", f"📊 System Resources:")
        log_info("system.health", f"  • CPU: {cpu_percent:.1f}%")
        log_info("system.health", f"  • Memory: {memory_percent:.1f}%")
        log_info("system.health", f"  • Disk: {disk_percent:.1f}%")
        
        if cpu_percent > 90 or memory_percent > 90 or disk_percent > 90:
            log_info("system.health", "⚠️ High resource usage detected")
        else:
            log_info("system.health", "✅ System resources normal")
        
        # Check required files
        required_files = ['config.yaml', 'requirements.txt']
        for file in required_files:
            if os.path.exists(file):
                log_info("system.health", f"✅ {file} found")
            else:
                log_error("main.health_check", f"❌ {file} missing")
                return False
        
        log_info("system.health", "🎉 System health check complete")
        return True
        
    except Exception as e:
        log_error("main.health_check", f"Health check failed: {e}")
        return False

async def main():
    """Main entry point"""
    print_banner()
    
    # Parse command line arguments
    mode = sys.argv[1] if len(sys.argv) > 1 else "production"
    
    # Parse options
    symbols = ["BTC", "ETH", "ADA", "SOL", "MATIC"]
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    host = "localhost"
    port = 8765
    
    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == "--symbols" and i + 1 < len(sys.argv):
            symbols = sys.argv[i + 1].split(",")
        elif arg == "--start-date" and i + 1 < len(sys.argv):
            start_date = sys.argv[i + 1]
        elif arg == "--end-date" and i + 1 < len(sys.argv):
            end_date = sys.argv[i + 1]
        elif arg == "--host" and i + 1 < len(sys.argv):
            host = sys.argv[i + 1]
        elif arg == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])
    
    # Setup signal handlers
    setup_signal_handlers()
    
    # Start logging session
    start_logging_session()

    # Initialize observability endpoints
    metrics_host = os.getenv("METRICS_HOST", "0.0.0.0")
    metrics_port = int(os.getenv("METRICS_PORT", "9100"))
    init_metrics_server(port=metrics_port, host=metrics_host)

    global health_server
    health_host = os.getenv("HEALTH_HOST", "0.0.0.0")
    health_port = int(os.getenv("HEALTH_PORT", "8081"))
    health_server = await start_health_server(
        host=health_host,
        port=health_port,
        readiness_check=readiness_check,
    )
    
    # Send Telegram notification on startup
    try:
        send_telegram_message("✅ Hunter Trading Bot Started\n\nBot is now running and monitoring markets.", deduplicate=False, message_type="status")
    except Exception as e:
        log_info("main.telegram", f"Could not send startup Telegram notification: {e}")
    
    try:
        # Route to appropriate mode
        if mode == "production":
            success = await run_production_mode()
        elif mode == "trading":
            success = await run_enhanced_trading_mode()
        elif mode == "dashboard":
            success = await run_dashboard_mode(host, port)
        elif mode == "backtest":
            success = await run_backtest_mode(symbols, start_date, end_date)
        elif mode == "optimize":
            result = await run_optimization_mode(symbols, start_date, end_date)
            success = result is not None
        elif mode == "health":
            success = await check_system_health()
        elif mode == "help":
            print_usage()
            success = True
        else:
            log_error("main.unknown_mode", f"Unknown mode: {mode}")
            print_usage()
            success = False
        
        if success:
            log_info("main.success", "✅ Operation completed successfully")
        else:
            log_error("main.operation_failed", "❌ Operation failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        log_info("main.interrupt", "🛑 Operation interrupted by user")
    except Exception as e:
        log_error("main.fatal_error", f"❌ Fatal error: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        if production_manager:
            try:
                await production_manager.auto_recovery.stop()
            except:
                pass
        
        end_logging_session()
        
        # Send Telegram notification on shutdown
        try:
            send_telegram_message("🛑 Hunter Trading Bot Stopped\n\nBot has been shut down.", deduplicate=False, message_type="status")
        except Exception as e:
            log_info("main.telegram", f"Could not send shutdown Telegram notification: {e}")
        
        if health_server:
            try:
                await health_server.stop()
            except Exception as err:
                log_error("main.health_server_stop", f"Failed to stop health server: {err}")
        
        log_info("main.shutdown", "👋 Hunter Trading Bot shutdown complete")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Shutdown complete")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
