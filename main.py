#!/usr/bin/env python3
"""
Practical Sustainable Trading Bot
Realistic approach for consistent 10-20% gains
"""

import os
import sys
import time
import json
from datetime import datetime
from collections import defaultdict
from typing import Dict, List

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from logger import log_event

# Import new systems from reorganized structure
from src.ai.ai_circuit_breaker import circuit_breaker_manager, check_ai_module_health, get_ai_module_status
from src.config.config_validator import config_validator, get_validated_config, validate_config
from src.monitoring.performance_monitor import performance_monitor, record_trade_metrics, start_trading_session, end_trading_session
from src.core.centralized_risk_manager import centralized_risk_manager, assess_trade_risk, update_trade_result, is_circuit_breaker_active
from src.monitoring.structured_logger import structured_logger, log_info, log_error, log_trade, log_performance, start_logging_session, end_logging_session

def _get_recent_trade_error(symbol: str) -> str:
    """Get the most recent error details for a token from logs"""
    try:
        import json
        import os
        from datetime import datetime, timedelta
        
        log_file = "logs/hunter.log"
        if not os.path.exists(log_file):
            return "No error details available"
        
        # Look for recent trade errors in the last 5 minutes
        cutoff_time = datetime.now() - timedelta(minutes=5)
        recent_errors = []
        
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())
                    if (log_entry.get('event') in ['trade.error', 'trade.slice.failure', 'trade.end'] and 
                        log_entry.get('level') in ['ERROR', 'WARNING'] and
                        symbol.lower() in str(log_entry.get('context', {})).lower()):
                        
                        # Parse timestamp
                        log_time = datetime.fromisoformat(log_entry['timestamp'].replace('Z', '+00:00'))
                        if log_time.replace(tzinfo=None) > cutoff_time:
                            error_msg = log_entry.get('context', {}).get('error', 'Unknown error')
                            recent_errors.append(error_msg)
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
        
        if recent_errors:
            return recent_errors[-1]  # Return the most recent error
        else:
            return "No recent error details found in logs"
            
    except Exception as e:
        return f"Error retrieving trade error details: {e}"

def log_print(msg):
    """Safe print function with error handling"""
    try:
        print(msg)
        log_event("main.info", message=msg, level="INFO")
    except OSError as e:
        if e.errno == 32:  # Broken pipe
            log_event("main.warning", message=f"Broken pipe error (errno 32): {e}", level="WARNING")
        else:
            log_event("main.error", message=f"OS error in print: {e}", level="ERROR")
    except Exception as e:
        # Handle any other print errors
        log_event("main.error", message=f"Print error: {e}", level="ERROR")

# Add live trading safety check
def check_live_trading_ready():
    """Verify system is ready for live trading"""
    log_print("🔍 Checking live trading readiness...")
    
    # Validate configuration first
    log_print("📋 Validating configuration...")
    if not validate_config():
        log_print("❌ ERROR: Configuration validation failed!")
        return False
    
    try:
        config = get_validated_config()
        log_print("✅ Configuration validation passed")
    except Exception as e:
        log_print(f"❌ ERROR: Configuration validation error: {e}")
        return False
    
    # Check AI module health
    log_print("🤖 Checking AI module health...")
    ai_health = check_ai_module_health()
    if not ai_health['overall_healthy']:
        log_print(f"⚠️ WARNING: {len(ai_health['unhealthy_modules'])} AI modules unhealthy")
        log_print(f"   Unhealthy modules: {', '.join(ai_health['unhealthy_modules'])}")
    else:
        log_print("✅ All AI modules healthy")
    
    # Check if test mode is disabled
    if config.test_mode:
        log_print("❌ ERROR: test_mode is still enabled! Disable in config.yaml")
        return False
    
    # Check wallet balance
    try:
        from src.core.risk_manager import _get_wallet_balance_usd
        eth_balance = _get_wallet_balance_usd("ethereum")
        sol_balance = _get_wallet_balance_usd("solana")
        log_print(f"💰 Wallet balances - ETH: ${eth_balance:.2f}, SOL: ${sol_balance:.2f}")
        
        if eth_balance < 10 and sol_balance < 10:
            log_print("⚠️ WARNING: Low wallet balance - ensure sufficient funds for trading")
    except Exception as e:
        log_print(f"⚠️ Could not check wallet balance: {e}")
    
    # Display configuration summary
    config_summary = config_validator.get_config_summary()
    log_print(f"⚙️ Configuration - Trade: ${config_summary['trading']['trade_amount_usd']}, TP: {config_summary['trading']['take_profit']*100:.0f}%, SL: {config_summary['trading']['stop_loss']*100:.0f}%")
    
    log_print("✅ Live trading system ready!")
    return True

# Import modules from reorganized structure
from src.utils.token_scraper import fetch_trending_tokens
from src.execution.multi_chain_executor import execute_trade
from src.monitoring.telegram_bot import send_telegram_message, send_periodic_status_report

# Import AI modules from reorganized structure
from src.ai.ai_sentiment_analyzer import AISentimentAnalyzer
from src.ai.ai_price_predictor import AIPricePredictor
from src.ai.ai_risk_assessor import AIRiskAssessor
from src.ai.ai_execution_optimizer import AIExecutionOptimizer
from src.ai.ai_microstructure_analyzer import AIMarketMicrostructureAnalyzer
from src.ai.ai_portfolio_optimizer import AIPortfolioOptimizer
from src.ai.ai_pattern_recognizer import AIPatternRecognizer
from src.ai.ai_market_intelligence_aggregator import AIMarketIntelligenceAggregator
from src.ai.ai_predictive_analytics_engine import AIPredictiveAnalyticsEngine
from src.ai.ai_dynamic_strategy_selector import AIDynamicStrategySelector
from src.ai.ai_risk_prediction_prevention_system import AIRiskPredictionPreventionSystem
from src.ai.ai_market_regime_transition_detector import AIMarketRegimeTransitionDetector
from src.ai.ai_liquidity_flow_analyzer import AILiquidityFlowAnalyzer
from src.ai.ai_multi_timeframe_analysis_engine import AIMultiTimeframeAnalysisEngine
from src.ai.ai_market_cycle_predictor import AIMarketCyclePredictor
from src.ai.ai_drawdown_protection_system import AIDrawdownProtectionSystem
from src.ai.ai_performance_attribution_analyzer import AIPerformanceAttributionAnalyzer
from src.ai.ai_market_anomaly_detector import AIMarketAnomalyDetector
from src.ai.ai_portfolio_rebalancing_engine import AIPortfolioRebalancingEngine
from src.ai.ai_emergency_stop_system import AIEmergencyStopSystem
from src.ai.ai_position_size_validator import AIPositionSizeValidator
from src.ai.ai_trade_execution_monitor import AITradeExecutionMonitor
from src.ai.ai_market_condition_guardian import AIMarketConditionGuardian
from src.ai.ai_market_regime_detector import AIMarketRegimeDetector

# Initialize AI modules
ai_sentiment_analyzer = AISentimentAnalyzer()
ai_price_predictor = AIPricePredictor()
ai_risk_assessor = AIRiskAssessor()
ai_execution_optimizer = AIExecutionOptimizer()
ai_microstructure_analyzer = AIMarketMicrostructureAnalyzer()
ai_portfolio_optimizer = AIPortfolioOptimizer()
ai_pattern_recognizer = AIPatternRecognizer()
ai_market_intelligence = AIMarketIntelligenceAggregator()
ai_predictive_analytics = AIPredictiveAnalyticsEngine()
ai_dynamic_strategy_selector = AIDynamicStrategySelector()
ai_risk_prediction_prevention = AIRiskPredictionPreventionSystem()
ai_market_regime_transition_detector = AIMarketRegimeTransitionDetector()
ai_liquidity_flow_analyzer = AILiquidityFlowAnalyzer()
ai_multi_timeframe_analysis_engine = AIMultiTimeframeAnalysisEngine()
ai_market_cycle_predictor = AIMarketCyclePredictor()
ai_drawdown_protection_system = AIDrawdownProtectionSystem()
ai_performance_attribution_analyzer = AIPerformanceAttributionAnalyzer()
ai_market_anomaly_detector = AIMarketAnomalyDetector()
ai_portfolio_rebalancing_engine = AIPortfolioRebalancingEngine()
ai_emergency_stop_system = AIEmergencyStopSystem()
ai_position_size_validator = AIPositionSizeValidator()
ai_trade_execution_monitor = AITradeExecutionMonitor()
ai_market_condition_guardian = AIMarketConditionGuardian()
ai_market_regime_detector = AIMarketRegimeDetector()

# Import other modules from reorganized structure
from src.core.performance_tracker import performance_tracker
from src.core.strategy import calculate_practical_quality_score, check_practical_buy_signal, get_dynamic_position_size, get_practical_take_profit

def practical_trade_loop():
    """
    Main trading loop focused on practical, sustainable gains with market regime awareness.
    """
    log_print("🌱 Starting AI-Enhanced Sustainable Trading Loop")
    log_print("🎯 Strategy: Consistent 10-20% gains with quality focus and market awareness")
    
    # Start performance monitoring session
    start_trading_session()
    
    # Check AI module health before starting
    ai_status = get_ai_module_status()
    log_print(f"🤖 AI Status: {ai_status}")
    
    # Get performance summary
    perf_summary = performance_monitor.get_performance_summary()
    log_print(f"📊 Performance: {perf_summary['real_time_metrics']['success_rate']:.1%} success rate, {perf_summary['real_time_metrics']['trades_per_hour']} trades/hour")
    
    # Check market regime first
    regime_data = ai_market_regime_detector.detect_market_regime()
    regime = regime_data['regime']
    confidence = regime_data['confidence']
    strategy = regime_data['strategy']
    
    # Initialize market_data for AI modules that require it
    market_data = {
        'timestamp': datetime.now().isoformat(),
        'regime': regime,
        'volatility': 0.2,
        'price': 0,  # Will be updated per token
        'volume': 0,  # Will be updated per token
        'liquidity': 0,  # Will be updated per token
        'current_price': 0,
        'volume_24h': 0,
        'avg_volume_24h': 0,
        'avg_volume_7d': 0,
        'avg_volume_30d': 0,
        'current_sentiment': 0.5,
        'sentiment_24h_ago': 0.5,
        'sentiment_7d_ago': 0.5,
        'sentiment_30d_ago': 0.5,
        'current_volatility': 0.2,
        'volatility_24h_ago': 0.2,
        'volatility_7d_ago': 0.2,
        'volatility_30d_ago': 0.2,
        'btc_correlation': 0.5,
        'eth_correlation': 0.5,
        'market_correlation': 0.5,
        'news_sentiment': 0.5,
        'news_impact_score': 0.5,
        'breaking_news_count': 0,
        'major_news_count': 0,
        'market_trend': 'neutral',
        'market_sentiment': 0.5
    }
    
    log_print(f"🎯 Market Regime: {regime} (confidence: {confidence:.2f}, strategy: {strategy})")
    
    # Check if trading should proceed in current regime
    should_trade, reason = ai_market_regime_detector.should_trade_in_current_regime()
    if not should_trade:
        log_print(f"⏸️ Trading paused: {reason}")
        return
    
    # Show regime recommendations
    recommendations = regime_data['recommendations']
    if recommendations:
        log_print(f"💡 Regime Recommendations:")
        for rec in recommendations[:3]:  # Show top 3 recommendations
            log_print(f"  • {rec}")
    
    # Check risk status
    from src.core.centralized_risk_manager import get_risk_summary
    risk_status = get_risk_summary()
    log_print(f"🧯 Risk status: {risk_status}")
    
    # Fetch trending tokens
    log_print("🔍 Fetching trending tokens...")
    tokens = fetch_trending_tokens(limit=30)  # Moderate limit for quality focus
    
    if not tokens:
        log_print("😴 No tokens found this cycle")
        return
    
    log_print(f"📊 Found {len(tokens)} tokens to evaluate")
    
    # Filter for practical sustainable opportunities
    practical_tokens = []
    
    for token in tokens:
        try:
            symbol = token.get("symbol", "UNKNOWN")
            address = token.get("address", "").lower()
            
            if not address:
                continue
            
            # Calculate AI-enhanced quality score
            quality_score = calculate_practical_quality_score(token)
            token['ai_enhanced_quality_score'] = quality_score
            
            # Check if token meets practical buy signal criteria
            if check_practical_buy_signal(token, regime_data):
                # Calculate practical position size
                position_size = get_dynamic_position_size(token, regime_data)
                token['practical_position_size'] = position_size
                
                # Calculate practical take profit
                tp = get_practical_take_profit(token)
                token['practical_tp'] = tp
                
                practical_tokens.append(token)
                
        except Exception as e:
            log_print(f"⚠️ Error processing token {token.get('symbol', 'UNKNOWN')}: {e}")
            continue
    
    if not practical_tokens:
        log_print("❌ No practical sustainable opportunities found")
        return
    
    # Sort by AI-enhanced quality score
    practical_tokens.sort(key=lambda x: x.get('ai_enhanced_quality_score', 0), reverse=True)
    
    log_print(f"✅ Found {len(practical_tokens)} practical opportunities")
    
    # Display top opportunities
    log_print("\n🎯 Top Opportunities:")
    for i, token in enumerate(practical_tokens[:5]):  # Show top 5
        symbol = token.get('symbol', 'UNKNOWN')
        quality = token.get('ai_enhanced_quality_score', 0)
        position_size = token.get('practical_position_size', 5)
        tp = token.get('practical_tp', 0.12)
        volume = float(token.get('volume24h', 0))
        liquidity = float(token.get('liquidity', 0))
        
        # Get AI analysis results for display
        sentiment = token.get('ai_sentiment', {})
        sentiment_category = sentiment.get('category', 'unknown')
        prediction = token.get('ai_prediction', {})
        success_prob = prediction.get('overall_success_probability', 0)
        prediction_confidence = prediction.get('confidence_level', 'unknown')
        
        log_print(f"  {i+1}. {symbol} - AI Quality: {quality:.1f}, Sentiment: {sentiment_category}, Prediction: {success_prob:.2f} ({prediction_confidence}), Position: ${position_size:.1f}, TP: {tp*100:.1f}%, Vol: ${volume:,.0f}, Liq: ${liquidity:,.0f}")
    
    # Track results
    rejections = defaultdict(list)
    successful_trades = []
    
    # Process top 3 opportunities (quality focus)
    for token in practical_tokens[:3]:
        try:
            symbol = token.get("symbol", "UNKNOWN")
            address = token.get("address", "").lower()
            quality_score = token.get("ai_enhanced_quality_score", 0)
            position_size = token.get("practical_position_size", 5)
            tp = token.get("practical_tp", 0.12)
            sentiment = token.get("ai_sentiment", {})
            sentiment_category = sentiment.get("category", "unknown")
            
            # Update market_data with current token data for AI modules
            price = float(token.get("priceUsd", 0))
            volume_24h = float(token.get("volume24h", 0))
            liquidity = float(token.get("liquidity", 0))
            market_data.update({
                'timestamp': datetime.now().isoformat(),
                'price': price,
                'volume': volume_24h,
                'liquidity': liquidity,
                'current_price': price,
                'volume_24h': volume_24h,
                'avg_volume_24h': volume_24h,
                'avg_volume_7d': volume_24h,
                'avg_volume_30d': volume_24h
            })
            
            # Use cached regime data to avoid duplicate calls
            # regime is already available from the initial detection above
            
            # Get prediction data for display
            prediction = token.get("ai_prediction", {})
            success_prob = prediction.get("overall_success_probability", 0)
            prediction_confidence = prediction.get("confidence_level", "unknown")
            expected_return = prediction.get("expected_return", 0)
            
            log_print(f"\n🎯 Processing {symbol} (AI Quality: {quality_score:.1f}, Sentiment: {sentiment_category}, Prediction: {success_prob:.2f} ({prediction_confidence}), Regime: {regime}, Position: ${position_size:.1f}, TP: {tp*100:.1f}%, Expected Return: {expected_return:.1f}%)")
            
            if not address:
                log_print("⚠️ Missing address - skipping")
                rejections["missing_address"].append(symbol)
                continue
            
            # Check if already holding this token
            if performance_tracker.is_token_held(address):
                log_print(f"⚠️ Already holding {symbol} - skipping")
                rejections["already_held"].append(symbol)
                continue
            
            # Check circuit breaker
            if is_circuit_breaker_active():
                log_print("⚠️ Circuit breaker active - skipping trades")
                rejections["circuit_breaker"].append(symbol)
                continue
            
            # Execute trade with new systems
            start_time = time.time()
            
            try:
                result = execute_trade(
                    token_address=address,
                    amount_usd=position_size,
                    chain=token.get("chain", "ethereum"),
                    take_profit=tp,
                    stop_loss=0.08,  # 8% stop loss
                    symbol=symbol
                )
                
                execution_time = (time.time() - start_time) * 1000
                
                if result.get("success", False):
                    log_print(f"✅ Trade successful: {symbol}")
                    successful_trades.append({
                        'symbol': symbol,
                        'amount': position_size,
                        'execution_time': execution_time,
                        'result': result
                    })
                    
                    # Record trade metrics
                    record_trade_metrics(
                        symbol=symbol,
                        chain=token.get("chain", "ethereum"),
                        amount_usd=position_size,
                        success=True,
                        execution_time_ms=execution_time,
                        profit_loss_usd=0.0,  # Will be updated when position closes
                        quality_score=quality_score,
                        risk_score=0.0  # Will be calculated by risk manager
                    )
                    
                    # Update risk manager
                    update_trade_result(True, 0.0)
                    
                else:
                    error_msg = result.get("error", "Unknown error")
                    log_print(f"❌ Trade failed: {symbol} - {error_msg}")
                    rejections["trade_failed"].append(f"{symbol}: {error_msg}")
                    
                    # Record failed trade metrics
                    record_trade_metrics(
                        symbol=symbol,
                        chain=token.get("chain", "ethereum"),
                        amount_usd=position_size,
                        success=False,
                        execution_time_ms=execution_time,
                        profit_loss_usd=0.0,
                        quality_score=quality_score,
                        risk_score=0.0,
                        error_message=error_msg
                    )
                    
                    # Update risk manager
                    update_trade_result(False, 0.0)
                    
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                error_msg = str(e)
                log_print(f"❌ Trade error: {symbol} - {error_msg}")
                rejections["trade_error"].append(f"{symbol}: {error_msg}")
                
                # Record error metrics
                record_trade_metrics(
                    symbol=symbol,
                    chain=token.get("chain", "ethereum"),
                    amount_usd=position_size,
                    success=False,
                    execution_time_ms=execution_time,
                    profit_loss_usd=0.0,
                    quality_score=quality_score,
                    risk_score=0.0,
                    error_message=error_msg
                )
                
                # Update risk manager
                update_trade_result(False, 0.0)
                
        except Exception as e:
            log_print(f"⚠️ Error processing {symbol}: {e}")
            rejections["processing_error"].append(f"{symbol}: {e}")
            continue
    
    # End trading session
    end_trading_session()
    
    # Summary
    log_print(f"\n📊 Trading Summary:")
    log_print(f"  • Opportunities found: {len(practical_tokens)}")
    log_print(f"  • Trades executed: {len(successful_trades)}")
    log_print(f"  • Rejections: {sum(len(v) for v in rejections.values())}")
    
    if rejections:
        log_print(f"\n❌ Rejection Summary:")
        for reason, tokens in rejections.items():
            if tokens:
                log_print(f"  • {reason}: {len(tokens)} tokens")
                for token in tokens[:3]:  # Show first 3
                    log_print(f"    - {token}")
                if len(tokens) > 3:
                    log_print(f"    ... and {len(tokens) - 3} more")

def main():
    """Main entry point"""
    log_print("🌱 Starting Sustainable Trading Bot - LIVE MODE")
    log_print("🎯 Strategy: Consistent 10-20% gains")
    log_print("📊 Focus: Quality over quantity")
    
    # Start structured logging session
    start_logging_session()
    
    # Check live trading readiness
    if not check_live_trading_ready():
        log_print("❌ System not ready for live trading. Exiting.")
        end_logging_session()
        return
    
    # Check if async trading is enabled
    try:
        config = get_validated_config()
        if hasattr(config, 'async_trading_enabled') and config.async_trading_enabled:
            log_print("🚀 Starting async trading mode...")
            from src.execution.async_trading_loop import start_async_trading
            start_async_trading()
            return
    except Exception as e:
        log_print(f"⚠️ Could not check async trading config: {e}")
    
    # Fall back to synchronous trading
    log_print("🔄 Starting synchronous trading mode...")
    
    # Send startup notification
    try:
        send_telegram_message(
            "🌱 Sustainable Trading Bot Started\n"
            "🎯 Strategy: 10-20% consistent gains\n"
            "📊 Quality over quantity approach\n"
            "🤖 AI-enhanced decision making\n"
            "🛡️ Advanced risk management"
        )
    except Exception as e:
        log_print(f"⚠️ Could not send startup notification: {e}")
    
    # Main trading loop
    try:
        while True:
            practical_trade_loop()
            
            # Wait before next cycle
            log_print("⏰ Waiting 10 minutes before next cycle...")
            time.sleep(600)  # 10 minutes
            
    except KeyboardInterrupt:
        log_print("🛑 Bot stopped by user")
    except Exception as e:
        log_print(f"❌ Fatal error: {e}")
    finally:
        end_logging_session()
        log_print("👋 Bot shutdown complete")

if __name__ == "__main__":
    main()
