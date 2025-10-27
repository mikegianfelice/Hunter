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
from logger import log_event

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
        return f"Error retrieving details: {str(e)}"

def log_print(msg):
    """Print to console and log to file"""
    print(msg)  # Print to console
    log_event("main.info", message=msg)  # Log to centralized logger

def safe_print(msg):
    """Print with broken pipe error handling"""
    try:
        print(msg)
    except BrokenPipeError:
        # Handle broken pipe error gracefully
        log_event("main.warning", message="Broken pipe error in print - connection may have been closed", level="WARNING")
        pass
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
    
    # Check if test mode is disabled
    from config_loader import get_config_bool
    if get_config_bool("test_mode", True):
        log_print("❌ ERROR: test_mode is still enabled! Disable in config.yaml")
        return False
    
    # Check wallet balance
    try:
        from risk_manager import _get_wallet_balance_usd
        eth_balance = _get_wallet_balance_usd("ethereum")
        sol_balance = _get_wallet_balance_usd("solana")
        log_print(f"💰 Wallet balances - ETH: ${eth_balance:.2f}, SOL: ${sol_balance:.2f}")
        
        if eth_balance < 10 and sol_balance < 10:
            log_print("⚠️ WARNING: Low wallet balance - ensure sufficient funds for trading")
    except Exception as e:
        log_print(f"⚠️ Could not check wallet balance: {e}")
    
    # Check configuration
    from config_loader import get_config_float
    trade_amount = get_config_float("trade_amount_usd", 5)
    take_profit = get_config_float("take_profit", 0.15)
    stop_loss = get_config_float("stop_loss", 0.08)
    
    log_print(f"⚙️ Configuration - Trade: ${trade_amount}, TP: {take_profit*100:.0f}%, SL: {stop_loss*100:.0f}%")
    
    log_print("✅ Live trading system ready!")
    return True

# Import modules
from token_scraper import fetch_trending_tokens
from multi_chain_executor import execute_trade
from telegram_bot import send_telegram_message, send_periodic_status_report
from risk_manager import allow_new_trade, register_buy, status_summary
from config_loader import get_config, get_config_float, get_config_int
from performance_tracker import performance_tracker
from ai_sentiment_analyzer import ai_sentiment_analyzer
from ai_market_regime_detector import ai_market_regime_detector
from ai_price_predictor import ai_price_predictor
from ai_portfolio_optimizer import ai_portfolio_optimizer
from ai_risk_assessor import ai_risk_assessor
from ai_pattern_recognizer import ai_pattern_recognizer
from ai_execution_optimizer import ai_execution_optimizer
from ai_microstructure_analyzer import ai_microstructure_analyzer
from ai_market_intelligence_aggregator import ai_market_intelligence_aggregator
from ai_predictive_analytics_engine import ai_predictive_analytics_engine
from ai_dynamic_strategy_selector import ai_dynamic_strategy_selector
from ai_risk_prediction_prevention_system import ai_risk_prediction_prevention_system
from ai_market_regime_transition_detector import ai_market_regime_transition_detector
from ai_liquidity_flow_analyzer import ai_liquidity_flow_analyzer
from ai_multi_timeframe_analysis_engine import ai_multi_timeframe_analysis_engine
from ai_market_cycle_predictor import ai_market_cycle_predictor
from ai_drawdown_protection_system import ai_drawdown_protection_system
from ai_performance_attribution_analyzer import ai_performance_attribution_analyzer
from ai_market_anomaly_detector import ai_market_anomaly_detector
from ai_portfolio_rebalancing_engine import ai_portfolio_rebalancing_engine
from ai_emergency_stop_system import ai_emergency_stop_system
from ai_position_size_validator import ai_position_size_validator
from ai_trade_execution_monitor import ai_trade_execution_monitor
from ai_market_condition_guardian import ai_market_condition_guardian
from market_data_fetcher import market_data_fetcher

def calculate_ai_enhanced_quality_score(token: Dict) -> float:
    """
    Calculate AI-enhanced quality score combining traditional metrics, sentiment analysis, and price prediction.
    This provides the most comprehensive assessment of token quality, sentiment, and success probability.
    """
    # Get traditional quality score (0-100)
    traditional_score = calculate_practical_quality_score(token)
    
    # Get AI sentiment analysis
    sentiment_analysis = ai_sentiment_analyzer.analyze_token_sentiment(token)
    sentiment_score = sentiment_analysis['score']  # 0-1
    sentiment_confidence = sentiment_analysis['confidence']  # 0-1
    
    # Get AI price prediction
    price_prediction = ai_price_predictor.predict_token_success(token)
    success_probability = price_prediction['overall_success_probability']  # 0-1
    prediction_confidence = price_prediction['confidence_level']
    
    # Convert scores to 0-100 scale
    sentiment_score_100 = sentiment_score * 100
    prediction_score_100 = success_probability * 100
    
    # Weight the scores based on confidence and importance
    # Traditional: 40%, Sentiment: 30%, Prediction: 30%
    traditional_weight = 0.4
    sentiment_weight = 0.3 + (sentiment_confidence * 0.1)  # 30-40% based on confidence
    prediction_weight = 0.3 + (0.1 if prediction_confidence == 'high' else 0)  # 30-40% based on confidence
    
    # Normalize weights
    total_weight = traditional_weight + sentiment_weight + prediction_weight
    traditional_weight /= total_weight
    sentiment_weight /= total_weight
    prediction_weight /= total_weight
    
    # Calculate weighted average
    ai_enhanced_score = (
        traditional_score * traditional_weight + 
        sentiment_score_100 * sentiment_weight + 
        prediction_score_100 * prediction_weight
    )
    
    # Apply AI-based adjustments
    sentiment_category = sentiment_analysis['category']
    prediction_confidence = price_prediction['confidence_level']
    
    # Sentiment adjustments
    if sentiment_category in ['very_positive', 'positive'] and sentiment_confidence > 0.7:
        ai_enhanced_score *= 1.05  # +5% for positive sentiment
    elif sentiment_category in ['very_negative', 'negative'] and sentiment_confidence > 0.7:
        ai_enhanced_score *= 0.95  # -5% for negative sentiment
    
    # Prediction adjustments
    if prediction_confidence == 'high' and success_probability > 0.75:
        ai_enhanced_score *= 1.1  # +10% for high-confidence high-probability predictions
    elif prediction_confidence == 'low' and success_probability < 0.45:
        ai_enhanced_score *= 0.9  # -10% for low-confidence low-probability predictions
    
    # Ensure score stays within bounds
    ai_enhanced_score = max(0, min(100, ai_enhanced_score))
    
    # Store AI data in token for later use
    token['ai_sentiment'] = sentiment_analysis
    token['ai_prediction'] = price_prediction
    token['ai_enhanced'] = True
    
    print(f"🧠 AI Enhanced Quality: {token.get('symbol', 'UNKNOWN')} - Traditional: {traditional_score:.1f}, Sentiment: {sentiment_score_100:.1f}, Prediction: {prediction_score_100:.1f}, Final: {ai_enhanced_score:.1f}")
    
    return ai_enhanced_score

def calculate_practical_quality_score(token: Dict) -> float:
    """
    Calculate a practical quality score (0-100) for token selection.
    Focus on volume, liquidity, and price stability.
    """
    volume_24h = float(token.get("volume24h", 0))
    liquidity = float(token.get("liquidity", 0))
    price = float(token.get("priceUsd", 0))
    symbol = token.get("symbol", "").upper()
    chain_id = token.get("chainId", "ethereum").lower()
    
    score = 0.0
    
    # Volume scoring (0-40 points)
    if volume_24h >= 1000000:  # $1M+ volume
        score += 40
    elif volume_24h >= 500000:  # $500k+ volume
        score += 35
    elif volume_24h >= 100000:  # $100k+ volume
        score += 30
    elif volume_24h >= 50000:   # $50k+ volume
        score += 25
    elif volume_24h >= 25000:   # $25k+ volume
        score += 20
    elif volume_24h >= 10000:   # $10k+ volume
        score += 15
    else:
        score += 0
    
    # Liquidity scoring (0-40 points)
    if liquidity >= 2000000:  # $2M+ liquidity
        score += 40
    elif liquidity >= 1000000:  # $1M+ liquidity
        score += 35
    elif liquidity >= 500000:  # $500k+ liquidity
        score += 30
    elif liquidity >= 250000:  # $250k+ liquidity
        score += 25
    elif liquidity >= 100000:  # $100k+ liquidity
        score += 20
    elif liquidity >= 50000:   # $50k+ liquidity
        score += 15
    else:
        score += 0
    
    # Price stability scoring (0-20 points)
    if price >= 0.01:  # $0.01+ tokens
        score += 20
    elif price >= 0.001:  # $0.001+ tokens
        score += 15
    elif price >= 0.0001:  # $0.0001+ tokens
        score += 10
    elif price >= 0.00001:  # $0.00001+ tokens
        score += 5
    else:
        score += 0
    
    return max(0, min(100, score))

def check_practical_buy_signal(token: Dict, regime_data: Dict = None) -> bool:
    """
    Check if token meets practical sustainable trading criteria with market regime awareness.
    Focus on quality over quantity, but be realistic and market-aware.
    """
    address = token.get("address", "").lower()
    price = float(token.get("priceUsd", 0))
    volume_24h = float(token.get("volume24h", 0))
    liquidity = float(token.get("liquidity", 0))
    symbol = token.get("symbol", "")
    
    # Initialize market_data with comprehensive default values - DO NOT REDEFINE
    market_data = {
        'timestamp': datetime.now().isoformat(),
        'regime': 'normal',
        'volatility': 0.2,
        'price': price,
        'volume': volume_24h,
        'liquidity': liquidity,
        'current_price': price,
        'volume_24h': volume_24h,
        'avg_volume_24h': volume_24h,
        'avg_volume_7d': volume_24h,
        'avg_volume_30d': volume_24h,
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
    
    safe_print(f"🔍 Evaluating {symbol} for practical sustainable trading...")
    
    # === SAFETY SYSTEM CHECKS ===
    
    # 1. AI Emergency Stop System Check
    try:
        portfolio_data = {
            'total_value': sum(pos.get('position_size_usd', 0) for pos in performance_tracker.get_open_trades()),
            'initial_value': 10000,  # Mock initial value
            'timestamp': datetime.now().isoformat()
        }
        trade_history = performance_tracker.get_trade_history() if hasattr(performance_tracker, 'get_trade_history') else []
        system_errors = []  # Mock system errors
        # Use existing market_data - do not redefine
        
        emergency_analysis = ai_emergency_stop_system.check_emergency_conditions(
            portfolio_data, trade_history, market_data, system_errors
        )
        
        if emergency_analysis['emergency_level'] in ['emergency', 'critical']:
            safe_print(f"🚨 {symbol}: Emergency stop activated - {emergency_analysis['emergency_level']}")
            return False
            
    except Exception as e:
        safe_print(f"⚠️ Emergency stop check failed: {e}")
    
    # 2. AI Market Condition Guardian Check
    try:
        # Update market_data with current values - do not redefine
        # Use real market volatility
        try:
            real_vol = market_data_fetcher.get_market_volatility(hours=24)
        except Exception:
            real_vol = 0.3
        market_data.update({
            'volatility': real_vol,
            'timestamp': datetime.now().isoformat()
        })
        news_data = {'sentiment': 'neutral', 'impact': 0.3}
        historical_data = {'price': price, 'volume': volume_24h}
        
        guardian_analysis = ai_market_condition_guardian.check_market_conditions(
            market_data, token, news_data, historical_data
        )
        
        if not guardian_analysis['trading_safety'] in ['safe', 'cautious']:
            safe_print(f"🛡️ {symbol}: Market conditions unsafe - {guardian_analysis['condition_level']}")
            return False
            
    except Exception as e:
        safe_print(f"⚠️ Market guardian check failed: {e}")
    
    # 3. AI Position Size Validator Check
    try:
        proposed_amount = get_config_float("trade_amount_usd", 5.0)
        # Get real wallet balance for the token's chain
        chain_id = token.get("chainId", "ethereum").lower()
        from risk_manager import _get_wallet_balance_usd
        wallet_balance = _get_wallet_balance_usd(chain_id)
        current_positions = performance_tracker.get_open_trades()
        market_conditions = {'regime': 'normal', 'volatility': 0.2}
        
        validation_analysis = ai_position_size_validator.validate_position_size(
            token, proposed_amount, wallet_balance, current_positions, market_conditions
        )
        
        if validation_analysis['validation_result'] in ['critical', 'rejected']:
            print(f"🔍 {symbol}: Position size validation failed - {validation_analysis['validation_result']}")
            return False
            
    except Exception as e:
        print(f"⚠️ Position size validation failed: {e}")
    
    # 4. AI Trade Execution Monitor Check
    try:
        trade_data = {
            'trade_id': f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'status': 'pending',
            'success': False,
            'start_time': datetime.now().isoformat(),
            'retry_count': 0
        }
        execution_history = []
        market_conditions = {'regime': 'normal'}
        
        monitoring_analysis = ai_trade_execution_monitor.monitor_trade_execution(
            trade_data, execution_history, market_conditions
        )
        
        if monitoring_analysis['monitoring_level'] in ['critical', 'emergency']:
            print(f"🔍 {symbol}: Execution monitoring critical - {monitoring_analysis['monitoring_level']}")
            return False
            
    except Exception as e:
        print(f"⚠️ Execution monitoring failed: {e}")
    
    # === END SAFETY SYSTEM CHECKS ===
    
    # Basic requirements
    if not address or price <= 0.00001:  # $0.00001 minimum price
        safe_print(f"❌ {symbol}: Missing address or price too low (${price:.6f})")
        return False
    
    # Use provided regime data or get it if not provided
    if regime_data is None:
        regime_data = ai_market_regime_detector.detect_market_regime()
    regime = regime_data['regime']
    quality_threshold_adjustment = regime_data['quality_threshold_adjustment']
    
    # Update market_data with detected regime - do not redefine
    # Refresh with real volatility
    try:
        real_vol = market_data_fetcher.get_market_volatility(hours=24)
    except Exception:
        real_vol = 0.3
    market_data.update({
        'timestamp': datetime.now().isoformat(),
        'regime': regime,
        'volatility': real_vol
    })
    
    # Dynamic volume requirement based on market regime
    base_volume_requirement = 25000
    if regime == 'bear_market':
        volume_requirement = base_volume_requirement * 1.5  # Higher volume in bear market
    elif regime == 'high_volatility':
        volume_requirement = base_volume_requirement * 2.0  # Much higher volume in high volatility
    else:
        volume_requirement = base_volume_requirement
    
    if volume_24h < volume_requirement:
        print(f"❌ {symbol}: Volume too low for {regime} regime (${volume_24h:,.0f} < ${volume_requirement:,.0f})")
        return False
    
    # Dynamic liquidity requirement based on market regime
    base_liquidity_requirement = 75000
    if regime == 'bear_market':
        liquidity_requirement = base_liquidity_requirement * 1.5
    elif regime == 'high_volatility':
        liquidity_requirement = base_liquidity_requirement * 2.0
    else:
        liquidity_requirement = base_liquidity_requirement
    
    if liquidity < liquidity_requirement:
        print(f"❌ {symbol}: Liquidity too low for {regime} regime (${liquidity:,.0f} < ${liquidity_requirement:,.0f})")
        return False
    
    # AI-enhanced quality score requirement with regime adjustment
    quality_score = calculate_ai_enhanced_quality_score(token)
    adjusted_quality_threshold = 45 + quality_threshold_adjustment
    
    if quality_score < adjusted_quality_threshold:
        print(f"❌ {symbol}: AI-enhanced quality score too low for {regime} regime ({quality_score:.1f} < {adjusted_quality_threshold})")
        return False
    
    # AI risk assessment
    risk_assessment = ai_risk_assessor.assess_token_risk(token)
    risk_score = risk_assessment['overall_risk_score']
    risk_category = risk_assessment['risk_category']
    should_trade = risk_assessment['should_trade']
    
    if not should_trade:
        print(f"❌ {symbol}: High risk detected (score: {risk_score:.2f}, category: {risk_category})")
        return False
    
    if risk_category == 'high_risk':
        print(f"❌ {symbol}: High risk category - avoiding trade")
        return False
    
    # AI pattern recognition
    pattern_recognition = ai_pattern_recognizer.recognize_patterns(token)
    pattern_strength = pattern_recognition['pattern_strength']
    overall_signal = pattern_recognition['overall_signal']
    confidence_level = pattern_recognition['confidence_level']
    
    # Check for strong negative patterns
    if overall_signal in ['strong_sell', 'sell'] and pattern_strength > 0.6:
        print(f"❌ {symbol}: Bearish pattern detected ({overall_signal}, strength: {pattern_strength:.2f})")
        return False
    
    # Require positive patterns for trading
    if overall_signal not in ['strong_buy', 'buy', 'weak_buy'] and pattern_strength > 0.4:
        print(f"❌ {symbol}: No bullish pattern detected ({overall_signal}, strength: {pattern_strength:.2f})")
        return False
    
    # AI microstructure analysis
    microstructure_analysis = ai_microstructure_analyzer.analyze_market_microstructure(token, 5.0)
    microstructure_score = microstructure_analysis['microstructure_score']
    execution_recommendation = microstructure_analysis['execution_recommendations']['execution_recommendation']
    risk_category = microstructure_analysis['risk_metrics']['risk_category']
    
    # Check microstructure execution recommendation
    if execution_recommendation == 'avoid_execution':
        print(f"❌ {symbol}: Microstructure analysis recommends avoiding execution")
        return False
    
    # Check microstructure risk
    if risk_category in ['very_high', 'high']:
        print(f"❌ {symbol}: High microstructure risk detected ({risk_category})")
        return False
    
    # AI market intelligence analysis
    intelligence_analysis = ai_market_intelligence_aggregator.analyze_market_intelligence(token, 5.0)
    intelligence_score = intelligence_analysis['intelligence_score']
    trading_recommendation = intelligence_analysis['trading_recommendations']['trading_recommendation']
    market_sentiment = intelligence_analysis['market_sentiment']['sentiment_category']
    
    # Check market intelligence trading recommendation
    if trading_recommendation in ['sell', 'strong_sell']:
        print(f"❌ {symbol}: Market intelligence recommends {trading_recommendation}")
        return False
    
    # Check market sentiment
    if market_sentiment in ['very_negative', 'negative']:
        print(f"❌ {symbol}: Negative market sentiment detected ({market_sentiment})")
        return False
    
    # AI predictive analytics analysis
    prediction_analysis = ai_predictive_analytics_engine.predict_price_movement(token, 5.0)
    prediction_score = prediction_analysis['prediction_score']
    trading_signal = prediction_analysis['trading_signals']['trading_signal']
    confidence_level = prediction_analysis['confidence_level']
    
    # Check predictive analytics trading signal
    if trading_signal in ['sell', 'strong_sell']:
        print(f"❌ {symbol}: Predictive analytics recommends {trading_signal}")
        return False
    
    # Check prediction confidence
    if confidence_level == 'low' and prediction_score < 0.4:
        print(f"❌ {symbol}: Low prediction confidence ({confidence_level}) with score {prediction_score:.2f}")
        return False
    
    # AI dynamic strategy selection
    market_conditions = {
        'bull_market_probability': 0.6 if regime == 'bull' else 0.3,
        'bear_market_probability': 0.3 if regime == 'bear' else 0.6,
        'sideways_market_probability': 0.4 if regime == 'sideways' else 0.3,
        'volatile_market_probability': 0.5,
        'volatility_score': 0.5,
        'volatility_trend': 'stable',
        'volatility_regime': 'moderate'
    }
    
    strategy_selection = ai_dynamic_strategy_selector.select_optimal_strategy(token, 5.0, market_conditions)
    selected_strategy = strategy_selection['selected_strategy']
    strategy_confidence = strategy_selection['strategy_confidence']
    
    # Check strategy confidence
    if strategy_confidence == 'low' and selected_strategy in ['breakout_strategy', 'momentum_strategy']:
        print(f"❌ {symbol}: Low strategy confidence ({strategy_confidence}) for {selected_strategy}")
        return False
    
    # AI risk prediction and prevention analysis
    risk_analysis = ai_risk_prediction_prevention_system.predict_risk(token, 5.0)
    risk_score = risk_analysis['risk_score']
    risk_level = risk_analysis['risk_level']
    risk_confidence = risk_analysis['risk_confidence']
    
    # Check critical risk levels
    if risk_level in ['critical', 'high']:
        print(f"❌ {symbol}: High risk detected ({risk_level}) with score {risk_score:.2f}")
        return False
    
    # Check risk confidence for medium risk
    if risk_level == 'medium' and risk_confidence == 'high' and risk_score > 0.6:
        print(f"❌ {symbol}: Medium risk with high confidence ({risk_score:.2f})")
        return False
    
    # AI market regime transition detection
    transition_analysis = ai_market_regime_transition_detector.detect_regime_transition(token, 5.0)
    transition_probability = transition_analysis['transition_probability']
    transition_confidence = transition_analysis['transition_confidence']
    
    # Check for high probability regime transition
    if transition_probability > 0.8 and transition_confidence == 'high':
        print(f"❌ {symbol}: High probability regime transition detected ({transition_probability:.2f})")
        return False
    
    # AI liquidity flow analysis
    liquidity_analysis = ai_liquidity_flow_analyzer.analyze_liquidity_flow(token, 5.0)
    liquidity_score = liquidity_analysis['liquidity_flow_score']
    liquidity_recommendations = liquidity_analysis['liquidity_recommendations']
    
    # Check liquidity flow recommendation
    if any('avoid trading' in rec.lower() for rec in liquidity_recommendations):
        print(f"❌ {symbol}: Liquidity flow analysis recommends avoiding trading")
        return False
    
    # AI multi-timeframe analysis
    timeframe_analysis = ai_multi_timeframe_analysis_engine.analyze_multi_timeframe(token, 5.0, market_data)
    timeframe_score = timeframe_analysis['overall_score']
    timeframe_confirmation = timeframe_analysis['signal_confirmation']['confirmation_direction']
    
    # Check multi-timeframe signal
    if timeframe_confirmation in ['sell', 'strong_sell']:
        print(f"❌ {symbol}: Multi-timeframe analysis recommends {timeframe_confirmation}")
        return False
    
    # AI market cycle prediction
    cycle_analysis = ai_market_cycle_predictor.predict_market_cycle(token, 5.0, market_data)
    cycle_phase = cycle_analysis['current_cycle_phase']
    cycle_confidence = cycle_analysis.get('cycle_confidence', 'medium')  # Default to medium if not available
    
    # Check market cycle phase
    if cycle_phase in ['decline', 'trough'] and cycle_confidence == 'high':
        print(f"❌ {symbol}: Unfavorable market cycle phase ({cycle_phase})")
        return False
    
    # AI drawdown protection analysis
    portfolio_data = {
        'total_value': sum(pos.get('position_size_usd', 0) for pos in performance_tracker.get_open_trades()),
        'position_count': len(performance_tracker.get_open_trades()),
        'timestamp': datetime.now().isoformat()
    }
    trade_history = performance_tracker.get_trade_history()
    # Update market_data with regime info - do not redefine
    market_data.update({'timestamp': datetime.now().isoformat(), 'regime': regime})
    
    drawdown_analysis = ai_drawdown_protection_system.analyze_drawdown_protection(portfolio_data, trade_history, market_data)
    drawdown_severity = drawdown_analysis['drawdown_severity']
    protection_urgency = drawdown_analysis['protection_urgency']
    
    # Check drawdown protection urgency
    if protection_urgency in ['urgent', 'emergency']:
        print(f"❌ {symbol}: High drawdown protection urgency ({protection_urgency})")
        return False
    
    # AI performance attribution analysis
    attribution_analysis = ai_performance_attribution_analyzer.analyze_performance_attribution(token, 5.0)
    attribution_score = attribution_analysis['attribution_score']
    performance_recommendation = attribution_analysis['performance_recommendations']['performance_recommendation']
    
    # Check performance attribution recommendation
    if performance_recommendation == 'avoid_trading':
        print(f"❌ {symbol}: Performance attribution analysis recommends avoiding trading")
        return False
    
    # AI market anomaly detection
    anomaly_analysis = ai_market_anomaly_detector.detect_market_anomalies(token, market_data, {})
    anomaly_score = anomaly_analysis['anomaly_score']
    anomaly_severity = anomaly_analysis['anomaly_severity']
    
    # Check for high anomaly severity
    if anomaly_severity in ['major', 'extreme']:
        print(f"❌ {symbol}: High anomaly severity detected ({anomaly_severity})")
        return False
    
    # AI portfolio rebalancing analysis
    current_positions = []
    for pos in performance_tracker.get_open_trades():
        current_positions.append({
            'symbol': pos.get('symbol', 'UNKNOWN'),
            'position_size_usd': pos.get('position_size_usd', 0),
            'entry_price': pos.get('entry_price', 0),
            'volume_24h': pos.get('volume_24h', 0),
            'liquidity': pos.get('liquidity', 0),
            'chainId': pos.get('chain_id', 'ethereum')
        })
    
    rebalancing_analysis = ai_portfolio_rebalancing_engine.optimize_portfolio_allocation(current_positions, market_data)
    rebalancing_urgency = rebalancing_analysis['rebalancing_needs']['rebalancing_urgency']
    portfolio_efficiency = rebalancing_analysis['portfolio_efficiency']
    
    # Check portfolio rebalancing urgency
    if rebalancing_urgency in ['urgent', 'emergency']:
        print(f"❌ {symbol}: High portfolio rebalancing urgency ({rebalancing_urgency})")
        return False
    
    print(f"✅ {symbol}: Quality score {quality_score:.1f}, Volume ${volume_24h:,.0f}, Liquidity ${liquidity:,.0f}, Risk: {risk_category} ({risk_score:.2f}), Pattern: {overall_signal} ({pattern_strength:.2f}), Microstructure: {microstructure_score:.2f}, Intelligence: {intelligence_score:.2f}, Sentiment: {market_sentiment}, Prediction: {prediction_score:.2f} ({trading_signal}), Strategy: {selected_strategy} ({strategy_confidence}), Risk: {risk_level} ({risk_score:.2f}), Transition: {transition_probability:.2f}, Liquidity: {liquidity_score:.2f}, Timeframe: {timeframe_score:.2f}, Cycle: {cycle_phase}, Drawdown: {drawdown_severity}, Attribution: {attribution_score:.2f}, Anomaly: {anomaly_severity}, Rebalancing: {rebalancing_urgency} (Regime: {regime})")
    return True

def get_dynamic_position_size(token: Dict, regime_data: Dict = None) -> float:
    """
    Calculate dynamic position size based on AI-enhanced token quality, risk factors, market regime, and portfolio optimization.
    Higher quality tokens get larger positions, but with safety limits, market awareness, and portfolio optimization.
    """
    base_amount = 5.0  # $5 base position
    quality_score = calculate_ai_enhanced_quality_score(token)
    volume_24h = float(token.get("volume24h", 0))
    liquidity = float(token.get("liquidity", 0))
    
    # Use provided regime data or get it if not provided
    if regime_data is None:
        regime_data = ai_market_regime_detector.detect_market_regime()
    regime = regime_data['regime']
    position_multiplier = regime_data['position_multiplier']
    
    # Get portfolio optimization for position sizing
    try:
        # Get current open positions for portfolio analysis
        open_positions = performance_tracker.get_open_trades()
        
        # Create position data for portfolio optimization
        current_positions = []
        for pos in open_positions:
            current_positions.append({
                'symbol': pos.get('symbol', 'UNKNOWN'),
                'value': pos.get('position_size_usd', 0),
                'quality_score': pos.get('quality_score', 50),
                'risk_score': 0.5,  # Default risk score
                'expected_return': 0.12,  # Default expected return
                'sector': 'crypto'  # Default sector
            })
        
        # Add current token to analysis
        current_positions.append({
            'symbol': token.get('symbol', 'UNKNOWN'),
            'value': base_amount,
            'quality_score': quality_score,
            'risk_score': 0.5,
            'expected_return': 0.12,
            'sector': 'crypto'
        })
        
        # Get portfolio optimization
        available_capital = 50.0  # Assume $50 available capital
        portfolio_optimization = ai_portfolio_optimizer.optimize_portfolio(
            current_positions, available_capital
        )
        
        # Get optimized allocation for this token
        optimized_allocation = portfolio_optimization.get('optimized_allocation', {})
        token_symbol = token.get('symbol', 'UNKNOWN')
        
        if token_symbol in optimized_allocation:
            portfolio_weight = optimized_allocation[token_symbol]['weight']
            portfolio_position_size = optimized_allocation[token_symbol]['position_size']
            
            # Use portfolio-optimized position size
            position_size = portfolio_position_size
            print(f"🎯 Portfolio-optimized position size: ${position_size:.1f} (weight: {portfolio_weight:.1%})")
            
        else:
            # Fallback to standard calculation
            position_size = base_amount
            print(f"⚠️ Portfolio optimization unavailable, using base amount: ${position_size:.1f}")
            
    except Exception as e:
        print(f"⚠️ Portfolio optimization failed: {e}, using base amount")
        position_size = base_amount
    
    # Start with base amount if portfolio optimization failed
    if position_size == 0:
        position_size = base_amount
    
    # Apply AI risk assessment adjustment
    try:
        risk_assessment = ai_risk_assessor.assess_token_risk(token)
        risk_adjustment = risk_assessment['position_adjustment']
        risk_category = risk_assessment['risk_category']
        
        # Apply risk-based position adjustment
        position_size *= risk_adjustment
        
        if risk_category == 'medium_risk':
            print(f"⚠️ Risk adjustment: {risk_adjustment:.1%} (medium risk)")
        elif risk_category == 'low_risk':
            print(f"✅ Risk adjustment: {risk_adjustment:.1%} (low risk)")
            
    except Exception as e:
        print(f"⚠️ Risk assessment failed: {e}, using base position size")
    
    # Quality-based sizing (primary factor)
    if quality_score >= 80:  # Excellent quality
        position_size *= 1.5  # $7.50 (50% increase)
    elif quality_score >= 70:  # High quality
        position_size *= 1.3  # $6.50 (30% increase)
    elif quality_score >= 60:  # Good quality
        position_size *= 1.1  # $5.50 (10% increase)
    elif quality_score < 50:  # Lower quality
        position_size *= 0.8  # $4.00 (20% decrease)
    
    # Volume-based adjustments (secondary factor)
    if volume_24h >= 1000000:  # $1M+ volume - very liquid
        position_size *= 1.1  # +10% for high volume
    elif volume_24h >= 500000:  # $500k+ volume - good liquidity
        position_size *= 1.05  # +5% for good volume
    elif volume_24h < 50000:  # <$50k volume - lower liquidity
        position_size *= 0.9  # -10% for lower volume
    
    # Liquidity-based adjustments (tertiary factor)
    if liquidity >= 2000000:  # $2M+ liquidity - very safe
        position_size *= 1.05  # +5% for high liquidity
    elif liquidity < 100000:  # <$100k liquidity - riskier
        position_size *= 0.9  # -10% for lower liquidity
    
    # Apply market regime adjustments
    position_size *= position_multiplier
    
    # Apply safety limits
    min_position = 2.0  # $2 minimum
    max_position = 10.0  # $10 maximum (respects per_trade_max_usd limit)
    
    position_size = max(min_position, min(max_position, position_size))
    
    # Round to nearest $0.50 for practical trading
    position_size = round(position_size * 2) / 2
    
    symbol = token.get('symbol', 'UNKNOWN')
    print(f"💰 {symbol}: Position ${position_size:.1f} (quality: {quality_score:.1f}, regime: {regime}, multiplier: {position_multiplier:.1f}x)")
    return position_size

def get_practical_take_profit(token: Dict) -> float:
    """
    Calculate practical take profit based on AI-enhanced token quality.
    Focus on achievable 10-20% gains.
    """
    base_tp = 0.12  # 12% base
    quality_score = calculate_ai_enhanced_quality_score(token)
    volume_24h = float(token.get("volume24h", 0))
    liquidity = float(token.get("liquidity", 0))
    
    tp = base_tp
    
    # Quality-based adjustments
    if quality_score >= 70:
        tp += 0.03  # +3% for high quality
    elif quality_score >= 60:
        tp += 0.02  # +2% for good quality
    elif quality_score < 50:
        tp -= 0.02  # -2% for lower quality
    
    # Volume-based adjustments
    if volume_24h >= 500000:  # $500k+ volume
        tp += 0.02  # +2% for high volume
    elif volume_24h < 50000:  # <$50k volume
        tp -= 0.02  # -2% for low volume
    
    # Liquidity-based adjustments
    if liquidity >= 1000000:  # $1M+ liquidity
        tp += 0.02  # +2% for high liquidity
    elif liquidity < 100000:  # <$100k liquidity
        tp -= 0.02  # -2% for low liquidity
    
    # Apply limits
    tp_min = 0.08  # 8% minimum
    tp_max = 0.20  # 20% maximum
    
    tp = max(tp_min, min(tp_max, tp))
    
    print(f"🎯 {token.get('symbol', 'UNKNOWN')}: TP {tp*100:.1f}% (quality: {quality_score:.1f})")
    return tp

def practical_trade_loop():
    """
    Main trading loop focused on practical, sustainable gains with market regime awareness.
    """
    log_print("🌱 Starting AI-Enhanced Sustainable Trading Loop")
    log_print("🎯 Strategy: Consistent 10-20% gains with quality focus and market awareness")
    
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
    risk_status = status_summary()
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
        if check_practical_buy_signal(token, regime_data):
            # Add AI-enhanced quality score, position size, and take profit to token data
            token["ai_enhanced_quality_score"] = calculate_ai_enhanced_quality_score(token)
            token["practical_position_size"] = get_dynamic_position_size(token, regime_data)
            token["practical_tp"] = get_practical_take_profit(token)
            practical_tokens.append(token)
    
    if not practical_tokens:
        log_print("❌ No practical sustainable opportunities found")
        return
    
    # Sort by AI-enhanced quality score (highest first)
    practical_tokens.sort(key=lambda x: x.get("ai_enhanced_quality_score", 0), reverse=True)
    
    log_print(f"✅ Found {len(practical_tokens)} practical sustainable opportunities")
    
    # Show top opportunities
    for i, token in enumerate(practical_tokens[:5]):
        symbol = token.get("symbol", "UNKNOWN")
        quality = token.get("ai_enhanced_quality_score", 0)
        position_size = token.get("practical_position_size", 5)
        tp = token.get("practical_tp", 0)
        volume = float(token.get("volume24h", 0))
        liquidity = float(token.get("liquidity", 0))
        sentiment = token.get("ai_sentiment", {})
        sentiment_category = sentiment.get("category", "unknown")
        prediction = token.get("ai_prediction", {})
        success_prob = prediction.get("overall_success_probability", 0)
        prediction_confidence = prediction.get("confidence_level", "unknown")
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
            
            # Risk manager check with dynamic position size
            allowed, reason = allow_new_trade(position_size, address, token.get("chainId", "ethereum"))
            if not allowed:
                log_print(f"🛑 Risk manager blocked: {reason}")
                rejections["risk_blocked"].append(symbol)
                continue
            
            # Get AI execution optimization
            execution_optimization = ai_execution_optimizer.optimize_trade_execution(token, position_size)
            
            # Check if execution is recommended
            success_prediction = execution_optimization['success_prediction']
            if success_prediction['recommendation'] == 'avoid':
                log_print(f"❌ Execution not recommended: {success_prediction['success_category']} success probability")
                rejections["execution_blocked"].append(symbol)
                continue
            
            # Display execution optimization details
            optimal_route = execution_optimization['optimal_route']
            slippage_optimization = execution_optimization['slippage_optimization']
            gas_optimization = execution_optimization['gas_optimization']
            
            log_print(f"⚡ Execution optimization for {symbol}:")
            log_print(f"  • Route: {optimal_route['dex']} (score: {optimal_route['score']:.2f})")
            log_print(f"  • Slippage: {slippage_optimization['target_slippage']:.1%} (max: {slippage_optimization['max_slippage']:.1%})")
            log_print(f"  • Gas: ${gas_optimization['optimized_gas_cost']:.4f} (efficiency: {gas_optimization['gas_efficiency']:.1%})")
            log_print(f"  • Success probability: {success_prediction['success_probability']:.1%}")
            
            # Execute trade with dynamic position size
            log_print(f"🚀 Executing ${position_size:.1f} trade for {symbol}...")
            tx_hash, success = execute_trade(token, position_size)
            
            if success:
                successful_trades.append((symbol, address))
                register_buy(position_size)
                
                # Log trade entry for performance tracking
                trade_id = performance_tracker.log_trade_entry(token, position_size, quality_score)
                
                # Send notification
                send_telegram_message(
                    f"✅ AI-Enhanced Sustainable Trade\n"
                    f"Token: {symbol}\n"
                    f"AI Quality Score: {quality_score:.1f}\n"
                    f"Sentiment: {sentiment_category}\n"
                    f"Success Probability: {success_prob:.2f} ({prediction_confidence})\n"
                    f"Expected Return: {expected_return:.1f}%\n"
                    f"Market Regime: {regime}\n"
                    f"Position Size: ${position_size:.1f}\n"
                    f"Take Profit: {tp*100:.1f}%\n"
                    f"TX: {tx_hash}",
                    message_type="trade_success"
                )
                
                log_print(f"✅ Successfully traded {symbol} - AI Quality: {quality_score:.1f}, Sentiment: {sentiment_category}, Position: ${position_size:.1f}, TP: {tp*100:.1f}%")
                log_print(f"📊 Trade logged with ID: {trade_id}")
            else:
                log_print(f"❌ Trade failed for {symbol}")
                rejections["execution_failed"].append(symbol)
                
                # Send Telegram alert for execution failures
                try:
                    # Get more detailed error information from recent logs
                    error_details = _get_recent_trade_error(symbol)
                    
                    send_telegram_message(
                        f"❌ Trade Execution Failed\n"
                        f"Token: {symbol}\n"
                        f"AI Quality Score: {quality_score:.1f}\n"
                        f"Position Size: ${position_size:.1f}\n"
                        f"Status: Execution returned False\n"
                        f"Error: {error_details}\n"
                        f"Time: {datetime.now().strftime('%H:%M:%S')}",
                        message_type="trade_failure"
                    )
                except Exception as telegram_err:
                    log_print(f"⚠️ Failed to send failure notification: {telegram_err}")
            
            # Small delay between trades
            time.sleep(3)
            
        except Exception as e:
            error_msg = str(e) if e else "Unknown error"
            symbol = token.get('symbol', 'UNKNOWN') if token else 'UNKNOWN'
            log_print(f"🔥 Error processing {symbol}: {error_msg}")
            rejections["error"].append(symbol)
            
            # Send Telegram alert for execution errors
            try:
                send_telegram_message(
                    f"🔥 Trade Execution Error\n"
                    f"Token: {symbol}\n"
                    f"Error: {error_msg}\n"
                    f"⚠️ Manual intervention may be required",
                    message_type="trade_error"
                )
            except Exception as telegram_err:
                log_print(f"⚠️ Failed to send error notification: {telegram_err}")
    
    # Print summary
    _print_practical_summary(rejections, successful_trades)
    
    # Show performance insights
    _show_performance_insights()
    
    # Show portfolio optimization insights
    _show_portfolio_insights()
    
    # Show AI risk assessment insights
    _show_risk_insights()
    
    # Show AI pattern recognition insights
    _show_pattern_insights()
    
    # Show AI execution optimization insights
    _show_execution_insights()
    
    # Show AI microstructure analysis insights
    _show_microstructure_insights()
    
    # Show AI market intelligence insights
    _show_market_intelligence_insights()
    
    # Show AI predictive analytics insights
    _show_predictive_analytics_insights()
    
    # Show AI dynamic strategy selector insights
    _show_dynamic_strategy_insights()
    
    # Show AI risk prediction and prevention insights
    _show_risk_prediction_insights()
    
    # Show AI market regime transition insights
    _show_regime_transition_insights()
    
    # Show AI liquidity flow insights
    _show_liquidity_flow_insights()
    
    # Show AI multi-timeframe analysis insights
    _show_multi_timeframe_insights()
    
    # Show AI market cycle prediction insights
    _show_market_cycle_insights()
    
    # Show AI drawdown protection insights
    _show_drawdown_protection_insights()
    
    # Show AI performance attribution insights
    _show_performance_attribution_insights()
    
    # Show AI market anomaly detection insights
    _show_market_anomaly_insights()
    
    # Show AI portfolio rebalancing insights
    _show_portfolio_rebalancing_insights()

def _print_practical_summary(rejections, successful_trades):
    """Print trading summary"""
    total_evaluated = sum(len(v) for v in rejections.values()) + len(successful_trades)
    
    log_print("\n📋 Practical Sustainable Trading Summary")
    log_print(f"• Tokens evaluated: {total_evaluated}")
    log_print(f"• Successful trades: {len(successful_trades)}")
    
    if successful_trades:
        symbols = [s for s, _ in successful_trades]
        log_print(f"  ↳ {', '.join(symbols)}")
    
    if rejections:
        for reason, tokens in rejections.items():
            if tokens:
                log_print(f"• Rejected ({reason}): {len(tokens)}")
                log_print(f"  ↳ {', '.join(tokens[:3])}{'...' if len(tokens) > 3 else ''}")

def _show_performance_insights():
    """Show performance insights from recent trading"""
    try:
        # Get recent performance summary
        summary = performance_tracker.get_performance_summary(7)  # Last 7 days
        
        if summary['total_trades'] > 0:
            log_print("\n📊 Performance Insights (Last 7 Days)")
            log_print(f"• Total Trades: {summary['total_trades']}")
            log_print(f"• Win Rate: {summary['win_rate']:.1f}%")
            log_print(f"• Avg PnL: ${summary['avg_pnl']:.2f}")
            log_print(f"• Total PnL: ${summary['total_pnl']:.2f}")
            
            # Show quality tier performance
            if summary['quality_analysis']:
                log_print("\n🎯 Quality Tier Performance:")
                for tier, stats in summary['quality_analysis'].items():
                    if stats['trades'] > 0:
                        log_print(f"  • {tier.upper()}: {stats['trades']} trades, {stats['win_rate']:.1f}% win rate, ${stats['avg_pnl']:.2f} avg")
        else:
            log_print("\n📊 No recent trades to analyze yet")
            
    except Exception as e:
        log_print(f"⚠️ Could not generate performance insights: {e}")

def _show_portfolio_insights():
    """Show portfolio optimization insights"""
    try:
        # Get current open positions
        open_positions = performance_tracker.get_open_trades()
        
        if not open_positions:
            log_print("\n📊 Portfolio Insights: No open positions to analyze")
            return
        
        # Create position data for portfolio analysis
        current_positions = []
        for pos in open_positions:
            current_positions.append({
                'symbol': pos.get('symbol', 'UNKNOWN'),
                'value': pos.get('position_size_usd', 0),
                'quality_score': pos.get('quality_score', 50),
                'risk_score': 0.5,
                'expected_return': 0.12,
                'sector': 'crypto'
            })
        
        # Get portfolio insights
        available_capital = 50.0  # Assume $50 available capital
        portfolio_insights = ai_portfolio_optimizer.get_portfolio_insights(
            current_positions, available_capital
        )
        
        log_print("\n📊 Portfolio Optimization Insights")
        log_print(f"• Total Positions: {portfolio_insights['total_positions']}")
        log_print(f"• Total Capital: ${portfolio_insights['total_capital']:.1f}")
        
        # Show portfolio metrics
        metrics = portfolio_insights.get('portfolio_metrics', {})
        if metrics:
            log_print(f"• Expected Return: {metrics.get('expected_return', 0)*100:.1f}%")
            log_print(f"• Portfolio Volatility: {metrics.get('portfolio_volatility', 0)*100:.1f}%")
            log_print(f"• Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            log_print(f"• Risk-Adjusted Return: {metrics.get('risk_adjusted_return', 0)*100:.1f}%")
        
        # Show insights
        insights = portfolio_insights.get('insights', [])
        if insights:
            log_print("\n💡 Portfolio Insights:")
            for insight in insights[:3]:  # Show top 3 insights
                log_print(f"  • {insight}")
        
        # Show recommendations
        recommendations = portfolio_insights.get('recommendations', [])
        if recommendations:
            log_print("\n🎯 Recommendations:")
            for rec in recommendations[:2]:  # Show top 2 recommendations
                log_print(f"  • {rec}")
                
    except Exception as e:
        log_print(f"⚠️ Could not generate portfolio insights: {e}")

def _show_risk_insights():
    """Show AI risk assessment insights"""
    try:
        # Get current open positions for risk analysis
        open_positions = performance_tracker.get_open_trades()
        
        if not open_positions:
            log_print("\n⚠️ Risk Assessment: No open positions to analyze")
            return
        
        # Create position data for risk analysis
        current_positions = []
        for pos in open_positions:
            current_positions.append({
                'symbol': pos.get('symbol', 'UNKNOWN'),
                'value': pos.get('position_size_usd', 0),
                'quality_score': pos.get('quality_score', 50),
                'risk_score': 0.5,
                'expected_return': 0.12,
                'sector': 'crypto'
            })
        
        # Get risk summary
        risk_summary = ai_risk_assessor.get_risk_summary(current_positions)
        
        log_print("\n⚠️ AI Risk Assessment Insights")
        log_print(f"• Total Positions: {risk_summary['total_tokens']}")
        log_print(f"• High Risk: {risk_summary['high_risk_tokens']}")
        log_print(f"• Medium Risk: {risk_summary['medium_risk_tokens']}")
        log_print(f"• Low Risk: {risk_summary['low_risk_tokens']}")
        log_print(f"• Overall Risk Level: {risk_summary['overall_risk_level']}")
        
        # Show individual position risk assessments
        risk_summaries = risk_summary.get('risk_summaries', [])
        if risk_summaries:
            log_print("\n📊 Position Risk Analysis:")
            for risk_info in risk_summaries[:5]:  # Show top 5 positions
                symbol = risk_info['symbol']
                risk_score = risk_info['risk_score']
                risk_category = risk_info['risk_category']
                should_trade = risk_info['should_trade']
                
                status = "✅" if should_trade else "❌"
                log_print(f"  {status} {symbol}: {risk_category} ({risk_score:.2f})")
        
        # Show risk recommendations
        if risk_summary['overall_risk_level'] == 'high':
            log_print("\n🚨 High Risk Alert:")
            log_print("  • Consider reducing position sizes")
            log_print("  • Monitor positions closely")
            log_print("  • Consider closing high-risk positions")
        elif risk_summary['overall_risk_level'] == 'medium':
            log_print("\n⚠️ Medium Risk Alert:")
            log_print("  • Monitor risk factors")
            log_print("  • Consider position adjustments")
        else:
            log_print("\n✅ Low Risk Portfolio:")
            log_print("  • Good risk management")
            log_print("  • Continue monitoring")
                
    except Exception as e:
        log_print(f"⚠️ Could not generate risk insights: {e}")

def _show_pattern_insights():
    """Show AI pattern recognition insights"""
    try:
        # Get current open positions for pattern analysis
        open_positions = performance_tracker.get_open_trades()
        
        if not open_positions:
            log_print("\n🔍 Pattern Recognition: No open positions to analyze")
            return
        
        # Create position data for pattern analysis
        current_positions = []
        for pos in open_positions:
            current_positions.append({
                'symbol': pos.get('symbol', 'UNKNOWN'),
                'priceUsd': pos.get('entry_price', 0),
                'volume24h': pos.get('volume_24h', 0),
                'priceChange24h': 0,  # Simulate price change
                'liquidity': pos.get('liquidity', 0)
            })
        
        # Get pattern summary
        pattern_summary = ai_pattern_recognizer.get_pattern_summary(current_positions)
        
        log_print("\n🔍 AI Pattern Recognition Insights")
        log_print(f"• Total Positions: {pattern_summary['total_tokens']}")
        log_print(f"• Strong Patterns: {pattern_summary['strong_patterns']}")
        log_print(f"• Moderate Patterns: {pattern_summary['moderate_patterns']}")
        log_print(f"• Weak Patterns: {pattern_summary['weak_patterns']}")
        log_print(f"• Overall Pattern Quality: {pattern_summary['overall_pattern_quality']}")
        
        # Show individual position pattern analysis
        pattern_summaries = pattern_summary.get('pattern_summaries', [])
        if pattern_summaries:
            log_print("\n📊 Position Pattern Analysis:")
            for pattern_info in pattern_summaries[:5]:  # Show top 5 positions
                symbol = pattern_info['symbol']
                pattern_strength = pattern_info['pattern_strength']
                overall_signal = pattern_info['overall_signal']
                confidence_level = pattern_info['confidence_level']
                
                signal_emoji = "📈" if "buy" in overall_signal else "📉" if "sell" in overall_signal else "➡️"
                log_print(f"  {signal_emoji} {symbol}: {overall_signal} (strength: {pattern_strength:.2f}, confidence: {confidence_level})")
        
        # Show pattern recommendations
        if pattern_summary['overall_pattern_quality'] == 'high':
            log_print("\n✅ Strong Pattern Portfolio:")
            log_print("  • Excellent pattern formation")
            log_print("  • High confidence signals")
            log_print("  • Continue monitoring patterns")
        elif pattern_summary['overall_pattern_quality'] == 'medium':
            log_print("\n⚠️ Moderate Pattern Portfolio:")
            log_print("  • Mixed pattern quality")
            log_print("  • Monitor for pattern improvements")
            log_print("  • Consider pattern-based adjustments")
        else:
            log_print("\n⚠️ Weak Pattern Portfolio:")
            log_print("  • Limited pattern formation")
            log_print("  • Wait for clearer patterns")
            log_print("  • Consider pattern-based filtering")
                
    except Exception as e:
        log_print(f"⚠️ Could not generate pattern insights: {e}")

def _show_execution_insights():
    """Show AI execution optimization insights"""
    try:
        # Get current open positions for execution analysis
        open_positions = performance_tracker.get_open_trades()
        
        if not open_positions:
            log_print("\n⚡ Execution Optimization: No open positions to analyze")
            return
        
        # Create position data for execution analysis
        current_positions = []
        trade_amounts = []
        for pos in open_positions:
            current_positions.append({
                'symbol': pos.get('symbol', 'UNKNOWN'),
                'priceUsd': pos.get('entry_price', 0),
                'volume24h': pos.get('volume_24h', 0),
                'liquidity': pos.get('liquidity', 0),
                'chainId': pos.get('chain_id', 'ethereum')
            })
            trade_amounts.append(pos.get('position_size_usd', 5.0))
        
        # Get execution summary
        execution_summary = ai_execution_optimizer.get_execution_summary(current_positions, trade_amounts)
        
        log_print("\n⚡ AI Execution Optimization Insights")
        log_print(f"• Total Positions: {execution_summary['total_tokens']}")
        log_print(f"• High Efficiency: {execution_summary['high_efficiency']}")
        log_print(f"• Medium Efficiency: {execution_summary['medium_efficiency']}")
        log_print(f"• Low Efficiency: {execution_summary['low_efficiency']}")
        log_print(f"• Overall Efficiency: {execution_summary['overall_efficiency']}")
        
        # Show individual position execution analysis
        execution_summaries = execution_summary.get('execution_summaries', [])
        if execution_summaries:
            log_print("\n📊 Position Execution Analysis:")
            for exec_info in execution_summaries[:5]:  # Show top 5 positions
                symbol = exec_info['symbol']
                strategy_score = exec_info['strategy_score']
                success_probability = exec_info['success_probability']
                recommendation = exec_info['recommendation']
                
                efficiency_emoji = "⚡" if strategy_score > 0.8 else "🔧" if strategy_score > 0.6 else "⚠️"
                log_print(f"  {efficiency_emoji} {symbol}: {recommendation} (efficiency: {strategy_score:.2f}, success: {success_probability:.1%})")
        
        # Show execution recommendations
        if execution_summary['overall_efficiency'] == 'high':
            log_print("\n✅ High Efficiency Portfolio:")
            log_print("  • Excellent execution optimization")
            log_print("  • Optimal routing and timing")
            log_print("  • Continue current execution strategy")
        elif execution_summary['overall_efficiency'] == 'medium':
            log_print("\n⚠️ Medium Efficiency Portfolio:")
            log_print("  • Mixed execution efficiency")
            log_print("  • Consider execution optimizations")
            log_print("  • Monitor execution performance")
        else:
            log_print("\n⚠️ Low Efficiency Portfolio:")
            log_print("  • Poor execution efficiency")
            log_print("  • Optimize execution strategy")
            log_print("  • Consider alternative routing")
                
    except Exception as e:
        log_print(f"⚠️ Could not generate execution insights: {e}")

def _show_microstructure_insights():
    """Show AI microstructure analysis insights"""
    try:
        # Get current open positions for microstructure analysis
        open_positions = performance_tracker.get_open_trades()
        
        if not open_positions:
            log_print("\n🔍 Microstructure Analysis: No open positions to analyze")
            return
        
        # Create position data for microstructure analysis
        current_positions = []
        trade_amounts = []
        for pos in open_positions:
            current_positions.append({
                'symbol': pos.get('symbol', 'UNKNOWN'),
                'priceUsd': pos.get('entry_price', 0),
                'volume24h': pos.get('volume_24h', 0),
                'liquidity': pos.get('liquidity', 0),
                'chainId': pos.get('chain_id', 'ethereum')
            })
            trade_amounts.append(pos.get('position_size_usd', 5.0))
        
        # Get microstructure summary
        microstructure_summary = ai_microstructure_analyzer.get_microstructure_summary(current_positions, trade_amounts)
        
        log_print("\n🔍 AI Microstructure Analysis Insights")
        log_print(f"• Total Positions: {microstructure_summary['total_tokens']}")
        log_print(f"• High Quality: {microstructure_summary['high_quality']}")
        log_print(f"• Medium Quality: {microstructure_summary['medium_quality']}")
        log_print(f"• Low Quality: {microstructure_summary['low_quality']}")
        log_print(f"• Overall Quality: {microstructure_summary['overall_quality']}")
        
        # Show individual position microstructure analysis
        microstructure_summaries = microstructure_summary.get('microstructure_summaries', [])
        if microstructure_summaries:
            log_print("\n📊 Position Microstructure Analysis:")
            for micro_info in microstructure_summaries[:5]:  # Show top 5 positions
                symbol = micro_info['symbol']
                microstructure_score = micro_info['microstructure_score']
                risk_category = micro_info['risk_category']
                execution_recommendation = micro_info['execution_recommendation']
                
                quality_emoji = "🔍" if microstructure_score > 0.8 else "🔧" if microstructure_score > 0.6 else "⚠️"
                log_print(f"  {quality_emoji} {symbol}: {execution_recommendation} (score: {microstructure_score:.2f}, risk: {risk_category})")
        
        # Show microstructure recommendations
        if microstructure_summary['overall_quality'] == 'high':
            log_print("\n✅ High Quality Microstructure:")
            log_print("  • Excellent market conditions")
            log_print("  • Optimal execution environment")
            log_print("  • Continue current strategy")
        elif microstructure_summary['overall_quality'] == 'medium':
            log_print("\n⚠️ Medium Quality Microstructure:")
            log_print("  • Mixed market conditions")
            log_print("  • Monitor microstructure changes")
            log_print("  • Consider execution optimizations")
        else:
            log_print("\n⚠️ Low Quality Microstructure:")
            log_print("  • Poor market conditions")
            log_print("  • High execution risk")
            log_print("  • Consider waiting for better conditions")
                
    except Exception as e:
        log_print(f"⚠️ Could not generate microstructure insights: {e}")

def _show_market_intelligence_insights():
    """Show AI market intelligence insights"""
    try:
        # Get current open positions for intelligence analysis
        open_positions = performance_tracker.get_open_trades()
        
        if not open_positions:
            log_print("\n🧠 Market Intelligence: No open positions to analyze")
            return
        
        # Create position data for intelligence analysis
        current_positions = []
        trade_amounts = []
        for pos in open_positions:
            current_positions.append({
                'symbol': pos.get('symbol', 'UNKNOWN'),
                'priceUsd': pos.get('entry_price', 0),
                'volume24h': pos.get('volume_24h', 0),
                'liquidity': pos.get('liquidity', 0),
                'chainId': pos.get('chain_id', 'ethereum')
            })
            trade_amounts.append(pos.get('position_size_usd', 5.0))
        
        # Get intelligence summary
        intelligence_summary = ai_market_intelligence_aggregator.get_intelligence_summary(current_positions, trade_amounts)
        
        log_print("\n🧠 AI Market Intelligence Insights")
        log_print(f"• Total Positions: {intelligence_summary['total_tokens']}")
        log_print(f"• High Intelligence: {intelligence_summary['high_intelligence']}")
        log_print(f"• Medium Intelligence: {intelligence_summary['medium_intelligence']}")
        log_print(f"• Low Intelligence: {intelligence_summary['low_intelligence']}")
        log_print(f"• Overall Intelligence: {intelligence_summary['overall_intelligence']}")
        
        # Show individual position intelligence analysis
        intelligence_summaries = intelligence_summary.get('intelligence_summaries', [])
        if intelligence_summaries:
            log_print("\n📊 Position Intelligence Analysis:")
            for intel_info in intelligence_summaries[:5]:  # Show top 5 positions
                symbol = intel_info['symbol']
                intelligence_score = intel_info['intelligence_score']
                market_sentiment = intel_info['market_sentiment']
                trading_recommendation = intel_info['trading_recommendation']
                
                intelligence_emoji = "🧠" if intelligence_score > 0.8 else "🔍" if intelligence_score > 0.6 else "⚠️"
                log_print(f"  {intelligence_emoji} {symbol}: {trading_recommendation} (score: {intelligence_score:.2f}, sentiment: {market_sentiment})")
        
        # Show intelligence recommendations
        if intelligence_summary['overall_intelligence'] == 'high':
            log_print("\n✅ High Intelligence Portfolio:")
            log_print("  • Excellent market intelligence")
            log_print("  • Strong news and social sentiment")
            log_print("  • High influencer activity")
            log_print("  • Continue current strategy")
        elif intelligence_summary['overall_intelligence'] == 'medium':
            log_print("\n⚠️ Medium Intelligence Portfolio:")
            log_print("  • Mixed market intelligence")
            log_print("  • Monitor news and social sentiment")
            log_print("  • Consider intelligence-based adjustments")
        else:
            log_print("\n⚠️ Low Intelligence Portfolio:")
            log_print("  • Limited market intelligence")
            log_print("  • Weak news and social sentiment")
            log_print("  • Consider waiting for better conditions")
                
    except Exception as e:
        log_print(f"⚠️ Could not generate market intelligence insights: {e}")

def _show_predictive_analytics_insights():
    """Show AI predictive analytics insights"""
    try:
        # Get current open positions for predictive analysis
        open_positions = performance_tracker.get_open_trades()
        
        if not open_positions:
            log_print("\n🔮 Predictive Analytics: No open positions to analyze")
            return
        
        # Create position data for predictive analysis
        current_positions = []
        trade_amounts = []
        for pos in open_positions:
            current_positions.append({
                'symbol': pos.get('symbol', 'UNKNOWN'),
                'priceUsd': pos.get('entry_price', 0),
                'volume24h': pos.get('volume_24h', 0),
                'liquidity': pos.get('liquidity', 0),
                'chainId': pos.get('chain_id', 'ethereum')
            })
            trade_amounts.append(pos.get('position_size_usd', 5.0))
        
        # Get prediction summary
        prediction_summary = ai_predictive_analytics_engine.get_prediction_summary(current_positions, trade_amounts)
        
        log_print("\n🔮 AI Predictive Analytics Insights")
        log_print(f"• Total Positions: {prediction_summary['total_tokens']}")
        log_print(f"• High Prediction: {prediction_summary['high_prediction']}")
        log_print(f"• Medium Prediction: {prediction_summary['medium_prediction']}")
        log_print(f"• Low Prediction: {prediction_summary['low_prediction']}")
        log_print(f"• Overall Prediction: {prediction_summary['overall_prediction']}")
        
        # Show individual position prediction analysis
        prediction_summaries = prediction_summary.get('prediction_summaries', [])
        if prediction_summaries:
            log_print("\n📊 Position Prediction Analysis:")
            for pred_info in prediction_summaries[:5]:  # Show top 5 positions
                symbol = pred_info['symbol']
                prediction_score = pred_info['prediction_score']
                trading_signal = pred_info['trading_signal']
                confidence_level = pred_info['confidence_level']
                
                prediction_emoji = "🔮" if prediction_score > 0.8 else "📈" if prediction_score > 0.6 else "⚠️"
                log_print(f"  {prediction_emoji} {symbol}: {trading_signal} (score: {prediction_score:.2f}, confidence: {confidence_level})")
        
        # Show prediction recommendations
        if prediction_summary['overall_prediction'] == 'high':
            log_print("\n✅ High Prediction Portfolio:")
            log_print("  • Excellent prediction accuracy")
            log_print("  • Strong technical and sentiment signals")
            log_print("  • High confidence predictions")
            log_print("  • Continue current strategy")
        elif prediction_summary['overall_prediction'] == 'medium':
            log_print("\n⚠️ Medium Prediction Portfolio:")
            log_print("  • Mixed prediction accuracy")
            log_print("  • Monitor prediction signals")
            log_print("  • Consider prediction-based adjustments")
        else:
            log_print("\n⚠️ Low Prediction Portfolio:")
            log_print("  • Limited prediction accuracy")
            log_print("  • Weak prediction signals")
            log_print("  • Consider waiting for better conditions")
                
    except Exception as e:
        log_print(f"⚠️ Could not generate predictive analytics insights: {e}")

def _show_dynamic_strategy_insights():
    """Show AI dynamic strategy selector insights"""
    try:
        # Get current open positions for strategy analysis
        open_positions = performance_tracker.get_open_trades()
        
        if not open_positions:
            log_print("\n🎯 Dynamic Strategy: No open positions to analyze")
            return
        
        # Create position data for strategy analysis
        current_positions = []
        trade_amounts = []
        market_conditions = []
        for pos in open_positions:
            current_positions.append({
                'symbol': pos.get('symbol', 'UNKNOWN'),
                'priceUsd': pos.get('entry_price', 0),
                'volume24h': pos.get('volume_24h', 0),
                'liquidity': pos.get('liquidity', 0),
                'chainId': pos.get('chain_id', 'ethereum')
            })
            trade_amounts.append(pos.get('position_size_usd', 5.0))
            
            # Create market conditions for each position
            market_conditions.append({
                'bull_market_probability': 0.6,
                'bear_market_probability': 0.3,
                'sideways_market_probability': 0.4,
                'volatile_market_probability': 0.5,
                'volatility_score': 0.5,
                'volatility_trend': 'stable',
                'volatility_regime': 'moderate'
            })
        
        # Get strategy summary
        strategy_summary = ai_dynamic_strategy_selector.get_strategy_summary(current_positions, trade_amounts, market_conditions)
        
        log_print("\n🎯 AI Dynamic Strategy Selector Insights")
        log_print(f"• Total Positions: {strategy_summary['total_tokens']}")
        log_print(f"• Most Common Strategy: {strategy_summary['most_common_strategy']}")
        log_print(f"• Overall Confidence: {strategy_summary['overall_confidence']}")
        
        # Show strategy distribution
        strategy_counts = strategy_summary.get('strategy_counts', {})
        if strategy_counts:
            log_print(f"\n📊 Strategy Distribution:")
            for strategy, count in strategy_counts.items():
                strategy_name = strategy.replace('_strategy', '').replace('_', ' ').title()
                log_print(f"  • {strategy_name}: {count}")
        
        # Show confidence distribution
        confidence_counts = strategy_summary.get('confidence_counts', {})
        if confidence_counts:
            log_print(f"\n🎯 Confidence Distribution:")
            for confidence, count in confidence_counts.items():
                confidence_emoji = "🟢" if confidence == 'high' else "🟡" if confidence == 'medium' else "🔴"
                log_print(f"  {confidence_emoji} {confidence.title()}: {count}")
        
        # Show individual position strategy analysis
        strategy_summaries = strategy_summary.get('strategy_summaries', [])
        if strategy_summaries:
            log_print("\n📊 Position Strategy Analysis:")
            for strategy_info in strategy_summaries[:5]:  # Show top 5 positions
                symbol = strategy_info['symbol']
                selected_strategy = strategy_info['selected_strategy']
                strategy_confidence = strategy_info['strategy_confidence']
                regime = strategy_info['regime']
                
                strategy_emoji = "🎯" if strategy_confidence == 'high' else "📈" if strategy_confidence == 'medium' else "⚠️"
                strategy_name = selected_strategy.replace('_strategy', '').replace('_', ' ').title()
                log_print(f"  {strategy_emoji} {symbol}: {strategy_name} (confidence: {strategy_confidence}, regime: {regime})")
        
        # Show strategy recommendations
        if strategy_summary['overall_confidence'] == 'high':
            log_print("\n✅ High Strategy Confidence:")
            log_print("  • Excellent strategy selection")
            log_print("  • Strong market regime alignment")
            log_print("  • High historical performance")
            log_print("  • Continue current strategy mix")
        elif strategy_summary['overall_confidence'] == 'medium':
            log_print("\n⚠️ Medium Strategy Confidence:")
            log_print("  • Mixed strategy performance")
            log_print("  • Monitor strategy effectiveness")
            log_print("  • Consider strategy adjustments")
        else:
            log_print("\n⚠️ Low Strategy Confidence:")
            log_print("  • Limited strategy confidence")
            log_print("  • Weak market regime alignment")
            log_print("  • Consider strategy optimization")
                
    except Exception as e:
        log_print(f"⚠️ Could not generate dynamic strategy insights: {e}")

def _show_risk_prediction_insights():
    """Show AI risk prediction and prevention insights"""
    try:
        # Get current open positions for risk analysis
        open_positions = performance_tracker.get_open_trades()
        
        if not open_positions:
            log_print("\n🛡️ Risk Prediction: No open positions to analyze")
            return
        
        # Create position data for risk analysis
        current_positions = []
        trade_amounts = []
        for pos in open_positions:
            current_positions.append({
                'symbol': pos.get('symbol', 'UNKNOWN'),
                'priceUsd': pos.get('entry_price', 0),
                'volume24h': pos.get('volume_24h', 0),
                'liquidity': pos.get('liquidity', 0),
                'chainId': pos.get('chain_id', 'ethereum')
            })
            trade_amounts.append(pos.get('position_size_usd', 5.0))
        
        # Get risk summary
        risk_summary = ai_risk_prediction_prevention_system.get_risk_summary(current_positions, trade_amounts)
        
        log_print("\n🛡️ AI Risk Prediction & Prevention Insights")
        log_print(f"• Total Positions: {risk_summary['total_tokens']}")
        log_print(f"• Critical Risk: {risk_summary['critical_risk']}")
        log_print(f"• High Risk: {risk_summary['high_risk']}")
        log_print(f"• Medium Risk: {risk_summary['medium_risk']}")
        log_print(f"• Low Risk: {risk_summary['low_risk']}")
        log_print(f"• Overall Risk: {risk_summary['overall_risk']}")
        
        # Show individual position risk analysis
        risk_summaries = risk_summary.get('risk_summaries', [])
        if risk_summaries:
            log_print("\n📊 Position Risk Analysis:")
            for risk_info in risk_summaries[:5]:  # Show top 5 positions
                symbol = risk_info['symbol']
                risk_score = risk_info['risk_score']
                risk_level = risk_info['risk_level']
                risk_confidence = risk_info['risk_confidence']
                
                risk_emoji = "🔴" if risk_level == 'critical' else "🟠" if risk_level == 'high' else "🟡" if risk_level == 'medium' else "🟢"
                log_print(f"  {risk_emoji} {symbol}: {risk_level} (score: {risk_score:.2f}, confidence: {risk_confidence})")
        
        # Show risk recommendations
        if risk_summary['overall_risk'] == 'critical':
            log_print("\n🚨 CRITICAL RISK DETECTED:")
            log_print("  • Exit all positions immediately")
            log_print("  • Stop all new trades")
            log_print("  • Activate emergency protocols")
        elif risk_summary['overall_risk'] == 'high':
            log_print("\n⚠️ HIGH RISK DETECTED:")
            log_print("  • Reduce position sizes significantly")
            log_print("  • Avoid new positions")
            log_print("  • Monitor for exit signals")
        elif risk_summary['overall_risk'] == 'medium':
            log_print("\n⚠️ MEDIUM RISK DETECTED:")
            log_print("  • Use conservative position sizing")
            log_print("  • Monitor risk indicators")
            log_print("  • Consider risk-adjusted strategies")
        else:
            log_print("\n✅ LOW RISK CONDITIONS:")
            log_print("  • Normal trading conditions")
            log_print("  • Continue current strategy")
            log_print("  • Monitor for risk changes")
                
    except Exception as e:
        log_print(f"⚠️ Could not generate risk prediction insights: {e}")

def main():
    """Main entry point"""
    log_print("🌱 Starting Sustainable Trading Bot - LIVE MODE")
    log_print("🎯 Strategy: Consistent 10-20% gains")
    log_print("📊 Focus: Quality over quantity")
    
    # Check live trading readiness
    if not check_live_trading_ready():
        log_print("❌ System not ready for live trading. Exiting.")
        return
    
    # Send startup notification
    try:
        send_telegram_message(
            "🌱 Sustainable Trading Bot Started\n"
            "🎯 Strategy: 10-20% consistent gains\n"
            "📊 Quality over quantity approach\n"
            "✅ Live trading enabled",
            message_type="status"
        )
    except Exception as e:
        log_print(f"⚠️ Could not send startup notification: {e}")
    
    # Main trading loop
    while True:
        try:
            # Check if it's time to send periodic status report (every 6 hours)
            try:
                send_periodic_status_report()
            except Exception as e:
                log_print(f"⚠️ Could not send periodic status: {e}")
            
            practical_trade_loop()
            
            # Wait before next cycle (longer for sustainable trading)
            wait_time = 600  # 10 minutes between cycles
            log_print(f"⏰ Waiting {wait_time//60} minutes before next cycle...")
            time.sleep(wait_time)
            
        except KeyboardInterrupt:
            log_print("🛑 Bot stopped by user")
            try:
                send_telegram_message("🛑 Sustainable Trading Bot stopped by user")
            except:
                pass
            break
        except Exception as e:
            error_msg = str(e) if e else "Unknown error"
            log_print(f"🔥 Bot error: {error_msg}")
            try:
                send_telegram_message(f"🔥 Bot error: {error_msg}")
            except Exception as telegram_error:
                log_print(f"⚠️ Failed to send telegram error message: {telegram_error}")
            time.sleep(60)  # Wait 1 minute before retry

def _show_regime_transition_insights():
    """Show AI market regime transition insights"""
    try:
        # Get current open positions for transition analysis
        open_positions = performance_tracker.get_open_trades()
        
        if not open_positions:
            log_print("\n🔄 Regime Transition: No open positions to analyze")
            return
        
        # Create position data for transition analysis
        current_positions = []
        for pos in open_positions:
            current_positions.append({
                'symbol': pos.get('symbol', 'UNKNOWN'),
                'priceUsd': pos.get('entry_price', 0),
                'volume24h': pos.get('volume_24h', 0),
                'liquidity': pos.get('liquidity', 0),
                'chainId': pos.get('chain_id', 'ethereum')
            })
        
        # Get transition summary
        # Create market_data_list and current_regimes for each position
        market_data_list = []
        current_regimes = []
        for pos in current_positions:
            market_data_list.append({
                'timestamp': datetime.now().isoformat(),
                'price': float(pos.get('priceUsd', 0)),
                'volume': float(pos.get('volume24h', 0)),
                'liquidity': float(pos.get('liquidity', 0)),
                'volatility': 0.2,
                'regime': 'normal'
            })
            current_regimes.append('normal')  # Default regime
        
        transition_summary = ai_market_regime_transition_detector.get_transition_summary(market_data_list, current_regimes)
        
        log_print("\n🔄 AI Market Regime Transition Insights")
        log_print(f"• Total Positions: {transition_summary['total_tokens']}")
        log_print(f"• Transition Probability: {transition_summary['transition_probability']:.2f}")
        log_print(f"• Transition Confidence: {transition_summary['transition_confidence']}")
        log_print(f"• Current Regime: {transition_summary['current_regime']}")
        log_print(f"• Predicted Regime: {transition_summary['predicted_regime']}")
        
        # Show transition recommendations
        if transition_summary['transition_probability'] > 0.7:
            log_print("\n⚠️ HIGH TRANSITION PROBABILITY DETECTED:")
            log_print("  • Monitor market conditions closely")
            log_print("  • Consider reducing position sizes")
            log_print("  • Prepare for regime change")
        else:
            log_print("\n✅ STABLE REGIME CONDITIONS:")
            log_print("  • Continue current strategy")
            log_print("  • Monitor for regime changes")
            log_print("  • Maintain position sizes")
                
    except Exception as e:
        log_print(f"⚠️ Could not generate regime transition insights: {e}")

def _show_liquidity_flow_insights():
    """Show AI liquidity flow insights"""
    try:
        # Get current open positions for liquidity analysis
        open_positions = performance_tracker.get_open_trades()
        
        if not open_positions:
            log_print("\n💧 Liquidity Flow: No open positions to analyze")
            return
        
        # Create position data for liquidity analysis
        current_positions = []
        for pos in open_positions:
            current_positions.append({
                'symbol': pos.get('symbol', 'UNKNOWN'),
                'priceUsd': pos.get('entry_price', 0),
                'volume24h': pos.get('volume24h', 0),
                'liquidity': pos.get('liquidity', 0),
                'chainId': pos.get('chain_id', 'ethereum')
            })
        
        # Get liquidity flow summary
        # Create market_data_list for each position
        market_data_list = []
        for pos in current_positions:
            market_data_list.append({
                'timestamp': datetime.now().isoformat(),
                'price': float(pos.get('priceUsd', 0)),
                'volume': float(pos.get('volume24h', 0)),
                'liquidity': float(pos.get('liquidity', 0)),
                'volatility': 0.2,
                'regime': 'normal'
            })
        
        liquidity_summary = ai_liquidity_flow_analyzer.get_liquidity_summary(current_positions, market_data_list)
        
        log_print("\n💧 AI Liquidity Flow Insights")
        log_print(f"• Total Positions: {liquidity_summary['total_tokens']}")
        log_print(f"• Average Liquidity Score: {liquidity_summary['average_liquidity_score']:.2f}")
        log_print(f"• Flow Recommendation: {liquidity_summary['flow_recommendation']}")
        log_print(f"• Market Imbalance: {liquidity_summary['market_imbalance']:.2f}")
        
        # Show liquidity flow recommendations
        if liquidity_summary['flow_recommendation'] == 'avoid_trading':
            log_print("\n⚠️ LIQUIDITY FLOW WARNING:")
            log_print("  • Avoid new positions")
            log_print("  • Monitor existing positions")
            log_print("  • Consider reducing exposure")
        else:
            log_print("\n✅ FAVORABLE LIQUIDITY CONDITIONS:")
            log_print("  • Continue trading strategy")
            log_print("  • Monitor liquidity changes")
            log_print("  • Maintain current positions")
                
    except Exception as e:
        log_print(f"⚠️ Could not generate liquidity flow insights: {e}")

def _show_multi_timeframe_insights():
    """Show AI multi-timeframe analysis insights"""
    try:
        # Get current open positions for timeframe analysis
        open_positions = performance_tracker.get_open_trades()
        
        if not open_positions:
            log_print("\n⏰ Multi-Timeframe: No open positions to analyze")
            return
        
        # Create position data for timeframe analysis
        current_positions = []
        for pos in open_positions:
            current_positions.append({
                'symbol': pos.get('symbol', 'UNKNOWN'),
                'priceUsd': pos.get('entry_price', 0),
                'volume24h': pos.get('volume24h', 0),
                'liquidity': pos.get('liquidity', 0),
                'chainId': pos.get('chain_id', 'ethereum')
            })
        
        # Get timeframe analysis summary
        # Create market_data_list for each position
        market_data_list = []
        for pos in current_positions:
            market_data_list.append({
                'timestamp': datetime.now().isoformat(),
                'price': float(pos.get('priceUsd', 0)),
                'volume': float(pos.get('volume24h', 0)),
                'liquidity': float(pos.get('liquidity', 0)),
                'volatility': 0.2,
                'regime': 'normal'
            })
        
        timeframe_summary = ai_multi_timeframe_analysis_engine.get_multi_timeframe_summary(current_positions, market_data_list)
        
        log_print("\n⏰ AI Multi-Timeframe Analysis Insights")
        log_print(f"• Total Positions: {timeframe_summary['total_tokens']}")
        log_print(f"• Average Timeframe Score: {timeframe_summary['average_timeframe_score']:.2f}")
        log_print(f"• Overall Signal: {timeframe_summary['overall_signal']}")
        log_print(f"• Signal Confidence: {timeframe_summary['signal_confidence']}")
        
        # Show timeframe recommendations
        if timeframe_summary['overall_signal'] in ['sell', 'strong_sell']:
            log_print("\n⚠️ BEARISH TIMEFRAME SIGNALS:")
            log_print("  • Consider reducing positions")
            log_print("  • Monitor for exit signals")
            log_print("  • Avoid new positions")
        else:
            log_print("\n✅ BULLISH TIMEFRAME SIGNALS:")
            log_print("  • Continue current strategy")
            log_print("  • Monitor timeframe alignment")
            log_print("  • Consider new positions")
                
    except Exception as e:
        log_print(f"⚠️ Could not generate multi-timeframe insights: {e}")

def _show_market_cycle_insights():
    """Show AI market cycle prediction insights"""
    try:
        # Get current open positions for cycle analysis
        open_positions = performance_tracker.get_open_trades()
        
        if not open_positions:
            log_print("\n🔄 Market Cycle: No open positions to analyze")
            return
        
        # Create position data for cycle analysis
        current_positions = []
        for pos in open_positions:
            current_positions.append({
                'symbol': pos.get('symbol', 'UNKNOWN'),
                'priceUsd': pos.get('entry_price', 0),
                'volume24h': pos.get('volume24h', 0),
                'liquidity': pos.get('liquidity', 0),
                'chainId': pos.get('chain_id', 'ethereum')
            })
        
        # Get cycle analysis summary
        # Create market_data_list for each position
        market_data_list = []
        for pos in current_positions:
            market_data_list.append({
                'timestamp': datetime.now().isoformat(),
                'price': float(pos.get('priceUsd', 0)),
                'volume': float(pos.get('volume24h', 0)),
                'liquidity': float(pos.get('liquidity', 0)),
                'volatility': 0.2,
                'regime': 'normal'
            })
        
        cycle_summary = ai_market_cycle_predictor.get_cycle_summary(market_data_list)
        
        log_print("\n🔄 AI Market Cycle Prediction Insights")
        log_print(f"• Total Markets: {cycle_summary.get('total_markets', 0)}")
        log_print(f"• Overall Cycle Phase: {cycle_summary.get('overall_cycle_phase', 'unknown')}")
        log_print(f"• Cycle Confidence: {cycle_summary.get('cycle_confidence', 'medium')}")
        log_print(f"• Accumulation: {cycle_summary.get('accumulation_count', 0)}, Markup: {cycle_summary.get('markup_count', 0)}, Distribution: {cycle_summary.get('distribution_count', 0)}, Markdown: {cycle_summary.get('markdown_count', 0)}")
        
        # Show cycle recommendations
        if cycle_summary.get('overall_cycle_phase', 'unknown') in ['decline', 'trough']:
            log_print("\n⚠️ UNFAVORABLE CYCLE PHASE:")
            log_print("  • Reduce position sizes")
            log_print("  • Monitor for cycle change")
            log_print("  • Avoid new positions")
        else:
            log_print("\n✅ FAVORABLE CYCLE PHASE:")
            log_print("  • Continue current strategy")
            log_print("  • Monitor cycle progression")
            log_print("  • Consider new positions")
                
    except Exception as e:
        log_print(f"⚠️ Could not generate market cycle insights: {e}")

def _show_drawdown_protection_insights():
    """Show AI drawdown protection insights"""
    try:
        # Get current portfolio data
        open_positions = performance_tracker.get_open_trades()
        trade_history = performance_tracker.get_trade_history() if hasattr(performance_tracker, 'get_trade_history') else []
        
        if not open_positions:
            log_print("\n🛡️ Drawdown Protection: No open positions to analyze")
            return
        
        # Create portfolio data
        portfolio_data = {
            'total_value': sum(pos.get('position_size_usd', 0) for pos in open_positions),
            'position_count': len(open_positions),
            'timestamp': datetime.now().isoformat()
        }
        # Initialize market_data for this function scope
        market_data = {'timestamp': datetime.now().isoformat()}
        
        # Get drawdown protection summary
        drawdown_summary = ai_drawdown_protection_system.get_drawdown_summary([portfolio_data], [trade_history], [market_data])
        
        log_print("\n🛡️ AI Drawdown Protection Insights")
        log_print(f"• Total Value: ${portfolio_data['total_value']:,.0f}")
        log_print(f"• Position Count: {portfolio_data['position_count']}")
        log_print(f"• Current Drawdown: {drawdown_summary['current_drawdown']:.1%}")
        log_print(f"• Drawdown Severity: {drawdown_summary['drawdown_severity']}")
        log_print(f"• Protection Urgency: {drawdown_summary['protection_urgency']}")
        
        # Show drawdown protection recommendations
        if drawdown_summary['protection_urgency'] in ['urgent', 'emergency']:
            log_print("\n🚨 HIGH DRAWDOWN PROTECTION URGENCY:")
            log_print("  • Reduce position sizes immediately")
            log_print("  • Consider emergency stop-loss")
            log_print("  • Monitor for recovery signals")
        else:
            log_print("\n✅ NORMAL DRAWDOWN CONDITIONS:")
            log_print("  • Continue current strategy")
            log_print("  • Monitor drawdown levels")
            log_print("  • Maintain position sizes")
                
    except Exception as e:
        log_print(f"⚠️ Could not generate drawdown protection insights: {e}")

def _show_performance_attribution_insights():
    """Show AI performance attribution insights"""
    try:
        # Get current open positions for attribution analysis
        open_positions = performance_tracker.get_open_trades()
        
        if not open_positions:
            log_print("\n📊 Performance Attribution: No open positions to analyze")
            return
        
        # Create position data for attribution analysis
        current_positions = []
        for pos in open_positions:
            current_positions.append({
                'symbol': pos.get('symbol', 'UNKNOWN'),
                'priceUsd': pos.get('entry_price', 0),
                'volume24h': pos.get('volume24h', 0),
                'liquidity': pos.get('liquidity', 0),
                'chainId': pos.get('chain_id', 'ethereum')
            })
        
        # Get attribution analysis summary
        # Create required data structures
        portfolio_data_list = []
        trade_history_list = []
        market_data_list = []
        performance_metrics_list = []
        
        for pos in current_positions:
            portfolio_data_list.append({
                'total_value': pos.get('position_size_usd', 0),
                'position_count': 1,
                'timestamp': datetime.now().isoformat()
            })
            trade_history_list.append([])  # Empty trade history for now
            market_data_list.append({
                'timestamp': datetime.now().isoformat(),
                'price': float(pos.get('priceUsd', 0)),
                'volume': float(pos.get('volume24h', 0)),
                'liquidity': float(pos.get('liquidity', 0)),
                'volatility': 0.2,
                'regime': 'normal'
            })
            performance_metrics_list.append({
                'total_return': 0.0,
                'volatility': 0.2,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            })
        
        attribution_summary = ai_performance_attribution_analyzer.get_attribution_summary(
            portfolio_data_list, trade_history_list, market_data_list, performance_metrics_list
        )
        
        log_print("\n📊 AI Performance Attribution Insights")
        log_print(f"• Total Positions: {attribution_summary['total_tokens']}")
        log_print(f"• Average Attribution Score: {attribution_summary['average_attribution_score']:.2f}")
        log_print(f"• Performance Recommendation: {attribution_summary['performance_recommendation']}")
        log_print(f"• Risk-Adjusted Return: {attribution_summary['risk_adjusted_return']:.2f}")
        
        # Show attribution recommendations
        if attribution_summary['performance_recommendation'] == 'avoid_trading':
            log_print("\n⚠️ PERFORMANCE ATTRIBUTION WARNING:")
            log_print("  • Avoid new positions")
            log_print("  • Review current strategy")
            log_print("  • Consider strategy adjustment")
        else:
            log_print("\n✅ FAVORABLE PERFORMANCE ATTRIBUTION:")
            log_print("  • Continue current strategy")
            log_print("  • Monitor performance drivers")
            log_print("  • Consider new positions")
                
    except Exception as e:
        log_print(f"⚠️ Could not generate performance attribution insights: {e}")

def _show_market_anomaly_insights():
    """Show AI market anomaly detection insights"""
    try:
        # Get current open positions for anomaly analysis
        open_positions = performance_tracker.get_open_trades()
        
        if not open_positions:
            log_print("\n🔍 Market Anomaly: No open positions to analyze")
            return
        
        # Create position data for anomaly analysis
        current_positions = []
        for pos in open_positions:
            current_positions.append({
                'symbol': pos.get('symbol', 'UNKNOWN'),
                'priceUsd': pos.get('entry_price', 0),
                'volume24h': pos.get('volume24h', 0),
                'liquidity': pos.get('liquidity', 0),
                'chainId': pos.get('chain_id', 'ethereum')
            })
        
        # Get anomaly analysis summary
        anomaly_summary = ai_market_anomaly_detector.get_anomaly_summary(current_positions)
        
        log_print("\n🔍 AI Market Anomaly Detection Insights")
        log_print(f"• Total Positions: {anomaly_summary['total_tokens']}")
        log_print(f"• Average Anomaly Score: {anomaly_summary['average_anomaly_score']:.2f}")
        log_print(f"• Anomaly Severity: {anomaly_summary['anomaly_severity']}")
        log_print(f"• Opportunities: {anomaly_summary['opportunities']}")
        log_print(f"• Risks: {anomaly_summary['risks']}")
        
        # Show anomaly recommendations
        if anomaly_summary['anomaly_severity'] in ['major', 'extreme']:
            log_print("\n⚠️ HIGH ANOMALY SEVERITY DETECTED:")
            log_print("  • Monitor positions closely")
            log_print("  • Consider reducing exposure")
            log_print("  • Watch for market changes")
        else:
            log_print("\n✅ NORMAL MARKET CONDITIONS:")
            log_print("  • Continue current strategy")
            log_print("  • Monitor for anomalies")
            log_print("  • Maintain position sizes")
                
    except Exception as e:
        log_print(f"⚠️ Could not generate market anomaly insights: {e}")

def _show_portfolio_rebalancing_insights():
    """Show AI portfolio rebalancing insights"""
    try:
        # Get current open positions for rebalancing analysis
        open_positions = performance_tracker.get_open_trades()
        
        if not open_positions:
            log_print("\n⚖️ Portfolio Rebalancing: No open positions to analyze")
            return
        
        # Create position data for rebalancing analysis
        current_positions = []
        for pos in open_positions:
            current_positions.append({
                'symbol': pos.get('symbol', 'UNKNOWN'),
                'position_size_usd': pos.get('position_size_usd', 0),
                'entry_price': pos.get('entry_price', 0),
                'volume_24h': pos.get('volume_24h', 0),
                'liquidity': pos.get('liquidity', 0),
                'chainId': pos.get('chain_id', 'ethereum')
            })
        
        # Initialize market_data for this function scope
        market_data = {'timestamp': datetime.now().isoformat()}
        
        # Get rebalancing analysis summary
        rebalancing_summary = ai_portfolio_rebalancing_engine.get_rebalancing_summary(current_positions, market_data)
        
        log_print("\n⚖️ AI Portfolio Rebalancing Insights")
        log_print(f"• Total Positions: {rebalancing_summary['total_positions']}")
        log_print(f"• Total Value: ${rebalancing_summary['total_value']:,.0f}")
        log_print(f"• Risk Score: {rebalancing_summary['risk_score']:.2f}")
        log_print(f"• Diversification Score: {rebalancing_summary['diversification_score']:.2f}")
        log_print(f"• Rebalancing Urgency: {rebalancing_summary['rebalancing_urgency']}")
        log_print(f"• Positions to Rebalance: {rebalancing_summary['positions_to_rebalance']}")
        
        # Show rebalancing recommendations
        if rebalancing_summary['rebalancing_urgency'] in ['urgent', 'emergency']:
            log_print("\n⚠️ HIGH REBALANCING URGENCY:")
            log_print("  • Execute rebalancing immediately")
            log_print("  • Adjust position sizes")
            log_print("  • Monitor portfolio efficiency")
        else:
            log_print("\n✅ PORTFOLIO WELL BALANCED:")
            log_print("  • Continue current allocation")
            log_print("  • Monitor rebalancing needs")
            log_print("  • Maintain position sizes")
                
    except Exception as e:
        log_print(f"⚠️ Could not generate portfolio rebalancing insights: {e}")

if __name__ == "__main__":
    main()
