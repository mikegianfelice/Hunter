#!/usr/bin/env python3
"""
Practical Sustainable Trading Bot
Realistic approach for consistent 10-20% gains
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from collections import defaultdict
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('practical_sustainable.log', mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def log_print(msg):
    """Print to console and log to file"""
    logger.info(msg)

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
from telegram_bot import send_telegram_message
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

def check_practical_buy_signal(token: Dict) -> bool:
    """
    Check if token meets practical sustainable trading criteria with market regime awareness.
    Focus on quality over quantity, but be realistic and market-aware.
    """
    address = token.get("address", "").lower()
    price = float(token.get("priceUsd", 0))
    volume_24h = float(token.get("volume24h", 0))
    liquidity = float(token.get("liquidity", 0))
    symbol = token.get("symbol", "")
    
    print(f"🔍 Evaluating {symbol} for practical sustainable trading...")
    
    # Basic requirements
    if not address or price <= 0.00001:  # $0.00001 minimum price
        print(f"❌ {symbol}: Missing address or price too low (${price:.6f})")
        return False
    
    # Get market regime for dynamic thresholds
    regime_data = ai_market_regime_detector.detect_market_regime()
    regime = regime_data['regime']
    quality_threshold_adjustment = regime_data['quality_threshold_adjustment']
    
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
    
    print(f"✅ {symbol}: Quality score {quality_score:.1f}, Volume ${volume_24h:,.0f}, Liquidity ${liquidity:,.0f}, Risk: {risk_category} ({risk_score:.2f}), Pattern: {overall_signal} ({pattern_strength:.2f}), Microstructure: {microstructure_score:.2f} (Regime: {regime})")
    return True

def get_dynamic_position_size(token: Dict) -> float:
    """
    Calculate dynamic position size based on AI-enhanced token quality, risk factors, market regime, and portfolio optimization.
    Higher quality tokens get larger positions, but with safety limits, market awareness, and portfolio optimization.
    """
    base_amount = 5.0  # $5 base position
    quality_score = calculate_ai_enhanced_quality_score(token)
    volume_24h = float(token.get("volume24h", 0))
    liquidity = float(token.get("liquidity", 0))
    
    # Get market regime for position sizing adjustments
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
        if check_practical_buy_signal(token):
            # Add AI-enhanced quality score, position size, and take profit to token data
            token["ai_enhanced_quality_score"] = calculate_ai_enhanced_quality_score(token)
            token["practical_position_size"] = get_dynamic_position_size(token)
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
            
            # Get current regime for notifications
            current_regime_data = ai_market_regime_detector.detect_market_regime()
            regime = current_regime_data['regime']
            
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
                    f"TX: {tx_hash}"
                )
                
                log_print(f"✅ Successfully traded {symbol} - AI Quality: {quality_score:.1f}, Sentiment: {sentiment_category}, Position: ${position_size:.1f}, TP: {tp*100:.1f}%")
                log_print(f"📊 Trade logged with ID: {trade_id}")
            else:
                log_print(f"❌ Trade failed for {symbol}")
                rejections["execution_failed"].append(symbol)
            
            # Small delay between trades
            time.sleep(3)
            
        except Exception as e:
            log_print(f"🔥 Error processing {token.get('symbol', 'UNKNOWN')}: {e}")
            rejections["error"].append(token.get("symbol", "UNKNOWN"))
    
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
            "✅ Live trading enabled"
        )
    except Exception as e:
        log_print(f"⚠️ Could not send startup notification: {e}")
    
    # Main trading loop
    while True:
        try:
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
            log_print(f"🔥 Bot error: {e}")
            try:
                send_telegram_message(f"🔥 Bot error: {e}")
            except:
                pass
            time.sleep(60)  # Wait 1 minute before retry

if __name__ == "__main__":
    main()
