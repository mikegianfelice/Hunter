#!/usr/bin/env python3
"""
Exit Validation Script
Tests momentum decay and structure failure exits against mock price data
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.core.strategy import _calculate_momentum_score
from src.config.config_loader import get_config_values, get_config_float, get_config_int, get_config_bool

def simulate_momentum_decay_exit():
    """Test momentum decay exit logic"""
    print("=" * 60)
    print("TEST 1: Momentum Decay Exit")
    print("=" * 60)
    
    config = get_config_values()
    
    # Simulate entry momentum
    entry_token = {
        "address": "test_token_123",
        "symbol": "TEST",
        "priceUsd": 1.0,
        "priceChange5m": 5.0,  # 5% in 5m
        "priceChange1h": 3.0,  # 3% in 1h
        "priceChange24h": 10.0,  # 10% in 24h
        "candles_validated": False,
    }
    
    entry_momentum, entry_source, entry_data = _calculate_momentum_score(entry_token, config)
    print(f"Entry Momentum: {entry_momentum:.4f} ({entry_source})")
    print(f"Entry Data: 5m={entry_data.get('momentum_5m')}, 1h={entry_data.get('momentum_1h')}, 24h={entry_data.get('momentum_24h')}")
    
    # Simulate decay scenarios
    decay_scenarios = [
        {
            "name": "Momentum drops below threshold",
            "priceChange5m": 0.3,  # 0.3% (below 0.5% threshold)
            "priceChange1h": 0.2,
            "priceChange24h": 5.0,
        },
        {
            "name": "Momentum decays significantly",
            "priceChange5m": 1.0,  # 1% (was 5%)
            "priceChange1h": 0.5,  # 0.5% (was 3%)
            "priceChange24h": 8.0,
        },
        {
            "name": "Momentum maintained",
            "priceChange5m": 4.5,  # Still strong
            "priceChange1h": 2.8,
            "priceChange24h": 9.5,
        },
    ]
    
    momentum_decay_threshold = get_config_float("momentum_decay_threshold", 0.005)
    momentum_decay_delta = get_config_float("momentum_decay_delta", 0.01)
    
    for scenario in decay_scenarios:
        print(f"\nScenario: {scenario['name']}")
        current_token = entry_token.copy()
        current_token.update({
            "priceChange5m": scenario["priceChange5m"],
            "priceChange1h": scenario["priceChange1h"],
            "priceChange24h": scenario["priceChange24h"],
        })
        
        current_momentum, current_source, _ = _calculate_momentum_score(current_token, config)
        if current_momentum is None:
            print(f"  ❌ No momentum data available")
            continue
        
        print(f"  Current Momentum: {current_momentum:.4f} ({current_source})")
        
        # Check absolute threshold
        if current_momentum < momentum_decay_threshold:
            print(f"  ✅ EXIT TRIGGERED: Below threshold ({current_momentum:.4f} < {momentum_decay_threshold:.4f})")
        else:
            print(f"  ⏸️  No threshold exit: {current_momentum:.4f} >= {momentum_decay_threshold:.4f}")
        
        # Check decay delta
        if entry_momentum is not None:
            decay_amount = entry_momentum - current_momentum
            print(f"  Decay Amount: {decay_amount:.4f}")
            if decay_amount > momentum_decay_delta:
                print(f"  ✅ EXIT TRIGGERED: Decay delta exceeded ({decay_amount:.4f} > {momentum_decay_delta:.4f})")
            else:
                print(f"  ⏸️  No decay exit: {decay_amount:.4f} <= {momentum_decay_delta:.4f}")

def simulate_structure_failure_exit():
    """Test structure failure exit logic"""
    print("\n" + "=" * 60)
    print("TEST 2: Structure Failure Exit")
    print("=" * 60)
    
    config = {
        'structure_failure_min_age_seconds': get_config_int("structure_failure_min_age_seconds", 300),
        'structure_break_threshold': get_config_float("structure_break_threshold", 0.02),
        'structure_lookback_candles': get_config_int("structure_lookback_candles", 20),
    }
    
    # Simulate candles with swing low
    candles = []
    base_price = 1.0
    for i in range(20):
        candles.append({
            'open': base_price + i * 0.01,
            'high': base_price + i * 0.01 + 0.02,
            'low': base_price + i * 0.01 - 0.01,
            'close': base_price + i * 0.01 + 0.01,
            'volume': 1000,
        })
    
    # Set swing low at candle 10
    swing_low_price = base_price + 10 * 0.01 - 0.01  # 1.09
    candles[10]['low'] = swing_low_price
    
    print(f"Swing Low: ${swing_low_price:.6f}")
    
    # Test scenarios
    scenarios = [
        {
            "name": "Price breaks below swing low by 2%",
            "current_price": swing_low_price * 0.98,  # 2% below
            "position_age_seconds": 600,  # 10 minutes
        },
        {
            "name": "Price breaks below swing low by 1%",
            "current_price": swing_low_price * 0.99,  # 1% below
            "position_age_seconds": 600,
        },
        {
            "name": "Price above swing low",
            "current_price": swing_low_price * 1.01,  # 1% above
            "position_age_seconds": 600,
        },
        {
            "name": "Position too new",
            "current_price": swing_low_price * 0.98,
            "position_age_seconds": 60,  # 1 minute (too new)
        },
    ]
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        current_price = scenario['current_price']
        position_age = scenario['position_age_seconds']
        
        if position_age < config['structure_failure_min_age_seconds']:
            print(f"  ⏸️  Position too new ({position_age}s < {config['structure_failure_min_age_seconds']}s)")
            continue
        
        # Find swing low
        lookback = config['structure_lookback_candles']
        lookback_candles = candles[-lookback:] if len(candles) >= lookback else candles
        swing_low = min(float(c.get('low', current_price)) for c in lookback_candles)
        
        break_pct = (swing_low - current_price) / swing_low if swing_low > 0 else 0
        
        print(f"  Current Price: ${current_price:.6f}")
        print(f"  Swing Low: ${swing_low:.6f}")
        print(f"  Break %: {break_pct*100:.2f}%")
        
        if break_pct >= config['structure_break_threshold']:
            print(f"  ✅ EXIT TRIGGERED: Break threshold exceeded ({break_pct*100:.2f}% >= {config['structure_break_threshold']*100:.2f}%)")
        else:
            print(f"  ⏸️  No exit: {break_pct*100:.2f}% < {config['structure_break_threshold']*100:.2f}%")

def simulate_vwap_entry_filter():
    """Test VWAP entry filter logic"""
    print("\n" + "=" * 60)
    print("TEST 3: VWAP Entry Filter")
    print("=" * 60)
    
    config = get_config_values()
    
    scenarios = [
        {
            "name": "Price above VWAP (allowed)",
            "price": 1.05,
            "vwap": 1.0,
            "expected": True,
        },
        {
            "name": "Price below VWAP (blocked)",
            "price": 0.95,
            "vwap": 1.0,
            "expected": False,
        },
        {
            "name": "Price extended above VWAP (requires extra confirmation)",
            "price": 1.10,  # 10% above
            "vwap": 1.0,
            "expected": True,  # Still allowed but logged as extended
        },
        {
            "name": "VWAP unavailable (proceed)",
            "price": 1.0,
            "vwap": None,
            "expected": True,
        },
    ]
    
    vwap_entry_required = config.get('VWAP_ENTRY_REQUIRED', True)
    max_extension = config.get('MAX_VWAP_EXTENSION_PCT', 0.05)
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        price = scenario['price']
        vwap = scenario['vwap']
        
        if vwap is None:
            print(f"  ✅ VWAP unavailable - proceeding (not blocking)")
            continue
        
        price_vs_vwap_pct = ((price - vwap) / vwap) if vwap > 0 else 0.0
        
        if vwap_entry_required and price < vwap:
            print(f"  ❌ BLOCKED: Price ${price:.6f} < VWAP ${vwap:.6f} ({price_vs_vwap_pct*100:.2f}% below)")
        elif price_vs_vwap_pct > max_extension:
            print(f"  ⚠️  EXTENDED: Price ${price:.6f} is {price_vs_vwap_pct*100:.2f}% above VWAP (threshold: {max_extension*100:.2f}%)")
            print(f"     → Requires extra momentum confirmation")
        else:
            print(f"  ✅ ALLOWED: Price ${price:.6f} vs VWAP ${vwap:.6f} ({price_vs_vwap_pct*100:.2f}%)")

def simulate_regime_enforcement():
    """Test regime enforcement logic"""
    print("\n" + "=" * 60)
    print("TEST 4: Regime Enforcement")
    print("=" * 60)
    
    bear_threshold = get_config_float("regime_bear_market_threshold", 0.60)
    vol_threshold = get_config_float("regime_high_volatility_threshold", 0.50)
    
    scenarios = [
        {
            "regime": "bear_market",
            "confidence": 0.65,
            "expected_block": True,
        },
        {
            "regime": "bear_market",
            "confidence": 0.55,
            "expected_block": False,
        },
        {
            "regime": "high_volatility",
            "confidence": 0.55,
            "expected_block": True,
        },
        {
            "regime": "high_volatility",
            "confidence": 0.45,
            "expected_block": False,
        },
        {
            "regime": "sideways_market",
            "confidence": 0.70,
            "expected_block": False,  # Throttle, not block
        },
        {
            "regime": "bull_market",
            "confidence": 0.80,
            "expected_block": False,
        },
    ]
    
    for scenario in scenarios:
        regime = scenario['regime']
        confidence = scenario['confidence']
        
        print(f"\nRegime: {regime}, Confidence: {confidence:.2f}")
        
        if regime == 'bear_market' and confidence > bear_threshold:
            print(f"  ❌ BLOCKED: Bear market confidence {confidence:.2f} > {bear_threshold:.2f}")
        elif regime == 'high_volatility' and confidence > vol_threshold:
            print(f"  ❌ BLOCKED: High volatility confidence {confidence:.2f} > {vol_threshold:.2f}")
        elif regime in ['sideways_market', 'recovery_market']:
            throttle_multiplier = get_config_float("regime_throttle_position_multiplier", 0.7)
            print(f"  ⚠️  THROTTLED: {regime} regime - position size reduced by {throttle_multiplier:.1f}x")
        else:
            print(f"  ✅ ALLOWED: {regime} regime")

if __name__ == "__main__":
    print("Exit Validation Script")
    print("Testing momentum decay, structure failure, VWAP filter, and regime enforcement\n")
    
    try:
        simulate_momentum_decay_exit()
        simulate_structure_failure_exit()
        simulate_vwap_entry_filter()
        simulate_regime_enforcement()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
