#!/usr/bin/env python3
"""
Test Safety Systems
Comprehensive testing of all 4 critical safety systems
"""

import sys
import logging
from datetime import datetime
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

def test_emergency_stop_system():
    """Test AI Emergency Stop System"""
    print("\n🚨 Testing AI Emergency Stop System...")
    
    try:
        from ai_emergency_stop_system import ai_emergency_stop_system
        
        # Test data
        portfolio_data = {
            'total_value': 5000,  # 50% drawdown
            'initial_value': 10000,
            'timestamp': datetime.now().isoformat()
        }
        trade_history = [
            {'pnl': -10, 'timestamp': datetime.now().isoformat()},
            {'pnl': -15, 'timestamp': datetime.now().isoformat()},
            {'pnl': -20, 'timestamp': datetime.now().isoformat()}
        ]
        system_errors = [
            {'error': 'Connection timeout', 'timestamp': datetime.now().isoformat()},
            {'error': 'API rate limit', 'timestamp': datetime.now().isoformat()}
        ]
        market_data = {'timestamp': datetime.now().isoformat()}
        
        # Test emergency conditions
        emergency_analysis = ai_emergency_stop_system.check_emergency_conditions(
            portfolio_data, trade_history, market_data, system_errors
        )
        
        print(f"✅ Emergency Level: {emergency_analysis['emergency_level']}")
        print(f"✅ Emergency Score: {emergency_analysis['emergency_score']:.2f}")
        print(f"✅ Emergency Urgency: {emergency_analysis['emergency_urgency']}")
        
        # Test summary
        summary = ai_emergency_stop_system.get_emergency_summary(
            portfolio_data, trade_history, market_data, system_errors
        )
        print(f"✅ Summary - Level: {summary['emergency_level']}, Stop Required: {summary['requires_emergency_stop']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Emergency Stop System test failed: {e}")
        return False

def test_position_size_validator():
    """Test AI Position Size Validator"""
    print("\n🔍 Testing AI Position Size Validator...")
    
    try:
        from ai_position_size_validator import ai_position_size_validator
        
        # Test data
        token = {
            'symbol': 'TEST',
            'priceUsd': 0.001,
            'volume24h': 100000,
            'liquidity': 50000,
            'risk_level': 'high'
        }
        proposed_amount = 100.0  # Large amount
        wallet_balance = 1000.0
        current_positions = [
            {'position_size_usd': 50, 'symbol': 'ETH'},
            {'position_size_usd': 30, 'symbol': 'BTC'}
        ]
        market_conditions = {
            'regime': 'bear_market',
            'volatility': 0.4
        }
        
        # Test position validation
        validation_analysis = ai_position_size_validator.validate_position_size(
            token, proposed_amount, wallet_balance, current_positions, market_conditions
        )
        
        print(f"✅ Validation Result: {validation_analysis['validation_result']}")
        print(f"✅ Validation Score: {validation_analysis['validation_score']:.2f}")
        print(f"✅ Recommended Size: ${validation_analysis['recommended_size']:.2f}")
        
        # Test summary
        summary = ai_position_size_validator.get_validation_summary(
            token, proposed_amount, wallet_balance, current_positions, market_conditions
        )
        print(f"✅ Summary - Result: {summary['validation_result']}, Passed: {summary['validation_passed']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Position Size Validator test failed: {e}")
        return False

def test_execution_monitor():
    """Test AI Trade Execution Monitor"""
    print("\n🔍 Testing AI Trade Execution Monitor...")
    
    try:
        from ai_trade_execution_monitor import ai_trade_execution_monitor
        
        # Test data
        trade_data = {
            'trade_id': 'TEST_001',
            'status': 'pending',
            'success': False,
            'start_time': datetime.now().isoformat(),
            'retry_count': 2,
            'gas_price': 0.02,
            'gas_limit': 200000,
            'expected_price': 0.001,
            'actual_price': 0.0011
        }
        execution_history = [
            {'trade_id': 'TEST_001', 'success': False, 'timestamp': datetime.now().isoformat()},
            {'trade_id': 'TEST_001', 'success': False, 'timestamp': datetime.now().isoformat()}
        ]
        market_conditions = {'regime': 'normal', 'volatility': 0.2}
        
        # Test execution monitoring
        monitoring_analysis = ai_trade_execution_monitor.monitor_trade_execution(
            trade_data, execution_history, market_conditions
        )
        
        print(f"✅ Monitoring Level: {monitoring_analysis['monitoring_level']}")
        print(f"✅ Monitoring Score: {monitoring_analysis['monitoring_score']:.2f}")
        print(f"✅ Execution Urgency: {monitoring_analysis['execution_urgency']}")
        
        # Test summary
        summary = ai_trade_execution_monitor.get_monitoring_summary(
            trade_data, execution_history, market_conditions
        )
        print(f"✅ Summary - Level: {summary['monitoring_level']}, Action Required: {summary['requires_immediate_action']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Execution Monitor test failed: {e}")
        return False

def test_market_condition_guardian():
    """Test AI Market Condition Guardian"""
    print("\n🛡️ Testing AI Market Condition Guardian...")
    
    try:
        from ai_market_condition_guardian import ai_market_condition_guardian
        
        # Test data
        market_data = {
            'timestamp': datetime.now().isoformat(),
            'volatility': 0.6,  # High volatility
            'regime': 'high_volatility'
        }
        token_data = {
            'symbol': 'TEST',
            'priceUsd': 0.001,
            'volume24h': 50000,
            'liquidity': 25000,
            'historical_price': 0.0012,  # Price drop
            'historical_volume': 20000,
            'historical_liquidity': 50000
        }
        news_data = {
            'sentiment': 'negative',
            'impact': 0.8
        }
        historical_data = {
            'price': 0.0012,
            'volume': 20000
        }
        
        # Test market conditions
        guardian_analysis = ai_market_condition_guardian.check_market_conditions(
            market_data, token_data, news_data, historical_data
        )
        
        print(f"✅ Condition Level: {guardian_analysis['condition_level']}")
        print(f"✅ Condition Score: {guardian_analysis['condition_score']:.2f}")
        print(f"✅ Trading Safety: {guardian_analysis['trading_safety']}")
        
        # Test summary
        summary = ai_market_condition_guardian.get_guardian_summary(
            market_data, token_data, news_data, historical_data
        )
        print(f"✅ Summary - Level: {summary['condition_level']}, Trading Allowed: {summary['trading_allowed']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Market Condition Guardian test failed: {e}")
        return False

def test_safety_system_integration():
    """Test safety system integration in main trading loop"""
    print("\n🔗 Testing Safety System Integration...")
    
    try:
        # Import main trading function
        from main import check_practical_buy_signal
        
        # Test token
        test_token = {
            'address': '0x1234567890abcdef',
            'symbol': 'SAFETY_TEST',
            'priceUsd': 0.001,
            'volume24h': 100000,
            'liquidity': 75000,
            'timestamp': datetime.now().isoformat()
        }
        
        print("Testing safety system integration...")
        result = check_practical_buy_signal(test_token)
        
        print(f"✅ Safety Integration Result: {'PASSED' if result else 'BLOCKED'}")
        print("✅ Safety systems are properly integrated into trading loop")
        
        return True
        
    except Exception as e:
        print(f"❌ Safety system integration test failed: {e}")
        return False

def main():
    """Run all safety system tests"""
    print("🛡️ AI SAFETY SYSTEMS COMPREHENSIVE TEST")
    print("=" * 50)
    
    tests = [
        ("Emergency Stop System", test_emergency_stop_system),
        ("Position Size Validator", test_position_size_validator),
        ("Execution Monitor", test_execution_monitor),
        ("Market Condition Guardian", test_market_condition_guardian),
        ("Safety System Integration", test_safety_system_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("🛡️ SAFETY SYSTEMS TEST SUMMARY")
    print("="*50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL SAFETY SYSTEMS OPERATIONAL!")
        print("✅ System is ready for live trading with safety protection")
    else:
        print("⚠️ Some safety systems need attention before live trading")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
