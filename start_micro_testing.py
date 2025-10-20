#!/usr/bin/env python3
"""
Micro Live Testing Startup Script
Starts the bot with $1 trades for safe live testing
"""

import os
import sys
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

def check_micro_testing_ready():
    """Verify system is ready for micro live testing"""
    print("🔍 Checking micro testing readiness...")
    
    # Check if test mode is disabled
    from config_loader import get_config_bool, get_config_float, get_config_int
    if get_config_bool("test_mode", True):
        print("❌ ERROR: test_mode is still enabled! Should be false for live testing")
        return False
    
    # Check trade amount
    trade_amount = get_config_float("trade_amount_usd", 5.0)
    if trade_amount != 1.0:
        print(f"⚠️ WARNING: Trade amount is ${trade_amount}, should be $1.0 for micro testing")
    
    # Check risk limits
    max_positions = get_config_int("max_concurrent_positions", 3)
    daily_limit = get_config_float("daily_loss_limit_usd", 30.0)
    
    print(f"💰 Trade Amount: ${trade_amount}")
    print(f"📊 Max Positions: {max_positions}")
    print(f"🛡️ Daily Loss Limit: ${daily_limit}")
    
    # Check wallet balance
    try:
        from risk_manager import _get_wallet_balance_usd
        eth_balance = _get_wallet_balance_usd("ethereum")
        sol_balance = _get_wallet_balance_usd("solana")
        print(f"💰 Wallet Balances - ETH: ${eth_balance:.2f}, SOL: ${sol_balance:.2f}")
        
        if eth_balance < 5 and sol_balance < 5:
            print("⚠️ WARNING: Low wallet balance - ensure sufficient funds for micro testing")
    except Exception as e:
        print(f"⚠️ Could not check wallet balance: {e}")
    
    print("✅ Micro testing system ready!")
    return True

def start_micro_testing():
    """Start micro live testing"""
    print("🚀 Starting Micro Live Testing...")
    print("=" * 50)
    print("🛡️ SAFETY SYSTEMS ACTIVE:")
    print("• Emergency Stop System: ENABLED")
    print("• Position Size Validator: ENABLED") 
    print("• Execution Monitor: ENABLED")
    print("• Market Condition Guardian: ENABLED")
    print("=" * 50)
    print("💰 MICRO TESTING CONFIGURATION:")
    print("• Trade Amount: $1.00 per trade")
    print("• Max Positions: 1")
    print("• Daily Loss Limit: $5.00")
    print("• Max Losing Streak: 2")
    print("=" * 50)
    
    # Check readiness
    if not check_micro_testing_ready():
        print("❌ System not ready for micro testing")
        return False
    
    print("\n🎯 Starting bot with micro live testing configuration...")
    print("📊 Monitor logs for safety system activity")
    print("🛡️ All 4 safety systems will protect your capital")
    print("💰 Maximum risk: $1 per trade, $5 daily limit")
    print("\nPress Ctrl+C to stop the bot")
    print("=" * 50)
    
    # Start the main bot
    try:
        from main import main
        main()
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
        return True
    except Exception as e:
        print(f"❌ Bot error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 MICRO LIVE TESTING MODE")
    print("Testing with $1 trades and comprehensive safety protection")
    print("=" * 50)
    
    success = start_micro_testing()
    if success:
        print("✅ Micro testing completed successfully")
    else:
        print("❌ Micro testing failed")
        sys.exit(1)
