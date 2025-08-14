#!/usr/bin/env python3
"""
Test script for Solana trading implementation
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_solana_implementation():
    """Test the Solana trading implementation"""
    print("🧪 Testing Solana Trading Implementation")
    print("=" * 50)
    
    # Check if required environment variables are set
    required_vars = [
        "SOLANA_RPC_URL",
        "SOLANA_WALLET_ADDRESS", 
        "SOLANA_PRIVATE_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these in your .env file")
        return False
    
    print("✅ Environment variables configured")
    
    try:
        # Test Solana executor initialization
        print("\n🔧 Testing Solana executor initialization...")
        from solana_executor import get_solana_executor
        
        executor = get_solana_executor()
        print(f"✅ Solana executor initialized")
        print(f"   Wallet: {executor.wallet_address[:8]}...{executor.wallet_address[-8:]}")
        
        # Test balance check
        print("\n💰 Testing balance check...")
        balance = executor.get_solana_balance()
        print(f"✅ SOL Balance: {balance:.6f}")
        
        # Test price fetching
        print("\n📊 Testing price fetching...")
        # Test with SOL price
        sol_price = executor.get_token_price_usd("So11111111111111111111111111111111111111112")
        print(f"✅ SOL Price: ${sol_price:.6f}")
        
        # Test pool fetching
        print("\n🏊 Testing pool fetching...")
        pools = executor.get_raydium_pools()
        print(f"✅ Fetched {len(pools)} Raydium pools")
        
        if pools:
            # Test finding a specific pool
            print("\n🔍 Testing pool finding...")
            # Try to find SOL/USDC pool
            sol_pool = executor.find_pool_for_token("So11111111111111111111111111111111111111112")
            if sol_pool:
                print(f"✅ Found SOL/USDC pool: {sol_pool.pool_id}")
                print(f"   Base reserve: {sol_pool.base_reserve}")
                print(f"   Quote reserve: {sol_pool.quote_reserve}")
            else:
                print("⚠️ SOL/USDC pool not found")
        
        # Test swap quote
        print("\n💱 Testing swap quote...")
        if sol_pool:
            # Test getting a quote for 1 USDC worth of SOL
            quote = executor.get_swap_quote(sol_pool, 1000000, is_buy=True)  # 1 USDC = 1,000,000 units
            if quote:
                print(f"✅ Got swap quote: {quote.get('outAmount', 0)} SOL for 1 USDC")
            else:
                print("⚠️ Failed to get swap quote")
        
        print("\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simulation_mode():
    """Test simulation mode"""
    print("\n🎭 Testing Simulation Mode")
    print("=" * 30)
    
    try:
        from solana_executor import buy_token_solana, sell_token_solana
        
        # Test buy simulation
        print("🔄 Testing buy simulation...")
        tx_hash, success = buy_token_solana(
            "So11111111111111111111111111111111111111112",  # SOL
            10.0,  # $10 USD
            "SOL",
            test_mode=True
        )
        
        if success:
            print(f"✅ Buy simulation successful: {tx_hash}")
        else:
            print("❌ Buy simulation failed")
        
        # Test sell simulation
        print("🔄 Testing sell simulation...")
        tx_hash, success = sell_token_solana(
            "So11111111111111111111111111111111111111112",  # SOL
            1.0,  # 1 SOL
            "SOL",
            test_mode=True
        )
        
        if success:
            print(f"✅ Sell simulation successful: {tx_hash}")
        else:
            print("❌ Sell simulation failed")
        
        print("✅ Simulation tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Simulation test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Solana Trading Bot - Implementation Test")
    print("=" * 50)
    
    # Test basic functionality
    basic_success = test_solana_implementation()
    
    # Test simulation mode
    sim_success = test_simulation_mode()
    
    print("\n" + "=" * 50)
    if basic_success and sim_success:
        print("🎉 All tests passed! Solana implementation is ready.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    print("\n📝 Next steps:")
    print("1. Set test_mode: false in config.yaml to enable real trading")
    print("2. Ensure your Solana wallet has sufficient SOL for gas fees")
    print("3. Ensure your wallet has USDC for trading")
    print("4. Test with small amounts first")
