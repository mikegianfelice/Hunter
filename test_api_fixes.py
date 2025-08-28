#!/usr/bin/env python3
"""
Test script to verify API fixes work
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_jupiter_api():
    """Test Jupiter API with improved error handling"""
    print("🧪 Testing Jupiter API fixes...")
    
    from jupiter_lib import JupiterCustomLib
    from secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY
    
    # Initialize Jupiter lib
    jupiter = JupiterCustomLib(SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY)
    
    # Test with a known token (USDC -> SOL)
    usdc_mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    sol_mint = "So11111111111111111111111111111111111111112"
    
    print(f"🔄 Testing USDC -> SOL quote...")
    quote = jupiter.get_quote(usdc_mint, sol_mint, 1000000)  # 1 USDC
    
    if quote:
        print(f"✅ Quote successful: {quote.get('inAmount')} -> {quote.get('outAmount')}")
        
        # Test swap transaction
        print(f"🔄 Testing swap transaction...")
        swap_tx = jupiter.get_swap_transaction(quote)
        
        if swap_tx:
            print(f"✅ Swap transaction generated successfully")
            return True
        else:
            print(f"❌ Swap transaction failed")
            return False
    else:
        print(f"❌ Quote failed")
        return False

def test_ethereum_api():
    """Test Ethereum API"""
    print("🧪 Testing Ethereum API...")
    
    try:
        from uniswap_executor import get_eth_price_usd
        price = get_eth_price_usd()
        if price:
            print(f"✅ ETH price: ${price}")
            return True
        else:
            print(f"❌ ETH price fetch failed")
            return False
    except Exception as e:
        print(f"❌ Ethereum API error: {e}")
        return False

def main():
    print("🔧 Testing API Fixes")
    print("=" * 40)
    
    # Test Jupiter API
    jupiter_success = test_jupiter_api()
    
    # Test Ethereum API
    eth_success = test_ethereum_api()
    
    print("\n📊 Results:")
    print(f"Jupiter API: {'✅ PASS' if jupiter_success else '❌ FAIL'}")
    print(f"Ethereum API: {'✅ PASS' if eth_success else '❌ FAIL'}")
    
    if jupiter_success and eth_success:
        print("\n🎉 All API tests passed! The fixes are working.")
    else:
        print("\n⚠️ Some API tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
