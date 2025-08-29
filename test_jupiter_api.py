#!/usr/bin/env python3
"""
Jupiter API Diagnostic Test Script
Tests Jupiter API with tokens that meet requirements to identify exact issues
"""

import requests
import json
import time
from typing import Dict, Any, Optional

def test_jupiter_quote_api(input_mint: str, output_mint: str, amount: str, description: str):
    """Test Jupiter quote API with specific parameters"""
    print(f"\n🔍 Testing Jupiter Quote API: {description}")
    print(f"   Input: {input_mint}")
    print(f"   Output: {output_mint}")
    print(f"   Amount: {amount}")
    
    url = "https://quote-api.jup.ag/v6/quote"
    params = {
        "inputMint": input_mint,
        "outputMint": output_mint,
        "amount": amount,
        "slippageBps": 200,  # 2% slippage
        "onlyDirectRoutes": "false",
        "asLegacyTransaction": "false"
    }
    
    try:
        print(f"   📡 Requesting: {url}")
        print(f"   📋 Params: {json.dumps(params, indent=6)}")
        
        response = requests.get(url, params=params, timeout=15)
        
        print(f"   📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Success! Response keys: {list(data.keys())}")
            
            if data.get("inAmount") and data.get("outAmount"):
                in_amount = int(data["inAmount"])
                out_amount = int(data["outAmount"])
                print(f"   💰 Quote: {in_amount} -> {out_amount}")
                
                if data.get("swapUsdValue"):
                    print(f"   💵 USD Value: ${data['swapUsdValue']}")
                
                return True
            else:
                print(f"   ⚠️ Missing quote data: {data}")
                return False
        else:
            print(f"   ❌ Error Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   💥 Exception: {e}")
        return False

def test_sol_price_apis():
    """Test all SOL price APIs"""
    print("\n🌞 Testing SOL Price APIs")
    
    # Test 1: CoinGecko
    print("\n1️⃣ CoinGecko SOL Price:")
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd"
        response = requests.get(url, timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            price = data.get("solana", {}).get("usd")
            print(f"   ✅ SOL Price: ${price}")
        else:
            print(f"   ❌ Error: {response.text}")
    except Exception as e:
        print(f"   💥 Exception: {e}")
    
    # Test 2: Jupiter Quote API for SOL price
    print("\n2️⃣ Jupiter Quote API for SOL Price:")
    success = test_jupiter_quote_api(
        "So11111111111111111111111111111111111111112",  # SOL
        "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
        "1000000000",  # 1 SOL in lamports
        "SOL -> USDC (1 SOL)"
    )
    
    # Test 3: DexScreener
    print("\n3️⃣ DexScreener SOL Price:")
    try:
        url = "https://api.dexscreener.com/latest/dex/tokens/So11111111111111111111111111111111111111112"
        response = requests.get(url, timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            pairs = data.get("pairs", [])
            if pairs:
                for pair in pairs[:3]:  # Show first 3 pairs
                    price = pair.get("priceUsd")
                    dex = pair.get("dexId")
                    print(f"   ✅ {dex}: ${price}")
        else:
            print(f"   ❌ Error: {response.text}")
    except Exception as e:
        print(f"   💥 Exception: {e}")

def test_token_validation():
    """Test Jupiter pre-check validation with various tokens"""
    print("\n🎯 Testing Token Validation")
    
    # Test tokens from recent logs that had issues
    test_tokens = [
        {
            "symbol": "MOON",
            "address": "4KeTt4e3vs9KKvRWttdzYQretUpZSkQhXs5y6QURzd7y",
            "description": "MOON token that failed Jupiter pre-check"
        },
        {
            "symbol": "DTF6900", 
            "address": "6M9rLwZ9rDWAED7U552NwBDtrTy3nWiepBjGv268LFWR",
            "description": "DTF6900 token with address validation issue"
        },
        {
            "symbol": "SR",
            "address": "7EtoZGFSuNnCGPDWtaZAe44nxiMyC5uPR8Lc7q6HqEvy", 
            "description": "SR token that was allowed despite validation issue"
        },
        {
            "symbol": "HOT",
            "address": "35QDv6oiyA5uKvT4ivWsxxr16E3VtWuymbq6kegZuYUm",
            "description": "HOT token that was not tradeable"
        }
    ]
    
    for token in test_tokens:
        print(f"\n🔍 Testing {token['symbol']}: {token['description']}")
        
        # Test Jupiter pre-check (same as strategy.py)
        url = "https://quote-api.jup.ag/v6/quote"
        params = {
            "inputMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
            "outputMint": token["address"],
            "amount": "1000000",  # 1 USDC test amount
            "slippageBps": 100,  # 1% slippage
            "onlyDirectRoutes": "false",
            "asLegacyTransaction": "false"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get("data"):
                    print(f"   ✅ {token['symbol']} is tradeable")
                else:
                    error_msg = data.get('error', 'Unknown error')
                    print(f"   ❌ {token['symbol']} not tradeable - {error_msg}")
            elif response.status_code == 400:
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', 'Bad Request')
                    print(f"   ❌ {token['symbol']} 400 error - {error_msg}")
                    
                    if "cannot be parsed" in error_msg.lower() or "invalid" in error_msg.lower():
                        print(f"   🔍 Address validation issue detected")
                    elif "not tradable" in error_msg.lower():
                        print(f"   🔍 Token not tradeable issue detected")
                except:
                    print(f"   ❌ {token['symbol']} 400 error - could not parse")
            else:
                print(f"   ⚠️ {token['symbol']} status {response.status_code}")
                
        except Exception as e:
            print(f"   💥 {token['symbol']} exception: {e}")

def test_quote_with_good_tokens():
    """Test Jupiter quotes with tokens that should work"""
    print("\n✅ Testing Jupiter Quotes with Good Tokens")
    
    # Test with some well-known Solana tokens
    good_tokens = [
        {
            "symbol": "BONK",
            "address": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
            "description": "BONK token (should be tradeable)"
        },
        {
            "symbol": "JUP",
            "address": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
            "description": "Jupiter token (should be tradeable)"
        }
    ]
    
    for token in good_tokens:
        print(f"\n🔍 Testing {token['symbol']}: {token['description']}")
        
        # Test USDC -> Token
        success = test_jupiter_quote_api(
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
            token["address"],
            "1000000",  # 1 USDC
            f"USDC -> {token['symbol']} (1 USDC)"
        )
        
        if success:
            print(f"   ✅ {token['symbol']} quote successful")
        else:
            print(f"   ❌ {token['symbol']} quote failed")

def test_address_validation():
    """Test address format validation"""
    print("\n🔍 Testing Address Format Validation")
    
    test_addresses = [
        {
            "address": "4KeTt4e3vs9KKvRWttdzYQretUpZSkQhXs5y6QURzd7y",
            "length": len("4KeTt4e3vs9KKvRWttdzYQretUpZSkQhXs5y6QURzd7y"),
            "description": "MOON token address"
        },
        {
            "address": "6M9rLwZ9rDWAED7U552NwBDtrTy3nWiepBjGv268LFWR", 
            "length": len("6M9rLwZ9rDWAED7U552NwBDtrTy3nWiepBjGv268LFWR"),
            "description": "DTF6900 token address"
        },
        {
            "address": "jG97DK3ASuZzYi5zBkQdr6Zf7s1ekdXkJbhvyyMiLmW",
            "length": len("jG97DK3ASuZzYi5zBkQdr6Zf7s1ekdXkJbhvyyMiLmW"),
            "description": "DTF6900 variant address (43 chars)"
        }
    ]
    
    for addr_info in test_addresses:
        print(f"\n📏 {addr_info['description']}:")
        print(f"   Address: {addr_info['address']}")
        print(f"   Length: {addr_info['length']} characters")
        
        # Check if it's a valid base58 string
        try:
            import base58
            decoded = base58.b58decode(addr_info['address'])
            print(f"   ✅ Valid base58 (decoded length: {len(decoded)})")
        except Exception as e:
            print(f"   ❌ Invalid base58: {e}")

def main():
    """Run all Jupiter API tests"""
    print("🚀 Jupiter API Diagnostic Test Suite")
    print("=" * 50)
    
    # Test 1: SOL Price APIs
    test_sol_price_apis()
    
    # Test 2: Token Validation
    test_token_validation()
    
    # Test 3: Quote with Good Tokens
    test_quote_with_good_tokens()
    
    # Test 4: Address Validation
    test_address_validation()
    
    print("\n" + "=" * 50)
    print("🏁 Jupiter API Diagnostic Complete")
    print("Check the output above for specific issues and patterns")

if __name__ == "__main__":
    main()
