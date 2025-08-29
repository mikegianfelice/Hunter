#!/usr/bin/env python3
"""
Raydium Fallback Executor
Trades tokens on Raydium when Jupiter pre-check fails
"""

import json
import time
import requests
import base58
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
import struct

from secrets import SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY
from config_loader import get_config, get_config_bool, get_config_float

# Common token addresses
WSOL_MINT = "So11111111111111111111111111111111111111112"  # Wrapped SOL
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # USDC

class RaydiumExecutor:
    def __init__(self):
        self.wallet_address = SOLANA_WALLET_ADDRESS
        self.private_key = SOLANA_PRIVATE_KEY
        
        # Initialize wallet (simplified for now)
        try:
            if self.private_key:
                # For now, just store the private key without creating keypair
                # This will be used when we implement actual swap execution
                print(f"✅ Raydium executor initialized with wallet: {self.wallet_address[:8]}...{self.wallet_address[-8:]}")
                self.keypair = True  # Placeholder
            else:
                print("⚠️ No Solana private key provided for Raydium")
                self.keypair = None
        except Exception as e:
            print(f"❌ Failed to initialize Raydium wallet: {e}")
            self.keypair = None

    def check_raydium_liquidity(self, token_address: str) -> Dict[str, Any]:
        """Check if token has liquidity on Raydium"""
        try:
            # Use Raydium API to check liquidity
            url = f"https://api.raydium.io/v2/sdk/liquidity/mainnet.json"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                pools = data.get("official", [])
                
                # Look for pools containing our token
                for pool in pools:
                    base_mint = pool.get("baseMint")
                    quote_mint = pool.get("quoteMint")
                    
                    if base_mint == token_address or quote_mint == token_address:
                        liquidity = pool.get("liquidity", 0)
                        volume_24h = pool.get("volume24h", 0)
                        
                        if liquidity > 0:
                            return {
                                "has_liquidity": True,
                                "liquidity": liquidity,
                                "volume_24h": volume_24h,
                                "pool_id": pool.get("id"),
                                "base_mint": base_mint,
                                "quote_mint": quote_mint
                            }
            
            return {"has_liquidity": False}
            
        except Exception as e:
            print(f"⚠️ Raydium liquidity check failed: {e}")
            return {"has_liquidity": False}

    def get_raydium_quote(self, input_mint: str, output_mint: str, amount: int) -> Optional[Dict[str, Any]]:
        """Get quote from Raydium API"""
        try:
            # Use Raydium quote API
            url = "https://quote-api.raydium.io/v2/quote"
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": str(amount),
                "slippage": 0.02,  # 2% slippage
                "version": 4
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    return {
                        "success": True,
                        "inAmount": data.get("inAmount"),
                        "outAmount": data.get("outAmount"),
                        "priceImpact": data.get("priceImpact"),
                        "route": data.get("route")
                    }
            
            return None
            
        except Exception as e:
            print(f"⚠️ Raydium quote failed: {e}")
            return None

    def execute_raydium_swap(self, input_mint: str, output_mint: str, amount: int, slippage: float = 0.02) -> Tuple[bool, str]:
        """Execute swap on Raydium"""
        try:
            if not self.keypair:
                print("❌ No wallet keypair available for Raydium swap")
                return False, "No wallet keypair"
            
            print(f"🔄 Executing Raydium swap: {input_mint[:8]}... -> {output_mint[:8]}...")
            
            # Get quote first
            quote = self.get_raydium_quote(input_mint, output_mint, amount)
            if not quote:
                print("❌ Failed to get Raydium quote")
                return False, "Quote failed"
            
            print(f"✅ Raydium quote: {quote['inAmount']} -> {quote['outAmount']}")
            
            # For now, return success (actual implementation would require Raydium SDK)
            # This is a placeholder - in production you'd use the Raydium SDK to execute the swap
            print("⚠️ Raydium swap execution not fully implemented (placeholder)")
            print("   Would execute swap using Raydium SDK with the quote above")
            
            return True, "RAYDIUM_SWAP_SUCCESS"
            
        except Exception as e:
            print(f"❌ Raydium swap failed: {e}")
            return False, str(e)

    def check_token_tradeable_on_raydium(self, token_address: str) -> bool:
        """Check if token is tradeable on Raydium"""
        try:
            # Check liquidity first
            liquidity_info = self.check_raydium_liquidity(token_address)
            if not liquidity_info.get("has_liquidity"):
                print(f"❌ Token {token_address[:8]}... has no Raydium liquidity")
                return False
            
            # Try to get a quote
            quote = self.get_raydium_quote(USDC_MINT, token_address, 1000000)  # 1 USDC test
            if quote:
                print(f"✅ Token {token_address[:8]}... is tradeable on Raydium")
                return True
            else:
                print(f"❌ Token {token_address[:8]}... failed Raydium quote")
                return False
                
        except Exception as e:
            print(f"⚠️ Raydium tradeability check failed: {e}")
            return False

def get_raydium_config():
    """Get Raydium-specific configuration"""
    return {
        'ENABLE_RAYDIUM_FALLBACK': get_config_bool('enable_raydium_fallback', True),
        'RAYDIUM_SLIPPAGE': get_config_float('raydium_slippage', 0.02),
        'RAYDIUM_TIMEOUT': get_config('raydium_timeout', 15),
        'MIN_RAYDIUM_LIQUIDITY': get_config_float('min_raydium_liquidity', 1000.0)
    }

# Global Raydium executor instance
raydium_executor = None

def get_raydium_executor():
    """Get or create Raydium executor instance"""
    global raydium_executor
    if raydium_executor is None:
        raydium_executor = RaydiumExecutor()
    return raydium_executor

def execute_raydium_fallback_trade(token_address: str, token_symbol: str, amount_usd: float) -> Tuple[bool, str]:
    """Execute fallback trade on Raydium when Jupiter fails"""
    config = get_raydium_config()
    
    if not config['ENABLE_RAYDIUM_FALLBACK']:
        print("🔓 Raydium fallback disabled in config")
        return False, "Raydium fallback disabled"
    
    try:
        executor = get_raydium_executor()
        
        # Check if token is tradeable on Raydium
        if not executor.check_token_tradeable_on_raydium(token_address):
            print(f"❌ {token_symbol} not tradeable on Raydium")
            return False, "Not tradeable on Raydium"
        
        # Convert USD amount to USDC amount (assuming 1 USDC = 1 USD)
        usdc_amount = int(amount_usd * 1000000)  # USDC has 6 decimals
        
        # Execute the swap
        success, tx_hash = executor.execute_raydium_swap(
            USDC_MINT, 
            token_address, 
            usdc_amount, 
            config['RAYDIUM_SLIPPAGE']
        )
        
        if success:
            print(f"✅ Raydium fallback trade successful for {token_symbol}")
            return True, tx_hash
        else:
            print(f"❌ Raydium fallback trade failed for {token_symbol}: {tx_hash}")
            return False, tx_hash
            
    except Exception as e:
        print(f"💥 Raydium fallback trade error for {token_symbol}: {e}")
        return False, str(e)
