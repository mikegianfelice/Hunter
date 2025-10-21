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

from secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY
from config_loader import get_config, get_config_bool, get_config_float
from raydium_lib import RaydiumCustomLib

# Common token addresses
WSOL_MINT = "So11111111111111111111111111111111111111112"  # Wrapped SOL
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # USDC

class RaydiumExecutor:
    def __init__(self):
        self.wallet_address = SOLANA_WALLET_ADDRESS
        self.private_key = SOLANA_PRIVATE_KEY
        
        # Initialize custom Raydium library
        try:
            if self.private_key:
                self.raydium_lib = RaydiumCustomLib(
                    SOLANA_RPC_URL,
                    self.wallet_address,
                    self.private_key
                )
                print(f"✅ Raydium executor initialized with wallet: {self.wallet_address[:8]}...{self.wallet_address[-8:]}")
                self.keypair = True
            else:
                print("⚠️ No Solana private key provided for Raydium")
                self.raydium_lib = None
                self.keypair = None
        except Exception as e:
            print(f"❌ Failed to initialize Raydium wallet: {e}")
            self.raydium_lib = None
            self.keypair = None

    def check_raydium_liquidity(self, token_address: str) -> Dict[str, Any]:
        """Check if token has liquidity on Raydium"""
        try:
            # Try Raydium API v2 first
            url = "https://api.raydium.io/v2/sdk/liquidity/mainnet.json"
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
            
            # Try alternative Raydium API endpoint
            url2 = "https://api.raydium.io/v2/main/pool"
            response2 = requests.get(url2, timeout=10)
            
            if response2.status_code == 200:
                data2 = response2.json()
                pools2 = data2.get("pools", [])
                
                for pool in pools2:
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
        """Get quote from Raydium API using DexScreener as fallback"""
        try:
            # Try Raydium API v2 for quotes first
            url = "https://api.raydium.io/v2/sdk/quote"
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
            elif response.status_code == 500:
                print(f"⚠️ Raydium SDK quote API server error (500) - trying alternative...")
            else:
                print(f"⚠️ Raydium SDK quote returned {response.status_code}")
            
            # Try alternative Raydium quote endpoint
            url2 = "https://api.raydium.io/v2/main/quote"
            response2 = requests.get(url2, params=params, timeout=15)
            
            if response2.status_code == 200:
                data2 = response2.json()
                if data2.get("success"):
                    return {
                        "success": True,
                        "inAmount": data2.get("inAmount"),
                        "outAmount": data2.get("outAmount"),
                        "priceImpact": data2.get("priceImpact"),
                        "route": data2.get("route")
                    }
            elif response2.status_code == 500:
                print(f"⚠️ Raydium main quote API also returning 500 server error")
            else:
                print(f"⚠️ Raydium main quote returned {response2.status_code}")
            
            # Fallback: Use DexScreener to get price and calculate quote
            print("   🔄 Using DexScreener fallback for quote...")
            url3 = f"https://api.dexscreener.com/latest/dex/tokens/{output_mint}"
            response3 = requests.get(url3, timeout=10)
            
            if response3.status_code == 200:
                data3 = response3.json()
                pairs = data3.get("pairs", [])
                
                # Find Raydium pair
                for pair in pairs:
                    dex_id = pair.get("dexId", "").lower()
                    if "raydium" in dex_id:
                        price_usd = float(pair.get("priceUsd", 0))
                        if price_usd > 0:
                            # Calculate quote based on price
                            input_usd = amount / 1000000  # Convert from USDC decimals
                            output_tokens = input_usd / price_usd
                            
                            return {
                                "success": True,
                                "inAmount": str(amount),
                                "outAmount": str(int(output_tokens * 1000000000)),  # Convert to token decimals
                                "priceImpact": 0.1,  # Estimated
                                "route": "raydium-dexscreener-fallback"
                            }
            
            return None
            
        except Exception as e:
            print(f"⚠️ Raydium quote failed: {e}")
            return None

    def get_raydium_swap_transaction(self, quote: Dict[str, Any], slippage: float) -> Optional[str]:
        """Get swap transaction from Raydium API"""
        try:
            # Use Raydium's swap API to get transaction
            url = "https://api.raydium.io/v2/sdk/swap"
            
            payload = {
                "quoteResponse": quote,
                "userPublicKey": self.wallet_address,
                "wrapUnwrapSOL": True,
                "computeUnitPriceMicroLamports": 1000,
                "asLegacyTransaction": True,
                "useSharedAccounts": False,
                "maxAccounts": 16
            }
            
            response = requests.post(url, json=payload, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if "swapTransaction" in data:
                    print(f"✅ Raydium swap transaction generated")
                    return data["swapTransaction"]
                else:
                    print(f"❌ No swap transaction in response: {data}")
                    return None
            else:
                print(f"❌ Raydium swap API failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Failed to get Raydium swap transaction: {e}")
            return None

    def sign_raydium_transaction(self, transaction_data: str) -> Optional[str]:
        """Sign Raydium transaction with wallet"""
        try:
            if not self.keypair:
                print("❌ No keypair available for signing")
                return None
            
            # Decode transaction
            import base64
            decoded_bytes = base64.b64decode(transaction_data)
            
            # Parse transaction structure
            num_signatures = decoded_bytes[0]
            signature_length = 64
            signatures_end = 1 + (num_signatures * signature_length)
            message_bytes = decoded_bytes[signatures_end:]
            
            # Sign the message
            signature = self.keypair.sign_message(message_bytes)
            signature_bytes = bytes(signature)
            
            # Reconstruct transaction with signature
            import struct
            reconstructed = struct.pack('B', num_signatures)
            reconstructed += signature_bytes
            reconstructed += message_bytes
            
            # Encode back to base64
            signed_transaction = base64.b64encode(reconstructed).decode('utf-8')
            
            print(f"✅ Raydium transaction signed successfully")
            return signed_transaction
            
        except Exception as e:
            print(f"❌ Raydium transaction signing failed: {e}")
            return None

    def send_raydium_transaction(self, signed_transaction: str) -> Optional[str]:
        """Send signed Raydium transaction to Solana network"""
        try:
            # Check transaction size
            import base64
            decoded = base64.b64decode(signed_transaction)
            tx_size = len(decoded)
            max_size = 1644
            
            if tx_size > max_size:
                print(f"❌ Transaction too large: {tx_size} bytes (max: {max_size})")
                return None
            
            # Send to Solana RPC
            rpc_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "sendTransaction",
                "params": [
                    signed_transaction,
                    {
                        "encoding": "base64",
                        "skipPreflight": True,
                        "maxRetries": 3
                    }
                ]
            }
            
            try:
                response = requests.post(self.rpc_url, json=rpc_payload, timeout=30)
            except requests.exceptions.ConnectionError as e:
                error_msg = str(e).lower()
                if "broken pipe" in error_msg or "errno 32" in error_msg:
                    print(f"❌ Broken pipe error sending Raydium transaction: Connection closed unexpectedly")
                else:
                    print(f"❌ Connection error sending Raydium transaction: {e}")
                return None
            except OSError as e:
                if e.errno == 32:  # Broken pipe
                    print(f"❌ Broken pipe error (errno 32) sending Raydium transaction: {e}")
                else:
                    print(f"❌ OS error sending Raydium transaction: {e}")
                return None
            
            if response.status_code == 200:
                result = response.json()
                if "result" in result:
                    tx_hash = result["result"]
                    print(f"✅ Raydium transaction sent: {tx_hash}")
                    return tx_hash
                elif "error" in result:
                    error_msg = result["error"]
                    print(f"❌ RPC error: {error_msg}")
                    return None
            else:
                print(f"❌ RPC request failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Failed to send Raydium transaction: {e}")
            return None

    def execute_raydium_swap(self, input_mint: str, output_mint: str, amount: int, slippage: float = 0.10) -> Tuple[bool, str]:
        """Execute real swap on Raydium using custom library"""
        try:
            if not self.raydium_lib:
                print("❌ No Raydium library available for swap")
                return False, "No Raydium library"
            
            print(f"🔄 Executing real Raydium swap: {input_mint[:8]}... -> {output_mint[:8]}...")
            
            # Use custom Raydium library to execute the swap
            tx_hash, success = self.raydium_lib.execute_swap(input_mint, output_mint, amount, slippage)
            
            if success:
                print(f"✅ Real Raydium swap executed! TX: {tx_hash}")
                return True, tx_hash
            else:
                print(f"❌ Raydium swap failed: {tx_hash}")
                return False, tx_hash
            
        except Exception as e:
            print(f"❌ Raydium swap failed: {e}")
            return False, str(e)

    def check_token_tradeable_on_raydium(self, token_address: str) -> bool:
        """Check if token is tradeable on Raydium"""
        try:
            # First try DexScreener to check if token has Raydium pairs
            url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                pairs = data.get("pairs", [])
                
                if pairs is None:
                    print(f"❌ Token {token_address[:8]}... has no pairs data")
                    return False
                
                # Look for Raydium pairs
                for pair in pairs:
                    dex_id = pair.get("dexId", "").lower()
                    if "raydium" in dex_id:
                        liquidity_data = pair.get("liquidity", {})
                        if isinstance(liquidity_data, dict):
                            liquidity = float(liquidity_data.get("usd", 0))
                        else:
                            liquidity = float(liquidity_data or 0)
                        
                        if liquidity > 1000:  # Minimum $1000 liquidity
                            print(f"✅ Token {token_address[:8]}... has Raydium liquidity: ${liquidity:,.2f}")
                            return True
            
            # Fallback to direct Raydium API check
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
