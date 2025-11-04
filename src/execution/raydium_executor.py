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

from ..config.secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY
from ..config.config_loader import get_config, get_config_bool, get_config_float
from .raydium_lib import RaydiumCustomLib
from ..monitoring.structured_logger import log_info, log_error

# Common token addresses
WSOL_MINT = "So11111111111111111111111111111111111111112"  # Wrapped SOL
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # USDC

def get_solana_base_currency():
    """Get the configured base currency for Solana trading (SOL or USDC)"""
    return get_config("solana_base_currency", "USDC")

class RaydiumExecutor:
    def __init__(self):
        self.wallet_address = SOLANA_WALLET_ADDRESS
        self.private_key = SOLANA_PRIVATE_KEY
        self.rpc_url = SOLANA_RPC_URL
        
        # Initialize custom Raydium library
        try:
            if self.private_key:
                self.raydium_lib = RaydiumCustomLib(
                    SOLANA_RPC_URL,
                    self.wallet_address,
                    self.private_key
                )
                log_info("raydium.init", wallet=self.wallet_address)
                # keypair management is handled by RaydiumCustomLib; keep a flag only
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
            log_info("raydium.liquidity_check_error", level="WARNING", error=str(e))
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
            log_info("raydium.quote.dexscreener_fallback")
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
            log_info("raydium.quote.error", level="WARNING", error=str(e))
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
                    log_info("raydium.swap.tx_generated")
                    return data["swapTransaction"]
                else:
                    print(f"❌ No swap transaction in response: {data}")
                    return None
            else:
                log_error("raydium.swap.api_failed", status=response.status_code)
                return None
                
        except Exception as e:
            log_error("raydium.swap.tx_error", error=str(e))
            return None

    def sign_raydium_transaction(self, transaction_data: str) -> Optional[str]:
        """Sign Raydium transaction with wallet via custom library"""
        try:
            if not self.raydium_lib:
                print("❌ No Raydium library available for signing")
                return None
            signed_transaction = self.raydium_lib.sign_transaction(transaction_data)
            if signed_transaction:
                print(f"✅ Raydium transaction signed successfully")
                return signed_transaction
            print(f"❌ Raydium library failed to sign transaction")
            return None
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
                log_error("raydium.swap.tx_too_large", size=tx_size, max=max_size)
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
                    log_error("raydium.swap.rpc_broken_pipe")
                else:
                    log_error("raydium.swap.rpc_conn_error", error=str(e))
                return None
            except OSError as e:
                if e.errno == 32:  # Broken pipe
                    log_error("raydium.swap.rpc_broken_pipe_os", error=str(e))
                else:
                    log_error("raydium.swap.rpc_os_error", error=str(e))
                return None
            
            if response.status_code == 200:
                result = response.json()
                if "result" in result:
                    tx_hash = result["result"]
                    log_info("raydium.swap.sent", tx_hash=tx_hash)
                    return tx_hash
                elif "error" in result:
                    error_msg = result["error"]
                    log_error("raydium.swap.rpc_error", error=error_msg)
                    return None
            else:
                log_error("raydium.swap.rpc_http_failed", status=response.status_code)
                return None
                
        except Exception as e:
            log_error("raydium.swap.send_exception", error=str(e))
            return None

    def execute_trade(self, token_address: str, amount_usd: float, is_buy: bool = True) -> Tuple[bool, str]:
        """
        Compatibility wrapper for execute_trade method (matches Jupiter interface)
        Converts USD amount to appropriate token amounts and executes swap
        """
        try:
            if not self.raydium_lib:
                return False, "No Raydium library"
            
            # Get base currency configuration
            base_currency = get_solana_base_currency()
            
            # Convert USD to appropriate input amount
            if is_buy:
                # Buying token
                if base_currency == "USDC":
                    # Buying token with USDC (1 USDC = $1, 6 decimals)
                    amount = int(float(amount_usd) * 1_000_000)  # USDC has 6 decimals
                    input_mint = USDC_MINT
                    output_mint = token_address
                    log_info("raydium.trade.buy_with_usdc", amount_usd=amount_usd, usdc_amount=amount)
                else:
                    # Legacy: Buying token with SOL
                    try:
                        # Import with fallback
                        try:
                            from src.utils.utils import get_sol_price_usd
                        except ImportError:
                            try:
                                from ..utils.utils import get_sol_price_usd
                            except ImportError:
                                import sys
                                import os
                                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
                                from src.utils.utils import get_sol_price_usd
                        
                        sol_price = get_sol_price_usd()
                        if sol_price <= 0:
                            return False, "Cannot get SOL price"
                        sol_amount = amount_usd / sol_price
                        amount = int(sol_amount * 1_000_000_000)  # SOL has 9 decimals
                        input_mint = WSOL_MINT
                        output_mint = token_address
                    except Exception as e:
                        return False, f"Failed to convert USD to SOL: {e}"
            else:
                # Selling token - need raw token amount in smallest units
                # CRITICAL: We need the actual token balance in raw units, not USD amount
                raw_token_balance = self.get_token_raw_balance(token_address)
                if raw_token_balance is None or raw_token_balance <= 0:
                    return False, "No token balance available"
                
                input_mint = token_address
                # Determine output currency based on base currency configuration
                if base_currency == "USDC":
                    output_mint = USDC_MINT
                    log_info("raydium.trade.sell_to_usdc", token=token_address)
                else:
                    output_mint = WSOL_MINT
                    log_info("raydium.trade.sell_to_sol", token=token_address)
                amount = raw_token_balance  # Use raw token amount for swap
            
            # Execute swap with default slippage
            return self.execute_raydium_swap(input_mint, output_mint, amount, slippage=0.10)
            
        except Exception as e:
            log_error("raydium.execute_trade.exception", error=str(e))
            return False, str(e)

    def get_token_raw_balance(self, token_mint: str) -> Optional[int]:
        """
        Get raw token balance in smallest units (not UI amount)
        Returns the actual token amount needed for swap quotes
        """
        try:
            rpc_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTokenAccountsByOwner",
                "params": [
                    self.wallet_address,
                    {
                        "mint": token_mint
                    },
                    {
                        "encoding": "jsonParsed"
                    }
                ]
            }
            
            response = requests.post(self.rpc_url, json=rpc_payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                if "result" in result and "value" in result["result"]:
                    accounts = result["result"]["value"]
                    if accounts:
                        # Get the first account's raw balance (in smallest units)
                        account_info = accounts[0]["account"]["data"]["parsed"]["info"]
                        raw_amount_str = account_info["tokenAmount"]["amount"]  # This is a string of the raw amount
                        return int(raw_amount_str)
                    else:
                        return 0
                else:
                    log_error("raydium.token_balance.rpc_error", error=result.get('error', 'Unknown'))
                    return None
            else:
                log_error("raydium.token_balance.http_error", status=response.status_code)
                return None
        except Exception as e:
            log_error("raydium.token_balance.exception", error=str(e))
            return None

    def execute_raydium_swap(self, input_mint: str, output_mint: str, amount: int, slippage: float = 0.10) -> Tuple[bool, str]:
        """Execute real swap on Raydium using custom library"""
        try:
            if not self.raydium_lib:
                print("❌ No Raydium library available for swap")
                return False, "No Raydium library"
            
            log_info("raydium.swap.start", input=input_mint, output=output_mint, amount=amount)
            
            # Use custom Raydium library to execute the swap
            tx_hash, success = self.raydium_lib.execute_swap(input_mint, output_mint, amount, slippage)
            
            if success:
                log_info("raydium.swap.executed", tx_hash=tx_hash)
                return True, tx_hash
            else:
                log_error("raydium.swap.failed", tx_hash=tx_hash)
                return False, tx_hash
            
        except Exception as e:
            log_error("raydium.swap.exception", error=str(e))
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
                            log_info("raydium.liquidity", token=token_address, liquidity=liquidity)
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
            log_info("raydium.tradeability_error", level="WARNING", error=str(e))
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
