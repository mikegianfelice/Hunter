#!/usr/bin/env python3
"""
Custom Raydium Library - Direct transaction handling for Raydium swaps
"""

import requests
import json
import base64
import struct
import time
from typing import Tuple, Optional, Dict, Any, List
from solders.keypair import Keypair
import base58

class RaydiumCustomLib:
    def __init__(self, rpc_url: str, wallet_address: str, private_key: str):
        self.rpc_url = rpc_url
        self.wallet_address = wallet_address
        self.private_key = private_key
        
        # Initialize wallet
        try:
            if self.private_key:
                secret_key_bytes = base58.b58decode(self.private_key)
                self.keypair = Keypair.from_bytes(secret_key_bytes)
                print(f"✅ Custom Raydium lib initialized with wallet: {self.keypair.pubkey()}...{str(self.keypair.pubkey())[-8:]}")
            else:
                print("⚠️ No Solana private key provided")
                self.keypair = None
        except Exception as e:
            print(f"❌ Failed to initialize wallet: {e}")
            self.keypair = None

    def get_quote(self, input_mint: str, output_mint: str, amount: int, slippage: float = 0.10) -> Dict[str, Any]:
        """Get swap quote from Raydium API with enhanced error handling"""
        try:
            # Try Raydium API v2 for quotes
            url = "https://api.raydium.io/v2/sdk/quote"
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": str(amount),
                "slippage": str(slippage),
                "version": 4
            }
            
            # Try multiple times with different strategies
            for attempt in range(3):
                try:
                    response = requests.get(url, params=params, timeout=15)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data and not data.get("error") and data.get("success") != False:
                            print(f"✅ Raydium quote: {data.get('inAmount', 'N/A')} -> {data.get('outAmount', 'N/A')}")
                            return data
                        else:
                            error_msg = data.get('error', 'Unknown error')
                            print(f"⚠️ Raydium quote failed (attempt {attempt + 1}/3): {error_msg}")
                            
                            # Try with different parameters
                            if attempt == 1:
                                print(f"🔄 Trying with different version...")
                                params["version"] = 3
                                continue
                            elif attempt == 2:
                                print(f"🔄 Trying with smaller amount...")
                                amount = int(amount * 0.5)
                                params["amount"] = str(amount)
                                continue
                            
                            return {}
                    else:
                        print(f"⚠️ Raydium quote failed (attempt {attempt + 1}/3): {response.status_code}")
                        
                        # Handle 500 server errors specifically
                        if response.status_code == 500:
                            print(f"⚠️ Raydium API server error (500) - trying alternative approach...")
                            if attempt < 2:
                                time.sleep(3)  # Wait longer for server errors
                                continue
                            else:
                                # Try alternative endpoint on final attempt
                                print(f"🔄 Trying alternative Raydium endpoint...")
                                return self._try_alternative_raydium_quote(input_mint, output_mint, amount, slippage)
                        
                        # Try with different parameters on 400/404 errors
                        if (response.status_code == 400 or response.status_code == 404) and attempt < 2:
                            if attempt == 0:
                                print(f"🔄 Trying with different version...")
                                params["version"] = 3
                            elif attempt == 1:
                                print(f"🔄 Trying with smaller amount...")
                                amount = int(amount * 0.5)
                                params["amount"] = str(amount)
                            continue
                        
                        # If we can't retry, continue to next attempt
                        continue
                        
                except requests.exceptions.Timeout:
                    print(f"⚠️ Raydium quote timeout (attempt {attempt + 1}/3)")
                    if attempt < 2:
                        time.sleep(2)
                    continue
                except Exception as e:
                    print(f"⚠️ Raydium quote error (attempt {attempt + 1}/3): {e}")
                    if attempt < 2:
                        time.sleep(1)
                    continue
            
            print(f"❌ All Raydium quote attempts failed")
            
            # Fallback: Use DexScreener to get price and calculate quote
            print(f"🔄 Using DexScreener fallback for quote...")
            try:
                url3 = f"https://api.dexscreener.com/latest/dex/tokens/{output_mint}"
                response3 = requests.get(url3, timeout=10)
                
                if response3.status_code == 200:
                    data3 = response3.json()
                    pairs = data3.get("pairs", [])
                    
                    print(f"Found {len(pairs)} pairs in DexScreener")
                    
                    # Find Raydium pair
                    for pair in pairs:
                        dex_id = pair.get("dexId", "").lower()
                        if "raydium" in dex_id:
                            price_usd = float(pair.get("priceUsd", 0))
                            if price_usd > 0:
                                # Calculate quote based on price
                                input_usd = amount / 1000000  # Convert from USDC decimals
                                output_tokens = input_usd / price_usd
                                
                                print(f"✅ DexScreener fallback quote: {amount} -> {int(output_tokens * 1000000000)}")
                                print(f"   Price: ${price_usd}")
                                print(f"   Rate: 1 USDC = {output_tokens * 1000000000:,.0f} tokens")
                                
                                return {
                                    "success": True,
                                    "inAmount": str(amount),
                                    "outAmount": str(int(output_tokens * 1000000000)),  # Convert to token decimals
                                    "priceImpact": 0.1,  # Estimated
                                    "route": "raydium-dexscreener-fallback"
                                }
                    
                    print(f"⚠️ No Raydium pairs found in DexScreener data")
                else:
                    print(f"❌ DexScreener API failed: {response3.status_code}")
            except Exception as e:
                print(f"⚠️ DexScreener fallback failed: {e}")
            
            return {}
            
        except Exception as e:
            print(f"❌ Raydium quote error: {e}")
            return {}

    def _try_alternative_raydium_quote(self, input_mint: str, output_mint: str, amount: int, slippage: float) -> Dict[str, Any]:
        """Try alternative Raydium endpoints when main quote API fails"""
        try:
            # Try Raydium main quote endpoint as alternative
            url = "https://api.raydium.io/v2/main/quote"
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": str(amount),
                "slippage": str(slippage),
                "version": "4"
            }
            
            print(f"🔄 Trying Raydium main quote endpoint...")
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data and not data.get("error") and data.get("success") != False:
                    print(f"✅ Raydium alternative quote: {data.get('inAmount', 'N/A')} -> {data.get('outAmount', 'N/A')}")
                    return data
                else:
                    print(f"⚠️ Raydium alternative quote failed: {data.get('error', 'Unknown error')}")
            elif response.status_code == 500:
                print(f"⚠️ Raydium main quote also returning 500 server error")
            else:
                print(f"⚠️ Raydium main quote returned {response.status_code}")
            
            # If alternative also fails, try DexScreener fallback
            return self._try_dexscreener_fallback(output_mint, amount)
            
        except Exception as e:
            print(f"❌ Raydium alternative quote error: {e}")
            return self._try_dexscreener_fallback(output_mint, amount)

    def _try_dexscreener_fallback(self, output_mint: str, amount: int) -> Dict[str, Any]:
        """Try DexScreener as final fallback for quote calculation"""
        try:
            print(f"🔄 Using DexScreener fallback for quote...")
            url = f"https://api.dexscreener.com/latest/dex/tokens/{output_mint}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                pairs = data.get("pairs", [])
                
                # Find Raydium pair
                for pair in pairs:
                    dex_id = pair.get("dexId", "").lower()
                    if "raydium" in dex_id:
                        price_usd = float(pair.get("priceUsd", 0))
                        if price_usd > 0:
                            # Calculate quote based on price
                            input_usd = amount / 1000000  # Convert from USDC decimals
                            output_tokens = input_usd / price_usd
                            
                            print(f"✅ DexScreener fallback quote: {amount} -> {int(output_tokens * 1000000000)}")
                            print(f"   Price: ${price_usd}")
                            
                            return {
                                "success": True,
                                "inAmount": str(amount),
                                "outAmount": str(int(output_tokens * 1000000000)),  # Convert to token decimals
                                "priceImpact": 0.1,  # Estimated
                                "route": "raydium-dexscreener-fallback"
                            }
                
                print(f"⚠️ No Raydium pairs found in DexScreener data")
            else:
                print(f"❌ DexScreener API failed: {response.status_code}")
        except Exception as e:
            print(f"⚠️ DexScreener fallback failed: {e}")
        
        return {}

    def get_swap_transaction(self, quote_response: Dict[str, Any]) -> str:
        """Get swap transaction from Raydium API with enhanced error handling"""
        try:
            url = "https://api.raydium.io/v2/sdk/swap"
            payload = {
                "quoteResponse": quote_response,
                "userPublicKey": self.wallet_address,
                "wrapUnwrapSOL": True,
                "computeUnitPriceMicroLamports": 1000,
                "asLegacyTransaction": True,
                "useSharedAccounts": False,
                "maxAccounts": 16
            }
            
            # Try multiple times with different strategies
            for attempt in range(3):
                try:
                    response = requests.post(url, json=payload, timeout=15)
                    
                    if response.status_code == 200:
                        swap_data = response.json()
                        if "swapTransaction" in swap_data:
                            print(f"✅ Raydium swap transaction generated successfully")
                            return swap_data["swapTransaction"]
                        else:
                            error_msg = swap_data.get("error", "No swap transaction in response")
                            print(f"⚠️ Raydium swap failed (attempt {attempt + 1}/3): {error_msg}")
                            
                            # Try with different parameters
                            if attempt == 1:
                                print(f"🔄 Trying with minimal accounts...")
                                payload["maxAccounts"] = 8
                                continue
                            elif attempt == 2:
                                print(f"🔄 Trying with compute budget disabled...")
                                payload["computeUnitPriceMicroLamports"] = 0
                                continue
                            
                            return ""
                    else:
                        print(f"⚠️ Raydium swap request failed (attempt {attempt + 1}/3): {response.status_code}")
                        
                        # Try with different parameters on 400 errors
                        if response.status_code == 400 and attempt < 2:
                            if attempt == 0:
                                print(f"🔄 Trying with minimal accounts...")
                                payload["maxAccounts"] = 8
                            elif attempt == 1:
                                print(f"🔄 Trying with compute budget disabled...")
                                payload["computeUnitPriceMicroLamports"] = 0
                            continue
                        
                        return ""
                        
                except requests.exceptions.Timeout:
                    print(f"⚠️ Raydium swap timeout (attempt {attempt + 1}/3)")
                    if attempt < 2:
                        time.sleep(2)
                    continue
                except Exception as e:
                    print(f"⚠️ Raydium swap error (attempt {attempt + 1}/3): {e}")
                    if attempt < 2:
                        time.sleep(1)
                    continue
            
            print(f"❌ All Raydium swap attempts failed")
            return ""
            
        except Exception as e:
            print(f"❌ Raydium swap error: {e}")
            return ""

    def decode_transaction(self, transaction_data: str) -> Dict[str, Any]:
        """Decode base64 transaction data"""
        try:
            decoded_bytes = base64.b64decode(transaction_data)
            
            # Parse transaction structure manually
            # Solana transaction format: [signatures][message]
            
            # Read number of signatures (1 byte)
            num_signatures = decoded_bytes[0]
            
            # Each signature is 64 bytes
            signature_length = 64
            signatures_end = 1 + (num_signatures * signature_length)
            
            # Extract signatures
            signatures = []
            for i in range(num_signatures):
                start = 1 + (i * signature_length)
                end = start + signature_length
                signature = decoded_bytes[start:end]
                signatures.append(base58.b58encode(signature).decode('utf-8'))
            
            # Extract message (rest of the data)
            message_bytes = decoded_bytes[signatures_end:]
            
            return {
                "num_signatures": num_signatures,
                "signatures": signatures,
                "message": base58.b58encode(message_bytes).decode('utf-8'),
                "message_bytes": message_bytes
            }
        except Exception as e:
            print(f"❌ Transaction decode error: {e}")
            return {}

    def sign_transaction(self, transaction_data: str) -> str:
        """Sign transaction with wallet"""
        try:
            if not self.keypair:
                print("❌ No keypair available for signing")
                return ""
            
            # Decode transaction
            decoded = self.decode_transaction(transaction_data)
            if not decoded:
                return ""
            
            # Get message bytes
            message_bytes = decoded["message_bytes"]
            
            # Sign the message
            signature = self.keypair.sign_message(message_bytes)
            signature_bytes = bytes(signature)
            
            # Reconstruct transaction with signature
            # Format: [num_signatures][signature][message]
            num_signatures = 1
            reconstructed = struct.pack('B', num_signatures)  # 1 byte for num signatures
            reconstructed += signature_bytes  # 64 bytes for signature
            reconstructed += message_bytes  # message bytes
            
            # Encode back to base64
            signed_transaction = base64.b64encode(reconstructed).decode('utf-8')
            
            print(f"✅ Raydium transaction signed successfully")
            return signed_transaction
            
        except Exception as e:
            print(f"❌ Raydium transaction signing error: {e}")
            return ""

    def send_transaction(self, signed_transaction: str) -> Tuple[str, bool]:
        """Send signed transaction to Solana network"""
        try:
            # Check transaction size before sending
            decoded = base64.b64decode(signed_transaction)
            tx_size = len(decoded)
            max_size = 1644  # Maximum transaction size
            
            if tx_size > max_size:
                print(f"❌ Transaction too large: {tx_size} bytes (max: {max_size})")
                return f"Transaction too large: {tx_size} bytes", False
            
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
                    print(f"❌ Broken pipe error sending transaction: Connection closed unexpectedly")
                else:
                    print(f"❌ Connection error sending transaction: {e}")
                return "", False
            except OSError as e:
                if e.errno == 32:  # Broken pipe
                    print(f"❌ Broken pipe error (errno 32) sending transaction: {e}")
                else:
                    print(f"❌ OS error sending transaction: {e}")
                return "", False
            
            if response.status_code == 200:
                result = response.json()
                if "result" in result:
                    tx_hash = result["result"]
                    print(f"✅ Raydium transaction sent: {tx_hash}")
                    return tx_hash, True
                elif "error" in result:
                    error_msg = result["error"]
                    print(f"❌ RPC error: {error_msg}")
                    return str(error_msg), False
            else:
                print(f"❌ RPC request failed: {response.status_code}")
                return "", False
                
        except Exception as e:
            print(f"❌ Send Raydium transaction error: {e}")
            return "", False

    def execute_swap(self, input_mint: str, output_mint: str, amount: int, slippage: float = 0.10) -> Tuple[str, bool]:
        """Execute complete Raydium swap process with enhanced error handling"""
        try:
            print(f"🔄 Executing Raydium swap: {input_mint[:8]}... -> {output_mint[:8]}...")
            
            # Step 1: Get quote
            quote = self.get_quote(input_mint, output_mint, amount, slippage)
            if not quote:
                print(f"❌ Failed to get Raydium quote for swap")
                return "", False
            
            # Step 2: Get swap transaction
            transaction_data = self.get_swap_transaction(quote)
            if not transaction_data:
                print(f"❌ Failed to get Raydium swap transaction")
                return "", False
            
            # Step 3: Sign transaction
            signed_transaction = self.sign_transaction(transaction_data)
            if not signed_transaction:
                print(f"❌ Failed to sign Raydium transaction")
                return "", False
            
            # Step 4: Send transaction
            tx_hash, success = self.send_transaction(signed_transaction)
            if success:
                return tx_hash, True
            
            # Step 5: If failed due to size, try with smaller amount
            print(f"🔄 Raydium transaction failed, trying with smaller amount...")
            smaller_amount = int(amount * 0.5)  # Try with 50% of original amount
            
            quote = self.get_quote(input_mint, output_mint, smaller_amount, slippage)
            if not quote:
                return "", False
            
            transaction_data = self.get_swap_transaction(quote)
            if not transaction_data:
                return "", False
            
            signed_transaction = self.sign_transaction(transaction_data)
            if not signed_transaction:
                return "", False
            
            tx_hash, success = self.send_transaction(signed_transaction)
            if success:
                return tx_hash, True
            
            # Step 6: If still failed, try with even smaller amount
            print(f"🔄 Still failed, trying with minimal amount...")
            minimal_amount = int(amount * 0.25)  # Try with 25% of original amount
            
            quote = self.get_quote(input_mint, output_mint, minimal_amount, slippage)
            if not quote:
                return "", False
            
            transaction_data = self.get_swap_transaction(quote)
            if not transaction_data:
                return "", False
            
            signed_transaction = self.sign_transaction(transaction_data)
            if not signed_transaction:
                return "", False
            
            tx_hash, success = self.send_transaction(signed_transaction)
            if success:
                return tx_hash, True
            
            # Step 7: Final attempt with tiny amount
            print(f"🔄 Final attempt with tiny amount...")
            tiny_amount = int(amount * 0.1)  # Try with 10% of original amount
            
            quote = self.get_quote(input_mint, output_mint, tiny_amount, slippage)
            if not quote:
                return "", False
            
            transaction_data = self.get_swap_transaction(quote)
            if not transaction_data:
                return "", False
            
            signed_transaction = self.sign_transaction(transaction_data)
            if not signed_transaction:
                return "", False
            
            tx_hash, success = self.send_transaction(signed_transaction)
            return tx_hash, success
            
        except Exception as e:
            print(f"❌ Raydium swap execution error: {e}")
            return "", False

    def get_balance(self) -> float:
        """Get SOL balance"""
        try:
            rpc_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getBalance",
                "params": [str(self.keypair.pubkey())]
            }
            
            response = requests.post(self.rpc_url, json=rpc_payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                if "result" in result and "value" in result["result"]:
                    balance_lamports = result["result"]["value"]
                    return float(balance_lamports) / 1_000_000_000
            return 0.0
        except Exception as e:
            print(f"❌ Error getting balance: {e}")
            return 0.0
