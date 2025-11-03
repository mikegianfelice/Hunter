#!/usr/bin/env python3
"""
Custom Jupiter Library - Direct transaction handling for Jupiter v6
"""

import os
import requests
import json
import base64
import struct
import time
from typing import Tuple, Optional, Dict, Any, List
from solders.keypair import Keypair
import base58
from ..utils.http_utils import get_json, post_json

# Jupiter API Configuration
# Default to free public API: https://lite-api.jup.ag (Ultra API - no API key required)
# For paid access, set JUPITER_API_BASE=https://api.jup.ag and JUPITER_API_KEY=your_key
# Ultra API documentation: https://dev.jup.ag/api-reference/ultra/order
JUPITER_API_BASE = os.getenv("JUPITER_API_BASE", "https://lite-api.jup.ag").rstrip("/")
JUPITER_API_KEY = (os.getenv("JUPITER_API_KEY") or "").strip()
JUPITER_HEADERS = {"X-API-KEY": JUPITER_API_KEY} if JUPITER_API_KEY else None

class JupiterCustomLib:
    def __init__(self, rpc_url: str, wallet_address: str, private_key: str):
        self.rpc_url = rpc_url
        self.wallet_address = wallet_address
        self.private_key = private_key
        
        # Initialize wallet
        try:
            if self.private_key:
                secret_key_bytes = base58.b58decode(self.private_key)
                self.keypair = Keypair.from_bytes(secret_key_bytes)
                print(f"✅ Custom Jupiter lib initialized with wallet: {self.keypair.pubkey()}...{str(self.keypair.pubkey())[-8:]}")
            else:
                print("⚠️ No Solana private key provided")
                self.keypair = None
        except Exception as e:
            print(f"❌ Failed to initialize wallet: {e}")
            self.keypair = None

    def get_quote(self, input_mint: str, output_mint: str, amount: int, slippage: float = 0.15, 
                  route_preferences: Dict[str, Any] = None, use_exactout: bool = False) -> Dict[str, Any]:
        """Get swap quote from Jupiter Ultra API with enhanced error handling
        
        Uses Jupiter Ultra API (https://dev.jup.ag/api-reference/ultra/order)
        Default endpoint: https://lite-api.jup.ag (free tier, no API key required)
        
        Configuration:
        - Set JUPITER_API_BASE env var to change endpoint (default: https://lite-api.jup.ag)
        - Set JUPITER_API_KEY env var for paid API access (not required for free tier)
        
        If a 401 or 429 error is returned, the function will return empty dict to trigger
        Raydium fallback.
        """
        try:
            # Use Jupiter Ultra API endpoint
            base_url = f"{JUPITER_API_BASE}/ultra/v1/order"
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": str(amount),
                "slippageBps": int(slippage * 10000),
                "taker": self.wallet_address  # Include taker to get transaction in response
            }
            
            # Apply route preferences (if needed)
            if route_preferences:
                if route_preferences.get('excludeDexes'):
                    params["excludeDexes"] = route_preferences['excludeDexes']
                if route_preferences.get('excludeRouters'):
                    params["excludeRouters"] = route_preferences['excludeRouters']
            
            # Use ExactOut for sketchy tokens
            if use_exactout:
                params["swapMode"] = "ExactOut"
                print(f"🔄 Using ExactOut mode for sketchy token")
            
            # Build URL with params
            url = f"{base_url}?{'&'.join(f'{k}={v}' for k, v in params.items())}"
            
            # Try multiple times with different strategies
            for attempt in range(3):
                try:
                    # Use http_utils with retry logic for better network error handling
                    # Include API key header if configured
                    data = get_json(url, headers=JUPITER_HEADERS, timeout=15, retries=2, backoff=1.0)
                    
                    # Ultra API response structure
                    if data and data.get("inAmount") and data.get("outAmount"):
                        print(f"✅ Jupiter Ultra quote: {data.get('inAmount', 'N/A')} -> {data.get('outAmount', 'N/A')}")
                        # Store requestId for execution (Ultra API requirement)
                        if data.get("requestId"):
                            data["requestId"] = data["requestId"]
                        # Transaction may already be in response if taker is provided
                        if data.get("transaction"):
                            data["transaction"] = data["transaction"]
                        # Add success field for compatibility
                        data["success"] = True
                        return data
                    else:
                        error_msg = data.get('errorMessage', data.get('error', 'Unknown error'))
                        print(f"⚠️ Jupiter quote failed (attempt {attempt + 1}/3): {error_msg}")
                        
                        # If it's a liquidity issue, try with smaller amount
                        if "insufficient" in str(error_msg).lower() or "liquidity" in str(error_msg).lower():
                            print(f"🔄 Trying with smaller amount due to liquidity issues...")
                            amount = int(amount * 0.5)  # Try with 50% of original amount
                            params["amount"] = str(amount)
                            url = f"{base_url}?{'&'.join(f'{k}={v}' for k, v in params.items())}"
                            continue
                        
                        return {}
                        
                except requests.exceptions.HTTPError as e:
                    status_code = e.response.status_code if hasattr(e, 'response') else None
                    print(f"⚠️ Jupiter quote HTTP error (attempt {attempt + 1}/3): {status_code}")
                    
                    # If it's a 401 (Unauthorized) or 429 (Rate Limited), fail fast to fallback to Raydium
                    if status_code in (401, 429):
                        if status_code == 401:
                            print(f"⚠️ Jupiter API returned 401 Unauthorized - API key may be required")
                        else:
                            print(f"⚠️ Jupiter API returned 429 Rate Limited - too many requests")
                        print(f"🔄 Will fallback to Raydium executor...")
                        return {}  # Return early to trigger Raydium fallback
                    
                    # If it's a 400 error, try with different parameters
                    if status_code == 400:
                        print(f"🔄 Trying alternative quote method for 400 error...")
                        # Try with legacy transaction format
                        params["asLegacyTransaction"] = "true"
                        url = f"{base_url}?{'&'.join(f'{k}={v}' for k, v in params.items())}"
                        continue
                    
                    if attempt < 2:
                        time.sleep(2)
                        continue
                    return {}
                    
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                    print(f"⚠️ Network error getting Jupiter quote (attempt {attempt + 1}/3): {type(e).__name__}")
                    if attempt < 2:
                        time.sleep(2)
                    continue
                    
                except Exception as e:
                    print(f"⚠️ Jupiter quote error (attempt {attempt + 1}/3): {e}")
                    if attempt < 2:
                        time.sleep(1)
                    continue
            
            print(f"❌ All Jupiter quote attempts failed")
            return {}
            
        except Exception as e:
            print(f"❌ Jupiter quote error: {e}")
            return {}

    def get_swap_transaction(self, quote_response: Dict[str, Any]) -> str:
        """Get swap transaction from Jupiter Ultra API with enhanced error handling
        
        Uses Jupiter Ultra API execute endpoint (https://dev.jup.ag/api-reference/ultra/execute)
        Default endpoint: https://lite-api.jup.ag (free tier, no API key required)
        
        Configuration:
        - Set JUPITER_API_BASE env var to change endpoint (default: https://lite-api.jup.ag)
        - Set JUPITER_API_KEY env var for paid API access (not required for free tier)
        
        Note: If quote_response already contains a transaction (from get_quote with taker),
        this function will return it directly. Otherwise, it will call the execute endpoint.
        """
        try:
            # If transaction is already in quote response (from Ultra API with taker), use it
            if quote_response.get("transaction"):
                transaction = quote_response["transaction"]
                if transaction and transaction.strip():
                    print(f"✅ Swap transaction from quote response")
                    return transaction
            
            # Otherwise, use execute endpoint with requestId
            request_id = quote_response.get("requestId")
            if not request_id:
                print(f"⚠️ No requestId in quote response, cannot execute swap")
                return ""
            
            url = f"{JUPITER_API_BASE}/ultra/v1/execute"
            payload = {
                "requestId": request_id
            }
            
            # Try multiple times with different strategies
            for attempt in range(3):
                try:
                    # Use post_json with retry logic for better network error handling
                    # Include API key header if configured
                    swap_data = post_json(url, payload, headers=JUPITER_HEADERS, timeout=15, retries=2, backoff=1.0)
                    
                    if "transaction" in swap_data and swap_data["transaction"]:
                        print(f"✅ Swap transaction generated successfully via execute endpoint")
                        return swap_data["transaction"]
                    else:
                        error_msg = swap_data.get("errorMessage", swap_data.get("error", "No transaction in response"))
                        print(f"⚠️ Jupiter swap failed (attempt {attempt + 1}/3): {error_msg}")
                        
                        # Check for specific error codes
                        error_code = swap_data.get("errorCode")
                        if error_code:
                            print(f"   Error code: {error_code}")
                        
                        if attempt < 2:
                            time.sleep(2)
                            continue
                        
                        return ""
                        
                except requests.exceptions.HTTPError as e:
                    status_code = e.response.status_code if hasattr(e, 'response') else None
                    print(f"⚠️ Jupiter swap HTTP error (attempt {attempt + 1}/3): {status_code}")
                    
                    # If it's a 401 (Unauthorized) or 429 (Rate Limited), fail fast to fallback to Raydium
                    if status_code in (401, 429):
                        if status_code == 401:
                            print(f"⚠️ Jupiter API returned 401 Unauthorized - API key may be required")
                        else:
                            print(f"⚠️ Jupiter API returned 429 Rate Limited - too many requests")
                        print(f"🔄 Will fallback to Raydium executor...")
                        return ""  # Return early to trigger Raydium fallback
                    
                    # Try with different parameters on 400 errors
                    if status_code == 400 and attempt < 2:
                        if attempt == 0:
                            print(f"🔄 Trying with minimal accounts...")
                            payload["maxAccounts"] = 8
                        elif attempt == 1:
                            print(f"🔄 Trying with compute budget disabled...")
                            payload["computeUnitPriceMicroLamports"] = 0
                        continue
                    
                    if attempt < 2:
                        time.sleep(2)
                        continue
                    return ""
                    
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                    print(f"⚠️ Network error getting swap transaction (attempt {attempt + 1}/3): {type(e).__name__}")
                    if attempt < 2:
                        time.sleep(2)
                    continue
                    
                except Exception as e:
                    print(f"⚠️ Jupiter swap error (attempt {attempt + 1}/3): {e}")
                    if attempt < 2:
                        time.sleep(1)
                    continue
            
            print(f"❌ All Jupiter swap attempts failed")
            return ""
            
        except Exception as e:
            print(f"❌ Jupiter swap error: {e}")
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
            
            print(f"✅ Transaction signed successfully")
            return signed_transaction
            
        except Exception as e:
            print(f"❌ Transaction signing error: {e}")
            return ""

    def send_transaction(self, signed_transaction: str) -> Tuple[str, bool]:
        """Send signed transaction to Solana network"""
        try:
            # Check transaction size before sending
            import base64
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
                    print(f"❌ Broken pipe error sending Jupiter transaction: Connection closed unexpectedly")
                else:
                    print(f"❌ Connection error sending Jupiter transaction: {e}")
                return "", False
            except OSError as e:
                if e.errno == 32:  # Broken pipe
                    print(f"❌ Broken pipe error (errno 32) sending Jupiter transaction: {e}")
                else:
                    print(f"❌ OS error sending Jupiter transaction: {e}")
                return "", False
            
            if response.status_code == 200:
                result = response.json()
                if "result" in result:
                    tx_hash = result["result"]
                    try:
                        print(f"✅ Transaction sent: {tx_hash}")
                    except BrokenPipeError:
                        pass
                    return tx_hash, True
                elif "error" in result:
                    error_msg = result["error"]
                    try:
                        print(f"❌ RPC error: {error_msg}")
                    except BrokenPipeError:
                        pass
                    return str(error_msg), False
            else:
                try:
                    print(f"❌ RPC request failed: {response.status_code}")
                except BrokenPipeError:
                    pass
                return "", False
                
        except Exception as e:
            try:
                print(f"❌ Send transaction error: {e}")
            except BrokenPipeError:
                pass
            return "", False

    def execute_swap(self, input_mint: str, output_mint: str, amount: int, slippage: float = 0.10) -> Tuple[str, bool]:
        """Execute complete swap process with enhanced error handling"""
        try:
            try:
                print(f"🔄 Executing swap: {input_mint[:8]}... -> {output_mint[:8]}...")
            except BrokenPipeError:
                pass
            
            # Step 1: Get quote
            quote = self.get_quote(input_mint, output_mint, amount, slippage)
            if not quote:
                try:
                    print(f"❌ Failed to get quote for swap")
                except BrokenPipeError:
                    pass
                return "", False
            
            # Step 2: Get swap transaction
            transaction_data = self.get_swap_transaction(quote)
            if not transaction_data:
                try:
                    print(f"❌ Failed to get swap transaction")
                except BrokenPipeError:
                    pass
                return "", False
            
            # Step 3: Sign transaction
            signed_transaction = self.sign_transaction(transaction_data)
            if not signed_transaction:
                try:
                    print(f"❌ Failed to sign transaction")
                except BrokenPipeError:
                    pass
                return "", False
            
            # Step 4: Send transaction
            tx_hash, success = self.send_transaction(signed_transaction)
            if success:
                return tx_hash, True
            
            # Step 5: If failed due to size, try with smaller amount
            print(f"🔄 Transaction failed, trying with smaller amount...")
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
            print(f"❌ Swap execution error: {e}")
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

    def get_sol_balance(self) -> float:
        """Get SOL balance (alias for get_balance)"""
        return self.get_balance()

    def get_token_balance(self, token_mint: str):
        """
        Get token balance for a specific mint
        
        Returns:
            float: Token balance (> 0 if position exists)
            None: If RPC call failed or error occurred (unknown state - don't remove position)
        """
        try:
            # Get token accounts for the wallet
            rpc_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTokenAccountsByOwner",
                "params": [
                    str(self.keypair.pubkey()),
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
                        # Get the first account's balance
                        account_info = accounts[0]["account"]["data"]["parsed"]["info"]
                        balance = float(account_info["tokenAmount"]["uiAmount"])
                        return balance
                    else:
                        # No token accounts found - this could mean:
                        # 1. Trade never executed (no account created)
                        # 2. Position was sold (account closed)
                        # Return 0.0 to indicate no balance (but only if we're sure)
                        return 0.0
                else:
                    # RPC response error - don't trust this, return None
                    print(f"⚠️ RPC response error for token balance check: {result.get('error', 'Unknown')}")
                    return None
            else:
                # HTTP error - don't trust this, return None
                print(f"⚠️ HTTP error {response.status_code} getting token balance")
                return None
        except Exception as e:
            print(f"❌ Error getting token balance: {e}")
            # Return None to indicate check failed (not zero balance)
            return None

    def swap_tokens(self, input_mint: str, output_mint: str, amount_in: float, slippage_bps: int) -> Dict[str, Any]:
        """Execute token swap and return result dict"""
        try:
            # Convert amount to lamports (for SOL) or token units
            if input_mint == "So11111111111111111111111111111111111111112":  # WSOL
                amount_lamports = int(amount_in * 1_000_000_000)
            else:
                # For tokens, assume 6 decimals (like USDC)
                amount_lamports = int(amount_in * 1_000_000)
            
            tx_hash, success = self.execute_swap(input_mint, output_mint, amount_lamports, slippage_bps / 10000)
            
            return {
                "success": success,
                "tx_hash": tx_hash if success else None,
                "amount_in": amount_in,
                "input_mint": input_mint,
                "output_mint": output_mint
            }
        except Exception as e:
            print(f"❌ Swap tokens error: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# Utility functions
def create_jupiter_lib(rpc_url: str, wallet_address: str, private_key: str) -> JupiterCustomLib:
    """Create Jupiter custom library instance"""
    return JupiterCustomLib(rpc_url, wallet_address, private_key)
