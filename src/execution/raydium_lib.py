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

    def _get_priority_fee(self, priority_level: str = "h") -> int:
        """Get priority fee from Raydium API
        
        Args:
            priority_level: "vh" (very high), "h" (high), or "m" (medium)
        
        Returns:
            Priority fee in microLamports, or default 1000 if API fails
        """
        try:
            url = "https://api.raydium.io/priority-fee"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("success") and data.get("data", {}).get("default", {}).get(priority_level):
                    fee = int(data["data"]["default"][priority_level])
                    print(f"✅ Raydium priority fee ({priority_level}): {fee} microLamports")
                    return fee
        except Exception as e:
            print(f"⚠️ Failed to get priority fee: {e}")
        # Return default if API fails
        return 1000

    def _get_token_account(self, token_mint: str, wallet_address: str) -> Optional[str]:
        """Get token account address for a given mint and wallet
        
        If token account doesn't exist, returns None (ATA will be created automatically)
        
        Args:
            token_mint: Token mint address
            wallet_address: Wallet address
            
        Returns:
            Token account address or None if account doesn't exist
        """
        try:
            # Check if token account exists via RPC
            rpc_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTokenAccountsByOwner",
                "params": [
                    wallet_address,
                    {"mint": token_mint},
                    {"encoding": "jsonParsed"}
                ]
            }
            
            response = requests.post(self.rpc_url, json=rpc_payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                if "result" in result and result["result"].get("value"):
                    accounts = result["result"]["value"]
                    if accounts and len(accounts) > 0:
                        # Extract account address from response
                        account_info = accounts[0]
                        if isinstance(account_info, dict):
                            # Response format: {"pubkey": "...", "account": {...}}
                            account_address = account_info.get("pubkey") or account_info.get("account", {}).get("data", {}).get("parsed", {}).get("info", {}).get("mint")
                            if account_address:
                                return account_address
            # Account doesn't exist - will be created automatically (return None)
            return None
        except Exception as e:
            print(f"⚠️ Error getting token account: {e}")
            return None

    def get_quote(self, input_mint: str, output_mint: str, amount: int, slippage: float = 0.10) -> Dict[str, Any]:
        """Get swap quote from Raydium Trade API (official endpoint)
        
        Reference: https://docs.raydium.io/raydium/traders/trade-api
        
        Uses official endpoint: https://transaction-v1.raydium.io/compute/swap-base-in
        """
        try:
            # Use official Raydium Trade API endpoint
            base_url = "https://transaction-v1.raydium.io/compute/swap-base-in"
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": str(amount),
                "slippageBps": str(int(slippage * 10000)),  # Convert to basis points (0.01%)
                "txVersion": "LEGACY"  # Use LEGACY for compatibility, or "V0" for versioned
            }
            
            # Try multiple times with different strategies
            for attempt in range(3):
                try:
                    response = requests.get(base_url, params=params, timeout=15)
                    
                    if response.status_code == 200:
                        data = response.json()
                        # Official API response structure: {success: bool, data: {...}}
                        if data and data.get("success") and data.get("data"):
                            swap_response = data["data"]
                            print(f"✅ Raydium quote: {swap_response.get('inAmount', 'N/A')} -> {swap_response.get('outAmount', 'N/A')}")
                            # Return full response structure as swapResponse for transaction endpoint
                            return {
                                "success": True,
                                "swapResponse": data,  # Full response for transaction endpoint
                                "data": swap_response
                            }
                        else:
                            error_msg = data.get('error', 'Unknown error')
                            print(f"⚠️ Raydium quote failed (attempt {attempt + 1}/3): {error_msg}")
                            
                            # Try with smaller amount on retry
                            if attempt < 2:
                                print(f"🔄 Trying with smaller amount...")
                                amount = int(amount * 0.5)
                                params["amount"] = str(amount)
                                continue
                            
                            return {}
                    else:
                        print(f"⚠️ Raydium quote failed (attempt {attempt + 1}/3): {response.status_code}")
                        
                        # Handle errors
                        if response.status_code == 500:
                            print(f"⚠️ Raydium API server error (500) - retrying...")
                            if attempt < 2:
                                time.sleep(3)
                                continue
                        
                        if (response.status_code == 400 or response.status_code == 404) and attempt < 2:
                            print(f"🔄 Trying with smaller amount...")
                            amount = int(amount * 0.5)
                            params["amount"] = str(amount)
                            continue
                        
                        if attempt < 2:
                            time.sleep(2)
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
            return {}
            
        except Exception as e:
            print(f"❌ Raydium quote error: {e}")
            return {}

    def get_swap_transaction(self, quote_response: Dict[str, Any], input_mint: str = None, output_mint: str = None) -> str:
        """Get swap transaction from Raydium Trade API (official endpoint)
        
        Reference: https://docs.raydium.io/raydium/traders/trade-api
        
        Uses official endpoint: https://transaction-v1.raydium.io/transaction/swap-base-in
        """
        try:
            # Extract swapResponse from quote - must be the full response from quote endpoint
            swap_response = quote_response.get("swapResponse")
            if not swap_response:
                print(f"❌ Quote response missing swapResponse field")
                return ""
            
            # Determine if we need to wrap/unwrap SOL
            WSOL_MINT = "So11111111111111111111111111111111111111112"
            wrap_sol = input_mint == WSOL_MINT if input_mint else False
            unwrap_sol = output_mint == WSOL_MINT if output_mint else False
            
            # Get token accounts (optional - will auto-create if None)
            input_account = None
            output_account = None
            
            if input_mint and input_mint != WSOL_MINT:
                input_account = self._get_token_account(input_mint, self.wallet_address)
            
            if output_mint and output_mint != WSOL_MINT:
                output_account = self._get_token_account(output_mint, self.wallet_address)
            
            # Get priority fee
            priority_fee = self._get_priority_fee("h")  # Use "high" priority
            
            # Use official Raydium Trade API transaction endpoint
            url = "https://transaction-v1.raydium.io/transaction/swap-base-in"
            payload = {
                "swapResponse": swap_response,  # Full response from quote endpoint
                "txVersion": "LEGACY",  # or "V0" for versioned transactions
                "wallet": self.wallet_address,
                "wrapSol": wrap_sol,
                "unwrapSol": unwrap_sol,
                "computeUnitPriceMicroLamports": str(priority_fee)
            }
            
            # Add token accounts if provided (None means ATA will be created automatically)
            if input_account:
                payload["inputAccount"] = input_account
            if output_account:
                payload["outputAccount"] = output_account
            
            # Try multiple times with different strategies
            for attempt in range(3):
                try:
                    response = requests.post(url, json=payload, timeout=15)
                    
                    if response.status_code == 200:
                        swap_data = response.json()
                        # Official API response: {success: bool, data: [{transaction: string}]}
                        if swap_data.get("success") and swap_data.get("data"):
                            transactions = swap_data["data"]
                            if transactions and len(transactions) > 0:
                                # Return first transaction (Raydium may return multiple)
                                transaction = transactions[0].get("transaction")
                                if transaction:
                                    print(f"✅ Raydium swap transaction generated successfully ({len(transactions)} tx(s))")
                                    return transaction
                            
                        error_msg = swap_data.get("error", "No transaction in response")
                        print(f"⚠️ Raydium swap failed (attempt {attempt + 1}/3): {error_msg}")
                        
                        # Try with lower priority fee on retry
                        if attempt < 2:
                            print(f"🔄 Trying with lower priority fee...")
                            payload["computeUnitPriceMicroLamports"] = str(priority_fee // 2)
                            continue
                        
                        return ""
                    else:
                        print(f"⚠️ Raydium swap request failed (attempt {attempt + 1}/3): {response.status_code}")
                        
                        # Try with lower priority fee on 400 errors
                        if response.status_code == 400 and attempt < 2:
                            print(f"🔄 Trying with lower priority fee...")
                            payload["computeUnitPriceMicroLamports"] = str(priority_fee // 2)
                            continue
                        
                        if attempt < 2:
                            time.sleep(2)
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
            if not quote or not quote.get("swapResponse"):
                print(f"❌ Failed to get Raydium quote for swap")
                return "", False
            
            # Step 2: Get swap transaction (pass mints for SOL wrapping logic)
            transaction_data = self.get_swap_transaction(quote, input_mint, output_mint)
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
