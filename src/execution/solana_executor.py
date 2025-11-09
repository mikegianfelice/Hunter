import json
import time
import requests
import base58
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from solana.rpc.api import Client
from solders.transaction import Transaction
from solders.keypair import Keypair
from solders.pubkey import Pubkey as PublicKey
from solders.system_program import TransferParams, transfer
from solana.rpc.commitment import Commitment
import struct

from src.config.secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY
from src.execution.jupiter_lib import JUPITER_API_BASE, JUPITER_HEADERS

# Common token addresses
WSOL_MINT = "So11111111111111111111111111111111111111112"  # Wrapped SOL
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # USDC

class SimpleSolanaExecutor:
    def __init__(self):
        self.client = Client(SOLANA_RPC_URL)
        self.wallet_address = SOLANA_WALLET_ADDRESS
        self.private_key = SOLANA_PRIVATE_KEY
        
        # Initialize wallet
        try:
            if self.private_key:
                self.keypair = Keypair.from_bytes(base58.b58decode(self.private_key))
                print(f"✅ Solana wallet initialized: {self.keypair.pubkey()}...{str(self.keypair.pubkey())[-8:]}")
            else:
                print("⚠️ No Solana private key provided")
                self.keypair = None
        except Exception as e:
            print(f"❌ Failed to initialize Solana wallet: {e}")
            self.keypair = None

    def get_token_price_usd(self, token_address: str) -> float:
        """Get token price in USD using multiple sources with retry logic"""
        import time
        
        # Import here to avoid circular imports
        from src.utils.utils import get_sol_price_usd
        
        # If the token is SOL, use the utility function
        sol_mint = "So11111111111111111111111111111111111111112"
        if token_address == sol_mint:
            sol_price = get_sol_price_usd()
            if sol_price is None:
                print(f"❌ Cannot get SOL price - skipping SOL token")
                return 0.0
            return sol_price
        
        # Try DexScreener API for token price first (direct price, no SOL dependency)
        for attempt in range(2):
            try:
                url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
                response = requests.get(url, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    pairs = data.get("pairs", [])
                    if pairs:
                        # Get the first pair with a valid price
                        for pair in pairs:
                            price = float(pair.get("priceUsd", 0))
                            if price > 0:
                                print(f"✅ Token price from DexScreener: ${price}")
                                return price
            except Exception as e:
                print(f"⚠️ DexScreener price API error (attempt {attempt + 1}/2): {e}")
            
            if attempt < 1:
                time.sleep(1)
        
        # Try Birdeye API for Solana tokens (direct price, no SOL dependency)
        try:
            url = f"https://public-api.birdeye.so/public/price?address={token_address}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("success") and data.get("data", {}).get("value"):
                    price = float(data["data"]["value"])
                    print(f"✅ Token price from Birdeye: ${price}")
                    return price
        except Exception as e:
            print(f"⚠️ Birdeye price API error: {e}")
        
        # Fallback to CoinGecko for common tokens (direct price, no SOL dependency)
        token_mapping = {
            "So11111111111111111111111111111111111111112": "solana",
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": "usd-coin",
            "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB": "tether",
            "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So": "msol",
            "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263": "bonk",
            "7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr": "pepe",
            "EPeUFDgHRxs9xxEPVaL6kfGQvCon7jmAWKVUHuux1Tpz": "jito"
        }
        
        if token_address in token_mapping:
            import os
            coingecko_key = os.getenv("COINGECKO_API_KEY", "").strip()
            for attempt in range(2):
                try:
                    coingecko_id = token_mapping[token_address]
                    base_url = (
                        "https://pro-api.coingecko.com/api/v3"
                        if coingecko_key
                        else "https://api.coingecko.com/api/v3"
                    )
                    url = f"{base_url}/simple/price?ids={coingecko_id}&vs_currencies=usd"
                    headers = {}
                    if coingecko_key:
                        headers["x-cg-pro-api-key"] = coingecko_key
                    
                    response = requests.get(url, headers=headers, timeout=15)
                    if response.status_code == 200:
                        data = response.json()
                        if coingecko_id in data and "usd" in data[coingecko_id]:
                            price = float(data[coingecko_id]["usd"])
                            print(f"✅ CoinGecko price for {token_address[:8]}...{token_address[-8:]}: ${price}")
                            return price
                except Exception as e:
                    print(f"⚠️ CoinGecko price API error (attempt {attempt + 1}/2): {e}")
                
                if attempt < 1:
                    time.sleep(1)
        
        # If all APIs fail, return a small positive value to prevent false delisting
        # This indicates "price unknown" rather than "price is zero"
        print(f"⚠️ Token not found in any price API: {token_address[:8]}...{token_address[-8:]}")
        print(f"🔄 Returning fallback price to prevent false delisting - actual price unknown")
        return 0.000001  # Small positive value instead of 0

    def get_jupiter_quote(self, input_mint: str, output_mint: str, amount: int, slippage: float = 0.10) -> Dict[str, Any]:
        """Get swap quote from Jupiter with better error handling
        
        NOTE: Jupiter API has changed to api.jup.ag (may require authentication)
        """
        try:
            url = f"{JUPITER_API_BASE}/ultra/v1/order"
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": str(amount),
                "slippageBps": int(slippage * 10000),
                "taker": self.wallet_address
            }

            response = requests.get(url, params=params, headers=JUPITER_HEADERS, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data and not data.get("error"):
                    print(f"✅ Jupiter quote: {data.get('inAmount', 'N/A')} -> {data.get('outAmount', 'N/A')}")
                    return data
                error_msg = data.get('error', 'Unknown error')
                print(f"⚠️ Jupiter quote failed: {error_msg}")
                return {}

            if response.status_code == 400:
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', 'Bad Request')
                    print(f"⚠️ Jupiter quote 400 error: {error_msg}")

                    if amount > 1_000_000:
                        smaller_amount = amount // 2
                        print(f"🔄 Retrying with smaller amount: {smaller_amount}")
                        params["amount"] = str(smaller_amount)
                        retry_response = requests.get(url, params=params, headers=JUPITER_HEADERS, timeout=15)
                        if retry_response.status_code == 200:
                            retry_data = retry_response.json()
                            if retry_data and not retry_data.get("error"):
                                print(f"✅ Jupiter quote with smaller amount: {retry_data.get('inAmount', 'N/A')} -> {retry_data.get('outAmount', 'N/A')}")
                                return retry_data
                except Exception as e:
                    print(f"⚠️ Could not parse Jupiter 400 error: {e}")

                return {}

            print(f"⚠️ Jupiter quote failed: {response.status_code} - {response.text}")
            return {}
        except Exception as e:
            print(f"❌ Jupiter quote error: {e}")
            return {}

    def execute_jupiter_swap(self, quote_response: Dict[str, Any]) -> Tuple[str, bool]:
        """Execute swap using Jupiter"""
        try:
            if not quote_response:
                return "", False
            
            # Get swap transaction  
            url = f"{JUPITER_API_BASE}/ultra/v1/swap"
            payload = {
                "quoteResponse": quote_response,
                "userPublicKey": self.wallet_address,
                "wrapAndUnwrapSol": True,
                "computeUnitPriceMicroLamports": 1000,
                "asLegacyTransaction": False,
                "useSharedAccounts": True
            }
            
            print(f"🌐 Requesting swap transaction from Jupiter...")
            response = requests.post(url, json=payload, timeout=15)
            
            if response.status_code == 200:
                swap_data = response.json()
                
                if not self.keypair:
                    print("❌ No Solana keypair available for signing")
                    return "", False
                
                # Check if we have the transaction data
                transaction_data = swap_data.get("transaction") or swap_data.get("swapTransaction")
                if not transaction_data:
                    print(f"❌ No swap transaction in response: {swap_data}")
                    return "", False
                
                # Sign and send transaction
                print(f"📝 Transaction data length: {len(transaction_data)}")
                
                try:
                    print(f"📡 Processing Jupiter Ultra transaction...")
                    
                    # Decode base64 transaction data
                    import base64
                    decoded_data = base64.b64decode(transaction_data)
                    print(f"✅ Base64 decoded, length: {len(decoded_data)} bytes")
                    
                    # Try to deserialize and sign the transaction
                    try:
                        from solders.transaction import Transaction
                        transaction = Transaction.deserialize(decoded_data)
                        print(f"✅ Transaction deserialized, {len(transaction.instructions)} instructions")
                        
                        # Sign the transaction
                        transaction.sign(self.keypair)
                        print(f"✅ Transaction signed")
                        
                        # Serialize the signed transaction
                        signed_data = transaction.serialize()
                        signed_base64 = base64.b64encode(signed_data).decode('utf-8')
                        print(f"✅ Transaction serialized and encoded")
                        
                        # Send the signed transaction via RPC
                        rpc_payload = {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "sendTransaction",
                            "params": [
                                signed_base64,
                                {
                                    "encoding": "base64",
                                    "skipPreflight": False,
                                    "maxRetries": 3
                                }
                            ]
                        }
                        
                        try:
                            rpc_response = requests.post(SOLANA_RPC_URL, json=rpc_payload, timeout=30)
                        except requests.exceptions.ConnectionError as e:
                            error_msg = str(e).lower()
                            if "broken pipe" in error_msg or "errno 32" in error_msg:
                                print(f"❌ Broken pipe error sending Solana transaction: Connection closed unexpectedly")
                            else:
                                print(f"❌ Connection error sending Solana transaction: {e}")
                            return "", False
                        except OSError as e:
                            if e.errno == 32:  # Broken pipe
                                print(f"❌ Broken pipe error (errno 32) sending Solana transaction: {e}")
                            else:
                                print(f"❌ OS error sending Solana transaction: {e}")
                            return "", False
                        
                        if rpc_response.status_code == 200:
                            rpc_result = rpc_response.json()
                            if "result" in rpc_result:
                                tx_hash = rpc_result["result"]
                                print(f"✅ Transaction sent: {tx_hash}")
                                
                                # CRITICAL: Wait for transaction confirmation
                                print(f"⏳ Waiting for transaction confirmation...")
                                confirmed = self._confirm_transaction(tx_hash, max_retries=30)
                                if not confirmed:
                                    print(f"❌ Transaction {tx_hash} failed to confirm or was rejected")
                                    return tx_hash, False
                                
                                # Verify transaction succeeded
                                print(f"🔍 Verifying transaction status...")
                                tx_success = self._verify_transaction_success(tx_hash)
                                if not tx_success:
                                    print(f"❌ Transaction {tx_hash} failed on-chain")
                                    return tx_hash, False
                                
                                print(f"✅ Transaction confirmed and successful: {tx_hash}")
                                return tx_hash, True
                            elif "error" in rpc_result:
                                error_msg = rpc_result["error"]
                                print(f"❌ RPC error: {error_msg}")
                                
                                # If it's a signature verification error, try a different approach
                                if "signature verification" in str(error_msg).lower():
                                    print(f"🔄 Trying alternative transaction approach...")
                                    return self._try_alternative_jupiter_swap(quote_response)
                                else:
                                    return "", False
                        else:
                            print(f"❌ RPC request failed: {rpc_response.status_code}")
                            return "", False
                            
                    except Exception as deserialize_error:
                        print(f"❌ Transaction deserialization failed: {deserialize_error}")
                        print(f"🔄 Trying alternative approach...")
                        return self._try_alternative_jupiter_swap(quote_response)
                        
                except Exception as tx_error:
                    print(f"❌ Transaction processing error: {tx_error}")
                    return "", False
            else:
                print(f"❌ Jupiter swap request failed: {response.status_code}")
                print(f"Response: {response.text}")
                return "", False
                
        except Exception as e:
            print(f"❌ Jupiter swap error: {e}")
            return "", False

    def _try_alternative_jupiter_swap(self, quote_response: Dict[str, Any]) -> Tuple[str, bool]:
        """Alternative Jupiter swap method"""
        try:
            print(f"🔄 Trying alternative Jupiter swap method...")
            
            # Try using Jupiter's POST endpoint with different parameters
            url = "https://api.jup.ag/v6/swap"
            payload = {
                "quoteResponse": quote_response,
                "userPublicKey": self.wallet_address,
                "wrapAndUnwrapSol": True,
                "computeUnitPriceMicroLamports": 1000,
                "asLegacyTransaction": True,  # Try legacy format
                "useSharedAccounts": False
            }
            
            response = requests.post(url, json=payload, timeout=15)
            
            if response.status_code == 200:
                swap_data = response.json()
                
                if "swapTransaction" in swap_data:
                    transaction_data = swap_data["swapTransaction"]
                    print(f"📝 Alternative transaction data length: {len(transaction_data)}")
                    
                    # Try to send directly via RPC
                    rpc_payload = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "sendTransaction",
                        "params": [
                            transaction_data,
                            {
                                "encoding": "base64",
                                "skipPreflight": True,
                                "maxRetries": 3
                            }
                        ]
                    }
                    
                    rpc_response = requests.post(SOLANA_RPC_URL, json=rpc_payload, timeout=30)
                    
                    if rpc_response.status_code == 200:
                        rpc_result = rpc_response.json()
                        if "result" in rpc_result:
                            tx_hash = rpc_result["result"]
                            print(f"✅ Transaction sent (alternative): {tx_hash}")
                            
                            # CRITICAL: Wait for transaction confirmation
                            print(f"⏳ Waiting for transaction confirmation...")
                            confirmed = self._confirm_transaction(tx_hash, max_retries=30)
                            if not confirmed:
                                print(f"❌ Transaction {tx_hash} failed to confirm or was rejected")
                                return tx_hash, False
                            
                            # Verify transaction succeeded
                            print(f"🔍 Verifying transaction status...")
                            tx_success = self._verify_transaction_success(tx_hash)
                            if not tx_success:
                                print(f"❌ Transaction {tx_hash} failed on-chain")
                                return tx_hash, False
                            
                            print(f"✅ Transaction confirmed and successful: {tx_hash}")
                            return tx_hash, True
                        elif "error" in rpc_result:
                            print(f"❌ Alternative RPC error: {rpc_result['error']}")
                            return "", False
                    else:
                        print(f"❌ Alternative RPC request failed: {rpc_response.status_code}")
                        return "", False
                else:
                    print(f"❌ No swap transaction in alternative response")
                    return "", False
            else:
                print(f"❌ Alternative Jupiter swap request failed: {response.status_code}")
                return "", False
                
        except Exception as e:
            print(f"❌ Alternative Jupiter swap error: {e}")
            return "", False

    def execute_trade(self, token_address: str, amount_usd: float, is_buy: bool = True) -> Tuple[str, bool]:
        """Execute trade using Jupiter (most reliable Solana DEX aggregator)"""
        try:
            print(f"🚀 Executing {'buy' if is_buy else 'sell'} for {token_address[:8]}...{token_address[-8:]}")
            
            # Validate minimum trade amount (Jupiter requires reasonable amounts)
            min_trade_usd = 1.0  # Minimum $1 USD
            if amount_usd < min_trade_usd:
                print(f"❌ Trade amount ${amount_usd} too small (minimum ${min_trade_usd})")
                return "", False
            
            # Get token liquidity to adjust trade amount
            try:
                from src.core.strategy import _get_token_liquidity
                liquidity = _get_token_liquidity(token_address)
                if liquidity and liquidity < amount_usd * 2:  # If liquidity is less than 2x trade amount
                    adjusted_amount = min(amount_usd, liquidity * 0.1)  # Use 10% of liquidity or original amount
                    print(f"🔄 Adjusting trade amount from ${amount_usd} to ${adjusted_amount} due to low liquidity (${liquidity})")
                    amount_usd = adjusted_amount
                    # Re-check minimum after adjustment
                    if amount_usd < min_trade_usd:
                        print(f"❌ Adjusted amount ${amount_usd} too small (minimum ${min_trade_usd})")
                        return "", False
            except Exception as e:
                print(f"⚠️ Could not get liquidity info: {e}")
            
            if is_buy:
                # Buying token with SOL (wrap/unwrap enabled in Jupiter)
                from src.utils.utils import get_sol_price_usd
                sol_price = get_sol_price_usd()
                if sol_price <= 0:
                    print("❌ Cannot get SOL price")
                    return "", False
                
                # Balance gate - check available SOL balance before trading
                try:
                    available_sol = self.get_solana_balance()  # in SOL
                    available_usd = float(available_sol) * float(sol_price)
                    
                    # Require 5% buffer for fees/slippage
                    buffer_pct = 0.05
                    required_usd = float(amount_usd) * (1.0 + buffer_pct)
                    
                    if available_usd < required_usd:
                        print(f"❌ Insufficient SOL balance: ${available_usd:.2f} available, ${required_usd:.2f} required")
                        return "", False
                except Exception as e:
                    print(f"⚠️ Balance check error: {e}")
                    # Fail safe: block the trade if we can't verify balance
                    return "", False
                
                sol_amount = amount_usd / sol_price
                sol_amount_lamports = int(sol_amount * 1_000_000_000)  # SOL has 9 decimals
                
                # Ensure minimum SOL amount (0.01 SOL equivalent to ~$1-2)
                min_sol_lamports = int(0.01 * 1_000_000_000)  # 0.01 SOL
                if sol_amount_lamports < min_sol_lamports:
                    print(f"❌ SOL amount too small: {sol_amount_lamports} lamports (minimum {min_sol_lamports})")
                    return "", False
                
                input_mint = WSOL_MINT
                output_mint = token_address
                amount = sol_amount_lamports
            else:
                # Selling token for USDC
                # CRITICAL: We need the actual token balance in raw units, not USD amount
                print(f"🔍 [SELL DEBUG] Token: {token_address[:8]}...{token_address[-8:]}")
                print(f"🔍 [SELL DEBUG] Amount USD: {amount_usd}")
                print(f"🔍 [SELL DEBUG] Getting raw token balance...")
                
                raw_token_balance = self.get_token_raw_balance(token_address)
                print(f"🔍 [SELL DEBUG] Raw balance result: {raw_token_balance}")
                
                if raw_token_balance is None:
                    print(f"❌ [SELL ERROR] Raw balance check failed (returned None) - RPC error or account issue")
                    print(f"   Possible causes:")
                    print(f"   - RPC connection failed")
                    print(f"   - Token account not found")
                    print(f"   - Network timeout")
                    return "", False
                elif raw_token_balance <= 0:
                    print(f"❌ [SELL ERROR] Raw balance is <= 0: {raw_token_balance}")
                    print(f"   Token may have been sold already or balance is zero")
                    return "", False
                
                input_mint = token_address
                output_mint = USDC_MINT  # Always sell to USDC
                amount = raw_token_balance  # Use raw token amount for swap
                print(f"💵 Selling {token_address[:8]}...{token_address[-8:]} for USDC")
                print(f"🔍 [SELL DEBUG] Swap params: input={input_mint[:8]}..., output={output_mint[:8]}..., amount={amount}")
            
            # Use higher slippage for small trades (microcaps)
            slippage = 0.15 if amount_usd < 50 else 0.10
            print(f"🎯 Using slippage: {slippage*100:.1f}%")
            
            # Get quote
            print(f"📊 Getting Jupiter quote: {input_mint[:8]}... -> {output_mint[:8]}... (amount: {amount})")
            quote = self.get_jupiter_quote(input_mint, output_mint, amount, slippage=slippage)
            
            if not quote:
                print(f"❌ [SELL ERROR] No quote available for {token_address[:8]}...{token_address[-8:]} -> USDC")
                print(f"   Possible causes:")
                print(f"   - Token may be delisted or have no liquidity")
                print(f"   - Jupiter API rate limit or error")
                print(f"   - Insufficient liquidity for swap")
                print(f"   - Token pair not available on Jupiter")
                return "", False
            
            print(f"✅ Quote received successfully")
            
            # Verify we're selling to USDC for sells
            if not is_buy:
                # The quote response should have outAmount which indicates USDC will be received
                out_amount = quote.get('outAmount', '0')
                if out_amount and int(out_amount) > 0:
                    print(f"✅ Quote confirmed: Will receive {int(out_amount) / 1_000_000:.2f} USDC (raw: {out_amount})")
                else:
                    print(f"⚠️ Warning: Quote shows zero USDC output")
            
            # Execute swap
            print(f"🔍 [SELL DEBUG] Executing Jupiter swap...")
            tx_hash, success = self.execute_jupiter_swap(quote)
            
            if success:
                print(f"✅ [SELL SUCCESS] Swap executed successfully: {tx_hash}")
            else:
                print(f"❌ [SELL FAILED] Swap execution failed (tx_hash: {tx_hash})")
                print(f"   Check transaction status if tx_hash is provided")
            
            return tx_hash, success
            
        except Exception as e:
            print(f"❌ Trade execution error: {e}")
            return "", False

    def get_solana_balance(self) -> float:
        """Get SOL balance"""
        try:
            balance = self.client.get_balance(PublicKey.from_string(self.wallet_address))
            if balance.value:
                return float(balance.value) / 1_000_000_000  # Convert lamports to SOL
            return 0.0
        except Exception as e:
            print(f"❌ Error getting SOL balance: {e}")
            return 0.0

    def _confirm_transaction(self, tx_hash: str, max_retries: int = 30) -> bool:
        """Wait for transaction to be confirmed on-chain"""
        import time
        
        for attempt in range(max_retries):
            try:
                rpc_payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getSignatureStatuses",
                    "params": [[tx_hash], {"searchTransactionHistory": True}]
                }
                
                response = requests.post(SOLANA_RPC_URL, json=rpc_payload, timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    if "result" in result and result["result"]["value"]:
                        status = result["result"]["value"][0]
                        if status:
                            err = status.get("err")
                            if err is None:
                                # Transaction confirmed without error
                                print(f"✅ Transaction confirmed on-chain")
                                return True
                            else:
                                # Transaction failed
                                print(f"❌ Transaction failed: {err}")
                                return False
                
                # Transaction not yet confirmed, wait and retry
                if attempt < max_retries - 1:
                    time.sleep(1)
                    
            except Exception as e:
                print(f"⚠️ Error checking transaction status: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        
        print(f"⚠️ Transaction confirmation timeout after {max_retries} attempts")
        return False

    def _verify_transaction_success(self, tx_hash: str) -> bool:
        """Verify transaction actually succeeded by checking transaction details"""
        try:
            rpc_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTransaction",
                "params": [
                    tx_hash,
                    {
                        "encoding": "jsonParsed",
                        "maxSupportedTransactionVersion": 0
                    }
                ]
            }
            
            response = requests.post(SOLANA_RPC_URL, json=rpc_payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                if "result" in result and result["result"]:
                    tx_data = result["result"]
                    meta = tx_data.get("meta", {})
                    
                    # Check if transaction succeeded
                    if meta.get("err") is not None:
                        print(f"❌ Transaction error in meta: {meta['err']}")
                        return False
                    
                    # Transaction succeeded
                    return True
            
            return False
        except Exception as e:
            print(f"⚠️ Error verifying transaction: {e}")
            # If we can't verify, assume it might have succeeded (conservative)
            return True

    def get_token_raw_balance(self, token_mint: str) -> Optional[int]:
        """
        Get raw token balance in smallest units (not UI amount) using ATA method.
        Returns the actual token amount needed for swap quotes.
        Uses ATA (Associated Token Account) method which is more reliable.
        """
        try:
            from src.utils.solana_ata_utils import get_associated_token_address
            from src.utils.http_utils import post_json
            
            if not self.keypair:
                print("⚠️ No keypair available for balance check")
                return None
            
            # Calculate ATA address (more efficient and reliable than searching)
            wallet_address = str(self.keypair.pubkey())
            ata_address = get_associated_token_address(wallet_address, token_mint)
            
            if not ata_address:
                print(f"⚠️ Failed to calculate ATA address for {token_mint[:8]}...{token_mint[-8:]}")
                # Fallback to old method if ATA calculation fails
                return self._get_token_raw_balance_fallback(token_mint)
            
            # Use getTokenAccountBalance (more efficient than getTokenAccountsByOwner)
            rpc_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTokenAccountBalance",
                "params": [
                    ata_address,
                    {
                        "commitment": "confirmed"
                    }
                ]
            }
            
            # Use http_utils for retry logic
            result = post_json(SOLANA_RPC_URL, rpc_payload, timeout=15, retries=2, backoff=1.0)
            
            if result and "result" in result and "value" in result["result"]:
                token_amount = result["result"]["value"]
                raw_amount_str = token_amount.get("amount", "0")
                raw_balance = int(raw_amount_str)
                print(f"✅ Raw token balance retrieved: {raw_balance} (via ATA: {ata_address[:8]}...)")
                return raw_balance
            else:
                # Check if account doesn't exist (token not held)
                if result and "error" in result:
                    error_code = result["error"].get("code", 0)
                    if error_code == -32602:  # Invalid params - account might not exist
                        print(f"⚠️ Token account not found (no balance) for {token_mint[:8]}...")
                        return 0
                
                print(f"⚠️ No balance found for token {token_mint[:8]}...{token_mint[-8:]}")
                # Fallback to old method
                return self._get_token_raw_balance_fallback(token_mint)
            
        except Exception as e:
            print(f"❌ Error getting raw token balance via ATA: {e}")
            import traceback
            print(traceback.format_exc())
            # Fallback to old method
            return self._get_token_raw_balance_fallback(token_mint)
    
    def _get_token_raw_balance_fallback(self, token_mint: str) -> Optional[int]:
        """
        Fallback method to get raw token balance using getTokenAccountsByOwner.
        Used when ATA method fails.
        """
        try:
            print(f"🔄 Trying fallback method for token balance...")
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
            
            response = requests.post(SOLANA_RPC_URL, json=rpc_payload, timeout=10)
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
                    print(f"⚠️ RPC response error for token raw balance (fallback): {result.get('error', 'Unknown')}")
                    return None
            else:
                print(f"⚠️ HTTP error {response.status_code} getting token raw balance (fallback)")
                return None
        except Exception as e:
            print(f"❌ Error in fallback token raw balance: {e}")
            return None

# Legacy functions for backward compatibility
def get_token_price_usd(token_address: str) -> float:
    """Legacy function for getting token price"""
    executor = SimpleSolanaExecutor()
    return executor.get_token_price_usd(token_address)

def get_solana_balance() -> float:
    """Legacy function for getting SOL balance"""
    executor = SimpleSolanaExecutor()
    return executor.get_solana_balance()

def execute_solana_trade(token_address: str, amount_usd: float, is_buy: bool = True) -> Tuple[str, bool]:
    """Legacy function for executing trades"""
    executor = SimpleSolanaExecutor()
    return executor.execute_trade(token_address, amount_usd, is_buy)

# Additional functions for multi-chain compatibility
def buy_token_solana(token_address: str, amount_usd: float, symbol: str = "", test_mode: bool = False) -> Tuple[str, bool]:
    """Compatibility wrapper that delegates to the guardrail-enabled Jupiter executor."""
    from src.execution.jupiter_executor import buy_token_solana as _buy_token_solana

    return _buy_token_solana(
        token_address=token_address,
        amount_usd=amount_usd,
        symbol=symbol,
        test_mode=test_mode,
    )

def sell_token_solana(token_address: str, amount_usd: float, symbol: str = "", test_mode: bool = False) -> Tuple[str, bool]:
    """Compatibility wrapper that delegates to the guardrail-enabled Jupiter executor."""
    from src.execution.jupiter_executor import sell_token_solana as _sell_token_solana

    return _sell_token_solana(
        token_address=token_address,
        amount_usd=amount_usd,
        symbol=symbol,
        test_mode=test_mode,
    )

def get_solana_executor():
    """Get Solana executor instance (for backward compatibility)"""
    return SimpleSolanaExecutor()
