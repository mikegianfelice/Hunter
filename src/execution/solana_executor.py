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
        from utils import get_sol_price_usd
        
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
            for attempt in range(2):
                try:
                    coingecko_id = token_mapping[token_address]
                    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coingecko_id}&vs_currencies=usd"
                    
                    response = requests.get(url, timeout=15)
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

    def get_jupiter_quote(self, input_mint: str, output_mint: str, amount: int, slippage: float = 0.02) -> Dict[str, Any]:
        """Get swap quote from Jupiter with better error handling
        
        NOTE: Jupiter API has changed to api.jup.ag (may require authentication)
        """
        try:
            url = "https://api.jup.ag/v6/quote"
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": str(amount),
                "slippageBps": int(slippage * 10000),  # Convert to basis points
                "onlyDirectRoutes": "false",
                "asLegacyTransaction": "false"
            }
            
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                # Jupiter v6 returns quote data directly, not wrapped in "data" field
                if data and not data.get("error"):
                    print(f"✅ Jupiter quote: {data.get('inAmount', 'N/A')} -> {data.get('outAmount', 'N/A')}")
                    return data
                else:
                    error_msg = data.get('error', 'Unknown error')
                    print(f"⚠️ Jupiter quote failed: {error_msg}")
                    return {}
            elif response.status_code == 400:
                # Try to get more details about the 400 error
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', 'Bad Request')
                    print(f"⚠️ Jupiter quote 400 error: {error_msg}")
                    
                    # Try with a smaller amount if it's a liquidity issue
                    if amount > 1000000:  # If amount > 1 USDC
                        smaller_amount = amount // 2
                        print(f"🔄 Retrying with smaller amount: {smaller_amount}")
                        params["amount"] = str(smaller_amount)
                        
                        retry_response = requests.get(url, params=params, timeout=15)
                        if retry_response.status_code == 200:
                            retry_data = retry_response.json()
                            if retry_data and not retry_data.get("error"):
                                print(f"✅ Jupiter quote with smaller amount: {retry_data.get('inAmount', 'N/A')} -> {retry_data.get('outAmount', 'N/A')}")
                                return retry_data
                except Exception as e:
                    print(f"⚠️ Could not parse Jupiter 400 error: {e}")
                
                return {}
            else:
                print(f"⚠️ Jupiter quote failed: {response.status_code}")
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
            url = "https://api.jup.ag/v6/swap"
            payload = {
                "quoteResponse": quote_response,
                "userPublicKey": self.wallet_address,
                "wrapUnwrapSOL": True,
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
                if "swapTransaction" not in swap_data:
                    print(f"❌ No swap transaction in response: {swap_data}")
                    return "", False
                
                # Sign and send transaction
                transaction_data = swap_data["swapTransaction"]
                print(f"📝 Transaction data length: {len(transaction_data)}")
                
                try:
                    # For Jupiter v6, we need to properly sign the transaction
                    print(f"📡 Processing Jupiter v6 transaction...")
                    
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
                                print(f"✅ Jupiter swap executed: {tx_hash}")
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
                "wrapUnwrapSOL": True,
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
                            print(f"✅ Alternative Jupiter swap executed: {tx_hash}")
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
            
            # Get token liquidity to adjust trade amount
            try:
                from strategy import _get_token_liquidity
                liquidity = _get_token_liquidity(token_address)
                if liquidity and liquidity < amount_usd * 2:  # If liquidity is less than 2x trade amount
                    adjusted_amount = min(amount_usd, liquidity * 0.1)  # Use 10% of liquidity or original amount
                    print(f"🔄 Adjusting trade amount from ${amount_usd} to ${adjusted_amount} due to low liquidity (${liquidity})")
                    amount_usd = adjusted_amount
            except Exception as e:
                print(f"⚠️ Could not get liquidity info: {e}")
            
            # Convert USD amount to USDC (assuming 1 USDC = 1 USD)
            usdc_amount = int(amount_usd * 1_000_000)  # USDC has 6 decimals
            
            if is_buy:
                # Buying token with USDC
                input_mint = USDC_MINT
                output_mint = token_address
            else:
                # Selling token for USDC
                input_mint = token_address
                output_mint = USDC_MINT
            
            # Get quote
            quote = self.get_jupiter_quote(input_mint, output_mint, usdc_amount)
            if not quote:
                print(f"❌ No quote available for {token_address[:8]}...{token_address[-8:]}")
                return "", False
            
            # Execute swap
            tx_hash, success = self.execute_jupiter_swap(quote)
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
    """Buy token on Solana (for multi-chain compatibility)"""
    executor = SimpleSolanaExecutor()
    
    if test_mode:
        # In test mode, still use real market data for quotes but don't execute
        # Get quote to validate trade would work
        try:
            from utils import get_sol_price_usd
            sol_price = get_sol_price_usd()
            if sol_price <= 0:
                return None, False
            sol_amount = amount_usd / sol_price
            sol_amount_lamports = int(sol_amount * 1_000_000_000)
            quote = executor.get_jupiter_quote(WSOL_MINT, token_address, sol_amount_lamports)
            if quote:
                print(f"🔄 Test mode: Validated Solana buy for {symbol} ({token_address[:8]}...{token_address[-8:]}) - Transaction not sent")
                return None, True
        except Exception as e:
            print(f"⚠️ Test mode validation failed: {e}")
            return None, False
    
    return executor.execute_trade(token_address, amount_usd, is_buy=True)

def sell_token_solana(token_address: str, amount_usd: float, symbol: str = "", test_mode: bool = False) -> Tuple[str, bool]:
    """Sell token on Solana (for multi-chain compatibility)"""
    executor = SimpleSolanaExecutor()
    
    if test_mode:
        # In test mode, still use real market data for quotes but don't execute
        # Get quote to validate trade would work
        try:
            usdc_amount = int(amount_usd * 1_000_000)
            quote = executor.get_jupiter_quote(token_address, USDC_MINT, usdc_amount)
            if quote:
                print(f"🔄 Test mode: Validated Solana sell for {symbol} ({token_address[:8]}...{token_address[-8:]}) - Transaction not sent")
                return None, True
        except Exception as e:
            print(f"⚠️ Test mode validation failed: {e}")
            return None, False
    
    return executor.execute_trade(token_address, amount_usd, is_buy=False)

def get_solana_executor():
    """Get Solana executor instance (for backward compatibility)"""
    return SimpleSolanaExecutor()
