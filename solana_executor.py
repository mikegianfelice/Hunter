import json
import time
import requests
import base58
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from solana.rpc.api import Client
from solana.transaction import Transaction
from solana.keypair import Keypair
from solana.publickey import PublicKey
from solana.system_program import TransferParams, transfer
from solana.rpc.commitment import Commitment
import struct

from secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY

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
                self.keypair = Keypair.from_secret_key(base58.b58decode(self.private_key))
                print(f"✅ Solana wallet initialized: {self.keypair.public_key}...{str(self.keypair.public_key)[-8:]}")
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
            return get_sol_price_usd()
        
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
        
        print(f"⚠️ Token not found in any price API: {token_address[:8]}...{token_address[-8:]}")
        return 0.0

    def get_jupiter_quote(self, input_mint: str, output_mint: str, amount: int, slippage: float = 0.02) -> Dict[str, Any]:
        """Get swap quote from Jupiter with better error handling"""
        try:
            url = "https://quote-api.jup.ag/v6/quote"
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
                if data.get("data"):
                    print(f"✅ Jupiter quote: {data['data']['inputAmount']} -> {data['data']['outputAmount']}")
                    return data["data"]
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
                            if retry_data.get("data"):
                                print(f"✅ Jupiter quote with smaller amount: {retry_data['data']['inputAmount']} -> {retry_data['data']['outputAmount']}")
                                return retry_data["data"]
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
            url = "https://quote-api.jup.ag/v6/swap"
            payload = {
                "quoteResponse": quote_response,
                "userPublicKey": self.wallet_address,
                "wrapUnwrapSOL": True
            }
            
            response = requests.post(url, json=payload, timeout=15)
            if response.status_code == 200:
                swap_data = response.json()
                
                if not self.keypair:
                    print("❌ No Solana keypair available for signing")
                    return "", False
                
                # Sign and send transaction
                transaction_data = swap_data["swapTransaction"]
                transaction = Transaction.deserialize(base58.b58decode(transaction_data))
                
                # Sign transaction
                transaction.sign(self.keypair)
                
                # Send transaction
                tx_hash = self.client.send_transaction(transaction)
                
                if tx_hash.value:
                    print(f"✅ Jupiter swap executed: {tx_hash.value}")
                    return tx_hash.value, True
                else:
                    print(f"❌ Jupiter swap failed: {tx_hash}")
                    return "", False
            else:
                print(f"❌ Jupiter swap request failed: {response.status_code}")
                return "", False
                
        except Exception as e:
            print(f"❌ Jupiter swap error: {e}")
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
            balance = self.client.get_balance(PublicKey(self.wallet_address))
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
    if test_mode:
        print(f"🔄 Simulating Solana buy for {symbol} ({token_address[:8]}...{token_address[-8:]})")
        return f"simulated_solana_tx_{int(time.time())}", True
    
    executor = SimpleSolanaExecutor()
    return executor.execute_trade(token_address, amount_usd, is_buy=True)

def sell_token_solana(token_address: str, amount_usd: float, symbol: str = "", test_mode: bool = False) -> Tuple[str, bool]:
    """Sell token on Solana (for multi-chain compatibility)"""
    if test_mode:
        print(f"🔄 Simulating Solana sell for {symbol} ({token_address[:8]}...{token_address[-8:]})")
        return f"simulated_solana_tx_{int(time.time())}", True
    
    executor = SimpleSolanaExecutor()
    return executor.execute_trade(token_address, amount_usd, is_buy=False)

def get_solana_executor():
    """Get Solana executor instance (for backward compatibility)"""
    return SimpleSolanaExecutor()
