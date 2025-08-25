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
        """Get token price in USD using multiple sources"""
        # Try Jupiter price API first
        try:
            url = "https://price.jup.ag/v4/price"
            params = {"ids": token_address}
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("data") and token_address in data["data"]:
                    price = float(data["data"][token_address]["price"])
                    print(f"✅ Jupiter price for {token_address[:8]}...{token_address[-8:]}: ${price}")
                    return price
        except Exception as e:
            print(f"⚠️ Jupiter price API error: {e}")
        
        # Fallback to CoinGecko for common tokens
        token_mapping = {
            "So11111111111111111111111111111111111111112": "solana",
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": "usd-coin"
        }
        
        if token_address in token_mapping:
            try:
                coingecko_id = token_mapping[token_address]
                url = f"https://api.coingecko.com/api/v3/simple/price?ids={coingecko_id}&vs_currencies=usd"
                
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if coingecko_id in data and "usd" in data[coingecko_id]:
                        price = float(data[coingecko_id]["usd"])
                        print(f"✅ CoinGecko price for {token_address[:8]}...{token_address[-8:]}: ${price}")
                        return price
            except Exception as e:
                print(f"⚠️ CoinGecko price API error: {e}")
        
        print(f"⚠️ Token not found in any price API: {token_address[:8]}...{token_address[-8:]}")
        return 0.0

    def get_jupiter_quote(self, input_mint: str, output_mint: str, amount: int, slippage: float = 0.02) -> Dict[str, Any]:
        """Get swap quote from Jupiter"""
        try:
            url = "https://quote-api.jup.ag/v6/quote"
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": str(amount),
                "slippageBps": int(slippage * 10000),  # Convert to basis points
                "onlyDirectRoutes": False,
                "asLegacyTransaction": False
            }
            
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data.get("data"):
                    print(f"✅ Jupiter quote: {data['data']['inputAmount']} -> {data['data']['outputAmount']}")
                    return data["data"]
                else:
                    print(f"⚠️ Jupiter quote failed: {data.get('error', 'Unknown error')}")
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
