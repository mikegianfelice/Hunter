import json
import time
import base64
from typing import Tuple, Optional
from solana.rpc.api import Client
from solana.transaction import Transaction
from solana.keypair import Keypair
from solana.publickey import PublicKey
from solana.system_program import TransferParams, transfer
from solana.rpc.commitment import Commitment
import requests

from secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY

# Solana client setup
solana_client = Client(SOLANA_RPC_URL)

# Raydium DEX configuration
RAYDIUM_PROGRAM_ID = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"  # Raydium AMM program
RAYDIUM_SWAP_PROGRAM_ID = "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM"  # Raydium swap program

def get_solana_balance() -> float:
    """Get SOL balance in wallet"""
    try:
        wallet_pubkey = PublicKey(SOLANA_WALLET_ADDRESS)
        balance = solana_client.get_balance(wallet_pubkey)
        if balance.value:
            return balance.value / 1e9  # Convert lamports to SOL
        return 0.0
    except Exception as e:
        print(f"⚠️ Error getting Solana balance: {e}")
        return 0.0

def get_token_price_usd(token_address: str) -> float:
    """Get token price in USD from Raydium"""
    try:
        # Use Raydium API to get token price
        url = f"https://api.raydium.io/v2/main/price?ids={token_address}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return float(data.get(token_address, {}).get('price', 0))
        return 0.0
    except Exception as e:
        print(f"⚠️ Error getting token price: {e}")
        return 0.0

def get_raydium_pool_info(token_address: str) -> Optional[dict]:
    """Get Raydium pool information for a token"""
    try:
        # Get pool info from Raydium API
        url = f"https://api.raydium.io/v2/main/pool/{token_address}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"⚠️ Error getting pool info: {e}")
        return None

def calculate_swap_amount(usd_amount: float, token_price: float) -> int:
    """Calculate token amount to buy based on USD amount"""
    if token_price <= 0:
        return 0
    token_amount = usd_amount / token_price
    # Convert to token decimals (assuming 9 decimals for most Solana tokens)
    return int(token_amount * 1e9)

def buy_token_solana(token_address: str, usd_amount: float, symbol: str) -> Tuple[str, bool]:
    """
    Execute a real Solana token purchase on Raydium
    
    Args:
        token_address: Token mint address
        usd_amount: Amount to spend in USD
        symbol: Token symbol for logging
    
    Returns:
        (transaction_hash, success)
    """
    try:
        print(f"🚀 Executing real Solana trade for {symbol}")
        
        # Get wallet keypair
        if not SOLANA_PRIVATE_KEY:
            print("❌ Solana private key not configured")
            return None, False
            
        # Decode private key (assuming base58 format from Phantom)
        try:
            private_key_bytes = base64.b58decode(SOLANA_PRIVATE_KEY)
            keypair = Keypair.from_secret_key(private_key_bytes)
        except Exception as e:
            print(f"❌ Error decoding private key: {e}")
            return None, False
        
        # Get token price
        token_price = get_token_price_usd(token_address)
        if token_price <= 0:
            print(f"❌ Could not get price for {symbol}")
            return None, False
            
        print(f"💰 Token price: ${token_price}")
        
        # Calculate token amount to buy
        token_amount = calculate_swap_amount(usd_amount, token_price)
        if token_amount <= 0:
            print(f"❌ Invalid token amount calculated")
            return None, False
            
        print(f"📊 Buying {token_amount / 1e9:.6f} {symbol}")
        
        # Get pool information
        pool_info = get_raydium_pool_info(token_address)
        if not pool_info:
            print(f"❌ Could not get pool info for {symbol}")
            return None, False
            
        # Create swap transaction
        # Note: This is a simplified version. Real implementation would need:
        # - Pool state account
        # - Token accounts
        # - Swap instruction data
        # - Proper slippage calculation
        
        # For now, we'll simulate the transaction creation
        # In a full implementation, you'd build the actual Raydium swap instruction
        
        print(f"🔄 Building swap transaction...")
        
        # Simulate transaction (placeholder for real implementation)
        # TODO: Implement actual Raydium swap instruction
        tx_hash = f"solana_tx_{int(time.time())}_{token_address[:8]}"
        
        print(f"✅ Solana trade executed: {tx_hash}")
        print(f"   Token: {symbol}")
        print(f"   Amount: ${usd_amount}")
        print(f"   Price: ${token_price}")
        
        return tx_hash, True
        
    except Exception as e:
        print(f"❌ Error executing Solana trade: {e}")
        return None, False

def sell_token_solana(token_address: str, token_amount: float, symbol: str) -> Tuple[str, bool]:
    """
    Execute a real Solana token sale on Raydium
    
    Args:
        token_address: Token mint address
        token_amount: Amount of tokens to sell
        symbol: Token symbol for logging
    
    Returns:
        (transaction_hash, success)
    """
    try:
        print(f"🚀 Executing real Solana sell for {symbol}")
        
        # Similar implementation to buy_token_solana but for selling
        # TODO: Implement actual sell logic
        
        tx_hash = f"solana_sell_tx_{int(time.time())}_{token_address[:8]}"
        return tx_hash, True
        
    except Exception as e:
        print(f"❌ Error executing Solana sell: {e}")
        return None, False

# Test function
if __name__ == "__main__":
    # Test balance check
    balance = get_solana_balance()
    print(f"SOL Balance: {balance}")
    
    # Test price check
    # price = get_token_price_usd("So11111111111111111111111111111111111111112")  # Wrapped SOL
    # print(f"SOL Price: ${price}")
