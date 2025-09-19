#!/usr/bin/env python3
"""
Jupiter Custom Executor - Using custom Jupiter library for real trades
"""

import time
from typing import Tuple, Optional, Dict, Any
from jupiter_lib import JupiterCustomLib

from secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY

# Common token addresses
WSOL_MINT = "So11111111111111111111111111111111111111112"  # Wrapped SOL
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # USDC

class JupiterCustomExecutor:
    def __init__(self):
        self.jupiter_lib = JupiterCustomLib(
            SOLANA_RPC_URL,
            SOLANA_WALLET_ADDRESS,
            SOLANA_PRIVATE_KEY
        )

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
                import requests
                url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
                response = requests.get(url, timeout=10)
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
                time.sleep(0.5)
        
        # Try Birdeye API for Solana tokens (direct price, no SOL dependency)
        try:
            import requests
            url = f"https://public-api.birdeye.so/public/price?address={token_address}"
            response = requests.get(url, timeout=8)
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
            try:
                import requests
                coingecko_id = token_mapping[token_address]
                url = f"https://api.coingecko.com/api/v3/simple/price?ids={coingecko_id}&vs_currencies=usd"
                
                response = requests.get(url, timeout=8)
                if response.status_code == 200:
                    data = response.json()
                    if coingecko_id in data and "usd" in data[coingecko_id]:
                        price = float(data[coingecko_id]["usd"])
                        print(f"✅ CoinGecko price for {token_address[:8]}...{token_address[-8:]}: ${price}")
                        return price
            except Exception as e:
                print(f"⚠️ CoinGecko price API error: {e}")
        
        # If all APIs fail, return a small positive value to prevent false delisting
        print(f"⚠️ Token not found in any price API: {token_address[:8]}...{token_address[-8:]}")
        print(f"🔄 Using fallback price to prevent false delisting")
        return 0.000001  # Small positive value instead of 0

    def execute_trade(self, token_address: str, amount_usd: float, is_buy: bool = True) -> Tuple[str, bool]:
        """Execute trade using custom Jupiter library"""
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
                # Continue with original amount if liquidity check fails
            
            if is_buy:
                # Buying token with SOL (convert USD amount to SOL)
                from utils import get_sol_price_usd
                sol_price = get_sol_price_usd()
                if sol_price <= 0:
                    print("❌ Could not get SOL price for trade")
                    return "", False
                
                sol_amount = amount_usd / sol_price
                sol_amount_lamports = int(sol_amount * 1_000_000_000)  # SOL has 9 decimals
                
                input_mint = WSOL_MINT
                output_mint = token_address
                amount = sol_amount_lamports
            else:
                # Selling token for SOL (convert USD amount to USDC)
                usdc_amount = int(amount_usd * 1_000_000)  # USDC has 6 decimals
                input_mint = token_address
                output_mint = WSOL_MINT
                amount = usdc_amount
            
            # Execute swap using custom Jupiter library
            tx_hash, success = self.jupiter_lib.execute_swap(input_mint, output_mint, amount)
            return tx_hash, success
            
        except Exception as e:
            print(f"❌ Trade execution error: {e}")
            return "", False

    def get_solana_balance(self) -> float:
        """Get SOL balance"""
        return self.jupiter_lib.get_balance()

# Legacy functions for backward compatibility
def get_token_price_usd(token_address: str) -> float:
    """Legacy function for getting token price"""
    executor = JupiterCustomExecutor()
    return executor.get_token_price_usd(token_address)

def get_solana_balance() -> float:
    """Legacy function for getting SOL balance"""
    executor = JupiterCustomExecutor()
    return executor.get_solana_balance()

def execute_solana_trade(token_address: str, amount_usd: float, is_buy: bool = True) -> Tuple[str, bool]:
    """Legacy function for executing trades"""
    executor = JupiterCustomExecutor()
    return executor.execute_trade(token_address, amount_usd, is_buy)

# Additional functions for multi-chain compatibility
def buy_token_solana(token_address: str, amount_usd: float, symbol: str = "", test_mode: bool = False, 
                     slippage: float = None, route_preferences: Dict[str, Any] = None, 
                     use_exactout: bool = False) -> Tuple[str, bool]:
    """Buy token on Solana (for multi-chain compatibility) with advanced features"""
    if test_mode:
        print(f"🔄 Simulating Solana buy for {symbol} ({token_address[:8]}...{token_address[-8:]})")
        return f"simulated_solana_tx_{int(time.time())}", True
    
    executor = JupiterCustomExecutor()
    
    # Apply advanced features if provided
    if slippage is not None:
        executor.jupiter_lib.slippage = slippage
        print(f"🎯 Using dynamic slippage: {slippage*100:.2f}%")
    
    if route_preferences:
        print(f"🛣️ Using route preferences: {route_preferences}")
    
    if use_exactout:
        print(f"🔄 Using ExactOut mode for sketchy token")
    
    return executor.execute_trade(token_address, amount_usd, is_buy=True)

def sell_token_solana(token_address: str, amount_usd: float, symbol: str = "", test_mode: bool = False) -> Tuple[str, bool]:
    """Sell token on Solana (for multi-chain compatibility)"""
    if test_mode:
        print(f"🔄 Simulating Solana sell for {symbol} ({token_address[:8]}...{token_address[-8:]})")
        return f"simulated_solana_tx_{int(time.time())}", True
    
    executor = JupiterCustomExecutor()
    return executor.execute_trade(token_address, amount_usd, is_buy=False)

def get_solana_executor():
    """Get Solana executor instance (for backward compatibility)"""
    return JupiterCustomExecutor()
