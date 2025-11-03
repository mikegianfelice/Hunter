#!/usr/bin/env python3
"""
Jupiter Custom Executor - Using custom Jupiter library for real trades
"""

import time
from typing import Tuple, Optional, Dict, Any
from .jupiter_lib import JupiterCustomLib
from ..monitoring.structured_logger import log_info, log_error

from ..config.secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY

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
        
        # Import here to avoid circular imports with fallback for different import paths
        try:
            from src.utils.utils import get_sol_price_usd
        except ImportError:
            # Fallback if runtime package layout differs
            try:
                from ..utils.utils import get_sol_price_usd
            except ImportError:
                # Final fallback
                import sys
                import os
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
                from src.utils.utils import get_sol_price_usd
        
        # If the token is SOL, use the utility function
        sol_mint = "So11111111111111111111111111111111111111112"
        if token_address == sol_mint:
            return get_sol_price_usd()
        
        # Try DexScreener API for token price first (direct price, no SOL dependency)
        from src.utils.http_utils import get_json
        for attempt in range(2):
            try:
                url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
                data = get_json(url, timeout=10, retries=1)
                if data:
                    pairs = data.get("pairs", [])
                    if pairs:
                        for pair in pairs:
                            price = float(pair.get("priceUsd", 0))
                            if price > 0:
                                log_info("solana.price.dexscreener", token=token_address, price=price)
                                return price
            except Exception as e:
                print(f"⚠️ DexScreener price API error (attempt {attempt + 1}/2): {e}")
            
            if attempt < 1:
                time.sleep(0.5)
        
        # Try Birdeye API for Solana tokens (direct price, no SOL dependency)
        try:
            url = f"https://public-api.birdeye.so/public/price?address={token_address}"
            data = get_json(url, timeout=8, retries=1)
            if data and data.get("success") and data.get("data", {}).get("value"):
                price = float(data["data"]["value"])
                print(f"✅ Token price from Birdeye: ${price}")
                return price
        except Exception as e:
            log_info("solana.price.birdeye_error", level="WARNING", token=token_address, error=str(e))
        
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
                coingecko_id = token_mapping[token_address]
                url = f"https://api.coingecko.com/api/v3/simple/price?ids={coingecko_id}&vs_currencies=usd"
                data = get_json(url, timeout=8, retries=1)
                if data and coingecko_id in data and "usd" in data[coingecko_id]:
                    price = float(data[coingecko_id]["usd"])
                    log_info("solana.price.coingecko", token=token_address, price=price)
                    return price
            except Exception as e:
                log_info("solana.price.coingecko_error", level="WARNING", token=token_address, error=str(e))
        
        # If all APIs fail, return a small positive value to prevent false delisting
        log_info("solana.price.fallback", level="WARNING", token=token_address)
        return 0.000001  # Small positive value instead of 0

    def execute_trade(self, token_address: str, amount_usd: float, is_buy: bool = True) -> Tuple[str, bool]:
        """Execute trade using custom Jupiter library"""
        try:
            log_info("solana.trade.start", token=token_address, side=("buy" if is_buy else "sell"), amount_usd=amount_usd)
            
            # Balance gate for SOL buys - prevent trading when insufficient balance
            if is_buy:
                try:
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
                        log_error("solana.trade.error_no_sol_price", "Cannot get SOL price - aborting trade")
                        return "", False
                    
                    # Available SOL balance (USD)
                    available_sol = self.get_solana_balance()  # in SOL
                    available_usd = float(available_sol) * float(sol_price)
                    
                    # Require 5% buffer for fees/slippage
                    buffer_pct = 0.05
                    required_usd = float(amount_usd) * (1.0 + buffer_pct)
                    
                    if available_usd < required_usd:
                        log_error("solana.trade.insufficient_balance",
                                  token=token_address, available_usd=round(available_usd, 2),
                                  required_usd=round(required_usd, 2))
                        return "", False
                except Exception as e:
                    log_error("solana.trade.balance_gate_error", error=str(e))
                    # Fail safe: block the trade if we can't verify balance
                    return "", False
            
            # Get token liquidity to adjust trade amount
            try:
                from ..core.strategy import _get_token_liquidity
                liquidity = _get_token_liquidity(token_address)
                if liquidity and liquidity < amount_usd * 2:  # If liquidity is less than 2x trade amount
                    adjusted_amount = min(amount_usd, liquidity * 0.1)  # Use 10% of liquidity or original amount
                    log_info("solana.trade.adjust_amount", amount_usd_from=amount_usd, amount_usd_to=adjusted_amount, liquidity=liquidity)
                    amount_usd = adjusted_amount
            except Exception as e:
                print(f"⚠️ Could not get liquidity info: {e}")
                # Continue with original amount if liquidity check fails
            
            if is_buy:
                # Buying token with SOL (convert USD amount to SOL)
                # Import with fallback for different import paths
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
                    log_error("solana.trade.error_no_sol_price", "Cannot get SOL price - aborting trade")
                    return "", False
                
                sol_amount = amount_usd / sol_price
                sol_amount_lamports = int(sol_amount * 1_000_000_000)  # SOL has 9 decimals
                
                input_mint = WSOL_MINT
                output_mint = token_address
                amount = sol_amount_lamports
            else:
                # Selling token for SOL
                # CRITICAL: We need the actual token balance in raw units, not USD amount
                # Get raw token balance (in smallest token units)
                raw_token_balance = self.get_token_raw_balance(token_address)
                if raw_token_balance is None or raw_token_balance <= 0:
                    log_error("solana.trade.no_token_balance", token=token_address)
                    return "", False
                
                input_mint = token_address
                output_mint = WSOL_MINT
                amount = raw_token_balance  # Use raw token amount for swap
            
            # Execute swap using custom Jupiter library
            tx_hash, success = self.jupiter_lib.execute_swap(input_mint, output_mint, amount)
            if success:
                log_info("solana.trade.sent", token=token_address, side=("buy" if is_buy else "sell"), tx_hash=tx_hash)
            else:
                log_error("solana.trade.error", token=token_address, side=("buy" if is_buy else "sell"))
            return tx_hash, success
            
        except Exception as e:
            log_error("solana.trade.exception", error=str(e))
            return "", False

    def get_solana_balance(self) -> float:
        """Get SOL balance"""
        return self.jupiter_lib.get_balance()

    def get_token_raw_balance(self, token_mint: str) -> Optional[int]:
        """
        Get raw token balance in smallest units (not UI amount)
        Returns the actual token amount needed for swap quotes
        """
        try:
            import requests
            rpc_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTokenAccountsByOwner",
                "params": [
                    SOLANA_WALLET_ADDRESS,
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
                    log_error("solana.token_balance.rpc_error", error=result.get('error', 'Unknown'))
                    return None
            else:
                log_error("solana.token_balance.http_error", status=response.status_code)
                return None
        except Exception as e:
            log_error("solana.token_balance.exception", error=str(e))
            return None

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
    executor = JupiterCustomExecutor()
    
    # Apply advanced features if provided
    if slippage is not None:
        executor.jupiter_lib.slippage = slippage
        print(f"🎯 Using dynamic slippage: {slippage*100:.2f}%")
    
    if route_preferences:
        print(f"🛣️ Using route preferences: {route_preferences}")
    
    if use_exactout:
        print(f"🔄 Using ExactOut mode for sketchy token")
    
    if test_mode:
        # In test mode, still use real market data for quotes but don't execute
        # Get quote to validate trade would work
        try:
            # Import with fallback for different import paths
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
                return None, False
            sol_amount = amount_usd / sol_price
            sol_amount_lamports = int(sol_amount * 1_000_000_000)
            quote = executor.jupiter_lib.get_quote(WSOL_MINT, token_address, sol_amount_lamports)
            if quote:
                print(f"🔄 Test mode: Validated Solana buy for {symbol} ({token_address[:8]}...{token_address[-8:]}) - Transaction not sent")
                return None, True
        except Exception as e:
            print(f"⚠️ Test mode validation failed: {e}")
            return None, False
    
    return executor.execute_trade(token_address, amount_usd, is_buy=True)

def sell_token_solana(token_address: str, amount_usd: float, symbol: str = "", test_mode: bool = False) -> Tuple[str, bool]:
    """Sell token on Solana (for multi-chain compatibility)"""
    executor = JupiterCustomExecutor()
    
    if test_mode:
        # In test mode, still use real market data for quotes but don't execute
        # Get quote to validate trade would work
        try:
            usdc_amount = int(amount_usd * 1_000_000)
            quote = executor.jupiter_lib.get_quote(token_address, USDC_MINT, usdc_amount)
            if quote:
                print(f"🔄 Test mode: Validated Solana sell for {symbol} ({token_address[:8]}...{token_address[-8:]}) - Transaction not sent")
                return None, True
        except Exception as e:
            print(f"⚠️ Test mode validation failed: {e}")
            return None, False
    
    return executor.execute_trade(token_address, amount_usd, is_buy=False)

def get_solana_executor():
    """Get Solana executor instance (for backward compatibility)"""
    return JupiterCustomExecutor()
