#!/usr/bin/env python3
"""
Jupiter Custom Executor - Using custom Jupiter library for real trades
"""

import time
from typing import Tuple, Optional, Dict, Any
from .jupiter_lib import JupiterCustomLib
from .execution_policy import enforce_slippage_limit
from ..monitoring.structured_logger import log_info, log_error
from ..monitoring.metrics import record_trade_attempt, record_trade_failure, record_trade_success
from ..utils.idempotency import (
    build_trade_intent,
    register_trade_intent,
    mark_trade_intent_pending,
    mark_trade_intent_completed,
    mark_trade_intent_failed,
)
from ..config.config_validator import get_execution_config

from ..config.secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY
from ..config.config_loader import get_config

# Common token addresses
WSOL_MINT = "So11111111111111111111111111111111111111112"  # Wrapped SOL
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # USDC

CHAIN_NAME = "solana"

def get_solana_base_currency():
    """Get the configured base currency for Solana trading (SOL or USDC)"""
    return get_config("solana_base_currency", "USDC")

def get_min_sol_for_fees():
    """Get minimum SOL required for transaction fees"""
    return float(get_config("solana_min_sol_for_fees", 0.05))

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
                import os
                coingecko_id = token_mapping[token_address]
                coingecko_key = os.getenv("COINGECKO_API_KEY", "").strip()
                url = f"https://api.coingecko.com/api/v3/simple/price?ids={coingecko_id}&vs_currencies=usd"
                headers = {}
                if coingecko_key:
                    url += f"&api_key={coingecko_key}"
                    headers["x-cg-demo-api-key"] = coingecko_key
                data = get_json(url, headers=headers if headers else None, timeout=8, retries=1)
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
        print(f"🔍 [DEBUG] execute_trade called: token={token_address}, amount_usd={amount_usd}, is_buy={is_buy}")
        try:
            log_info("solana.trade.start", token=token_address, side=("buy" if is_buy else "sell"), amount_usd=amount_usd)
            
            # Get base currency configuration
            base_currency = get_solana_base_currency()
            min_sol_for_fees = get_min_sol_for_fees()
            
            # Balance gate for buys - prevent trading when insufficient balance
            if is_buy:
                try:
                    # Check SOL balance for transaction fees (always required)
                    available_sol = self.get_solana_balance()
                    if available_sol < min_sol_for_fees:
                        log_error("solana.trade.insufficient_sol_for_fees",
                                  token=token_address, available_sol=round(available_sol, 6),
                                  required_sol=round(min_sol_for_fees, 6))
                        return "", False
                    
                    # Check base currency balance for trading
                    if base_currency == "USDC":
                        # Check USDC balance
                        available_usdc = self.get_usdc_balance()
                        required_usd = float(amount_usd)
                        
                        # Require 5% buffer for slippage
                        buffer_pct = 0.05
                        required_with_buffer = required_usd * (1.0 + buffer_pct)
                        
                        if available_usdc < required_with_buffer:
                            log_error("solana.trade.insufficient_usdc_balance",
                                      token=token_address, available_usdc=round(available_usdc, 2),
                                      required_usdc=round(required_with_buffer, 2))
                            return "", False
                        
                        log_info("solana.trade.balance_check_passed", 
                                available_usdc=round(available_usdc, 2),
                                required_usdc=round(required_with_buffer, 2),
                                available_sol=round(available_sol, 6))
                    else:
                        # Legacy SOL-based trading (backward compatibility)
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
                        
                        available_usd = float(available_sol) * float(sol_price)
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
                # Determine base currency and set input/output mints
                if base_currency == "USDC":
                    # Buying token with USDC (1 USDC = $1, 6 decimals)
                    usdc_amount = int(float(amount_usd) * 1_000_000)  # USDC has 6 decimals
                    input_mint = USDC_MINT
                    output_mint = token_address
                    amount = usdc_amount
                    log_info("solana.trade.buy_with_usdc", amount_usd=amount_usd, usdc_amount=usdc_amount)
                else:
                    # Legacy: Buying token with SOL
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
                # Selling token
                # CRITICAL: We need the actual token balance in raw units, not USD amount
                # Get raw token balance (in smallest token units)
                print(f"🔍 [DEBUG] [SELL] Getting raw token balance for {token_address}...")
                raw_token_balance = self.get_token_raw_balance(token_address)
                print(f"🔍 [DEBUG] [SELL] Raw token balance result: {raw_token_balance}, type={type(raw_token_balance)}")
                
                if raw_token_balance is None:
                    print(f"❌ [ERROR] [SELL] Raw token balance is None - RPC call failed or token not found")
                    log_error("solana.trade.no_token_balance", token=token_address)
                    return "", False
                elif raw_token_balance <= 0:
                    print(f"❌ [ERROR] [SELL] Raw token balance is <= 0: {raw_token_balance}")
                    log_error("solana.trade.no_token_balance", token=token_address)
                    return "", False
                
                print(f"✅ [SELL] Raw token balance OK: {raw_token_balance}")
                input_mint = token_address
                
                # Determine output currency based on base currency configuration
                if base_currency == "USDC":
                    output_mint = USDC_MINT
                    log_info("solana.trade.sell_to_usdc", token=token_address)
                else:
                    output_mint = WSOL_MINT
                    log_info("solana.trade.sell_to_sol", token=token_address)
                
                amount = raw_token_balance  # Use raw token amount for swap
                print(f"🔍 [DEBUG] [SELL] Swap parameters: input_mint={input_mint}, output_mint={output_mint}, amount={amount}")
            
            # Execute swap using custom Jupiter library
            print(f"🔍 [DEBUG] Calling jupiter_lib.execute_swap...")
            tx_hash, success = self.jupiter_lib.execute_swap(input_mint, output_mint, amount)
            print(f"🔍 [DEBUG] execute_swap result: tx_hash={tx_hash}, success={success}, type(tx_hash)={type(tx_hash)}")
            
            if success:
                log_info("solana.trade.sent", token=token_address, side=("buy" if is_buy else "sell"), tx_hash=tx_hash)
                print(f"✅ [SUCCESS] Trade executed successfully: tx_hash={tx_hash}")
            else:
                log_error("solana.trade.error", token=token_address, side=("buy" if is_buy else "sell"))
                print(f"❌ [ERROR] Trade execution failed: tx_hash={tx_hash}")
            return tx_hash, success
            
        except Exception as e:
            print(f"❌ [EXCEPTION] Exception in execute_trade: {e}")
            import traceback
            print(f"🔍 [DEBUG] Exception traceback:\n{traceback.format_exc()}")
            log_error("solana.trade.exception", error=str(e))
            return "", False

    def get_solana_balance(self) -> float:
        """Get SOL balance"""
        return self.jupiter_lib.get_balance()
    
    def get_usdc_balance(self) -> Optional[float]:
        """
        Get USDC balance in USD (USDC has 6 decimals, 1 USDC = $1)
        
        Returns:
            float: USDC balance in USD
            None: If balance check failed after retries (rate limit or other error)
        """
        # Retry up to 3 times if we get None (rate limit errors)
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            balance = self.jupiter_lib.get_token_balance(USDC_MINT)
            
            if balance is not None:
                # Success - return the balance
                return float(balance)
            
            # If we got None, it might be a rate limit error
            if attempt < max_retries - 1:
                # Wait before retrying with exponential backoff
                wait_time = retry_delay * (2 ** attempt)
                print(f"⚠️ USDC balance check failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time:.1f}s...")
                import time
                time.sleep(wait_time)
            else:
                # All retries exhausted
                print(f"⚠️ Failed to get USDC balance after {max_retries} attempts (rate limit or RPC error)")
                # Return None to indicate failure (not zero balance)
                # This allows risk manager to handle gracefully
                return None
        
        return None

    def get_token_raw_balance(self, token_mint: str) -> Optional[int]:
        """
        Get raw token balance in smallest units (not UI amount)
        Returns the actual token amount needed for swap quotes
        """
        print(f"🔍 [DEBUG] get_token_raw_balance called: token_mint={token_mint}")
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
            
            print(f"🔍 [DEBUG] Sending RPC request to {SOLANA_RPC_URL}...")
            response = requests.post(SOLANA_RPC_URL, json=rpc_payload, timeout=10)
            print(f"🔍 [DEBUG] RPC response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"🔍 [DEBUG] RPC response result keys: {result.keys() if isinstance(result, dict) else 'not a dict'}")
                
                if "result" in result and "value" in result["result"]:
                    accounts = result["result"]["value"]
                    print(f"🔍 [DEBUG] Found {len(accounts)} token account(s)")
                    
                    if accounts:
                        # Get the first account's raw balance (in smallest units)
                        account_info = accounts[0]["account"]["data"]["parsed"]["info"]
                        raw_amount_str = account_info["tokenAmount"]["amount"]  # This is a string of the raw amount
                        raw_balance = int(raw_amount_str)
                        print(f"✅ [DEBUG] Raw token balance: {raw_balance} (from string: {raw_amount_str})")
                        return raw_balance
                    else:
                        print(f"⚠️ [DEBUG] No token accounts found for mint {token_mint}")
                        return 0
                else:
                    error_msg = result.get('error', 'Unknown')
                    print(f"❌ [ERROR] RPC error in result: {error_msg}")
                    log_error("solana.token_balance.rpc_error", error=error_msg)
                    return None
            else:
                print(f"❌ [ERROR] HTTP error: status={response.status_code}")
                log_error("solana.token_balance.http_error", status=response.status_code)
                return None
        except Exception as e:
            print(f"❌ [EXCEPTION] Error in get_token_raw_balance: {e}")
            import traceback
            print(f"🔍 [DEBUG] Exception traceback:\n{traceback.format_exc()}")
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
    """Buy token on Solana (for multi-chain compatibility) with guardrails and metrics."""
    trade_started = time.time()
    record_trade_attempt(CHAIN_NAME, "buy")

    execution_cfg = get_execution_config()
    default_slippage = execution_cfg.max_slippage_percent_by_chain.get(
        CHAIN_NAME, execution_cfg.max_slippage_percent
    )
    requested_slippage = slippage if slippage is not None else default_slippage
    effective_slippage = enforce_slippage_limit(CHAIN_NAME, requested_slippage)

    intent = build_trade_intent(
        chain=CHAIN_NAME,
        side="buy",
        token_address=token_address,
        symbol=symbol,
        quantity=float(amount_usd),
        metadata={"slippage": effective_slippage},
    )
    registered, existing = register_trade_intent(intent)
    if not registered:
        record_trade_failure(CHAIN_NAME, "buy", "duplicate_intent")
        log_warning_context = existing or {}
        log_info("solana.trade.duplicate", "Duplicate Solana buy intent skipped", context=log_warning_context)
        return "", False
    mark_trade_intent_pending(intent.intent_id)

    executor = JupiterCustomExecutor()
    executor.jupiter_lib.slippage = effective_slippage

    if route_preferences:
        log_info("solana.trade.route_preferences", route_preferences=route_preferences)
    if use_exactout:
        log_info("solana.trade.use_exactout", note="Using ExactOut mode for swap")

    def abort(reason: str, message: str, **context) -> Tuple[str, bool]:
        mark_trade_intent_failed(intent.intent_id, reason)
        latency_ms = int((time.time() - trade_started) * 1000)
        record_trade_failure(CHAIN_NAME, "buy", reason, latency_ms=latency_ms)
        log_error(f"solana.trade.{reason}", message, context=context)
        return "", False

    def succeed(tx_hash: Optional[str]) -> Tuple[str, bool]:
        mark_trade_intent_completed(intent.intent_id, tx_hash=tx_hash)
        latency_ms = int((time.time() - trade_started) * 1000)
        record_trade_success(
            CHAIN_NAME,
            "buy",
            latency_ms=latency_ms,
            slippage_bps=effective_slippage * 10000.0,
        )
        log_info("solana.trade.completed", token=token_address, symbol=symbol, tx_hash=tx_hash)
        return tx_hash or "", True

    if test_mode:
        try:
            try:
                from src.utils.utils import get_sol_price_usd
            except ImportError:
                from ..utils.utils import get_sol_price_usd
            sol_price = get_sol_price_usd()
            if sol_price <= 0:
                return abort("price_unavailable", "Cannot fetch SOL price for test mode validation")
            sol_amount = amount_usd / sol_price
            sol_amount_lamports = int(sol_amount * 1_000_000_000)
            quote = executor.jupiter_lib.get_quote(WSOL_MINT, token_address, sol_amount_lamports)
            if quote:
                log_info("solana.trade.test_mode_valid", token=token_address, symbol=symbol)
                return succeed(None)
            return abort("test_mode_quote_failed", "Test mode quote failed", token=token_address)
        except Exception as exc:
            return abort("test_mode_exception", "Test mode validation failed", error=str(exc))

    try:
        tx_hash, success = executor.execute_trade(token_address, amount_usd, is_buy=True)
    except Exception as exc:
        return abort("execution_exception", "Executor raised exception", error=str(exc))

    if success:
        return succeed(tx_hash)
    return abort("execution_failed", "Executor returned failure", token=token_address, amount_usd=amount_usd)

def sell_token_solana(token_address: str, amount_usd: float, symbol: str = "", test_mode: bool = False) -> Tuple[str, bool]:
    """Sell token on Solana (for multi-chain compatibility) with guardrails."""
    trade_started = time.time()
    record_trade_attempt(CHAIN_NAME, "sell")

    execution_cfg = get_execution_config()
    default_slippage = execution_cfg.max_slippage_percent_by_chain.get(
        CHAIN_NAME, execution_cfg.max_slippage_percent
    )
    effective_slippage = enforce_slippage_limit(CHAIN_NAME, default_slippage)

    intent = build_trade_intent(
        chain=CHAIN_NAME,
        side="sell",
        token_address=token_address,
        symbol=symbol,
        quantity=float(amount_usd),
        metadata={"slippage": effective_slippage},
    )
    registered, existing = register_trade_intent(intent)
    if not registered:
        record_trade_failure(CHAIN_NAME, "sell", "duplicate_intent")
        log_info("solana.trade.duplicate_sell", token=token_address, existing_intent=existing)
        return "", False
    mark_trade_intent_pending(intent.intent_id)

    executor = JupiterCustomExecutor()
    executor.jupiter_lib.slippage = effective_slippage

    def abort(reason: str, message: str, **context) -> Tuple[str, bool]:
        mark_trade_intent_failed(intent.intent_id, reason)
        latency_ms = int((time.time() - trade_started) * 1000)
        record_trade_failure(CHAIN_NAME, "sell", reason, latency_ms=latency_ms)
        log_error(f"solana.sell.{reason}", message, context=context)
        return "", False

    def succeed(tx_hash: Optional[str]) -> Tuple[str, bool]:
        mark_trade_intent_completed(intent.intent_id, tx_hash=tx_hash)
        latency_ms = int((time.time() - trade_started) * 1000)
        record_trade_success(
            CHAIN_NAME,
            "sell",
            latency_ms=latency_ms,
            slippage_bps=effective_slippage * 10000.0,
        )
        log_info("solana.sell.completed", token=token_address, symbol=symbol, tx_hash=tx_hash)
        return tx_hash or "", True

    if test_mode:
        try:
            usdc_amount = int(amount_usd * 1_000_000)
            quote = executor.jupiter_lib.get_quote(token_address, USDC_MINT, usdc_amount)
            if quote:
                log_info("solana.sell.test_mode_valid", token=token_address, amount_usd=amount_usd)
                return succeed(None)
            return abort("test_mode_quote_failed", "Test mode quote failed", token=token_address)
        except Exception as exc:
            return abort("test_mode_exception", "Test mode validation failed", error=str(exc))

    try:
        tx_hash, success = executor.execute_trade(token_address, amount_usd, is_buy=False)
    except Exception as exc:
        return abort("execution_exception", "Executor raised exception", error=str(exc))

    if success:
        return succeed(tx_hash)
    return abort("execution_failed", "Executor returned failure", token=token_address, amount_usd=amount_usd)

def get_solana_executor():
    """Get Solana executor instance (for backward compatibility)"""
    return JupiterCustomExecutor()
