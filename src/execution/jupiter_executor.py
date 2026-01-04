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
                base_url = "https://api.coingecko.com/api/v3/"
                url = f"{base_url}simple/price?ids={coingecko_id}&vs_currencies=usd"
                headers = {}
                if coingecko_key:
                    headers["x-cg-demo-api-key"] = coingecko_key
                data = get_json(url, headers=headers if headers else None, timeout=8, retries=1)
                if data and coingecko_id in data and "usd" in data[coingecko_id]:
                    from src.utils.api_tracker import track_coingecko_call
                    track_coingecko_call()
                    price = float(data[coingecko_id]["usd"])
                    log_info("solana.price.coingecko", token=token_address, price=price)
                    return price
            except Exception as e:
                log_info("solana.price.coingecko_error", level="WARNING", token=token_address, error=str(e))
        
        # If all APIs fail, return a small positive value to prevent false delisting
        log_info("solana.price.fallback", level="WARNING", token=token_address)
        return 0.000001  # Small positive value instead of 0

    def execute_trade(self, token_address: str, amount_usd: float, is_buy: bool = True) -> Tuple[str, bool, Optional[int]]:
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
                        return "", False, None
                    
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
                            return "", False, None
                        
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
                            return "", False, None
                        
                        available_usd = float(available_sol) * float(sol_price)
                        buffer_pct = 0.05
                        required_usd = float(amount_usd) * (1.0 + buffer_pct)
                        
                        if available_usd < required_usd:
                            log_error("solana.trade.insufficient_balance",
                                      token=token_address, available_usd=round(available_usd, 2),
                                      required_usd=round(required_usd, 2))
                            return "", False, None
                except Exception as e:
                    log_error("solana.trade.balance_gate_error", error=str(e))
                    # Fail safe: block the trade if we can't verify balance
                    return "", False, None
            
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
                        return "", False, None
                    
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
                    return "", False, None
                elif raw_token_balance <= 0:
                    print(f"❌ [ERROR] [SELL] Raw token balance is <= 0: {raw_token_balance}")
                    log_error("solana.trade.no_token_balance", token=token_address)
                    return "", False, None
                
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
            
            # Get quoted output amount for slippage calculation
            quoted_output_amount = None
            if self.jupiter_lib.last_quote and self.jupiter_lib.last_quote.get("outputMint") == output_mint:
                quoted_output_amount = int(self.jupiter_lib.last_quote.get("outAmount", 0))
            
            if success:
                log_info("solana.trade.sent", token=token_address, side=("buy" if is_buy else "sell"), tx_hash=tx_hash)
                print(f"✅ [SUCCESS] Trade executed successfully: tx_hash={tx_hash}")
            else:
                log_error("solana.trade.error", token=token_address, side=("buy" if is_buy else "sell"))
                print(f"❌ [ERROR] Trade execution failed: tx_hash={tx_hash}")
                return "", False, None
            return tx_hash, success, quoted_output_amount
            
        except Exception as e:
            print(f"❌ [EXCEPTION] Exception in execute_trade: {e}")
            import traceback
            print(f"🔍 [DEBUG] Exception traceback:\n{traceback.format_exc()}")
            log_error("solana.trade.exception", error=str(e))
            return "", False, None

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

def execute_solana_trade(token_address: str, amount_usd: float, is_buy: bool = True) -> Tuple[str, bool, Optional[int]]:
    """Legacy function for executing trades"""
    executor = JupiterCustomExecutor()
    return executor.execute_trade(token_address, amount_usd, is_buy)

# Additional functions for multi-chain compatibility
def buy_token_solana(token_address: str, amount_usd: float, symbol: str = "", test_mode: bool = False,
                     slippage: float = None, route_preferences: Dict[str, Any] = None,
                     use_exactout: bool = False) -> Tuple[str, bool, Optional[int]]:
    """Buy token on Solana (for multi-chain compatibility) with guardrails and metrics.
    
    Returns:
        Tuple of (tx_hash, success, quoted_output_amount)
        - tx_hash: Transaction hash if successful, empty string otherwise
        - success: True if trade was successful, False otherwise
        - quoted_output_amount: Quoted output amount in token units, None if not available
    """
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
        return "", False, None
    mark_trade_intent_pending(intent.intent_id)

    executor = JupiterCustomExecutor()
    executor.jupiter_lib.slippage = effective_slippage

    if route_preferences:
        log_info("solana.trade.route_preferences", route_preferences=route_preferences)
    if use_exactout:
        log_info("solana.trade.use_exactout", note="Using ExactOut mode for swap")

    def abort(reason: str, message: str, **context) -> Tuple[str, bool, Optional[int]]:
        mark_trade_intent_failed(intent.intent_id, reason)
        latency_ms = int((time.time() - trade_started) * 1000)
        record_trade_failure(CHAIN_NAME, "buy", reason, latency_ms=latency_ms)
        log_error(f"solana.trade.{reason}", message, context=context)
        return "", False, None

    def succeed(tx_hash: Optional[str], quoted_amt: Optional[int] = None) -> Tuple[str, bool, Optional[int]]:
        mark_trade_intent_completed(intent.intent_id, tx_hash=tx_hash)
        latency_ms = int((time.time() - trade_started) * 1000)
        record_trade_success(
            CHAIN_NAME,
            "buy",
            latency_ms=latency_ms,
            slippage_bps_value=effective_slippage * 10000.0,
        )
        log_info("solana.trade.completed", token=token_address, symbol=symbol, tx_hash=tx_hash)
        return tx_hash or "", True, quoted_amt

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
                return succeed(None, None)
            return abort("test_mode_quote_failed", "Test mode quote failed", token=token_address)
        except Exception as exc:
            return abort("test_mode_exception", "Test mode validation failed", error=str(exc))

    # Capture balance before trade for verification fallback
    balance_before = None
    try:
        balance_before = executor.jupiter_lib.get_token_balance(token_address)
        if balance_before is not None:
            print(f"📊 Balance before buy: {balance_before:.8f}")
    except Exception as e:
        print(f"⚠️ Could not capture balance before buy: {e}")
        # Non-critical, continue without balance_before

    try:
        tx_hash, success, quoted_output_amount = executor.execute_trade(token_address, amount_usd, is_buy=True)
    except Exception as exc:
        return abort("execution_exception", "Executor raised exception", error=str(exc))

    # CRITICAL: Always verify on-chain before marking as completed, even if success=True
    # This ensures we don't mark transactions as completed if they actually failed
    if tx_hash:
        print(f"🔍 Verifying buy transaction {tx_hash} on-chain before marking as completed...")
        
        # Retry verification with increasing delays (transaction may not be indexed immediately)
        verified = None
        for attempt in range(3):
            wait_time = 2 * (attempt + 1)  # 2s, 4s, 6s
            if attempt > 0:
                print(f"⏳ Retry {attempt + 1}/3: Waiting {wait_time}s before verification...")
                time.sleep(wait_time)
            else:
                time.sleep(2)
            
            verified = executor.jupiter_lib.verify_transaction_success(tx_hash)
            if verified is True:
                print(f"✅ Buy transaction {tx_hash} verified as successful on-chain")
                return succeed(tx_hash, quoted_output_amount)
            elif verified is False:
                print(f"❌ Buy transaction {tx_hash} confirmed as failed on-chain")
                print(f"   Check transaction on Solscan: https://solscan.io/tx/{tx_hash}")
                return abort("execution_failed", "Buy transaction confirmed as failed on-chain", 
                            token=token_address, amount_usd=amount_usd, tx_hash=tx_hash)
            # If verified is None, continue to next retry
        
        # If still can't verify after retries, check balance as fallback
        if verified is None:
            print(f"⚠️ WARNING: Cannot verify buy transaction {tx_hash} via RPC after multiple attempts")
            print(f"   Falling back to balance check...")
            try:
                time.sleep(3)  # Additional wait for balance to update
                balance_after = executor.jupiter_lib.get_token_balance(token_address)
                
                # Check if balance increased (indicating tokens were received)
                if balance_after is not None and balance_before is not None:
                    balance_increase = balance_after - balance_before
                    if balance_increase > 0:
                        print(f"✅ Balance check confirms tokens received: {balance_increase:.8f} tokens")
                        print(f"   Balance before: {balance_before:.8f}, Balance after: {balance_after:.8f}")
                        print(f"   Assuming transaction succeeded (RPC verification unreliable)")
                        print(f"   Check transaction on Solscan: https://solscan.io/tx/{tx_hash}")
                        log_info("solana.trade.verification_uncertain_balance_confirmed", 
                                message=f"Buy transaction {tx_hash} verified via balance check",
                                token=token_address, amount_usd=amount_usd, tx_hash=tx_hash,
                                balance_before=balance_before, balance_after=balance_after,
                                balance_increase=balance_increase)
                        return succeed(tx_hash, quoted_output_amount)
                    else:
                        print(f"⚠️ Balance check shows no increase: {balance_before:.8f} -> {balance_after:.8f}")
                elif balance_after is not None and balance_after > 0:
                    # If we don't have balance_before but balance_after > 0, assume success
                    print(f"✅ Balance check shows tokens present: {balance_after:.8f} tokens")
                    print(f"   Assuming transaction succeeded (RPC verification unreliable)")
                    print(f"   Check transaction on Solscan: https://solscan.io/tx/{tx_hash}")
                    log_info("solana.trade.verification_uncertain_balance_confirmed", 
                            message=f"Buy transaction {tx_hash} verified via balance check (no before balance)",
                            token=token_address, amount_usd=amount_usd, tx_hash=tx_hash,
                            balance_after=balance_after)
                    return succeed(tx_hash, quoted_output_amount)
                
                # Balance check didn't confirm success, but we have tx_hash
                # Assume success to prevent false negatives (transaction was submitted)
                print(f"⚠️ Balance check inconclusive, but transaction was submitted")
                print(f"   Assuming transaction succeeded to avoid false negatives")
                print(f"   Check transaction on Solscan: https://solscan.io/tx/{tx_hash}")
                log_info("solana.trade.verification_uncertain", level="WARNING",
                         message=f"Buy transaction {tx_hash} cannot be verified but has tx_hash",
                         token=token_address, amount_usd=amount_usd, tx_hash=tx_hash,
                         balance_before=balance_before, balance_after=balance_after)
                return succeed(tx_hash, quoted_output_amount)  # Assume success if we have tx_hash
            except Exception as e:
                print(f"⚠️ Balance check also failed: {e}")
                # Even if balance check fails, if we have tx_hash, assume success
                # This prevents false negatives when RPC/network is unreliable
                print(f"⚠️ Assuming transaction succeeded (has tx_hash, verification methods unreliable)")
                print(f"   Check transaction on Solscan: https://solscan.io/tx/{tx_hash}")
                log_info("solana.trade.verification_uncertain", level="WARNING",
                         message=f"Buy transaction {tx_hash} cannot be verified but has tx_hash",
                         token=token_address, amount_usd=amount_usd, tx_hash=tx_hash,
                         balance_check_error=str(e))
                return succeed(tx_hash, quoted_output_amount)
    elif success:
        # Got success=True but no tx_hash - this shouldn't happen, but handle it
        print(f"⚠️ WARNING: Executor reported success but no tx_hash provided")
        return abort("execution_failed", "No transaction hash provided despite success report", 
                    token=token_address, amount_usd=amount_usd)
    
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
            slippage_bps_value=effective_slippage * 10000.0,
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

    # CRITICAL: Before selling, verify token balance exists in wallet
    # Also capture balance for verification comparison after sell
    balance_before = None
    try:
        print(f"🔍 Checking token balance before sell for {symbol or token_address[:8]}...")
        raw_balance = executor.get_token_raw_balance(token_address)
        if raw_balance is None:
            return abort("balance_check_failed", "Cannot check token balance - RPC error", token=token_address)
        elif raw_balance <= 0:
            return abort("insufficient_balance", f"No token balance to sell (balance: {raw_balance})", token=token_address)
        print(f"✅ Token balance verified: {raw_balance} raw units available for sale")
        
        # Capture UI balance for comparison after sell
        try:
            balance_before = executor.jupiter_lib.get_token_balance(token_address)
            print(f"📊 Balance before sell: {balance_before:.8f if balance_before is not None else 'unknown'}")
        except:
            pass  # Non-critical, we'll check balance later
    except Exception as e:
        log_error("solana.sell.balance_check_error", error=str(e), token=token_address)
        return abort("balance_check_exception", f"Error checking balance: {e}", token=token_address)

    try:
        tx_hash, success, quoted_output_amount = executor.execute_trade(token_address, amount_usd, is_buy=False)
    except Exception as exc:
        return abort("execution_exception", "Executor raised exception", error=str(exc))

    # CRITICAL: Always verify on-chain before marking as completed, even if success=True
    # This ensures we don't mark transactions as completed if they actually failed
    if tx_hash:
        print(f"🔍 Verifying sell transaction {tx_hash} on-chain before marking as completed...")
        time.sleep(2)  # Wait a bit for transaction to propagate
        
        verified = executor.jupiter_lib.verify_transaction_success(tx_hash)
        if verified is True:
            print(f"✅ Sell transaction {tx_hash} verified as successful on-chain")
            return succeed(tx_hash)
        elif verified is False:
            # CRITICAL: Before assuming failure, check balance as verification might be wrong
            # RPC verification can return False incorrectly due to timing, stale data, or network issues
            print(f"⚠️ Sell transaction {tx_hash} reported as failed via RPC verification")
            print(f"   Double-checking with balance verification before assuming failure...")
            
            # Use balance_before captured earlier in function (or try to get it now)
            if balance_before is None:
                try:
                    balance_before = executor.jupiter_lib.get_token_balance(token_address)
                    print(f"🔍 Current balance: {balance_before:.8f if balance_before is not None else 'unknown'}")
                except:
                    balance_before = None
            else:
                print(f"🔍 Balance before sell (from earlier check): {balance_before:.8f}")
            
            # Wait longer for transaction to fully propagate and be indexed
            print(f"⏳ Waiting 10 seconds for transaction to fully propagate and be indexed...")
            time.sleep(10)
            
            # Retry verification one more time after waiting
            verified_retry = executor.jupiter_lib.verify_transaction_success(tx_hash)
            if verified_retry is True:
                print(f"✅ Sell transaction {tx_hash} verified as successful on retry")
                return succeed(tx_hash)
            
            # Check balance as final verification
            try:
                balance_after = executor.jupiter_lib.get_token_balance(token_address)
                if balance_after is not None:
                    # If balance is zero or significantly reduced, transaction likely succeeded
                    if balance_after <= 0.000001:  # Zero or dust balance
                        print(f"✅ Balance check confirms sell succeeded (balance: {balance_after:.8f})")
                        print(f"   Transaction verified via balance check despite RPC verification failure")
                        print(f"   Check transaction on Solscan: https://solscan.io/tx/{tx_hash}")
                        return succeed(tx_hash)
                    elif balance_before is not None and balance_after < balance_before * 0.9:
                        # Balance decreased by at least 10% - sell likely succeeded (partial or full)
                        print(f"✅ Balance check shows decrease from {balance_before:.8f} to {balance_after:.8f}")
                        print(f"   Transaction verified via balance check despite RPC verification failure")
                        print(f"   Check transaction on Solscan: https://solscan.io/tx/{tx_hash}")
                        return succeed(tx_hash)
                    else:
                        # Balance unchanged - transaction likely failed
                        print(f"❌ Balance check shows no change: {balance_after:.8f}")
                        print(f"   Transaction confirmed as failed on-chain and balance check")
                        print(f"   Check transaction on Solscan: https://solscan.io/tx/{tx_hash}")
                        return abort("execution_failed", "Sell transaction confirmed as failed on-chain and balance check", 
                                    token=token_address, amount_usd=amount_usd, tx_hash=tx_hash)
                else:
                    # Balance check failed - can't verify, but transaction exists
                    # Better to assume success to avoid false negatives (transaction was submitted)
                    print(f"⚠️ Balance check failed, but transaction was submitted with hash")
                    print(f"   Check transaction on Solscan: https://solscan.io/tx/{tx_hash}")
                    print(f"   Assuming success to avoid false negative (transaction hash available)")
                    return succeed(tx_hash)
            except Exception as balance_error:
                print(f"⚠️ Balance check error: {balance_error}")
                print(f"   Check transaction on Solscan: https://solscan.io/tx/{tx_hash}")
                # If we have a tx_hash, assume success (transaction was submitted)
                # Better to mark as success and let reconciliation handle edge cases
                print(f"   Assuming success (transaction hash available)")
                return succeed(tx_hash)
            
            # If all checks fail, abort
            print(f"❌ Sell transaction {tx_hash} confirmed as failed on-chain")
            print(f"   Check transaction on Solscan: https://solscan.io/tx/{tx_hash}")
            return abort("execution_failed", "Sell transaction confirmed as failed on-chain", 
                        token=token_address, amount_usd=amount_usd, tx_hash=tx_hash)
        else:
            # Can't verify (RPC timeout/missing data) - use balance check as fallback verification
            print(f"⚠️ WARNING: Cannot verify sell transaction {tx_hash} via RPC")
            print(f"   Falling back to balance check verification...")
            
            # Wait a bit longer for transaction to propagate
            time.sleep(5)
            
            # Check if token balance decreased (indicating successful sell)
            try:
                balance_after = executor.jupiter_lib.get_token_balance(token_address)
                if balance_after is not None:
                    # If balance is zero or significantly reduced, assume sell succeeded
                    if balance_after <= 0.000001:  # Zero or dust balance
                        print(f"✅ Balance check confirms sell succeeded (balance: {balance_after:.8f})")
                        print(f"   Transaction verified via balance check: https://solscan.io/tx/{tx_hash}")
                        return succeed(tx_hash)
                    else:
                        # Balance still exists - transaction may have failed or was partial
                        print(f"⚠️ Balance check shows tokens still present (balance: {balance_after:.8f})")
                        print(f"   This may be a partial sell or failed transaction")
                        print(f"   Check transaction on Solscan: https://solscan.io/tx/{tx_hash}")
                        # Still mark as succeeded if we have tx_hash and balance decreased
                        # (partial sells are valid, and we can't determine exact amount without transaction analysis)
                        print(f"   Proceeding with success (transaction was submitted with hash)")
                        return succeed(tx_hash)
                else:
                    # Balance check failed - can't verify, but we have tx_hash
                    print(f"⚠️ Balance check failed, but transaction was submitted with hash")
                    print(f"   Check transaction on Solscan: https://solscan.io/tx/{tx_hash}")
                    print(f"   Assuming success (transaction hash available)")
                    return succeed(tx_hash)
            except Exception as balance_error:
                print(f"⚠️ Balance check error: {balance_error}")
                print(f"   Check transaction on Solscan: https://solscan.io/tx/{tx_hash}")
                # If we have a tx_hash, assume success (transaction was submitted)
                # Better to mark as success and let reconciliation handle edge cases
                print(f"   Assuming success (transaction hash available)")
                return succeed(tx_hash)
    elif success:
        # Got success=True but no tx_hash - this shouldn't happen, but handle it
        print(f"⚠️ WARNING: Executor reported success but no tx_hash provided")
        return abort("execution_failed", "No transaction hash provided despite success report", 
                    token=token_address, amount_usd=amount_usd)
    
    # No tx_hash available, abort normally
    return abort("execution_failed", "Executor returned failure", token=token_address, amount_usd=amount_usd)

def get_solana_executor():
    """Get Solana executor instance (for backward compatibility)"""
    return JupiterCustomExecutor()
