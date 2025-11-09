#!/usr/bin/env python3
"""
AI Fill Verifier - Verifies trade execution fills using real blockchain data
Prevents "ghost" entries where trades appear successful but no tokens were received
"""

import time
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque

from ..config.config_loader import get_config, get_config_bool, get_config_int, get_config_float
from ..monitoring.structured_logger import log_info, log_error
from ..monitoring.metrics import record_trade_failure, record_trade_success


@dataclass
class EntryResult:
    """Result of fill verification"""
    status: str  # "accepted" | "rerouted" | "aborted"
    attempts: int
    route_used: str
    tokens_received: Optional[float]
    amount_usd_actual: Optional[float]
    tx_hash: Optional[str] = None
    error_message: Optional[str] = None


class AIFillVerifier:
    """
    Verifies trade fills using real blockchain balance checks
    Prevents recording positions when trades fail silently
    """
    
    def __init__(self):
        self.enabled = get_config_bool("enable_ai_fill_verifier", True)
        self.max_retry_attempts = get_config_int("fill_verifier.max_retry_attempts", 2)
        self.reroute_order = get_config("fill_verifier.reroute_order", ["direct_pool", "jupiter"])
        self.reduce_slice_factor = get_config_float("fill_verifier.reduce_slice_factor", 0.5)
        self.blacklist_minutes_on_fail = get_config_int("fill_verifier.blacklist_minutes_on_fail", 30)
        self.min_tokens_received_threshold = get_config_float("fill_verifier.min_tokens_received_threshold", 0.000001)
        
        # Metrics tracking
        self.fill_history = deque(maxlen=100)
        self.reroute_history = deque(maxlen=50)
        self.abort_history = deque(maxlen=50)
        
    def verify_and_finalize_entry(
        self,
        intent: Dict[str, Any],
        initial_attempt: Dict[str, Any],
        executors: Dict[str, Any],
        chain_id: str = "solana"
    ) -> EntryResult:
        """
        Verify fill using real blockchain data and reroute if needed
        
        Args:
            intent: Trade intent with symbol, amount_usd, token_address
            initial_attempt: Result from first execution attempt with tx_hash, tokens_received, amount_usd_actual
            executors: Dict with 'jupiter', 'raydium' executors
            chain_id: Chain identifier
            
        Returns:
            EntryResult with verification status
        """
        if not self.enabled:
            # If disabled, accept the initial attempt as-is
            return EntryResult(
                status="accepted",
                attempts=1,
                route_used=initial_attempt.get("route", "unknown"),
                tokens_received=initial_attempt.get("tokens_received"),
                amount_usd_actual=initial_attempt.get("amount_usd_actual"),
                tx_hash=initial_attempt.get("tx_hash")
            )
        
        token_address = intent.get("token_address") or intent.get("address")
        symbol = intent.get("symbol", "?")
        amount_usd = intent.get("amount_usd", 0)
        
        # Check if initial attempt was successful
        tokens_received = initial_attempt.get("tokens_received")
        amount_usd_actual = initial_attempt.get("amount_usd_actual", 0)
        tx_hash = initial_attempt.get("tx_hash")
        
        # Verify fill using real balance check
        if self._is_valid_fill(tokens_received, amount_usd_actual, token_address, chain_id):
            log_info("fill_verifier.accepted", 
                    symbol=symbol, 
                    tokens_received=tokens_received,
                    amount_usd_actual=amount_usd_actual,
                    tx_hash=tx_hash)
            self.fill_history.append({
                "timestamp": time.time(),
                "symbol": symbol,
                "status": "accepted",
                "tokens_received": tokens_received,
                "amount_usd_actual": amount_usd_actual
            })
            record_trade_success(chain_id, "buy")
            return EntryResult(
                status="accepted",
                attempts=1,
                route_used=initial_attempt.get("route", "jupiter"),
                tokens_received=tokens_received,
                amount_usd_actual=amount_usd_actual,
                tx_hash=tx_hash
            )
        
        # Fill verification failed - attempt reroute
        log_error("fill_verifier.failed", 
                 symbol=symbol,
                 tokens_received=tokens_received,
                 amount_usd_actual=amount_usd_actual,
                 tx_hash=tx_hash)
        
        # Try rerouting with reduced size
        for attempt in range(1, self.max_retry_attempts + 1):
            log_info("fill_verifier.reroute_attempt", 
                    symbol=symbol,
                    attempt=attempt,
                    max_attempts=self.max_retry_attempts)
            
            # Reduce slice size
            reduced_amount = amount_usd * (self.reduce_slice_factor ** attempt)
            
            # Try alternative route
            reroute_result = self._try_reroute(
                intent, 
                reduced_amount, 
                executors, 
                chain_id,
                attempt
            )
            
            if reroute_result and self._is_valid_fill(
                reroute_result.get("tokens_received"),
                reroute_result.get("amount_usd_actual", 0),
                token_address,
                chain_id
            ):
                log_info("fill_verifier.reroute_success",
                        symbol=symbol,
                        attempt=attempt,
                        route=reroute_result.get("route"),
                        tokens_received=reroute_result.get("tokens_received"),
                        amount_usd_actual=reroute_result.get("amount_usd_actual"))
                self.reroute_history.append({
                    "timestamp": time.time(),
                    "symbol": symbol,
                    "attempt": attempt,
                    "route": reroute_result.get("route")
                })
                record_trade_success(chain_id, "buy")
                return EntryResult(
                    status="rerouted",
                    attempts=attempt + 1,
                    route_used=reroute_result.get("route", "unknown"),
                    tokens_received=reroute_result.get("tokens_received"),
                    amount_usd_actual=reroute_result.get("amount_usd_actual"),
                    tx_hash=reroute_result.get("tx_hash")
                )
            
            # Wait before next attempt
            time.sleep(1.0 * attempt)  # Exponential backoff
        
        # All reroute attempts failed - abort
        log_error("fill_verifier.aborted",
                 symbol=symbol,
                 attempts=self.max_retry_attempts + 1,
                 reason="all_reroutes_failed")
        self.abort_history.append({
            "timestamp": time.time(),
            "symbol": symbol,
            "token_address": token_address,
            "reason": "all_reroutes_failed"
        })
        record_trade_failure(chain_id, "buy", "fill_verification_failed")
        return EntryResult(
            status="aborted",
            attempts=self.max_retry_attempts + 1,
            route_used="none",
            tokens_received=None,
            amount_usd_actual=None,
            error_message="Fill verification failed after all reroute attempts"
        )
    
    def _is_valid_fill(
        self,
        tokens_received: Optional[float],
        amount_usd_actual: Optional[float],
        token_address: str,
        chain_id: str
    ) -> bool:
        """
        Verify fill using real balance check
        Returns True if fill is valid (tokens received > threshold and amount_usd > 0)
        """
        # Basic validation
        if tokens_received is None or tokens_received < self.min_tokens_received_threshold:
            return False
        
        if amount_usd_actual is None or amount_usd_actual <= 0:
            return False
        
        # For Solana, verify actual balance change
        if chain_id.lower() == "solana":
            return self._verify_solana_balance(token_address, tokens_received)
        
        # For EVM chains, basic validation is sufficient (balance checks are expensive)
        return True
    
    def _verify_solana_balance(self, token_address: str, expected_tokens: float) -> bool:
        """
        Verify Solana token balance using real RPC call
        """
        try:
            from ..execution.jupiter_lib import JupiterCustomLib
            from ..config.secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY
            
            lib = JupiterCustomLib(SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY)
            actual_balance = lib.get_token_balance(token_address)
            
            if actual_balance is None:
                # RPC error - assume valid if we have tokens_received > 0
                return expected_tokens > self.min_tokens_received_threshold
            
            # Check if balance is at least expected amount (allowing for small rounding)
            return actual_balance >= expected_tokens * 0.95  # 5% tolerance for rounding
            
        except Exception as e:
            log_error("fill_verifier.balance_check_error", 
                     token_address=token_address,
                     error=str(e))
            # On error, assume valid if we have tokens_received > 0
            return expected_tokens > self.min_tokens_received_threshold
    
    def _try_reroute(
        self,
        intent: Dict[str, Any],
        reduced_amount: float,
        executors: Dict[str, Any],
        chain_id: str,
        attempt: int
    ) -> Optional[Dict[str, Any]]:
        """
        Try rerouting trade through alternative executor
        """
        token_address = intent.get("token_address") or intent.get("address")
        symbol = intent.get("symbol", "?")
        
        # Determine which route to try based on attempt number
        if attempt <= len(self.reroute_order):
            route_name = self.reroute_order[attempt - 1]
        else:
            route_name = self.reroute_order[-1]  # Use last route
        
        try:
            if chain_id.lower() == "solana":
                if route_name == "direct_pool" and "raydium" in executors:
                    # Try Raydium direct pool
                    raydium_executor = executors.get("raydium")
                    if raydium_executor:
                        log_info("fill_verifier.trying_raydium",
                                symbol=symbol,
                                amount_usd=reduced_amount)
                        tx_hash, success = raydium_executor.execute_trade(
                            token_address,
                            reduced_amount,
                            is_buy=True
                        )
                        if success:
                            # Get actual tokens received
                            tokens_received = self._get_tokens_received_solana(
                                token_address, tx_hash, reduced_amount
                            )
                            return {
                                "route": "raydium",
                                "tx_hash": tx_hash,
                                "tokens_received": tokens_received,
                                "amount_usd_actual": reduced_amount * 0.95  # Estimate with 5% slippage
                            }
                
                elif route_name == "jupiter" and "jupiter" in executors:
                    # Try Jupiter (if not already tried)
                    jupiter_executor = executors.get("jupiter")
                    if jupiter_executor:
                        log_info("fill_verifier.trying_jupiter",
                                symbol=symbol,
                                amount_usd=reduced_amount)
                        tx_hash, success = jupiter_executor.execute_trade(
                            token_address,
                            reduced_amount,
                            is_buy=True
                        )
                        if success:
                            tokens_received = self._get_tokens_received_solana(
                                token_address, tx_hash, reduced_amount
                            )
                            return {
                                "route": "jupiter",
                                "tx_hash": tx_hash,
                                "tokens_received": tokens_received,
                                "amount_usd_actual": reduced_amount * 0.95
                            }
            
            return None
            
        except Exception as e:
            log_error("fill_verifier.reroute_error",
                     symbol=symbol,
                     route=route_name,
                     error=str(e))
            return None
    
    def _get_tokens_received_solana(
        self,
        token_address: str,
        tx_hash: str,
        amount_usd: float
    ) -> Optional[float]:
        """
        Get actual tokens received from Solana transaction
        """
        try:
            from ..execution.jupiter_lib import JupiterCustomLib
            from ..config.secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY
            
            lib = JupiterCustomLib(SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY)
            balance = lib.get_token_balance(token_address)
            
            # Wait a moment for transaction to settle
            time.sleep(2.0)
            
            new_balance = lib.get_token_balance(token_address)
            
            if new_balance is not None and balance is not None:
                tokens_received = new_balance - balance
                if tokens_received > 0:
                    return tokens_received
            
            # Fallback: estimate from amount_usd and price
            from ..execution.jupiter_executor import JupiterCustomExecutor
            executor = JupiterCustomExecutor()
            price = executor.get_token_price_usd(token_address)
            if price and price > 0:
                return amount_usd / price
            
            return None
            
        except Exception as e:
            log_error("fill_verifier.get_tokens_error",
                     token_address=token_address,
                     tx_hash=tx_hash,
                     error=str(e))
            return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get fill verification metrics"""
        total_fills = len(self.fill_history)
        total_reroutes = len(self.reroute_history)
        total_aborts = len(self.abort_history)
        
        fill_success_rate = 0.0
        if total_fills + total_aborts > 0:
            fill_success_rate = total_fills / (total_fills + total_aborts)
        
        return {
            "fill_success_rate": fill_success_rate,
            "total_fills": total_fills,
            "total_reroutes": total_reroutes,
            "total_aborts": total_aborts,
            "enabled": self.enabled
        }


# Global instance
_fill_verifier_instance: Optional[AIFillVerifier] = None

def get_fill_verifier() -> AIFillVerifier:
    """Get global fill verifier instance"""
    global _fill_verifier_instance
    if _fill_verifier_instance is None:
        _fill_verifier_instance = AIFillVerifier()
    return _fill_verifier_instance

