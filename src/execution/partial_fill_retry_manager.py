#!/usr/bin/env python3
"""
Partial Fill Retry Manager
Handles automatic retry of partially filled positions with safety checks
"""

import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from ..config.config_loader import get_config_bool, get_config_float, get_config_int, get_config
from ..monitoring.structured_logger import log_info, log_error
from ..storage.positions import load_positions, upsert_position
from ..utils.position_sync import create_position_key


class PartialFillRetryManager:
    """
    Manages automatic retry of partially filled positions
    """
    
    def __init__(self):
        self.enabled = get_config_bool("enable_partial_fill_retry", True)
        self.config = get_config("partial_fill_retry", {}) or {}
        
        # Load configuration with defaults
        self.min_unfilled_usd = get_config_float("partial_fill_retry.min_unfilled_usd", 5.0)
        self.min_fill_ratio = get_config_float("partial_fill_retry.min_fill_ratio", 0.95)
        self.max_retry_attempts = get_config_int("partial_fill_retry.max_retry_attempts", 2)
        self.max_time_window_seconds = get_config_int("partial_fill_retry.max_time_window_seconds", 300)
        self.require_price_stability = get_config_bool("partial_fill_retry.require_price_stability", True)
        self.max_price_change_pct = get_config_float("partial_fill_retry.max_price_change_pct", 0.05)
        self.check_risk_limits = get_config_bool("partial_fill_retry.check_risk_limits", True)
        self.check_token_tradeability = get_config_bool("partial_fill_retry.check_token_tradeability", True)
        self.retry_priority = self.config.get("retry_priority", "low")
        
    def detect_partial_fills(self) -> List[Dict[str, Any]]:
        """
        Scan open positions and detect partial fills that need retry
        
        Returns:
            List of positions eligible for retry with retry metadata
        """
        if not self.enabled:
            return []
        
        positions = load_positions()
        retry_candidates = []
        
        for position_key, position_data in positions.items():
            try:
                # Check if position has partial fill indicators
                intended_size = position_data.get("intended_position_size_usd")
                actual_size = position_data.get("position_size_usd", 0)
                
                # Skip if no intended size recorded (old positions)
                if not intended_size or intended_size <= 0:
                    continue
                
                # Calculate fill ratio
                fill_ratio = actual_size / intended_size if intended_size > 0 else 1.0
                unfilled_amount = intended_size - actual_size
                
                # Check if eligible for retry
                if self._is_eligible_for_retry(
                    position_key, 
                    position_data, 
                    intended_size, 
                    actual_size, 
                    fill_ratio, 
                    unfilled_amount
                ):
                    retry_candidates.append({
                        "position_key": position_key,
                        "position_data": position_data,
                        "intended_size": intended_size,
                        "actual_size": actual_size,
                        "unfilled_amount": unfilled_amount,
                        "fill_ratio": fill_ratio,
                        "retry_count": position_data.get("partial_fill_retry_count", 0)
                    })
                    
            except Exception as e:
                log_error("partial_fill_retry.detect_error",
                         f"Error detecting partial fill for {position_key}: {e}")
                continue
        
        if retry_candidates:
            log_info("partial_fill_retry.detected",
                    f"Found {len(retry_candidates)} positions eligible for retry",
                    {"count": len(retry_candidates)})
        
        return retry_candidates
    
    def _is_eligible_for_retry(
        self,
        position_key: str,
        position_data: Dict[str, Any],
        intended_size: float,
        actual_size: float,
        fill_ratio: float,
        unfilled_amount: float
    ) -> bool:
        """
        Check if a position is eligible for retry based on all safety conditions
        """
        # 1. Minimum fill ratio check - only retry if significant portion unfilled
        if fill_ratio >= self.min_fill_ratio:
            return False
        
        # 2. Minimum unfilled amount check
        if unfilled_amount < self.min_unfilled_usd:
            return False
        
        # 3. Retry attempt limit check
        retry_count = position_data.get("partial_fill_retry_count", 0)
        if retry_count >= self.max_retry_attempts:
            log_info("partial_fill_retry.max_attempts_reached",
                    f"Position {position_key} has reached max retry attempts ({retry_count})")
            return False
        
        # 4. Time window check
        timestamp_str = position_data.get("timestamp")
        if timestamp_str:
            try:
                if isinstance(timestamp_str, str):
                    entry_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')).timestamp()
                else:
                    entry_time = float(timestamp_str)
                
                time_since_fill = time.time() - entry_time
                if time_since_fill > self.max_time_window_seconds:
                    log_info("partial_fill_retry.time_window_expired",
                            f"Position {position_key} retry window expired ({time_since_fill:.0f}s)")
                    return False
            except Exception as e:
                log_error("partial_fill_retry.timestamp_error",
                         f"Error parsing timestamp for {position_key}: {e}")
                # If we can't parse timestamp, be conservative and skip
                return False
        
        # 5. Price stability check (if enabled)
        if self.require_price_stability:
            if not self._check_price_stability(position_data):
                log_info("partial_fill_retry.price_unstable",
                        f"Position {position_key} price moved too much, skipping retry")
                return False
        
        # 6. Risk limit check (if enabled)
        if self.check_risk_limits:
            if not self._check_risk_limits(unfilled_amount, position_data):
                log_info("partial_fill_retry.risk_limits_blocked",
                        f"Position {position_key} retry blocked by risk limits")
                return False
        
        # 7. Token tradeability check (if enabled)
        if self.check_token_tradeability:
            if not self._check_token_tradeability(position_data):
                log_info("partial_fill_retry.token_not_tradeable",
                        f"Position {position_key} token no longer tradeable")
                return False
        
        return True
    
    def _check_price_stability(self, position_data: Dict[str, Any]) -> bool:
        """Check if token price hasn't moved significantly since entry"""
        try:
            entry_price = float(position_data.get("entry_price", 0))
            if entry_price <= 0:
                return False
            
            # Get current price
            token_address = position_data.get("address")
            chain_id = position_data.get("chain_id", "solana")
            current_price = self._get_current_token_price(token_address, chain_id)
            
            if current_price is None or current_price <= 0:
                # If we can't get price, be conservative and allow retry
                # (price check failure shouldn't block retry)
                return True
            
            # Calculate price change
            price_change_pct = abs(current_price - entry_price) / entry_price
            return price_change_pct <= self.max_price_change_pct
            
        except Exception as e:
            log_error("partial_fill_retry.price_check_error", f"Error checking price stability: {e}")
            # On error, allow retry (fail open)
            return True
    
    def _get_current_token_price(self, token_address: str, chain_id: str) -> Optional[float]:
        """Get current token price"""
        try:
            if chain_id.lower() == "solana":
                from ..execution.solana_executor import get_token_price_usd
                return get_token_price_usd(token_address)
            elif chain_id.lower() in ["ethereum", "base", "arbitrum", "polygon"]:
                # For EVM chains, you may need to implement price fetching
                # For now, return None to allow retry (fail open)
                return None
            return None
        except Exception as e:
            log_error("partial_fill_retry.price_fetch_error", f"Error fetching price: {e}")
            return None
    
    def _check_risk_limits(self, unfilled_amount: float, position_data: Dict[str, Any]) -> bool:
        """Check if retry would violate risk limits"""
        try:
            from ..core.risk_manager import allow_new_trade
            
            token_address = position_data.get("address")
            chain_id = position_data.get("chain_id", "solana")
            
            allowed, reason = allow_new_trade(unfilled_amount, token_address, chain_id)
            return allowed
            
        except Exception as e:
            log_error("partial_fill_retry.risk_check_error", f"Error checking risk limits: {e}")
            # On error, allow retry (fail open)
            return True
    
    def _check_token_tradeability(self, position_data: Dict[str, Any]) -> bool:
        """Check if token is still tradeable"""
        try:
            token_address = position_data.get("address")
            chain_id = position_data.get("chain_id", "solana")
            
            # Basic check - verify we can get price (indicates token exists)
            current_price = self._get_current_token_price(token_address, chain_id)
            return current_price is not None and current_price > 0
            
        except Exception as e:
            log_error("partial_fill_retry.tradeability_check_error", f"Error checking tradeability: {e}")
            # On error, allow retry (fail open)
            return True
    
    def prepare_retry_token(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare token dict for retry execution
        
        Returns:
            Token dict formatted for trading engine
        """
        position_data = candidate["position_data"]
        unfilled_amount = candidate["unfilled_amount"]
        
        # Build token dict similar to what trading engine expects
        token = {
            "address": position_data.get("address"),
            "symbol": position_data.get("symbol", "?"),
            "chain": position_data.get("chain_id", "solana"),
            "priceUsd": position_data.get("entry_price", 0),
            "position_size_usd": unfilled_amount,  # Use unfilled amount as position size
            "recommended_position_size": unfilled_amount,
            "is_partial_fill_retry": True,
            "original_position_key": candidate["position_key"],
            "original_intended_size": candidate["intended_size"],
            "original_actual_size": candidate["actual_size"],
            "retry_attempt": candidate["retry_count"] + 1
        }
        
        return token
    
    def mark_retry_attempted(self, position_key: str, success: bool):
        """Mark that a retry attempt was made"""
        try:
            positions = load_positions()
            if position_key in positions:
                position_data = positions[position_key]
                retry_count = position_data.get("partial_fill_retry_count", 0)
                position_data["partial_fill_retry_count"] = retry_count + 1
                position_data["last_retry_attempt"] = datetime.now().isoformat()
                position_data["last_retry_success"] = success
                upsert_position(position_key, position_data)
        except Exception as e:
            log_error("partial_fill_retry.mark_error", f"Error marking retry attempt: {e}")
    
    def update_position_after_retry(
        self, 
        position_key: str, 
        retry_filled_amount: float,
        new_total_size: float
    ):
        """Update position size after successful retry"""
        try:
            positions = load_positions()
            if position_key in positions:
                position_data = positions[position_key]
                position_data["position_size_usd"] = new_total_size
                position_data["last_retry_filled"] = retry_filled_amount
                position_data["last_retry_update"] = datetime.now().isoformat()
                upsert_position(position_key, position_data)
                
                log_info("partial_fill_retry.position_updated",
                        f"Updated position {position_key} after retry: ${new_total_size:.2f}",
                        {
                            "position_key": position_key,
                            "new_total_size": new_total_size,
                            "retry_filled": retry_filled_amount
                        })
        except Exception as e:
            log_error("partial_fill_retry.update_error", f"Error updating position after retry: {e}")


# Global instance
_retry_manager_instance: Optional[PartialFillRetryManager] = None

def get_partial_fill_retry_manager() -> PartialFillRetryManager:
    """Get global partial fill retry manager instance"""
    global _retry_manager_instance
    if _retry_manager_instance is None:
        _retry_manager_instance = PartialFillRetryManager()
    return _retry_manager_instance

