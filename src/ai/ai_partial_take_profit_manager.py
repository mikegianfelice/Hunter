#!/usr/bin/env python3
"""
AI Partial Take-Profit Manager - Manages partial profit taking and trailing stops
Uses real position data and current prices to optimize exit timing
"""

import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

from ..config.config_loader import get_config, get_config_bool, get_config_float
from ..monitoring.structured_logger import log_info, log_error


@dataclass
class PartialTPAction:
    """Action to take for partial profit management"""
    type: str  # "sell" | "move_stop" | "none"
    size_pct: float  # Percentage of position to sell (0.0-1.0)
    reason: str
    new_stop_price: Optional[float] = None


class AIPartialTPManager:
    """
    Manages partial take-profit and trailing stops using real position data
    """
    
    def __init__(self):
        self.enabled = get_config_bool("enable_ai_partial_tp_manager", True)
        self.first_take_pct = get_config_float("partial_tp_manager.first_take_pct", 0.50)
        self.first_take_trigger_pct = get_config_float("partial_tp_manager.first_take_trigger_pct", 0.05)  # Default 5% (was 10%)
        self.trailing_stop_initial = get_config_float("partial_tp_manager.trailing_stop_initial", 0.04)  # Default 4% (was 6%)
        self.trail_tighten_trigger_pct = get_config_float("partial_tp_manager.trail_tighten_trigger_pct", 0.15)
        self.trailing_stop_tightened = get_config_float("partial_tp_manager.trailing_stop_tightened", 0.05)
        self.hard_stop_loss_pct = get_config_float("partial_tp_manager.hard_stop_loss_pct", 0.07)
        # Apply slippage buffer to hard stop loss
        stop_loss_slippage_buffer = get_config_float("stop_loss_slippage_buffer", 0.15)
        self.effective_hard_stop_loss_pct = self.hard_stop_loss_pct * (1 + stop_loss_slippage_buffer)
        
        # Track partial TP state per position
        self.position_states: Dict[str, Dict[str, Any]] = {}
        
    def evaluate_and_manage(
        self,
        position: Dict[str, Any],
        current_price: float,
        volatility_score: Optional[float] = None
    ) -> List[PartialTPAction]:
        """
        Evaluate position and return actions for partial TP management
        
        Args:
            position: Position dict with entry_price, position_size_usd, symbol, address, etc.
            current_price: Current token price in USD
            volatility_score: Optional volatility score (0-1)
            
        Returns:
            List of actions to take
        """
        if not self.enabled:
            return []
        
        actions = []
        position_key = self._get_position_key(position)
        entry_price = float(position.get("entry_price", 0))
        position_size_usd = float(position.get("position_size_usd", 0))
        
        if entry_price <= 0 or current_price <= 0:
            return actions
        
        # Calculate unrealized PnL
        pnl_pct = (current_price - entry_price) / entry_price
        state = self.position_states.get(position_key, {
            "partial_taken": False,
            "trailing_stop_active": False,
            "peak_price": entry_price,
            "trail_tightened": False
        })
        
        # Update peak price
        if current_price > state["peak_price"]:
            state["peak_price"] = current_price
        
        # Check hard stop loss first (using effective stop loss with slippage buffer)
        if pnl_pct <= -self.effective_hard_stop_loss_pct:
            # Log the threshold, not the actual loss (which may be worse due to slippage/delays)
            threshold_pct = self.hard_stop_loss_pct * 100
            effective_threshold_pct = self.effective_hard_stop_loss_pct * 100
            actions.append(PartialTPAction(
                type="sell",
                size_pct=1.0,
                reason=f"hard_stop_loss_{threshold_pct:.1f}%_triggered_at_{pnl_pct:.2%}_effective_{effective_threshold_pct:.1f}%"
            ))
            log_info("partial_tp.hard_stop",
                    symbol=position.get("symbol", "?"),
                    pnl_pct=pnl_pct,
                    entry_price=entry_price,
                    current_price=current_price)
            return actions
        
        # Check for first partial take-profit
        if not state["partial_taken"] and pnl_pct >= self.first_take_trigger_pct:
            actions.append(PartialTPAction(
                type="sell",
                size_pct=self.first_take_pct,
                reason=f"partial_tp_{pnl_pct:.2%}"
            ))
            state["partial_taken"] = True
            state["trailing_stop_active"] = True
            state["trailing_stop_price"] = current_price * (1 - self.trailing_stop_initial)
            log_info("partial_tp.first_take",
                    symbol=position.get("symbol", "?"),
                    pnl_pct=pnl_pct,
                    size_pct=self.first_take_pct,
                    entry_price=entry_price,
                    current_price=current_price)
        
        # Check for trail tightening
        if state["trailing_stop_active"] and not state["trail_tightened"]:
            if pnl_pct >= self.trail_tighten_trigger_pct:
                state["trail_tightened"] = True
                state["trailing_stop_price"] = current_price * (1 - self.trailing_stop_tightened)
                actions.append(PartialTPAction(
                    type="move_stop",
                    size_pct=0.0,
                    reason=f"tighten_trail_{pnl_pct:.2%}",
                    new_stop_price=state["trailing_stop_price"]
                ))
                log_info("partial_tp.tighten_trail",
                        symbol=position.get("symbol", "?"),
                        pnl_pct=pnl_pct,
                        new_stop_price=state["trailing_stop_price"])
        
        # Check trailing stop
        if state["trailing_stop_active"]:
            trailing_stop_price = state.get("trailing_stop_price", entry_price * (1 - self.trailing_stop_initial))
            
            # Update trailing stop if price moved up
            if current_price > state["peak_price"]:
                if state["trail_tightened"]:
                    trailing_stop_price = current_price * (1 - self.trailing_stop_tightened)
                else:
                    trailing_stop_price = current_price * (1 - self.trailing_stop_initial)
                state["trailing_stop_price"] = trailing_stop_price
                state["peak_price"] = current_price
            
            # Check if trailing stop hit
            if current_price <= trailing_stop_price:
                remaining_size_pct = 1.0 - (self.first_take_pct if state["partial_taken"] else 0.0)
                actions.append(PartialTPAction(
                    type="sell",
                    size_pct=remaining_size_pct,
                    reason=f"trailing_stop_{trailing_stop_price:.6f}"
                ))
                log_info("partial_tp.trail_hit",
                        symbol=position.get("symbol", "?"),
                        current_price=current_price,
                        stop_price=trailing_stop_price,
                        pnl_pct=pnl_pct)
        
        # Save state
        self.position_states[position_key] = state
        
        return actions
    
    def _get_position_key(self, position: Dict[str, Any]) -> str:
        """Get unique key for position"""
        address = position.get("address") or position.get("token_address", "")
        chain_id = position.get("chain_id", "solana")
        return f"{chain_id}:{address.lower()}"
    
    def clear_position_state(self, position_key: str):
        """Clear state when position is closed"""
        if position_key in self.position_states:
            del self.position_states[position_key]
    
    def get_position_state(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """Get current state for position"""
        position_key = self._get_position_key(position)
        return self.position_states.get(position_key, {
            "partial_taken": False,
            "trailing_stop_active": False,
            "peak_price": position.get("entry_price", 0),
            "trail_tightened": False
        })


# Global instance
_partial_tp_instance: Optional[AIPartialTPManager] = None

def get_partial_tp_manager() -> AIPartialTPManager:
    """Get global partial TP manager instance"""
    global _partial_tp_instance
    if _partial_tp_instance is None:
        _partial_tp_instance = AIPartialTPManager()
    return _partial_tp_instance

