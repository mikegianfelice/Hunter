#!/usr/bin/env python3
"""
AI Time-Window Scheduler - Gates trade entries based on execution quality metrics
Uses real execution data to determine optimal trading windows
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import statistics

from ..config.config_loader import get_config, get_config_bool, get_config_float, get_config_int
from ..monitoring.structured_logger import log_info, log_error
from ..monitoring.metrics import trade_success, trade_failures, trade_latency_ms, slippage_bps


@dataclass
class WindowDecision:
    """Decision from time window scheduler"""
    should_trade: bool
    score: float  # 0.0-1.0
    next_check_in_s: int
    reason: str


# State persistence path
SCHEDULER_STATE_PATH = Path("data/time_window_scheduler_state.json")


class AITimeWindowScheduler:
    """
    Schedules trades based on execution quality metrics
    Gates entries during low-quality windows to improve fill rates
    """
    
    def __init__(self):
        self.enabled = get_config_bool("enable_ai_time_window_scheduler", True)
        self.min_window_score_to_trade = get_config_float("time_window_scheduler.min_window_score_to_trade", 0.60)
        self.pause_on_volatility_spike = get_config_bool("time_window_scheduler.pause_on_volatility_spike", True)
        self.volatility_spike_threshold = get_config_float("time_window_scheduler.volatility_spike_threshold", 0.70)
        self.review_interval_seconds = get_config_int("time_window_scheduler.review_interval_seconds", 300)
        
        # Track execution metrics over time windows
        self.window_size_seconds = 300  # 5 minute windows
        self.execution_history: deque = deque(maxlen=100)
        
        # Load persisted state or use defaults
        self._load_state()
        
        # Ensure minimum floor is at least equal to threshold to prevent blocking at floor
        # This prevents the score from getting stuck at 0.50 when threshold is 0.55
        min_score_floor = max(0.50, self.min_window_score_to_trade - 0.05)  # Floor is threshold - 5%
        if self.current_window_score < min_score_floor:
            log_info("time_window_scheduler.score_adjusted",
                    old_score=self.current_window_score,
                    new_score=min_score_floor,
                    reason="below_minimum_floor")
            self.current_window_score = min_score_floor
        
    def should_trade_now(
        self,
        recent_exec_metrics: Optional[Dict[str, Any]] = None,
        market_quality_metrics: Optional[Dict[str, Any]] = None
    ) -> WindowDecision:
        """
        Determine if trading should proceed based on current window quality
        
        Args:
            recent_exec_metrics: Optional dict with fill_success_rate, avg_slippage, etc.
            market_quality_metrics: Optional dict with volatility, liquidity, etc.
            
        Returns:
            WindowDecision with should_trade flag and score
        """
        if not self.enabled:
            return WindowDecision(
                should_trade=True,
                score=1.0,
                next_check_in_s=self.review_interval_seconds,
                reason="scheduler_disabled"
            )
        
        current_time = time.time()
        
        # Check if we're in a pause period
        if self.is_paused and current_time < self.pause_until:
            remaining_pause = int(self.pause_until - current_time)
            return WindowDecision(
                should_trade=False,
                score=0.0,
                next_check_in_s=min(remaining_pause, self.review_interval_seconds),
                reason=f"paused_until_{remaining_pause}s"
            )
        
        # Force recalculation on first call (when last_review_time is 0.0) or if score is still at default
        # This ensures we get a proper score calculation even if we haven't hit the review interval yet
        force_recalculate = (self.last_review_time == 0.0) or (self.current_window_score == 0.0)
        
        # Update window score periodically or on first call
        if force_recalculate or current_time - self.last_review_time >= self.review_interval_seconds:
            old_score = self.current_window_score
            self.current_window_score = self._calculate_window_score(
                recent_exec_metrics,
                market_quality_metrics
            )
            self.last_review_time = current_time
            
            # Persist state after score update
            self._save_state()
            
            # Log score changes for debugging
            if abs(old_score - self.current_window_score) > 0.05:
                log_info("time_window_scheduler.score_changed",
                        old_score=old_score,
                        new_score=self.current_window_score,
                        threshold=self.min_window_score_to_trade)
            
            # Check for volatility spike
            if self.pause_on_volatility_spike and market_quality_metrics:
                volatility = market_quality_metrics.get("volatility", 0.0)
                if volatility >= self.volatility_spike_threshold:
                    self.is_paused = True
                    self.pause_until = current_time + (self.review_interval_seconds * 2)  # Pause for 2 review intervals
                    log_info("time_window_scheduler.pause_volatility",
                            volatility=volatility,
                            threshold=self.volatility_spike_threshold,
                            pause_until=self.pause_until)
                    self._save_state()
                    return WindowDecision(
                        should_trade=False,
                        score=0.0,
                        next_check_in_s=self.review_interval_seconds,
                        reason=f"volatility_spike_{volatility:.2f}"
                    )
            
            # Clear pause if conditions improved
            if self.is_paused and current_time >= self.pause_until:
                self.is_paused = False
                log_info("time_window_scheduler.resume", score=self.current_window_score)
                self._save_state()
        
        # Make decision based on window score
        should_trade = self.current_window_score >= self.min_window_score_to_trade
        
        if not should_trade:
            log_info("time_window_scheduler.blocked",
                    score=self.current_window_score,
                    threshold=self.min_window_score_to_trade)
        
        return WindowDecision(
            should_trade=should_trade,
            score=self.current_window_score,
            next_check_in_s=self.review_interval_seconds,
            reason=f"window_score_{self.current_window_score:.2f}"
        )
    
    def _calculate_window_score(
        self,
        recent_exec_metrics: Optional[Dict[str, Any]],
        market_quality_metrics: Optional[Dict[str, Any]]
    ) -> float:
        """
        Calculate window quality score (0.0-1.0) based on execution metrics
        with automatic recovery mechanism
        """
        score_components = []
        weights = []
        
        # 1. Fill success rate (40% weight)
        fill_success_rate = self._get_fill_success_rate(recent_exec_metrics)
        score_components.append(fill_success_rate)
        weights.append(0.40)
        
        # 2. Average slippage (30% weight) - lower is better
        avg_slippage = self._get_avg_slippage(recent_exec_metrics)
        slippage_score = max(0.0, 1.0 - (avg_slippage / 0.05))  # Normalize to 0-1, 5% slippage = 0 score
        score_components.append(slippage_score)
        weights.append(0.30)
        
        # 3. Execution latency (15% weight) - lower is better
        avg_latency = self._get_avg_latency(recent_exec_metrics)
        latency_score = max(0.0, 1.0 - (avg_latency / 5000.0))  # Normalize to 0-1, 5s latency = 0 score
        score_components.append(latency_score)
        weights.append(0.15)
        
        # 4. Market quality (15% weight)
        market_score = self._get_market_quality_score(market_quality_metrics)
        score_components.append(market_score)
        weights.append(0.15)
        
        # Weighted average
        if sum(weights) > 0:
            weighted_score = sum(s * w for s, w in zip(score_components, weights)) / sum(weights)
        else:
            weighted_score = 0.5  # Default neutral score
        
        # Automatic recovery mechanism: if no recent bad data, gradually improve score
        current_time = time.time()
        time_since_last_review = current_time - self.last_review_time if self.last_review_time > 0 else 0
        
        # If we have no recent execution history and no metrics provided, default to optimistic score
        # This prevents blocking trades when there's insufficient data (e.g., after position closes)
        has_recent_data = (
            len(self.execution_history) > 0 or
            (recent_exec_metrics and recent_exec_metrics.get("fill_success_rate") is not None)
        )
        
        if not has_recent_data:
            # No recent data - use optimistic default to avoid blocking trades unnecessarily
            # This handles the case where positions just closed and we don't have recent buy execution data
            weighted_score = max(weighted_score, 0.70)  # At least 70% if no recent data
        else:
            # Automatic recovery: if score is low but no recent failures, gradually improve
            # Recovery rate: 0.01 (1%) per 5 minutes of good behavior
            recovery_rate = 0.01  # 1% per review interval
            recovery_intervals = max(0, int(time_since_last_review / self.review_interval_seconds))
            
            # Only apply recovery if current score is below threshold and we have recent good data
            if weighted_score < self.min_window_score_to_trade and recovery_intervals > 0:
                # Check if recent metrics are actually good (no recent failures)
                recent_fill_rate = fill_success_rate
                recent_slippage = avg_slippage
                recent_latency = avg_latency
                
                # If recent metrics are good, apply recovery
                if (recent_fill_rate >= 0.8 and recent_slippage <= 0.03 and recent_latency <= 3000):
                    recovery_bonus = min(0.10, recovery_intervals * recovery_rate)  # Cap at 10% recovery
                    weighted_score = min(1.0, weighted_score + recovery_bonus)
        
        # Minimum score floor: prevent score from getting stuck below threshold
        # Floor should be at least threshold - 5% to allow some buffer
        min_score_floor = max(0.50, self.min_window_score_to_trade - 0.05)
        weighted_score = max(min_score_floor, weighted_score)
        
        # Improved recovery mechanism: if score is stuck at floor, gradually recover
        # This prevents permanent blocking when metrics are temporarily bad
        if weighted_score <= min_score_floor + 0.01:  # Within 1% of floor
            current_time = time.time()
            time_since_last_review = current_time - self.last_review_time if self.last_review_time > 0 else 0
            
            # If we've been at the floor for a while, start gradual recovery
            # Recovery rate: 0.02 (2%) per review interval when stuck at floor
            recovery_intervals = max(0, int(time_since_last_review / self.review_interval_seconds))
            if recovery_intervals > 0:
                # More forgiving recovery - don't require perfect metrics
                # Just check that we're not getting worse
                recent_fill_rate = fill_success_rate
                recent_slippage = avg_slippage
                
                # If metrics aren't catastrophically bad, allow recovery
                if recent_fill_rate >= 0.5 and recent_slippage <= 0.10:  # More lenient thresholds
                    recovery_bonus = min(0.15, recovery_intervals * 0.02)  # Cap at 15% recovery
                    weighted_score = min(1.0, weighted_score + recovery_bonus)
                    log_info("time_window_scheduler.floor_recovery",
                            old_score=min_score_floor,
                            new_score=weighted_score,
                            recovery_intervals=recovery_intervals)
        
        return max(0.0, min(1.0, weighted_score))
    
    def _get_fill_success_rate(self, metrics: Optional[Dict[str, Any]]) -> float:
        """Get fill success rate from metrics or Prometheus"""
        if metrics and "fill_success_rate" in metrics:
            return float(metrics["fill_success_rate"])
        
        # Try to get from execution history
        if len(self.execution_history) > 0:
            recent = list(self.execution_history)[-20:]  # Last 20 executions
            successful = sum(1 for e in recent if e.get("success", False))
            if len(recent) > 0:
                return successful / len(recent)
        
        # Default: assume good if no data
        return 0.85
    
    def _get_avg_slippage(self, metrics: Optional[Dict[str, Any]]) -> float:
        """Get average slippage from metrics"""
        if metrics and "avg_slippage" in metrics:
            return float(metrics["avg_slippage"])
        
        # Try to get from execution history
        if len(self.execution_history) > 0:
            recent = [e.get("slippage", 0) for e in list(self.execution_history)[-20:] if e.get("slippage") is not None]
            if recent:
                return statistics.mean(recent)
        
        # Default: assume 2% slippage
        return 0.02
    
    def _get_avg_latency(self, metrics: Optional[Dict[str, Any]]) -> float:
        """Get average execution latency"""
        if metrics and "avg_latency_ms" in metrics:
            return float(metrics["avg_latency_ms"])
        
        # Try to get from execution history
        if len(self.execution_history) > 0:
            recent = [e.get("latency_ms", 0) for e in list(self.execution_history)[-20:] if e.get("latency_ms") is not None]
            if recent:
                return statistics.mean(recent)
        
        # Default: assume 2s latency
        return 2000.0
    
    def _get_market_quality_score(self, metrics: Optional[Dict[str, Any]]) -> float:
        """Get market quality score based on volatility and liquidity"""
        if not metrics:
            return 0.7  # Default neutral
        
        volatility = metrics.get("volatility", 0.5)
        liquidity = metrics.get("liquidity_score", 0.5)
        
        # Lower volatility = higher score
        volatility_score = max(0.0, 1.0 - volatility)
        
        # Higher liquidity = higher score
        liquidity_score = min(1.0, liquidity)
        
        # Combined score
        return (volatility_score * 0.6 + liquidity_score * 0.4)
    
    def record_execution(
        self,
        success: bool,
        slippage: Optional[float] = None,
        latency_ms: Optional[float] = None
    ):
        """Record execution result for metrics tracking"""
        self.execution_history.append({
            "timestamp": time.time(),
            "success": success,
            "slippage": slippage,
            "latency_ms": latency_ms
        })
    
    def _load_state(self) -> None:
        """Load scheduler state from persistent storage"""
        try:
            if SCHEDULER_STATE_PATH.exists():
                state = json.loads(SCHEDULER_STATE_PATH.read_text(encoding="utf-8"))
                self.current_window_score = float(state.get("current_window_score", 0.75))
                self.last_review_time = float(state.get("last_review_time", 0.0))
                self.is_paused = bool(state.get("is_paused", False))
                self.pause_until = float(state.get("pause_until", 0.0))
                
                # Load execution history (limited to recent entries)
                history = state.get("execution_history", [])
                # Only keep entries from last 24 hours
                current_time = time.time()
                day_ago = current_time - 86400
                for entry in history:
                    if entry.get("timestamp", 0) > day_ago:
                        self.execution_history.append(entry)
                
                log_info("time_window_scheduler.state_loaded",
                        score=self.current_window_score,
                        history_size=len(self.execution_history))
            else:
                # First run - use defaults
                self.last_review_time = 0.0
                self.current_window_score = 0.75  # Default to 75% - optimistic but safe
                self.is_paused = False
                self.pause_until = 0.0
        except Exception as e:
            log_error("time_window_scheduler.load_state_error", error=str(e))
            # On error, use defaults
            self.last_review_time = 0.0
            self.current_window_score = 0.75
            self.is_paused = False
            self.pause_until = 0.0
    
    def _save_state(self) -> None:
        """Save scheduler state to persistent storage"""
        try:
            SCHEDULER_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "current_window_score": self.current_window_score,
                "last_review_time": self.last_review_time,
                "is_paused": self.is_paused,
                "pause_until": self.pause_until,
                # Save recent execution history (last 50 entries)
                "execution_history": list(self.execution_history)[-50:]
            }
            SCHEDULER_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except Exception as e:
            log_error("time_window_scheduler.save_state_error", error=str(e))
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get scheduler metrics"""
        return {
            "enabled": self.enabled,
            "current_window_score": self.current_window_score,
            "is_paused": self.is_paused,
            "pause_until": self.pause_until,
            "execution_history_size": len(self.execution_history),
            "fill_success_rate": self._get_fill_success_rate(None),
            "avg_slippage": self._get_avg_slippage(None),
            "avg_latency_ms": self._get_avg_latency(None)
        }


# Global instance
_time_window_scheduler_instance: Optional[AITimeWindowScheduler] = None

def get_time_window_scheduler() -> AITimeWindowScheduler:
    """Get global time window scheduler instance"""
    global _time_window_scheduler_instance
    if _time_window_scheduler_instance is None:
        _time_window_scheduler_instance = AITimeWindowScheduler()
    return _time_window_scheduler_instance

