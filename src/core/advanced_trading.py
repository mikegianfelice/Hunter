#!/usr/bin/env python3
"""
Advanced Trading Engine
Sophisticated execution logic for enhanced trading performance
"""

import logging
from typing import Dict, List, Tuple, Any
from src.config.config_loader import get_config_float, get_config_bool

logger = logging.getLogger(__name__)

class AdvancedTrading:
    """Advanced trading engine with sophisticated execution logic"""
    
    def __init__(self):
        self.min_liquidity_threshold = get_config_float("min_liquidity_usd", 10000)
        self.max_slippage = get_config_float("max_slippage", 0.5)  # 50%
        self.min_slippage = get_config_float("min_slippage", 0.01)  # 1%
        self.sketchy_token_threshold = get_config_float("sketchy_token_threshold", 0.3)
        
    def enhanced_preflight_check(self, token: Dict, amount_usd: float) -> Tuple[bool, str]:
        """
        Enhanced preflight check with comprehensive validation
        
        Args:
            token: Token information dictionary
            amount_usd: Trade amount in USD
            
        Returns:
            Tuple of (success, reason)
        """
        try:
            symbol = token.get("symbol", "?")
            address = token.get("address", "")
            chain_id = token.get("chainId", "ethereum").lower()
            
            # Basic validation
            if not address:
                return False, "No token address provided"
            
            if amount_usd <= 0:
                return False, f"Invalid trade amount: ${amount_usd}"
            
            # Check if token is too new (less than 1 hour old)
            if "created_at" in token:
                from datetime import datetime, timedelta
                created_time = datetime.fromisoformat(token["created_at"].replace('Z', '+00:00'))
                if datetime.now(created_time.tzinfo) - created_time < timedelta(hours=1):
                    return False, "Token too new (less than 1 hour old)"
            
            # Check liquidity if available
            liquidity = token.get("liquidity_usd", 0)
            if liquidity > 0 and liquidity < self.min_liquidity_threshold:
                return False, f"Insufficient liquidity: ${liquidity:.2f} < ${self.min_liquidity_threshold}"
            
            # Check market cap if available
            market_cap = token.get("market_cap", 0)
            if market_cap > 0 and market_cap < 100000:  # Less than $100k market cap
                return False, f"Market cap too low: ${market_cap:.2f}"
            
            # Check for suspicious patterns
            if self._is_suspicious_token(token):
                return False, "Token flagged as suspicious"
            
            # Check trade amount vs liquidity ratio
            if liquidity > 0:
                liquidity_ratio = amount_usd / liquidity
                if liquidity_ratio > 0.1:  # More than 10% of liquidity
                    return False, f"Trade too large for liquidity: {liquidity_ratio*100:.1f}%"
            
            return True, "Preflight check passed"
            
        except Exception as e:
            logger.error(f"Preflight check error: {e}")
            return False, f"Preflight check failed: {e}"
    
    def calculate_order_slices(self, amount_usd: float, token: Dict) -> List[float]:
        """
        Calculate optimal order slices for large trades
        
        Args:
            amount_usd: Total trade amount
            token: Token information
            
        Returns:
            List of slice amounts
        """
        try:
            # For small trades, no slicing needed
            if amount_usd <= 50:
                return [amount_usd]
            
            # For medium trades, 2 slices
            elif amount_usd <= 200:
                return [amount_usd * 0.6, amount_usd * 0.4]
            
            # For large trades, 3-4 slices
            elif amount_usd <= 500:
                return [amount_usd * 0.4, amount_usd * 0.3, amount_usd * 0.3]
            
            # For very large trades, 4-5 slices
            else:
                slice_count = min(5, max(4, int(amount_usd / 100)))
                slice_size = amount_usd / slice_count
                return [slice_size] * slice_count
                
        except Exception as e:
            logger.error(f"Error calculating order slices: {e}")
            return [amount_usd]  # Fallback to single slice
    
    def calculate_dynamic_slippage(self, token: Dict, base_slippage: float) -> float:
        """
        Calculate dynamic slippage based on token characteristics
        
        Args:
            token: Token information
            base_slippage: Base slippage from config
            
        Returns:
            Calculated slippage percentage
        """
        try:
            slippage = base_slippage
            
            # Adjust based on liquidity
            liquidity = token.get("liquidity_usd", 0)
            if liquidity > 0:
                if liquidity < 50000:  # Low liquidity
                    slippage *= 1.5
                elif liquidity > 500000:  # High liquidity
                    slippage *= 0.8
            
            # Adjust based on market cap
            market_cap = token.get("market_cap", 0)
            if market_cap > 0:
                if market_cap < 1000000:  # Less than $1M
                    slippage *= 1.3
                elif market_cap > 10000000:  # More than $10M
                    slippage *= 0.9
            
            # Adjust based on age
            if "created_at" in token:
                from datetime import datetime, timedelta
                try:
                    created_time = datetime.fromisoformat(token["created_at"].replace('Z', '+00:00'))
                    age_hours = (datetime.now(created_time.tzinfo) - created_time).total_seconds() / 3600
                    if age_hours < 6:  # Very new token
                        slippage *= 1.4
                    elif age_hours < 24:  # New token
                        slippage *= 1.2
                except:
                    pass  # If date parsing fails, use default slippage
            
            # Adjust based on volatility indicators
            if self._is_volatile_token(token):
                slippage *= 1.2
            
            # Ensure slippage is within bounds
            slippage = max(self.min_slippage, min(slippage, self.max_slippage))
            
            return slippage
            
        except Exception as e:
            logger.error(f"Error calculating dynamic slippage: {e}")
            return base_slippage  # Fallback to base slippage
    
    def should_use_exactout(self, token: Dict) -> bool:
        """
        Determine if ExactOut should be used for this token
        
        Args:
            token: Token information
            
        Returns:
            True if ExactOut should be used
        """
        try:
            # Use ExactOut for sketchy tokens
            if self._is_sketchy_token(token):
                return True
            
            # Use ExactOut for very new tokens
            if "created_at" in token:
                from datetime import datetime, timedelta
                try:
                    created_time = datetime.fromisoformat(token["created_at"].replace('Z', '+00:00'))
                    age_hours = (datetime.now(created_time.tzinfo) - created_time).total_seconds() / 3600
                    if age_hours < 2:  # Less than 2 hours old
                        return True
                except:
                    pass
            
            # Use ExactOut for low liquidity tokens
            liquidity = token.get("liquidity_usd", 0)
            if liquidity > 0 and liquidity < 20000:  # Less than $20k liquidity
                return True
            
            # Use ExactOut for small market cap tokens
            market_cap = token.get("market_cap", 0)
            if market_cap > 0 and market_cap < 500000:  # Less than $500k market cap
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error determining ExactOut usage: {e}")
            return False  # Default to False
    
    def get_route_preferences(self, token: Dict) -> Dict[str, Any]:
        """
        Get route preferences for this token
        
        Args:
            token: Token information
            
        Returns:
            Dictionary of route preferences
        """
        try:
            preferences = {
                "max_hops": 3,
                "prefer_direct": True,
                "avoid_wrapped": True,
                "min_liquidity_threshold": 1000,
                "max_price_impact": 0.05,  # 5%
                "prefer_stable_routes": False
            }
            
            # Adjust preferences based on token characteristics
            liquidity = token.get("liquidity_usd", 0)
            if liquidity < 50000:  # Low liquidity
                preferences["max_hops"] = 2
                preferences["prefer_direct"] = True
                preferences["min_liquidity_threshold"] = 500
            
            # For sketchy tokens, be more restrictive
            if self._is_sketchy_token(token):
                preferences["max_hops"] = 1
                preferences["prefer_direct"] = True
                preferences["max_price_impact"] = 0.03  # 3%
            
            return preferences
            
        except Exception as e:
            logger.error(f"Error getting route preferences: {e}")
            return {
                "max_hops": 3,
                "prefer_direct": True,
                "avoid_wrapped": True,
                "min_liquidity_threshold": 1000,
                "max_price_impact": 0.05
            }
    
    def _is_suspicious_token(self, token: Dict) -> bool:
        """Check if token has suspicious characteristics"""
        try:
            symbol = token.get("symbol", "").upper()
            name = token.get("name", "").upper()
            
            # Check for suspicious keywords
            suspicious_keywords = [
                "TEST", "FAKE", "SCAM", "RUG", "PULL", "HONEYPOT",
                "PONZI", "PYRAMID", "PUMP", "DUMP", "MOON", "LAMBO"
            ]
            
            for keyword in suspicious_keywords:
                if keyword in symbol or keyword in name:
                    return True
            
            # Check for very short or very long names
            if len(symbol) < 2 or len(symbol) > 20:
                return True
            
            # Check for excessive special characters
            special_chars = sum(1 for c in symbol if not c.isalnum())
            if special_chars > len(symbol) * 0.3:  # More than 30% special chars
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking suspicious token: {e}")
            return False
    
    def _is_sketchy_token(self, token: Dict) -> bool:
        """Check if token is considered sketchy"""
        try:
            # Very new tokens
            if "created_at" in token:
                from datetime import datetime, timedelta
                try:
                    created_time = datetime.fromisoformat(token["created_at"].replace('Z', '+00:00'))
                    age_hours = (datetime.now(created_time.tzinfo) - created_time).total_seconds() / 3600
                    if age_hours < 1:  # Less than 1 hour old
                        return True
                except:
                    pass
            
            # Low liquidity
            liquidity = token.get("liquidity_usd", 0)
            if liquidity > 0 and liquidity < 10000:  # Less than $10k
                return True
            
            # Low market cap
            market_cap = token.get("market_cap", 0)
            if market_cap > 0 and market_cap < 100000:  # Less than $100k
                return True
            
            # Suspicious characteristics
            if self._is_suspicious_token(token):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking sketchy token: {e}")
            return False
    
    def _is_volatile_token(self, token: Dict) -> bool:
        """Check if token is highly volatile"""
        try:
            # Check price change if available
            price_change_24h = token.get("price_change_24h", 0)
            if abs(price_change_24h) > 50:  # More than 50% change in 24h
                return True
            
            # Check volume spike if available
            volume_24h = token.get("volume_24h", 0)
            market_cap = token.get("market_cap", 0)
            if volume_24h > 0 and market_cap > 0:
                volume_ratio = volume_24h / market_cap
                if volume_ratio > 2:  # Volume more than 2x market cap
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking volatile token: {e}")
            return False

# Create global instance
advanced_trading = AdvancedTrading()
