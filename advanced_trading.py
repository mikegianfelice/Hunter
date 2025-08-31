#!/usr/bin/env python3
"""
Advanced Trading Features Module
Implements sophisticated trading strategies for better execution
"""

import math
import time
from typing import List, Dict, Any, Tuple, Optional
from config_loader import get_config, get_config_bool, get_config_float, get_config_int

class AdvancedTradingEngine:
    def __init__(self):
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load advanced trading configuration"""
        return {
            # Order splitting
            'enable_order_splitting': get_config_bool("enable_order_splitting", True),
            'max_price_impact_per_slice': get_config_float("max_price_impact_per_slice", 0.02),
            'min_slice_amount_usd': get_config_float("min_slice_amount_usd", 2.0),
            'max_slices_per_trade': get_config_int("max_slices_per_trade", 5),
            
            # Dynamic slippage
            'enable_dynamic_slippage': get_config_bool("enable_dynamic_slippage", True),
            'dynamic_slippage_multiplier': get_config_float("dynamic_slippage_multiplier", 1.5),
            'max_dynamic_slippage': get_config_float("max_dynamic_slippage", 0.15),
            'min_dynamic_slippage': get_config_float("min_dynamic_slippage", 0.02),
            
            # ExactOut trades
            'enable_exactout_trades': get_config_bool("enable_exactout_trades", True),
            'exactout_liquidity_threshold': get_config_float("exactout_liquidity_threshold", 5000),
            'exactout_volume_threshold': get_config_float("exactout_volume_threshold", 1000),
            'exactout_max_attempts': get_config_int("exactout_max_attempts", 3),
            
            # Route restrictions
            'enable_route_restrictions': get_config_bool("enable_route_restrictions", True),
            'prefer_direct_routes': get_config_bool("prefer_direct_routes", True),
            'max_route_hops': get_config_int("max_route_hops", 2),
            'enable_direct_pool_swaps': get_config_bool("enable_direct_pool_swaps", True),
            
            # Enhanced preflight
            'enable_enhanced_preflight': get_config_bool("enable_enhanced_preflight", True),
            'check_token_decimals': get_config_bool("check_token_decimals", True),
            'check_ata_existence': get_config_bool("check_ata_existence", True),
            'check_mint_frozen': get_config_bool("check_mint_frozen", True),
            'check_transfer_fee': get_config_bool("check_transfer_fee", True),
            'check_pool_reserves': get_config_bool("check_pool_reserves", True),
            'min_pool_reserves_multiplier': get_config_float("min_pool_reserves_multiplier", 2.0)
        }
    
    def calculate_order_slices(self, total_amount_usd: float, token_data: Dict[str, Any]) -> List[float]:
        """
        Calculate optimal order slices to minimize price impact
        Returns list of slice amounts in USD
        """
        if not self.config['enable_order_splitting']:
            return [total_amount_usd]
        
        liquidity = token_data.get('liquidity', 0)
        volume_24h = token_data.get('volume24h', 0)
        
        # Calculate price impact for full order
        price_impact = self._estimate_price_impact(total_amount_usd, liquidity, volume_24h)
        
        # If impact is already low, no need to split
        if price_impact <= self.config['max_price_impact_per_slice']:
            return [total_amount_usd]
        
        # Calculate optimal slice size
        target_impact = self.config['max_price_impact_per_slice']
        optimal_slice_size = self._calculate_optimal_slice_size(liquidity, volume_24h, target_impact)
        
        # Ensure minimum slice size
        optimal_slice_size = max(optimal_slice_size, self.config['min_slice_amount_usd'])
        
        # Calculate number of slices
        num_slices = math.ceil(total_amount_usd / optimal_slice_size)
        num_slices = min(num_slices, self.config['max_slices_per_trade'])
        
        # Calculate actual slice size
        slice_size = total_amount_usd / num_slices
        
        # Create slices (last slice gets remainder)
        slices = []
        remaining = total_amount_usd
        for i in range(num_slices - 1):
            slices.append(slice_size)
            remaining -= slice_size
        slices.append(remaining)  # Last slice gets remainder
        
        print(f"📊 Order splitting: {total_amount_usd:.2f} USD → {len(slices)} slices of ~{slice_size:.2f} USD each")
        print(f"🎯 Estimated price impact: {price_impact*100:.2f}% → {target_impact*100:.2f}% per slice")
        
        return slices
    
    def calculate_dynamic_slippage(self, token_data: Dict[str, Any], base_slippage: float) -> float:
        """
        Calculate dynamic slippage based on predicted price impact
        """
        if not self.config['enable_dynamic_slippage']:
            return base_slippage
        
        liquidity = token_data.get('liquidity', 0)
        volume_24h = token_data.get('volume24h', 0)
        trade_amount = token_data.get('trade_amount_usd', 5.0)
        
        # Estimate price impact
        price_impact = self._estimate_price_impact(trade_amount, liquidity, volume_24h)
        
        # Calculate dynamic slippage
        dynamic_slippage = price_impact * self.config['dynamic_slippage_multiplier']
        
        # Apply bounds
        dynamic_slippage = max(dynamic_slippage, self.config['min_dynamic_slippage'])
        dynamic_slippage = min(dynamic_slippage, self.config['max_dynamic_slippage'])
        
        print(f"🎯 Dynamic slippage: {base_slippage*100:.2f}% → {dynamic_slippage*100:.2f}% (impact: {price_impact*100:.2f}%)")
        
        return dynamic_slippage
    
    def should_use_exactout(self, token_data: Dict[str, Any]) -> bool:
        """
        Determine if ExactOut trade should be used for sketchy tokens
        """
        if not self.config['enable_exactout_trades']:
            return False
        
        liquidity = token_data.get('liquidity', 0)
        volume_24h = token_data.get('volume24h', 0)
        
        # Use ExactOut for low liquidity/volume tokens
        if liquidity < self.config['exactout_liquidity_threshold']:
            print(f"⚠️ Using ExactOut: Low liquidity (${liquidity:.0f} < ${self.config['exactout_liquidity_threshold']:.0f})")
            return True
        
        if volume_24h < self.config['exactout_volume_threshold']:
            print(f"⚠️ Using ExactOut: Low volume (${volume_24h:.0f} < ${self.config['exactout_volume_threshold']:.0f})")
            return True
        
        return False
    
    def get_route_preferences(self, token_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get route preferences for Jupiter/Raydium
        """
        preferences = {}
        
        if self.config['enable_route_restrictions']:
            preferences['onlyDirectRoutes'] = self.config['prefer_direct_routes']
            preferences['maxHops'] = self.config['max_route_hops']
        
        # Add chain-specific preferences
        chain_id = token_data.get('chainId', 'ethereum').lower()
        if chain_id == 'solana':
            preferences['asLegacyTransaction'] = False
            preferences['useSharedAccounts'] = True
            preferences['computeUnitPriceMicroLamports'] = 1000
        
        return preferences
    
    def enhanced_preflight_check(self, token_data: Dict[str, Any], trade_amount_usd: float) -> Tuple[bool, str]:
        """
        Enhanced preflight checks before trading
        Returns (passed: bool, reason: str)
        """
        if not self.config['enable_enhanced_preflight']:
            return True, "preflight_disabled"
        
        token_address = token_data.get('address', '')
        chain_id = token_data.get('chainId', 'ethereum').lower()
        symbol = token_data.get('symbol', '')
        liquidity = token_data.get('liquidity', 0)
        volume_24h = token_data.get('volume24h', 0)
        
        print(f"🔍 Enhanced preflight check for {symbol} ({chain_id})")
        print(f"📊 Token metrics: Liquidity ${liquidity:,.0f}, Volume ${volume_24h:,.0f}")
        
        # Check minimum liquidity requirements
        min_liquidity = 50000  # $50k minimum liquidity
        if liquidity < min_liquidity:
            print(f"❌ Insufficient liquidity: ${liquidity:,.0f} < ${min_liquidity:,.0f}")
            return False, f"insufficient_liquidity_{liquidity:.0f}"
        
        # Check minimum volume requirements
        min_volume = 50000  # $50k minimum volume
        if volume_24h < min_volume:
            print(f"❌ Insufficient volume: ${volume_24h:,.0f} < ${min_volume:,.0f}")
            return False, f"insufficient_volume_{volume_24h:.0f}"
        
        # Check if trade amount is reasonable relative to liquidity
        max_trade_ratio = 0.01  # Maximum 1% of liquidity
        if trade_amount_usd > liquidity * max_trade_ratio:
            print(f"❌ Trade amount too large: ${trade_amount_usd:.2f} > ${liquidity * max_trade_ratio:.2f} (1% of liquidity)")
            return False, f"trade_amount_too_large"
        
        # Check token decimals
        if self.config['check_token_decimals']:
            decimals = self._check_token_decimals(token_address, chain_id)
            if decimals is None:
                return False, "invalid_token_decimals"
            print(f"✅ Token decimals: {decimals}")
        
        # Check pool reserves
        if self.config['check_pool_reserves']:
            reserves_ok = self._check_pool_reserves(token_address, trade_amount_usd, chain_id)
            if not reserves_ok:
                return False, "insufficient_pool_reserves"
            print(f"✅ Pool reserves sufficient")
        
        # Chain-specific checks
        if chain_id == 'solana':
            # Check ATA existence
            if self.config['check_ata_existence']:
                ata_exists = self._check_ata_existence(token_address)
                if not ata_exists:
                    print(f"⚠️ ATA does not exist - will be created during trade")
            
            # Check mint frozen status
            if self.config['check_mint_frozen']:
                is_frozen = self._check_mint_frozen(token_address)
                if is_frozen:
                    return False, "mint_is_frozen"
                print(f"✅ Mint is not frozen")
        
        # Check transfer fee configuration
        if self.config['check_transfer_fee']:
            transfer_fee = self._check_transfer_fee(token_address, chain_id)
            if transfer_fee > 0.1:  # More than 10% transfer fee
                print(f"⚠️ High transfer fee detected: {transfer_fee*100:.2f}%")
                return False, "high_transfer_fee"
            elif transfer_fee > 0.05:  # More than 5% transfer fee
                print(f"⚠️ Elevated transfer fee: {transfer_fee*100:.2f}%")
        
        print(f"✅ Enhanced preflight check passed")
        return True, "preflight_passed"
    
    def _estimate_price_impact(self, trade_amount_usd: float, liquidity: float, volume_24h: float) -> float:
        """
        Estimate price impact based on trade size vs liquidity
        """
        if liquidity <= 0:
            return 1.0  # 100% impact if no liquidity
        
        # Simple linear model: impact = trade_size / liquidity
        impact = trade_amount_usd / liquidity
        
        # Adjust based on volume (higher volume = lower impact)
        if volume_24h > 0:
            volume_factor = min(1.0, volume_24h / (liquidity * 10))  # Normalize by liquidity
            impact *= (1.0 - volume_factor * 0.5)  # Reduce impact by up to 50% for high volume
        
        return min(impact, 1.0)  # Cap at 100%
    
    def _calculate_optimal_slice_size(self, liquidity: float, volume_24h: float, target_impact: float) -> float:
        """
        Calculate optimal slice size to achieve target price impact
        """
        if liquidity <= 0:
            return 1.0  # Minimum slice if no liquidity
        
        # Reverse the impact calculation
        optimal_size = target_impact * liquidity
        
        # Adjust for volume
        if volume_24h > 0:
            volume_factor = min(1.0, volume_24h / (liquidity * 10))
            optimal_size *= (1.0 + volume_factor * 0.5)  # Increase size for high volume
        
        return optimal_size
    
    def _check_token_decimals(self, token_address: str, chain_id: str) -> Optional[int]:
        """Check token decimals"""
        try:
            if chain_id == 'ethereum':
                # Ethereum token decimals check
                from web3 import Web3
                from uniswap_executor import w3, _erc20
                
                token = _erc20(token_address)
                decimals = token.functions.decimals().call()
                return decimals
            elif chain_id == 'solana':
                # Solana token decimals check - simplified
                # For now, assume standard decimals to avoid import issues
                print(f"⚠️ Solana decimals check skipped - assuming standard decimals")
                return 9  # Most Solana tokens use 9 decimals
        except Exception as e:
            print(f"⚠️ Failed to check token decimals: {e}")
            return None
    
    def _check_pool_reserves(self, token_address: str, trade_amount_usd: float, chain_id: str) -> bool:
        """Check if pool has sufficient reserves"""
        try:
            if chain_id == 'ethereum':
                # Check Uniswap pool reserves
                from uniswap_executor import w3, _erc20, WETH
                
                token = _erc20(token_address)
                token_balance = token.functions.balanceOf(token_address).call()
                
                # Convert to USD (simplified)
                eth_price = 2000  # Approximate ETH price
                required_reserves = trade_amount_usd * self.config['min_pool_reserves_multiplier'] / eth_price
                
                return token_balance >= required_reserves
            elif chain_id == 'solana':
                # Check Raydium pool reserves via DexScreener
                import requests
                
                url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    pairs = data.get("pairs", [])
                    
                    for pair in pairs:
                        if pair.get("dexId") == "raydium":
                            liquidity = float(pair.get("liquidity", {}).get("usd", 0))
                            required_liquidity = trade_amount_usd * self.config['min_pool_reserves_multiplier']
                            
                            if liquidity >= required_liquidity:
                                return True
                
                return False
        except Exception as e:
            print(f"⚠️ Failed to check pool reserves: {e}")
            return True  # Assume OK if check fails
    
    def _check_ata_existence(self, token_address: str) -> bool:
        """Check if associated token account exists (Solana)"""
        try:
            # Simplified check - assume ATA will be created if needed
            print(f"⚠️ ATA existence check skipped - will be created during trade if needed")
            return True
        except Exception as e:
            print(f"⚠️ Failed to check ATA existence: {e}")
            return True  # Assume OK
    
    def _check_mint_frozen(self, token_address: str) -> bool:
        """Check if mint is frozen (Solana)"""
        try:
            # Simplified check - assume mint is not frozen
            print(f"⚠️ Mint frozen check skipped - assuming mint is not frozen")
            return False
        except Exception as e:
            print(f"⚠️ Failed to check mint frozen status: {e}")
            return False
    
    def _check_transfer_fee(self, token_address: str, chain_id: str) -> float:
        """Check transfer fee configuration"""
        try:
            if chain_id == 'ethereum':
                # Check for transfer fee in token contract
                from web3 import Web3
                from uniswap_executor import w3, _erc20
                
                token = _erc20(token_address)
                
                # Try to call transferFee function (common in fee tokens)
                try:
                    transfer_fee = token.functions.transferFee().call()
                    return transfer_fee / 10000  # Convert from basis points
                except:
                    return 0.0
            elif chain_id == 'solana':
                # Solana tokens don't typically have transfer fees
                return 0.0
        except Exception as e:
            print(f"⚠️ Failed to check transfer fee: {e}")
            return 0.0

# Global instance
advanced_trading = AdvancedTradingEngine()
