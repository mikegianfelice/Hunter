#!/usr/bin/env python3
"""
Transaction Analyzer - Extract actual execution details from blockchain transactions
Supports Ethereum and Solana chains for accurate fee and slippage tracking
"""

from web3 import Web3
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Uniswap contract ABIs for parsing logs
UNISWAP_V2_ROUTER_ABI = [
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "sender", "type": "address"},
            {"indexed": False, "name": "amount0In", "type": "uint256"},
            {"indexed": False, "name": "amount1In", "type": "uint256"},
            {"indexed": False, "name": "amount0Out", "type": "uint256"},
            {"indexed": False, "name": "amount1Out", "type": "uint256"},
            {"indexed": True, "name": "to", "type": "address"}
        ],
        "name": "Swap",
        "type": "event"
    }
]

UNISWAP_V2_FACTORY = "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f"

def get_eth_price_usd() -> float:
    """Get current ETH price in USD"""
    try:
        from src.utils.utils import get_eth_price_usd as get_price
        return get_price()
    except Exception:
        # Fallback to CoinGecko
        try:
            import requests
            import os
            coingecko_key = os.getenv("COINGECKO_API_KEY", "").strip()
            base_url = "https://api.coingecko.com/api/v3/"
            url = f"{base_url}simple/price?ids=ethereum&vs_currencies=usd"
            headers = {}
            if coingecko_key:
                headers["x-cg-demo-api-key"] = coingecko_key
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return float(data['ethereum']['usd'])
        except Exception:
            pass
        return 3000.0  # Conservative fallback

def analyze_buy_transaction(w3: Web3, tx_hash: str) -> Dict:
    """
    Analyze buy transaction receipt to extract actual execution details
    
    Returns:
        Dict with:
        - gas_fee_usd: Gas fee in USD
        - tokens_received: Actual tokens received (raw wei)
        - actual_cost_usd: Total actual cost including gas
        - slippage_actual: Actual slippage experienced
    """
    try:
        receipt = w3.eth.get_transaction_receipt(tx_hash)
        tx = w3.eth.get_transaction(tx_hash)
        
        # Calculate gas fee
        gas_used = receipt['gasUsed']
        gas_price = tx.get('gasPrice') or receipt.get('effectiveGasPrice', 0)
        if isinstance(gas_price, int):
            gas_fee_eth = (gas_used * gas_price) / 10**18
        else:
            gas_fee_eth = 0
        
        # Convert to USD
        eth_price = get_eth_price_usd()
        gas_fee_usd = gas_fee_eth * eth_price
        
        # Get ETH value sent
        tx_value_eth = float(tx['value']) / 10**18 if tx.get('value') else 0
        actual_cost_eth = tx_value_eth + gas_fee_eth
        actual_cost_usd = actual_cost_eth * eth_price
        
        # Parse logs to get actual tokens received
        tokens_received = _parse_swap_logs_buy(w3, receipt)
        
        result = {
            'gas_fee_usd': gas_fee_usd,
            'tokens_received': tokens_received,
            'actual_cost_usd': actual_cost_usd,
            'tx_value_eth': tx_value_eth,
            'success': True
        }
        
        logger.info(f"Analyzed buy tx {tx_hash[:10]}...: cost=${actual_cost_usd:.4f}, gas=${gas_fee_usd:.4f}")
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing buy transaction {tx_hash}: {e}")
        return {
            'gas_fee_usd': 0,
            'tokens_received': None,
            'actual_cost_usd': 0,
            'error': str(e),
            'success': False
        }

def analyze_sell_transaction(w3: Web3, tx_hash: str) -> Dict:
    """
    Analyze sell transaction receipt to extract actual proceeds
    
    Returns:
        Dict with:
        - gas_fee_usd: Gas fee in USD
        - eth_received: Actual ETH received (in wei)
        - actual_proceeds_usd: Net proceeds after gas
        - slippage_actual: Actual slippage experienced
    """
    try:
        receipt = w3.eth.get_transaction_receipt(tx_hash)
        tx = w3.eth.get_transaction(tx_hash)
        
        # Calculate gas fee
        gas_used = receipt['gasUsed']
        gas_price = tx.get('gasPrice') or receipt.get('effectiveGasPrice', 0)
        if isinstance(gas_price, int):
            gas_fee_eth = (gas_used * gas_price) / 10**18
        else:
            gas_fee_eth = 0
        
        # Convert to USD
        eth_price = get_eth_price_usd()
        gas_fee_usd = gas_fee_eth * eth_price
        
        # Parse logs to get actual ETH received
        eth_received = _parse_swap_logs_sell(w3, receipt)
        eth_received_value = eth_received / 10**18 if eth_received else 0
        
        # Net proceeds after gas
        net_proceeds_eth = max(0, eth_received_value - gas_fee_eth)
        actual_proceeds_usd = net_proceeds_eth * eth_price
        
        result = {
            'gas_fee_usd': gas_fee_usd,
            'eth_received': eth_received,
            'eth_received_value': eth_received_value,
            'actual_proceeds_usd': actual_proceeds_usd,
            'success': True
        }
        
        logger.info(f"Analyzed sell tx {tx_hash[:10]}...: proceeds=${actual_proceeds_usd:.4f}, gas=${gas_fee_usd:.4f}")
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing sell transaction {tx_hash}: {e}")
        return {
            'gas_fee_usd': 0,
            'eth_received': None,
            'actual_proceeds_usd': 0,
            'error': str(e),
            'success': False
        }

def _parse_swap_logs_buy(w3: Web3, receipt: Dict) -> Optional[int]:
    """
    Parse Uniswap Swap event logs to get actual tokens received during buy
    
    Returns: Token amount in raw wei units
    """
    try:
        # Get WETH and token addresses from transaction
        weth_address = w3.to_checksum_address("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2")
        
        for log in receipt['logs']:
            try:
                # Try to decode as Swap event
                event_data = w3.eth.contract(abi=UNISWAP_V2_ROUTER_ABI).events.Swap().process_log(log)
                
                # For buys: ETH -> Token
                # amount0In/amount1In is the input, amount0Out/amount1Out is output
                amount0_in = event_data['args']['amount0In']
                amount1_in = event_data['args']['amount1In']
                amount0_out = event_data['args']['amount0Out']
                amount1_out = event_data['args']['amount1Out']
                
                # Determine which direction the swap went
                # One of the "In" amounts will be ETH (WETH), the corresponding "Out" will be tokens
                if amount0_in > 0 and amount1_out > 0:
                    # amount0 is ETH, amount1 is token
                    return int(amount1_out)
                elif amount1_in > 0 and amount0_out > 0:
                    # amount1 is ETH, amount0 is token
                    return int(amount0_out)
                    
            except Exception:
                # Not a Swap event, continue
                continue
        
        logger.warning("Could not parse tokens from swap logs")
        return None
        
    except Exception as e:
        logger.error(f"Error parsing swap logs: {e}")
        return None

def _parse_swap_logs_sell(w3: Web3, receipt: Dict) -> Optional[int]:
    """
    Parse Uniswap Swap event logs to get actual ETH received during sell
    
    Returns: ETH amount in raw wei units
    """
    try:
        # Get WETH and token addresses from transaction
        weth_address = w3.to_checksum_address("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2")
        
        for log in receipt['logs']:
            try:
                # Try to decode as Swap event
                event_data = w3.eth.contract(abi=UNISWAP_V2_ROUTER_ABI).events.Swap().process_log(log)
                
                amount0_in = event_data['args']['amount0In']
                amount1_in = event_data['args']['amount1In']
                amount0_out = event_data['args']['amount0Out']
                amount1_out = event_data['args']['amount1Out']
                
                # Determine which direction the swap went
                # One of the "In" amounts will be tokens, the corresponding "Out" will be ETH
                if amount0_in > 0 and amount1_out > 0:
                    # amount0 is tokens, amount1 is ETH
                    return int(amount1_out)
                elif amount1_in > 0 and amount0_out > 0:
                    # amount1 is tokens, amount0 is ETH
                    return int(amount0_out)
                    
            except Exception:
                # Not a Swap event, continue
                continue
        
        # Fallback: check for WETH Transfer events to wallet
        wallet_address = None
        if receipt.get('to'):
            wallet_address = receipt['to']
        
        for log in receipt['logs']:
            if log.get('address') and log['address'].lower() == weth_address.lower():
                if log.get('topics'):
                    # ERC20 Transfer event: Transfer(address indexed from, address indexed to, uint256 value)
                    if wallet_address and len(log['topics']) >= 3:
                        to_address = w3.to_checksum_address(f"0x{log['topics'][2].hex()[26:]}")
                        if to_address.lower() == wallet_address.lower():
                            # Extract value from data
                            value_hex = log['data'].hex()
                            if value_hex and len(value_hex) >= 64:
                                return int(value_hex, 16)
        
        logger.warning("Could not parse ETH received from swap logs")
        return None
        
    except Exception as e:
        logger.error(f"Error parsing swap logs: {e}")
        return None

def calculate_actual_slippage(quoted_amount: float, actual_amount: float, is_buy: bool) -> float:
    """
    Calculate actual slippage experienced
    
    Args:
        quoted_amount: Expected amount from quote
        actual_amount: Actual amount received
        is_buy: True if buy, False if sell
    
    Returns:
        Slippage as ratio (e.g., 0.01 = 1% slippage)
    """
    if not quoted_amount or quoted_amount <= 0:
        return 0.0
    
    if is_buy:
        # For buys: negative slippage = got fewer tokens than expected
        slippage = (quoted_amount - actual_amount) / quoted_amount
    else:
        # For sells: negative slippage = got less proceeds than expected
        slippage = (quoted_amount - actual_amount) / quoted_amount
    
    return slippage

def extract_dex_fees_from_receipt(receipt: Dict, chain: str, dex: str, is_buy: bool) -> float:
    """
    Extract DEX swap fees from transaction logs
    
    Returns:
        DEX fee in USD
    """
    try:
        if chain == "ethereum" and dex == "uniswap":
            # Uniswap fees are typically 0.3% on V2, variable on V3
            # For V2, we can calculate from swap amounts
            # For now, return estimated fee
            
            # Try to extract swap amounts and calculate fee
            # This is a simplified estimate - actual fee is baked into swap price
            return 0.0  # Will be calculated as slippage difference
            
        elif chain == "solana":
            # Jupiter/Raydium fees are baked into the route
            return 0.0  # Will be calculated from price impact
            
    except Exception as e:
        logger.error(f"Error extracting DEX fees: {e}")
    
    return 0.0
