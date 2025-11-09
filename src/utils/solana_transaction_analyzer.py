#!/usr/bin/env python3
"""
Solana Transaction Analyzer - Extract actual execution details from Solana transactions
Analyzes Jupiter swaps to calculate gas fees and actual amounts
"""

import requests
import json
import base64
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

def get_sol_price_usd() -> float:
    """Get current SOL price in USD"""
    try:
        from src.utils.utils import get_sol_price_usd as get_price
        return get_price()
    except Exception:
        # Fallback to CoinGecko
        try:
            import os
            coingecko_key = os.getenv("COINGECKO_API_KEY", "").strip()
            url = "https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd"
            headers = {}
            if coingecko_key:
                url += f"&api_key={coingecko_key}"
                headers["x-cg-demo-api-key"] = coingecko_key
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return float(data['solana']['usd'])
        except Exception:
            pass
        return 150.0  # Conservative fallback

def analyze_jupiter_transaction(rpc_url: str, tx_signature: str, wallet_address: str, is_buy: bool = True) -> Dict:
    """
    Analyze Jupiter swap transaction to extract actual execution details
    
    Args:
        rpc_url: Solana RPC URL
        tx_signature: Transaction signature
        wallet_address: Wallet address to track balance changes
        is_buy: True if buy transaction, False if sell
    
    Returns:
        Dict with:
        - gas_fee_usd: Transaction fee in USD
        - tokens_received: Actual tokens received (for buys)
        - sol_received: Actual SOL/USDC received (for sells)
        - actual_cost_usd: Total actual cost including fee
        - actual_proceeds_usd: Net proceeds after fee
    """
    try:
        # Get transaction details
        response = requests.post(rpc_url, json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTransaction",
            "params": [
                tx_signature,
                {
                    "encoding": "jsonParsed",
                    "maxSupportedTransactionVersion": 0
                }
            ]
        }, timeout=30)
        
        if response.status_code != 200:
            logger.error(f"RPC request failed: {response.status_code}")
            return _empty_result()
        
        result = response.json()
        if 'error' in result:
            logger.error(f"RPC error: {result['error']}")
            return _empty_result()
        
        tx_data = result.get('result', {})
        if not tx_data:
            logger.error("No transaction data returned")
            return _empty_result()
        
        meta = tx_data.get('meta', {})
        account_keys = tx_data.get('transaction', {}).get('message', {}).get('accountKeys', [])
        
        # Find wallet account index
        wallet_index = None
        if isinstance(account_keys, list):
            for i, key in enumerate(account_keys):
                if isinstance(key, dict) and key.get('pubkey') == wallet_address:
                    wallet_index = i
                    break
                elif isinstance(key, str) and key == wallet_address:
                    wallet_index = i
                    break
        
        if wallet_index is None:
            logger.warning(f"Wallet not found in transaction accounts")
            # Try alternative lookup
            try:
                pre_balances = meta.get('preBalances', [])
                for i in range(len(pre_balances)):
                    account_key = account_keys[i] if i < len(account_keys) else None
                    if account_key:
                        account_addr = account_key if isinstance(account_key, str) else account_key.get('pubkey', '')
                        if account_addr == wallet_address:
                            wallet_index = i
                            break
            except Exception as e:
                logger.error(f"Error finding wallet index: {e}")
        
        if wallet_index is None:
            logger.error("Could not determine wallet account index")
            return _empty_result()
        
        # Extract balance changes
        pre_balances = meta.get('preBalances', [])
        post_balances = meta.get('postBalances', [])
        
        if wallet_index >= len(pre_balances) or wallet_index >= len(post_balances):
            logger.error("Wallet index out of bounds")
            return _empty_result()
        
        pre_balance = pre_balances[wallet_index] / 10**9  # Convert lamports to SOL
        post_balance = post_balances[wallet_index] / 10**9
        balance_change = post_balance - pre_balance
        
        # Transaction fee in lamports
        fee_lamports = meta.get('fee', 0)
        fee_sol = fee_lamports / 10**9
        sol_price = get_sol_price_usd()
        fee_usd = fee_sol * sol_price
        
        # Parse token transfers
        token_transfers = _parse_token_transfers(tx_data, wallet_address)
        
        result = {
            'gas_fee_usd': fee_usd,
            'sol_balance_change': balance_change,
            'token_transfers': token_transfers,
            'success': True
        }
        
        if is_buy:
            # For buys, calculate actual cost
            # Balance change should be negative (we spent SOL)
            tokens_received = sum([t['amount'] for t in token_transfers if t['direction'] == 'receive'])
            result['tokens_received'] = tokens_received
            result['actual_cost_usd'] = abs(balance_change) * sol_price + fee_usd
        else:
            # For sells, calculate proceeds
            sol_received = max(0, balance_change - fee_sol) if balance_change > 0 else 0
            result['sol_received'] = sol_received
            result['actual_proceeds_usd'] = sol_received * sol_price
        
        logger.info(f"Analyzed Solana tx {tx_signature[:8]}...: fee=${fee_usd:.4f}, balance_change={balance_change:.6f} SOL")
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing Solana transaction {tx_signature}: {e}")
        return _empty_result()

def _parse_token_transfers(tx_data: Dict, wallet_address: str) -> List[Dict]:
    """
    Parse token transfer events from transaction
    
    Returns:
        List of transfers with direction and amount
    """
    transfers = []
    try:
        meta = tx_data.get('meta', {})
        account_keys = tx_data.get('transaction', {}).get('message', {}).get('accountKeys', [])
        
        # Parse pre/post token balances
        pre_token_balances = meta.get('preTokenBalances', [])
        post_token_balances = meta.get('postTokenBalances', [])
        
        # Create maps by account index
        pre_map = {b['accountIndex']: b for b in pre_token_balances}
        post_map = {b['accountIndex']: b for b in post_token_balances}
        
        # Find all token accounts
        token_accounts = set(pre_map.keys()) | set(post_map.keys())
        
        for account_idx in token_accounts:
            pre_bal = pre_map.get(account_idx, {})
            post_bal = post_map.get(account_idx, {})
            
            # Check if this account belongs to our wallet
            if account_idx >= len(account_keys):
                continue
                
            account_key = account_keys[account_idx]
            account_addr = account_key if isinstance(account_key, str) else account_key.get('pubkey', '')
            
            # For now, we'll track all changes - can filter by wallet if needed
            pre_amount = float(pre_bal.get('uiTokenAmount', {}).get('uiAmount', 0) or 0)
            post_amount = float(post_bal.get('uiTokenAmount', {}).get('uiAmount', 0) or 0)
            amount_change = post_amount - pre_amount
            
            if abs(amount_change) > 0:
                transfers.append({
                    'account_index': account_idx,
                    'account_address': account_addr,
                    'direction': 'receive' if amount_change > 0 else 'send',
                    'amount': abs(amount_change),
                    'token_mint': pre_bal.get('mint') or post_bal.get('mint')
                })
        
        return transfers
        
    except Exception as e:
        logger.error(f"Error parsing token transfers: {e}")
        return []

def _empty_result() -> Dict:
    """Return empty result structure"""
    return {
        'gas_fee_usd': 0,
        'tokens_received': 0,
        'sol_received': 0,
        'actual_cost_usd': 0,
        'actual_proceeds_usd': 0,
        'error': 'Could not analyze transaction',
        'success': False
    }

def calculate_actual_slippage(quoted_amount: float, actual_amount: float, is_buy: bool) -> float:
    """
    Calculate actual slippage experienced on Solana
    
    Args:
        quoted_amount: Expected amount from Jupiter quote
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
