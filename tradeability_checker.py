#!/usr/bin/env python3
"""
Tradeability Checker - Pre-filter tokens to only include tradeable ones
"""

import requests
import time
from typing import Dict, List, Tuple, Optional

def check_jupiter_tradeability(token_address: str, chain_id: str = "solana") -> bool:
    """
    Check if a token is tradeable on Jupiter
    Returns True if tradeable, False otherwise
    """
    if chain_id.lower() != "solana":
        return True  # Skip check for non-Solana chains
    
    try:
        # Test with SOL -> token quote
        sol_mint = "So11111111111111111111111111111111111111112"
        url = "https://quote-api.jup.ag/v6/quote"
        params = {
            "inputMint": sol_mint,
            "outputMint": token_address,
            "amount": "1000000000",  # 1 SOL in lamports
            "slippageBps": 500,  # 5% slippage for test
            "onlyDirectRoutes": "true"
        }
        
        response = requests.get(url, params=params, timeout=3)  # Shorter timeout
        
        if response.status_code == 200:
            data = response.json()
            # Check if we got valid quote data (Jupiter returns data at root level)
            if data.get("outAmount") or data.get("inAmount"):
                return True
            else:
                return False
        elif response.status_code == 400:
            # Check if it's a "not tradeable" error
            try:
                error_data = response.json()
                error_msg = error_data.get('error', '').lower()
                if 'not tradable' in error_msg or 'not tradeable' in error_msg:
                    return False
                elif 'input and output mints are not allowed to be equal' in error_msg:
                    # This means the token is SOL itself, which is tradeable
                    return True
                # For other 400 errors, assume it might be tradeable
                return True
            except:
                return True
        
        return False
        
    except Exception as e:
        print(f"⚠️ Jupiter tradeability check failed for {token_address[:8]}...{address[-8:]}: {e}")
        return True  # Assume tradeable if check fails

def check_raydium_tradeability(token_address: str, chain_id: str = "solana") -> bool:
    """
    Check if a token is tradeable on Raydium
    Returns True if tradeable, False otherwise
    """
    if chain_id.lower() != "solana":
        return True  # Skip check for non-Solana chains
    
    try:
        # Try to get token info from Raydium API
        url = f"https://api.raydium.io/v2/sdk/token/raydium.mainnet.json"
        
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            official_tokens = data.get("official", [])
            
            # Check if token is in official Raydium tokens
            for token_info in official_tokens:
                if token_info.get("mint", "").lower() == token_address.lower():
                    return True
        
        # If not in official list, try a quote test
        return _test_raydium_quote(token_address)
        
    except Exception as e:
        print(f"⚠️ Raydium tradeability check failed for {token_address[:8]}...{address[-8:]}: {e}")
        return True  # Assume tradeable if check fails

def _test_raydium_quote(token_address: str) -> bool:
    """Test if we can get a quote from Raydium for this token"""
    try:
        # Try to get pool info for the token
        url = "https://api.raydium.io/v2/main/price"
        params = {"ids": token_address}
        
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("data") and token_address in data["data"]:
                return True
        
        return False
        
    except Exception as e:
        print(f"⚠️ Raydium quote test failed for {token_address[:8]}...{address[-8:]}: {e}")
        return False

def check_ethereum_tradeability(token_address: str, chain_id: str = "ethereum") -> bool:
    """
    Check if a token is tradeable on Ethereum (Uniswap)
    Returns True if tradeable, False otherwise
    """
    if chain_id.lower() != "ethereum":
        return True  # Skip check for non-Ethereum chains
    
    try:
        # Test with a small quote from Uniswap
        from utils import fetch_token_price_usd
        price = fetch_token_price_usd(token_address)
        
        # If we can get a price, assume it's tradeable
        return price is not None and price > 0
        
    except Exception as e:
        print(f"⚠️ Ethereum tradeability check failed for {token_address[:8]}...{address[-8:]}: {e}")
        return True  # Assume tradeable if check fails

def is_token_tradeable(token_data: Dict) -> Tuple[bool, str]:
    """
    Check if a token is tradeable on its respective chain
    Returns (is_tradeable: bool, reason: str)
    """
    token_address = token_data.get("address", "")
    chain_id = token_data.get("chainId", "ethereum").lower()
    symbol = token_data.get("symbol", "UNKNOWN")
    
    if not token_address:
        return False, "no_address"
    
    # Check tradeability based on chain
    if chain_id == "solana":
        # For Solana, check both Jupiter and Raydium
        jupiter_ok = check_jupiter_tradeability(token_address, chain_id)
        raydium_ok = check_raydium_tradeability(token_address, chain_id)
        
        if jupiter_ok or raydium_ok:
            return True, "tradeable_on_solana"
        else:
            return False, "not_tradeable_on_solana"
    
    elif chain_id == "ethereum":
        ethereum_ok = check_ethereum_tradeability(token_address, chain_id)
        if ethereum_ok:
            return True, "tradeable_on_ethereum"
        else:
            return False, "not_tradeable_on_ethereum"
    
    elif chain_id == "base":
        # For Base, assume tradeable for now (similar to Ethereum)
        return True, "tradeable_on_base"
    
    else:
        # For unknown chains, assume tradeable
        return True, "unknown_chain_assume_tradeable"

def filter_tradeable_tokens(tokens: List[Dict], max_checks: int = 50) -> List[Dict]:
    """
    Filter a list of tokens to only include tradeable ones
    Returns list of tradeable tokens
    """
    if not tokens:
        return []
    
    tradeable_tokens = []
    checked_count = 0
    
    print(f"🔍 Filtering {len(tokens)} tokens for tradeability...")
    
    for token in tokens:
        if checked_count >= max_checks:
            print(f"⚠️ Reached max tradeability checks ({max_checks}), skipping remaining tokens")
            break
        
        is_tradeable, reason = is_token_tradeable(token)
        checked_count += 1
        
        if is_tradeable:
            tradeable_tokens.append(token)
            print(f"✅ {token.get('symbol', '?')} - tradeable ({reason})")
        else:
            print(f"❌ {token.get('symbol', '?')} - not tradeable ({reason})")
        
        # Small delay to avoid overwhelming APIs
        time.sleep(0.05)  # Reduced delay
    
    print(f"📊 Tradeability filter: {len(tradeable_tokens)}/{checked_count} tokens are tradeable")
    return tradeable_tokens

def quick_tradeability_check(token_address: str, chain_id: str = "solana") -> bool:
    """
    Quick tradeability check for a single token
    Returns True if tradeable, False otherwise
    """
    if chain_id.lower() == "solana":
        return check_jupiter_tradeability(token_address, chain_id)
    elif chain_id.lower() == "ethereum":
        return check_ethereum_tradeability(token_address, chain_id)
    else:
        return True  # Assume tradeable for other chains
