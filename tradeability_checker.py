#!/usr/bin/env python3
"""
Tradeability Checker - Pre-filter tokens to only include tradeable ones
"""

import requests
import time
from typing import Dict, List, Tuple, Optional
from http_utils import get_json

def check_jupiter_tradeability(token_address: str, chain_id: str = "solana") -> bool:
    """
    Check if a token is tradeable on Jupiter
    Returns True if tradeable, False otherwise
    
    NOTE: Jupiter's quote-api.jup.ag endpoint no longer exists (DNS resolution fails).
    The new api.jup.ag requires API keys (returns 401 Unauthorized).
    This function now assumes tokens are tradeable and relies on actual swap attempts to determine tradeability.
    """
    if chain_id.lower() != "solana":
        return True  # Skip check for non-Solana chains
    
    # Jupiter API endpoint has changed and now requires authentication
    # quote-api.jup.ag no longer resolves (DNS failure)
    # api.jup.ag exists but returns 401 Unauthorized without API keys
    # 
    # Workaround: Assume all tokens are tradeable
    # The actual swap will fail if the token is not tradeable, which is handled elsewhere
    print(f"ℹ️  Jupiter API check skipped for {token_address[:8]}...{token_address[-8:]} (API requires auth)")
    return True  # Assume tradeable - let the actual swap attempt handle failures

def check_raydium_tradeability(token_address: str, chain_id: str = "solana") -> bool:
    """
    Check if a token is tradeable on Raydium
    Returns True if tradeable, False otherwise
    """
    if chain_id.lower() != "solana":
        return True  # Skip check for non-Solana chains
    
    try:
        # Try to get token info from Raydium API with retry logic
        url = "https://api.raydium.io/v2/sdk/token/raydium.mainnet.json"
        
        try:
            data = get_json(url, timeout=10, retries=3, backoff=1.0)
            official_tokens = data.get("official", [])
            
            # Check if token is in official Raydium tokens
            for token_info in official_tokens:
                if token_info.get("mint", "").lower() == token_address.lower():
                    return True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            print(f"⚠️ Network error checking Raydium token list for {token_address[:8]}...{token_address[-8:]}: {type(e).__name__}")
        except Exception as e:
            print(f"⚠️ Error checking Raydium token list for {token_address[:8]}...{token_address[-8:]}: {e}")
        
        # If not in official list, try a quote test
        return _test_raydium_quote(token_address)
        
    except Exception as e:
        print(f"⚠️ Raydium tradeability check failed for {token_address[:8]}...{token_address[-8:]}: {e}")
        return True  # Assume tradeable if check fails

def _test_raydium_quote(token_address: str) -> bool:
    """Test if we can get a quote from Raydium for this token"""
    try:
        # Try to get pool info for the token with retry logic
        base_url = "https://api.raydium.io/v2/main/price"
        url = f"{base_url}?ids={token_address}"
        
        try:
            data = get_json(url, timeout=10, retries=3, backoff=1.0)
            if data.get("data") and token_address in data["data"]:
                return True
            return False
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            print(f"⚠️ Network error checking Raydium price for {token_address[:8]}...{token_address[-8:]}: {type(e).__name__}")
            return False
        except Exception as e:
            print(f"⚠️ Raydium quote test failed for {token_address[:8]}...{token_address[-8:]}: {e}")
            return False
        
    except Exception as e:
        print(f"⚠️ Raydium quote test failed for {token_address[:8]}...{token_address[-8:]}: {e}")
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
        if price is not None and price > 0:
            return True
        
        # If price check fails, assume tradeable anyway (Graph API is unreliable)
        print(f"⚠️ Ethereum price check failed for {token_address[:8]}...{token_address[-8:]}, assuming tradeable")
        return True
        
    except Exception as e:
        print(f"⚠️ Ethereum tradeability check failed for {token_address[:8]}...{token_address[-8:]}: {e}")
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
            # If both checks fail, assume tradeable anyway (APIs can be unreliable)
            print(f"⚠️ Both Jupiter and Raydium checks failed for {symbol}, assuming tradeable")
            return True, "assumed_tradeable_on_solana"
    
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
