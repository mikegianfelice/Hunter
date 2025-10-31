#!/usr/bin/env python3
"""
Tradeability Checker - Pre-filter tokens to only include tradeable ones
"""

import requests
import time
from typing import Dict, List, Tuple, Optional
from src.utils.http_utils import get_json
from src.utils.address_utils import detect_chain_from_address, normalize_evm_address, validate_chain_address_match

# Circuit breaker for Raydium API failures
_raydium_circuit_breaker = {
    "failures": 0,
    "last_failure": 0,
    "is_open": False,
    "failure_threshold": 5,  # Open circuit after 5 consecutive failures
    "recovery_timeout": 300  # 5 minutes before trying again
}

def _check_circuit_breaker() -> bool:
    """Check if Raydium circuit breaker is open"""
    global _raydium_circuit_breaker
    
    if not _raydium_circuit_breaker["is_open"]:
        return False
    
    current_time = time.time()
    if current_time - _raydium_circuit_breaker["last_failure"] > _raydium_circuit_breaker["recovery_timeout"]:
        # Reset circuit breaker
        _raydium_circuit_breaker["is_open"] = False
        _raydium_circuit_breaker["failures"] = 0
        print("🔄 Raydium circuit breaker reset - trying API again")
        return False
    
    return True

def _record_circuit_breaker_failure():
    """Record a failure in the circuit breaker"""
    global _raydium_circuit_breaker
    
    _raydium_circuit_breaker["failures"] += 1
    _raydium_circuit_breaker["last_failure"] = time.time()
    
    if _raydium_circuit_breaker["failures"] >= _raydium_circuit_breaker["failure_threshold"]:
        _raydium_circuit_breaker["is_open"] = True
        print(f"⚠️ Raydium circuit breaker opened after {_raydium_circuit_breaker['failures']} failures")

def _record_circuit_breaker_success():
    """Record a success in the circuit breaker"""
    global _raydium_circuit_breaker
    
    _raydium_circuit_breaker["failures"] = 0
    _raydium_circuit_breaker["is_open"] = False

def _check_dexscreener_tradeability(token_address: str) -> bool:
    """Helper function to check tradeability via DexScreener"""
    try:
        url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
        data = get_json(url, timeout=12, retries=2, backoff=1.5)
        
        if data and data.get("pairs"):
            pairs = data["pairs"]
            if pairs:
                richest = max(pairs, key=lambda p: float((p.get("liquidity") or {}).get("usd") or 0))
                liquidity_usd = float((richest.get("liquidity") or {}).get("usd") or 0)
                txns = richest.get("txns", {}).get("h24", {})
                tx_count = int(txns.get("buys") or 0) + int(txns.get("sells") or 0)
                price_usd = float(richest.get("priceUsd") or 0)
                
                if liquidity_usd >= 15000 and tx_count >= 50 and price_usd > 0:
                    return True
        return False
    except Exception:
        return False

def check_jupiter_tradeability(token_address: str, chain_id: str = "solana") -> bool:
    """
    Check if a token is tradeable on Solana using real market data from DexScreener and Jupiter price API.
    Returns True if tradeable (has liquidity and trading activity), False otherwise.
    """
    if chain_id.lower() != "solana":
        return False  # Only check Solana tokens
    
    try:
        # Method 1: Check DexScreener for real trading pairs and liquidity
        url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
        data = get_json(url, timeout=12, retries=2, backoff=1.5)
        
        if data and data.get("pairs"):
            pairs = data["pairs"]
            if pairs:
                # Find the pair with highest liquidity
                richest = max(pairs, key=lambda p: float((p.get("liquidity") or {}).get("usd") or 0))
                
                liquidity_usd = float((richest.get("liquidity") or {}).get("usd") or 0)
                txns = richest.get("txns", {}).get("h24", {})
                tx_count = int(txns.get("buys") or 0) + int(txns.get("sells") or 0)
                price_usd = float(richest.get("priceUsd") or 0)
                
                # Real tradeability criteria: minimum liquidity and trading activity
                if liquidity_usd >= 15000 and tx_count >= 50 and price_usd > 0:
                    print(f"✅ Jupiter tradeability confirmed via DexScreener: ${liquidity_usd:,.0f} liquidity, {tx_count} txns/24h")
                    return True
        
        # Method 2: Fallback to Jupiter price API (no auth required for price endpoint)
        try:
            price_url = f"https://price.jup.ag/v4/price?ids={token_address}"
            price_data = get_json(price_url, timeout=10, retries=1, backoff=1.0)
            
            if price_data and price_data.get("data") and token_address in price_data["data"]:
                price_info = price_data["data"][token_address]
                price = float(price_info.get("price") or 0)
                
                if price > 0:
                    print(f"✅ Jupiter price available for {token_address[:8]}...{token_address[-8:]}: ${price}")
                    return True
        except Exception as e:
            print(f"⚠️ Jupiter price API check failed: {e}")
        
        # No valid trading data found
        print(f"❌ No tradeable pairs found for {token_address[:8]}...{token_address[-8:]}")
        return False
        
    except Exception as e:
        print(f"⚠️ Jupiter tradeability check failed for {token_address[:8]}...{token_address[-8:]}: {e}")
        return False  # Return False on error instead of assuming tradeable

def check_raydium_tradeability(token_address: str, chain_id: str = "solana") -> bool:
    """
    Check if a token is tradeable on Raydium with enhanced error handling and circuit breaker
    Returns True if tradeable, False otherwise
    """
    if chain_id.lower() != "solana":
        return False  # Only check Solana tokens
    
    # Check circuit breaker first
    if _check_circuit_breaker():
        print(f"⚠️ Raydium circuit breaker is open - using DexScreener fallback for {token_address[:8]}...{token_address[-8:]}")
        # Use DexScreener as fallback when circuit breaker is open
        return _check_dexscreener_tradeability(token_address)
    
    try:
        # Try to get token info from Raydium API with retry logic
        url = "https://api.raydium.io/v2/sdk/token/raydium.mainnet.json"
        
        try:
            data = get_json(url, timeout=10, retries=2, backoff=1.5)
            official_tokens = data.get("official", [])
            
            # Check if token is in official Raydium tokens
            for token_info in official_tokens:
                if token_info.get("mint", "").lower() == token_address.lower():
                    print(f"✅ Token {token_address[:8]}... found in official Raydium list")
                    _record_circuit_breaker_success()  # Record success
                    return True
        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 500:
                print(f"⚠️ Raydium token list API server error (500) for {token_address[:8]}...{token_address[-8:]}")
                _record_circuit_breaker_failure()  # Record failure
                # Continue to quote test despite 500 error
            else:
                print(f"⚠️ Raydium token list HTTP error for {token_address[:8]}...{token_address[-8:]}: {e.response.status_code if e.response else 'Unknown'}")
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            print(f"⚠️ Network error checking Raydium token list for {token_address[:8]}...{token_address[-8:]}: {type(e).__name__}")
        except Exception as e:
            print(f"⚠️ Error checking Raydium token list for {token_address[:8]}...{token_address[-8:]}: {e}")
        
        # If not in official list, try a quote test
        result = _test_raydium_quote(token_address)
        if result:
            _record_circuit_breaker_success()  # Record success
        else:
            _record_circuit_breaker_failure()  # Record failure
        return result
        
    except Exception as e:
        print(f"⚠️ Raydium tradeability check failed for {token_address[:8]}...{token_address[-8:]}: {e}")
        _record_circuit_breaker_failure()  # Record failure
        # Fallback to DexScreener instead of assuming tradeable
        return _check_dexscreener_tradeability(token_address)

def _test_raydium_quote(token_address: str) -> bool:
    """Test if we can get a quote from Raydium for this token with enhanced error handling"""
    try:
        # Try to get pool info for the token with retry logic and better error handling
        base_url = "https://api.raydium.io/v2/main/price"
        url = f"{base_url}?ids={token_address}"
        
        try:
            # Use http_utils with enhanced retry logic for 500 errors
            data = get_json(url, timeout=15, retries=2, backoff=2.0)
            if data.get("data") and token_address in data["data"]:
                _record_circuit_breaker_success()  # Record success
                return True
            return False
        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 500:
                print(f"⚠️ Raydium API server error (500) for {token_address[:8]}...{token_address[-8:]}, trying alternative endpoint...")
                _record_circuit_breaker_failure()  # Record failure
                # Try alternative Raydium endpoint
                return _test_raydium_alternative(token_address)
            else:
                print(f"⚠️ Raydium HTTP error for {token_address[:8]}...{token_address[-8:]}: {e.response.status_code if e.response else 'Unknown'}")
                _record_circuit_breaker_failure()  # Record failure
                return False
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            print(f"⚠️ Network error checking Raydium price for {token_address[:8]}...{token_address[-8:]}: {type(e).__name__}")
            _record_circuit_breaker_failure()  # Record failure
            return False
        except Exception as e:
            print(f"⚠️ Raydium quote test failed for {token_address[:8]}...{token_address[-8:]}: {e}")
            _record_circuit_breaker_failure()  # Record failure
            return False
        
    except Exception as e:
        print(f"⚠️ Raydium quote test failed for {token_address[:8]}...{token_address[-8:]}: {e}")
        _record_circuit_breaker_failure()  # Record failure
        return False

def _test_raydium_alternative(token_address: str) -> bool:
    """Try alternative Raydium endpoints when main price API fails"""
    try:
        # Try Raydium SDK quote endpoint as alternative
        url = "https://api.raydium.io/v2/sdk/quote"
        params = {
            "inputMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
            "outputMint": token_address,
            "amount": "1000000",  # 1 USDC
            "slippage": "0.02",
            "version": "4"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("success") and data.get("outAmount"):
                    print(f"✅ Raydium alternative quote successful for {token_address[:8]}...{token_address[-8:]}")
                    _record_circuit_breaker_success()  # Record success
                    return True
            elif response.status_code == 500:
                print(f"⚠️ Raydium SDK also returning 500 for {token_address[:8]}...{token_address[-8:]}")
                _record_circuit_breaker_failure()  # Record failure
                return False
            else:
                print(f"⚠️ Raydium SDK returned {response.status_code} for {token_address[:8]}...{token_address[-8:]}")
                _record_circuit_breaker_failure()  # Record failure
                return False
        except Exception as e:
            print(f"⚠️ Raydium SDK alternative failed for {token_address[:8]}...{token_address[-8:]}: {e}")
            _record_circuit_breaker_failure()  # Record failure
            return False
            
    except Exception as e:
        print(f"⚠️ Raydium alternative test failed for {token_address[:8]}...{token_address[-8:]}: {e}")
        _record_circuit_breaker_failure()  # Record failure
        return False

def check_ethereum_tradeability(token_address: str, chain_id: str = "ethereum") -> bool:
    """
    Check if a token is tradeable on EVM chains using real market data from DexScreener.
    Returns True if tradeable (has liquidity and trading activity), False otherwise.
    """
    if chain_id.lower() not in ["ethereum", "base", "arbitrum", "polygon", "bsc"]:
        return False  # Only check supported EVM chains
    
    try:
        # Use DexScreener to check real trading pairs and liquidity
        url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
        data = get_json(url, timeout=12, retries=2, backoff=1.5)
        
        if not data or not data.get("pairs"):
            print(f"❌ No trading pairs found for {token_address[:8]}...{token_address[-8:]} on {chain_id}")
            return False
        
        pairs = data["pairs"]
        # Filter pairs for the specific chain
        chain_pairs = [p for p in pairs if p.get("chainId", "").lower() == chain_id.lower()]
        
        if not chain_pairs:
            print(f"❌ No pairs found on {chain_id} for {token_address[:8]}...{token_address[-8:]}")
            return False
        
        # Find the pair with highest liquidity
        richest = max(chain_pairs, key=lambda p: float((p.get("liquidity") or {}).get("usd") or 0))
        
        liquidity_usd = float((richest.get("liquidity") or {}).get("usd") or 0)
        txns = richest.get("txns", {}).get("h24", {})
        tx_count = int(txns.get("buys") or 0) + int(txns.get("sells") or 0)
        price_usd = float(richest.get("priceUsd") or 0)
        
        # Real tradeability criteria: minimum liquidity and trading activity
        if liquidity_usd >= 20000 and tx_count >= 75 and price_usd > 0:
            print(f"✅ {chain_id} tradeability confirmed: ${liquidity_usd:,.0f} liquidity, {tx_count} txns/24h")
            return True
        else:
            print(f"❌ {chain_id} token insufficient: ${liquidity_usd:,.0f} liquidity, {tx_count} txns/24h")
            return False
        
    except Exception as e:
        print(f"⚠️ {chain_id} tradeability check failed for {token_address[:8]}...{token_address[-8:]}: {e}")
        return False  # Return False on error instead of assuming tradeable

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
    
    # Use standardized chain/address validation
    is_valid, corrected_chain, error_message = validate_chain_address_match(token_address, chain_id)
    
    if not is_valid:
        return False, f"invalid_chain_address: {error_message}"
    
    # Update chain if it was corrected
    if corrected_chain != chain_id:
        chain_id = corrected_chain
    
    # Normalize EVM addresses
    detected = detect_chain_from_address(token_address)
    if detected == "evm":
        token_address = normalize_evm_address(token_address)

    # Check tradeability based on chain using real market data
    if chain_id == "solana":
        # For Solana, check both Jupiter and Raydium
        jupiter_ok = check_jupiter_tradeability(token_address, chain_id)
        raydium_ok = check_raydium_tradeability(token_address, chain_id)
        
        if jupiter_ok or raydium_ok:
            return True, "tradeable_on_solana"
        else:
            # Both checks failed - token is not tradeable
            return False, "not_tradeable_on_solana"
    
    elif chain_id in ["ethereum", "base", "arbitrum", "polygon", "bsc"]:
        # Use real DexScreener check for all EVM chains
        is_tradeable = check_ethereum_tradeability(token_address, chain_id)
        if is_tradeable:
            return True, f"tradeable_on_{chain_id}"
        else:
            return False, f"not_tradeable_on_{chain_id}"
    
    else:
        # Unknown/unsupported chains - return False instead of assuming
        return False, f"unsupported_chain_{chain_id}"

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
    Quick tradeability check for a single token using real market data
    Returns True if tradeable, False otherwise
    """
    if chain_id.lower() == "solana":
        return check_jupiter_tradeability(token_address, chain_id)
    elif chain_id.lower() in ["ethereum", "base", "arbitrum", "polygon", "bsc"]:
        return check_ethereum_tradeability(token_address, chain_id)
    else:
        return False  # Return False for unsupported chains instead of assuming
