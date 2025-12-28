#!/usr/bin/env python3
"""
Jupiter Token Verification - Check if tokens are verified on Jupiter
Uses Jupiter's static token list (no API calls needed after initial fetch)
"""

import json
import time
import requests
from pathlib import Path
from typing import Set, Optional, Dict
from datetime import datetime, timedelta

# Cache file location
CACHE_DIR = Path("data")
CACHE_FILE = CACHE_DIR / "jupiter_token_list.json"
CACHE_TTL_HOURS = 24  # Refresh token list daily

# Jupiter token list endpoints
JUPITER_STRICT_LIST_URL = "https://token.jup.ag/strict"
JUPITER_ALL_LIST_URL = "https://token.jup.ag/all"


def _load_cached_token_list() -> Optional[Dict]:
    """Load cached Jupiter token list"""
    if not CACHE_FILE.exists():
        return None
    
    try:
        with open(CACHE_FILE, 'r') as f:
            data = json.load(f)
            # Check if cache is still valid
            cache_time = datetime.fromisoformat(data.get('cached_at', '2000-01-01'))
            if datetime.now() - cache_time < timedelta(hours=CACHE_TTL_HOURS):
                return data
    except Exception as e:
        print(f"⚠️ Error loading Jupiter token list cache: {e}")
    
    return None


def _save_token_list_cache(token_addresses: Set[str], source: str = "strict"):
    """Save token list to cache"""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            'token_addresses': list(token_addresses),
            'count': len(token_addresses),
            'source': source,
            'cached_at': datetime.now().isoformat()
        }
        with open(CACHE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Cached {len(token_addresses)} Jupiter verified tokens")
    except Exception as e:
        print(f"⚠️ Error saving Jupiter token list cache: {e}")


def _fetch_jupiter_token_list(use_strict: bool = True) -> Set[str]:
    """
    Fetch Jupiter token list from their API
    Returns set of token addresses (mint addresses)
    """
    url = JUPITER_STRICT_LIST_URL if use_strict else JUPITER_ALL_LIST_URL
    token_addresses = set()
    
    try:
        print(f"🔄 Fetching Jupiter {'strict' if use_strict else 'all'} token list...")
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            tokens = response.json()
            
            # Jupiter token list format: list of objects with 'address' or 'mint' field
            for token in tokens:
                # Handle different possible formats
                address = token.get('address') or token.get('mint') or token.get('mintAddress')
                if address:
                    token_addresses.add(address)
            
            print(f"✅ Fetched {len(token_addresses)} tokens from Jupiter")
            return token_addresses
        else:
            print(f"⚠️ Jupiter API returned status {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ Error fetching Jupiter token list: {e}")
    
    return token_addresses


def get_jupiter_verified_tokens(force_refresh: bool = False) -> Set[str]:
    """
    Get set of Jupiter-verified token addresses
    Uses cache if available and fresh, otherwise fetches from Jupiter
    """
    # Try cache first (unless force refresh)
    if not force_refresh:
        cached = _load_cached_token_list()
        if cached:
            addresses = set(cached.get('token_addresses', []))
            print(f"✅ Using cached Jupiter token list ({len(addresses)} tokens)")
            return addresses
    
    # Fetch fresh list
    token_addresses = _fetch_jupiter_token_list(use_strict=True)
    
    if token_addresses:
        _save_token_list_cache(token_addresses, source="strict")
        return token_addresses
    else:
        # Fallback to cache even if stale
        cached = _load_cached_token_list()
        if cached:
            addresses = set(cached.get('token_addresses', []))
            print(f"⚠️ Using stale cache ({len(addresses)} tokens) - Jupiter fetch failed")
            return addresses
    
    return set()


def is_token_verified_jupiter(token_address: str, chain_id: str = "solana") -> bool:
    """
    Check if a token is verified on Jupiter
    
    Args:
        token_address: Token mint address
        chain_id: Chain identifier (only 'solana' is supported)
    
    Returns:
        True if token is verified on Jupiter, False otherwise
    """
    if chain_id.lower() != "solana":
        # Jupiter only supports Solana
        return False
    
    if not token_address:
        return False
    
    # Normalize address (case-insensitive comparison)
    token_address = token_address.strip()
    
    # Get verified tokens
    verified_tokens = get_jupiter_verified_tokens()
    
    # Check if token is in the list (case-insensitive)
    is_verified = token_address in verified_tokens or token_address.lower() in {t.lower() for t in verified_tokens}
    
    return is_verified


def refresh_jupiter_token_list() -> int:
    """
    Force refresh of Jupiter token list
    Returns number of tokens in the refreshed list
    """
    token_addresses = _fetch_jupiter_token_list(use_strict=True)
    if token_addresses:
        _save_token_list_cache(token_addresses, source="strict")
        return len(token_addresses)
    return 0
