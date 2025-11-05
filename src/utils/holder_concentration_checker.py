"""
Holder Concentration Checker
Checks the percentage of tokens held by top 10 holders to detect potential rug pull risks
"""

import requests
import time
from typing import Dict, Optional
from src.config.secrets import SOLANA_RPC_URL
from src.config.config_loader import get_config, get_config_int, get_config_float, get_config_bool
from src.utils.advanced_cache import cache_get, cache_set


def check_holder_concentration(token_address: str, chain_id: str = "solana") -> Dict:
    """
    Check holder concentration for a token by analyzing top 10 holders percentage
    
    Args:
        token_address: Token mint address (for Solana) or contract address
        chain_id: Chain identifier ("solana", "ethereum", "base")
    
    Returns:
        Dict with:
            - top_10_percentage: float - Percentage owned by top 10 holders
            - is_safe: bool - Whether concentration is below threshold
            - risk_level: str - "low", "medium", "high", "critical"
            - error: Optional[str] - Error message if check failed
            - total_supply: Optional[float] - Total token supply
            - top_10_balance: Optional[float] - Combined balance of top 10 holders
    """
    # Check cache first
    cache_key = f"holder_concentration_{chain_id}_{token_address}"
    cache_ttl = get_config_int("holder_concentration_cache_minutes", 10) * 60  # Convert to seconds
    
    cached_result = cache_get(cache_key)
    if cached_result is not None:
        return cached_result
    
    # Initialize result structure
    result = {
        "top_10_percentage": 0.0,
        "is_safe": True,
        "risk_level": "unknown",
        "error": None,
        "total_supply": None,
        "top_10_balance": None
    }
    
    try:
        if chain_id.lower() == "solana":
            result = _check_solana_holder_concentration(token_address)
        elif chain_id.lower() in ["ethereum", "base"]:
            # Ethereum/Base support can be added later
            result["error"] = f"Holder concentration check not yet implemented for {chain_id}"
            result["is_safe"] = True  # Fail-safe: allow trading if check unavailable
            return result
        else:
            result["error"] = f"Unsupported chain: {chain_id}"
            result["is_safe"] = True  # Fail-safe
            return result
        
        # Determine risk level
        threshold = get_config_float("holder_concentration_threshold", 60.0)
        
        if result.get("error"):
            # If there was an error, fail-safe (allow trading)
            result["is_safe"] = True
            result["risk_level"] = "unknown"
        else:
            percentage = result["top_10_percentage"]
            
            if percentage >= threshold:
                result["is_safe"] = False
                result["risk_level"] = "critical"
            elif percentage >= threshold * 0.8:
                result["is_safe"] = False
                result["risk_level"] = "high"
            elif percentage >= threshold * 0.6:
                result["is_safe"] = True
                result["risk_level"] = "medium"
            else:
                result["is_safe"] = True
                result["risk_level"] = "low"
        
        # Cache the result
        cache_set(cache_key, result, ttl=cache_ttl)
        
        return result
        
    except Exception as e:
        # Fail-safe: allow trading if check fails
        result["error"] = str(e)
        result["is_safe"] = True
        result["risk_level"] = "unknown"
        return result


def _check_solana_holder_concentration(token_address: str) -> Dict:
    """
    Check holder concentration for Solana token using RPC calls
    
    Uses getTokenLargestAccounts to get top 10 holders
    Uses getTokenSupply to get total supply
    """
    result = {
        "top_10_percentage": 0.0,
        "is_safe": True,
        "risk_level": "unknown",
        "error": None,
        "total_supply": None,
        "top_10_balance": None
    }
    
    try:
        rpc_url = SOLANA_RPC_URL or "https://api.mainnet-beta.solana.com"
        
        # Step 1: Get total token supply
        supply_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTokenSupply",
            "params": [token_address]
        }
        
        supply_response = requests.post(rpc_url, json=supply_payload, timeout=15)
        
        if supply_response.status_code != 200:
            result["error"] = f"RPC request failed: HTTP {supply_response.status_code}"
            return result
        
        supply_data = supply_response.json()
        
        if "error" in supply_data:
            result["error"] = f"RPC error getting supply: {supply_data['error'].get('message', 'Unknown error')}"
            return result
        
        if "result" not in supply_data or "value" not in supply_data["result"]:
            result["error"] = "Invalid RPC response structure for token supply"
            return result
        
        total_supply_ui = float(supply_data["result"]["value"]["uiAmount"] or 0)
        total_supply_raw = int(supply_data["result"]["value"]["amount"] or 0)
        
        if total_supply_ui == 0 or total_supply_raw == 0:
            result["error"] = "Token supply is zero or invalid"
            return result
        
        result["total_supply"] = total_supply_ui
        
        # Step 2: Get top 10 largest accounts
        accounts_payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "getTokenLargestAccounts",
            "params": [
                token_address,
                {
                    "limit": 10
                }
            ]
        }
        
        accounts_response = requests.post(rpc_url, json=accounts_payload, timeout=15)
        
        if accounts_response.status_code != 200:
            result["error"] = f"RPC request failed: HTTP {accounts_response.status_code}"
            return result
        
        accounts_data = accounts_response.json()
        
        if "error" in accounts_data:
            result["error"] = f"RPC error getting largest accounts: {accounts_data['error'].get('message', 'Unknown error')}"
            return result
        
        if "result" not in accounts_data or "value" not in accounts_data["result"]:
            result["error"] = "Invalid RPC response structure for largest accounts"
            return result
        
        largest_accounts = accounts_data["result"]["value"]
        
        if not largest_accounts or len(largest_accounts) == 0:
            result["error"] = "No holder accounts found"
            return result
        
        # Calculate total balance of top 10 holders
        top_10_balance_raw = 0
        for account in largest_accounts:
            balance_raw = int(account.get("amount", 0))
            top_10_balance_raw += balance_raw
        
        # Convert to UI amount (considering decimals)
        decimals = supply_data["result"]["value"].get("decimals", 0)
        top_10_balance_ui = top_10_balance_raw / (10 ** decimals) if decimals > 0 else top_10_balance_raw
        
        result["top_10_balance"] = top_10_balance_ui
        
        # Calculate percentage
        if total_supply_ui > 0:
            percentage = (top_10_balance_ui / total_supply_ui) * 100
            result["top_10_percentage"] = round(percentage, 2)
        else:
            result["error"] = "Cannot calculate percentage: total supply is zero"
            return result
        
        return result
        
    except requests.exceptions.Timeout:
        result["error"] = "RPC request timeout"
        return result
    except requests.exceptions.RequestException as e:
        result["error"] = f"RPC request error: {str(e)}"
        return result
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"
        return result

