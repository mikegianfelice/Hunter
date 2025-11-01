#!/usr/bin/env python3
"""
Preflight health check system
Validates secrets, RPC connectivity, and basic functionality before startup
"""

from typing import Dict, Any, List
from ..monitoring.structured_logger import log_info, log_error, log_warning
from ..config.secrets import (
    SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY,
    WALLET_ADDRESS, PRIVATE_KEY, INFURA_URL
)


async def check_solana_readiness() -> Dict[str, Any]:
    """Check if Solana trading infrastructure is ready"""
    result = {
        "ready": False,
        "checks": {},
        "errors": []
    }
    
    # Check if secrets are available
    if not SOLANA_WALLET_ADDRESS or not SOLANA_PRIVATE_KEY:
        result["errors"].append("Missing Solana wallet address or private key")
        result["checks"]["secrets"] = False
        return result
    
    result["checks"]["secrets"] = True
    
    # Check RPC connectivity
    try:
        import requests
        rpc_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getHealth"
        }
        response = requests.post(SOLANA_RPC_URL, json=rpc_payload, timeout=5)
        if response.status_code == 200:
            result["checks"]["rpc_connectivity"] = True
        else:
            result["checks"]["rpc_connectivity"] = False
            result["errors"].append(f"Solana RPC returned status {response.status_code}")
    except Exception as e:
        result["checks"]["rpc_connectivity"] = False
        result["errors"].append(f"Solana RPC connectivity failed: {str(e)}")
    
    # Check wallet balance (lightweight check)
    try:
        from ..execution.jupiter_executor import get_solana_balance
        balance = get_solana_balance()
        if balance is not None and balance >= 0:
            result["checks"]["wallet_balance"] = True
            result["balance"] = balance
            if balance < 0.01:
                result["errors"].append(f"Low SOL balance: {balance:.4f} SOL")
        else:
            result["checks"]["wallet_balance"] = False
            result["errors"].append("Could not fetch wallet balance")
    except Exception as e:
        result["checks"]["wallet_balance"] = False
        result["errors"].append(f"Wallet balance check failed: {str(e)}")
    
    # Test Jupiter quote capability
    try:
        from ..execution.jupiter_executor import JupiterCustomExecutor
        executor = JupiterCustomExecutor()
        # Just verify the executor can be initialized
        if executor.jupiter_lib:
            result["checks"]["jupiter_executor"] = True
        else:
            result["checks"]["jupiter_executor"] = False
            result["errors"].append("Jupiter executor initialization failed")
    except Exception as e:
        result["checks"]["jupiter_executor"] = False
        result["errors"].append(f"Jupiter executor check failed: {str(e)}")
    
    # Determine overall readiness
    all_checks_passed = all(result["checks"].values())
    result["ready"] = all_checks_passed and len(result["errors"]) == 0
    
    return result


async def check_evm_readiness() -> Dict[str, Any]:
    """Check if EVM chains are ready (Ethereum, Base, etc.)"""
    result = {
        "ready": False,
        "checks": {},
        "errors": []
    }
    
    # Check if secrets are available
    if not WALLET_ADDRESS or not PRIVATE_KEY:
        result["errors"].append("Missing EVM wallet address or private key")
        result["checks"]["secrets"] = False
        return result
    
    result["checks"]["secrets"] = True
    
    # Check RPC connectivity
    try:
        import requests
        rpc_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_blockNumber",
            "params": []
        }
        response = requests.post(INFURA_URL, json=rpc_payload, timeout=5)
        if response.status_code == 200:
            result["checks"]["rpc_connectivity"] = True
        else:
            result["checks"]["rpc_connectivity"] = False
            result["errors"].append(f"EVM RPC returned status {response.status_code}")
    except Exception as e:
        result["checks"]["rpc_connectivity"] = False
        result["errors"].append(f"EVM RPC connectivity failed: {str(e)}")
    
    result["ready"] = all(result["checks"].values()) and len(result["errors"]) == 0
    return result


async def run_preflight_checks(chains: List[str] = None) -> Dict[str, Any]:
    """
    Run comprehensive preflight checks for trading readiness
    
    Args:
        chains: List of chains to check (e.g., ["solana", "ethereum", "base"])
                If None, checks all configured chains
    
    Returns:
        Dict with overall status and detailed results per chain
    """
    if chains is None:
        chains = ["solana", "ethereum", "base"]
    
    log_info("preflight.start", "🔍 Running preflight health checks...")
    
    results = {
        "overall_ready": True,
        "chains": {},
        "warnings": [],
        "errors": []
    }
    
    # Check Solana if in chains
    if "solana" in chains:
        log_info("preflight.solana", "Checking Solana readiness...")
        solana_result = await check_solana_readiness()
        results["chains"]["solana"] = solana_result
        
        if not solana_result["ready"]:
            results["overall_ready"] = False
            results["errors"].extend([f"Solana: {err}" for err in solana_result["errors"]])
        
        if solana_result.get("balance") is not None:
            balance = solana_result["balance"]
            if balance < 0.01:
                results["warnings"].append(f"Solana wallet has low balance: {balance:.4f} SOL")
            
            log_info("preflight.solana", 
                    f"Solana ready: {solana_result['ready']}, Balance: {balance:.4f} SOL" if solana_result['ready'] else 
                    f"Solana not ready: {', '.join(solana_result['errors'])}")
    
    # Check EVM chains if in chains
    evm_chains = [c for c in chains if c in ["ethereum", "base", "arbitrum", "polygon"]]
    if evm_chains:
        log_info("preflight.evm", f"Checking EVM chains readiness ({', '.join(evm_chains)})...")
        evm_result = await check_evm_readiness()
        results["chains"]["evm"] = evm_result
        
        if not evm_result["ready"]:
            results["overall_ready"] = False
            results["errors"].extend([f"EVM: {err}" for err in evm_result["errors"]])
        
        else:
            log_info("preflight.evm", "EVM chains ready")
    
    # Summary
    if results["overall_ready"]:
        log_info("preflight.success", "✅ All preflight checks passed")
    else:
        log_error("preflight.failed", f"❌ Preflight checks failed: {len(results['errors'])} error(s)")
        for error in results["errors"]:
            log_error("preflight.error_detail", f"  - {error}")
    
    if results["warnings"]:
        for warning in results["warnings"]:
            log_warning("preflight.warning", f"⚠️  {warning}")
    
    return results

