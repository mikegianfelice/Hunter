"""
Utility to sync positions between performance_data.json and open_positions.json
Ensures positions are always tracked even if initial logging fails.
Now validates wallet balances before syncing to prevent manually closed positions from being re-added.
"""
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PERFORMANCE_DATA_FILE = PROJECT_ROOT / "data" / "performance_data.json"
OPEN_POSITIONS_FILE = PROJECT_ROOT / "data" / "open_positions.json"


def _check_token_balance_on_chain(token_address: str, chain_id: str) -> float:
    """
    Check token balance on the specified chain.
    Returns balance amount (0.0 if balance is zero, -1.0 if check failed).
    """
    try:
        chain_lower = chain_id.lower()
        
        if chain_lower == "solana":
            from src.execution.jupiter_lib import JupiterCustomLib
            from src.config.secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY
            lib = JupiterCustomLib(SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY)
            balance = lib.get_token_balance(token_address)
            # None means check failed (error) - return -1.0 to indicate unknown state
            if balance is None:
                return -1.0
            return float(balance)
            
        elif chain_lower == "base":
            from src.execution.base_executor import get_token_balance
            balance = get_token_balance(token_address)
            return float(balance or 0.0)
            
        elif chain_lower == "ethereum":
            # Use web3 to check ERC20 token balance
            from web3 import Web3
            from src.config.secrets import INFURA_URL, WALLET_ADDRESS
            
            w3 = Web3(Web3.HTTPProvider(INFURA_URL))
            if not w3.is_connected():
                return -1.0
            
            # ERC20 ABI minimal - just balanceOf and decimals
            erc20_abi = json.loads("""[
                {"constant":true,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"type":"function"},
                {"constant":true,"inputs":[{"name":"owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"type":"function"}
            ]""")
            
            wallet = Web3.to_checksum_address(WALLET_ADDRESS)
            token_addr = Web3.to_checksum_address(token_address)
            token_contract = w3.eth.contract(address=token_addr, abi=erc20_abi)
            
            balance_wei = token_contract.functions.balanceOf(wallet).call()
            decimals = token_contract.functions.decimals().call()
            balance = float(balance_wei) / (10 ** decimals)
            return balance
            
        else:
            # For other chains, return -1.0 to indicate check not implemented
            return -1.0
            
    except Exception as e:
        # If balance check fails, return -1.0 to indicate unknown state
        return -1.0


def sync_position_from_performance_data(token_address: str, symbol: str, chain_id: str, entry_price: float, position_size_usd: float = None) -> bool:
    """Manually sync a position to open_positions.json (legacy function for backward compatibility)"""
    # Use address as key for backward compatibility
    return sync_position_from_performance_data_with_key(
        token_address, token_address, symbol, chain_id, entry_price, position_size_usd
    )


def sync_position_from_performance_data_with_key(position_key: str, token_address: str, symbol: str, chain_id: str, entry_price: float, position_size_usd: float = None, trade_id: str = None, verify_balance: bool = True) -> bool:
    """
    Manually sync a position to open_positions.json with a custom key.
    If verify_balance is True (default), checks wallet balance before syncing.
    Returns False if balance check fails (zero balance or check error).
    """
    try:
        # CRITICAL: Check wallet balance BEFORE syncing to prevent manually closed positions from being added
        if verify_balance:
            balance = _check_token_balance_on_chain(token_address, chain_id)
            
            if balance == -1.0:
                # Balance check failed - skip to be safe (don't add if we can't verify)
                print(f"⏸️  Skipping sync for {symbol} ({token_address[:8]}...{token_address[-8:]}) - balance check failed")
                return False
            elif balance <= 0.0 or balance < 0.000001:
                # Zero or dust balance - position was manually closed, don't sync it
                print(f"🚫 Skipping sync for {symbol} ({token_address[:8]}...{token_address[-8:]}) - zero/dust balance ({balance:.8f}) detected (manually closed)")
                # Mark as closed in performance_data to prevent future syncs
                _mark_trade_as_closed_in_performance_data(token_address, chain_id, trade_id)
                return False
            else:
                # Has balance - safe to sync
                print(f"✓ {symbol} ({token_address[:8]}...{token_address[-8:]}) has balance {balance:.6f} - syncing")
        # If position_size_usd not provided, try to get it from performance_data.json
        if position_size_usd is None:
            try:
                if PERFORMANCE_DATA_FILE.exists():
                    with open(PERFORMANCE_DATA_FILE, "r") as f:
                        perf_data = json.load(f)
                        trades = perf_data.get("trades", [])
                        # Find matching trade
                        if trade_id:
                            matching_trades = [t for t in trades if t.get("id") == trade_id]
                        else:
                            matching_trades = [t for t in trades 
                                             if t.get("address", "").lower() == token_address.lower() 
                                             and t.get("status") == "open"]
                        if matching_trades:
                            # Sort by entry_time (most recent first)
                            matching_trades.sort(key=lambda x: x.get("entry_time", ""), reverse=True)
                            position_size_usd = matching_trades[0].get("position_size_usd", 0.0)
            except Exception:
                pass
        
        OPEN_POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing positions
        if OPEN_POSITIONS_FILE.exists():
            try:
                with open(OPEN_POSITIONS_FILE, "r") as f:
                    positions = json.load(f) or {}
            except (json.JSONDecodeError, IOError):
                positions = {}
        else:
            positions = {}
        
        # Add or update position
        position_data = {
            "entry_price": float(entry_price),
            "chain_id": chain_id.lower(),
            "symbol": symbol,
            "address": token_address,  # Store original address for compatibility
            "timestamp": datetime.now().isoformat()
        }
        
        # Include trade_id if available
        if trade_id:
            position_data["trade_id"] = trade_id
        
        # Include position_size_usd if available
        if position_size_usd is not None and position_size_usd > 0:
            position_data["position_size_usd"] = float(position_size_usd)
        
        positions[position_key] = position_data
        
        # Atomic write
        temp_file = OPEN_POSITIONS_FILE.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(positions, f, indent=2)
        temp_file.replace(OPEN_POSITIONS_FILE)
        
        size_str = f" (${position_size_usd:.2f})" if position_size_usd else ""
        trade_str = f" [Trade: {trade_id}]" if trade_id else ""
        print(f"📝 Synced position: {symbol} ({token_address[:8]}...{token_address[-8:]}) on {chain_id.upper()} @ ${entry_price:.6f}{size_str}{trade_str}")
        return True
    except Exception as e:
        print(f"⚠️ Failed to sync position: {e}")
        return False


def _mark_trade_as_closed_in_performance_data(token_address: str, chain_id: str, trade_id: str = None):
    """Mark a trade as manually closed in performance_data.json"""
    try:
        if not PERFORMANCE_DATA_FILE.exists():
            return
        
        with open(PERFORMANCE_DATA_FILE, "r") as f:
            perf_data = json.load(f)
        
        trades = perf_data.get("trades", [])
        updated = False
        
        for trade in trades:
            if trade.get("status") == "open":
                # Match by trade_id if available, otherwise by address and chain
                if trade_id and trade.get("id") == trade_id:
                    trade['status'] = 'manual_close'
                    trade['exit_time'] = datetime.now().isoformat()
                    trade['exit_price'] = 0.0
                    trade['pnl_usd'] = 0.0
                    trade['pnl_percent'] = 0.0
                    updated = True
                    break
                elif trade.get("address", "").lower() == token_address.lower() and trade.get("chain", "").lower() == chain_id.lower():
                    trade['status'] = 'manual_close'
                    trade['exit_time'] = datetime.now().isoformat()
                    trade['exit_price'] = 0.0
                    trade['pnl_usd'] = 0.0
                    trade['pnl_percent'] = 0.0
                    updated = True
                    break
        
        if updated:
            # Save updated performance_data
            temp_file = PERFORMANCE_DATA_FILE.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(perf_data, f, indent=2)
            temp_file.replace(PERFORMANCE_DATA_FILE)
            print(f"📝 Marked trade as manually closed in performance_data.json")
    except Exception as e:
        print(f"⚠️ Failed to mark trade as closed: {e}")


def sync_all_open_positions(verify_balances: bool = True) -> Dict[str, bool]:
    """
    Sync all open positions from performance_data.json to open_positions.json.
    If verify_balances is True (default), only syncs positions that have on-chain balances.
    Returns dict of {token_address: success_bool}
    """
    results = {}
    
    try:
        if not PERFORMANCE_DATA_FILE.exists():
            return results
        
        # Load performance data
        try:
            with open(PERFORMANCE_DATA_FILE, "r") as f:
                perf_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ Failed to read performance_data.json: {e}")
            return results
        
        # Load existing open positions
        if OPEN_POSITIONS_FILE.exists():
            try:
                with open(OPEN_POSITIONS_FILE, "r") as f:
                    open_positions = json.load(f) or {}
            except (json.JSONDecodeError, IOError):
                open_positions = {}
        else:
            open_positions = {}
        
        # Find all open trades in performance_data
        # Track ALL open positions, including multiple positions for the same token
        trades = perf_data.get("trades", [])
        open_trades = [t for t in trades if t.get("status") == "open"]
        
        synced_count = 0
        skipped_no_balance = 0
        for trade in open_trades:
            address = trade.get("address", "")
            if not address:
                continue
            
            symbol = trade.get("symbol", "?")
            chain = trade.get("chain", "ethereum").lower()
            entry_price = float(trade.get("entry_price", 0))
            position_size_usd = trade.get("position_size_usd", 0.0)
            trade_id = trade.get("id", "")
            entry_time = trade.get("entry_time", "")
            
            if entry_price <= 0:
                continue
            
            # Create composite key to support multiple positions per token
            # Use trade_id if available, otherwise use address_entrytime
            if trade_id:
                # Use trade_id as part of the key to make it unique
                # Format: "address_tradeid" for tracking, but keep address accessible
                position_key = f"{address}_{trade_id}"
            else:
                # Fallback: use address with entry_time
                position_key = f"{address}_{entry_time.replace(':', '-')}"
            
            # Skip if already tracked
            if position_key in open_positions:
                continue
            
            # CRITICAL: Check balance BEFORE syncing if verify_balances is enabled
            if verify_balances:
                balance = _check_token_balance_on_chain(address, chain)
                
                if balance == -1.0:
                    # Balance check failed - skip to be safe (don't add if we can't verify)
                    print(f"⏸️  Skipping sync for {symbol} ({address[:8]}...{address[-8:]}) - balance check failed")
                    skipped_no_balance += 1
                    continue
                elif balance <= 0.0 or balance < 0.000001:
                    # Zero or dust balance - position was manually closed, don't sync it
                    print(f"🚫 Skipping sync for {symbol} ({address[:8]}...{address[-8:]}) - zero/dust balance ({balance:.8f}) detected (manually closed)")
                    # Mark as closed in performance_data to prevent future syncs
                    _mark_trade_as_closed_in_performance_data(address, chain, trade_id)
                    skipped_no_balance += 1
                    continue
                else:
                    # Has balance - safe to sync
                    print(f"✓ {symbol} ({address[:8]}...{address[-8:]}) has balance {balance:.6f} - syncing")
            
            # Sync position with composite key (verify_balance is passed through)
            success = sync_position_from_performance_data_with_key(
                position_key, address, symbol, chain, entry_price, position_size_usd, trade_id, verify_balance=verify_balances
            )
            results[position_key] = success
            if success:
                synced_count += 1
            
            # Brief delay to avoid file lock issues
            time.sleep(0.1)
        
        if synced_count > 0:
            print(f"✅ Synced {synced_count} position(s) from performance_data.json to open_positions.json")
        if skipped_no_balance > 0:
            print(f"🚫 Skipped {skipped_no_balance} position(s) with zero balance (manually closed)")
        
        return results
        
    except Exception as e:
        print(f"⚠️ Error syncing positions: {e}")
        return results


if __name__ == "__main__":
    # Run sync when called directly (with balance verification enabled by default)
    results = sync_all_open_positions(verify_balances=True)
    if results:
        print(f"Synced {sum(1 for v in results.values() if v)} position(s)")
    else:
        print("No positions to sync")

