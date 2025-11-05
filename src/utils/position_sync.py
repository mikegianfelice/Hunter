"""
Utility to sync positions between performance_data.json and open_positions.json
Ensures positions are always tracked even if initial logging fails.
Now validates wallet balances before syncing to prevent manually closed positions from being re-added.
Supports wallet scanning to discover positions not tracked in performance_data.json.
"""
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List

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


def get_all_wallet_token_holdings(chain_id: str, min_balance_threshold: float = 0.000001) -> List[Dict]:
    """
    Scan wallet and get all token holdings for a specific chain.
    
    Args:
        chain_id: Chain to scan ("solana", "ethereum", "base")
        min_balance_threshold: Minimum balance to include (default: 0.000001)
    
    Returns:
        List of dicts with token info: [{"address": str, "balance": float, "symbol": str, "price_usd": float, "value_usd": float}]
    """
    holdings = []
    
    try:
        chain_lower = chain_id.lower()
        
        if chain_lower == "solana":
            from src.execution.jupiter_lib import JupiterCustomLib
            from src.config.secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY
            from src.utils.utils import fetch_token_price_usd
            
            lib = JupiterCustomLib(SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY)
            
            if not lib.keypair:
                print("⚠️ No Solana keypair available for wallet scan")
                return holdings
            
            # Get all token accounts for the wallet using getTokenAccountsByOwner (without mint filter)
            rpc_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTokenAccountsByOwner",
                "params": [
                    str(lib.keypair.pubkey()),
                    {
                        "programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"  # SPL Token Program
                    },
                    {
                        "encoding": "jsonParsed"
                    }
                ]
            }
            
            try:
                from src.utils.http_utils import post_json
                result = post_json(SOLANA_RPC_URL, rpc_payload, timeout=30, retries=2, backoff=1.0)
                
                if result and "result" in result and "value" in result["result"]:
                    accounts = result["result"]["value"]
                    print(f"🔍 Found {len(accounts)} token account(s) in Solana wallet")
                    
                    for account in accounts:
                        try:
                            account_info = account.get("account", {}).get("data", {}).get("parsed", {}).get("info", {})
                            token_amount = account_info.get("tokenAmount", {})
                            
                            mint = account_info.get("mint", "")
                            balance = float(token_amount.get("uiAmount", 0))
                            
                            # Skip zero or dust balances
                            if balance <= min_balance_threshold:
                                continue
                            
                            # Get token symbol and price
                            symbol = "UNKNOWN"
                            price_usd = 0.0
                            
                            # First, try to get symbol from existing performance_data.json (if token was previously traded)
                            try:
                                if PERFORMANCE_DATA_FILE.exists():
                                    with open(PERFORMANCE_DATA_FILE, "r") as f:
                                        perf_data_check = json.load(f)
                                        trades_check = perf_data_check.get("trades", [])
                                        # Find most recent trade with this address
                                        matching_trades = [t for t in trades_check 
                                                         if t.get("address", "").lower() == mint.lower()]
                                        if matching_trades:
                                            # Use symbol from most recent trade
                                            matching_trades.sort(key=lambda x: x.get("entry_time", ""), reverse=True)
                                            symbol = matching_trades[0].get("symbol", "UNKNOWN")
                            except Exception:
                                pass  # If we can't read performance_data, continue with UNKNOWN
                            
                            # Try to get price (for Solana tokens, try Solana-specific price fetcher)
                            try:
                                # Try Solana price fetcher first
                                from src.execution.solana_executor import get_token_price_usd as get_solana_price
                                price_usd = get_solana_price(mint)
                                
                                # If that fails, try the generic price fetcher
                                if not price_usd or price_usd <= 0:
                                    price_usd = fetch_token_price_usd(mint)
                                
                                # If we still don't have a symbol and have a price, use mint address as fallback
                                if symbol == "UNKNOWN" and price_usd and price_usd > 0:
                                    symbol = mint[:8] + "..." + mint[-8:]
                            except Exception as e:
                                print(f"⚠️ Could not get price for {mint[:8]}...{mint[-8:]}: {e}")
                                # Use mint address as symbol if we still don't have one
                                if symbol == "UNKNOWN":
                                    symbol = mint[:8] + "..." + mint[-8:]
                            
                            value_usd = balance * price_usd if price_usd > 0 else 0.0
                            
                            holdings.append({
                                "address": mint,
                                "balance": balance,
                                "symbol": symbol,
                                "price_usd": price_usd,
                                "value_usd": value_usd,
                                "chain_id": "solana"
                            })
                            
                        except Exception as e:
                            print(f"⚠️ Error parsing token account: {e}")
                            continue
                
            except Exception as e:
                print(f"⚠️ Error scanning Solana wallet: {e}")
                return holdings
        
        elif chain_lower in ["ethereum", "base"]:
            # For Ethereum/Base, we would need to:
            # 1. Query token transfers from/to wallet (via indexer)
            # 2. Or maintain a list of known tokens and check balances
            # For now, we'll skip this as it's more complex and requires external services
            print(f"⚠️ Wallet scanning for {chain_lower} not yet implemented")
            return holdings
        
        else:
            print(f"⚠️ Unsupported chain for wallet scanning: {chain_id}")
            return holdings
    
    except Exception as e:
        print(f"⚠️ Error in get_all_wallet_token_holdings: {e}")
        return holdings
    
    return holdings


def reconcile_wallet_with_positions(
    min_balance_threshold: float = 0.000001,
    discover_missing_positions: bool = True,
    auto_close_zero_balance: bool = True
) -> Dict:
    """
    Reconcile wallet holdings with tracked positions in open_positions.json and performance_data.json.
    
    Args:
        min_balance_threshold: Minimum balance to consider as a valid position
        discover_missing_positions: If True, add wallet tokens not in tracking
        auto_close_zero_balance: If True, mark positions with zero balance as closed
    
    Returns:
        Dict with reconciliation results: {"added": int, "removed": int, "updated": int, "errors": List[str]}
    """
    results = {
        "added": 0,
        "removed": 0,
        "updated": 0,
        "errors": []
    }
    
    try:
        # Load existing data
        if OPEN_POSITIONS_FILE.exists():
            try:
                with open(OPEN_POSITIONS_FILE, "r") as f:
                    open_positions = json.load(f) or {}
            except (json.JSONDecodeError, IOError):
                open_positions = {}
        else:
            open_positions = {}
        
        if PERFORMANCE_DATA_FILE.exists():
            try:
                with open(PERFORMANCE_DATA_FILE, "r") as f:
                    perf_data = json.load(f)
                    trades = perf_data.get("trades", [])
            except (json.JSONDecodeError, IOError):
                trades = []
                perf_data = {"trades": [], "daily_stats": {}}
        else:
            trades = []
            perf_data = {"trades": [], "daily_stats": {}}
        
        # Get all wallet holdings (for Solana - can extend to other chains)
        print("🔍 Scanning wallet for token holdings...")
        wallet_holdings = get_all_wallet_token_holdings("solana", min_balance_threshold)
        print(f"📊 Found {len(wallet_holdings)} token holding(s) in wallet")
        
        # Create a set of wallet token addresses for quick lookup
        wallet_addresses = {h["address"].lower() for h in wallet_holdings}
        
        # Create a mapping of address -> holding for quick access
        wallet_holdings_map = {h["address"].lower(): h for h in wallet_holdings}
        
        # Step 1: Check existing positions - remove those with zero balance
        if auto_close_zero_balance:
            positions_to_remove = []
            for position_key, position_data in list(open_positions.items()):
                if isinstance(position_data, dict):
                    chain_id = position_data.get("chain_id", "ethereum").lower()
                    token_address = position_data.get("address", position_key)
                    if "_" in position_key and not position_data.get("address"):
                        token_address = position_key.split("_")[0]
                    symbol = position_data.get("symbol", "?")
                else:
                    chain_id = "ethereum"
                    token_address = position_key
                    symbol = "?"
                
                # Check if token is in wallet
                token_address_lower = token_address.lower()
                if token_address_lower not in wallet_addresses:
                    # Check balance to confirm
                    balance = _check_token_balance_on_chain(token_address, chain_id)
                    if balance == 0.0 or (balance > 0 and balance < min_balance_threshold):
                        # Position closed - mark as removed
                        positions_to_remove.append(position_key)
                        
                        # Mark as closed in performance_data
                        trade_id = position_data.get("trade_id") if isinstance(position_data, dict) else None
                        _mark_trade_as_closed_in_performance_data(token_address, chain_id, trade_id)
                        
                        print(f"✅ Removed position: {symbol} ({token_address[:8]}...{token_address[-8:]}) - zero balance")
            
            for key in positions_to_remove:
                open_positions.pop(key, None)
                results["removed"] += 1
        
        # Step 2: Discover missing positions from wallet
        if discover_missing_positions:
            for holding in wallet_holdings:
                token_address = holding["address"]
                token_address_lower = token_address.lower()
                balance = holding["balance"]
                price_usd = holding.get("price_usd", 0.0)
                symbol = holding.get("symbol", "UNKNOWN")
                chain_id = holding.get("chain_id", "solana")
                
                # Check if this token is already tracked
                already_tracked = False
                for position_key, position_data in open_positions.items():
                    if isinstance(position_data, dict):
                        pos_address = position_data.get("address", position_key).lower()
                        if "_" in position_key and not position_data.get("address"):
                            pos_address = position_key.split("_")[0].lower()
                    else:
                        pos_address = position_key.lower()
                    
                    if pos_address == token_address_lower:
                        already_tracked = True
                        break
                
                if not already_tracked:
                    # New position discovered - add it
                    # Use current price as entry price (best guess)
                    entry_price = price_usd if price_usd > 0 else 0.0
                    position_size_usd = balance * price_usd if price_usd > 0 else 0.0
                    
                    # Create position key
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    position_key = f"{token_address}_{symbol}_{timestamp_str}"
                    
                    # Add to open_positions
                    position_data = {
                        "entry_price": entry_price,
                        "chain_id": chain_id,
                        "symbol": symbol,
                        "address": token_address,
                        "timestamp": datetime.now().isoformat(),
                        "position_size_usd": position_size_usd,
                        "discovered": True,  # Flag to indicate this was auto-discovered
                        "entry_price_estimated": True  # Flag to indicate entry price is estimated
                    }
                    
                    open_positions[position_key] = position_data
                    
                    # Add to performance_data as an open trade
                    trade_id = f"{symbol}_{timestamp_str}"
                    trade = {
                        "id": trade_id,
                        "symbol": symbol,
                        "address": token_address,
                        "chain": chain_id,
                        "entry_time": datetime.now().isoformat(),
                        "entry_price": entry_price,
                        "position_size_usd": position_size_usd,
                        "quality_score": 0.0,  # Unknown quality for discovered positions
                        "volume_24h": 0.0,
                        "liquidity": 0.0,
                        "exit_time": None,
                        "exit_price": None,
                        "pnl_usd": None,
                        "pnl_percent": None,
                        "status": "open",
                        "take_profit_target": None,
                        "stop_loss_target": None,
                        "discovered": True,  # Flag to indicate this was auto-discovered
                        "entry_price_estimated": True
                    }
                    
                    trades.append(trade)
                    perf_data["trades"] = trades
                    
                    # Update position key to include trade_id
                    position_data["trade_id"] = trade_id
                    # Update the key to use trade_id format
                    new_position_key = f"{token_address}_{trade_id}"
                    open_positions[new_position_key] = position_data
                    if position_key != new_position_key:
                        open_positions.pop(position_key, None)
                    
                    results["added"] += 1
                    print(f"✅ Discovered new position: {symbol} ({token_address[:8]}...{token_address[-8:]}) - balance: {balance:.6f}, value: ${position_size_usd:.2f}")
        
        # Save updated files
        if results["added"] > 0 or results["removed"] > 0:
            # Save open_positions
            temp_file = OPEN_POSITIONS_FILE.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(open_positions, f, indent=2)
            temp_file.replace(OPEN_POSITIONS_FILE)
            
            # Save performance_data
            perf_data["last_updated"] = datetime.now().isoformat()
            temp_file = PERFORMANCE_DATA_FILE.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(perf_data, f, indent=2)
            temp_file.replace(PERFORMANCE_DATA_FILE)
            
            print(f"✅ Reconciliation complete: {results['added']} added, {results['removed']} removed")
    
    except Exception as e:
        error_msg = f"Error in reconcile_wallet_with_positions: {e}"
        print(f"⚠️ {error_msg}")
        results["errors"].append(error_msg)
        import traceback
        print(traceback.format_exc())
    
    return results


def update_open_positions_from_wallet() -> Dict:
    """
    Wrapper function to update open_positions.json from wallet holdings.
    Uses configuration values for thresholds and options.
    """
    try:
        from src.config.config_loader import get_config, get_config_float, get_config_bool
        
        # Get nested config values
        pos_recon = get_config("position_reconciliation", {})
        if isinstance(pos_recon, dict):
            min_balance = pos_recon.get("min_balance_threshold", 0.000001)
            discover_missing = pos_recon.get("discover_missing_positions", True)
            auto_close = pos_recon.get("auto_close_zero_balance", True)
        else:
            # Fallback to defaults if config structure is wrong
            min_balance = 0.000001
            discover_missing = True
            auto_close = True
        
        return reconcile_wallet_with_positions(
            min_balance_threshold=float(min_balance),
            discover_missing_positions=bool(discover_missing),
            auto_close_zero_balance=bool(auto_close)
        )
    except Exception as e:
        print(f"⚠️ Error in update_open_positions_from_wallet: {e}")
        import traceback
        print(traceback.format_exc())
        return {"added": 0, "removed": 0, "updated": 0, "errors": [str(e)]}


if __name__ == "__main__":
    # Run sync when called directly (with balance verification enabled by default)
    print("🔄 Running position sync from performance_data.json...")
    results = sync_all_open_positions(verify_balances=True)
    if results:
        print(f"✅ Synced {sum(1 for v in results.values() if v)} position(s)")
    else:
        print("No positions to sync")
    
    # Also run wallet reconciliation
    print("\n🔄 Running wallet reconciliation...")
    recon_results = update_open_positions_from_wallet()
    print(f"✅ Reconciliation: {recon_results['added']} added, {recon_results['removed']} removed")

