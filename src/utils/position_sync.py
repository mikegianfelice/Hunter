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
from typing import Any, Dict, Optional, List, Tuple, Iterable
from src.storage.positions import load_positions as load_positions_store, replace_positions
from src.storage.performance import load_performance_data, replace_performance_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PERFORMANCE_DATA_FILE = PROJECT_ROOT / "data" / "performance_data.json"
OPEN_POSITIONS_FILE = PROJECT_ROOT / "data" / "open_positions.json"

_EXCLUDED_MINTS_BY_CHAIN = {
    "solana": {
        "so11111111111111111111111111111111111111112",  # SOL / wSOL
        "epjfwdd5aufqssqem2qn1xzybapc8g4weggkzwytdt1v",  # USDC
        "es9vmfrzacermjfrf4h2fyd4kconky11mcce8benwnyb",  # USDT
    },
}

_EXCLUDED_SYMBOLS_BY_CHAIN = {
    "solana": {"SOL", "WSOL", "USDC", "USDT"},
}


def is_native_gas_token(
    token_address: Optional[str],
    symbol: Optional[str],
    chain_id: Optional[str],
) -> bool:
    """
    Return True when the token represents a native gas asset that we should never
    track as an open position (e.g., SOL used for fees).
    """
    chain_lower = (chain_id or "").strip().lower()
    address_lower = (token_address or "").strip().lower()
    symbol_upper = (symbol or "").strip().upper()

    excluded_mints = _EXCLUDED_MINTS_BY_CHAIN.get(chain_lower, set())
    excluded_symbols = _EXCLUDED_SYMBOLS_BY_CHAIN.get(chain_lower, set())

    if address_lower and address_lower in excluded_mints:
        return True
    if symbol_upper and symbol_upper in excluded_symbols:
        return True

    return False

def create_position_key(
    token_address: str,
    *,
    trade_id: Optional[str] = None,
    entry_time: Optional[str] = None,
) -> str:
    """
    Create the canonical key for entries in open_positions.json.

    Keys are the normalized token mint/address (lowercase). The optional arguments
    remain for backward compatibility with older call sites but no longer affect
    the generated key.
    """
    address = (token_address or "").strip()

    if not address:
        raise ValueError("token_address is required to build a position key")

    return address.lower()


def split_position_key(position_key: str) -> Tuple[str, Optional[str]]:
    """
    Split a legacy composite key back into address and suffix.

    Mint-only keys simply return (key, None). The legacy suffix (trade/timestamp)
    is preserved for migration scenarios.
    """
    if "_" not in position_key:
        return position_key, None

    address, suffix = position_key.split("_", 1)
    return address, suffix


def resolve_token_address(position_key: str, position_data: Any) -> str:
    """Resolve the token address represented by a position entry."""
    if isinstance(position_data, dict):
        explicit = position_data.get("address")
        if explicit:
            return explicit

    address, _ = split_position_key(position_key)
    return address


def find_position_key_by_address(token_address: str, chain_id: Optional[str] = None) -> Optional[str]:
    """
    Find position key by token address. Handles both canonical and composite keys.
    Returns the key if found, None otherwise.
    """
    positions = load_positions_store()
    token_address_lower = token_address.lower()
    
    for key, pos_data in positions.items():
        if isinstance(pos_data, dict):
            # Check chain_id if provided
            if chain_id:
                pos_chain = (pos_data.get("chain_id") or "").lower()
                if pos_chain != chain_id.lower():
                    continue
            
            # Check address
            pos_addr = (pos_data.get("address") or key).lower()
            if pos_addr == token_address_lower:
                return key
        else:
            # Legacy format - check key directly
            if key.lower() == token_address_lower:
                return key
    
    return None


def _check_token_balance_on_chain(token_address: str, chain_id: str, use_cache: bool = True) -> float:
    """
    Check token balance on the specified chain with caching support.
    Returns balance amount (0.0 if balance is zero, -1.0 if check failed).
    
    Args:
        token_address: Token contract address
        chain_id: Chain identifier
        use_cache: If True, use cached balance if available and fresh
    """
    # Try cache first if enabled
    if use_cache:
        from src.utils.balance_cache import get_cached_token_balance, update_token_balance_cache
        from src.config.config_loader import get_config_bool
        
        if get_config_bool("balance_cache_enabled", True):
            cached_balance, is_valid = get_cached_token_balance(token_address, chain_id)
            if is_valid:
                return float(cached_balance)
    
    try:
        chain_lower = chain_id.lower()
        balance = None
        
        if chain_lower == "solana":
            from src.execution.jupiter_lib import JupiterCustomLib
            from src.config.secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY
            lib = JupiterCustomLib(SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY)
            balance = lib.get_token_balance(token_address)
            # None means check failed (error) - return -1.0 to indicate unknown state
            if balance is None:
                return -1.0
            balance = float(balance)
            
        elif chain_lower == "base":
            from src.execution.base_executor import get_token_balance
            balance = get_token_balance(token_address)
            balance = float(balance or 0.0)
            
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
            
        else:
            # For other chains, return -1.0 to indicate check not implemented
            return -1.0
        
        # Cache successful balance check
        if balance is not None and use_cache:
            from src.utils.balance_cache import update_token_balance_cache
            from src.config.config_loader import get_config_bool
            if get_config_bool("balance_cache_enabled", True):
                update_token_balance_cache(token_address, chain_id, balance)
        
        return balance
            
    except Exception as e:
        # If balance check fails, return -1.0 to indicate unknown state
        return -1.0


def sync_position_from_performance_data(
    token_address: str,
    symbol: str,
    chain_id: str,
    entry_price: float,
    position_size_usd: float = None,
) -> bool:
    """Manually sync a position to open_positions.json (legacy function)."""

    trade_id = None
    entry_time = None
    try:
        perf_data = load_performance_data()
        trades = perf_data.get("trades", [])
        matching_trades = [
            t
            for t in trades
            if t.get("address", "").lower() == token_address.lower()
            and t.get("status") == "open"
        ]
        if matching_trades:
            matching_trades.sort(key=lambda x: x.get("entry_time", ""), reverse=True)
            latest_trade = matching_trades[0]
            trade_id = latest_trade.get("id")
            entry_time = latest_trade.get("entry_time")
            if position_size_usd is None:
                position_size_usd = latest_trade.get("position_size_usd", 0.0)
    except Exception:
        pass

    position_key = create_position_key(
        token_address,
        trade_id=trade_id,
        entry_time=entry_time,
    )

    return sync_position_from_performance_data_with_key(
        position_key,
        token_address,
        symbol,
        chain_id,
        entry_price,
        position_size_usd,
        trade_id,
        entry_time,
    )


def sync_position_from_performance_data_with_key(
    position_key: str,
    token_address: str,
    symbol: str,
    chain_id: str,
    entry_price: float,
    position_size_usd: float = None,
    trade_id: str = None,
    entry_time: Optional[str] = None,
    verify_balance: bool = True,
) -> bool:
    """
    Manually sync a position to open_positions.json with a custom key.
    If verify_balance is True (default), checks wallet balance before syncing.
    Returns False if balance check fails (zero balance or check error).
    """
    try:
        # Align key with canonical schema (mint-only)
        canonical_key = create_position_key(
            token_address,
            trade_id=trade_id,
            entry_time=entry_time,
        )
        if position_key != canonical_key:
            position_key = canonical_key

        OPEN_POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Load existing positions once for both skip and update flows.
        positions = _load_open_positions()

        # Never track native gas/base tokens (e.g., SOL / USDC on Solana).
        if is_native_gas_token(token_address, symbol, chain_id):
            print(
                f"⛽️ Skipping native gas token {symbol or token_address} "
                f"on {chain_id.upper()} - not tracked in open_positions"
            )

            removed = False
            normalized_address = (token_address or "").lower()
            # Remove any existing entries that map to the same token.
            for existing_key, existing_value in list(positions.items()):
                resolved_address = resolve_token_address(existing_key, existing_value).lower()
                if normalized_address and resolved_address == normalized_address:
                    positions.pop(existing_key, None)
                    removed = True

            if removed:
                _save_open_positions(positions)
                print(f"🧹 Removed native gas token entry {token_address} from open_positions.json")

            _mark_trade_as_closed_in_performance_data(token_address, chain_id, trade_id)
            return False

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
                perf_data = load_performance_data()
                trades = perf_data.get("trades", [])
                if trade_id:
                    matching_trades = [t for t in trades if t.get("id") == trade_id]
                else:
                    matching_trades = [
                        t
                        for t in trades
                        if t.get("address", "").lower() == token_address.lower()
                        and t.get("status") == "open"
                    ]
                if matching_trades:
                    matching_trades.sort(key=lambda x: x.get("entry_time", ""), reverse=True)
                    position_size_usd = matching_trades[0].get("position_size_usd", 0.0)
            except Exception:
                pass
        
        # Add or update position
        existing_entry = positions.get(position_key)

        timestamp_value = entry_time
        if not timestamp_value and isinstance(existing_entry, dict):
            timestamp_value = existing_entry.get("timestamp")
        if not timestamp_value:
            timestamp_value = datetime.now().isoformat()

        # CRITICAL: Preserve original_entry_price if it exists (for partial TP positions)
        # Only update entry_price if original_entry_price doesn't exist OR if explicitly updating entry price
        if isinstance(existing_entry, dict) and existing_entry.get("original_entry_price"):
            # Position has partial TP history - preserve original_entry_price
            position_data = {
                "entry_price": float(existing_entry.get("entry_price", entry_price)),  # Preserve existing entry_price
                "original_entry_price": float(existing_entry["original_entry_price"]),  # Always preserve
                "chain_id": chain_id.lower(),
                "symbol": symbol,
                "address": token_address,  # Store original address for compatibility
                "timestamp": timestamp_value,
            }
            
            # Check if entry_price is being changed significantly (more than 1%)
            existing_entry_price = existing_entry.get("entry_price", 0)
            if existing_entry_price > 0:
                price_diff_pct = abs(float(entry_price) - existing_entry_price) / existing_entry_price
                if price_diff_pct > 0.01:  # More than 1% difference
                    print(f"⚠️ WARNING: Entry price changed significantly for {symbol}: ${existing_entry_price:.6f} -> ${entry_price:.6f} (diff: {price_diff_pct*100:.2f}%)")
                    print(f"   Preserving original_entry_price: ${existing_entry['original_entry_price']:.6f}")
        else:
            # New position or no partial TP history - use provided entry_price
            position_data = {
                "entry_price": float(entry_price),
                "chain_id": chain_id.lower(),
                "symbol": symbol,
                "address": token_address,  # Store original address for compatibility
                "timestamp": timestamp_value,
            }
        
        # Include trade_id if available
        if trade_id:
            position_data["trade_id"] = trade_id
        elif isinstance(existing_entry, dict) and existing_entry.get("trade_id"):
            position_data["trade_id"] = existing_entry.get("trade_id")
        
        # Include position_size_usd if available
        if position_size_usd is not None and position_size_usd > 0:
            position_data["position_size_usd"] = float(position_size_usd)
        elif isinstance(existing_entry, dict) and existing_entry.get("position_size_usd"):
            position_data["position_size_usd"] = float(existing_entry.get("position_size_usd"))

        # Store entry volume if not already present (for volume deterioration tracking)
        if "entry_volume_24h_avg" not in position_data:
            try:
                from src.utils.market_data_fetcher import MarketDataFetcher
                market_fetcher = MarketDataFetcher()
                volume_24h = market_fetcher.get_token_volume_cached(token_address, chain_id)
                if volume_24h and volume_24h > 0:
                    position_data["entry_volume_24h_avg"] = float(volume_24h)
            except Exception as e:
                # Non-critical - log but don't fail
                print(f"⚠️ Could not fetch entry volume for {symbol}: {e}")

        # Preserve additional metadata flags when present (including partial TP flags)
        if isinstance(existing_entry, dict):
            for key in ("discovered", "entry_price_estimated", "entry_volume_24h_avg", 
                       "partial_sell_taken", "partial_sell_pct", "partial_sell_tx", 
                       "partial_sell_price", "partial_sell_time"):
                if key in existing_entry and key not in position_data:
                    position_data[key] = existing_entry[key]
        
        # Remove any legacy keys that map to the same token address
        address_lower = token_address.lower()
        for existing_key, existing_value in list(positions.items()):
            if existing_key == position_key:
                continue
            if resolve_token_address(existing_key, existing_value).lower() == address_lower:
                positions.pop(existing_key, None)

        positions[position_key] = position_data
        
        # Atomic write
        _save_open_positions(positions)
        
        size_str = f" (${position_size_usd:.2f})" if position_size_usd else ""
        trade_str = f" [Trade: {trade_id}]" if trade_id else ""
        print(f"📝 Synced position: {symbol} ({token_address[:8]}...{token_address[-8:]}) on {chain_id.upper()} @ ${entry_price:.6f}{size_str}{trade_str}")
        return True
    except Exception as e:
        print(f"⚠️ Failed to sync position: {e}")
        return False


def _mark_trade_as_closed_in_performance_data(token_address: str, chain_id: str, trade_id: str = None):
    """Mark a trade as manually closed in performance storage"""
    try:
        perf_data = load_performance_data()
        trades = perf_data.get("trades", [])
        updated = False

        for trade in trades:
            if trade.get("status") == "open":
                if trade_id and trade.get("id") == trade_id:
                    _close_trade_record(trade)
                    updated = True
                    break
                elif trade.get("address", "").lower() == token_address.lower() and trade.get("chain", "").lower() == chain_id.lower():
                    _close_trade_record(trade)
                    updated = True
                    break

        if updated:
            perf_data["trades"] = trades
            perf_data.setdefault("daily_stats", perf_data.get("daily_stats", {}))
            perf_data["last_updated"] = datetime.now().isoformat()
            replace_performance_data(perf_data)
            print(f"📝 Marked trade as manually closed in performance history")
    except Exception as e:
        print(f"⚠️ Failed to mark trade as closed: {e}")


def _close_trade_record(trade: Dict[str, Any]) -> None:
    trade['status'] = 'manual_close'
    trade['exit_time'] = datetime.now().isoformat()
    trade['exit_price'] = 0.0
    trade['pnl_usd'] = 0.0
    trade['pnl_percent'] = 0.0


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
            perf_data = load_performance_data()
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ Failed to read performance_data.json: {e}")
            return results
        
        # Load existing open positions
        positions = _load_open_positions()
        
        # Find all open trades in performance data
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
            position_key = create_position_key(
                address,
                trade_id=trade_id,
                entry_time=entry_time,
            )
            
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
                position_key,
                address,
                symbol,
                chain,
                entry_price,
                position_size_usd,
                trade_id,
                entry_time,
                verify_balance=verify_balances,
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
    perf_data_snapshot = {}
    try:
        perf_data_snapshot = load_performance_data()
    except Exception:
        perf_data_snapshot = {}

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
                            
                            # First, try to get symbol from stored performance history (if token was previously traded)
                            trades_check = perf_data_snapshot.get("trades", [])
                            matching_trades = [
                                t for t in trades_check if t.get("address", "").lower() == mint.lower()
                            ]
                            if matching_trades:
                                matching_trades.sort(key=lambda x: x.get("entry_time", ""), reverse=True)
                                symbol = matching_trades[0].get("symbol", "UNKNOWN")

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
        open_positions = _load_open_positions()
        perf_data = load_performance_data()
        trades = perf_data.get("trades", [])

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
                    symbol = position_data.get("symbol", "?")
                else:
                    chain_id = "ethereum"
                    symbol = "?"

                token_address = resolve_token_address(position_key, position_data)
                
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
                # Get position data before removing (needed for cache invalidation)
                position_data = open_positions.get(key)
                open_positions.pop(key, None)
                results["removed"] += 1
                
                # Invalidate cache for removed position
                if position_data:
                    try:
                        from src.utils.balance_cache import invalidate_token_balance_cache
                        token_addr = resolve_token_address(key, position_data)
                        chain = position_data.get("chain_id", "solana") if isinstance(position_data, dict) else "solana"
                        invalidate_token_balance_cache(token_addr, chain)
                    except Exception as e:
                        print(f"⚠️ Failed to invalidate cache for {key}: {e}")
        
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
                    pos_address = resolve_token_address(position_key, position_data).lower()
                    if pos_address == token_address_lower:
                        already_tracked = True
                        break
                
                if not already_tracked:
                    # New position discovered - add it
                    # Use current price as entry price (best guess)
                    entry_price = price_usd if price_usd > 0 else 0.0
                    position_size_usd = balance * price_usd if price_usd > 0 else 0.0

                    timestamp = datetime.now()
                    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
                    entry_time = timestamp.isoformat()
                    trade_id = f"{symbol}_{timestamp_str}"
                    position_key = create_position_key(token_address, trade_id=trade_id)

                    # Add to open_positions
                    position_data = {
                        "entry_price": entry_price,
                        "chain_id": chain_id,
                        "symbol": symbol,
                        "address": token_address,
                        "timestamp": entry_time,
                        "position_size_usd": position_size_usd,
                        "discovered": True,  # Flag to indicate this was auto-discovered
                        "entry_price_estimated": True,  # Flag to indicate entry price is estimated
                        "trade_id": trade_id,
                    }
                    
                    # Store entry volume for volume deterioration tracking
                    try:
                        from src.utils.market_data_fetcher import MarketDataFetcher
                        market_fetcher = MarketDataFetcher()
                        volume_24h = market_fetcher.get_token_volume_cached(token_address, chain_id)
                        if volume_24h and volume_24h > 0:
                            position_data["entry_volume_24h_avg"] = float(volume_24h)
                    except Exception as e:
                        # Non-critical - log but don't fail
                        print(f"⚠️ Could not fetch entry volume for {symbol}: {e}")

                    open_positions[position_key] = position_data

                    # Add to performance_data as an open trade
                    trade = {
                        "id": trade_id,
                        "symbol": symbol,
                        "address": token_address,
                        "chain": chain_id,
                        "entry_time": entry_time,
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
 
                    results["added"] += 1
                    print(f"✅ Discovered new position: {symbol} ({token_address[:8]}...{token_address[-8:]}) - balance: {balance:.6f}, value: ${position_size_usd:.2f}")
        
        # Save updated files
        if results["added"] > 0 or results["removed"] > 0:
            _save_open_positions(open_positions)
            perf_data["last_updated"] = datetime.now().isoformat()
            replace_performance_data(perf_data)

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


def migrate_open_positions_to_canonical_keys(dry_run: bool = False) -> Dict[str, int]:
    """Re-key open_positions.json entries using the canonical schema."""

    stats = {"rekeyed": 0, "deduped": 0, "converted": 0, "errors": 0, "skipped_gas_tokens": 0}

    if not OPEN_POSITIONS_FILE.exists():
        return stats

    try:
        with open(OPEN_POSITIONS_FILE, "r") as fh:
            positions = json.load(fh) or {}
    except (json.JSONDecodeError, OSError) as exc:
        stats["errors"] += 1
        print(f"⚠️ Failed to load open_positions.json for migration: {exc}")
        return stats

    perf_trades: List[Dict[str, Any]] = []
    try:
        perf_payload = load_performance_data()
        perf_trades = perf_payload.get("trades", []) or []
    except Exception:
        perf_trades = []

    def _match_trade_id(address: str, chain_id: str) -> Optional[str]:
        candidates = [
            t
            for t in perf_trades
            if (t.get("address", "").lower() == address.lower())
            and (t.get("status", "").lower() == "open")
            and (not chain_id or t.get("chain", "").lower() == chain_id.lower())
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda x: x.get("entry_time", ""), reverse=True)
        return candidates[0].get("id")

    def _timestamp_weight(ts: Optional[str]) -> float:
        if not ts:
            return 0.0
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0

    def _merge_position_entries(preferred: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
        merged: Dict[str, Any] = dict(fallback or {})
        for key, value in (preferred or {}).items():
            if value is None or value == "":
                if key not in merged or merged.get(key) in (None, ""):
                    merged[key] = value
            else:
                merged[key] = value
        return merged

    new_positions: Dict[str, Any] = {}

    for original_key, raw_value in positions.items():
        if isinstance(raw_value, dict):
            position_data = dict(raw_value)
        else:
            stats["converted"] += 1
            position_data = {
                "entry_price": float(raw_value or 0.0),
                "chain_id": "ethereum",
                "symbol": "?",
            }

        address = resolve_token_address(original_key, position_data)
        suffix = split_position_key(original_key)[1]
        chain_id = position_data.get("chain_id", "ethereum")

        if is_native_gas_token(address, position_data.get("symbol"), chain_id):
            stats["skipped_gas_tokens"] += 1
            continue

        trade_id = position_data.get("trade_id")
        if not trade_id:
            trade_id = _match_trade_id(address, chain_id)
            if trade_id:
                position_data["trade_id"] = trade_id

        entry_time = position_data.get("timestamp") or suffix

        position_data["address"] = address

        canonical_key = create_position_key(address)

        if canonical_key != original_key:
            stats["rekeyed"] += 1

        existing = new_positions.get(canonical_key)
        if isinstance(existing, dict):
            stats["deduped"] += 1
            existing_ts = existing.get("timestamp")
            candidate_ts = position_data.get("timestamp")
            if _timestamp_weight(candidate_ts) >= _timestamp_weight(existing_ts):
                new_positions[canonical_key] = _merge_position_entries(position_data, existing)
            else:
                new_positions[canonical_key] = _merge_position_entries(existing, position_data)
        else:
            new_positions[canonical_key] = position_data

    if dry_run or new_positions == positions:
        return stats

    _save_open_positions(new_positions)

    return stats


def _load_open_positions() -> Dict[str, Any]:
    return load_positions_store()


def _save_open_positions(positions: Dict[str, Any]) -> None:
    replace_positions(positions)


def reconcile_position_sizes(
    *,
    threshold_pct: float = 5.0,
    min_balance_threshold: float = 1e-6,
    chains: Optional[Iterable[str]] = None,
    verify_balance: bool = True,
    dry_run: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Recalculate position_size_usd from on-chain balance * current price and
    update open_positions.json, performance_data.json, and hunter_state.db.

    Args:
        threshold_pct: Minimum percentage difference to trigger update (default 5%)
        min_balance_threshold: Minimum balance to consider as valid position (default 1e-6)
        chains: Optional list of chains to process (None = all chains)
        verify_balance: Whether to verify on-chain balance (default True)
        dry_run: If True, don't persist changes (default False)
        verbose: If True, print detailed logs (default False)

    Returns:
        Dict with stats: {"updated": int, "closed": int, "skipped": int, "errors": List[str]}
    """
    stats = {"updated": 0, "closed": 0, "skipped": 0, "errors": []}
    chains_set = {c.lower() for c in chains} if chains else None

    try:
        positions = load_positions_store()
        perf = load_performance_data()
        trades = perf.get("trades", [])
    except Exception as e:
        stats["errors"].append(f"load_error: {e}")
        if verbose:
            print(f"❌ Failed to load data: {e}")
        return stats

    updated_positions = dict(positions)
    any_change = False

    for pos_key, pos_data in list(positions.items()):
        try:
            if not isinstance(pos_data, dict):
                continue

            chain_id = pos_data.get("chain_id", "ethereum").lower()
            if chains_set and chain_id not in chains_set:
                stats["skipped"] += 1
                continue

            token_addr = resolve_token_address(pos_key, pos_data)
            symbol = pos_data.get("symbol", "?")

            if is_native_gas_token(token_addr, symbol, chain_id):
                stats["skipped"] += 1
                continue

            # Check balance
            balance = None
            if verify_balance:
                balance = _check_token_balance_on_chain(token_addr, chain_id)
                if balance is None or balance == -1.0:
                    # RPC check failed - try Helius as fallback (for Solana)
                    if chain_id.lower() == "solana":
                        try:
                            from src.utils.helius_client import HeliusClient
                            from src.config.secrets import HELIUS_API_KEY, SOLANA_WALLET_ADDRESS
                            if HELIUS_API_KEY and SOLANA_WALLET_ADDRESS:
                                if verbose:
                                    print(f"[fallback-helius] {symbol} {token_addr[:8]}...{token_addr[-8:]} - RPC failed, trying Helius...")
                                client = HeliusClient(HELIUS_API_KEY)
                                balances = client.get_address_balances(SOLANA_WALLET_ADDRESS)
                                tokens = balances.get("tokens", [])
                                for token in tokens:
                                    if (token.get("mint") or "").lower() == token_addr.lower():
                                        amount_raw = token.get("amount") or 0
                                        decimals = token.get("decimals") or 0
                                        balance = float(amount_raw) / (10 ** decimals) if decimals else float(amount_raw)
                                        if verbose:
                                            print(f"[fallback-helius-success] {symbol} balance={balance:.8f}")
                                        break
                        except Exception as helius_e:
                            if verbose:
                                print(f"[skip-balance] {symbol} {token_addr[:8]}...{token_addr[-8:]} - both RPC and Helius failed: {helius_e}")
                    
                    # If still no balance after fallback, skip (don't remove on uncertainty)
                    if balance is None or balance == -1.0:
                        stats["skipped"] += 1
                        if verbose:
                            print(f"[skip-balance] {symbol} {token_addr[:8]}...{token_addr[-8:]} - balance check failed")
                        continue

            if balance is not None and (balance <= 0 or balance < min_balance_threshold):
                # Position closed - remove from open_positions
                updated_positions.pop(pos_key, None)
                trade_id = pos_data.get("trade_id")
                _mark_trade_as_closed_in_performance_data(token_addr, chain_id, trade_id)
                
                # Invalidate cache for closed position
                try:
                    from src.utils.balance_cache import invalidate_token_balance_cache
                    invalidate_token_balance_cache(token_addr, chain_id)
                except Exception as e:
                    if verbose:
                        print(f"⚠️ Failed to invalidate cache: {e}")
                
                stats["closed"] += 1
                any_change = True
                if verbose:
                    print(f"[close] {symbol} {token_addr[:8]}...{token_addr[-8:]} - balance={balance:.8f} (zero/dust)")
                continue

            # Fetch current price
            price = 0.0
            try:
                if chain_id == "solana":
                    from src.execution.solana_executor import get_token_price_usd
                    price = float(get_token_price_usd(token_addr) or 0.0)
                    if price <= 0:
                        from src.utils.utils import fetch_token_price_usd
                        price = float(fetch_token_price_usd(token_addr) or 0.0)
                else:
                    from src.utils.utils import fetch_token_price_usd
                    price = float(fetch_token_price_usd(token_addr) or 0.0)
            except Exception as e:
                if verbose:
                    print(f"[skip-price] {symbol} {token_addr[:8]}...{token_addr[-8:]} - price fetch error: {e}")
                price = 0.0

            if price <= 0:
                stats["skipped"] += 1
                if verbose:
                    print(f"[skip-price] {symbol} {token_addr[:8]}...{token_addr[-8:]} - price unavailable")
                continue

            # Calculate actual position value
            if balance is None:
                # If balance verification was skipped, try to get it now
                balance = _check_token_balance_on_chain(token_addr, chain_id)
                if balance is None or balance == -1.0 or balance <= 0:
                    stats["skipped"] += 1
                    if verbose:
                        print(f"[skip-balance] {symbol} {token_addr[:8]}...{token_addr[-8:]} - cannot get balance")
                    continue

            actual_usd = balance * price
            logged_usd = float(pos_data.get("position_size_usd", 0.0) or 0.0)

            if logged_usd <= 0:
                # No logged size - always update
                diff_ratio = 1.0
            else:
                diff_ratio = abs(actual_usd - logged_usd) / logged_usd

            # Only update if discrepancy exceeds threshold
            if diff_ratio * 100 < threshold_pct:
                stats["skipped"] += 1
                if verbose:
                    print(f"[skip-threshold] {symbol} {token_addr[:8]}...{token_addr[-8:]} - diff {diff_ratio*100:.2f}% < {threshold_pct}%")
                continue

            # Update position data
            pos_data["position_size_usd"] = actual_usd
            pos_data["last_reconciled_at"] = datetime.now().isoformat()
            updated_positions[pos_key] = pos_data
            any_change = True
            stats["updated"] += 1

            # Update performance_data.json for matching open trade
            trade_id = pos_data.get("trade_id")
            for t in trades:
                if t.get("status") == "open" and (
                    (trade_id and t.get("id") == trade_id)
                    or (t.get("address", "").lower() == token_addr.lower() and t.get("chain", "").lower() == chain_id)
                ):
                    t["position_size_usd"] = actual_usd
                    if t.get("entry_amount_usd_actual") is None or t.get("entry_amount_usd_actual") == 0:
                        t["entry_amount_usd_actual"] = actual_usd
                    break

            if verbose:
                print(f"[update] {symbol} {token_addr[:8]}...{token_addr[-8:]} - ${logged_usd:.2f} -> ${actual_usd:.2f} ({diff_ratio*100:+.1f}%)")

        except Exception as e:
            error_msg = f"{pos_key}: {e}"
            stats["errors"].append(error_msg)
            if verbose:
                print(f"[error] {error_msg}")
            import traceback
            if verbose:
                print(traceback.format_exc())

    # Persist changes if not dry run
    if not dry_run and any_change:
        try:
            # Update open_positions.json and hunter_state.db via replace_positions
            replace_positions(updated_positions)
            
            # Update performance_data.json
            perf["trades"] = trades
            perf["last_updated"] = datetime.now().isoformat()
            replace_performance_data(perf)
            
            if verbose:
                print(f"✅ Persisted changes to open_positions.json, performance_data.json, and hunter_state.db")
        except Exception as e:
            error_msg = f"persist_error: {e}"
            stats["errors"].append(error_msg)
            if verbose:
                print(f"❌ Failed to persist changes: {e}")

    return stats


if __name__ == "__main__":
    # Ensure existing data uses mint-only keys before any syncs
    print("🔄 Migrating open_positions.json to mint-only keys...")
    migration_stats = migrate_open_positions_to_canonical_keys(dry_run=False)
    print(f"✅ Migration complete: {migration_stats}")

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

