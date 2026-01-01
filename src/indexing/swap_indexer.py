#!/usr/bin/env python3
"""
Continuous Swap Event Indexer

Monitors Solana blockchain for swap transactions and stores them in SQLite
for fast historical candlestick queries without API calls.
"""

import time
import logging
import threading
from typing import Optional, Dict, List, Set
from datetime import datetime, timedelta
from pathlib import Path

from solana.rpc.api import Client
from solders.pubkey import Pubkey

from src.config.secrets import SOLANA_RPC_URL, HELIUS_API_KEY
from src.storage.swap_events import (
    store_swap_event,
    get_latest_swap_time,
    get_swap_count,
)
from src.utils.api_tracker import track_helius_call

logger = logging.getLogger(__name__)

# DEX Program IDs
DEX_PROGRAMS = {
    "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8": "Raydium V4",
    "27haf8L6oxUeXrHrgEgsexjSY5hbVUWEmvv9Nyxg8vQv": "Raydium V3",
    "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP": "Orca V2",
    "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc": "Orca Whirlpool",
    "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4": "Jupiter V6",
}

# Common quote tokens
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDT_MINT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
WSOL_MINT = "So11111111111111111111111111111111111111112"

QUOTE_MINTS = {USDC_MINT.lower(), USDT_MINT.lower(), WSOL_MINT.lower()}


class SwapIndexer:
    """Continuous indexer for Solana swap events"""

    def __init__(
        self,
        rpc_url: Optional[str] = None,
        helius_api_key: Optional[str] = None,
        index_interval: int = 60,  # Index every 60 seconds
        max_pools_per_token: int = 5,
        lookback_hours: int = 24,  # Look back 24 hours on startup
    ):
        self.rpc_url = rpc_url or SOLANA_RPC_URL
        self.helius_api_key = helius_api_key or HELIUS_API_KEY
        self.index_interval = index_interval
        self.max_pools_per_token = max_pools_per_token
        self.lookback_hours = lookback_hours
        
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.tracked_tokens: Set[str] = set()
        self.tracked_pools: Set[str] = set()
        
        # Use Helius RPC if available, otherwise public RPC
        if self.helius_api_key and "helius-rpc.com" in self.rpc_url:
            self.client = Client(self.rpc_url)
        else:
            # Use public RPC
            self.client = Client(self.rpc_url)
            logger.info("Using public Solana RPC for swap indexing")

    def start(self) -> None:
        """Start the continuous indexing thread"""
        if self.running:
            logger.warning("Swap indexer already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._index_loop, daemon=True)
        self.thread.start()
        logger.info("Swap indexer started")

    def stop(self) -> None:
        """Stop the continuous indexing thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Swap indexer stopped")

    def add_token(self, token_address: str) -> None:
        """Add a token to track for swap events"""
        self.tracked_tokens.add(token_address.lower())
        logger.debug(f"Added token to tracking: {token_address[:8]}...")

    def add_pool(self, pool_address: str) -> None:
        """Add a pool to track for swap events"""
        self.tracked_pools.add(pool_address.lower())
        logger.debug(f"Added pool to tracking: {pool_address[:8]}...")

    def _index_loop(self) -> None:
        """Main indexing loop"""
        logger.info("Starting swap indexing loop...")
        
        # Initial backfill for tracked tokens
        if self.tracked_tokens:
            logger.info(f"Performing initial backfill for {len(self.tracked_tokens)} tokens...")
            self._backfill_tokens(list(self.tracked_tokens))
        
        # Continuous indexing
        while self.running:
            try:
                # Index new swaps for tracked tokens
                if self.tracked_tokens:
                    self._index_new_swaps(list(self.tracked_tokens))
                
                # Index new swaps for tracked pools
                if self.tracked_pools:
                    self._index_pool_swaps(list(self.tracked_pools))
                
                time.sleep(self.index_interval)
            except Exception as e:
                logger.error(f"Error in indexing loop: {e}", exc_info=True)
                time.sleep(self.index_interval)

    def _backfill_tokens(self, token_addresses: List[str]) -> None:
        """Backfill historical swaps for tokens"""
        end_time = time.time()
        start_time = end_time - (self.lookback_hours * 3600)
        
        for token_address in token_addresses:
            try:
                logger.info(f"Backfilling swaps for {token_address[:8]}...")
                swaps = self._fetch_swaps_for_token(token_address, start_time, end_time)
                stored = 0
                for swap in swaps:
                    if store_swap_event(**swap):
                        stored += 1
                logger.info(f"Stored {stored} swap events for {token_address[:8]}...")
            except Exception as e:
                logger.error(f"Error backfilling {token_address[:8]}...: {e}")

    def _index_new_swaps(self, token_addresses: List[str]) -> None:
        """Index new swaps since last check"""
        for token_address in token_addresses:
            try:
                # Get latest swap time for this token
                latest_time = get_latest_swap_time(token_address)
                start_time = latest_time if latest_time else (time.time() - 3600)  # Last hour if no data
                end_time = time.time()
                
                swaps = self._fetch_swaps_for_token(token_address, start_time, end_time)
                stored = 0
                for swap in swaps:
                    if store_swap_event(**swap):
                        stored += 1
                
                if stored > 0:
                    logger.debug(f"Indexed {stored} new swaps for {token_address[:8]}...")
            except Exception as e:
                logger.error(f"Error indexing swaps for {token_address[:8]}...: {e}")
    
    def backfill_missing_hours(
        self, token_address: str, missing_hours: float
    ) -> int:
        """
        Targeted backfill for missing hours only.
        Uses pagination to fetch all historical swaps in the time range.
        Returns number of swaps stored.
        """
        try:
            end_time = time.time()
            start_time = end_time - (missing_hours * 3600)
            
            logger.info(f"Backfilling {missing_hours:.1f} hours for {token_address[:8]}... (using pagination)")
            # Use pagination=True to fetch all historical swaps
            swaps = self._fetch_swaps_for_token(token_address, start_time, end_time, use_pagination=True)
            
            stored = 0
            for swap in swaps:
                if store_swap_event(**swap):
                    stored += 1
            
            logger.info(f"Stored {stored} swap events for {token_address[:8]}... ({missing_hours:.1f} hours)")
            return stored
            
        except Exception as e:
            logger.error(f"Error backfilling {token_address[:8]}...: {e}")
            return 0

    def _index_pool_swaps(self, pool_addresses: List[str]) -> None:
        """Index swaps from specific pools"""
        for pool_address in pool_addresses:
            try:
                latest_time = get_latest_swap_time()
                start_time = latest_time if latest_time else (time.time() - 3600)
                end_time = time.time()
                
                swaps = self._fetch_swaps_for_pool(pool_address, start_time, end_time)
                stored = 0
                for swap in swaps:
                    if store_swap_event(**swap):
                        stored += 1
                
                if stored > 0:
                    logger.debug(f"Indexed {stored} new swaps for pool {pool_address[:8]}...")
            except Exception as e:
                logger.error(f"Error indexing pool swaps {pool_address[:8]}...: {e}")

    def _fetch_swaps_for_token(
        self, token_address: str, start_time: float, end_time: float, use_pagination: bool = False
    ) -> List[Dict]:
        """
        Fetch swap transactions for a token.
        
        Args:
            use_pagination: If True, paginate through all historical transactions (for backfill)
        """
        swaps = []
        
        try:
            # Find pools containing this token
            pools = self._find_pools_for_token(token_address)
            
            # Query swaps from pools (only main pool for efficiency)
            for pool_address in pools[:self.max_pools_per_token]:
                pool_swaps = self._fetch_swaps_for_pool(
                    pool_address, start_time, end_time, token_address, use_pagination=use_pagination
                )
                swaps.extend(pool_swaps)
            
            # Note: Removed token mint query per architecture (only query pools)
            
        except Exception as e:
            logger.error(f"Error fetching swaps for token {token_address[:8]}...: {e}")
        
        return swaps

    def _fetch_swaps_for_pool(
        self, pool_address: str, start_time: float, end_time: float, filter_token: Optional[str] = None,
        use_pagination: bool = False
    ) -> List[Dict]:
        """Fetch swap transactions from a pool"""
        return self._fetch_swaps_for_address(pool_address, start_time, end_time, filter_token, use_pagination)

    def _fetch_swaps_for_address(
        self, address: str, start_time: float, end_time: float, filter_token: Optional[str] = None,
        use_pagination: bool = False
    ) -> List[Dict]:
        """
        Fetch swap transactions for an address (token or pool).
        
        Args:
            use_pagination: If True, paginate through all historical transactions until start_time
        """
        swaps = []
        
        try:
            pubkey = Pubkey.from_string(address)
            
            if use_pagination:
                # Paginated backfill: fetch all transactions in time range
                all_signatures = []
                before_signature = None
                max_pages = 100  # Safety limit for backfill
                
                for page in range(max_pages):
                    # Get signatures
                    if self.helius_api_key:
                        track_helius_call()
                    
                    sigs_response = self.client.get_signatures_for_address(
                        pubkey,
                        limit=1000,  # Max per page for backfill
                        before=before_signature,
                    )
                    
                    if not sigs_response.value:
                        break
                    
                    # Collect signatures in time range
                    page_signatures = []
                    found_old_enough = False
                    
                    for sig_info in sigs_response.value:
                        if not sig_info.block_time:
                            continue
                        
                        if sig_info.block_time < start_time:
                            found_old_enough = True
                            break
                        
                        if start_time <= sig_info.block_time <= end_time:
                            page_signatures.append((sig_info.signature, sig_info.block_time))
                    
                    all_signatures.extend(page_signatures)
                    
                    if found_old_enough:
                        break
                    
                    # Set up pagination for next page
                    if sigs_response.value:
                        before_signature = sigs_response.value[-1].signature
                    else:
                        break
                
                # Fetch transactions in batches
                batch_size = 50
                for i in range(0, len(all_signatures), batch_size):
                    batch = all_signatures[i:i + batch_size]
                    for sig, block_time in batch:
                        tx_data = self._get_transaction(sig)
                        if tx_data:
                            swap_data = self._parse_swap_transaction(tx_data, block_time, filter_token)
                            if swap_data:
                                swap_data["tx_signature"] = str(sig)
                                swaps.append(swap_data)
            else:
                # Incremental indexing: only recent transactions
                if self.helius_api_key:
                    track_helius_call()
                
                sigs_response = self.client.get_signatures_for_address(
                    pubkey,
                    limit=100,  # Reasonable limit for incremental indexing
                )
                
                if not sigs_response.value:
                    return []
                
                # Filter by time and fetch transactions
                for sig_info in sigs_response.value:
                    if not sig_info.block_time:
                        continue
                    
                    if sig_info.block_time < start_time:
                        break  # Signatures are in reverse chronological order
                    
                    if sig_info.block_time > end_time:
                        continue
                    
                    # Fetch transaction
                    tx_data = self._get_transaction(sig_info.signature)
                    if not tx_data:
                        continue
                    
                    # Parse swap
                    swap_data = self._parse_swap_transaction(tx_data, sig_info.block_time, filter_token)
                    if swap_data:
                        swap_data["tx_signature"] = str(sig_info.signature)  # Set signature
                        swaps.append(swap_data)
        
        except Exception as e:
            logger.error(f"Error fetching swaps for address {address[:8]}...: {e}")
        
        return swaps

    def _get_transaction(self, signature: str) -> Optional[Dict]:
        """Get transaction data"""
        try:
            if self.helius_api_key:
                track_helius_call()
            
            response = self.client.get_transaction(
                signature,
                max_supported_transaction_version=0,
            )
            
            if response.value:
                # Convert to dict format
                tx_dict = {
                    "meta": {
                        "err": response.value.transaction.meta.err if response.value.transaction.meta else None,
                        "preTokenBalances": [
                            {
                                "accountIndex": b.account_index,
                                "mint": str(b.mint) if b.mint else None,
                                "owner": str(b.owner) if b.owner else None,
                                "uiTokenAmount": {
                                    "uiAmount": b.ui_token_amount.ui_amount if b.ui_token_amount else None,
                                },
                            }
                            for b in (response.value.transaction.meta.pre_token_balances if response.value.transaction.meta else [])
                        ],
                        "postTokenBalances": [
                            {
                                "accountIndex": b.account_index,
                                "mint": str(b.mint) if b.mint else None,
                                "owner": str(b.owner) if b.owner else None,
                                "uiTokenAmount": {
                                    "uiAmount": b.ui_token_amount.ui_amount if b.ui_token_amount else None,
                                },
                            }
                            for b in (response.value.transaction.meta.post_token_balances if response.value.transaction.meta else [])
                        ],
                    },
                    "transaction": {
                        "message": {
                            "accountKeys": [str(k) for k in response.value.transaction.transaction.message.account_keys],
                        },
                    },
                }
                return tx_dict
        except Exception as e:
            logger.debug(f"Error fetching transaction {signature[:8]}...: {e}")
        
        return None

    def _parse_swap_transaction(
        self, tx_data: Dict, block_time: float, filter_token: Optional[str] = None
    ) -> Optional[Dict]:
        """Parse swap transaction and extract swap data with directional information"""
        try:
            # Extract additional metadata first
            meta = tx_data.get("meta", {})
            account_keys = tx_data.get("transaction", {}).get("message", {}).get("accountKeys", [])
            
            # Extract signer wallet (first account is always the signer)
            signer_wallet = account_keys[0] if account_keys else None
            
            if not filter_token or not signer_wallet:
                return None
            
            # Parse with directional information for signer wallet
            swap_data = self._parse_solana_swap_with_direction(
                tx_data, filter_token, signer_wallet, USDC_MINT, WSOL_MINT, block_time
            )
            
            if not swap_data or swap_data.get("price", 0) <= 0:
                return None
            
            # Find DEX program
            dex_program = None
            for key in account_keys:
                if key.lower() in [p.lower() for p in DEX_PROGRAMS.keys()]:
                    dex_program = DEX_PROGRAMS.get(key)
                    break
            
            # Extract pool address (first account that's not a DEX program)
            pool_address = None
            for key in account_keys:
                if key.lower() not in [p.lower() for p in DEX_PROGRAMS.keys()]:
                    pool_address = key
                    break
            
            return {
                "token_address": filter_token.lower() if filter_token else "",
                "pool_address": pool_address,
                "tx_signature": "",  # Will be set by caller
                "block_time": block_time,
                "price_usd": swap_data.get("price", 0),
                "volume_usd": swap_data.get("volume_usd"),
                "amount_in": swap_data.get("token_amount"),  # Absolute value for backward compatibility
                "amount_out": swap_data.get("quote_amount"),  # Absolute value for backward compatibility
                "base_mint": filter_token.lower() if filter_token else None,
                "quote_mint": swap_data.get("quote_mint", "").lower() if swap_data.get("quote_mint") else None,
                "dex_program": dex_program,
                "signer_wallet": signer_wallet,
                "token_delta": swap_data.get("token_delta"),  # Signed delta for signer
                "quote_delta": swap_data.get("quote_delta"),  # Signed delta for signer
            }
        
        except Exception as e:
            logger.debug(f"Error parsing swap transaction: {e}")
            return None
    
    def _parse_solana_swap_with_direction(
        self, tx_data: Dict, token_address: str, signer_wallet: str, 
        usdc_mint: str, wsol_mint: str, block_time: float
    ) -> Optional[Dict]:
        """
        Parse Solana transaction to extract swap data with directional balance changes for signer wallet.
        
        Returns swap data including signed deltas (token_delta, quote_delta) for the signer wallet.
        Positive token_delta = BUY (signer received tokens)
        Negative token_delta = SELL (signer sent tokens)
        """
        try:
            if not isinstance(tx_data, dict):
                return None
            
            meta = tx_data.get("meta", {})
            if not meta or meta.get("err") is not None:
                return None
            
            # Get token balances (pre and post)
            pre_token_balances = meta.get("preTokenBalances", [])
            post_token_balances = meta.get("postTokenBalances", [])
            
            if not pre_token_balances or not post_token_balances:
                return None
            
            # Get account keys to map account indices to wallet addresses
            account_keys = tx_data.get("transaction", {}).get("message", {}).get("accountKeys", [])
            if not account_keys:
                return None
            
            signer_wallet_lower = signer_wallet.lower()
            token_address_lower = token_address.lower()
            usdc_mint_lower = usdc_mint.lower()
            wsol_mint_lower = wsol_mint.lower()
            
            # Build maps: (account_index, mint) -> balance for signer's token accounts
            # Token accounts are owned by the signer wallet (matched by owner field)
            pre_balances = {}
            post_balances = {}
            
            for bal in pre_token_balances:
                account_idx = bal.get("accountIndex")
                if account_idx is not None:
                    ui_amount = bal.get("uiTokenAmount", {}).get("uiAmount", 0)
                    mint = bal.get("mint", "").lower()
                    owner = bal.get("owner", "").lower() if bal.get("owner") else None
                    # Match token accounts owned by the signer wallet
                    if owner and owner == signer_wallet_lower:
                        pre_balances[(account_idx, mint)] = float(ui_amount) if ui_amount is not None else 0.0
            
            for bal in post_token_balances:
                account_idx = bal.get("accountIndex")
                if account_idx is not None:
                    ui_amount = bal.get("uiTokenAmount", {}).get("uiAmount", 0)
                    mint = bal.get("mint", "").lower()
                    owner = bal.get("owner", "").lower() if bal.get("owner") else None
                    # Match token accounts owned by the signer wallet
                    if owner and owner == signer_wallet_lower:
                        post_balances[(account_idx, mint)] = float(ui_amount) if ui_amount is not None else 0.0
            
            # Calculate signed deltas for signer wallet
            token_delta = 0.0
            quote_delta = 0.0
            quote_mint = None
            
            all_keys = set(pre_balances.keys()) | set(post_balances.keys())
            for (account_idx, mint) in all_keys:
                pre_bal = pre_balances.get((account_idx, mint), 0.0)
                post_bal = post_balances.get((account_idx, mint), 0.0)
                delta = post_bal - pre_bal
                
                if abs(delta) < 0.000001:
                    continue
                
                if mint == token_address_lower:
                    token_delta += delta
                elif mint == usdc_mint_lower or mint == wsol_mint_lower:
                    quote_delta += delta
                    if quote_mint is None:
                        quote_mint = mint
            
            # Calculate absolute amounts and price for backward compatibility
            token_amount = abs(token_delta)
            quote_amount = abs(quote_delta)
            
            if token_amount <= 0 or quote_amount <= 0:
                return None
            
            price = quote_amount / token_amount if token_amount > 0 else 0
            if price <= 0:
                return None
            
            # Calculate volume in USD
            volume_usd = quote_amount if quote_mint == usdc_mint_lower else quote_amount * 150.0
            
            return {
                "timestamp": block_time,
                "token_amount": token_amount,  # Absolute for backward compatibility
                "quote_amount": quote_amount,  # Absolute for backward compatibility
                "token_delta": token_delta,  # Signed: positive = BUY, negative = SELL
                "quote_delta": quote_delta,  # Signed: positive = received quote, negative = paid quote
                "quote_mint": quote_mint,
                "price": price,
                "volume_usd": volume_usd,
            }
        
        except Exception as e:
            logger.debug(f"Error parsing Solana swap with direction: {e}")
            return None

    def _find_pools_for_token(self, token_address: str) -> List[str]:
        """Find liquidity pools containing this token using direct DEX program queries"""
        # Try direct DEX program query first (more efficient, no external API)
        pools = self._find_pools_via_program_accounts(token_address)
        if pools:
            logger.debug(f"Found {len(pools)} pools via program accounts for {token_address[:8]}...")
            return pools[:self.max_pools_per_token]
        
        # Fallback to DexScreener if program accounts query fails
        logger.debug(f"Program accounts query returned no pools, trying DexScreener fallback for {token_address[:8]}...")
        return self._find_pools_via_dexscreener(token_address)
    
    def _find_pools_via_program_accounts(self, token_address: str) -> List[str]:
        """Find pools directly from DEX programs using getProgramAccounts"""
        pools = []
        token_pubkey = Pubkey.from_string(token_address)
        
        try:
            # Try Raydium V4 pools first
            raydium_pools = self._find_raydium_pools(token_pubkey)
            pools.extend(raydium_pools)
            
            # Try Orca pools
            orca_pools = self._find_orca_pools(token_pubkey)
            pools.extend(orca_pools)
            
            # Remove duplicates
            return list(set(pools))
        except Exception as e:
            logger.debug(f"Error finding pools via program accounts: {e}")
            return []
    
    def _find_raydium_pools(self, token_pubkey: Pubkey) -> List[str]:
        """
        Find Raydium pools containing the token.
        
        Note: getProgramAccounts times out due to thousands of pools.
        Instead, we query transactions from the DEX program and filter for swaps
        involving our token, then extract pool addresses from those transactions.
        This is more efficient than querying all program accounts.
        """
        pools = []
        try:
            # Raydium V4 Program ID
            RAYDIUM_V4 = Pubkey.from_string("675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8")
            
            if self.helius_api_key:
                track_helius_call()
            
            # Query recent transactions from Raydium program
            # Look for swaps involving our token by checking transaction accounts
            # This is more efficient than getProgramAccounts
            
            # Get recent signatures from the program
            sigs_response = self.client.get_signatures_for_address(
                RAYDIUM_V4,
                limit=100,  # Check recent 100 transactions
            )
            
            if not sigs_response.value:
                return []
            
            # Extract unique pool addresses from transactions
            seen_pools = set()
            
            for sig_info in sigs_response.value[:50]:  # Limit to 50 to avoid too many RPC calls
                try:
                    # Get transaction to check accounts
                    # sig_info.signature is already a Signature object
                    tx_response = self.client.get_transaction(
                        sig_info.signature,
                        max_supported_transaction_version=0,
                    )
                    
                    if tx_response.value and tx_response.value.transaction:
                        # Check if transaction involves our token
                        accounts = tx_response.value.transaction.transaction.message.account_keys
                        
                        # Check token balances to see if our token is involved
                        meta = tx_response.value.transaction.meta
                        if meta and meta.pre_token_balances:
                            for balance in meta.pre_token_balances:
                                if balance.mint and str(balance.mint) == str(token_pubkey):
                                    # This transaction involves our token
                                    # Find pool address (usually one of the accounts)
                                    for account in accounts:
                                        account_str = str(account)
                                        # Pool addresses are typically not the program itself
                                        if account_str != str(RAYDIUM_V4) and account_str not in seen_pools:
                                            # This might be a pool address
                                            # Verify by checking if it's owned by Raydium
                                            try:
                                                account_info = self.client.get_account_info(account)
                                                if account_info.value and account_info.value.owner == RAYDIUM_V4:
                                                    pools.append(account_str)
                                                    seen_pools.add(account_str)
                                                    logger.debug(f"Found potential Raydium pool: {account_str[:8]}...")
                                            except:
                                                pass
                                    break
                except Exception as e:
                    logger.debug(f"Error checking transaction for pools: {e}")
                    continue
                    
        except Exception as e:
            logger.debug(f"Error querying Raydium pools via transactions: {e}")
        
        return pools
    
    def _find_orca_pools(self, token_pubkey: Pubkey) -> List[str]:
        """
        Find Orca pools containing the token.
        Uses same transaction-based approach as Raydium to avoid getProgramAccounts timeout.
        """
        pools = []
        try:
            # Orca Whirlpool Program ID
            ORCA_WHIRLPOOL = Pubkey.from_string("whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc")
            
            if self.helius_api_key:
                track_helius_call()
            
            # Query recent transactions from Orca program
            sigs_response = self.client.get_signatures_for_address(
                ORCA_WHIRLPOOL,
                limit=100,
            )
            
            if not sigs_response.value:
                return []
            
            seen_pools = set()
            
            for sig_info in sigs_response.value[:50]:
                try:
                    tx_response = self.client.get_transaction(
                        sig_info.signature,
                        max_supported_transaction_version=0,
                    )
                    
                    if tx_response.value and tx_response.value.transaction:
                        accounts = tx_response.value.transaction.transaction.message.account_keys
                        meta = tx_response.value.transaction.meta
                        
                        if meta and meta.pre_token_balances:
                            for balance in meta.pre_token_balances:
                                if balance.mint and str(balance.mint) == str(token_pubkey):
                                    for account in accounts:
                                        account_str = str(account)
                                        if account_str != str(ORCA_WHIRLPOOL) and account_str not in seen_pools:
                                            try:
                                                account_info = self.client.get_account_info(account)
                                                if account_info.value and account_info.value.owner == ORCA_WHIRLPOOL:
                                                    pools.append(account_str)
                                                    seen_pools.add(account_str)
                                                    logger.debug(f"Found potential Orca pool: {account_str[:8]}...")
                                            except:
                                                pass
                                    break
                except Exception as e:
                    logger.debug(f"Error checking Orca transaction: {e}")
                    continue
                    
        except Exception as e:
            logger.debug(f"Error querying Orca pools via transactions: {e}")
        
        return pools
    
    def _find_pools_via_dexscreener(self, token_address: str) -> List[str]:
        """Fallback: Find pools using DexScreener API"""
        try:
            import requests
            url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                pairs = data.get("pairs", [])
                
                pool_addresses = []
                for pair in pairs[:self.max_pools_per_token]:
                    pair_address = pair.get("pairAddress")
                    if pair_address and len(pair_address) in [43, 44]:
                        pool_addresses.append(pair_address)
                
                return pool_addresses
        except Exception as e:
            logger.debug(f"Error finding pools via DexScreener: {e}")
        
        return []


# Global indexer instance
_global_indexer: Optional[SwapIndexer] = None


def get_indexer() -> SwapIndexer:
    """Get or create global swap indexer instance"""
    global _global_indexer
    if _global_indexer is None:
        from src.config.config_loader import get_config_int
        index_interval = get_config_int("swap_indexer.index_interval_seconds", 60)
        max_pools = get_config_int("swap_indexer.max_pools_per_token", 5)
        lookback_hours = get_config_int("swap_indexer.lookback_hours", 24)
        
        _global_indexer = SwapIndexer(
            index_interval=index_interval,
            max_pools_per_token=max_pools,
            lookback_hours=lookback_hours,
        )
    return _global_indexer
