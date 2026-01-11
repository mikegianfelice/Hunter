#!/usr/bin/env python3
"""
Priority Token Indexer
Proactively identifies and indexes frequently-traded tokens for fast candle fetches.
"""

from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Tuple
import logging


logger = logging.getLogger(__name__)

# Import ESTABLISHED_TOKENS - handle import error gracefully
try:
    from src.utils.token_scraper import ESTABLISHED_TOKENS
except ImportError:
    ESTABLISHED_TOKENS = {}
    logger.warning("Could not import ESTABLISHED_TOKENS from token_scraper")


class PriorityTokenIndexer:
    """Extracts and prioritizes tokens for proactive indexing"""
    
    def __init__(self, helius_cache_file: str = "data/helius_wallet_value_cache.json"):
        self.helius_cache_file = helius_cache_file
        self.established_tokens = set(ESTABLISHED_TOKENS.keys())
    
    def get_top_tokens_from_trades(
        self, 
        days: int = 30,
        min_trades: int = 1,
        top_n: int = 50,
        chain_filter: str = "solana"
    ) -> List[Tuple[str, int]]:
        """
        Extract top N most-traded tokens from historical trades.
        Uses helius_wallet_value_cache.json as the source of truth.
        
        Args:
            days: Look back N days
            min_trades: Minimum number of trades to include
            top_n: Return top N tokens
            chain_filter: Only include tokens from this chain (default: solana)
        
        Returns:
            List of (token_address, trade_count) tuples, sorted by frequency
        """
        try:
            from pathlib import Path
            import json
            
            cache_file = Path(self.helius_cache_file)
            if not cache_file.exists():
                logger.info(f"Helius cache file not found: {self.helius_cache_file}")
                return []
            
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            completed_trades = cache_data.get('completed_trades', [])
            
            if not completed_trades:
                logger.info("No completed trades found in Helius cache")
                return []
            
            # Filter by date range (use UTC for timezone-aware comparison)
            from datetime import timezone
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            # Count token frequency
            token_counts = Counter()
            for trade in completed_trades:
                # Get buy_time for date filtering
                buy_time_str = trade.get('buy_time', '')
                if not buy_time_str:
                    continue
                
                # Parse datetime (format: "2025-11-17 13:56:29+00:00")
                try:
                    # Handle both formats: "2025-11-17 13:56:29+00:00" and ISO format
                    if 'T' in buy_time_str:
                        buy_time = datetime.fromisoformat(buy_time_str.replace('Z', '+00:00'))
                    else:
                        # Format: "2025-11-17 13:56:29+00:00"
                        buy_time = datetime.strptime(buy_time_str, "%Y-%m-%d %H:%M:%S%z")
                    
                    # Filter by date
                    if buy_time < cutoff_date:
                        continue
                except (ValueError, TypeError) as e:
                    logger.debug(f"Could not parse buy_time '{buy_time_str}': {e}")
                    continue
                
                # Get mint address (token address)
                mint = trade.get('mint', '').strip()
                if mint:
                    token_counts[mint.lower()] += 1
            
            # Filter by minimum trades and get top N
            filtered = [
                (addr, count) 
                for addr, count in token_counts.items() 
                if count >= min_trades
            ]
            filtered.sort(key=lambda x: x[1], reverse=True)
            
            top_tokens = filtered[:top_n]
            
            logger.info(
                f"Extracted {len(top_tokens)} top tokens from {len(completed_trades)} completed trades "
                f"(last {days} days, min {min_trades} trades)"
            )
            
            return top_tokens
            
        except Exception as e:
            logger.error(f"Error extracting top tokens from Helius cache: {e}", exc_info=True)
            return []
    
    def get_established_tokens(self, chain_filter: str = "solana") -> List[str]:
        """
        Get established tokens (from ESTABLISHED_TOKENS list).
        
        Args:
            chain_filter: Only return tokens if we know their chain
                          (for now, return all - filtering can be added later)
        
        Returns:
            List of token addresses
        """
        # For Solana, we can identify Solana tokens by their format or known addresses
        # SOL, BONK, mSOL are Solana tokens
        # For now, return all established tokens - they'll be filtered by chain later
        return list(self.established_tokens)
    
    def get_priority_tokens(
        self,
        from_trades_days: int = 30,
        from_trades_min: int = 1,
        from_trades_top_n: int = 50,
        include_established: bool = True,
        chain_filter: str = "solana"
    ) -> Set[str]:
        """
        Get all priority tokens for indexing.
        
        Returns:
            Set of token addresses to prioritize
        """
        priority = set()
        
        # Add top tokens from historical trades
        top_traded = self.get_top_tokens_from_trades(
            days=from_trades_days,
            min_trades=from_trades_min,
            top_n=from_trades_top_n,
            chain_filter=chain_filter
        )
        for addr, _ in top_traded:
            priority.add(addr.lower())
        
        # Add established tokens
        if include_established:
            established = self.get_established_tokens(chain_filter)
            for addr in established:
                priority.add(addr.lower())
        
        logger.info(f"Total priority tokens: {len(priority)}")
        return priority
    
    def add_tokens_to_indexer(
        self,
        indexer,
        priority_tokens: Set[str],
        verbose: bool = False
    ) -> Dict[str, int]:
        """
        Add priority tokens to swap indexer.
        
        Args:
            indexer: SwapIndexer instance
            priority_tokens: Set of token addresses to add
            verbose: Log each token added
        
        Returns:
            Dict with stats: {'added': count, 'already_tracked': count}
        """
        stats = {'added': 0, 'already_tracked': 0}
        
        for token_address in priority_tokens:
            token_lower = token_address.lower()
            
            # Check if already tracked
            if token_lower in indexer.tracked_tokens:
                stats['already_tracked'] += 1
                if verbose:
                    logger.debug(f"Token {token_lower[:8]}... already tracked")
                continue
            
            # Add to indexer
            try:
                indexer.add_token(token_address)
                stats['added'] += 1
                if verbose:
                    logger.info(f"Added priority token to indexer: {token_lower[:8]}...")
            except Exception as e:
                logger.warning(f"Failed to add token {token_lower[:8]}...: {e}")
        
        logger.info(
            f"Indexer update: {stats['added']} added, "
            f"{stats['already_tracked']} already tracked"
        )
        
        return stats


def get_priority_tokens_for_indexing(
    from_trades_days: int = 30,
    from_trades_min: int = 1,
    from_trades_top_n: int = 50,
    include_established: bool = True,
    chain_filter: str = "solana"
) -> Set[str]:
    """
    Convenience function to get priority tokens.
    
    Usage:
        priority_tokens = get_priority_tokens_for_indexing()
        indexer = get_indexer()
        for token in priority_tokens:
            indexer.add_token(token)
    """
    indexer = PriorityTokenIndexer()
    return indexer.get_priority_tokens(
        from_trades_days=from_trades_days,
        from_trades_min=from_trades_min,
        from_trades_top_n=from_trades_top_n,
        include_established=include_established,
        chain_filter=chain_filter
    )
