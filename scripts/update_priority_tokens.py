#!/usr/bin/env python3
"""
Update Priority Tokens for Swap Indexer
Can be run standalone or as a scheduled task.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.config.config_loader import get_config, get_config_int
from src.indexing.swap_indexer import get_indexer
from src.utils.priority_token_indexer import PriorityTokenIndexer


def main():
    """Update priority tokens in swap indexer"""
    print("🔄 Updating priority tokens for swap indexer...\n")
    
    # Swap indexer must be enabled to update priority tokens
    if not get_config("swap_indexer.enabled", False):
        print("⚠️  Swap indexer is disabled in config (swap_indexer.enabled: false)")
        return
    if not get_config("swap_indexer.enable_priority_indexing", True):
        print("⚠️  Priority indexing is disabled in config")
        return
    
    try:
        # Get indexer
        indexer = get_indexer()
        
        if not indexer.running:
            print("⚠️  Swap indexer is not running. Starting it...")
            indexer.start()
        
        # Get priority indexer
        priority_indexer = PriorityTokenIndexer()
        
        # Get config values
        from_trades_days = get_config_int("swap_indexer.priority_indexing.from_trades_days", 30)
        from_trades_min = get_config_int("swap_indexer.priority_indexing.from_trades_min_trades", 1)
        from_trades_top_n = get_config_int("swap_indexer.priority_indexing.from_trades_top_n", 50)
        include_established = get_config("swap_indexer.priority_indexing.include_established", True)
        chain_filter = "solana"
        
        print(f"📊 Extracting priority tokens...")
        print(f"   - From trades: last {from_trades_days} days")
        print(f"   - Minimum trades: {from_trades_min}")
        print(f"   - Top N: {from_trades_top_n}")
        print(f"   - Include established: {include_established}\n")
        
        # Get priority tokens
        priority_tokens = priority_indexer.get_priority_tokens(
            from_trades_days=from_trades_days,
            from_trades_min=from_trades_min,
            from_trades_top_n=from_trades_top_n,
            include_established=include_established,
            chain_filter=chain_filter
        )
        
        print(f"✅ Found {len(priority_tokens)} priority tokens\n")
        
        # Add to indexer
        print("📝 Adding tokens to indexer...")
        stats = priority_indexer.add_tokens_to_indexer(
            indexer,
            priority_tokens,
            verbose=True
        )
        
        print(f"\n✅ Complete!")
        print(f"   - Added: {stats['added']} tokens")
        print(f"   - Already tracked: {stats['already_tracked']} tokens")
        print(f"   - Total tracked: {len(indexer.tracked_tokens)} tokens")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Error updating priority tokens: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
