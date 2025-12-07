#!/usr/bin/env python3
"""
Reconcile position_size_usd using on-chain balances and live prices.
Updates open_positions.json, performance_data.json, and hunter_state.db.
"""

import argparse
from pathlib import Path
import sys

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.position_sync import reconcile_position_sizes  # noqa: E402

def main():
    parser = argparse.ArgumentParser(
        description="Reconcile position sizes from on-chain balances and current prices"
    )
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=5.0,
        help="Minimum percentage difference to trigger update (default: 5.0%%)"
    )
    parser.add_argument(
        "--min-balance",
        type=float,
        default=1e-6,
        help="Minimum balance threshold to treat as zero/dust (default: 1e-6)"
    )
    parser.add_argument(
        "--chain",
        action="append",
        help="Restrict to specific chain(s): solana|ethereum|base (can be repeated for multiple chains)"
    )
    parser.add_argument(
        "--no-verify-balance",
        action="store_true",
        help="Skip balance verification (not recommended)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without persisting to files"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed logging for each position"
    )
    args = parser.parse_args()

    print("🔄 Starting position size reconciliation...")
    if args.dry_run:
        print("⚠️  DRY RUN MODE - No changes will be persisted")
    
    stats = reconcile_position_sizes(
        threshold_pct=args.threshold_pct,
        min_balance_threshold=args.min_balance,
        chains=args.chain,
        verify_balance=not args.no_verify_balance,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    print("\n" + "=" * 60)
    print("📊 Reconciliation Summary")
    print("=" * 60)
    print(f"✅ Updated: {stats['updated']} position(s)")
    print(f"🚫 Closed:  {stats['closed']} position(s) (zero balance)")
    print(f"⏭️  Skipped: {stats['skipped']} position(s)")
    print(f"❌ Errors:   {len(stats['errors'])} error(s)")
    
    if stats["errors"]:
        print("\n⚠️  Errors encountered:")
        for err in stats["errors"]:
            print(f"   • {err}")
    
    if args.dry_run:
        print("\n💡 Run without --dry-run to apply changes")
    else:
        print("\n✅ Reconciliation complete!")

if __name__ == "__main__":
    main()

