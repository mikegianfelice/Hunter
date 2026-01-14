#!/usr/bin/env python3
"""
Manually run Helius reconciliation for open positions.
"""

import sys
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.helius_reconciliation import reconcile_positions_and_pnl

def main():
    print("🔄 Running Helius reconciliation...")
    print("=" * 60)
    
    result = reconcile_positions_and_pnl(limit=200)
    
    print("\n✅ Helius Reconciliation Complete!")
    print("=" * 60)
    print(f"Enabled: {result.get('enabled', False)}")
    
    if not result.get('enabled'):
        print(f"❌ Reason: {result.get('reason', 'Unknown')}")
        return
    
    if result.get('skipped'):
        print(f"⏭️  Skipped: {result.get('skipped')}")
        return
    
    print(f"📊 Open positions closed: {result.get('open_positions_closed', 0)}")
    print(f"✅ Open positions verified: {result.get('open_positions_verified', 0)}")
    print(f"🔄 Trades updated: {result.get('trades_updated', 0)}")
    print(f"⚠️  Issues: {len(result.get('issues', []))}")
    
    if result.get('issues'):
        print("\n⚠️  Issues encountered:")
        for issue in result['issues']:
            print(f"   • {issue}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
