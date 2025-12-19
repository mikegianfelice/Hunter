#!/usr/bin/env python3
"""
Analyze A/B test performance and recommend best configuration
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.ab_testing import ab_testing

def main():
    report = ab_testing.get_performance_report()
    
    print("=" * 60)
    print("A/B Testing Performance Report")
    print("=" * 60)
    
    for config in report['configs']:
        print(f"\n{config['name']}:")
        print(f"  Trades: {config['trades']}")
        print(f"  Win Rate: {config['win_rate']:.2%}")
        print(f"  Total PnL: ${config['total_pnl']:.2f}")
        print(f"  Avg PnL: ${config['avg_pnl']:.2f}")
    
    best = ab_testing.get_best_config()
    if best:
        print(f"\n🏆 Best Configuration: {best.name}")
        print(f"   Win Rate: {best.wins / best.trades_count:.2%}")
        print(f"   Total PnL: ${best.total_pnl:.2f}")
    else:
        print("\n⚠️  No configuration has enough trades yet (need at least 10)")

if __name__ == "__main__":
    main()

