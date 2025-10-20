#!/usr/bin/env python3
"""
Performance Dashboard for Sustainable Trading Bot
Interactive dashboard to view trading performance and analytics
"""

import sys
from performance_tracker import performance_tracker
from datetime import datetime, timedelta

def show_main_dashboard():
    """Show the main performance dashboard"""
    print("\n" + "="*60)
    print("📊 SUSTAINABLE TRADING PERFORMANCE DASHBOARD")
    print("="*60)
    
    # Overall performance
    summary = performance_tracker.get_performance_summary(30)
    print(f"\n🎯 OVERALL PERFORMANCE (Last 30 Days):")
    print(f"• Total Trades: {summary['total_trades']}")
    print(f"• Win Rate: {summary['win_rate']:.1f}%")
    print(f"• Average PnL: ${summary['avg_pnl']:.2f}")
    print(f"• Total PnL: ${summary['total_pnl']:.2f}")
    
    # Quality tier analysis
    if summary['quality_analysis']:
        print(f"\n📈 QUALITY TIER PERFORMANCE:")
        for tier, stats in summary['quality_analysis'].items():
            if stats['trades'] > 0:
                print(f"• {tier.upper()} Quality ({stats['trades']} trades):")
                print(f"  - Win Rate: {stats['win_rate']:.1f}%")
                print(f"  - Avg PnL: ${stats['avg_pnl']:.2f}")
                print(f"  - Total PnL: ${stats['total_pnl']:.2f}")
    
    # Recent trades
    recent_trades = performance_tracker.get_recent_trades(10)
    if recent_trades:
        print(f"\n🔄 RECENT TRADES:")
        for trade in recent_trades:
            status_emoji = "✅" if trade['pnl_usd'] and trade['pnl_usd'] > 0 else "❌" if trade['pnl_usd'] else "⏳"
            pnl_str = f"${trade['pnl_usd']:.2f}" if trade['pnl_usd'] else "Open"
            entry_date = datetime.fromisoformat(trade['entry_time']).strftime('%m/%d %H:%M')
            print(f"• {status_emoji} {trade['symbol']} - {pnl_str} (Quality: {trade['quality_score']:.1f}) - {entry_date}")
    
    # Open trades
    open_trades = performance_tracker.get_open_trades()
    if open_trades:
        print(f"\n⏳ OPEN TRADES ({len(open_trades)}):")
        for trade in open_trades:
            entry_date = datetime.fromisoformat(trade['entry_time']).strftime('%m/%d %H:%M')
            print(f"• {trade['symbol']} - ${trade['position_size_usd']:.1f} (Quality: {trade['quality_score']:.1f}) - {entry_date}")

def show_quality_analysis():
    """Show detailed quality vs performance analysis"""
    print("\n" + "="*60)
    print("🎯 QUALITY VS PERFORMANCE ANALYSIS")
    print("="*60)
    
    analysis = performance_tracker.get_quality_vs_performance()
    
    if 'message' in analysis:
        print(f"\n{analysis['message']}")
        return
    
    for tier, stats in analysis.items():
        print(f"\n📊 {tier.upper()} QUALITY TIER:")
        print(f"• Total Trades: {stats['trades']}")
        print(f"• Win Rate: {stats['win_rate']:.1f}%")
        print(f"• Average PnL: ${stats['avg_pnl']:.2f}")
        print(f"• Total PnL: ${stats['total_pnl']:.2f}")
        print(f"• Best Trade: ${stats['best_trade']:.2f}")
        print(f"• Worst Trade: ${stats['worst_trade']:.2f}")
        
        # Performance insights
        if stats['trades'] >= 5:  # Only show insights for meaningful sample size
            if stats['win_rate'] >= 70:
                print(f"  💚 EXCELLENT: High win rate suggests good quality selection")
            elif stats['win_rate'] >= 60:
                print(f"  🟢 GOOD: Solid performance, consider increasing position size")
            elif stats['win_rate'] >= 50:
                print(f"  🟡 AVERAGE: Decent performance, monitor closely")
            else:
                print(f"  🔴 POOR: Consider raising quality threshold for this tier")

def show_detailed_report():
    """Show detailed performance report"""
    print("\n" + "="*60)
    print("📋 DETAILED PERFORMANCE REPORT")
    print("="*60)
    
    report = performance_tracker.generate_performance_report()
    print(report)

def show_recent_trades():
    """Show recent trades in detail"""
    print("\n" + "="*60)
    print("🔄 RECENT TRADES DETAIL")
    print("="*60)
    
    recent_trades = performance_tracker.get_recent_trades(20)
    
    if not recent_trades:
        print("\nNo trades found.")
        return
    
    print(f"\n{'Symbol':<12} {'Quality':<8} {'Position':<10} {'PnL':<10} {'Status':<8} {'Date':<12}")
    print("-" * 70)
    
    for trade in recent_trades:
        symbol = trade['symbol'][:11]  # Truncate long symbols
        quality = f"{trade['quality_score']:.1f}"
        position = f"${trade['position_size_usd']:.1f}"
        
        if trade['pnl_usd'] is not None:
            pnl = f"${trade['pnl_usd']:.2f}"
            status = "✅ Win" if trade['pnl_usd'] > 0 else "❌ Loss"
        else:
            pnl = "Open"
            status = "⏳ Open"
        
        entry_date = datetime.fromisoformat(trade['entry_time']).strftime('%m/%d %H:%M')
        
        print(f"{symbol:<12} {quality:<8} {position:<10} {pnl:<10} {status:<8} {entry_date:<12}")

def show_open_trades():
    """Show currently open trades"""
    print("\n" + "="*60)
    print("⏳ OPEN TRADES")
    print("="*60)
    
    open_trades = performance_tracker.get_open_trades()
    
    if not open_trades:
        print("\nNo open trades.")
        return
    
    print(f"\n{'Symbol':<12} {'Quality':<8} {'Position':<10} {'Entry Price':<12} {'Entry Date':<12}")
    print("-" * 70)
    
    for trade in open_trades:
        symbol = trade['symbol'][:11]
        quality = f"{trade['quality_score']:.1f}"
        position = f"${trade['position_size_usd']:.1f}"
        entry_price = f"${trade['entry_price']:.6f}"
        entry_date = datetime.fromisoformat(trade['entry_time']).strftime('%m/%d %H:%M')
        
        print(f"{symbol:<12} {quality:<8} {position:<10} {entry_price:<12} {entry_date:<12}")

def cleanup_data():
    """Clean up old performance data"""
    print("\n🧹 Cleaning up old performance data...")
    performance_tracker.cleanup_old_data(90)  # Keep 90 days
    print("✅ Data cleanup completed")

def main():
    """Main dashboard interface"""
    while True:
        print("\n" + "="*60)
        print("📊 SUSTAINABLE TRADING DASHBOARD")
        print("="*60)
        print("1. Main Dashboard")
        print("2. Quality Analysis")
        print("3. Detailed Report")
        print("4. Recent Trades")
        print("5. Open Trades")
        print("6. Cleanup Data")
        print("7. Exit")
        
        try:
            choice = input("\nSelect option (1-7): ").strip()
            
            if choice == '1':
                show_main_dashboard()
            elif choice == '2':
                show_quality_analysis()
            elif choice == '3':
                show_detailed_report()
            elif choice == '4':
                show_recent_trades()
            elif choice == '5':
                show_open_trades()
            elif choice == '6':
                cleanup_data()
            elif choice == '7':
                print("\n👋 Goodbye!")
                break
            else:
                print("\n❌ Invalid option. Please select 1-7.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
