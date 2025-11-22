#!/usr/bin/env python3
"""
Performance Analysis Script - Last 5 Days
Analyzes trading performance metrics over the specified period
"""

from datetime import datetime, timedelta
from typing import Dict, List
import json
import csv
from collections import defaultdict
from pathlib import Path

# Find project root (where data/ folder exists)
def get_project_root():
    """Find the project root directory by looking for data/ folder"""
    current = Path(__file__).resolve()
    # If we're in scripts/, go up one level to project root
    if current.parent.name == 'scripts':
        return current.parent.parent
    # Otherwise, search up the directory tree for data/ folder
    for parent in current.parents:
        if (parent / 'data').exists():
            return parent
    # Fallback: assume we're in project root
    return current.parent

PROJECT_ROOT = get_project_root()

def load_performance_data():
    """Load performance data from JSON file"""
    try:
        data_file = PROJECT_ROOT / 'data' / 'performance_data.json'
        with open(data_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading performance data: {e}")
        return None

def load_trade_log():
    """Load trade log from CSV file"""
    trades = []
    try:
        log_file = PROJECT_ROOT / 'data' / 'trade_log.csv'
        with open(log_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    timestamp = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')
                    trades.append({
                        'timestamp': timestamp,
                        'token': row['token'],
                        'entry_price': float(row['entry_price']),
                        'exit_price': float(row['exit_price']),
                        'pnl_pct': float(row['pnl_pct']),
                        'reason': row['reason']
                    })
                except Exception as e:
                    continue
    except Exception as e:
        print(f"Error loading trade log: {e}")
    return trades

def analyze_last_5_days():
    """Analyze performance over the last 5 days"""
    data = load_performance_data()
    trade_log = load_trade_log()
    
    # Calculate date range (last 5 days)
    today = datetime.now()
    cutoff_date = today - timedelta(days=5)
    
    # Filter trade log entries from last 5 days
    recent_log_trades = [t for t in trade_log if t['timestamp'] >= cutoff_date]
    
    if not data and not recent_log_trades:
        return None
    
    trades = data.get('trades', []) if data else []
    daily_stats = data.get('daily_stats', {}) if data else {}
    
    # Filter trades from last 5 days
    recent_trades = []
    for trade in trades:
        try:
            entry_time = datetime.fromisoformat(trade['entry_time'].replace('Z', '+00:00').split('.')[0])
            if entry_time >= cutoff_date:
                recent_trades.append(trade)
        except Exception as e:
            continue
    
    # Helper function to identify failed entry attempts
    def is_failed_entry_attempt(trade):
        """Check if a trade represents a failed entry attempt (not an actual completed trade)."""
        entry_amount = trade.get('entry_amount_usd_actual', 0) or 0
        tokens_received = trade.get('entry_tokens_received')
        
        # If no entry amount or no tokens received, it's a failed entry
        if entry_amount == 0 or (tokens_received is None or (isinstance(tokens_received, (int, float)) and tokens_received == 0)):
            # Double-check: if exit_time is very close to entry_time, it was closed immediately
            try:
                entry_time_str = trade.get('entry_time', '')
                exit_time_str = trade.get('exit_time', '')
                if entry_time_str and exit_time_str:
                    entry_time = datetime.fromisoformat(entry_time_str.replace('Z', '+00:00').split('.')[0])
                    exit_time = datetime.fromisoformat(exit_time_str.replace('Z', '+00:00').split('.')[0])
                    time_diff = abs((exit_time - entry_time).total_seconds())
                    
                    # If closed within 2 seconds and exit_price = entry_price, it's a failed entry
                    if time_diff < 2.0:
                        exit_price = trade.get('exit_price', 0)
                        entry_price = trade.get('entry_price', 0)
                        if exit_price == entry_price or exit_price == 0:
                            return True
            except:
                pass
            
            return True
        
        return False
    
    # Separate closed and open trades, excluding failed entry attempts
    all_closed_trades = [t for t in recent_trades if t.get('status') in ['closed', 'manual_close', 'stopped_out'] and t.get('pnl_usd') is not None]
    closed_trades = [t for t in all_closed_trades if not is_failed_entry_attempt(t)]
    failed_entry_attempts = [t for t in all_closed_trades if is_failed_entry_attempt(t)]
    open_trades = [t for t in recent_trades if t.get('status') == 'open']
    
    # Overall statistics
    total_trades = len(recent_trades)
    closed_count = len(closed_trades)
    open_count = len(open_trades)
    
    # Position size analysis (calculate first)
    position_sizes = [t.get('position_size_usd', 0) for t in recent_trades if t.get('position_size_usd')]
    avg_position_size = (sum(position_sizes) / len(position_sizes)) if position_sizes else 14.52  # Default estimate
    
    # Use trade log for more accurate PnL if available
    if recent_log_trades:
        # Calculate PnL from trade log (more accurate)
        log_pnl_pct = [t['pnl_pct'] for t in recent_log_trades]
        log_total_pnl_pct = sum(log_pnl_pct) if log_pnl_pct else 0.0
        log_avg_pnl_pct = (log_total_pnl_pct / len(log_pnl_pct)) if log_pnl_pct else 0.0
        
        # Estimate USD PnL from average position size
        log_total_pnl_usd = sum([(t['pnl_pct'] / 100) * avg_position_size for t in recent_log_trades])
        log_avg_pnl_usd = (log_total_pnl_usd / len(recent_log_trades)) if recent_log_trades else 0.0
        
        log_winning = [t for t in recent_log_trades if t['pnl_pct'] > 0]
        log_losing = [t for t in recent_log_trades if t['pnl_pct'] <= 0]
        log_win_rate = (len(log_winning) / len(recent_log_trades) * 100) if recent_log_trades else 0.0
    else:
        log_total_pnl_usd = 0.0
        log_avg_pnl_usd = 0.0
        log_win_rate = 0.0
        log_winning = []
        log_losing = []
        log_pnl_pct = []
    
    # PnL calculations from performance data
    pnl_values = [t['pnl_usd'] for t in closed_trades if t.get('pnl_usd') is not None]
    total_pnl = sum(pnl_values) if pnl_values else 0.0
    avg_pnl = (total_pnl / len(pnl_values)) if pnl_values else 0.0
    
    # Use trade log data if available, otherwise use performance data
    if recent_log_trades:
        total_pnl = log_total_pnl_usd
        avg_pnl = log_avg_pnl_usd
        winning_trades = log_winning
        losing_trades = log_losing
        win_rate = log_win_rate
        closed_count = len(recent_log_trades)
    else:
        # Win rate from performance data
        winning_trades = [t for t in closed_trades if t.get('pnl_usd', 0) > 0]
        losing_trades = [t for t in closed_trades if t.get('pnl_usd', 0) <= 0]
        win_rate = (len(winning_trades) / closed_count * 100) if closed_count > 0 else 0.0
    
    # Create address to symbol mapping from performance data
    address_to_symbol = {}
    for trade in trades:
        addr = trade.get('address', '')
        symbol = trade.get('symbol', '')
        if addr and symbol:
            address_to_symbol[addr] = symbol
    
    # Best and worst trades
    if recent_log_trades:
        best_log_trade = max(recent_log_trades, key=lambda x: x['pnl_pct'])
        worst_log_trade = min(recent_log_trades, key=lambda x: x['pnl_pct'])
        
        best_token_addr = best_log_trade['token']
        best_symbol = address_to_symbol.get(best_token_addr, best_token_addr[:8] + '...')
        
        worst_token_addr = worst_log_trade['token']
        worst_symbol = address_to_symbol.get(worst_token_addr, worst_token_addr[:8] + '...')
        
        best_trade = {
            'symbol': best_symbol,
            'pnl_usd': (best_log_trade['pnl_pct'] / 100) * avg_position_size,
            'pnl_percent': best_log_trade['pnl_pct'],
            'entry_time': best_log_trade['timestamp'].isoformat(),
            'reason': best_log_trade['reason']
        }
        worst_trade = {
            'symbol': worst_symbol,
            'pnl_usd': (worst_log_trade['pnl_pct'] / 100) * avg_position_size,
            'pnl_percent': worst_log_trade['pnl_pct'],
            'entry_time': worst_log_trade['timestamp'].isoformat(),
            'reason': worst_log_trade['reason']
        }
    else:
        best_trade = max(closed_trades, key=lambda x: x.get('pnl_usd', float('-inf'))) if closed_trades else None
        worst_trade = min(closed_trades, key=lambda x: x.get('pnl_usd', float('inf'))) if closed_trades else None
    
    # Daily breakdown - use trade log if available
    daily_breakdown = {}
    
    if recent_log_trades:
        # Use trade log for daily breakdown
        for trade in recent_log_trades:
            date = trade['timestamp'].strftime('%Y-%m-%d')
            if date not in daily_breakdown:
                daily_breakdown[date] = {
                    'trades': 0,
                    'closed': 0,
                    'open': 0,
                    'total_pnl': 0.0,
                    'total_pnl_pct': 0.0,
                    'winning': 0,
                    'losing': 0
                }
            
            daily_breakdown[date]['trades'] += 1
            daily_breakdown[date]['closed'] += 1
            pnl_usd = (trade['pnl_pct'] / 100) * avg_position_size
            daily_breakdown[date]['total_pnl'] += pnl_usd
            daily_breakdown[date]['total_pnl_pct'] += trade['pnl_pct']
            if trade['pnl_pct'] > 0:
                daily_breakdown[date]['winning'] += 1
            else:
                daily_breakdown[date]['losing'] += 1
    else:
        # Use performance data
        for trade in recent_trades:
            date = trade['entry_time'][:10]  # YYYY-MM-DD
            if date not in daily_breakdown:
                daily_breakdown[date] = {
                    'trades': 0,
                    'closed': 0,
                    'open': 0,
                    'total_pnl': 0.0,
                    'total_pnl_pct': 0.0,
                    'winning': 0,
                    'losing': 0
                }
            
            daily_breakdown[date]['trades'] += 1
            if trade.get('status') == 'open':
                daily_breakdown[date]['open'] += 1
            else:
                daily_breakdown[date]['closed'] += 1
                if trade.get('pnl_usd') is not None:
                    daily_breakdown[date]['total_pnl'] += trade['pnl_usd']
                    if trade['pnl_usd'] > 0:
                        daily_breakdown[date]['winning'] += 1
                    else:
                        daily_breakdown[date]['losing'] += 1
    
    # Quality tier analysis (only for completed trades, excluding failed entry attempts)
    quality_tiers = {
        "excellent": (80, 100),
        "high": (70, 79),
        "good": (60, 69),
        "average": (50, 59),
        "low": (0, 49)
    }
    
    quality_analysis = {}
    for tier, (min_score, max_score) in quality_tiers.items():
        # Filter tier trades to exclude failed entry attempts
        tier_trades_raw = [t for t in closed_trades if min_score <= (t.get('quality_score', 0) * 100) <= max_score]
        tier_trades = [t for t in tier_trades_raw if not is_failed_entry_attempt(t)]
        if tier_trades:
            tier_pnl = [t['pnl_usd'] for t in tier_trades if t.get('pnl_usd') is not None]
            tier_wins = len([t for t in tier_trades if t.get('pnl_usd', 0) > 0])
            quality_analysis[tier] = {
                'trades': len(tier_trades),
                'win_rate': (tier_wins / len(tier_trades) * 100) if tier_trades else 0,
                'total_pnl': sum(tier_pnl) if tier_pnl else 0.0,
                'avg_pnl': (sum(tier_pnl) / len(tier_pnl)) if tier_pnl else 0.0
            }
    
    # Fee analysis
    total_fees = 0.0
    trades_with_fees = 0
    for trade in closed_trades:
        entry_fee = trade.get('entry_gas_fee_usd') or 0
        exit_fee = trade.get('exit_gas_fee_usd') or 0
        if entry_fee > 0 or exit_fee > 0:
            total_fees += entry_fee + exit_fee
            trades_with_fees += 1
    
    
    # Compile results
    analysis = {
        'period': {
            'start_date': cutoff_date.strftime('%Y-%m-%d'),
            'end_date': today.strftime('%Y-%m-%d'),
            'days': 5
        },
        'overall': {
            'total_trades': total_trades,
            'closed_trades': closed_count,
            'open_trades': open_count,
            'failed_entry_attempts': len(failed_entry_attempts),  # Track separately
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'avg_pnl': round(avg_pnl, 2),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'avg_position_size': round(avg_position_size, 2),
            'total_pnl_pct': round(sum(log_pnl_pct), 2) if recent_log_trades else 0.0,
            'avg_pnl_pct': round(log_avg_pnl_pct, 2) if recent_log_trades else 0.0
        },
        'best_trade': {
            'symbol': best_trade.get('symbol') if best_trade else None,
            'pnl_usd': round(best_trade.get('pnl_usd', 0), 2) if best_trade else None,
            'pnl_percent': round(best_trade.get('pnl_percent', 0), 2) if best_trade else None,
            'entry_time': best_trade.get('entry_time') if best_trade else None
        } if best_trade else None,
        'worst_trade': {
            'symbol': worst_trade.get('symbol') if worst_trade else None,
            'pnl_usd': round(worst_trade.get('pnl_usd', 0), 2) if worst_trade else None,
            'pnl_percent': round(worst_trade.get('pnl_percent', 0), 2) if worst_trade else None,
            'entry_time': worst_trade.get('entry_time') if worst_trade else None
        } if worst_trade else None,
        'daily_breakdown': daily_breakdown,
        'quality_analysis': quality_analysis,
        'fees': {
            'total_fees': round(total_fees, 4),
            'trades_with_fees': trades_with_fees,
            'avg_fee_per_trade': round(total_fees / trades_with_fees, 4) if trades_with_fees > 0 else 0.0
        }
    }
    
    return analysis

def print_analysis(analysis):
    """Print formatted analysis report"""
    if not analysis:
        print("❌ Could not generate analysis")
        return
    
    print("\n" + "="*70)
    print("📊 PERFORMANCE ANALYSIS - LAST 5 DAYS")
    print("="*70)
    
    period = analysis['period']
    print(f"\n📅 Period: {period['start_date']} to {period['end_date']} ({period['days']} days)")
    
    overall = analysis['overall']
    print(f"\n🎯 OVERALL STATISTICS:")
    print(f"   • Total Trades: {overall['total_trades']}")
    print(f"   • Closed Trades: {overall['closed_trades']}")
    print(f"   • Open Trades: {overall['open_trades']}")
    if overall.get('failed_entry_attempts', 0) > 0:
        print(f"   • Failed Entry Attempts: {overall['failed_entry_attempts']} (excluded from win rate)")
    print(f"   • Win Rate: {overall['win_rate']:.1f}%")
    print(f"   • Total PnL: ${overall['total_pnl']:.2f}")
    if overall.get('total_pnl_pct'):
        print(f"   • Total PnL %: {overall['total_pnl_pct']:.2f}%")
    print(f"   • Average PnL: ${overall['avg_pnl']:.2f}")
    if overall.get('avg_pnl_pct'):
        print(f"   • Average PnL %: {overall['avg_pnl_pct']:.2f}%")
    print(f"   • Winning Trades: {overall['winning_trades']}")
    print(f"   • Losing Trades: {overall['losing_trades']}")
    print(f"   • Avg Position Size: ${overall['avg_position_size']:.2f}")
    
    if analysis['best_trade']:
        bt = analysis['best_trade']
        print(f"\n🏆 BEST TRADE:")
        print(f"   • Symbol: {bt['symbol']}")
        print(f"   • PnL: ${bt['pnl_usd']:.2f} ({bt['pnl_percent']:.2f}%)")
        print(f"   • Entry Time: {bt['entry_time']}")
        if bt.get('reason'):
            print(f"   • Reason: {bt['reason']}")
    
    if analysis['worst_trade']:
        wt = analysis['worst_trade']
        print(f"\n📉 WORST TRADE:")
        print(f"   • Symbol: {wt['symbol']}")
        print(f"   • PnL: ${wt['pnl_usd']:.2f} ({wt['pnl_percent']:.2f}%)")
        print(f"   • Entry Time: {wt['entry_time']}")
        if wt.get('reason'):
            print(f"   • Reason: {wt['reason']}")
    
    if analysis['daily_breakdown']:
        print(f"\n📅 DAILY BREAKDOWN:")
        for date in sorted(analysis['daily_breakdown'].keys(), reverse=True):
            day = analysis['daily_breakdown'][date]
            win_rate_day = (day['winning'] / day['closed'] * 100) if day['closed'] > 0 else 0
            print(f"   • {date}:")
            print(f"     - Total: {day['trades']} trades ({day['closed']} closed, {day['open']} open)")
            if day['closed'] > 0:
                print(f"     - PnL: ${day['total_pnl']:.2f}")
                if day.get('total_pnl_pct'):
                    print(f"     - PnL %: {day['total_pnl_pct']:.2f}%")
                print(f"     - Win Rate: {win_rate_day:.1f}% ({day['winning']}W / {day['losing']}L)")
    
    if analysis['quality_analysis']:
        print(f"\n⭐ QUALITY TIER ANALYSIS:")
        for tier in ['excellent', 'high', 'good', 'average', 'low']:
            if tier in analysis['quality_analysis']:
                qa = analysis['quality_analysis'][tier]
                print(f"   • {tier.upper()}:")
                print(f"     - Trades: {qa['trades']}")
                print(f"     - Win Rate: {qa['win_rate']:.1f}%")
                print(f"     - Total PnL: ${qa['total_pnl']:.2f}")
                print(f"     - Avg PnL: ${qa['avg_pnl']:.2f}")
    
    fees = analysis['fees']
    if fees['trades_with_fees'] > 0:
        print(f"\n💰 FEE ANALYSIS:")
        print(f"   • Total Fees: ${fees['total_fees']:.4f}")
        print(f"   • Trades with Fees: {fees['trades_with_fees']}")
        print(f"   • Avg Fee per Trade: ${fees['avg_fee_per_trade']:.4f}")
    
    print("\n" + "="*70 + "\n")

if __name__ == '__main__':
    analysis = analyze_last_5_days()
    print_analysis(analysis)

