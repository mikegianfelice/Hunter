#!/usr/bin/env python3
"""
Plot Wallet Value Over Time
Visualizes PnL by plotting wallet value over the last 5 days
"""

import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict
import sys

def get_project_root():
    """Find the project root directory"""
    current = Path(__file__).resolve()
    if current.parent.name == 'scripts':
        return current.parent.parent
    for parent in current.parents:
        if (parent / 'data').exists():
            return parent
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

def estimate_position_size_from_trades(trades):
    """Estimate average position size from trade data"""
    position_sizes = []
    for trade in trades:
        if trade.get('position_size_usd'):
            position_sizes.append(trade['position_size_usd'])
    
    if position_sizes:
        return sum(position_sizes) / len(position_sizes)
    return 13.2  # Default estimate based on config

def calculate_wallet_value_over_time(days=5, initial_wallet_usd=None):
    """Calculate wallet value over time from trades"""
    data = load_performance_data()
    trade_log = load_trade_log()
    
    if not data and not trade_log:
        print("No trade data available")
        return None
    
    # Calculate date range
    today = datetime.now()
    start_date = today - timedelta(days=days)
    
    # Get all trades from performance data
    all_trades = data.get('trades', []) if data else []
    
    # Filter trades within date range
    recent_trades = []
    for trade in all_trades:
        try:
            entry_time_str = trade.get('entry_time', '')
            if not entry_time_str:
                continue
            
            # Parse entry time
            entry_time = datetime.fromisoformat(entry_time_str.replace('Z', '+00:00').split('.')[0])
            
            if entry_time >= start_date:
                recent_trades.append({
                    'entry_time': entry_time,
                    'exit_time': datetime.fromisoformat(trade.get('exit_time', today.isoformat()).replace('Z', '+00:00').split('.')[0]) if trade.get('exit_time') else None,
                    'position_size_usd': trade.get('position_size_usd', 0) or 0,
                    'pnl_usd': trade.get('pnl_usd'),
                    'pnl_percent': trade.get('pnl_percent'),
                    'status': trade.get('status', 'open'),
                    'symbol': trade.get('symbol', 'UNKNOWN')
                })
        except Exception as e:
            continue
    
    # Also process trade log for more granular data
    recent_log_trades = [t for t in trade_log if t['timestamp'] >= start_date]
    
    # Estimate average position size
    avg_position_size = estimate_position_size_from_trades(recent_trades)
    if not recent_trades:
        avg_position_size = 13.2  # Default
    
    # Estimate initial wallet value if not provided
    if initial_wallet_usd is None:
        # Try to estimate from open positions and recent trades
        open_positions_value = sum([t['position_size_usd'] for t in recent_trades if t['status'] == 'open'])
        
        # Sum all closed trade PnL
        closed_pnl = sum([t.get('pnl_usd', 0) or 0 for t in recent_trades if t.get('pnl_usd') is not None and t['status'] != 'open'])
        
        # Estimate: current wallet = open positions + cash - closed PnL losses
        # Reverse engineer: if we know current open positions, we can estimate starting
        # But we'll use a simpler approach: estimate from position sizes
        initial_wallet_usd = 300.0  # Default starting value (user mentioned $300 earlier)
    
    # Create time series of wallet value changes
    # Collect all trade events (entries and exits)
    events = []
    
    # Process closed trades from performance data
    for trade in recent_trades:
        if trade.get('exit_time') and trade.get('pnl_usd') is not None:
            events.append({
                'time': trade['exit_time'],
                'pnl_usd': trade.get('pnl_usd', 0) or 0,
                'type': 'exit',
                'symbol': trade.get('symbol', 'UNKNOWN')
            })
    
    # Process trade log entries (more granular)
    for trade in recent_log_trades:
        pnl_usd = (trade['pnl_pct'] / 100) * avg_position_size
        events.append({
            'time': trade['timestamp'],
            'pnl_usd': pnl_usd,
            'type': 'exit',
            'symbol': trade.get('token', 'UNKNOWN')[:8] + '...'
        })
    
    # Sort events by time
    events.sort(key=lambda x: x['time'])
    
    # Calculate cumulative wallet value
    time_points = [start_date]
    wallet_values = [initial_wallet_usd]
    
    current_wallet = initial_wallet_usd
    
    # Add daily points even if no trades
    for day_offset in range(1, days + 1):
        day_start = start_date + timedelta(days=day_offset)
        day_end = day_start + timedelta(days=1)
        
        # Process all events in this day
        day_events = [e for e in events if day_start <= e['time'] < day_end]
        
        # Add PnL from events
        for event in day_events:
            current_wallet += event['pnl_usd']
        
        # Add point at end of day
        time_points.append(day_end)
        wallet_values.append(current_wallet)
    
    # Also add points for each trade event for smoother curve
    detailed_time_points = [start_date]
    detailed_wallet_values = [initial_wallet_usd]
    current_detailed = initial_wallet_usd
    
    for event in events:
        if event['time'] >= start_date:
            detailed_time_points.append(event['time'])
            current_detailed += event['pnl_usd']
            detailed_wallet_values.append(current_detailed)
    
    # Add current time if we have open positions
    if any(t['status'] == 'open' for t in recent_trades):
        detailed_time_points.append(today)
        detailed_wallet_values.append(current_detailed)
    
    return {
        'daily_time_points': time_points,
        'daily_wallet_values': wallet_values,
        'detailed_time_points': detailed_time_points,
        'detailed_wallet_values': detailed_wallet_values,
        'events': events,
        'initial_wallet': initial_wallet_usd,
        'current_wallet': detailed_wallet_values[-1] if detailed_wallet_values else initial_wallet_usd,
        'total_pnl': sum([e['pnl_usd'] for e in events]),
        'total_trades': len(events)
    }

def plot_wallet_value(days=5, initial_wallet_usd=None, save_path=None):
    """Plot wallet value over time"""
    data = calculate_wallet_value_over_time(days, initial_wallet_usd)
    
    if not data:
        print("No data to plot")
        return
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Detailed wallet value over time
    ax1.plot(data['detailed_time_points'], data['detailed_wallet_values'], 
             marker='o', linewidth=2, markersize=4, color='#2E86AB', label='Wallet Value')
    
    # Add horizontal line for initial value
    ax1.axhline(y=data['initial_wallet'], color='gray', linestyle='--', 
                linewidth=1, alpha=0.5, label=f'Initial: ${data["initial_wallet"]:.2f}')
    
    # Add current value line
    ax1.axhline(y=data['current_wallet'], color='green', linestyle='--', 
                linewidth=1, alpha=0.5, label=f'Current: ${data["current_wallet"]:.2f}')
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax1.set_xlabel('Date & Time')
    ax1.set_ylabel('Wallet Value (USD)')
    ax1.set_title(f'Wallet Value Over Last {days} Days - Detailed View')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add PnL annotations for significant trades
    for event in data['events'][-10:]:  # Last 10 events
        event_idx = data['detailed_time_points'].index(event['time']) if event['time'] in data['detailed_time_points'] else None
        if event_idx and event_idx < len(data['detailed_wallet_values']):
            wallet_at_event = data['detailed_wallet_values'][event_idx]
            pnl_color = 'green' if event['pnl_usd'] > 0 else 'red'
            ax1.annotate(f"{'+' if event['pnl_usd'] > 0 else ''}${event['pnl_usd']:.2f}",
                        xy=(event['time'], wallet_at_event),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color=pnl_color, alpha=0.7)
    
    # Plot 2: Daily summary
    ax2.plot(data['daily_time_points'], data['daily_wallet_values'], 
             marker='s', linewidth=2, markersize=8, color='#A23B72', label='End of Day Value')
    
    # Fill area between initial and current
    ax2.fill_between(data['daily_time_points'], 
                     [data['initial_wallet']] * len(data['daily_time_points']),
                     data['daily_wallet_values'],
                     where=[v >= data['initial_wallet'] for v in data['daily_wallet_values']],
                     alpha=0.3, color='green', label='Profit')
    ax2.fill_between(data['daily_time_points'], 
                     [data['initial_wallet']] * len(data['daily_time_points']),
                     data['daily_wallet_values'],
                     where=[v < data['initial_wallet'] for v in data['daily_wallet_values']],
                     alpha=0.3, color='red', label='Loss')
    
    ax2.axhline(y=data['initial_wallet'], color='gray', linestyle='--', 
                linewidth=1, alpha=0.5, label=f'Initial: ${data["initial_wallet"]:.2f}')
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Wallet Value (USD)')
    ax2.set_title(f'Wallet Value Over Last {days} Days - Daily Summary')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add statistics text
    total_pnl = data['total_pnl']
    total_pnl_pct = ((data['current_wallet'] - data['initial_wallet']) / data['initial_wallet'] * 100) if data['initial_wallet'] > 0 else 0
    
    stats_text = f"""Statistics:
Initial Wallet: ${data['initial_wallet']:.2f}
Current Wallet: ${data['current_wallet']:.2f}
Total PnL: ${total_pnl:+.2f} ({total_pnl_pct:+.2f}%)
Total Trades: {data['total_trades']}
"""
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.savefig(PROJECT_ROOT / 'data' / 'wallet_value_plot.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {PROJECT_ROOT / 'data' / 'wallet_value_plot.png'}")
    
    plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot wallet value over time')
    parser.add_argument('--days', type=int, default=5, help='Number of days to plot (default: 5)')
    parser.add_argument('--initial', type=float, default=None, help='Initial wallet value in USD (default: estimated)')
    parser.add_argument('--save', type=str, default=None, help='Path to save plot (default: data/wallet_value_plot.png)')
    
    args = parser.parse_args()
    
    plot_wallet_value(days=args.days, initial_wallet_usd=args.initial, save_path=args.save)

