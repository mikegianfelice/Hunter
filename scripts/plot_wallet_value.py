#!/usr/bin/env python3
"""
Plot Wallet Value Over Time
Visualizes PnL by plotting wallet value over time
Includes percentage-based charts optimized for marketing
"""

import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for CI/headless environments
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

# Minimum date for chart display - bot started trading on this date
MIN_CHART_DATE = datetime(2025, 11, 16, 0, 0, 0)

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
    
    # Find the earliest trade date to set a minimum start date
    # This prevents showing empty days before trading actually started
    earliest_trade_date = None
    
    # Check performance data
    all_trades = data.get('trades', []) if data else []
    for trade in all_trades:
        try:
            entry_time_str = trade.get('entry_time', '')
            if entry_time_str:
                entry_time = datetime.fromisoformat(entry_time_str.replace('Z', '+00:00').split('.')[0])
                if earliest_trade_date is None or entry_time < earliest_trade_date:
                    earliest_trade_date = entry_time
        except Exception:
            continue
    
    # Check trade log
    for trade in trade_log:
        if trade.get('timestamp'):
            if earliest_trade_date is None or trade['timestamp'] < earliest_trade_date:
                earliest_trade_date = trade['timestamp']
    
    # Use the later of: calculated start_date, earliest trade date, or minimum chart date
    # This ensures we don't show empty days before trading started
    min_date = MIN_CHART_DATE
    if earliest_trade_date:
        min_date = max(MIN_CHART_DATE, earliest_trade_date.replace(hour=0, minute=0, second=0, microsecond=0))
    start_date = max(start_date, min_date)
    
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
        return None
    
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
        plt.savefig(PROJECT_ROOT / 'docs' / 'wallet_value_plot.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {PROJECT_ROOT / 'docs' / 'wallet_value_plot.png'}")
    
    plt.close()

def plot_percentage_pnl(days=30, initial_wallet_usd=None, save_path=None):
    """
    Plot PnL as percentage returns - optimized for marketing and GitHub display
    Clean, professional chart showing percentage returns over time
    """
    data = calculate_wallet_value_over_time(days, initial_wallet_usd)
    
    if not data:
        print("No data to plot")
        return None
    
    # Calculate percentage returns
    initial = data['initial_wallet']
    if initial <= 0:
        print("Invalid initial wallet value")
        return
    
    percentage_returns = [
        ((val - initial) / initial * 100) if initial > 0 else 0
        for val in data['detailed_wallet_values']
    ]
    
    # Calculate win rate from events
    winning_trades = len([e for e in data['events'] if e['pnl_usd'] > 0])
    total_trades = data['total_trades']
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Create professional figure with modern styling
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            plt.style.use('default')
    
    fig, ax = plt.subplots(figsize=(14, 7), facecolor='white')
    
    # Color scheme
    positive_color = '#10B981'  # Green
    negative_color = '#EF4444'    # Red
    line_color = '#2563EB'      # Blue
    grid_color = '#E5E7EB'      # Light gray
    
    # Plot percentage returns with smooth line
    ax.plot(data['detailed_time_points'], percentage_returns, 
             linewidth=3, color=line_color, label='Total Return', zorder=3)
    
    # Zero line (break-even)
    ax.axhline(y=0, color='#6B7280', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
    
    # Fill positive/negative areas
    ax.fill_between(data['detailed_time_points'], 0, percentage_returns,
                     where=[p >= 0 for p in percentage_returns],
                     alpha=0.2, color=positive_color, label='Profit Zone', zorder=2)
    ax.fill_between(data['detailed_time_points'], 0, percentage_returns,
                     where=[p < 0 for p in percentage_returns],
                     alpha=0.2, color=negative_color, label='Loss Zone', zorder=2)
    
    # Formatting for readability
    ax.set_xlabel('Date', fontsize=13, fontweight='bold', color='#1F2937')
    ax.set_ylabel('Return (%)', fontsize=13, fontweight='bold', color='#1F2937')
    
    # Title with key metric
    total_return = percentage_returns[-1] if percentage_returns else 0
    title_color = positive_color if total_return >= 0 else negative_color
    ax.set_title(f'Hunter Bot Performance: {total_return:+.2f}% over {days} days', 
                 fontsize=16, fontweight='bold', color=title_color, pad=20)
    
    # Grid styling
    ax.grid(True, alpha=0.3, color=grid_color, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Format x-axis dates
    if days <= 7:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    elif days <= 30:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, days // 7)))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=10)
    plt.setp(ax.yaxis.get_majorticklabels(), fontsize=10)
    
    # Add percentage sign to y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:+.0f}%'))
    
    # Stats box with key metrics
    total_return = percentage_returns[-1] if percentage_returns else 0
    avg_daily_return = total_return / days if days > 0 else 0
    
    stats_text = f"""Key Metrics
━━━━━━━━━━━━━━━━━━━━
Total Return: {total_return:+.2f}%
Avg Daily: {avg_daily_return:+.2f}%
Win Rate: {win_rate:.1f}%
Trades: {total_trades}
Period: {days} days"""
    
    # Position stats box in upper left
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='left',
            family='monospace', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#F9FAFB', 
                     edgecolor='#E5E7EB', linewidth=1.5, alpha=0.95),
            color='#1F2937')
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#D1D5DB')
    ax.spines['bottom'].set_color('#D1D5DB')
    
    plt.tight_layout()
    
    # Save with high quality
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', format='png')
        print(f"✅ Chart saved to: {save_path}")
    else:
        default_path = PROJECT_ROOT / 'docs' / 'performance_chart.png'
        default_path.parent.mkdir(exist_ok=True)
        plt.savefig(default_path, dpi=150, bbox_inches='tight', facecolor='white',
                   edgecolor='none', format='png')
        print(f"✅ Chart saved to: {default_path}")
    
    plt.close()  # Important: close to free memory
    return total_return, total_trades, win_rate

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot wallet value over time')
    parser.add_argument('--days', type=int, default=30, help='Number of days to plot (default: 30)')
    parser.add_argument('--initial', type=float, default=None, help='Initial wallet value in USD (default: estimated)')
    parser.add_argument('--save', type=str, default=None, help='Path to save plot (default: docs/performance_chart.png)')
    parser.add_argument('--type', type=str, choices=['percentage', 'value'], default='percentage',
                       help='Chart type: percentage (marketing) or value (detailed)')
    
    args = parser.parse_args()
    
    try:
        if args.type == 'percentage':
            result = plot_percentage_pnl(days=args.days, initial_wallet_usd=args.initial, save_path=args.save)
        else:
            result = plot_wallet_value(days=args.days, initial_wallet_usd=args.initial, save_path=args.save)
        
        if result is None:
            print("Warning: No data available to generate chart")
            sys.exit(0)  # Exit successfully even if no data
    except Exception as e:
        print(f"Error generating chart: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

