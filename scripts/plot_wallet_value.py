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
import numpy as np

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

def calculate_win_rate_from_trade_log(days=30):
    """
    Calculate win rate from trade_log.csv (closed trades only)
    This is the accurate source of truth for win rate statistics
    """
    trade_log = load_trade_log()
    
    if not trade_log:
        return 0.0, 0
    
    # Calculate date range
    today = datetime.now()
    start_date = today - timedelta(days=days)
    
    # Filter trades within date range
    recent_trades = [t for t in trade_log if t['timestamp'] >= start_date]
    
    if not recent_trades:
        return 0.0, 0
    
    # Count wins and losses (only closed trades with actual PnL)
    wins = 0
    losses = 0
    
    for trade in recent_trades:
        pnl_pct = trade.get('pnl_pct', 0)
        if pnl_pct > 0:
            wins += 1
        elif pnl_pct < 0:
            losses += 1
        # Breakeven trades (pnl_pct == 0) are not counted in win rate
    
    total_closed_trades = wins + losses
    win_rate = (wins / total_closed_trades * 100) if total_closed_trades > 0 else 0.0
    
    return win_rate, total_closed_trades

def load_position_sizes_from_performance_data():
    """
    Load actual position sizes from performance_data.json
    Returns a dictionary mapping (token_address, date) -> position_size_usd
    """
    perf_data = load_performance_data()
    if not perf_data:
        return {}
    
    position_sizes = {}
    trades = perf_data.get('trades', [])
    
    for trade in trades:
        try:
            entry_time_str = trade.get('entry_time', '')
            if not entry_time_str:
                continue
            
            # Parse entry time
            entry_time = datetime.fromisoformat(entry_time_str.replace('Z', '+00:00').split('.')[0])
            token = trade.get('address', '')
            
            # Get position size (try multiple fields)
            pos_size = (
                trade.get('position_size_usd') or 
                trade.get('entry_amount_usd_actual') or 
                trade.get('entry_amount_usd', 0)
            )
            
            if token and pos_size and pos_size > 0:
                # Use token + date as key (date only, not time, to match trades on same day)
                key = (token, entry_time.date().isoformat())
                position_sizes[key] = pos_size
        except Exception as e:
            continue
    
    return position_sizes

def get_position_size_for_trade(trade, position_sizes_map, fallback_avg=None):
    """
    Get actual position size for a trade from performance data
    Falls back to average if not found
    """
    token = trade.get('token', '')
    trade_date = trade['timestamp'].date().isoformat()
    key = (token, trade_date)
    
    if key in position_sizes_map:
        return position_sizes_map[key]
    
    # If not found, try to find any trade for this token on this date
    # (in case timestamp doesn't match exactly)
    for (map_token, map_date), pos_size in position_sizes_map.items():
        if map_token == token and map_date == trade_date:
            return pos_size
    
    # Fallback to average of all known position sizes
    if position_sizes_map and fallback_avg is None:
        fallback_avg = sum(position_sizes_map.values()) / len(position_sizes_map)
    
    # Final fallback
    return fallback_avg if fallback_avg else 13.2

def estimate_position_size_from_trades(trades):
    """Estimate average position size from trade data"""
    position_sizes = []
    for trade in trades:
        if trade.get('position_size_usd'):
            position_sizes.append(trade['position_size_usd'])
    
    if position_sizes:
        return sum(position_sizes) / len(position_sizes)
    return 13.2  # Default estimate based on config

def calculate_wallet_value_over_time_from_helius_wrapper(days=5, initial_wallet_usd=None):
    """Calculate wallet value using Helius transaction data (more accurate)"""
    try:
        # Import here to avoid circular dependencies
        import sys
        from pathlib import Path
        
        # Add scripts directory to path to import calculate_pnl_from_helius
        scripts_dir = Path(__file__).parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        
        from calculate_pnl_from_helius import calculate_wallet_value_over_time_from_helius
        
        # Calculate date range
        today = datetime.now()
        start_date = today - timedelta(days=days)
        
        # Use Nov 17, 2025 as the start if before that (when bot started)
        min_start = datetime(2025, 11, 17, 0, 0, 0)
        if start_date < min_start:
            start_date = min_start
        
        # Set default initial wallet if not provided
        if initial_wallet_usd is None:
            initial_wallet_usd = 200.0
        
        # Call Helius calculation function
        result = calculate_wallet_value_over_time_from_helius(
            start_date_str=start_date.strftime('%Y-%m-%d'),
            end_date_str=today.strftime('%Y-%m-%d'),
            initial_wallet=initial_wallet_usd,
            force_refresh=False  # Use cache for efficiency
        )
        
        if not result:
            return None
        
        # Convert Helius format to format expected by plotting functions
        time_points = result['time_points']
        wallet_values = result['wallet_values']
        events_raw = result.get('events', [])
        
        # Convert events to format expected by plotting functions
        events = []
        for event in events_raw:
            if event.get('type') == 'sell':
                events.append({
                    'time': event['time'],
                    'pnl_usd': event.get('pnl', 0.0),
                    'type': 'exit',
                    'symbol': event.get('mint', 'UNKNOWN')[:8] + '...' if event.get('mint') else 'UNKNOWN'
                })
        
        # Store raw events for percentage calculation (includes deposits/withdrawals)
        raw_events = events_raw
        
        # Create daily summary (sample points at end of each day)
        daily_time_points = [time_points[0]] if time_points else [start_date]
        daily_wallet_values = [wallet_values[0]] if wallet_values else [initial_wallet_usd]
        
        current_day = None
        for time_point, wallet_value in zip(time_points, wallet_values):
            day_start = time_point.replace(hour=0, minute=0, second=0, microsecond=0)
            if current_day != day_start:
                current_day = day_start
                daily_time_points.append(day_start)
                daily_wallet_values.append(wallet_value)
        
        # Add final point
        if time_points:
            daily_time_points.append(time_points[-1])
            daily_wallet_values.append(wallet_values[-1])
        
        return {
            'daily_time_points': daily_time_points,
            'daily_wallet_values': daily_wallet_values,
            'detailed_time_points': time_points,
            'detailed_wallet_values': wallet_values,
            'events': events,
            'raw_events': raw_events,  # Includes deposits/withdrawals for percentage calculation
            'initial_wallet': result['initial_wallet'],
            'current_wallet': result['final_wallet'],
            'total_pnl': result['total_pnl'],
            'trading_pnl': result.get('trading_pnl', result['total_pnl']),
            'total_deposits': result.get('total_deposits', 0.0),
            'total_withdrawals': result.get('total_withdrawals', 0.0),
            'total_trades': len(result.get('completed_trades', []))
        }
    except Exception as e:
        print(f"⚠️  Helius calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_wallet_value_over_time(days=5, initial_wallet_usd=None, use_helius=True):
    """
    Calculate wallet value over time from trades.
    Uses Helius data if use_helius=True (more accurate), otherwise falls back to trade_log.csv
    """
    # Try Helius first if requested
    if use_helius:
        helius_result = calculate_wallet_value_over_time_from_helius_wrapper(days, initial_wallet_usd)
        if helius_result:
            return helius_result
        print("⚠️  Helius calculation failed, falling back to trade_log.csv method")
    
    # Fallback to original method using trade_log.csv
    # Use only trade_log.csv for consistency with calculate_sharpe_r2.py
    trade_log = load_trade_log()
    
    if not trade_log:
        print("No trade data available")
        return None
    
    # Load actual position sizes from performance data
    position_sizes_map = load_position_sizes_from_performance_data()
    
    # Calculate average position size for fallback
    avg_position_size = None
    if position_sizes_map:
        avg_position_size = sum(position_sizes_map.values()) / len(position_sizes_map)
    else:
        avg_position_size = 13.2  # Default estimate
    
    # Calculate date range
    today = datetime.now()
    start_date = today - timedelta(days=days)
    
    # Find the earliest trade date to set a minimum start date
    # This prevents showing empty days before trading actually started
    earliest_trade_date = None
    
    # Check trade log for earliest date
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
    recent_log_trades = [t for t in trade_log if t['timestamp'] >= start_date]
    
    # Estimate initial wallet value if not provided
    if initial_wallet_usd is None:
        initial_wallet_usd = 300.0  # Default starting value
    
    # Create time series of wallet value changes
    # Collect all trade events (entries and exits)
    events = []
    
    # Process trade log entries using actual position sizes
    for trade in recent_log_trades:
        # Get actual position size for this trade
        position_size = get_position_size_for_trade(trade, position_sizes_map, avg_position_size)
        pnl_usd = (trade['pnl_pct'] / 100) * position_size
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
    
    # Add current time point
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

def plot_wallet_value(days=5, initial_wallet_usd=None, save_path=None, use_helius=True):
    """Plot wallet value over time"""
    data = calculate_wallet_value_over_time(days, initial_wallet_usd, use_helius=use_helius)
    
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

def plot_percentage_pnl(days=30, initial_wallet_usd=None, save_path=None, use_helius=True):
    """
    Plot PnL as percentage returns - optimized for marketing and GitHub display
    Clean, professional chart showing percentage returns over time
    """
    data = calculate_wallet_value_over_time(days, initial_wallet_usd, use_helius=use_helius)
    
    if not data:
        print("No data to plot")
        return None
    
    # Calculate percentage returns based on trading PnL and adjusted capital base
    initial = data['initial_wallet']
    if initial <= 0:
        print("Invalid initial wallet value")
        return
    
    # Get trading PnL and deposits/withdrawals data
    trading_pnl = data.get('trading_pnl', data.get('total_pnl', 0.0))
    total_deposits = data.get('total_deposits', 0.0)
    total_withdrawals = data.get('total_withdrawals', 0.0)
    
    # Calculate adjusted capital base for final percentage
    adjusted_capital_base = initial + total_deposits - total_withdrawals
    
    # Calculate percentage returns over time using events (if available)
    # This accounts for deposits/withdrawals properly
    percentage_returns = []
    if 'raw_events' in data and data['raw_events']:
        # Sort events by time
        sorted_events = sorted(data['raw_events'], key=lambda x: x['time'])
        
        # Build a dictionary of cumulative values at each event time
        event_cumulative = {}  # time -> (trading_pnl, deposits, withdrawals)
        cumulative_trading_pnl = 0.0
        cumulative_deposits = 0.0
        cumulative_withdrawals = 0.0
        
        # Process events once and build cumulative values
        for event in sorted_events:
            if event.get('type') == 'sell':
                cumulative_trading_pnl += event.get('pnl', 0.0)
            elif event.get('type') == 'deposit':
                cumulative_deposits += event.get('amount', 0.0)
            elif event.get('type') == 'withdrawal':
                cumulative_withdrawals += event.get('amount', 0.0)
            
            event_cumulative[event['time']] = (cumulative_trading_pnl, cumulative_deposits, cumulative_withdrawals)
        
        # For each time point, find the most recent event and calculate percentage
        for time_point, wallet_value in zip(data['detailed_time_points'], data['detailed_wallet_values']):
            # Find the most recent event up to this time point
            most_recent_cumulative = (0.0, 0.0, 0.0)
            most_recent_time = None
            for event_time, cum_vals in event_cumulative.items():
                if event_time <= time_point:
                    if most_recent_time is None or event_time > most_recent_time:
                        most_recent_time = event_time
                        most_recent_cumulative = cum_vals
            
            cum_trading_pnl, cum_deposits, cum_withdrawals = most_recent_cumulative
            
            # Calculate adjusted capital base at this point
            current_capital_base = initial + cum_deposits - cum_withdrawals
            
            # Calculate percentage return: trading PnL / capital base
            if current_capital_base > 0:
                pct_return = (cum_trading_pnl / current_capital_base) * 100
            else:
                pct_return = 0.0
            
            percentage_returns.append(pct_return)
    else:
        # Fallback: use final trading PnL percentage if events not available
        final_pct = (trading_pnl / adjusted_capital_base * 100) if adjusted_capital_base > 0 else 0.0
        # Create linear approximation (not ideal but better than nothing)
        num_points = len(data['detailed_wallet_values'])
        if num_points > 0:
            percentage_returns = [final_pct * (i / (num_points - 1)) for i in range(num_points)]
        else:
            percentage_returns = []
    
    # Calculate win rate from trade_log.csv (closed trades only - accurate source)
    win_rate, total_closed_trades = calculate_win_rate_from_trade_log(days)
    total_trades = total_closed_trades  # Use closed trades count for display
    
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
    
    # Calculate Sharpe ratio from daily returns
    # Group wallet values by date to match calculate_sharpe_r2.py behavior
    # This ensures we only calculate one return per day (not per trade) and exclude days with no trades
    sharpe_ratio = 0.0
    detailed_values = data.get('detailed_wallet_values', [])
    detailed_times = data.get('detailed_time_points', [])
    
    if len(detailed_values) > 1 and len(detailed_times) == len(detailed_values):
        # Group by date to get end-of-day wallet values (one per day)
        from collections import defaultdict
        daily_end_values = {}
        for i, (time_point, wallet_value) in enumerate(zip(detailed_times, detailed_values)):
            date_key = time_point.date()
            # Keep the last value for each day (end of day)
            daily_end_values[date_key] = wallet_value
        
        # Sort by date and calculate daily returns
        sorted_dates = sorted(daily_end_values.keys())
        daily_returns = []
        for i in range(1, len(sorted_dates)):
            prev_date = sorted_dates[i-1]
            curr_date = sorted_dates[i]
            prev_value = daily_end_values[prev_date]
            curr_value = daily_end_values[curr_date]
            
            # Skip days where portfolio value is zero or negative (can't calculate meaningful return)
            # This ensures consistency with calculate_sharpe_r2.py and proper volatility calculations
            if prev_value > 0:
                daily_return = (curr_value - prev_value) / prev_value
                daily_returns.append(daily_return)
        
        # Calculate Sharpe ratio (annualized, risk-free rate = 0 for crypto)
        if len(daily_returns) > 1:
            returns_array = np.array(daily_returns)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array, ddof=1)  # Sample standard deviation
            
            if std_return > 0:
                # Annualize: daily returns * 365, volatility * sqrt(365)
                annualized_return = mean_return * 365
                annualized_volatility = std_return * np.sqrt(365)
                risk_free_rate = 0.0  # Assume 0% for crypto
                sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    
    stats_text = f"""Key Metrics
━━━━━━━━━━━━━━━━━━━━
Total Return: {total_return:+.2f}%
Avg Daily: {avg_daily_return:+.2f}%
Sharpe Ratio: {sharpe_ratio:.2f}
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
    parser.add_argument('--use-helius', action='store_true', default=True,
                       help='Use Helius data for accurate chart generation (default: True)')
    parser.add_argument('--no-helius', dest='use_helius', action='store_false',
                       help='Use trade_log.csv instead of Helius data')
    
    args = parser.parse_args()
    
    try:
        if args.type == 'percentage':
            result = plot_percentage_pnl(days=args.days, initial_wallet_usd=args.initial, save_path=args.save, use_helius=args.use_helius)
        else:
            result = plot_wallet_value(days=args.days, initial_wallet_usd=args.initial, save_path=args.save, use_helius=args.use_helius)
        
        if result is None:
            print("Warning: No data available to generate chart")
            sys.exit(0)  # Exit successfully even if no data
    except Exception as e:
        print(f"Error generating chart: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

