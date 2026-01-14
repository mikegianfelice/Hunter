#!/usr/bin/env python3
"""
Plot Wallet Value Over Time
Visualizes PnL by plotting wallet value over time
Includes percentage-based charts optimized for marketing
"""

from datetime import datetime, timedelta, timezone
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

# Official start date when bot officially began running after major changes
OFFICIAL_START_DATE = datetime(2026, 1, 14, 0, 0, 0, tzinfo=timezone.utc)


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
        
        # Calculate date range (use timezone-aware datetimes)
        today = datetime.now(timezone.utc)
        calculated_start = (today - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
        # Enforce minimum start date of January 14, 2026 (official bot start date)
        start_date = max(calculated_start, OFFICIAL_START_DATE)
        
        # Set default initial wallet if not provided
        if initial_wallet_usd is None:
            initial_wallet_usd = 200.0
        
        # Call Helius calculation function (use a bit earlier start to ensure we have data)
        # but we'll filter to the exact date range requested
        # Use a 1-day buffer, but don't go before official start date
        helius_start_buffer = (start_date - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        helius_start_date = max(helius_start_buffer, OFFICIAL_START_DATE - timedelta(days=1))
        result = calculate_wallet_value_over_time_from_helius(
            start_date_str=helius_start_date.strftime('%Y-%m-%d'),
            end_date_str=today.strftime('%Y-%m-%d'),
            initial_wallet=initial_wallet_usd,
            force_refresh=False  # Use cache for efficiency
        )
        
        if not result:
            return None
        
        # Filter results to the exact date range requested
        time_points_all = result['time_points']
        wallet_values_all = result['wallet_values']
        events_raw_all = result.get('events', [])
        
        # Ensure start_date is timezone-aware for comparison (Helius returns timezone-aware datetimes)
        if time_points_all and isinstance(time_points_all[0], datetime):
            if time_points_all[0].tzinfo is not None and start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=timezone.utc)
        
        # Filter time_points and wallet_values to requested date range
        time_points = []
        wallet_values = []
        for tp, wv in zip(time_points_all, wallet_values_all):
            # Make sure tp is timezone-aware if start_date is
            if isinstance(tp, datetime):
                if start_date.tzinfo is not None and tp.tzinfo is None:
                    tp = tp.replace(tzinfo=timezone.utc)
                elif start_date.tzinfo is None and tp.tzinfo is not None:
                    tp = tp.replace(tzinfo=None)
            
            if tp >= start_date:
                time_points.append(tp)
                wallet_values.append(wv)
        
        # Filter events to requested date range
        events_raw = []
        for e in events_raw_all:
            event_time = e.get('time')
            if event_time:
                # Make sure event_time is comparable to start_date
                if isinstance(event_time, datetime):
                    if start_date.tzinfo is not None and event_time.tzinfo is None:
                        event_time = event_time.replace(tzinfo=timezone.utc)
                    elif start_date.tzinfo is None and event_time.tzinfo is not None:
                        event_time = event_time.replace(tzinfo=None)
                
                if event_time >= start_date:
                    events_raw.append(e)
        
        # Determine the actual starting wallet value at the beginning of the period
        # This should be the wallet value at the start of the period, not the original $200
        actual_starting_wallet = initial_wallet_usd
        
        # If no data points in range, use the last point from before the range as starting point
        if not time_points and time_points_all:
            # Find the last point before start_date to use as initial value
            for i in range(len(time_points_all) - 1, -1, -1):
                tp = time_points_all[i]
                if isinstance(tp, datetime):
                    if start_date.tzinfo is not None and tp.tzinfo is None:
                        tp = tp.replace(tzinfo=timezone.utc)
                    elif start_date.tzinfo is None and tp.tzinfo is not None:
                        tp = tp.replace(tzinfo=None)
                
                if tp < start_date:
                    time_points.insert(0, start_date)
                    wallet_values.insert(0, wallet_values_all[i])
                    # Update to the actual wallet value at period start
                    actual_starting_wallet = wallet_values_all[i]
                    break
        elif time_points and wallet_values:
            # Use the first wallet value in the period as the actual starting balance
            actual_starting_wallet = wallet_values[0]
        
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
            'initial_wallet': actual_starting_wallet,  # Use actual starting wallet at period start, not original $200
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
    Calculate wallet value over time from trades using Helius data (the source of truth).
    """
    if not use_helius:
        raise ValueError("Helius data is required. Set use_helius=True (default).")
    
    helius_result = calculate_wallet_value_over_time_from_helius_wrapper(days, initial_wallet_usd)
    if not helius_result:
        raise RuntimeError("Failed to calculate wallet value from Helius data. Please ensure Helius API is configured correctly.")
    
    return helius_result

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
    # For percentage calculation, we should only count deposits/withdrawals DURING the period
    # Calculate from raw_events if available (they're filtered to the period)
    if 'raw_events' in data and data['raw_events']:
        deposits_in_period = sum(e.get('amount', 0.0) for e in data['raw_events'] if e.get('type') == 'deposit')
        withdrawals_in_period = sum(e.get('amount', 0.0) for e in data['raw_events'] if e.get('type') == 'withdrawal')
        total_deposits = deposits_in_period
        total_withdrawals = withdrawals_in_period
    else:
        # Fallback to total deposits/withdrawals (may include pre-period)
        total_deposits = data.get('total_deposits', 0.0)
        total_withdrawals = data.get('total_withdrawals', 0.0)
    
    trading_pnl = data.get('trading_pnl', data.get('total_pnl', 0.0))
    
    # Calculate adjusted capital base for final percentage
    adjusted_capital_base = initial + total_deposits - total_withdrawals
    
    # Calculate percentage returns over time using events (if available)
    # This accounts for deposits/withdrawals properly
    # The first point should always be 0% (starting point)
    percentage_returns = []
    if 'raw_events' in data and data['raw_events']:
        # Sort events by time
        sorted_events = sorted(data['raw_events'], key=lambda x: x['time'])
        
        # Create a list of (time, trading_pnl_delta, deposit_delta, withdrawal_delta) for efficient lookup
        event_deltas = []
        for event in sorted_events:
            if event.get('type') == 'sell':
                event_deltas.append((event['time'], event.get('pnl', 0.0), 0.0, 0.0))
            elif event.get('type') == 'deposit':
                event_deltas.append((event['time'], 0.0, event.get('amount', 0.0), 0.0))
            elif event.get('type') == 'withdrawal':
                event_deltas.append((event['time'], 0.0, 0.0, event.get('amount', 0.0)))
        
        # Process time points with events incrementally
        # Since events are already filtered to the period, we can reset everything at the start
        cumulative_trading_pnl_since_start = 0.0
        cumulative_deposits_since_start = 0.0
        cumulative_withdrawals_since_start = 0.0
        event_index = 0
        
        # Use the initial wallet as the starting capital base (not the wallet value at period start)
        # The wallet value at period start may already include withdrawals from before the period,
        # which would incorrectly reduce the capital base and make returns appear worse
        # We'll add/subtract deposits/withdrawals that happen DURING the period
        period_start_capital_base = initial
        
        for time_point, wallet_value in zip(data['detailed_time_points'], data['detailed_wallet_values']):
            # First point should always be 0%
            if len(percentage_returns) == 0:
                percentage_returns.append(0.0)
                # Don't process events at first point - they'll be counted from second point onward
            else:
                # Process all events up to this time point (events during the period)
                while event_index < len(event_deltas) and event_deltas[event_index][0] <= time_point:
                    _, pnl_delta, deposit_delta, withdrawal_delta = event_deltas[event_index]
                    cumulative_trading_pnl_since_start += pnl_delta
                    cumulative_deposits_since_start += deposit_delta
                    cumulative_withdrawals_since_start += withdrawal_delta
                    event_index += 1
                
                # Calculate adjusted capital base at this point
                # Starting base + deposits during period - withdrawals during period
                current_capital_base = period_start_capital_base + cumulative_deposits_since_start - cumulative_withdrawals_since_start
                
                # Calculate percentage return: trading PnL since period start / current capital base
                if current_capital_base > 0:
                    pct_return = (cumulative_trading_pnl_since_start / current_capital_base) * 100
                else:
                    pct_return = 0.0
                
                percentage_returns.append(pct_return)
    else:
        # Fallback: use final trading PnL percentage if events not available
        final_pct = (trading_pnl / adjusted_capital_base * 100) if adjusted_capital_base > 0 else 0.0
        # Create linear approximation (not ideal but better than nothing)
        # First point should always be 0%
        num_points = len(data['detailed_wallet_values'])
        if num_points > 0:
            percentage_returns = [0.0] + [final_pct * (i / (num_points - 1)) for i in range(1, num_points)]
        else:
            percentage_returns = []
    
    # Calculate win rate from Helius sell events (always use Helius data)
    win_rate = 0.0
    total_trades = 0
    if 'raw_events' in data and data['raw_events']:
        sell_events = [e for e in data['raw_events'] if e.get('type') == 'sell']
        wins = 0
        losses = 0
        for event in sell_events:
            pnl = event.get('pnl', 0.0)
            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1
            # Breakeven trades (pnl == 0) are not counted in win rate
        total_closed_trades = wins + losses
        win_rate = (wins / total_closed_trades * 100) if total_closed_trades > 0 else 0.0
        total_trades = total_closed_trades
    
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
    
    # Calculate Sharpe ratio from daily percentage returns
    # Use percentage_returns which already accounts for deposits/withdrawals correctly
    sharpe_ratio = 0.0
    if len(percentage_returns) > 1 and len(data['detailed_time_points']) == len(percentage_returns):
        # Group percentage returns by date to get one return per day (end of day)
        from collections import defaultdict
        daily_end_percentages = {}
        for i, (time_point, pct_return) in enumerate(zip(data['detailed_time_points'], percentage_returns)):
            date_key = time_point.date() if hasattr(time_point, 'date') else time_point.date()
            # Keep the last value for each day (end of day)
            daily_end_percentages[date_key] = pct_return
        
        # Sort by date and calculate daily returns
        # Since percentage_returns are cumulative from period start, the difference between
        # consecutive days gives us the incremental return for that day
        sorted_dates = sorted(daily_end_percentages.keys())
        daily_returns = []
        for i in range(1, len(sorted_dates)):
            prev_date = sorted_dates[i-1]
            curr_date = sorted_dates[i]
            prev_pct = daily_end_percentages[prev_date]
            curr_pct = daily_end_percentages[curr_date]
            
            # Calculate incremental percentage point change
            # This represents the return for that day relative to the capital base
            # Since percentage_returns account for deposits/withdrawals, this is the correct daily return
            pct_point_change = curr_pct - prev_pct
            
            # Convert percentage points to decimal return (e.g., 1.5% -> 0.015)
            daily_return = pct_point_change / 100.0
            daily_returns.append(daily_return)
        
        # Calculate Sharpe ratio (annualized, risk-free rate = 0 for crypto)
        if len(daily_returns) > 1:
            returns_array = np.array(daily_returns)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array, ddof=1)  # Sample standard deviation
            
            if std_return > 0:
                # Annualize: mean daily return * 365, volatility * sqrt(365)
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
    # Helius is always used (it's the source of truth)
    # The use_helius parameter is kept for API compatibility but is always True
    
    args = parser.parse_args()
    
    try:
        # Always use Helius (it's the source of truth)
        if args.type == 'percentage':
            result = plot_percentage_pnl(days=args.days, initial_wallet_usd=args.initial, save_path=args.save, use_helius=True)
        else:
            result = plot_wallet_value(days=args.days, initial_wallet_usd=args.initial, save_path=args.save, use_helius=True)
        
        if result is None:
            print("Warning: No data available to generate chart")
            sys.exit(0)  # Exit successfully even if no data
    except Exception as e:
        print(f"Error generating chart: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

