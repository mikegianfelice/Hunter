#!/usr/bin/env python3
"""
Calculate win rate for the last 7 days using the charts win rate method.
This method uses Helius sell events and counts:
- Wins: pnl > 0
- Losses: pnl < 0
- Breakeven trades (pnl == 0) are not counted
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

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
CACHE_FILE = PROJECT_ROOT / "data" / "helius_wallet_value_cache.json"

def calculate_win_rate_charts_method(days=7):
    """
    Calculate win rate using the charts method (from plot_wallet_value.py).
    This method uses Helius sell events from the cache file.
    """
    try:
        if not CACHE_FILE.exists():
            print(f"Cache file not found: {CACHE_FILE}")
            return None
        
        # Load cache
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
        
        completed_trades = cache.get('completed_trades', [])
        
        if not completed_trades:
            print("No completed trades found in cache")
            return None
        
        # Calculate cutoff date (last N days)
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Get initial wallet and deposits/withdrawals for percentage calculation
        initial_wallet = cache.get('initial_wallet', 200.0)
        deposits = cache.get('deposits', [])
        withdrawals = cache.get('withdrawals', [])
        
        # Filter deposits and withdrawals from the last N days
        deposits_in_period = []
        withdrawals_in_period = []
        
        for deposit in deposits:
            deposit_time_str = deposit.get('time')
            if deposit_time_str:
                try:
                    if '+' in deposit_time_str:
                        deposit_time = datetime.fromisoformat(deposit_time_str)
                    else:
                        deposit_time = datetime.strptime(deposit_time_str, '%Y-%m-%d %H:%M:%S')
                        deposit_time = deposit_time.replace(tzinfo=timezone.utc)
                    if deposit_time >= cutoff_date:
                        deposits_in_period.append(deposit)
                except Exception:
                    pass
        
        for withdrawal in withdrawals:
            withdrawal_time_str = withdrawal.get('time')
            if withdrawal_time_str:
                try:
                    if '+' in withdrawal_time_str:
                        withdrawal_time = datetime.fromisoformat(withdrawal_time_str)
                    else:
                        withdrawal_time = datetime.strptime(withdrawal_time_str, '%Y-%m-%d %H:%M:%S')
                        withdrawal_time = withdrawal_time.replace(tzinfo=timezone.utc)
                    if withdrawal_time >= cutoff_date:
                        withdrawals_in_period.append(withdrawal)
                except Exception:
                    pass
        
        # Calculate total deposits and withdrawals in period
        total_deposits_in_period = sum(d.get('amount', 0.0) for d in deposits_in_period)
        total_withdrawals_in_period = sum(w.get('amount', 0.0) for w in withdrawals_in_period)
        
        # Filter trades from the last N days and count wins/losses, calculate PnL
        wins = 0
        losses = 0
        total_pnl_usd = 0.0
        
        for trade in completed_trades:
            sell_time_str = trade.get('sell_time')
            if not sell_time_str:
                continue
            
            # Parse sell time
            try:
                # Handle different time formats
                if '+' in sell_time_str:
                    sell_time = datetime.fromisoformat(sell_time_str)
                else:
                    # Try parsing without timezone
                    sell_time = datetime.strptime(sell_time_str, '%Y-%m-%d %H:%M:%S')
                    sell_time = sell_time.replace(tzinfo=timezone.utc)
            except Exception as e:
                print(f"Warning: Could not parse sell_time '{sell_time_str}': {e}")
                continue
            
            # Only count trades from the last N days
            if sell_time < cutoff_date:
                continue
            
            # Get PnL
            pnl = trade.get('pnl_usd', 0.0)
            total_pnl_usd += pnl
            
            # Count wins and losses (same logic as charts method)
            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1
            # Breakeven trades (pnl == 0) are not counted in win rate
        
        total_closed_trades = wins + losses
        win_rate = (wins / total_closed_trades * 100) if total_closed_trades > 0 else 0.0
        
        # Calculate percentage PnL
        # Need to get the wallet value at the start of the period
        # For simplicity, we'll use initial_wallet + all deposits before period - all withdrawals before period
        # But for accuracy, we should get the wallet value at cutoff_date
        # For now, use a simplified approach: adjusted capital base = initial + deposits - withdrawals
        # This is an approximation - the exact method would need to track wallet value over time
        
        # Get all deposits and withdrawals before the period
        deposits_before = 0.0
        for d in deposits:
            deposit_time_str = d.get('time')
            if deposit_time_str:
                try:
                    if '+' in deposit_time_str:
                        deposit_time = datetime.fromisoformat(deposit_time_str)
                    else:
                        deposit_time = datetime.strptime(deposit_time_str, '%Y-%m-%d %H:%M:%S')
                        deposit_time = deposit_time.replace(tzinfo=timezone.utc)
                    if deposit_time < cutoff_date:
                        deposits_before += d.get('amount', 0.0)
                except Exception:
                    pass
        
        withdrawals_before = 0.0
        for w in withdrawals:
            withdrawal_time_str = w.get('time')
            if withdrawal_time_str:
                try:
                    if '+' in withdrawal_time_str:
                        withdrawal_time = datetime.fromisoformat(withdrawal_time_str)
                    else:
                        withdrawal_time = datetime.strptime(withdrawal_time_str, '%Y-%m-%d %H:%M:%S')
                        withdrawal_time = withdrawal_time.replace(tzinfo=timezone.utc)
                    if withdrawal_time < cutoff_date:
                        withdrawals_before += w.get('amount', 0.0)
                except Exception:
                    pass
        
        # Adjusted capital base at start of period
        capital_base_at_start = initial_wallet + deposits_before - withdrawals_before
        
        # Adjusted capital base during period (for percentage calculation)
        # Use the capital base at start + deposits during period - withdrawals during period
        adjusted_capital_base = capital_base_at_start + total_deposits_in_period - total_withdrawals_in_period
        
        # Calculate percentage PnL
        pnl_percent = (total_pnl_usd / adjusted_capital_base * 100) if adjusted_capital_base > 0 else 0.0
        
        return {
            'win_rate': win_rate,
            'total_trades': total_closed_trades,
            'wins': wins,
            'losses': losses,
            'pnl_usd': total_pnl_usd,
            'pnl_percent': pnl_percent,
            'days': days,
            'cutoff_date': cutoff_date.strftime('%Y-%m-%d %H:%M:%S UTC'),
            'capital_base_at_start': capital_base_at_start,
            'adjusted_capital_base': adjusted_capital_base,
            'deposits_in_period': total_deposits_in_period,
            'withdrawals_in_period': total_withdrawals_in_period
        }
    
    except Exception as e:
        print(f"Error calculating win rate: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    result = calculate_win_rate_charts_method(days=7)
    
    if result:
        print(f"\n📊 Performance Summary (Last 7 Days) - Charts Method")
        print("=" * 70)
        print(f"Win Rate: {result['win_rate']:.2f}%")
        print(f"Total Closed Trades: {result['total_trades']}")
        print(f"Wins: {result['wins']}")
        print(f"Losses: {result['losses']}")
        print("-" * 70)
        print(f"PnL (USD): ${result['pnl_usd']:+.2f}")
        print(f"PnL (%): {result['pnl_percent']:+.2f}%")
        print("-" * 70)
        print(f"Capital Base at Period Start: ${result['capital_base_at_start']:.2f}")
        print(f"Adjusted Capital Base: ${result['adjusted_capital_base']:.2f}")
        if result['deposits_in_period'] > 0:
            print(f"Deposits in Period: ${result['deposits_in_period']:.2f}")
        if result['withdrawals_in_period'] > 0:
            print(f"Withdrawals in Period: ${result['withdrawals_in_period']:.2f}")
        print("-" * 70)
        print(f"Period: Last {result['days']} days")
        print(f"Cutoff Date: {result['cutoff_date']}")
        print("=" * 70)
        print(f"\nNote: Breakeven trades (pnl == 0) are excluded from win rate calculation.")
        print(f"Method: Uses Helius sell events (same as charts/plot_wallet_value.py)")
    else:
        print("Failed to calculate performance metrics")
