#!/usr/bin/env python3
"""
Calculate Sharpe Ratio and R-squared for Trading Bot Performance
"""

import json
import csv
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import statistics

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
                        'pnl_pct': float(row['pnl_pct']),
                    })
                except Exception as e:
                    continue
    except Exception as e:
        print(f"Error loading trade log: {e}")
    return trades

def load_performance_data():
    """Load performance data from JSON file"""
    try:
        data_file = PROJECT_ROOT / 'data' / 'performance_data.json'
        with open(data_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading performance data: {e}")
        return None

def calculate_daily_returns(trades: List[Dict], days: int = None, initial_wallet: float = 300.0, avg_position_size: float = 13.2) -> Tuple[List[float], List[datetime]]:
    """
    Calculate daily portfolio returns from trades.
    Converts trade-level PnL percentages to portfolio-level returns.
    
    Args:
        trades: List of trades with 'timestamp' and 'pnl_pct'
        days: Number of days to analyze (None for all)
        initial_wallet: Initial portfolio value in USD
        avg_position_size: Average position size in USD
    
    Returns: (daily_returns, dates)
    """
    if not trades:
        return [], []
    
    # Determine date range
    if days:
        cutoff_date = datetime.now() - timedelta(days=days)
        trades = [t for t in trades if t['timestamp'] >= cutoff_date]
    
    if not trades:
        return [], []
    
    # Sort trades by timestamp
    trades = sorted(trades, key=lambda x: x['timestamp'])
    
    # Group trades by date and calculate daily dollar PnL
    daily_pnl_usd = defaultdict(float)
    for trade in trades:
        date = trade['timestamp'].date()
        # Convert trade PnL percentage to dollar PnL
        # pnl_pct is return on position size, not portfolio
        pnl_usd = (trade['pnl_pct'] / 100.0) * avg_position_size
        daily_pnl_usd[date] += pnl_usd
    
    # Calculate portfolio value over time and daily returns
    sorted_dates = sorted(daily_pnl_usd.keys())
    daily_returns = []
    dates = []
    
    current_portfolio_value = initial_wallet
    
    for date in sorted_dates:
        # Calculate portfolio return for this day
        daily_pnl = daily_pnl_usd[date]
        daily_return = daily_pnl / current_portfolio_value if current_portfolio_value > 0 else 0.0
        
        daily_returns.append(daily_return)
        dates.append(datetime.combine(date, datetime.min.time()))
        
        # Update portfolio value for next day
        current_portfolio_value += daily_pnl
    
    return daily_returns, dates

def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0, periods_per_year: int = 365) -> float:
    """
    Calculate Sharpe Ratio.
    
    Args:
        returns: List of periodic returns (as decimals, e.g., 0.05 for 5%)
        risk_free_rate: Annual risk-free rate (default: 0.0 for crypto)
        periods_per_year: Number of periods per year (365 for daily, 252 for trading days)
    
    Returns:
        Sharpe ratio (annualized)
    """
    if not returns or len(returns) < 2:
        return 0.0
    
    returns_array = np.array(returns)
    mean_return = np.mean(returns_array)
    std_return = np.std(returns_array, ddof=1)  # Sample standard deviation
    
    if std_return == 0:
        return 0.0
    
    # Annualize returns and volatility
    annualized_return = mean_return * periods_per_year
    annualized_volatility = std_return * np.sqrt(periods_per_year)
    
    # Sharpe ratio
    sharpe = (annualized_return - risk_free_rate) / annualized_volatility
    
    return sharpe

def calculate_r_squared(bot_returns: List[float], benchmark_returns: List[float]) -> float:
    """
    Calculate R-squared (coefficient of determination).
    Measures how well bot returns fit the benchmark returns.
    
    Args:
        bot_returns: List of bot returns
        benchmark_returns: List of benchmark returns (same length)
    
    Returns:
        R-squared value (0 to 1, where 1 = perfect correlation)
    """
    if len(bot_returns) != len(benchmark_returns) or len(bot_returns) < 2:
        return 0.0
    
    bot_array = np.array(bot_returns)
    benchmark_array = np.array(benchmark_returns)
    
    # Calculate R-squared using linear regression
    # R² = 1 - (SS_res / SS_tot)
    # where SS_res = sum of squared residuals
    # and SS_tot = total sum of squares
    
    # Simple linear regression: bot_return = a + b * benchmark_return
    # Using numpy for efficiency
    if np.std(benchmark_array) == 0:
        return 0.0
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(bot_array, benchmark_array)[0, 1]
    
    # R-squared is the square of correlation coefficient
    r_squared = correlation ** 2 if not np.isnan(correlation) else 0.0
    
    return r_squared

def get_benchmark_returns(dates: List[datetime], benchmark_type: str = 'zero') -> List[float]:
    """
    Get benchmark returns for comparison.
    
    Args:
        dates: List of dates
        benchmark_type: 'zero' (assume 0% market return), 'sol' (SOL price), or 'market' (assume small positive)
    
    Returns:
        List of benchmark returns
    """
    if benchmark_type == 'zero':
        # Assume market returns 0% (neutral benchmark)
        return [0.0] * len(dates)
    elif benchmark_type == 'market':
        # Assume market returns 0.1% per day (roughly 36.5% annually)
        return [0.001] * len(dates)
    elif benchmark_type == 'sol':
        # Try to load SOL price data
        try:
            sol_file = PROJECT_ROOT / 'data' / 'sol_price_cache.json'
            # This is a simple cache, we'd need historical SOL prices for proper benchmark
            # For now, return zero
            return [0.0] * len(dates)
        except:
            return [0.0] * len(dates)
    else:
        return [0.0] * len(dates)

def estimate_position_size():
    """Estimate average position size from performance data"""
    data = load_performance_data()
    if data:
        trades = data.get('trades', [])
        position_sizes = [t.get('position_size_usd', 0) for t in trades if t.get('position_size_usd')]
        if position_sizes:
            return sum(position_sizes) / len(position_sizes)
    return 13.2  # Default estimate

def calculate_metrics(days: int = None, risk_free_rate: float = 0.0, benchmark_type: str = 'zero', initial_wallet: float = 300.0) -> Dict:
    """
    Calculate Sharpe ratio and R-squared for the bot.
    
    Args:
        days: Number of days to analyze (None for all data)
        risk_free_rate: Annual risk-free rate (default: 0.0)
        benchmark_type: Benchmark type ('zero', 'market', 'sol')
        initial_wallet: Initial portfolio value in USD (default: 300.0)
    
    Returns:
        Dictionary with metrics
    """
    # Load trade data
    trades = load_trade_log()
    
    if not trades:
        return {
            'error': 'No trade data available',
            'sharpe_ratio': 0.0,
            'r_squared': 0.0,
            'total_trades': 0,
            'period_days': 0
        }
    
    # Estimate position size
    avg_position_size = estimate_position_size()
    
    # Calculate daily returns (portfolio-level, not trade-level)
    daily_returns, dates = calculate_daily_returns(trades, days, initial_wallet, avg_position_size)
    
    if not daily_returns:
        return {
            'error': 'No returns data available',
            'sharpe_ratio': 0.0,
            'r_squared': 0.0,
            'total_trades': len(trades),
            'period_days': 0
        }
    
    # Calculate Sharpe ratio
    sharpe_ratio = calculate_sharpe_ratio(daily_returns, risk_free_rate, periods_per_year=365)
    
    # Get benchmark returns
    benchmark_returns = get_benchmark_returns(dates, benchmark_type)
    
    # Calculate R-squared
    r_squared = calculate_r_squared(daily_returns, benchmark_returns)
    
    # Additional statistics
    # Calculate total return using compound formula: (1 + r1) * (1 + r2) * ... - 1
    if daily_returns:
        total_return = np.prod([1 + r for r in daily_returns]) - 1
    else:
        total_return = 0.0
    
    mean_daily_return = statistics.mean(daily_returns) if daily_returns else 0.0
    std_daily_return = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 0.0
    annualized_return = mean_daily_return * 365
    annualized_volatility = std_daily_return * np.sqrt(365)
    
    # Calculate win rate
    winning_days = len([r for r in daily_returns if r > 0])
    win_rate = (winning_days / len(daily_returns) * 100) if daily_returns else 0.0
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'r_squared': r_squared,
        'total_trades': len(trades),
        'period_days': len(daily_returns),
        'total_return': total_return,
        'total_return_pct': total_return * 100,
        'mean_daily_return': mean_daily_return,
        'mean_daily_return_pct': mean_daily_return * 100,
        'annualized_return': annualized_return,
        'annualized_return_pct': annualized_return * 100,
        'annualized_volatility': annualized_volatility,
        'annualized_volatility_pct': annualized_volatility * 100,
        'win_rate': win_rate,
        'dates': [d.strftime('%Y-%m-%d') for d in dates],
        'benchmark_type': benchmark_type,
        'initial_wallet': initial_wallet,
        'avg_position_size': avg_position_size
    }

def print_metrics(metrics: Dict):
    """Print formatted metrics report"""
    if 'error' in metrics:
        print(f"❌ Error: {metrics['error']}")
        return
    
    print("\n" + "="*70)
    print("📊 SHARPE RATIO & R-SQUARED ANALYSIS")
    print("="*70)
    
    print(f"\n📅 Period: {metrics['period_days']} days")
    print(f"📈 Total Trades: {metrics['total_trades']}")
    
    print(f"\n🎯 KEY METRICS:")
    print(f"   • Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    
    # Sharpe ratio interpretation
    sharpe = metrics['sharpe_ratio']
    if sharpe > 2.0:
        interpretation = "Excellent (Very Good Risk-Adjusted Returns)"
    elif sharpe > 1.5:
        interpretation = "Very Good"
    elif sharpe > 1.0:
        interpretation = "Good"
    elif sharpe > 0.5:
        interpretation = "Acceptable"
    elif sharpe > 0:
        interpretation = "Below Average"
    else:
        interpretation = "Poor (Negative Risk-Adjusted Returns)"
    
    print(f"      └─ Interpretation: {interpretation}")
    print(f"   • R-Squared: {metrics['r_squared']:.4f} ({metrics['r_squared']*100:.2f}%)")
    
    # R-squared interpretation
    r2 = metrics['r_squared']
    if r2 > 0.7:
        r2_interp = "High correlation with benchmark"
    elif r2 > 0.5:
        r2_interp = "Moderate correlation with benchmark"
    elif r2 > 0.3:
        r2_interp = "Low correlation with benchmark"
    else:
        r2_interp = "Very low correlation (independent performance)"
    
    print(f"      └─ Interpretation: {r2_interp}")
    
    print(f"\n💰 RETURN STATISTICS:")
    print(f"   • Total Return: {metrics['total_return_pct']:+.2f}%")
    print(f"   • Mean Daily Return: {metrics['mean_daily_return_pct']:+.4f}%")
    print(f"   • Annualized Return: {metrics['annualized_return_pct']:+.2f}%")
    print(f"   • Annualized Volatility: {metrics['annualized_volatility_pct']:.2f}%")
    print(f"   • Win Rate: {metrics['win_rate']:.1f}%")
    
    print(f"\n📊 BENCHMARK:")
    print(f"   • Type: {metrics['benchmark_type']}")
    if metrics['benchmark_type'] == 'zero':
        print(f"      └─ Assumed 0% daily market return (neutral benchmark)")
    elif metrics['benchmark_type'] == 'market':
        print(f"      └─ Assumed 0.1% daily market return (~36.5% annually)")
    
    print(f"\n💼 PORTFOLIO ASSUMPTIONS:")
    print(f"   • Initial Wallet: ${metrics.get('initial_wallet', 300.0):.2f}")
    print(f"   • Avg Position Size: ${metrics.get('avg_position_size', 13.2):.2f}")
    
    print("\n" + "="*70 + "\n")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate Sharpe Ratio and R-squared for trading bot')
    parser.add_argument('--days', type=int, default=None, 
                       help='Number of days to analyze (default: all available data)')
    parser.add_argument('--risk-free-rate', type=float, default=0.0,
                       help='Annual risk-free rate (default: 0.0 for crypto)')
    parser.add_argument('--benchmark', type=str, choices=['zero', 'market', 'sol'], default='zero',
                       help='Benchmark type: zero (0%%), market (0.1%% daily), or sol (SOL price)')
    parser.add_argument('--initial-wallet', type=float, default=300.0,
                       help='Initial portfolio value in USD (default: 300.0)')
    
    args = parser.parse_args()
    
    metrics = calculate_metrics(
        days=args.days,
        risk_free_rate=args.risk_free_rate,
        benchmark_type=args.benchmark,
        initial_wallet=args.initial_wallet
    )
    
    print_metrics(metrics)

