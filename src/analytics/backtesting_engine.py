#!/usr/bin/env python3
"""
Advanced Backtesting Engine - Phase 4
Comprehensive backtesting with strategy optimization and performance analysis
"""

import asyncio
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import logging
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.monitoring.structured_logger import log_info, log_error
from src.config.config_validator import get_validated_config
from src.ai.ai_integration_engine import analyze_token_ai, AIAnalysisResult

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Individual trade record"""
    timestamp: str
    symbol: str
    action: str  # 'buy' or 'sell'
    price: float
    quantity: float
    value: float
    fees: float
    pnl: float
    reason: str
    ai_score: float
    confidence: float

@dataclass
class BacktestResult:
    """Comprehensive backtest results"""
    start_date: str
    end_date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_fees: float
    net_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    profit_factor: float
    avg_trade_pnl: float
    avg_winning_trade: float
    avg_losing_trade: float
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    max_consecutive_wins: int
    max_consecutive_losses: int
    trades_per_day: float
    volatility: float
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional Value at Risk 95%
    monthly_returns: List[float]
    equity_curve: List[float]
    drawdown_curve: List[float]
    trades: List[Trade]

class MarketDataSimulator:
    """Simulates realistic market data for backtesting"""
    
    def __init__(self, start_date: str, end_date: str, symbols: List[str]):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.symbols = symbols
        self.data = {}
        
    def generate_market_data(self) -> Dict[str, pd.DataFrame]:
        """Generate realistic market data for all symbols"""
        log_info(f"Generating market data from {self.start_date} to {self.end_date} for {len(self.symbols)} symbols")
        
        for symbol in self.symbols:
            self.data[symbol] = self._generate_symbol_data(symbol)
        
        return self.data
    
    def _generate_symbol_data(self, symbol: str) -> pd.DataFrame:
        """Generate realistic data for a single symbol"""
        # Create date range
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='1H')
        
        # Generate price data using geometric Brownian motion
        n_periods = len(date_range)
        dt = 1/24/365  # 1 hour in years
        
        # Parameters for different symbols
        if 'BTC' in symbol or 'ETH' in symbol:
            mu = 0.1  # 10% annual return
            sigma = 0.3  # 30% volatility
            initial_price = 50000 if 'BTC' in symbol else 3000
        else:
            mu = 0.15  # 15% annual return
            sigma = 0.5  # 50% volatility
            initial_price = np.random.uniform(0.001, 100)
        
        # Generate random returns
        returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_periods)
        
        # Calculate prices
        prices = [initial_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Generate other metrics
        volumes = np.random.lognormal(10, 1, n_periods)
        market_caps = prices * np.random.uniform(1000000, 10000000, n_periods)
        liquidity = np.random.uniform(50000, 5000000, n_periods)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': date_range,
            'price': prices,
            'volume_24h': volumes,
            'market_cap': market_caps,
            'liquidity': liquidity,
            'price_change_24h': np.concatenate([[0], np.diff(prices) / prices[:-1]]),
            'holders': np.random.randint(100, 50000, n_periods),
            'transactions_24h': np.random.randint(100, 10000, n_periods),
            'social_mentions': np.random.poisson(10, n_periods),
            'news_sentiment': np.random.uniform(0.2, 0.8, n_periods),
            'rsi': np.random.uniform(20, 80, n_periods),
            'macd': np.random.uniform(-0.1, 0.1, n_periods),
            'bollinger_upper': prices * 1.1,
            'bollinger_lower': prices * 0.9,
            'moving_avg_20': pd.Series(prices).rolling(20).mean().fillna(prices[0]),
            'moving_avg_50': pd.Series(prices).rolling(50).mean().fillna(prices[0])
        })
        
        return df

class StrategyOptimizer:
    """Optimizes trading strategy parameters using genetic algorithm"""
    
    def __init__(self, population_size: int = 50, generations: int = 100):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        
    def optimize_strategy(self, backtest_engine: 'BacktestEngine', 
                         parameter_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Optimize strategy parameters using genetic algorithm"""
        log_info(f"Starting strategy optimization with {self.population_size} individuals over {self.generations} generations")
        
        # Initialize population
        population = self._initialize_population(parameter_ranges)
        best_individual = None
        best_fitness = -float('inf')
        
        for generation in range(self.generations):
            # Evaluate fitness for each individual
            fitness_scores = []
            for individual in population:
                try:
                    fitness = self._evaluate_fitness(individual, backtest_engine)
                    fitness_scores.append(fitness)
                    
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_individual = individual.copy()
                        
                except Exception as e:
                    log_error(f"Error evaluating individual: {e}")
                    fitness_scores.append(-float('inf'))
            
            # Selection, crossover, and mutation
            new_population = []
            
            # Keep best individual (elitism)
            if best_individual:
                new_population.append(best_individual)
            
            # Generate new individuals
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                if np.random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2, parameter_ranges)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutate children
                if np.random.random() < self.mutation_rate:
                    child1 = self._mutate(child1, parameter_ranges)
                if np.random.random() < self.mutation_rate:
                    child2 = self._mutate(child2, parameter_ranges)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
            
            if generation % 10 == 0:
                log_info(f"Generation {generation}: Best fitness = {best_fitness:.4f}")
        
        log_info(f"Optimization complete. Best fitness: {best_fitness:.4f}")
        return best_individual if best_individual else {}
    
    def _initialize_population(self, parameter_ranges: Dict[str, Tuple[float, float]]) -> List[Dict[str, float]]:
        """Initialize random population"""
        population = []
        for _ in range(self.population_size):
            individual = {}
            for param, (min_val, max_val) in parameter_ranges.items():
                individual[param] = np.random.uniform(min_val, max_val)
            population.append(individual)
        return population
    
    def _evaluate_fitness(self, individual: Dict[str, float], backtest_engine: 'BacktestEngine') -> float:
        """Evaluate fitness of an individual (strategy parameters)"""
        try:
            # Run backtest with these parameters
            result = backtest_engine.run_backtest(individual)
            
            # Calculate fitness score (Sharpe ratio + win rate + profit factor)
            fitness = (
                result.sharpe_ratio * 0.4 +
                result.win_rate * 0.3 +
                result.profit_factor * 0.3
            )
            
            return fitness
            
        except Exception as e:
            log_error(f"Error evaluating fitness: {e}")
            return -float('inf')
    
    def _tournament_selection(self, population: List[Dict], fitness_scores: List[float], 
                            tournament_size: int = 3) -> Dict[str, float]:
        """Tournament selection for parent selection"""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_index].copy()
    
    def _crossover(self, parent1: Dict[str, float], parent2: Dict[str, float], 
                   parameter_ranges: Dict[str, Tuple[float, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Crossover two parents to create two children"""
        child1, child2 = {}, {}
        
        for param in parent1.keys():
            if np.random.random() < 0.5:
                child1[param] = parent1[param]
                child2[param] = parent2[param]
            else:
                child1[param] = parent2[param]
                child2[param] = parent1[param]
        
        return child1, child2
    
    def _mutate(self, individual: Dict[str, float], parameter_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Mutate an individual"""
        for param, (min_val, max_val) in parameter_ranges.items():
            if np.random.random() < 0.1:  # 10% chance to mutate each parameter
                # Gaussian mutation
                mutation = np.random.normal(0, (max_val - min_val) * 0.1)
                individual[param] = max(min_val, min(max_val, individual[param] + mutation))
        
        return individual

class BacktestEngine:
    """Main backtesting engine with comprehensive analysis"""
    
    def __init__(self, config: Any = None):
        self.config = config or get_validated_config()
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.drawdown_curve: List[float] = []
        self.initial_capital = 10000  # $10,000 starting capital
        self.current_capital = self.initial_capital
        self.position = 0
        self.position_value = 0
        self.position_price = 0
        
    def run_backtest(self, strategy_params: Dict[str, Any], 
                    market_data: Dict[str, pd.DataFrame],
                    start_date: str = None, end_date: str = None) -> BacktestResult:
        """Run comprehensive backtest with given strategy parameters"""
        log_info("Starting backtest with strategy parameters", **strategy_params)
        
        # Reset state
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.drawdown_curve = [0]
        self.current_capital = self.initial_capital
        self.position = 0
        self.position_value = 0
        self.position_price = 0
        
        # Get date range
        if start_date is None:
            start_date = min(df.index.min() for df in market_data.values())
        if end_date is None:
            end_date = max(df.index.max() for df in market_data.values())
        
        # Process each symbol
        for symbol, df in market_data.items():
            self._process_symbol(symbol, df, strategy_params, start_date, end_date)
        
        # Calculate results
        result = self._calculate_results(start_date, end_date)
        
        log_info(f"Backtest complete: {result.total_trades} trades, {result.win_rate:.2%} win rate, ${result.net_pnl:.2f} PnL")
        return result
    
    def _process_symbol(self, symbol: str, df: pd.DataFrame, 
                       strategy_params: Dict[str, Any], start_date: str, end_date: str):
        """Process a single symbol through the backtest"""
        # Filter data by date range
        df_filtered = df[(df.index >= start_date) & (df.index <= end_date)]
        
        for idx, row in df_filtered.iterrows():
            try:
                # Convert row to token data format
                token_data = {
                    "symbol": symbol,
                    "timestamp": idx.isoformat(),
                    "priceUsd": row['price'],
                    "volume24h": row['volume_24h'],
                    "marketCap": row['market_cap'],
                    "priceChange24h": row['price_change_24h'],
                    "liquidity": row['liquidity'],
                    "holders": row['holders'],
                    "transactions24h": row['transactions_24h'],
                    "social_mentions": row['social_mentions'],
                    "news_sentiment": row['news_sentiment'],
                    "technical_indicators": {
                        "rsi": row['rsi'],
                        "macd": row['macd'],
                        "bollinger_upper": row['bollinger_upper'],
                        "bollinger_lower": row['bollinger_lower'],
                        "moving_avg_20": row['moving_avg_20'],
                        "moving_avg_50": row['moving_avg_50']
                    }
                }
                
                # Simulate AI analysis (in real implementation, use actual AI)
                ai_result = self._simulate_ai_analysis(token_data, strategy_params)
                
                # Make trading decision
                action = self._make_trading_decision(ai_result, strategy_params)
                
                if action != "hold":
                    self._execute_trade(symbol, row['price'], action, ai_result, strategy_params)
                
                # Update equity curve
                self._update_equity_curve()
                
            except Exception as e:
                log_error(f"Error processing {symbol} at {idx}: {e}")
                continue
    
    def _simulate_ai_analysis(self, token_data: Dict[str, Any], 
                             strategy_params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate AI analysis (in real implementation, use actual AI engine)"""
        # Simplified AI simulation based on technical indicators
        price = token_data['priceUsd']
        rsi = token_data['technical_indicators']['rsi']
        macd = token_data['technical_indicators']['macd']
        ma_20 = token_data['technical_indicators']['moving_avg_20']
        volume = token_data['volume24h']
        liquidity = token_data['liquidity']
        
        # Calculate AI score
        score = 0.5  # Base score
        
        # RSI signals
        if rsi < 30:  # Oversold
            score += 0.2
        elif rsi > 70:  # Overbought
            score -= 0.2
        
        # MACD signals
        if macd > 0:
            score += 0.1
        else:
            score -= 0.1
        
        # Moving average signals
        if price > ma_20:
            score += 0.1
        else:
            score -= 0.1
        
        # Volume and liquidity
        if volume > 100000 and liquidity > 100000:
            score += 0.1
        
        # Clamp score
        score = max(0, min(1, score))
        
        return {
            "overall_score": score,
            "confidence": np.random.uniform(0.6, 0.95),
            "recommendations": {
                "action": "buy" if score > 0.7 else "sell" if score < 0.3 else "hold",
                "position_size": min(20, max(5, score * 25)),
                "take_profit": 0.15,
                "stop_loss": 0.08
            },
            "risk_assessment": {
                "risk_score": 1 - score,
                "risk_level": "low" if score > 0.7 else "high" if score < 0.3 else "medium"
            }
        }
    
    def _make_trading_decision(self, ai_result: Dict[str, Any], 
                              strategy_params: Dict[str, Any]) -> str:
        """Make trading decision based on AI analysis and strategy parameters"""
        score_threshold = strategy_params.get('score_threshold', 0.7)
        confidence_threshold = strategy_params.get('confidence_threshold', 0.8)
        
        overall_score = ai_result.get('overall_score', 0.5)
        confidence = ai_result.get('confidence', 0.5)
        recommendations = ai_result.get('recommendations', {})
        
        # Check thresholds
        if overall_score > score_threshold and confidence > confidence_threshold:
            return recommendations.get('action', 'buy')
        elif overall_score < (1 - score_threshold) and confidence > confidence_threshold:
            return 'sell'
        else:
            return 'hold'
    
    def _execute_trade(self, symbol: str, price: float, action: str, 
                      ai_result: Dict[str, Any], strategy_params: Dict[str, Any]):
        """Execute a trade"""
        try:
            if action == "buy" and self.position == 0:
                # Calculate position size
                position_size = ai_result.get('recommendations', {}).get('position_size', 10)
                max_position_value = self.current_capital * strategy_params.get('max_position_pct', 0.1)
                position_value = min(position_size * 100, max_position_value)  # $100 per position unit
                
                if position_value > 0:
                    quantity = position_value / price
                    fees = position_value * 0.001  # 0.1% fee
                    
                    trade = Trade(
                        timestamp=datetime.now().isoformat(),
                        symbol=symbol,
                        action="buy",
                        price=price,
                        quantity=quantity,
                        value=position_value,
                        fees=fees,
                        pnl=0,
                        reason=f"AI score: {ai_result.get('overall_score', 0):.2f}",
                        ai_score=ai_result.get('overall_score', 0),
                        confidence=ai_result.get('confidence', 0)
                    )
                    
                    self.trades.append(trade)
                    self.position = quantity
                    self.position_value = position_value
                    self.position_price = price
                    self.current_capital -= (position_value + fees)
                    
            elif action == "sell" and self.position > 0:
                # Close position
                sell_value = self.position * price
                fees = sell_value * 0.001
                pnl = sell_value - self.position_value - fees
                
                trade = Trade(
                    timestamp=datetime.now().isoformat(),
                    symbol=symbol,
                    action="sell",
                    price=price,
                    quantity=self.position,
                    value=sell_value,
                    fees=fees,
                    pnl=pnl,
                    reason=f"AI score: {ai_result.get('overall_score', 0):.2f}",
                    ai_score=ai_result.get('overall_score', 0),
                    confidence=ai_result.get('confidence', 0)
                )
                
                self.trades.append(trade)
                self.current_capital += sell_value - fees
                self.position = 0
                self.position_value = 0
                self.position_price = 0
                
        except Exception as e:
            log_error(f"Error executing trade: {e}")
    
    def _update_equity_curve(self):
        """Update equity curve with current portfolio value"""
        current_value = self.current_capital
        if self.position > 0:
            # Estimate current position value (simplified)
            current_value += self.position_value
        
        self.equity_curve.append(current_value)
        
        # Calculate drawdown
        peak = max(self.equity_curve)
        drawdown = (peak - current_value) / peak if peak > 0 else 0
        self.drawdown_curve.append(drawdown)
    
    def _calculate_results(self, start_date: str, end_date: str) -> BacktestResult:
        """Calculate comprehensive backtest results"""
        if not self.trades:
            return BacktestResult(
                start_date=start_date, end_date=end_date,
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, total_pnl=0, total_fees=0, net_pnl=0,
                max_drawdown=0, sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
                profit_factor=0, avg_trade_pnl=0, avg_winning_trade=0, avg_losing_trade=0,
                largest_win=0, largest_loss=0, consecutive_wins=0, consecutive_losses=0,
                max_consecutive_wins=0, max_consecutive_losses=0, trades_per_day=0,
                volatility=0, var_95=0, cvar_95=0, monthly_returns=[], equity_curve=[],
                drawdown_curve=[], trades=[]
            )
        
        # Basic statistics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.pnl > 0])
        losing_trades = len([t for t in self.trades if t.pnl < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # PnL statistics
        total_pnl = sum(t.pnl for t in self.trades)
        total_fees = sum(t.fees for t in self.trades)
        net_pnl = total_pnl - total_fees
        
        # Trade statistics
        pnls = [t.pnl for t in self.trades if t.pnl != 0]
        avg_trade_pnl = np.mean(pnls) if pnls else 0
        avg_winning_trade = np.mean([t.pnl for t in self.trades if t.pnl > 0]) if winning_trades > 0 else 0
        avg_losing_trade = np.mean([t.pnl for t in self.trades if t.pnl < 0]) if losing_trades > 0 else 0
        largest_win = max([t.pnl for t in self.trades if t.pnl > 0], default=0)
        largest_loss = min([t.pnl for t in self.trades if t.pnl < 0], default=0)
        
        # Consecutive wins/losses
        consecutive_wins, consecutive_losses = 0, 0
        max_consecutive_wins, max_consecutive_losses = 0, 0
        current_wins, current_losses = 0, 0
        
        for trade in self.trades:
            if trade.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            elif trade.pnl < 0:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
        
        # Risk metrics
        max_drawdown = max(self.drawdown_curve) if self.drawdown_curve else 0
        
        # Calculate returns for Sharpe ratio
        returns = []
        for i in range(1, len(self.equity_curve)):
            ret = (self.equity_curve[i] - self.equity_curve[i-1]) / self.equity_curve[i-1]
            returns.append(ret)
        
        if returns:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            sortino_ratio = np.mean(returns) / np.std([r for r in returns if r < 0]) * np.sqrt(252) if any(r < 0 for r in returns) else 0
            volatility = np.std(returns) * np.sqrt(252)
            
            # VaR and CVaR
            var_95 = np.percentile(returns, 5)
            cvar_95 = np.mean([r for r in returns if r <= var_95])
        else:
            sharpe_ratio = sortino_ratio = volatility = var_95 = cvar_95 = 0
        
        calmar_ratio = (self.equity_curve[-1] / self.equity_curve[0] - 1) / max_drawdown if max_drawdown > 0 else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Trades per day
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        days = (end_dt - start_dt).days
        trades_per_day = total_trades / days if days > 0 else 0
        
        # Monthly returns (simplified)
        monthly_returns = [0.1, 0.05, -0.02, 0.08, 0.12, -0.03]  # Placeholder
        
        return BacktestResult(
            start_date=start_date, end_date=end_date,
            total_trades=total_trades, winning_trades=winning_trades, losing_trades=losing_trades,
            win_rate=win_rate, total_pnl=total_pnl, total_fees=total_fees, net_pnl=net_pnl,
            max_drawdown=max_drawdown, sharpe_ratio=sharpe_ratio, sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio, profit_factor=profit_factor, avg_trade_pnl=avg_trade_pnl,
            avg_winning_trade=avg_winning_trade, avg_losing_trade=avg_losing_trade,
            largest_win=largest_win, largest_loss=largest_loss, consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses, max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses, trades_per_day=trades_per_day,
            volatility=volatility, var_95=var_95, cvar_95=cvar_95, monthly_returns=monthly_returns,
            equity_curve=self.equity_curve, drawdown_curve=self.drawdown_curve, trades=self.trades
        )
    
    def generate_report(self, result: BacktestResult, output_file: str = "backtest_report.html"):
        """Generate comprehensive HTML report"""
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Backtest Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                    .metric {{ margin: 10px 0; }}
                    .positive {{ color: green; }}
                    .negative {{ color: red; }}
                    .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                    table {{ width: 100%; border-collapse: collapse; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Backtest Report</h1>
                    <p>Period: {result.start_date} to {result.end_date}</p>
                </div>
                
                <div class="section">
                    <h2>Performance Summary</h2>
                    <div class="metric">Total Trades: {result.total_trades}</div>
                    <div class="metric">Win Rate: <span class="{'positive' if result.win_rate > 0.5 else 'negative'}">{result.win_rate:.2%}</span></div>
                    <div class="metric">Net PnL: <span class="{'positive' if result.net_pnl > 0 else 'negative'}">${result.net_pnl:.2f}</span></div>
                    <div class="metric">Total Fees: ${result.total_fees:.2f}</div>
                    <div class="metric">Max Drawdown: <span class="negative">{result.max_drawdown:.2%}</span></div>
                    <div class="metric">Sharpe Ratio: {result.sharpe_ratio:.2f}</div>
                    <div class="metric">Profit Factor: {result.profit_factor:.2f}</div>
                </div>
                
                <div class="section">
                    <h2>Trade Statistics</h2>
                    <div class="metric">Average Trade PnL: ${result.avg_trade_pnl:.2f}</div>
                    <div class="metric">Average Winning Trade: <span class="positive">${result.avg_winning_trade:.2f}</span></div>
                    <div class="metric">Average Losing Trade: <span class="negative">${result.avg_losing_trade:.2f}</span></div>
                    <div class="metric">Largest Win: <span class="positive">${result.largest_win:.2f}</span></div>
                    <div class="metric">Largest Loss: <span class="negative">${result.largest_loss:.2f}</span></div>
                    <div class="metric">Max Consecutive Wins: {result.max_consecutive_wins}</div>
                    <div class="metric">Max Consecutive Losses: {result.max_consecutive_losses}</div>
                </div>
                
                <div class="section">
                    <h2>Risk Metrics</h2>
                    <div class="metric">Volatility: {result.volatility:.2%}</div>
                    <div class="metric">VaR 95%: {result.var_95:.2%}</div>
                    <div class="metric">CVaR 95%: {result.cvar_95:.2%}</div>
                    <div class="metric">Sortino Ratio: {result.sortino_ratio:.2f}</div>
                    <div class="metric">Calmar Ratio: {result.calmar_ratio:.2f}</div>
                </div>
                
                <div class="section">
                    <h2>Recent Trades</h2>
                    <table>
                        <tr>
                            <th>Timestamp</th>
                            <th>Symbol</th>
                            <th>Action</th>
                            <th>Price</th>
                            <th>Quantity</th>
                            <th>PnL</th>
                            <th>AI Score</th>
                        </tr>
            """
            
            # Add recent trades
            for trade in result.trades[-20:]:  # Last 20 trades
                pnl_class = "positive" if trade.pnl > 0 else "negative" if trade.pnl < 0 else ""
                html_content += f"""
                        <tr>
                            <td>{trade.timestamp}</td>
                            <td>{trade.symbol}</td>
                            <td>{trade.action}</td>
                            <td>${trade.price:.4f}</td>
                            <td>{trade.quantity:.4f}</td>
                            <td class="{pnl_class}">${trade.pnl:.2f}</td>
                            <td>{trade.ai_score:.2f}</td>
                        </tr>
                """
            
            html_content += """
                    </table>
                </div>
            </body>
            </html>
            """
            
            # Write to file
            with open(output_file, 'w') as f:
                f.write(html_content)
            
            log_info(f"Backtest report generated: {output_file}")
            
        except Exception as e:
            log_error(f"Error generating report: {e}")

async def run_comprehensive_backtest(symbols: List[str], start_date: str, end_date: str, 
                                   strategy_params: Dict[str, Any] = None) -> BacktestResult:
    """Run a comprehensive backtest"""
    log_info(f"Starting comprehensive backtest for {len(symbols)} symbols from {start_date} to {end_date}")
    
    # Generate market data
    simulator = MarketDataSimulator(start_date, end_date, symbols)
    market_data = simulator.generate_market_data()
    
    # Initialize backtest engine
    engine = BacktestEngine()
    
    # Set default strategy parameters
    if strategy_params is None:
        strategy_params = {
            'score_threshold': 0.7,
            'confidence_threshold': 0.8,
            'max_position_pct': 0.1
        }
    
    # Run backtest
    result = engine.run_backtest(strategy_params, market_data, start_date, end_date)
    
    # Generate report
    engine.generate_report(result)
    
    return result

async def optimize_strategy(symbols: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
    """Optimize trading strategy using genetic algorithm"""
    log_info("Starting strategy optimization")
    
    # Generate market data
    simulator = MarketDataSimulator(start_date, end_date, symbols)
    market_data = simulator.generate_market_data()
    
    # Initialize optimizer and engine
    optimizer = StrategyOptimizer(population_size=30, generations=50)
    engine = BacktestEngine()
    
    # Define parameter ranges to optimize
    parameter_ranges = {
        'score_threshold': (0.5, 0.9),
        'confidence_threshold': (0.6, 0.95),
        'max_position_pct': (0.05, 0.2)
    }
    
    # Run optimization
    best_params = optimizer.optimize_strategy(engine, parameter_ranges)
    
    log_info(f"Strategy optimization complete. Best parameters: {best_params}")
    return best_params

if __name__ == "__main__":
    # Example usage
    symbols = ["BTC", "ETH", "ADA", "SOL", "MATIC"]
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    
    # Run backtest
    result = asyncio.run(run_comprehensive_backtest(symbols, start_date, end_date))
    print(f"Backtest complete: {result.total_trades} trades, {result.win_rate:.2%} win rate")
    
    # Optimize strategy
    best_params = asyncio.run(optimize_strategy(symbols, start_date, end_date))
    print(f"Best parameters: {best_params}")
