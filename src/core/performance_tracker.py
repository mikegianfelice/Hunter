#!/usr/bin/env python3
"""
Performance Tracking System for Sustainable Trading Bot
Tracks trades, analyzes performance by quality tiers, and provides insights
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import statistics
import os
import threading
import time

from src.storage.performance import (
    load_performance_data,
    replace_performance_data,
    set_json_path,
)


class PerformanceTracker:
    def __init__(self, data_file: str = "data/performance_data.json"):
        self.data_file = data_file
        self.trades = []
        self.daily_stats = {}
        self.quality_tiers = {
            "excellent": (80, 100),
            "high": (70, 79),
            "good": (60, 69),
            "average": (50, 59),
            "low": (0, 49)
        }

        # --- sync single-flight guards (prevent concurrent sync spawns) ---
        # Protects against multiple threads in this process starting syncs back-to-back.
        self._sync_spawn_lock = threading.Lock()
        self._last_sync_spawn_ts = 0.0
        # Debounce window to avoid spawning multiple syncs during bursts of trades.
        self._min_sync_spawn_interval_secs = 20.0

        self.load_data()
    
    def load_data(self):
        """Load existing performance data"""
        set_json_path(self.data_file)
        try:
            payload = load_performance_data()
            self.trades = payload.get('trades', [])
            self.daily_stats = payload.get('daily_stats', {})
        except Exception as e:
            print(f"⚠️ Could not load performance data: {e}")
            self.trades = []
            self.daily_stats = {}
    
    def save_data(self):
        """Persist performance data to storage"""
        try:
            data = {
                'trades': self.trades,
                'daily_stats': self.daily_stats,
                'last_updated': datetime.now().isoformat()
            }
            replace_performance_data(data)
            # Optionally sync to git for chart generation (non-blocking)
            self._maybe_sync_to_git()
        except Exception as e:
            print(f"⚠️ Could not save performance data: {e}")
    
    def _maybe_sync_to_git(self):
        """
        Optionally sync performance data to git for chart generation.

        This implementation enforces:
        - Single-flight per process: only one sync spawn at a time.
        - Debounce: avoid spawning multiple syncs during rapid trade bursts.
        - Respect existing on-disk locks (.sync_chart_data.lock, .git/index.lock).
        """
        try:
            # Check if auto-sync is enabled via environment variable
            auto_sync_env = os.getenv('AUTO_SYNC_CHART_DATA', '').lower() == 'true'
            
            # Check config if available
            auto_sync_config = False
            try:
                from src.config.config_loader import get_config_bool
                auto_sync_config = get_config_bool('auto_sync_chart_data', False)
            except Exception:
                # Config loader not available or failed – fall back to env only
                pass
            
            # Enable if either is set
            if not (auto_sync_env or auto_sync_config):
                return

            # Single-flight: if another thread is already spawning/running a sync, skip.
            if not self._sync_spawn_lock.acquire(blocking=False):
                return

            try:
                import subprocess
                from pathlib import Path

                now = time.time()

                # Debounce: don't spawn syncs more often than the configured interval.
                if (now - self._last_sync_spawn_ts) < self._min_sync_spawn_interval_secs:
                    return

                # Get project root directory (where script should run from)
                project_root = Path(__file__).parent.parent.parent
                script_path = project_root / 'scripts' / 'sync_chart_data.sh'
                
                if not script_path.exists():
                    print(f"⚠️ Sync script not found at {script_path}")
                    return

                sync_lock = project_root / '.sync_chart_data.lock'
                git_index_lock = project_root / '.git' / 'index.lock'

                # If a sync lock exists and is recent, assume another sync process is running.
                if sync_lock.exists():
                    try:
                        lock_age = now - sync_lock.stat().st_mtime
                        # Consider lock "active" for up to 10 minutes.
                        if lock_age < 600:
                            return
                    except Exception:
                        # If we can't stat the lock, be conservative and skip spawning.
                        return

                # If git index is locked and recent, don't spawn another git operation.
                if git_index_lock.exists():
                    try:
                        lock_age = now - git_index_lock.stat().st_mtime
                        if lock_age < 600:
                            return
                    except Exception:
                        return

                # Ensure script is executable
                os.chmod(script_path, 0o755)
                
                # Run in background, but log to file for debugging
                log_file = project_root / 'logs' / 'sync_chart_data.log'
                log_file.parent.mkdir(exist_ok=True)
                
                timestamp_str = datetime.now().isoformat()
                with open(log_file, 'a') as log:
                    log.write(f"\n[{timestamp_str}] Starting sync...\n")
                
                # Mark spawn time BEFORE launching to prevent double-spawn races
                self._last_sync_spawn_ts = now

                # Run script with proper working directory
                process = subprocess.Popen(
                    ['bash', str(script_path)],
                    cwd=str(project_root),  # Set working directory
                    stdout=open(log_file, 'a'),
                    stderr=subprocess.STDOUT,  # Redirect stderr to stdout
                    start_new_session=True  # Detach from parent process
                )
                
                # Log that we started the process
                with open(log_file, 'a') as log:
                    log.write(f"Sync process started with PID {process.pid}\n")
            finally:
                # Always release the lock, even if we early-returned or errored.
                self._sync_spawn_lock.release()
                
        except Exception as e:
            # Log error but don't interrupt trading
            try:
                from pathlib import Path
                project_root = Path(__file__).parent.parent.parent
                log_file = project_root / 'logs' / 'sync_chart_data.log'
                log_file.parent.mkdir(exist_ok=True)
                with open(log_file, 'a') as log:
                    ts = datetime.now().isoformat()
                    log.write(f"[{ts}] ERROR: {e}\n")
            except Exception:
                pass
            print(f"⚠️ Failed to start sync script: {e}")
    
    def log_trade_entry(self, token: Dict, position_size: float, quality_score: float, 
                       additional_data: Dict = None):
        """Log a trade entry with optional fee tracking data"""
        trade = {
            'id': f"{token.get('symbol', 'UNKNOWN')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'symbol': token.get('symbol', 'UNKNOWN'),
            'address': token.get('address', ''),
            'chain': token.get('chainId', 'ethereum'),
            'entry_time': datetime.now().isoformat(),
            'entry_price': float(token.get('priceUsd', 0)),
            'position_size_usd': position_size,
            'quality_score': quality_score,
            'volume_24h': float(token.get('volume24h', 0)),
            'liquidity': float(token.get('liquidity', 0)),
            'exit_time': None,
            'exit_price': None,
            'pnl_usd': None,
            'pnl_percent': None,
            'status': 'open',  # open, closed, stopped_out
            'take_profit_target': None,
            'stop_loss_target': None,
            # Fee tracking fields
            'buy_tx_hash': None,
            'entry_amount_usd_quoted': position_size,
            'entry_amount_usd_actual': None,
            'entry_tokens_received': None,
            'entry_gas_fee_usd': None,
            'buy_slippage_actual': None,
            # Window score tracking for entry timing analysis
            'window_score': None,
        }
        
        # Merge additional fee data if provided
        if additional_data:
            trade.update(additional_data)
        
        self.trades.append(trade)
        self.save_data()
        
        # After saving to performance_data, ensure it's synced to open_positions.json
        # ONLY if the trade actually succeeded (tokens were received)
        try:
            from src.utils.position_sync import sync_position_from_performance_data
            address = trade.get('address')
            symbol = trade.get('symbol', '?')
            chain = trade.get('chain', 'ethereum').lower()
            entry_price = float(trade.get('entry_price', 0))
            
            # Verify trade actually succeeded before syncing
            entry_amount = trade.get('entry_amount_usd_actual', 0) or 0
            tokens_received = trade.get('entry_tokens_received')
            
            if address and entry_price > 0:
                # Only sync if we have verified tokens received
                if entry_amount > 0 or (tokens_received is not None and tokens_received > 0):
                    sync_position_from_performance_data(address, symbol, chain, entry_price, position_size)
                else:
                    # Trade failed - mark as closed immediately
                    trade['status'] = 'manual_close'
                    trade['exit_time'] = trade.get('entry_time')
                    trade['exit_price'] = trade.get('entry_price', 0)
                    trade['pnl_usd'] = 0.0
                    trade['pnl_percent'] = 0.0
                    self.save_data()
                    print(f"⚠️ Trade {symbol} failed - no tokens received, marked as closed")
        except Exception as e:
            # Don't fail the whole operation if sync fails, but log it
            print(f"⚠️ Failed to sync position after logging: {e}")
        
        print(f"📊 Logged trade entry: {trade['symbol']} - ${position_size:.1f} (Quality: {quality_score:.1f})")
        return trade['id']
    
    def log_trade_exit(self, trade_id: str, exit_price: float, pnl_usd: float, status: str = 'closed',
                      additional_data: Dict = None):
        """Log a trade exit with optional fee tracking data"""
        for trade in self.trades:
            if trade['id'] == trade_id and trade['status'] == 'open':
                trade['exit_time'] = datetime.now().isoformat()
                trade['exit_price'] = exit_price
                trade['pnl_usd'] = pnl_usd
                trade['pnl_percent'] = (pnl_usd / trade['position_size_usd']) * 100
                trade['status'] = status
                
                # Initialize fee tracking fields if not present
                if 'exit_gas_fee_usd' not in trade:
                    trade['exit_gas_fee_usd'] = None
                if 'total_fees_usd' not in trade:
                    trade['total_fees_usd'] = None
                if 'pnl_after_fees_usd' not in trade:
                    trade['pnl_after_fees_usd'] = None
                if 'pnl_after_fees_percent' not in trade:
                    trade['pnl_after_fees_percent'] = None
                if 'sell_tx_hash' not in trade:
                    trade['sell_tx_hash'] = None
                if 'sell_slippage_actual' not in trade:
                    trade['sell_slippage_actual'] = None
                
                # Merge additional fee data if provided
                if additional_data:
                    trade.update(additional_data)
                    
                    # Calculate fee-adjusted PnL if we have fee data
                    if additional_data.get('total_fees_usd') is not None:
                        buy_gas = trade.get('entry_gas_fee_usd', 0) or 0
                        sell_gas = trade.get('exit_gas_fee_usd', 0) or 0
                        total_fees = buy_gas + sell_gas
                        
                        # Recalculate PnL after fees
                        entry_cost = trade.get('entry_amount_usd_actual') or trade['position_size_usd']
                        exit_proceeds = additional_data.get('actual_proceeds_usd')
                        
                        if exit_proceeds is not None:
                            trade['pnl_after_fees_usd'] = exit_proceeds - entry_cost - total_fees
                            trade['pnl_after_fees_percent'] = (trade['pnl_after_fees_usd'] / entry_cost) * 100 if entry_cost > 0 else 0
                        else:
                            # Fallback calculation using nominal PnL minus fees
                            trade['pnl_after_fees_usd'] = pnl_usd - total_fees
                            trade['pnl_after_fees_percent'] = (trade['pnl_after_fees_usd'] / trade['position_size_usd']) * 100
                
                # Update daily stats
                self._update_daily_stats(trade)
                self.save_data()
                
                # Print summary with fee info if available
                if trade.get('pnl_after_fees_usd') is not None:
                    print(f"📊 Logged trade exit: {trade['symbol']} - PnL: ${pnl_usd:.2f} ({trade['pnl_percent']:.1f}%), "
                          f"After Fees: ${trade['pnl_after_fees_usd']:.2f} ({trade['pnl_after_fees_percent']:.1f}%)")
                else:
                    print(f"📊 Logged trade exit: {trade['symbol']} - PnL: ${pnl_usd:.2f} ({trade['pnl_percent']:.1f}%)")
                return True
        
        print(f"⚠️ Could not find open trade with ID: {trade_id}")
        return False
    
    def _update_daily_stats(self, trade: Dict):
        """Update daily statistics"""
        date = trade['entry_time'][:10]  # YYYY-MM-DD
        
        if date not in self.daily_stats:
            self.daily_stats[date] = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0.0,
                'quality_tier_stats': defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0})
            }
        
        stats = self.daily_stats[date]
        stats['total_trades'] += 1
        stats['total_pnl'] += trade['pnl_usd']
        
        if trade['pnl_usd'] > 0:
            stats['winning_trades'] += 1
        else:
            stats['losing_trades'] += 1
        
        # Update quality tier stats
        quality_tier = self._get_quality_tier(trade['quality_score'])
        tier_stats = stats['quality_tier_stats'][quality_tier]
        tier_stats['trades'] += 1
        tier_stats['pnl'] += trade['pnl_usd']
        if trade['pnl_usd'] > 0:
            tier_stats['wins'] += 1
    
    def _get_quality_tier(self, quality_score: float) -> str:
        """Get quality tier based on score"""
        for tier, (min_score, max_score) in self.quality_tiers.items():
            if min_score <= quality_score <= max_score:
                return tier
        return 'low'
    
    def _is_failed_entry_attempt(self, trade: Dict) -> bool:
        """Check if a trade represents a failed entry attempt (not an actual completed trade).
        
        Failed entry attempts are identified by:
        - entry_amount_usd_actual is 0 or null
        - entry_tokens_received is null or 0
        - status is manual_close with exit_time very close to entry_time
        - exit_price equals entry_price (no actual trade occurred)
        """
        # Check if entry actually executed
        entry_amount = trade.get('entry_amount_usd_actual', 0) or 0
        tokens_received = trade.get('entry_tokens_received')
        
        # If no entry amount or no tokens received, it's a failed entry
        if entry_amount == 0 or (tokens_received is None or (isinstance(tokens_received, (int, float)) and tokens_received == 0)):
            # Double-check: if exit_time is very close to entry_time (within 1 second),
            # this confirms it was closed immediately after failed entry
            try:
                entry_time = datetime.fromisoformat(trade.get('entry_time', '').replace('Z', '+00:00'))
                exit_time_str = trade.get('exit_time')
                if exit_time_str:
                    exit_time = datetime.fromisoformat(exit_time_str.replace('Z', '+00:00'))
                    time_diff = abs((exit_time - entry_time).total_seconds())
                    
                    # If closed within 1 second and exit_price = entry_price, it's a failed entry
                    if time_diff < 2.0:
                        exit_price = trade.get('exit_price', 0)
                        entry_price = trade.get('entry_price', 0)
                        if exit_price == entry_price or exit_price == 0:
                            return True
            except:
                pass
            
            return True
        
        return False
    
    def get_performance_summary(self, days: int = 30) -> Dict:
        """Get performance summary for the last N days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_trades = [
            trade for trade in self.trades 
            if datetime.fromisoformat(trade['entry_time']) >= cutoff_date
        ]
        
        if not recent_trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_pnl': 0,
                'total_pnl': 0,
                'quality_analysis': {}
            }
        
        # Filter out failed entry attempts (only count actual completed trades)
        completed_trades = [t for t in recent_trades if not self._is_failed_entry_attempt(t)]
        failed_entry_attempts = [t for t in recent_trades if self._is_failed_entry_attempt(t)]
        
        # Overall stats (only for completed trades, excluding failed entry attempts)
        total_trades = len(completed_trades)
        winning_trades = len([t for t in completed_trades if t.get('pnl_usd') and t['pnl_usd'] > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        pnl_values = [t['pnl_usd'] for t in completed_trades if t.get('pnl_usd') is not None]
        avg_pnl = statistics.mean(pnl_values) if pnl_values else 0
        total_pnl = sum(pnl_values) if pnl_values else 0
        
        # Quality tier analysis (only for completed trades)
        quality_analysis = {}
        for tier in self.quality_tiers.keys():
            tier_trades = [t for t in completed_trades if self._get_quality_tier(t['quality_score']) == tier]
            if tier_trades:
                tier_wins = len([t for t in tier_trades if t.get('pnl_usd') and t['pnl_usd'] > 0])
                tier_pnl = [t['pnl_usd'] for t in tier_trades if t.get('pnl_usd') is not None]
                
                quality_analysis[tier] = {
                    'trades': len(tier_trades),
                    'win_rate': (tier_wins / len(tier_trades) * 100) if tier_trades else 0,
                    'avg_pnl': statistics.mean(tier_pnl) if tier_pnl else 0,
                    'total_pnl': sum(tier_pnl) if tier_pnl else 0
                }
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl,
            'quality_analysis': quality_analysis,
            'failed_entry_attempts': len(failed_entry_attempts),  # Track separately for visibility
            'period_days': days
        }
    
    def get_quality_vs_performance(self) -> Dict:
        """Analyze performance by quality tiers"""
        closed_trades = [t for t in self.trades if t['status'] == 'closed' and t['pnl_usd'] is not None]
        
        if not closed_trades:
            return {'message': 'No closed trades to analyze'}
        
        analysis = {}
        for tier, (min_score, max_score) in self.quality_tiers.items():
            tier_trades = [
                t for t in closed_trades 
                if min_score <= t['quality_score'] <= max_score
            ]
            
            if tier_trades:
                wins = len([t for t in tier_trades if t['pnl_usd'] > 0])
                pnl_values = [t['pnl_usd'] for t in tier_trades]
                
                analysis[tier] = {
                    'trades': len(tier_trades),
                    'win_rate': (wins / len(tier_trades) * 100) if tier_trades else 0,
                    'avg_pnl': statistics.mean(pnl_values),
                    'total_pnl': sum(pnl_values),
                    'best_trade': max(pnl_values),
                    'worst_trade': min(pnl_values)
                }
        
        return analysis
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """Get recent trades"""
        return sorted(self.trades, key=lambda x: x['entry_time'], reverse=True)[:limit]
    
    def get_open_trades(self, validate_balances: bool = False) -> List[Dict]:
        """
        Get currently open trades.
        If validate_balances is True, only returns trades that have on-chain wallet balances.
        This prevents manually closed positions from appearing as open.
        """
        open_trades = [t for t in self.trades if t['status'] == 'open']
        
        if not validate_balances:
            return open_trades
        
        # Validate balances for each open trade
        validated_trades = []
        for trade in open_trades:
            address = trade.get('address', '')
            chain = trade.get('chain', 'ethereum').lower()
            
            if not address:
                continue
            
            # Check wallet balance
            balance = self._check_token_balance_on_chain(address, chain)
            
            if balance == -1.0:
                # Balance check failed - keep trade to be safe (don't lose track of it)
                validated_trades.append(trade)
            elif balance <= 0.0 or balance < 0.000001:
                # Zero or dust balance - position was manually closed
                # Mark as closed in performance_data
                trade['status'] = 'manual_close'
                trade['exit_time'] = datetime.now().isoformat()
                trade['exit_price'] = 0.0
                trade['pnl_usd'] = 0.0
                trade['pnl_percent'] = 0.0
                # Don't add to validated_trades - it's closed
                print(f"📝 Marked trade {trade.get('symbol', '?')} as manually closed (zero balance)")
            else:
                # Has balance - trade is still open
                validated_trades.append(trade)
        
        # Save updated performance_data if any trades were marked as closed
        if len(validated_trades) < len(open_trades):
            self.save_data()
        
        return validated_trades
    
    def _check_token_balance_on_chain(self, token_address: str, chain_id: str) -> float:
        """
        Check token balance on the specified chain.
        Returns balance amount (0.0 if balance is zero, -1.0 if check failed).
        """
        try:
            chain_lower = chain_id.lower()
            
            if chain_lower == "solana":
                from src.execution.jupiter_lib import JupiterCustomLib
                from src.config.secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY
                lib = JupiterCustomLib(SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY)
                balance = lib.get_token_balance(token_address)
                # None means check failed (error) - return -1.0 to indicate unknown state
                if balance is None:
                    return -1.0
                return float(balance)
                
            elif chain_lower == "base":
                from src.execution.base_executor import get_token_balance
                balance = get_token_balance(token_address)
                return float(balance or 0.0)
                
            elif chain_lower == "ethereum":
                # Use web3 to check ERC20 token balance
                from web3 import Web3
                from src.config.secrets import INFURA_URL, WALLET_ADDRESS
                
                w3 = Web3(Web3.HTTPProvider(INFURA_URL))
                if not w3.is_connected():
                    return -1.0
                
                # ERC20 ABI minimal - just balanceOf and decimals
                erc20_abi = [
                    {"constant":True,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"type":"function"},
                    {"constant":True,"inputs":[{"name":"owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"type":"function"}
                ]
                
                wallet = Web3.to_checksum_address(WALLET_ADDRESS)
                token_addr = Web3.to_checksum_address(token_address)
                token_contract = w3.eth.contract(address=token_addr, abi=erc20_abi)
                
                balance_wei = token_contract.functions.balanceOf(wallet).call()
                decimals = token_contract.functions.decimals().call()
                balance = float(balance_wei) / (10 ** decimals)
                return balance
                
            else:
                # For other chains, return -1.0 to indicate check not implemented
                return -1.0
                
        except Exception as e:
            # If balance check fails, return -1.0 to indicate unknown state
            return -1.0
    
    def validate_and_update_open_trades(self) -> int:
        """
        Validate all open trades by checking wallet balances.
        Marks trades with zero balance as manually closed.
        Returns the number of trades that were marked as closed.
        """
        open_trades = self.get_open_trades(validate_balances=False)
        closed_count = 0
        
        for trade in open_trades:
            address = trade.get('address', '')
            chain = trade.get('chain', 'ethereum').lower()
            
            if not address:
                continue
            
            # Check wallet balance
            balance = self._check_token_balance_on_chain(address, chain)
            
            if balance == -1.0:
                # Balance check failed - skip to be safe
                continue
            elif balance <= 0.0 or balance < 0.000001:
                # Zero or dust balance - position was manually closed
                # Mark as closed in performance_data
                trade['status'] = 'manual_close'
                trade['exit_time'] = datetime.now().isoformat()
                trade['exit_price'] = 0.0
                trade['pnl_usd'] = 0.0
                trade['pnl_percent'] = 0.0
                closed_count += 1
                print(f"📝 Marked trade {trade.get('symbol', '?')} ({address[:8]}...{address[-8:]}) as manually closed (zero balance)")
        
        if closed_count > 0:
            self.save_data()
            print(f"✅ Validated open trades: {closed_count} trade(s) marked as manually closed")
        
        return closed_count
    
    def get_trade_history(self, limit: int = None) -> List[Dict]:
        """Get trade history (all trades or limited by count)"""
        if limit is None:
            return self.trades
        return sorted(self.trades, key=lambda x: x['entry_time'], reverse=True)[:limit]
    
    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report"""
        summary = self.get_performance_summary(30)
        quality_analysis = self.get_quality_vs_performance()
        
        report = f"""
📊 SUSTAINABLE TRADING PERFORMANCE REPORT
{'='*50}

🎯 OVERALL PERFORMANCE (Last 30 Days):
• Total Trades: {summary['total_trades']}
• Win Rate: {summary['win_rate']:.1f}%
• Average PnL: ${summary['avg_pnl']:.2f}
• Total PnL: ${summary['total_pnl']:.2f}

📈 QUALITY TIER ANALYSIS:
"""
        
        for tier, stats in quality_analysis.items():
            if stats['trades'] > 0:
                report += f"• {tier.upper()} Quality ({stats['trades']} trades):\n"
                report += f"  - Win Rate: {stats['win_rate']:.1f}%\n"
                report += f"  - Avg PnL: ${stats['avg_pnl']:.2f}\n"
                report += f"  - Total PnL: ${stats['total_pnl']:.2f}\n"
                report += f"  - Best: ${stats['best_trade']:.2f}, Worst: ${stats['worst_trade']:.2f}\n\n"
        
        # Recent trades
        recent = self.get_recent_trades(5)
        if recent:
            report += "🔄 RECENT TRADES:\n"
            for trade in recent:
                status_emoji = "✅" if trade['pnl_usd'] and trade['pnl_usd'] > 0 else "❌" if trade['pnl_usd'] else "⏳"
                pnl_str = f"${trade['pnl_usd']:.2f}" if trade['pnl_usd'] else "Open"
                report += f"• {status_emoji} {trade['symbol']} - {pnl_str} (Quality: {trade['quality_score']:.1f})\n"
        
        return report
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Remove old trade data to keep file size manageable"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Keep recent trades
        self.trades = [
            trade for trade in self.trades 
            if datetime.fromisoformat(trade['entry_time']) >= cutoff_date
        ]
        
        # Clean up daily stats
        cutoff_str = cutoff_date.strftime('%Y-%m-%d')
        self.daily_stats = {
            date: stats for date, stats in self.daily_stats.items() 
            if date >= cutoff_str
        }
        
        self.save_data()
        print(f"🧹 Cleaned up data older than {days_to_keep} days")
    
    def validate_and_fix_trades(self, min_balance_threshold: float = 0.000001) -> Dict:
        """
        Validate all trades and fix inconsistencies.
        Marks trades with zero balance as closed if they have 'open' status.
        
        Args:
            min_balance_threshold: Minimum balance to consider as a valid position
        
        Returns:
            Dict with validation results: {"fixed": int, "errors": List[str]}
        """
        results = {
            "fixed": 0,
            "errors": []
        }
        
        try:
            open_trades = [t for t in self.trades if t.get('status') == 'open']
            fixed_count = 0
            
            for trade in open_trades:
                address = trade.get('address', '')
                chain = trade.get('chain', 'ethereum').lower()
                
                if not address:
                    continue
                
                # Check wallet balance
                balance = self._check_token_balance_on_chain(address, chain)
                
                if balance == -1.0:
                    # Balance check failed - skip to be safe
                    continue
                elif balance <= 0.0 or balance < min_balance_threshold:
                    # Zero or dust balance - mark as closed
                    trade['status'] = 'manual_close'
                    trade['exit_time'] = datetime.now().isoformat()
                    trade['exit_price'] = 0.0
                    trade['pnl_usd'] = 0.0
                    trade['pnl_percent'] = 0.0
                    fixed_count += 1
                    print(f"📝 Fixed trade: {trade.get('symbol', '?')} ({address[:8]}...{address[-8:]}) - marked as closed (zero balance)")
            
            if fixed_count > 0:
                self.save_data()
                results["fixed"] = fixed_count
                print(f"✅ Validated trades: {fixed_count} trade(s) marked as closed")
        
        except Exception as e:
            error_msg = f"Error in validate_and_fix_trades: {e}"
            print(f"⚠️ {error_msg}")
            results["errors"].append(error_msg)
        
        return results

# Global instance
performance_tracker = PerformanceTracker()
