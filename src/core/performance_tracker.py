#!/usr/bin/env python3
"""
Performance Tracking System for Sustainable Trading Bot
Tracks trades, analyzes performance by quality tiers, and provides insights
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import statistics

class PerformanceTracker:
    def __init__(self, data_file: str = "performance_data.json"):
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
        self.load_data()
    
    def load_data(self):
        """Load existing performance data"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.trades = data.get('trades', [])
                    self.daily_stats = data.get('daily_stats', {})
            except Exception as e:
                print(f"⚠️ Could not load performance data: {e}")
                self.trades = []
                self.daily_stats = {}
    
    def save_data(self):
        """Save performance data to file"""
        try:
            data = {
                'trades': self.trades,
                'daily_stats': self.daily_stats,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save performance data: {e}")
    
    def log_trade_entry(self, token: Dict, position_size: float, quality_score: float):
        """Log a trade entry"""
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
            'stop_loss_target': None
        }
        
        self.trades.append(trade)
        self.save_data()
        
        print(f"📊 Logged trade entry: {trade['symbol']} - ${position_size:.1f} (Quality: {quality_score:.1f})")
        return trade['id']
    
    def log_trade_exit(self, trade_id: str, exit_price: float, pnl_usd: float, status: str = 'closed'):
        """Log a trade exit"""
        for trade in self.trades:
            if trade['id'] == trade_id and trade['status'] == 'open':
                trade['exit_time'] = datetime.now().isoformat()
                trade['exit_price'] = exit_price
                trade['pnl_usd'] = pnl_usd
                trade['pnl_percent'] = (pnl_usd / trade['position_size_usd']) * 100
                trade['status'] = status
                
                # Update daily stats
                self._update_daily_stats(trade)
                self.save_data()
                
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
        
        # Overall stats
        total_trades = len(recent_trades)
        winning_trades = len([t for t in recent_trades if t['pnl_usd'] and t['pnl_usd'] > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        pnl_values = [t['pnl_usd'] for t in recent_trades if t['pnl_usd'] is not None]
        avg_pnl = statistics.mean(pnl_values) if pnl_values else 0
        total_pnl = sum(pnl_values) if pnl_values else 0
        
        # Quality tier analysis
        quality_analysis = {}
        for tier in self.quality_tiers.keys():
            tier_trades = [t for t in recent_trades if self._get_quality_tier(t['quality_score']) == tier]
            if tier_trades:
                tier_wins = len([t for t in tier_trades if t['pnl_usd'] and t['pnl_usd'] > 0])
                tier_pnl = [t['pnl_usd'] for t in tier_trades if t['pnl_usd'] is not None]
                
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
    
    def get_open_trades(self) -> List[Dict]:
        """Get currently open trades"""
        return [t for t in self.trades if t['status'] == 'open']
    
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

# Global instance
performance_tracker = PerformanceTracker()
