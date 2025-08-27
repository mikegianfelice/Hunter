#!/usr/bin/env python3
"""
Quick Bot Status Checker
Run this to get a snapshot of bot status
"""

import json
import os
import csv
from datetime import datetime
import subprocess

def check_bot_status():
    print("🤖 Crypto Trading Bot Status Check")
    print("=" * 50)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if bot is running
    try:
        result = subprocess.run(['pgrep', '-f', 'python.*main.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Bot Status: RUNNING")
        else:
            print("❌ Bot Status: NOT RUNNING")
    except:
        print("❓ Bot Status: UNKNOWN")
    
    print()
    
    # Check open positions
    print("📈 Open Positions:")
    if os.path.exists("open_positions.json"):
        with open("open_positions.json", "r") as f:
            positions = json.load(f)
        if positions:
            print(f"   {len(positions)} position(s) open:")
            for addr, price in positions.items():
                print(f"   • {addr[:10]}...{addr[-10:]} @ ${price}")
        else:
            print("   No open positions")
    else:
        print("   No positions file found")
    
    print()
    
    # Check recent trades
    print("💰 Recent Trades:")
    if os.path.exists("trade_log.csv"):
        with open("trade_log.csv", "r") as f:
            reader = csv.DictReader(f)
            trades = list(reader)
        if trades:
            print(f"   Last 3 trades:")
            for trade in trades[-3:]:
                pnl = float(trade.get('pnl_pct', 0))
                status = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
                print(f"   {status} {trade.get('token', 'Unknown')[:10]}... "
                      f"PnL: {pnl:.1f}% ({trade.get('reason', 'Unknown')})")
        else:
            print("   No trades recorded")
    else:
        print("   No trade log found")
    
    print()
    
    # Check risk state
    print("🛡️ Risk Status:")
    if os.path.exists("risk_state.json"):
        with open("risk_state.json", "r") as f:
            risk = json.load(f)
        print(f"   PnL Today: ${risk.get('realized_pnl_usd', 0):.2f}")
        print(f"   Buys Today: {risk.get('buys_today', 0)}")
        print(f"   Sells Today: {risk.get('sells_today', 0)}")
        print(f"   Losing Streak: {risk.get('losing_streak', 0)}")
        if risk.get('paused_until', 0) > 0:
            print(f"   ⚠️ PAUSED until {datetime.fromtimestamp(risk['paused_until'])}")
    else:
        print("   No risk state file found")
    
    print()
    print("=" * 50)

if __name__ == "__main__":
    check_bot_status()
