#!/bin/bash

# Bot Monitoring Script
# Run this script to check bot status every few minutes

echo "🤖 Crypto Trading Bot Monitor"
echo "=============================="
echo "Started at: $(date)"
echo ""

while true; do
    echo "📊 Status Check at $(date)"
    echo "------------------------"
    
    # Check if bot process is running
    if pgrep -f "python.*main.py" > /dev/null; then
        echo "✅ Bot is RUNNING"
    else
        echo "❌ Bot is NOT RUNNING"
        echo "💡 Start with: python main.py"
    fi
    
    # Check open positions
    echo ""
    echo "📈 Open Positions:"
    if [ -f "open_positions.json" ]; then
        POSITIONS=$(cat open_positions.json | grep -o '"[^"]*"' | wc -l)
        if [ "$POSITIONS" -eq 0 ]; then
            echo "   No open positions"
        else
            echo "   $POSITIONS position(s) open"
            cat open_positions.json | grep -o '"[^"]*"' | head -5
        fi
    else
        echo "   No positions file found"
    fi
    
    # Check recent trades
    echo ""
    echo "💰 Recent Trades:"
    if [ -f "trade_log.csv" ]; then
        RECENT_TRADES=$(tail -5 trade_log.csv)
        if [ -z "$RECENT_TRADES" ]; then
            echo "   No recent trades"
        else
            echo "$RECENT_TRADES"
        fi
    else
        echo "   No trade log found"
    fi
    
    # Check risk state
    echo ""
    echo "🛡️ Risk Status:"
    if [ -f "risk_state.json" ]; then
        cat risk_state.json | grep -E '"realized_pnl_usd"|"buys_today"|"sells_today"|"losing_streak"'
    fi
    
    echo ""
    echo "⏰ Next check in 5 minutes..."
    echo "=============================="
    sleep 300  # Wait 5 minutes
done
