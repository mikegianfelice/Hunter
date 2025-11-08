#!/usr/bin/env python3
"""Test script to send a test Telegram status report"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.monitoring.telegram_bot import format_status_message, send_telegram_message

# Create mock data for testing
mock_risk_summary = {
    'buys_today': 0,
    'sells_today': 0,
    'realized_pnl_usd': 0,
    'losing_streak': 0,
    'open_positions': 2,
    'paused_until': 0
}

mock_recent_summary = {
    'total_trades': 0,
    'win_rate': 0,
    'total_pnl': 0,
    'avg_pnl': 0
}

mock_open_trades = []

mock_market_regime = {
    'regime': 'bull_market',
    'confidence': 0.85,
    'description': 'Strong bullish momentum detected with increasing volume and positive sentiment',
    'strategy': 'aggressive',
    'recommendations': [
        'Focus on high-quality tokens with strong fundamentals',
        'Consider increasing position sizes slightly',
        'Monitor for potential profit-taking opportunities'
    ]
}

# Format the test message
test_message = format_status_message(
    risk_summary=mock_risk_summary,
    recent_summary=mock_recent_summary,
    open_trades=mock_open_trades,
    market_regime=mock_market_regime
)

print("=" * 50)
print("TEST MESSAGE TO BE SENT:")
print("=" * 50)
print(test_message)
print("=" * 50)

# Send the message
print("\nSending test message to Telegram...")
result = send_telegram_message(
    test_message,
    markdown=True,
    deduplicate=False,
    message_type="test"
)

if result:
    print("✅ Test message sent successfully!")
else:
    print("❌ Failed to send test message. Check your Telegram configuration.")

