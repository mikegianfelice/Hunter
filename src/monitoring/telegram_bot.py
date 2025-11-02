# telegram_bot.py
import requests
import time
import hashlib
import sys
import os
# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config.secrets import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
import threading
from datetime import datetime

# Message deduplication cache
_sent_messages = {}
_message_cache_ttl = 300  # 5 minutes

# Rate limiting for frequent messages
_rate_limits = {}
_rate_limit_window = 60  # 1 minute
_max_messages_per_window = 5  # Max 5 messages per minute per type

# Periodic status tracking
_last_status_time = 0
_status_interval = 6 * 60 * 60  # 6 hours in seconds

def _cleanup_old_messages():
    """Remove old messages from the cache"""
    current_time = time.time()
    global _sent_messages
    _sent_messages = {
        msg: timestamp for msg, timestamp in _sent_messages.items()
        if current_time - timestamp < _message_cache_ttl
    }

def _get_message_fingerprint(message: str) -> str:
    """Create a fingerprint for message deduplication that ignores minor variations"""
    # Normalize message by removing timestamps, addresses, and other variable data
    normalized = message.lower()
    # Remove common variable patterns
    import re
    normalized = re.sub(r'0x[a-fA-F0-9]{8,}', 'ADDRESS', normalized)
    normalized = re.sub(r'\$[\d,]+\.?\d*', 'AMOUNT', normalized)
    normalized = re.sub(r'\d+\.\d+%', 'PERCENT', normalized)
    normalized = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', 'TIMESTAMP', normalized)
    normalized = re.sub(r'\d+\.\d+', 'NUMBER', normalized)
    return hashlib.md5(normalized.encode()).hexdigest()

def _check_rate_limit(message_type: str) -> bool:
    """Check if we're within rate limits for this message type"""
    current_time = time.time()
    global _rate_limits
    
    # Clean old entries
    _rate_limits = {
        msg_type: timestamps for msg_type, timestamps in _rate_limits.items()
        if any(t > current_time - _rate_limit_window for t in timestamps)
    }
    
    # Check current rate
    if message_type not in _rate_limits:
        _rate_limits[message_type] = []
    
    recent_messages = [t for t in _rate_limits[message_type] if t > current_time - _rate_limit_window]
    
    if len(recent_messages) >= _max_messages_per_window:
        return False
    
    # Add current message
    _rate_limits[message_type].append(current_time)
    return True

def send_telegram_message(message: str, markdown: bool = False, disable_preview: bool = True, deduplicate: bool = True, message_type: str = "general"):
    """
    Send a Telegram message using secrets loaded from .env (via secrets.py).
    - message: text to send
    - markdown: enable Telegram Markdown parsing
    - disable_preview: disable link previews
    - deduplicate: prevent sending the same message multiple times within 5 minutes
    - message_type: category for rate limiting (e.g., "error", "trade", "status")
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram not configured (missing TELEGRAM_BOT_TOKEN/CHAT_ID).")
        return False

    # Check rate limiting
    if not _check_rate_limit(message_type):
        print(f"📨 Rate limited: {message_type} messages (max {_max_messages_per_window}/min)")
        return False

    # Deduplication logic
    if deduplicate:
        _cleanup_old_messages()
        current_time = time.time()
        
        # Check if this message type was sent recently (using fingerprint)
        fingerprint = _get_message_fingerprint(message)
        if fingerprint in _sent_messages:
            print(f"📨 Skipping duplicate Telegram message: {message_type}")
            return True  # Return True since we "handled" it
        
        # Add to cache
        _sent_messages[fingerprint] = current_time

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "disable_web_page_preview": disable_preview,
    }
    if markdown:
        payload["parse_mode"] = "Markdown"

    try:
        resp = requests.post(url, json=payload, timeout=12)
        if resp.status_code != 200:
            # Fallback: if Markdown parsing fails, retry without formatting
            if (
                resp.status_code == 400
                and "can't parse entities" in resp.text.lower()
                and payload.get("parse_mode")
            ):
                print(f"⚠️ Telegram parse error detected, retrying without formatting...")
                payload_retry = {k: v for k, v in payload.items() if k != "parse_mode"}
                resp2 = requests.post(url, json=payload_retry, timeout=12)
                if resp2.status_code == 200:
                    print("📨 Telegram alert sent without formatting (parse error fallback).")
                    return True
                else:
                    print(f"❌ Telegram failed even without formatting: {resp2.text}")
                    return False
            
            print(f"❌ Telegram failed: {resp.text}")
            return False
        print("📨 Telegram alert sent!")
        return True
    except requests.RequestException as e:
        if hasattr(e, 'errno') and e.errno == 32:  # Broken pipe
            print(f"⚠️ Telegram broken pipe error (connection closed): {e}")
        else:
            print(f"❌ Telegram request error: {e}")
        return False
    except OSError as e:
        if e.errno == 32:  # Broken pipe
            print(f"⚠️ Telegram broken pipe error (OS): {e}")
        else:
            print(f"❌ Telegram OS error: {e}")
        return False
    except Exception as e:
        print(f"❌ Telegram unexpected error: {e}")
        return False

def send_periodic_status_report():
    """
    Send a comprehensive status report to Telegram every 6 hours.
    Includes bot status, buy/sell summary, and market conditions.
    """
    global _last_status_time
    current_time = time.time()
    
    # Check if it's time to send a status report (every 6 hours)
    if current_time - _last_status_time < _status_interval:
        return False
    
    # Update last status time before sending to prevent duplicate sends
    _last_status_time = current_time
    
    try:
        # Import here to avoid circular imports and use correct module paths
        from src.core.performance_tracker import performance_tracker
        from src.core.risk_manager import status_summary, get_tier_based_risk_limits
        from src.ai.ai_market_regime_detector import ai_market_regime_detector
        
        # Get bot status
        risk_summary = status_summary()
        
        # Get recent performance data
        recent_summary = performance_tracker.get_performance_summary(7)  # Last 7 days
        open_trades = performance_tracker.get_open_trades()
        
        # Get current market regime
        market_regime = ai_market_regime_detector.detect_market_regime()
        
        # Format the status message
        status_msg = format_status_message(risk_summary, recent_summary, open_trades, market_regime)
        
        # Send the message
        return send_telegram_message(status_msg, markdown=True, deduplicate=False)
    except Exception as e:
        print(f"⚠️ Could not send periodic status report: {e}")
        return False

def format_status_message(risk_summary, recent_summary, open_trades, market_regime=None):
    """Format a comprehensive status message"""
    from datetime import datetime
    import json
    import os
    
    # Helper to fetch current token price by chain/address
    def _current_price(chain: str, address: str) -> float:
        try:
            ch = (chain or 'ethereum').lower()
            if ch == 'solana':
                # Use Solana executor price fetcher
                from src.execution.solana_executor import get_token_price_usd
                px = get_token_price_usd(address)
                return float(px or 0.0)
            else:
                # EVM and others fallback to utils fetcher (Uniswap subgraph etc.)
                from src.utils.utils import fetch_token_price_usd
                px = fetch_token_price_usd(address)
                return float(px or 0.0)
        except Exception:
            return 0.0
    
    # Helper to load and calculate unrealized PnL from open_positions.json
    def _get_open_positions_pnl():
        """Load open positions from file and calculate unrealized PnL"""
        positions_file = "data/open_positions.json"
        if not os.path.exists(positions_file):
            return []
        
        try:
            with open(positions_file, 'r') as f:
                positions = json.load(f) or {}
        except Exception:
            return []
        
        pnl_lines = []
        for token_address, position_data in positions.items():
            try:
                if isinstance(position_data, dict):
                    entry_price = float(position_data.get("entry_price", 0))
                    chain_id = position_data.get("chain_id", "ethereum").lower()
                    symbol = position_data.get("symbol", "?")
                else:
                    # Legacy format
                    entry_price = float(position_data)
                    chain_id = "ethereum"
                    symbol = "?"
                
                if entry_price <= 0:
                    continue
                
                # Fetch current price
                current_price = _current_price(chain_id, token_address)
                
                if current_price > 0:
                    # Calculate PnL percentage
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    # Format prices with appropriate precision (5-6 decimals for small prices)
                    entry_str = f"{entry_price:.5f}".rstrip('0').rstrip('.')
                    current_str = f"{current_price:.6f}".rstrip('0').rstrip('.')
                    # Format as requested: SYMBOL: entry $X → current $Y → PnL Z%
                    pnl_lines.append(
                        f"{symbol}: entry ${entry_str} → current ${current_str} → PnL {pnl_pct:+.2f}%"
                    )
                else:
                    # If we can't get current price, still show the position
                    pnl_lines.append(
                        f"{symbol}: entry ${entry_price:.5f} → current N/A"
                    )
            except Exception as e:
                # Skip positions that fail to process
                continue
        
        return pnl_lines
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate total exposure from open trades
    total_exposure = sum(t.get('position_size_usd', 0) for t in open_trades)
    open_count = len(open_trades)
    
    # Compute unrealized PnL for open trades
    unrealized_total_usd = 0.0
    unrealized_lines = []
    if open_trades:
        for i, trade in enumerate(open_trades[:5], 1):
            try:
                symbol = trade.get('symbol', 'UNKNOWN')
                size_usd = float(trade.get('position_size_usd', 0) or 0)
                chain = trade.get('chain', trade.get('chainId', 'unknown'))
                address = trade.get('address') or ''
                entry = float(trade.get('entry_price', trade.get('priceUsd', 0)) or 0)
                price = _current_price(chain, address) if address else 0.0
                if entry > 0 and price > 0 and size_usd > 0:
                    pct = (price - entry) / entry
                    pnl_usd = pct * size_usd
                    unrealized_total_usd += pnl_usd
                    unrealized_lines.append(
                        f"  {i}. {symbol} [{chain}]: ${size_usd:.2f} | Entry ${entry:.6f} → Now ${price:.6f} | Unrealized: ${pnl_usd:.2f} ({pct*100:.1f}%)\n"
                    )
                else:
                    unrealized_lines.append(
                        f"  {i}. {symbol} [{chain}]: ${size_usd:.2f}\n"
                    )
            except Exception:
                # Fallback to basic line if anything fails
                symbol = trade.get('symbol', 'UNKNOWN')
                size_usd = trade.get('position_size_usd', 0)
                chain = trade.get('chain', trade.get('chainId', 'unknown'))
                unrealized_lines.append(f"  {i}. {symbol} [{chain}]: ${size_usd:.2f}\n")
    
    # Get tier information
    try:
        from src.core.risk_manager import get_tier_based_risk_limits
        tier_limits = get_tier_based_risk_limits()
        tier_name = tier_limits.get('TIER_NAME', 'unknown')
    except Exception:
        tier_name = 'unknown'
    
    msg = f"""🤖 *Bot Status Report* - {current_time}
━━━━━━━━━━━━━━━━━━━━━━━

✅ *Bot Status:* Online & Operational

📊 *Recent Activity (Last 7 Days):*
• Total Trades: {recent_summary.get('total_trades', 0)}
• Win Rate: {recent_summary.get('win_rate', 0):.1f}%
• Total PnL: ${recent_summary.get('total_pnl', 0):.2f}
• Avg PnL: ${recent_summary.get('avg_pnl', 0):.2f}

💼 *Current Positions:*
• Open Trades: {open_count}
• Approx. Exposure: ${total_exposure:.2f}
• Tier: {tier_name}
"""
    
    # Add open positions with unrealized PnL from open_positions.json
    open_positions_pnl = _get_open_positions_pnl()
    if open_positions_pnl:
        msg += "\n📈 *Open Positions Unrealized PnL:*\n"
        for pnl_line in open_positions_pnl:
            msg += f"{pnl_line}\n"
    elif open_trades:
        # Fallback to open_trades if available
        if unrealized_lines:
            for line in unrealized_lines:
                msg += line
        else:
            for i, trade in enumerate(open_trades[:5], 1):
                symbol = trade.get('symbol', 'UNKNOWN')
                size = trade.get('position_size_usd', 0)
                chain = trade.get('chain', trade.get('chainId', 'unknown'))
                msg += f"  {i}. {symbol} [{chain}]: ${size:.2f}\n"
        # Add total unrealized PnL summary line
        if unrealized_total_usd != 0:
            msg += f"• Unrealized PnL (est.): ${unrealized_total_usd:.2f}\n"
    else:
        msg += "  No open positions\n"
    
    # Add risk management info
    msg += f"""
🛡️ *Risk Management:*
• Daily Buys: {risk_summary.get('buys_today', 0)}
• Daily Sells: {risk_summary.get('sells_today', 0)}
• Realized PnL: ${risk_summary.get('realized_pnl_usd', 0):.2f}
• Losing Streak: {risk_summary.get('losing_streak', 0)}
• Open Positions: {risk_summary.get('open_positions', open_count)}
"""
    
    # Check if paused
    paused_until = risk_summary.get('paused_until', 0)
    if paused_until > time.time():
        pause_mins = int((paused_until - time.time()) / 60)
        msg += f"⏸️ *Status:* Paused for {pause_mins} more minutes\n"
    else:
        msg += "▶️ *Status:* Active Trading\n"
    
    # Add market conditions with regime information
    if market_regime:
        regime = market_regime.get('regime', 'unknown')
        confidence = market_regime.get('confidence', 0) * 100.0  # Convert to percentage
        description = market_regime.get('description', 'Unknown market condition')
        strategy = market_regime.get('strategy', 'neutral')
        recommendations = market_regime.get('recommendations', [])[:3]  # Show top 3
        
        # Format regime display with emoji
        regime_emoji = {
            'bull_market': '🐂',
            'bear_market': '🐻', 
            'sideways_market': '↔️',
            'high_volatility': '⚡',
            'recovery_market': '🔄'
        }.get(regime, '📊')
        
        msg += f"""
📈 *Market Conditions:*
{regime_emoji} *Regime:* {regime.replace('_', ' ').title()}
📊 *Confidence:* {confidence:.1f}%
💡 *Strategy:* {strategy.title()}
📝 *Description:* {description}
"""
        if recommendations:
            msg += "✅ *Focus Now:*\n"
            for r in recommendations:
                msg += f"• {r}\n"
    else:
        msg += """
📈 *Market Conditions:*
• Bot is monitoring opportunities
• Following sustainable trading strategy
• Target: 10-20% consistent gains
"""
    
    return msg