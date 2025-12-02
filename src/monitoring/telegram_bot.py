# telegram_bot.py
import requests
import time
import hashlib
import sys
import os
import atexit
from queue import Queue, Empty
# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config.secrets import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from src.monitoring.structured_logger import log_info, log_warning, log_error
import threading
from datetime import datetime
from src.storage.positions import load_positions as load_positions_store

_SESSION = requests.Session()

class TelegramDispatcher:
    def __init__(self, max_retries: int = 2, base_backoff: float = 2.0, worker_name: str = "telegram-dispatcher"):
        self._queue: Queue = Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._max_retries = max_retries
        self._base_backoff = base_backoff
        self._worker_name = worker_name

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, name=self._worker_name, daemon=True)
        self._thread.start()

    def stop(self, drain: bool = False, timeout: float = 5.0):
        self._stop_event.set()
        if drain:
            self._queue.join()
        if self._thread:
            self._thread.join(timeout=timeout)

    def submit(self, job: dict):
        self._queue.put(job)

    def _worker(self):
        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=0.5)
            except Empty:
                continue

            success = _send_message_sync(job)

            if success:
                if job.get("deduplicate") and job.get("fingerprint"):
                    _sent_messages[job["fingerprint"]] = time.time()
                _pending_fingerprints.discard(job.get("fingerprint"))
                self._queue.task_done()
                continue

            attempt = job.get("attempt", 0) + 1
            if attempt > self._max_retries or self._stop_event.is_set():
                log_error(
                    "telegram.async.failed",
                    "Telegram async send failed after retries",
                    context={"message_type": job.get("message_type"), "attempts": attempt},
                )
                _pending_fingerprints.discard(job.get("fingerprint"))
                self._queue.task_done()
                continue

            job["attempt"] = attempt
            delay = min(self._base_backoff ** attempt, 30)
            time.sleep(delay)
            self._queue.put(job)
            self._queue.task_done()

_dispatcher = TelegramDispatcher()

_pending_fingerprints: set[str] = set()

# Message deduplication cache
_sent_messages = {}
_message_cache_ttl = 300  # 5 minutes

# Rate limiting for frequent messages
_rate_limits = {}
_rate_limit_window = 60  # 1 minute
_max_messages_per_window = 5  # Max 5 messages per minute per type

# Periodic status tracking
_last_status_time = 0
_status_interval = 1 * 60 * 60  # 1 hour in seconds

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

def _send_message_sync(job: dict) -> bool:
    payload = dict(job.get("payload") or {})
    message_type = job.get("message_type", "general")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    try:
        response = _SESSION.post(url, json=payload, timeout=12)
    except requests.RequestException as e:
        if hasattr(e, "errno") and e.errno == 32:
            log_warning(
                "telegram.broken_pipe_request",
                "Telegram broken pipe error (connection closed)",
                context={"error": str(e), "message_type": message_type},
            )
        else:
            log_error(
                "telegram.request_exception",
                "Telegram request error",
                context={"error": str(e), "message_type": message_type},
            )
        return False
    except OSError as e:
        if getattr(e, "errno", None) == 32:
            log_warning(
                "telegram.broken_pipe_os",
                "Telegram broken pipe error (OS)",
                context={"error": str(e), "message_type": message_type},
            )
        else:
            log_error(
                "telegram.os_error",
                "Telegram OS error",
                context={"error": str(e), "message_type": message_type},
            )
        return False
    except Exception as e:
        log_error(
            "telegram.unexpected_error",
            "Telegram unexpected error",
            context={"error": str(e), "message_type": message_type},
        )
        return False

    if response.status_code == 200:
        log_info(
            "telegram.sent",
            "Telegram alert sent",
            context={"message_type": message_type},
        )
        return True

    if (
        response.status_code == 400
        and "can't parse entities" in response.text.lower()
        and payload.get("parse_mode")
    ):
        log_warning(
            "telegram.parse_error",
            "Telegram parse error detected, retrying without formatting",
            context={"status": response.status_code, "response": response.text[:120]},
        )
        payload_retry = {k: v for k, v in payload.items() if k != "parse_mode"}
        try:
            resp2 = _SESSION.post(url, json=payload_retry, timeout=12)
        except Exception as e:
            log_error(
                "telegram.send_failed_without_formatting",
                "Telegram send failed after removing formatting due to exception",
                context={"error": str(e), "message_type": message_type},
            )
            return False
        if resp2.status_code == 200:
            log_info(
                "telegram.sent_without_formatting",
                "Telegram alert sent without formatting after parse error fallback",
                context={"message_type": message_type},
            )
            return True
        log_error(
            "telegram.send_failed_without_formatting",
            "Telegram send failed after removing formatting",
            context={"status": resp2.status_code, "response": resp2.text[:200]},
        )
        return False

    log_error(
        "telegram.send_failed",
        "Telegram send failed",
        context={"status": response.status_code, "response": response.text[:200]},
    )
    return False

def send_telegram_message(
    message: str,
    markdown: bool = False,
    disable_preview: bool = True,
    deduplicate: bool = True,
    message_type: str = "general",
    async_mode: bool = True,
):
    """
    Send a Telegram message using secrets loaded from .env (via secrets.py).
    - message: text to send
    - markdown: enable Telegram Markdown parsing
    - disable_preview: disable link previews
    - deduplicate: prevent sending the same message multiple times within 5 minutes
    - message_type: category for rate limiting (e.g., "error", "trade", "status")
    - async_mode: enqueue message for asynchronous delivery (default). If False, send synchronously.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log_warning(
            "telegram.config.missing",
            "Telegram not configured (missing TELEGRAM_BOT_TOKEN/CHAT_ID).",
        )
        return False

    if not _check_rate_limit(message_type):
        log_warning(
            "telegram.rate_limited",
            "Telegram message rate limited",
            context={"message_type": message_type, "window": _rate_limit_window, "max_messages": _max_messages_per_window},
        )
        return False

    fingerprint = None
    if deduplicate:
        _cleanup_old_messages()
        fingerprint = _get_message_fingerprint(message)
        if fingerprint in _pending_fingerprints or fingerprint in _sent_messages:
            log_info(
                "telegram.duplicate_skipped",
                "Skipping duplicate Telegram message",
                context={"message_type": message_type},
            )
            return True
        _pending_fingerprints.add(fingerprint)

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "disable_web_page_preview": disable_preview,
    }
    if markdown:
        payload["parse_mode"] = "Markdown"

    job = {
        "payload": payload,
        "message_type": message_type,
        "deduplicate": deduplicate,
        "fingerprint": fingerprint,
        "attempt": 0,
    }

    if async_mode:
        _dispatcher.start()
        _dispatcher.submit(job)
        return True

    success = _send_message_sync(job)
    if deduplicate and fingerprint:
        if success:
            _sent_messages[fingerprint] = time.time()
        _pending_fingerprints.discard(fingerprint)
    return success

def send_periodic_status_report():
    """
    Send a comprehensive status report to Telegram every 1 hour.
    Includes bot status, buy/sell summary, and market conditions.
    """
    global _last_status_time
    current_time = time.time()
    
    # Check if it's time to send a status report (every 1 hour)
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
        log_warning(
            "telegram.periodic_status_failed",
            "Could not send periodic status report",
            context={"error": str(e)},
        )
        return False

def format_status_message(risk_summary, recent_summary, open_trades, market_regime=None):
    """Format a simplified status message with only Open Positions and Market Conditions"""
    from datetime import datetime
    import json
    import os
    import re
    
    # Helper to escape markdown special characters
    def _escape_markdown(text: str) -> str:
        """Escape markdown special characters to prevent parsing errors"""
        if not text:
            return text
        # Escape special markdown characters: _ * [ ] ( ) ~ ` > # + - = | { } . !
        return re.sub(r'([_*\[\]()~`>#+\-=|{}.!])', r'\\\1', str(text))
    
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
        positions = load_positions_store()
        if not positions:
            return [], 0.0
        
        pnl_lines = []
        total_unrealized_usd = 0.0
        
        for token_address, position_data in positions.items():
            try:
                if isinstance(position_data, dict):
                    entry_price = float(position_data.get("entry_price", 0))
                    chain_id = position_data.get("chain_id", "ethereum").lower()
                    symbol = position_data.get("symbol", "?")
                    position_size_usd = float(position_data.get("position_size_usd", 0))
                else:
                    # Legacy format
                    entry_price = float(position_data)
                    chain_id = "ethereum"
                    symbol = "?"
                    position_size_usd = 0.0
                
                if entry_price <= 0:
                    continue
                
                # Fetch current price
                current_price = _current_price(chain_id, token_address)
                
                if current_price > 0:
                    # Calculate PnL percentage
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    # Calculate unrealized USD if we have position size
                    unrealized_usd = None
                    if position_size_usd > 0:
                        unrealized_usd = (pnl_pct / 100) * position_size_usd
                        total_unrealized_usd += unrealized_usd
                    
                    # Format prices with appropriate precision (5-6 decimals for small prices)
                    entry_str = f"{entry_price:.5f}".rstrip('0').rstrip('.')
                    current_str = f"{current_price:.6f}".rstrip('0').rstrip('.')
                    # Format as requested: SYMBOL: entry $X → current $Y → PnL Z%
                    pnl_lines.append({
                        'symbol': symbol,
                        'entry': entry_str,
                        'current': current_str,
                        'pnl_pct': pnl_pct,
                        'unrealized_usd': unrealized_usd
                    })
                else:
                    # If we can't get current price, still show the position
                    pnl_lines.append({
                        'symbol': symbol,
                        'entry': f"{entry_price:.5f}".rstrip('0').rstrip('.'),
                        'current': 'N/A',
                        'pnl_pct': None,
                        'unrealized_usd': None
                    })
            except Exception as e:
                # Skip positions that fail to process
                continue
        
        return pnl_lines, total_unrealized_usd
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Build message starting with header
    msg = f"*Status Report* - {current_time}\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    
    # Add open positions with unrealized PnL
    msg += "*Open Positions Unrealized PnL:*\n"
    open_positions_pnl, total_unrealized_usd = _get_open_positions_pnl()
    
    if open_positions_pnl:
        for pos in open_positions_pnl:
            symbol = pos['symbol']
            entry = pos['entry']
            current = pos['current']
            
            if pos['pnl_pct'] is not None:
                pnl_pct = pos['pnl_pct']
                pnl_sign = "+" if pnl_pct >= 0 else ""
                pnl_emoji = "📈" if pnl_pct >= 0 else "📉"
                
                line = f"{pnl_emoji} {symbol}: entry ${entry} → current ${current} → PnL {pnl_sign}{pnl_pct:.2f}%"
                
                # Add USD unrealized if available
                if pos['unrealized_usd'] is not None:
                    usd_sign = "+" if pos['unrealized_usd'] >= 0 else ""
                    line += f" (${usd_sign}{pos['unrealized_usd']:.2f})"
                
                msg += f"{line}\n"
            else:
                msg += f"• {symbol}: entry ${entry} → current {current}\n"
        
        # Add total unrealized PnL if we have it
        if total_unrealized_usd != 0:
            total_sign = "+" if total_unrealized_usd >= 0 else ""
            msg += f"\n*Total Unrealized PnL:* ${total_sign}{total_unrealized_usd:.2f}\n"
    else:
        msg += "No open positions\n"
    
    # Add market conditions
    msg += "\n*Market Conditions:*\n"
    
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
        
        # Escape special characters for markdown
        regime_display = regime.replace('_', ' ').title()
        strategy_display = strategy.title()
        
        msg += f"{regime_emoji} *Regime:* {regime_display}\n"
        msg += f"📊 *Confidence:* {confidence:.1f}%\n"
        msg += f"💡 *Strategy:* {strategy_display}\n"
        # Escape description to prevent markdown parsing errors
        escaped_description = _escape_markdown(description)
        msg += f"📝 *Description:* {escaped_description}\n"
    else:
        msg += "• Bot is monitoring opportunities\n"
        msg += "• Following sustainable trading strategy\n"
        msg += "• Target: 10-20% consistent gains\n"
    
    return msg

def shutdown_telegram_dispatcher(drain: bool = False):
    """Stop the background telegram dispatcher."""
    _dispatcher.stop(drain=drain)


atexit.register(shutdown_telegram_dispatcher)