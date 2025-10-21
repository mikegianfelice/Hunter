# telegram_bot.py
import requests
import time
from secrets import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

# Message deduplication cache
_sent_messages = {}
_message_cache_ttl = 300  # 5 minutes

def _cleanup_old_messages():
    """Remove old messages from the cache"""
    current_time = time.time()
    global _sent_messages
    _sent_messages = {
        msg: timestamp for msg, timestamp in _sent_messages.items()
        if current_time - timestamp < _message_cache_ttl
    }

def send_telegram_message(message: str, markdown: bool = False, disable_preview: bool = True, deduplicate: bool = True):
    """
    Send a Telegram message using secrets loaded from .env (via secrets.py).
    - message: text to send
    - markdown: enable Telegram Markdown parsing
    - disable_preview: disable link previews
    - deduplicate: prevent sending the same message multiple times within 5 minutes
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram not configured (missing TELEGRAM_BOT_TOKEN/CHAT_ID).")
        return False

    # Deduplication logic
    if deduplicate:
        _cleanup_old_messages()
        current_time = time.time()
        
        # Check if this exact message was sent recently
        if message in _sent_messages:
            print(f"📨 Skipping duplicate Telegram message: {message[:50]}...")
            return True  # Return True since we "handled" it
        
        # Add to cache
        _sent_messages[message] = current_time

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