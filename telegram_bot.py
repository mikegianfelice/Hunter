# telegram_bot.py
import requests
from secrets import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

def send_telegram_message(message: str, markdown: bool = False, disable_preview: bool = True):
    """
    Send a Telegram message using secrets loaded from .env (via secrets.py).
    - message: text to send
    - markdown: enable Telegram Markdown parsing
    - disable_preview: disable link previews
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram not configured (missing TELEGRAM_BOT_TOKEN/CHAT_ID).")
        return False

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
        print(f"❌ Telegram request error: {e}")
        return False