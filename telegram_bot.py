import requests
import yaml

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

BOT_TOKEN = config["telegram_bot_token"]
CHAT_ID = config["telegram_chat_id"]

def send_telegram_message(message: str):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }

    response = requests.post(url, json=payload)

    if response.status_code != 200:
        print(f"❌ Telegram failed: {response.text}")
    else:
        print("📨 Telegram alert sent!")