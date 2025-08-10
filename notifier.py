import requests
import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{config['telegram_bot_token']}/sendMessage"
    data = {"chat_id": config["telegram_chat_id"], "text": message}
    requests.post(url, data=data)