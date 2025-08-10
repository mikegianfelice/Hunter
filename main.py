import time
import yaml
from token_scraper import fetch_trending_tokens  # ← renamed properly
from sentiment_scraper import get_sentiment_score
from strategy import check_buy_signal, get_dynamic_take_profit
from executor import execute_trade
from trade_logger import log_trade
from telegram_bot import send_telegram_message
from token_sniffer import check_token_safety as is_token_safe  # ← aliased for consistency
from cooldown_manager import is_in_cooldown, add_to_cooldown
from blacklist import is_blacklisted, add_to_blacklist

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

WALLET = config["wallet_address"]
TRADE_AMOUNT = config["trade_amount_usd"]
SLIPPAGE = config["slippage"]
TAKE_PROFIT = config["take_profit"]
STOP_LOSS = config["stop_loss"]
USE_DYNAMIC_TP = config.get("use_dynamic_tp", False)

def trade_loop():
    print("🔁 Starting trade loop...")
    tokens = fetch_trending_tokens(limit=10)

    for token in tokens:
        print(f"\n🚀 Evaluating token: {token}")

        if is_blacklisted(token):
            print("⛔ Token is blacklisted.")
            continue

        if is_in_cooldown(token):
            print("⏳ Token is in cooldown.")
            continue

        if not is_token_safe(token):
            print("⚠️ TokenSniffer marked as unsafe.")
            add_to_blacklist(token)
            continue

        sentiment = get_sentiment_score(token)
        if (
            sentiment["status"] == "blocked"
            or sentiment["mentions"] < 3
            or sentiment["score"] < 60
        ):
            print("📉 Token failed sentiment check.")
            continue

        if not check_buy_signal(token):
            print("❌ No buy signal.")
            continue

        tp = get_dynamic_take_profit(token) if USE_DYNAMIC_TP else TAKE_PROFIT

        tx_hash, success = execute_trade(token, TRADE_AMOUNT, SLIPPAGE, tp, STOP_LOSS)

        if success:
            send_telegram_message(
                f"✅ Bought {token}\n"
                f"TP: {tp * 100:.0f}% | SL: {STOP_LOSS * 100:.0f}%\n"
                f"TX: https://etherscan.io/tx/{tx_hash}"
            )
            log_trade(token, "BUY", tx_hash)
        else:
            add_to_cooldown(token)
            send_telegram_message(
                f"❌ Failed to buy {token}, added to cooldown."
            )

        time.sleep(3)

if __name__ == "__main__":
    while True:
        try:
            trade_loop()
            time.sleep(60)
        except Exception as e:
            print(f"🔥 Bot crashed: {e}")
            time.sleep(30)