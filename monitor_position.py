import time
from uniswap_executor import sell_token
from utils import fetch_token_price_usd
from telegram_bot import send_telegram_message
import yaml

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

TAKE_PROFIT = config.get("take_profit", 0.5)
STOP_LOSS = config.get("stop_loss", 0.25)

def monitor_position():
    try:
        with open("entry_price.txt", "r") as f:
            line = f.read().strip()
            token_address, entry_price_str = line.split(",")
            entry_price = float(entry_price_str)
    except FileNotFoundError:
        print("📭 No open position found.")
        return
    except Exception as e:
        print(f"⚠️ Failed to read entry_price.txt: {e}")
        return

    print(f"🔍 Monitoring token: {token_address}")
    print(f"🎯 Entry price: ${entry_price:.6f}")

    try:
        current_price = fetch_token_price_usd(token_address)
    except Exception as e:
        print(f"⚠️ Price fetch failed: {e}")
        return

    if current_price is None:
        print("⚠️ Could not fetch current price.")
        return

    print(f"📈 Current price: ${current_price:.6f}")

    gain = (current_price - entry_price) / entry_price
    print(f"📊 PnL: {gain*100:.2f}%")

    if gain >= TAKE_PROFIT:
        print("💰 Take-profit hit! Selling...")
        sell_token(token_address)
        send_telegram_message(
            f"💰 Take-profit triggered!\n"
            f"Token: {token_address}\n"
            f"Entry: ${entry_price:.6f}\n"
            f"Now: ${current_price:.6f} (+{gain*100:.2f}%)"
        )
        _clear_position()

    elif gain <= -STOP_LOSS:
        print("🛑 Stop-loss hit! Selling...")
        sell_token(token_address)
        send_telegram_message(
            f"🛑 Stop-loss triggered!\n"
            f"Token: {token_address}\n"
            f"Entry: ${entry_price:.6f}\n"
            f"Now: ${current_price:.6f} ({gain*100:.2f}%)"
        )
        _clear_position()
    else:
        print("⏳ Holding position.")


def _clear_position():
    try:
        open("entry_price.txt", "w").close()
        print("🧹 Position cleared.")
    except Exception as e:
        print(f"⚠️ Could not clear position: {e}")


if __name__ == "__main__":
    while True:
        monitor_position()
        time.sleep(30)  # Check every 30s