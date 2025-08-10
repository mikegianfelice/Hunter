import yaml
from strategy import calculate_indicators, check_buy_signal, get_dynamic_take_profit
from uniswap_executor import buy_token, sell_token
from telegram_bot import send_telegram_message

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

TRADE_AMOUNT_USD = config["trade_amount_usd"]
SLIPPAGE = config["slippage"]
TAKE_PROFIT = config["take_profit"]
USE_DYNAMIC_TP = config.get("use_dynamic_tp", False)
STOP_LOSS = config["stop_loss"]

# === Token to test with ===
TOKEN_ADDRESS = "0x6982508145454Ce325dDbE47a25d4ec3d2311933"  # PEPE
ETH_AMOUNT = 0.005  # ~$15 USD

# === Step 1: FORCE BUY SIGNAL ===
print("🔍 Forcing buy signal for test...")

# === Step 2: Determine take profit
if USE_DYNAMIC_TP:
    dynamic_tp = get_dynamic_take_profit(TOKEN_ADDRESS)
    tp_to_use = dynamic_tp
    print(f"📈 Using dynamic TP: {dynamic_tp:.2f}")
else:
    tp_to_use = TAKE_PROFIT
    print(f"📈 Using static TP: {tp_to_use:.2f}")

# === Step 3: Simulate buy (DRY RUN)
tx = buy_token(TOKEN_ADDRESS, ETH_AMOUNT, SLIPPAGE)

# === Step 4: Send Telegram alert
try:
    send_telegram_message(
        f"🤖 *DRY RUN TRADE EXECUTED!*\n\nBought `{ETH_AMOUNT} ETH` of token:\n`{TOKEN_ADDRESS}`\nTP set to `{tp_to_use * 100:.1f}%`"
    )
except Exception as e:
    print(f"⚠️ Telegram failed: {e}")