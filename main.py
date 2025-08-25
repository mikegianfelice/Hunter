import time
import yaml
from collections import defaultdict
from secrets import WALLET_ADDRESS, validate_secrets  # loaded from secure backend via secrets.py

from clear_state import ensure_mode_transition_clean

from token_scraper import fetch_trending_tokens
from sentiment_scraper import get_sentiment_score
from strategy import check_buy_signal, get_dynamic_take_profit, prune_price_memory
from multi_chain_executor import execute_trade
from telegram_bot import send_telegram_message
from token_sniffer import check_token_safety as is_token_safe
from cooldown import is_token_on_cooldown, update_cooldown_log
from blacklist_manager import is_blacklisted, add_to_blacklist, remove_from_blacklist
from risk_manager import allow_new_trade, register_buy, status_summary

# --- Load non-secret config ---
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f) or {}

# Current mode
TEST_MODE = bool(config.get("test_mode", True))

# One-time cleaner: if mode changed since last run (or reset_state_now: true)
ensure_mode_transition_clean(TEST_MODE, force_reset=bool(config.get("reset_state_now", False)))

# Pull settings from config
WALLET = WALLET_ADDRESS  # from .env
TRADE_AMOUNT = float(config.get("trade_amount_usd", 5))
SLIPPAGE = float(config.get("slippage", 0.02))
TAKE_PROFIT = float(config.get("take_profit", 0.5))
STOP_LOSS = float(config.get("stop_loss", 0.25))
USE_DYNAMIC_TP = bool(config.get("use_dynamic_tp", False))

# Normalize trusted token addresses to a lowercase set (robust to None/str/list)
def _normalize_trusted(val):
    if not val:
        return set()
    if isinstance(val, str):
        vals = [val]
    elif isinstance(val, (list, tuple, set)):
        vals = list(val)
    else:
        return set()
    return {str(a).strip().lower() for a in vals if isinstance(a, str) and a.strip()}

TRUSTED_TOKENS = _normalize_trusted(config.get("trusted_tokens"))

# ---- Debug rejection tracking keys ----
REJECT_BLACKLIST   = "blacklisted"
REJECT_COOLDOWN    = "cooldown"
REJECT_SNIFFER     = "tokensniffer_unsafe"
REJECT_SENTIMENT   = "sentiment_fail"
REJECT_BUY_SIGNAL  = "no_buy_signal"
REJECT_RISK        = "risk_blocked"
REJECT_MISSINGADDR = "missing_address"

def trade_loop():
    print("🔁 Starting trade loop...")

    # Housekeeping: prune stale price memory each loop
    removed = prune_price_memory()
    if removed:
        print(f"🧽 Price memory cleanup: removed {removed} stale entries")

    tokens = fetch_trending_tokens(limit=100)

    # optional: print current risk state each loop
    risk = status_summary()
    print(f"🧯 Risk status: {risk}")

    # For end-of-loop summary
    rejections = defaultdict(list)  # reason -> list of (symbol, addr)
    buys = []
    
    if not tokens:
        print("😴 No valid tokens found this cycle. Waiting for next discovery...")
        return

    for token in tokens:
        try:
            symbol = token.get("symbol", "UNKNOWN")
            address = (token.get("address") or "").lower()
            print(f"\n🚀 Evaluating token: {symbol}")

            if not address:
                print("⚠️ Missing token address; skipping.")
                rejections[REJECT_MISSINGADDR].append((symbol, ""))
                continue

            is_trusted = address in TRUSTED_TOKENS
            token["is_trusted"] = is_trusted  # let strategy relax logic for trusted tokens

            # Auto-unblacklist trusted tokens so they can pass
            if is_trusted and is_blacklisted(address):
                remove_from_blacklist(address)

            # --- Safety / hygiene (skip for trusted tokens) ---
            if not is_trusted:
                if is_blacklisted(address):
                    print("⛔ Token is blacklisted.")
                    rejections[REJECT_BLACKLIST].append((symbol, address))
                    continue

                if is_token_on_cooldown(address):
                    print("⏳ Token is in cooldown.")
                    rejections[REJECT_COOLDOWN].append((symbol, address))
                    continue

                # TokenSniffer / safety gate
                chain_id = token.get("chainId", "ethereum")
                print(f"🔍 Checking safety for {symbol} on {chain_id.upper()}")
                print(f"   Token data: {token}")
                if not is_token_safe(address, chain_id):
                    print("⚠️ TokenSniffer marked as unsafe.")
                    add_to_blacklist(address)
                    rejections[REJECT_SNIFFER].append((symbol, address))
                    continue
            else:
                print("🔓 Trusted token — skipping blacklist, cooldown, and TokenSniffer")

            # --- Sentiment (Ethereum only) ---
            chain_id = token.get("chainId", "ethereum").lower()
            if chain_id == "ethereum":
                sentiment = get_sentiment_score(token) or {}
                print(f"🧠 Sentiment for ${symbol}: {sentiment}")

                # Attach sentiment to token so strategy/executor can use it
                token["sent_score"] = sentiment.get("score")
                token["sent_mentions"] = sentiment.get("mentions")
                token["sent_status"] = sentiment.get("status")

                # Temporarily disable sentiment checks to allow trades
                print("🔓 Sentiment checks temporarily disabled")
                # if not is_trusted:
                #     if (
                #         sentiment.get("status") == "blocked"
                #         or (sentiment.get("mentions") or 0) < 1  # Reduced from 3 to 1
                #         or (sentiment.get("score") or 0) < 30   # Reduced from 60 to 30
                #     ):
                #         print("📉 Token failed sentiment check.")
                #         rejections[REJECT_SENTIMENT].append((symbol, address))
                #         continue
            else:
                # Skip sentiment for non-Ethereum chains
                print(f"🔓 Skipping sentiment for {chain_id.upper()} token (not required)")
                token["sent_score"] = 100  # Default high score for non-Ethereum
                token["sent_mentions"] = 10  # Default high mentions for non-Ethereum
                token["sent_status"] = "ok"

            # --- Strategy signal ---
            if not check_buy_signal(token):
                print("❌ No buy signal.")
                rejections[REJECT_BUY_SIGNAL].append((symbol, address))
                continue

            # --- Risk manager gate ---
            chain_id = token.get("chainId", "ethereum")
            allowed, reason = allow_new_trade(TRADE_AMOUNT, address, chain_id)
            if not allowed:
                print(f"🛑 Risk manager blocked trade: {reason}")
                send_telegram_message(f"🛑 Trade blocked by risk controls: {reason}")
                rejections[REJECT_RISK].append((symbol, address))
                continue

            # --- Determine TP (static or dynamic) ---
            tp = get_dynamic_take_profit(token) if USE_DYNAMIC_TP else TAKE_PROFIT

            # --- Execute trade (persists position, sends TG open alert, logs BUY) ---
            tx_hash, success = execute_trade(token, TRADE_AMOUNT)

            if success:
                buys.append((symbol, address))
                register_buy(TRADE_AMOUNT)  # inform risk manager
                send_telegram_message(
                    f"✅ Bought {symbol}\n"
                    f"TP: {tp * 100:.0f}% | SL: {STOP_LOSS * 100:.0f}%\n"
                    f"TX: https://etherscan.io/tx/{tx_hash}"
                )
            else:
                # Check if the failure might be due to delisting
                chain_id = token.get("chainId", "ethereum").lower()
                if chain_id == "solana":
                    try:
                        from solana_executor import get_token_price_usd
                        current_price = get_token_price_usd(address)
                        if current_price == 0:
                            print(f"🚨 Trade failed and token has zero price - likely delisted")
                            # Add to delisted tokens instead of cooldown
                            from strategy import _add_to_delisted_tokens
                            _add_to_delisted_tokens(address, symbol, "Trade failed + zero price")
                            rejections[REJECT_RISK].append((symbol, address))
                            continue
                    except Exception as e:
                        print(f"⚠️ Could not verify token status: {e}")
                
                # If not delisted, add to cooldown as usual
                update_cooldown_log(address)
                # Don't send telegram for WETH since it's intentionally skipped
                if symbol != "WETH":
                    send_telegram_message(f"❌ Failed to buy {symbol}, added to cooldown.")
                # Treat failed execution as implicitly rejected by risk/execution environment
                rejections[REJECT_RISK].append((symbol, address))

            # small pause between tokens
            time.sleep(3)

        except Exception as e:
            print(f"🔥 Error while evaluating {token}: {e}")

    # ---- End-of-loop debug summary ----
    _print_reject_summary(rejections, buys)


def _print_reject_summary(rejections, buys):
    total_eval = sum(len(v) for v in rejections.values()) + len(buys)
    print("\n📋 Evaluation summary")
    print(f"• Tokens evaluated: {total_eval}")
    print(f"• Buys executed:   {len(buys)}")
    if buys:
        sample = ", ".join([s for s, _ in buys[:5]])
        print(f"  ↳ {sample}{'…' if len(buys) > 5 else ''}")

    counts = {reason: len(items) for reason, items in rejections.items()}
    if not counts:
        print("• No rejections recorded this loop.")
        return

    # Order reasons by most frequent
    for reason, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"• Rejected ({reason}): {count}")
        samples = rejections[reason][:5]
        if samples:
            names = ", ".join([s for s, _ in samples])
            print(f"  ↳ {names}{'…' if count > 5 else ''}")

if __name__ == "__main__":
    # Validate secrets before starting
    if not validate_secrets():
        print("❌ Exiting due to missing secrets")
        exit(1)
    
    print("🔐 Secrets validated successfully")
    
    while True:
        try:
            trade_loop()
            time.sleep(60)  # wait before next discovery cycle
        except Exception as e:
            print(f"🔥 Bot crashed: {e}")
            time.sleep(30)