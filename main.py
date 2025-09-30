#!/usr/bin/env python3
"""
Crypto Trading Bot - Main Entry Point
Automatically clears Python cache to ensure latest code is used
"""

import os
import sys
import shutil

# Clear Python cache BEFORE any other imports
def clear_python_cache():
    """Clear Python cache to ensure we're using the latest code"""
    try:
        # Check if any .py files are newer than their .pyc files
        cache_needs_clearing = False
        
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.py'):
                    py_path = os.path.join(root, file)
                    pyc_path = os.path.join(root, file.replace('.py', '.pyc'))
                    
                    # Check if .pyc exists and if .py is newer
                    if os.path.exists(pyc_path):
                        py_mtime = os.path.getmtime(py_path)
                        pyc_mtime = os.path.getmtime(pyc_path)
                        if py_mtime > pyc_mtime:
                            cache_needs_clearing = True
                            break
        
        if cache_needs_clearing:
            print("🔄 Source files modified - clearing Python cache...")
            
            # Remove all .pyc files
            for root, dirs, files in os.walk('.'):
                for file in files:
                    if file.endswith('.pyc'):
                        os.remove(os.path.join(root, file))
            
            # Remove all __pycache__ directories
            for root, dirs, files in os.walk('.'):
                for dir_name in dirs:
                    if dir_name == '__pycache__':
                        shutil.rmtree(os.path.join(root, dir_name))
            
            print("🧹 Python cache cleared - using latest code")
        else:
            print("✅ Python cache is up to date")
            
    except Exception as e:
        print(f"⚠️ Cache clearing failed: {e}")

# Clear cache immediately
clear_python_cache()

# Now import all modules (they'll be fresh)
import time
import json
import yaml
import signal
from datetime import datetime
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from secrets import WALLET_ADDRESS, validate_secrets  # loaded from secure backend via secrets.py
from clear_state import ensure_mode_transition_clean

from token_scraper import fetch_trending_tokens
from sentiment_scraper import get_sentiment_score
from strategy import check_buy_signal, get_dynamic_take_profit, prune_price_memory
from multi_chain_executor import execute_trade
from telegram_bot import send_telegram_message
from token_sniffer import check_token_safety as is_token_safe
from cooldown import is_token_on_cooldown, update_cooldown_log
from blacklist_manager import (
    is_blacklisted, add_to_blacklist, remove_from_blacklist, 
    record_failure, review_blacklisted_tokens, get_blacklist_stats
)
from risk_manager import allow_new_trade, register_buy, status_summary

# --- Load non-secret config ---
from config_loader import get_config, get_config_bool, get_config_float, get_config_int

# Dynamic config loading
def get_main_config():
    """Get current configuration values dynamically"""
    return {
        'TEST_MODE': get_config_bool("test_mode", True),
        'TRADE_AMOUNT': get_config_float("trade_amount_usd", 5),
        'SLIPPAGE': get_config_float("slippage", 0.02),
        'TAKE_PROFIT': get_config_float("take_profit", 0.5),
        'STOP_LOSS': get_config_float("stop_loss", 0.25),
        'USE_DYNAMIC_TP': get_config_bool("use_dynamic_tp", False),
        'ENABLE_SMART_BLACKLIST_CLEANUP': get_config_bool("enable_smart_blacklist_cleanup", True),
        'BLACKLIST_CLEANUP_INTERVAL': get_config_int("blacklist_cleanup_interval", 6),
        'BLACKLIST_KEEP_FAILURE_THRESHOLD': get_config_int("blacklist_keep_failure_threshold", 3),
        'ENABLE_SMART_DELISTED_CLEANUP': get_config_bool("enable_smart_delisted_cleanup", True),
        'DELISTED_CLEANUP_INTERVAL': get_config_int("delisted_cleanup_interval", 12)
    }

# Current mode
config = get_main_config()
TEST_MODE = config['TEST_MODE']

# One-time cleaner: if mode changed since last run (or reset_state_now: true)
ensure_mode_transition_clean(TEST_MODE, force_reset=get_config_bool("reset_state_now", False))

# Pull settings from config
WALLET = WALLET_ADDRESS  # from .env
TRADE_AMOUNT = config['TRADE_AMOUNT']
SLIPPAGE = config['SLIPPAGE']
TAKE_PROFIT = config['TAKE_PROFIT']
STOP_LOSS = config['STOP_LOSS']
USE_DYNAMIC_TP = config['USE_DYNAMIC_TP']

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

TRUSTED_TOKENS = _normalize_trusted(get_config("trusted_tokens"))

# ---- Debug rejection tracking keys ----
REJECT_BLACKLIST   = "blacklisted"
REJECT_COOLDOWN    = "cooldown"
REJECT_SNIFFER     = "tokensniffer_unsafe"
REJECT_SENTIMENT   = "sentiment_fail"
REJECT_BUY_SIGNAL  = "no_buy_signal"
REJECT_RISK        = "risk_blocked"
REJECT_MISSINGADDR = "missing_address"

# Smart blacklist maintenance
ENABLE_SMART_BLACKLIST_CLEANUP = bool(config.get("enable_smart_blacklist_cleanup", True))
BLACKLIST_CLEANUP_INTERVAL = int(config.get("blacklist_cleanup_interval", 6))
BLACKLIST_KEEP_FAILURE_THRESHOLD = int(config.get("blacklist_keep_failure_threshold", 3))
blacklist_cleanup_counter = 0

# Smart delisted token cleanup
ENABLE_SMART_DELISTED_CLEANUP = bool(config.get("enable_smart_delisted_cleanup", True))
DELISTED_CLEANUP_INTERVAL = int(config.get("delisted_cleanup_interval", 12))  # Every 12 loops (12 hours)
delisted_cleanup_counter = 0

def smart_blacklist_maintenance():
    """Automatically clean blacklist to maintain trading opportunities"""
    global blacklist_cleanup_counter
    
    config = get_main_config()
    
    if not config['ENABLE_SMART_BLACKLIST_CLEANUP']:
        return
        
    blacklist_cleanup_counter += 1
    
    if blacklist_cleanup_counter < config['BLACKLIST_CLEANUP_INTERVAL']:
        return
    
    blacklist_cleanup_counter = 0  # Reset counter
    
    print("\n🧹 Running smart blacklist maintenance...")
    
    try:
        # Load current blacklist data
        if not os.path.exists("delisted_tokens.json"):
            return
        
        with open("delisted_tokens.json", "r") as f:
            data = json.load(f)
        
        delisted_tokens = data.get("delisted_tokens", [])
        failure_counts = data.get("failure_counts", {})
        
        if not delisted_tokens:
            print("✅ Blacklist is already clean")
            return
        
        # Identify high-risk tokens to keep
        high_risk_tokens = set()
        
        # Keep tokens with N+ failures (configurable threshold)
        for token, count in failure_counts.items():
            if count >= config['BLACKLIST_KEEP_FAILURE_THRESHOLD']:
                high_risk_tokens.add(token.lower())
        
        # Keep tokens that caused 100% losses (from trade log)
        if os.path.exists("trade_log.csv"):
            import csv
            with open("trade_log.csv", "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if float(row.get("pnl_pct", 0)) <= -99.9:  # 100% loss
                        high_risk_tokens.add(row.get("token", "").lower())
        
        # Calculate safe tokens to remove
        safe_to_remove = []
        for token in delisted_tokens:
            if token.lower() not in high_risk_tokens:
                safe_to_remove.append(token)
        
        if not safe_to_remove:
            print("⚠️ No safe tokens to remove - all are high-risk")
            return
        
        # Remove safe tokens (keep high-risk ones)
        original_count = len(delisted_tokens)
        delisted_tokens = [t for t in delisted_tokens if t not in safe_to_remove]
        
        # Update data
        data["delisted_tokens"] = delisted_tokens
        data["removed_count"] = original_count - len(delisted_tokens)
        data["remaining_count"] = len(delisted_tokens)
        data["quick_cleaned_at"] = datetime.now().isoformat()
        
        # Save updated data
        with open("delisted_tokens.json", "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Smart cleanup: removed {len(safe_to_remove)} safer tokens")
        print(f"📊 Kept {len(high_risk_tokens)} high-risk tokens")
        print(f"📊 Remaining blacklist: {len(delisted_tokens)} tokens")
        
    except Exception as e:
        print(f"⚠️ Blacklist maintenance failed: {e}")

def smart_delisted_cleanup():
    """Automatically clean delisted tokens list to maintain trading opportunities"""
    global delisted_cleanup_counter
    
    config = get_main_config()
    
    if not config['ENABLE_SMART_DELISTED_CLEANUP']:
        return
        
    delisted_cleanup_counter += 1
    
    if delisted_cleanup_counter < config['DELISTED_CLEANUP_INTERVAL']:
        return
    
    delisted_cleanup_counter = 0  # Reset counter
    
    print("\n🧹 Running smart delisted token cleanup...")
    
    try:
        from smart_blacklist_cleaner import clean_delisted_tokens
        result = clean_delisted_tokens()
        
        if result:
            removed_count = result.get("removed_count", 0)
            remaining_count = result.get("remaining_count", 0)
            print(f"✅ Delisted cleanup completed: {removed_count} tokens reactivated, {remaining_count} still delisted")
        else:
            print("⚠️ Delisted cleanup failed")
            
    except Exception as e:
        print(f"❌ Error during delisted cleanup: {e}")

def trade_loop():
    print("🔁 Starting trade loop...")

    # Smart blacklist maintenance (runs every 6 loops)
    smart_blacklist_maintenance()
    
    # Smart delisted token cleanup (runs every 12 loops)
    smart_delisted_cleanup()

    # Housekeeping: prune stale price memory each loop
    removed = prune_price_memory()
    if removed:
        print(f"🧽 Price memory cleanup: removed {removed} stale entries")

    tokens = fetch_trending_tokens(limit=200)  # Increased from 100 to 200 for more opportunities

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
            # Safely extract token information with defensive programming
            symbol = token.get("symbol", "UNKNOWN") if isinstance(token, dict) else "UNKNOWN"
            address = (token.get("address") or "").lower() if isinstance(token, dict) else ""
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
                
                # Quick tradeability check for non-trusted tokens
                from tradeability_checker import quick_tradeability_check
                chain_id = token.get("chainId", "ethereum")
                if not quick_tradeability_check(address, chain_id):
                    print("❌ Token is not tradeable on current DEXs.")
                    rejections[REJECT_RISK].append((symbol, address))  # Use REJECT_RISK for now
                    continue

                # TokenSniffer / safety gate
                chain_id = token.get("chainId", "ethereum")
                print(f"🔍 Checking safety for {symbol} on {chain_id.upper()}")
                print(f"   Token data: {token}")
                
                # Check if TokenSniffer is enabled
                if get_config_bool("enable_tokensniffer", False):
                    # Enhanced error handling for TokenSniffer
                    try:
                        if not is_token_safe(address, chain_id):
                            print("⚠️ TokenSniffer marked as unsafe.")
                            # Use failure tracking instead of immediate blacklisting
                            if record_failure(address, "tokensniffer_unsafe"):
                                rejections[REJECT_SNIFFER].append((symbol, address))
                                continue
                            else:
                                print("⚠️ Token marked as unsafe but not blacklisted (failure tracking)")
                                rejections[REJECT_SNIFFER].append((symbol, address))
                                continue
                    except Exception as e:
                        print(f"⚠️ TokenSniffer check failed for {symbol}: {e}")
                        # Record failure but don't blacklist for API errors
                        record_failure(address, "tokensniffer_api_error", should_blacklist=False)
                        print("⚠️ Skipping TokenSniffer check due to API error")
                        # Continue with evaluation instead of rejecting
                else:
                    print("🔓 TokenSniffer disabled in config - skipping safety check")
            else:
                print("🔓 Trusted token — skipping blacklist, cooldown, and TokenSniffer")

            # --- Sentiment (Ethereum only) ---
            chain_id = token.get("chainId", "ethereum").lower()
            config = get_main_config()
            
            # Check if sentiment checks are enabled
            if get_config_bool("enable_sentiment_checks", False):
                if chain_id == "ethereum":
                    sentiment = get_sentiment_score(token) or {}
                    print(f"🧠 Sentiment for ${symbol}: {sentiment}")

                    # Attach sentiment to token so strategy/executor can use it
                    token["sent_score"] = sentiment.get("score")
                    token["sent_mentions"] = sentiment.get("mentions")
                    token["sent_status"] = sentiment.get("status")

                    # Sentiment checks are enabled
                    if not is_trusted:
                        if (
                            sentiment.get("status") == "blocked"
                            or (sentiment.get("mentions") or 0) < 1
                            or (sentiment.get("score") or 0) < 30
                        ):
                            print("📉 Token failed sentiment check.")
                            rejections[REJECT_SENTIMENT].append((symbol, address))
                            continue
                else:
                    # Skip sentiment for non-Ethereum chains
                    print(f"🔓 Skipping sentiment for {chain_id.upper()} token (not required)")
                    token["sent_score"] = 100  # Default high score for non-Ethereum
                    token["sent_mentions"] = 10  # Default high mentions for non-Ethereum
                    token["sent_status"] = "ok"
            else:
                # Sentiment checks disabled - use default values
                print("🔓 Sentiment checks disabled in config")
                token["sent_score"] = 100  # Default high score
                token["sent_mentions"] = 10  # Default high mentions
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
                # Check if the failure might be due to delisting (only if enabled in config)
                from config_loader import get_config_values
                config = get_config_values()
                
                chain_id = token.get("chainId", "ethereum").lower()
                if chain_id == "solana" and config.get('ENABLE_PRE_BUY_DELISTING_CHECK', False):
                    try:
                        # Use the price data already available in the token object
                        current_price = float(token.get("priceUsd", 0))
                        volume_24h = float(token.get("volume24h", 0))
                        liquidity = float(token.get("liquidity", 0))
                        
                        # Only mark as delisted if:
                        # 1. Price is 0 from API check (not DexScreener data)
                        # 2. Volume/liquidity are very low
                        # 3. DexScreener data also shows low metrics
                        if current_price == 0 and volume_24h < 100 and liquidity < 500:
                            print(f"🚨 Trade failed and token has zero price with very low metrics - likely delisted")
                            # Add to delisted tokens instead of cooldown
                            from strategy import _add_to_delisted_tokens
                            _add_to_delisted_tokens(address, symbol, "Trade failed + zero price + low metrics")
                            rejections[REJECT_RISK].append((symbol, address))
                            continue
                        elif current_price == 0:
                            print(f"⚠️ Trade failed and token has zero price but decent metrics - not marking as delisted")
                        else:
                            print(f"✅ Token price verified: ${current_price}")
                    except Exception as e:
                        print(f"⚠️ Could not verify token status: {e}")
                else:
                    print(f"ℹ️ Delisting check disabled or non-Solana chain - skipping delisting verification")
                
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
            # Get token info safely for error reporting
            token_symbol = token.get("symbol", "UNKNOWN") if isinstance(token, dict) else "UNKNOWN"
            token_address = token.get("address", "N/A") if isinstance(token, dict) else "N/A"
            print(f"🔥 Error while evaluating {token_symbol} ({token_address}): {e}")

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
    
    # Initialize blacklist review counter
    blacklist_review_counter = 0
    
    while True:
        try:
            # Run smart blacklist maintenance every loop
            smart_blacklist_maintenance()
            smart_delisted_cleanup()
            
            # Periodic blacklist review (every 24 hours)
            blacklist_review_counter += 1
            if blacklist_review_counter >= 1440:  # 24 hours (1440 minutes)
                print("\n🔄 Running periodic blacklist review...")
                removed_count = review_blacklisted_tokens()
                if removed_count > 0:
                    print(f"✅ Removed {removed_count} old blacklisted tokens")
                
                # Print blacklist statistics
                stats = get_blacklist_stats()
                print(f"📊 Blacklist stats: {stats['blacklisted_count']} blacklisted, {stats['failure_tracking_count']} tracked")
                
                blacklist_review_counter = 0  # Reset counter
            
            trade_loop()
            time.sleep(60)  # wait before next discovery cycle
        except Exception as e:
            print(f"🔥 Bot crashed: {e}")
            time.sleep(30)