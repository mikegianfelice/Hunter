#!/usr/bin/env python3
"""
Background service to track 5-minute price snapshots for tracked tokens (pippin, fartcoin, usor).
Run this as a background process to continuously track prices every 5 minutes.
"""

import sys
import os
import time
import signal
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.storage.minute_price_tracker import (
    store_price_snapshot,
    TRACKED_TOKENS,
    INTERVAL_SECONDS,
)
from src.execution.jupiter_executor import JupiterCustomExecutor
from src.monitoring.structured_logger import log_info, log_error

running = True


def signal_handler(sig, frame):
    """Handle graceful shutdown"""
    global running
    print("\n🛑 Shutting down 5-minute price tracker...")
    running = False


def get_token_price(token_address: str) -> float:
    """Get current token price using Jupiter executor"""
    try:
        executor = JupiterCustomExecutor()
        price = executor.get_token_price_usd(token_address)
        return price if price and price > 0 else 0.0
    except Exception as e:
        log_error("price_tracker.price_fetch_error", f"Error fetching price for {token_address[:8]}...: {e}")
        return 0.0


def track_prices():
    """Main tracking loop - runs every 5 minutes"""
    global running
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🚀 Starting 5-minute price tracker...")
    print(f"📊 Tracking tokens: {list(TRACKED_TOKENS.keys())}")
    print(f"⏱️  Interval: {INTERVAL_SECONDS // 60} minutes")
    print(f"📈 Expected API calls: ~{len(TRACKED_TOKENS) * (24 * 60 // (INTERVAL_SECONDS // 60))} per day")
    
    while running:
        try:
            current_time = time.time()
            # Round down to nearest 5-minute interval
            current_interval = int(current_time // INTERVAL_SECONDS) * INTERVAL_SECONDS
            
            for token_name, token_address in TRACKED_TOKENS.items():
                price = get_token_price(token_address)
                
                # Fallback value from Jupiter executor when all APIs fail (0.000001)
                # Also check for 0.00001 which might be another fallback threshold
                FALLBACK_PRICE = 0.000001
                is_fallback = abs(price - FALLBACK_PRICE) < 1e-9 or abs(price - 0.00001) < 1e-9
                
                if price > 0 and not is_fallback:
                    stored = store_price_snapshot(token_address, price, current_time)
                    if stored:
                        log_info(
                            "price_tracker.snapshot_stored",
                            f"✅ {token_name}: ${price:.8f} at {datetime.fromtimestamp(current_interval).strftime('%Y-%m-%d %H:%M:%S')}",
                            token=token_name,
                            price=price,
                            timestamp=current_interval,
                        )
                    # If not stored, it means we already have data for this interval (duplicate)
                elif is_fallback:
                    log_error(
                        "price_tracker.fallback_price_skipped",
                        f"⚠️ {token_name}: Skipping fallback price ${price:.8f} (price fetch failed - all APIs unavailable)",
                        token=token_name,
                        price=price,
                    )
                else:
                    log_error(
                        "price_tracker.price_zero",
                        f"⚠️ {token_name}: Invalid price (${price})",
                        token=token_name,
                    )
            
            # Sleep until next 5-minute interval (with small buffer)
            sleep_time = INTERVAL_SECONDS - (time.time() % INTERVAL_SECONDS) + 1
            time.sleep(min(sleep_time, INTERVAL_SECONDS))
            
        except KeyboardInterrupt:
            running = False
            break
        except Exception as e:
            log_error("price_tracker.error", f"Error in tracking loop: {e}")
            time.sleep(INTERVAL_SECONDS)  # Wait 5 minutes before retrying
    
    print("👋 5-minute price tracker stopped")


if __name__ == "__main__":
    track_prices()
