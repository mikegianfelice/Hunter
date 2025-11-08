import time

from src.storage.positions import load_positions as load_positions_store
from src.storage.cooldown import (
    load_cooldown_log as load_cooldown_store,
    save_cooldown_log as save_cooldown_store,
)

COOLDOWN_HOURS = 1  # You can change this to lower value for 


def load_cooldown_log():
    log = load_cooldown_store() or {}
    # Normalize keys to lowercase for consistent lookups
    return {token.lower(): entry for token, entry in log.items()}


def save_cooldown_log(log):
    sanitized = {token.lower(): entry for token, entry in (log or {}).items()}
    save_cooldown_store(sanitized)

def is_token_on_cooldown(token_address):
    log = load_cooldown_log()
    now = time.time()
    key = (token_address or "").lower()
    if key in log:
        entry = log[key]
        if isinstance(entry, dict):
            last_failure = entry.get("last_failure", 0)
        else:
            last_failure = float(entry)
            log[key] = {"last_failure": last_failure, "failure_count": 1}
            save_cooldown_log(log)

        elapsed_hours = (now - last_failure) / 3600
        return elapsed_hours < COOLDOWN_HOURS
    return False


def update_cooldown_log(token_address):
    log = load_cooldown_log()
    now = time.time()
    key = (token_address or "").lower()

    if key not in log:
        log[key] = {"last_failure": now, "failure_count": 1}
    else:
        if isinstance(log[key], dict):
            log[key]["last_failure"] = now
            log[key]["failure_count"] = log[key].get("failure_count", 0) + 1
        else:
            log[key] = {"last_failure": now, "failure_count": 2}

    failure_count = log[key]["failure_count"]
    if failure_count >= 5:
        print(f"🚨 Token {token_address} failed {failure_count} times - checking if should auto-delist")
        
        # CRITICAL SAFEGUARD: Before auto-delisting, verify the token isn't in wallet
        # If it has balance, it's clearly not delisted - just a price API issue
        try:
            positions = load_positions_store()
            for position_key, position_data in positions.items():
                if isinstance(position_data, dict):
                    addr = position_data.get("address", position_key)
                else:
                    addr = position_key

                if "_" in position_key and isinstance(position_data, dict) and not position_data.get("address"):
                    addr = position_key.split("_")[0]

                if addr.lower() == token_address.lower():
                    print(f"🛡️ Token {token_address} is in open positions - skipping auto-delist to prevent false positive")
                    log[key]["failure_count"] = max(0, failure_count - 2)
                    save_cooldown_log(log)
                    return
        except Exception:
            pass

        # Check wallet balance (for Solana tokens, check if balance > 0)
        try:
            from src.monitoring.monitor_position import _check_token_balance_on_chain
            
            # Try to determine chain - for Solana addresses (44 chars), check Solana balance
            chain_id = "solana" if len(token_address) == 44 else "ethereum"
            balance = _check_token_balance_on_chain(token_address, chain_id)
            
            if balance == -1.0:
                # Balance check failed - can't verify, so don't mark as delisted
                print(f"⏸️  Cannot verify delisting: balance check failed. Keeping token active.")
                log[key]["failure_count"] = max(0, failure_count - 2)
                save_cooldown_log(log)
                return  # Don't delist - verification failed
            elif balance > 0:
                # Token exists in wallet - definitely not delisted, just price API issue
                print(f"✅ Token has balance ({balance:.6f}) - not delisted. Price API issue detected.")
                # Reset failure count since we know the token exists
                log[key]["failure_count"] = 0
                save_cooldown_log(log)
                return  # Don't delist - token has balance
        except Exception as e:
            print(f"⚠️ Balance check error for {token_address}: {e}")
            # On error, don't delist to be safe
            log[key]["failure_count"] = max(0, failure_count - 2)
            save_cooldown_log(log)
            return
        
        except Exception as e:
            print(f"⚠️ Error checking safeguards before auto-delist: {e}")
            # On error, don't delist to be safe
            log[key]["failure_count"] = max(0, failure_count - 2)
            save_cooldown_log(log)
            return
        
        # Only auto-delist if balance is 0 AND not in open positions
        print(f"🚨 Token {token_address} failed {failure_count} times - auto-delisting")
        try:
            from src.core.strategy import _add_to_delisted_tokens
            _add_to_delisted_tokens(token_address, "UNKNOWN", f"Auto-delisted after {failure_count} failures")
        except Exception as e:
            print(f"⚠️ Failed to auto-delist token: {e}")
    
    save_cooldown_log(log)