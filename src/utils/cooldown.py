import json
import time
import os

COOLDOWN_FILE = "cooldown_log.json"
COOLDOWN_HOURS = 1  # You can change this to lower value for 

def load_cooldown_log():
    if os.path.exists(COOLDOWN_FILE):
        with open(COOLDOWN_FILE, "r") as f:
            return json.load(f)
    return {}

def save_cooldown_log(log):
    with open(COOLDOWN_FILE, "w") as f:
        json.dump(log, f, indent=2)

def is_token_on_cooldown(token_address):
    log = load_cooldown_log()
    now = time.time()
    if token_address in log:
        # Handle both old and new formats
        if isinstance(log[token_address], dict):
            last_failure = log[token_address]["last_failure"]
        else:
            # Old format - convert timestamp to new format
            last_failure = log[token_address]
            log[token_address] = {"last_failure": last_failure, "failure_count": 1}
            save_cooldown_log(log)
        
        elapsed_hours = (now - last_failure) / 3600
        return elapsed_hours < COOLDOWN_HOURS
    return False

def update_cooldown_log(token_address):
    log = load_cooldown_log()
    now = time.time()
    
    # Track failure count
    if token_address not in log:
        log[token_address] = {"last_failure": now, "failure_count": 1}
    else:
        if isinstance(log[token_address], dict):
            log[token_address]["last_failure"] = now
            log[token_address]["failure_count"] = log[token_address].get("failure_count", 0) + 1
        else:
            # Convert old format to new format
            log[token_address] = {"last_failure": now, "failure_count": 2}
    
    # Check if token should be automatically delisted (5+ failures)
    failure_count = log[token_address]["failure_count"]
    if failure_count >= 5:
        print(f"🚨 Token {token_address} failed {failure_count} times - checking if should auto-delist")
        
        # CRITICAL SAFEGUARD: Before auto-delisting, verify the token isn't in wallet
        # If it has balance, it's clearly not delisted - just a price API issue
        try:
            # Check if token is in open positions (should never delist tracked positions)
            if os.path.exists("data/open_positions.json"):
                with open("data/open_positions.json", "r") as f:
                    positions = json.load(f)
                    
                # Check if any position matches this token address
                for position_key, position_data in positions.items():
                    if isinstance(position_data, dict):
                        addr = position_data.get("address", position_key)
                    else:
                        addr = position_key
                    
                    # Extract address from composite key if needed
                    if "_" in position_key and isinstance(position_data, dict):
                        if not position_data.get("address"):
                            addr = position_key.split("_")[0]
                    
                    if addr.lower() == token_address.lower():
                        print(f"🛡️ Token {token_address} is in open positions - skipping auto-delist to prevent false positive")
                        # Reset failure count to give it more time
                        log[token_address]["failure_count"] = max(0, failure_count - 2)
                        save_cooldown_log(log)
                        return  # Don't delist - position is tracked
            
            # Check wallet balance (for Solana tokens, check if balance > 0)
            try:
                from src.monitoring.monitor_position import _check_token_balance_on_chain
                
                # Try to determine chain - for Solana addresses (44 chars), check Solana balance
                chain_id = "solana" if len(token_address) == 44 else "ethereum"
                balance = _check_token_balance_on_chain(token_address, chain_id)
                
                if balance == -1.0:
                    # Balance check failed - can't verify, so don't mark as delisted
                    print(f"⏸️  Cannot verify delisting: balance check failed. Keeping token active.")
                    log[token_address]["failure_count"] = max(0, failure_count - 2)
                    save_cooldown_log(log)
                    return  # Don't delist - verification failed
                elif balance > 0:
                    # Token exists in wallet - definitely not delisted, just price API issue
                    print(f"✅ Token has balance ({balance:.6f}) - not delisted. Price API issue detected.")
                    # Reset failure count since we know the token exists
                    log[token_address]["failure_count"] = 0
                    save_cooldown_log(log)
                    return  # Don't delist - token has balance
            except Exception as e:
                print(f"⚠️ Balance check error for {token_address}: {e}")
                # On error, don't delist to be safe
                log[token_address]["failure_count"] = max(0, failure_count - 2)
                save_cooldown_log(log)
                return
            
        except Exception as e:
            print(f"⚠️ Error checking safeguards before auto-delist: {e}")
            # On error, don't delist to be safe
            log[token_address]["failure_count"] = max(0, failure_count - 2)
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