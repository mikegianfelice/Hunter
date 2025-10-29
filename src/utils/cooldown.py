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
        print(f"🚨 Token {token_address} failed {failure_count} times - auto-delisting")
        try:
            from strategy import _add_to_delisted_tokens
            _add_to_delisted_tokens(token_address, "UNKNOWN", f"Auto-delisted after {failure_count} failures")
        except Exception as e:
            print(f"⚠️ Failed to auto-delist token: {e}")
    
    save_cooldown_log(log)