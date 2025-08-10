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
        elapsed_hours = (now - log[token_address]) / 3600
        return elapsed_hours < COOLDOWN_HOURS
    return False

def update_cooldown_log(token_address):
    log = load_cooldown_log()
    log[token_address] = time.time()
    save_cooldown_log(log)