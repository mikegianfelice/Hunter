import json
import os

BLACKLIST_FILE = "blacklist.json"

def load_blacklist():
    if not os.path.exists(BLACKLIST_FILE):
        return []
    with open(BLACKLIST_FILE, "r") as file:
        try:
            return json.load(file)
        except json.JSONDecodeError:
            return []

def save_blacklist(blacklist):
    with open(BLACKLIST_FILE, "w") as file:
        json.dump(list(set(blacklist)), file, indent=2)

def add_to_blacklist(token_address):
    blacklist = load_blacklist()
    if token_address not in blacklist:
        blacklist.append(token_address)
        save_blacklist(blacklist)
        print(f"🛑 Token blacklisted: {token_address}")

def is_blacklisted(token_address):
    blacklist = load_blacklist()
    return token_address in blacklist