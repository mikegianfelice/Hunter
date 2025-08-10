import requests
import time

def check_token_safety(token_address):
    url = f"https://tokensniffer.com/api/v2/tokens/{token_address}"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"❌ TokenSniffer error for {token_address}")
            return False

        data = response.json()

        # Criteria: You can adjust as you wish
        if (
            not data["is_honeypot"]
            and data["liquidity"]["locked"]
            and data["buy_tax"] <= 10
            and data["sell_tax"] <= 10
            and data["score"] >= 50
        ):
            print(f"✅ TokenSniffer approved: {token_address}")
            return True
        else:
            print(f"⚠️ TokenSniffer rejected: {token_address}")
            return False

    except Exception as e:
        print(f"❌ Error checking TokenSniffer for {token_address}: {e}")
        return False