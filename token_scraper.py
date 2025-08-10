# token_scraper.py
import requests
from bs4 import BeautifulSoup

def fetch_trending_tokens(limit=10):
    url = "https://www.dextools.io/app/en/ether/pairs"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print("❌ Failed to fetch trending tokens.")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    token_addresses = []

    for a_tag in soup.select("a[href*='/ether/pair-explorer/']")[:limit]:
        href = a_tag.get("href")
        if href:
            parts = href.split("/")
            if len(parts) >= 5:
                token_address = parts[-1]
                if token_address.startswith("0x"):
                    token_addresses.append(token_address)

    print(f"🔁 Found {len(token_addresses)} trending tokens.")
    return token_addresses