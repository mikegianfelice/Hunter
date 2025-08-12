import requests
from bs4 import BeautifulSoup

# Weighted bullish keywords
BULLISH_KEYWORDS = {
    "100x": 3,
    "moon": 2,
    "pump": 2,
    "gem": 2,
    "bullish": 1,
    "buy": 1,
    "alpha": 1,
    "entry": 1,
    "launch": 1,
    "whale": 1,
    "undervalued": 1,
    "next big": 2,
    "going parabolic": 3,
    "ape": 2
}

# FUD keywords to disqualify posts
FUD_KEYWORDS = [
    "rug", "scam", "exit", "hack", "dump", "rekt", "liquidated", "dead coin", "honeypot", "exit scam"
]

def score_content(content):
    score = 0
    lowered = content.lower()
    if any(fud in lowered for fud in FUD_KEYWORDS):
        return -100, False
    for word, value in BULLISH_KEYWORDS.items():
        if word in lowered:
            score += value
    return score, True

def scrape_nitter(symbol):
    query = f"https://nitter.net/search?f=tweets&q=%24{symbol}"
    headers = {"User-Agent": "Mozilla/5.0"}
    mentions = 0
    score = 0
    try:
        response = requests.get(query, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        tweets = soup.find_all("div", class_="timeline-item")
        for tweet in tweets:
            content = tweet.text
            tweet_score, passed = score_content(content)
            if passed:
                score += tweet_score
                mentions += 1
            else:
                return {"score": 0, "mentions": mentions, "source": "nitter", "status": "blocked by FUD"}
        return {"score": score, "mentions": mentions, "source": "nitter", "status": "ok"}
    except Exception as e:
        print(f"❌ Twitter scrape failed for {symbol}: {e}")
        return {"score": 0, "mentions": 0, "source": "nitter", "status": "error"}

def scrape_reddit(symbol):
    query = f"https://www.reddit.com/search/?q={symbol}&sort=new"
    headers = {"User-Agent": "Mozilla/5.0"}
    mentions = 0
    score = 0
    try:
        response = requests.get(query, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        posts = soup.find_all("div", attrs={"data-testid": "post-container"})
        for post in posts:
            content = post.text
            post_score, passed = score_content(content)
            if passed:
                score += post_score
                mentions += 1
            else:
                return {"score": 0, "mentions": mentions, "source": "reddit", "status": "blocked by FUD"}
        return {"score": score, "mentions": mentions, "source": "reddit", "status": "ok"}
    except Exception as e:
        print(f"❌ Reddit scrape failed for {symbol}: {e}")
        return {"score": 0, "mentions": 0, "source": "reddit", "status": "error"}

def get_sentiment_score(token):
    # Extract symbol from token dict or use token directly if it's a string
    if isinstance(token, dict):
        symbol = token.get("symbol", "UNKNOWN")
    else:
        symbol = str(token)
    
    twitter = scrape_nitter(symbol)
    reddit = scrape_reddit(symbol)

    total_mentions = twitter["mentions"] + reddit["mentions"]
    raw_score = twitter["score"] + reddit["score"]

    # Normalize score between 0–100
    normalized_score = max(0, min(raw_score * 5, 100))

    sentiment = {
        "symbol": symbol,
        "mentions": total_mentions,
        "score": normalized_score,
        "source": "nitter+reddit",
        "status": "ok" if "blocked" not in twitter["status"] and "blocked" not in reddit["status"] else "blocked"
    }

    print(f"🧠 Sentiment for ${symbol}: {sentiment}")
    return sentiment