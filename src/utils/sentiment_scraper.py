import requests
from bs4 import BeautifulSoup
import time

# Weighted bullish keywords (enhanced)
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
    "ape": 2,
    "diamond hands": 2,
    "hodl": 1,
    "to the moon": 3,
    "breakout": 1,
    "surge": 1,
    "rally": 1,
    "bull run": 2,
    "green": 1,
    "profit": 1,
    "gains": 1
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

def scrape_twitter_alternative(symbol):
    """Try multiple Twitter alternatives for sentiment data"""
    # Multiple Twitter alternatives to try (updated with more reliable instances)
    alternatives = [
        "https://nitter.net",
        "https://nitter.1d4.us", 
        "https://nitter.kavin.rocks",
        "https://nitter.unixfox.eu",
        "https://nitter.privacydev.net",
        "https://nitter.fdn.fr",
        "https://nitter.actionsack.com"
    ]
    
    query = f"/search?f=tweets&q=%24{symbol}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    mentions = 0
    score = 0
    
    for base_url in alternatives:
        try:
            url = base_url + query
            response = requests.get(url, headers=headers, timeout=8)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                tweets = soup.find_all("div", class_="timeline-item")
                
                for tweet in tweets:
                    content = tweet.text
                    tweet_score, passed = score_content(content)
                    if passed:
                        score += tweet_score
                        mentions += 1
                    else:
                        return {"score": 0, "mentions": mentions, "source": "twitter", "status": "blocked by FUD"}
                
                print(f"✅ Twitter sentiment from {base_url}")
                return {"score": score, "mentions": mentions, "source": "twitter", "status": "ok"}
                
        except Exception as e:
            print(f"⚠️ Twitter alternative {base_url} failed: {e}")
            continue
    
    # If all Twitter alternatives fail, return neutral sentiment marked as unavailable
    print(f"❌ All Twitter alternatives failed for {symbol} - returning neutral sentiment (data unavailable)")
    return {"score": 50, "mentions": 0, "source": "twitter", "status": "unavailable"}

def scrape_reddit(symbol):
    """Scrape Reddit for sentiment data with better error handling"""
    query = f"https://www.reddit.com/search/?q={symbol}&sort=new"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    mentions = 0
    score = 0
    
    try:
        response = requests.get(query, headers=headers, timeout=10)
        if response.status_code == 200:
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
        else:
            print(f"⚠️ Reddit returned status {response.status_code}")
            return {"score": 0, "mentions": 0, "source": "reddit", "status": "error"}
            
    except Exception as e:
        print(f"❌ Reddit scrape failed for {symbol}: {e}")
        return {"score": 0, "mentions": 0, "source": "reddit", "status": "error"}

def get_fallback_sentiment(symbol):
    """Generate fallback sentiment based on symbol characteristics"""
    symbol_lower = symbol.lower()
    
    # Check for bullish indicators in symbol name
    bullish_indicators = ["moon", "pump", "gem", "diamond", "gold", "bull", "rocket", "mars"]
    bearish_indicators = ["dump", "bear", "crash", "dead", "scam"]
    
    score = 50  # Base neutral score
    mentions = 3  # Base mentions
    
    for indicator in bullish_indicators:
        if indicator in symbol_lower:
            score += 15
            mentions += 1
    
    for indicator in bearish_indicators:
        if indicator in symbol_lower:
            score -= 20
            mentions += 1
    
    return {
        "score": max(0, min(score, 100)),
        "mentions": max(1, mentions),
        "source": "fallback",
        "status": "fallback"
    }

def get_sentiment_score(token):
    # Extract symbol from token dict or use token directly if it's a string
    if isinstance(token, dict):
        symbol = token.get("symbol", "UNKNOWN")
    else:
        symbol = str(token)
    
    # Add small deterministic delay to avoid rate limiting without randomness
    time.sleep(1.0)
    
    try:
        twitter = scrape_twitter_alternative(symbol)
        reddit = scrape_reddit(symbol)

        total_mentions = twitter["mentions"] + reddit["mentions"]
        raw_score = twitter["score"] + reddit["score"]

        # If both sources failed, use fallback
        if twitter["status"] == "fallback" and reddit["status"] == "error":
            print(f"🔄 Using fallback sentiment for {symbol}")
            fallback = get_fallback_sentiment(symbol)
            return {
                "symbol": symbol,
                "mentions": fallback["mentions"],
                "score": fallback["score"],
                "source": "fallback",
                "status": "fallback"
            }
        
        # If we have very low mentions and scores, use fallback
        if total_mentions < 2 and raw_score < 10:
            print(f"🔄 Low sentiment data for {symbol}, using fallback")
            fallback = get_fallback_sentiment(symbol)
            return {
                "symbol": symbol,
                "mentions": fallback["mentions"],
                "score": fallback["score"],
                "source": "fallback",
                "status": "fallback"
            }

        # Normalize score between 0–100
        normalized_score = max(0, min(raw_score * 5, 100))

        sentiment = {
            "symbol": symbol,
            "mentions": total_mentions,
            "score": normalized_score,
            "source": f"{twitter['source']}+{reddit['source']}",
            "status": "ok" if twitter["status"] == "ok" and reddit["status"] == "ok" else "partial"
        }
        
        return sentiment
        
    except Exception as e:
        print(f"⚠️ Sentiment analysis failed for {symbol}: {e}")
        fallback = get_fallback_sentiment(symbol)
        return {
            "symbol": symbol,
            "mentions": fallback["mentions"],
            "score": fallback["score"],
            "source": "fallback",
            "status": "fallback"
        }