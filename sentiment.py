import requests

def get_sentiment_score(symbol):
    twitter_url = f"https://api.twitter.com/2/tweets/search/recent?query={symbol}"
    headers = {"Authorization": "Bearer AAAAAAAAAAAAAAAAAAAAAGf73AEAAAAAeFem8YMiP%2FzkkS8XSf77eQWw1iA%3Dadc8H9TGZkwpCvrIZTwdSbjU98d4egoWsb0DPEoc8beoQgZFDE"}
    try:
        response = requests.get(twitter_url, headers=headers)
        data = response.json()
        return len(data.get("data", []))
    except:
        return 0