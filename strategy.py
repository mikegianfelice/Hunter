import pandas as pd
import requests

def fetch_token_prices(token_address, interval="5m", limit=50):
    # NOTE: This is a placeholder. Replace with your real price data source.
    # You can use DexTools, Moralis, or your own Uniswap graph integration.
    
    print(f"🔍 Fetching mock price data for {token_address}")
    
    try:
        # Mock data: random walk prices for testing
        import numpy as np
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(limit)) + 100

        df = pd.DataFrame(prices, columns=["price"])
        return df
    except Exception as e:
        print(f"❌ Failed to fetch price data: {e}")
        return None

def calculate_indicators(token_address):
    df = fetch_token_prices(token_address)

    if df is None or df.empty:
        print("⚠️ Price data unavailable.")
        return None

    try:
        df["EMA10"] = df["price"].ewm(span=10).mean()
        df["EMA20"] = df["price"].ewm(span=20).mean()
        df["returns"] = df["price"].pct_change()
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"⚠️ Error calculating indicators: {e}")
        return None

def check_buy_signal(token_address):
    df = calculate_indicators(token_address)
    if df is None or df.empty:
        return False

    # Simple crossover strategy
    try:
        if (
            df["EMA10"].iloc[-1] > df["EMA20"].iloc[-1]
            and df["EMA10"].iloc[-2] <= df["EMA20"].iloc[-2]
        ):
            print("✅ BUY SIGNAL TRIGGERED")
            return True
        else:
            print("📉 No buy signal")
            return False
    except Exception as e:
        print(f"❌ Error in buy signal logic: {e}")
        return False

def get_dynamic_take_profit(token_address):
    df = calculate_indicators(token_address)
    if df is None or df.empty:
        return 0.3  # fallback TP = 30%

    # Use recent volatility to determine dynamic TP
    try:
        recent_volatility = df["returns"].std() * 100  # %
        dynamic_tp = min(max(recent_volatility * 2, 0.1), 1.0)
        return dynamic_tp  # e.g., 0.25 = 25%
    except Exception as e:
        print(f"⚠️ Failed to calculate dynamic TP: {e}")
        return 0.3