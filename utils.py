import requests

def fetch_token_price_usd(token_address):
    """
    Uses Uniswap V3 subgraph to fetch the latest USD price for a token.
    Returns None if price is not found.
    """
    token_address = token_address.lower()
    query = """
    {
      token(id: "%s") {
        derivedETH
      }
      bundle(id: "1") {
        ethPriceUSD
      }
    }
    """ % token_address

    url = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"
    response = requests.post(url, json={"query": query})

    if response.status_code != 200:
        print(f"⚠️ GraphQL error: {response.status_code}")
        return None

    data = response.json()["data"]
    token_data = data.get("token")
    bundle = data.get("bundle")

    if not token_data or not bundle:
        print("⚠️ Token or ETH price not found")
        return None

    derived_eth = float(token_data["derivedETH"])
    eth_price_usd = float(bundle["ethPriceUSD"])
    return derived_eth * eth_price_usd