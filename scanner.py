import requests
from bs4 import BeautifulSoup

def get_new_tokens():
    url = "https://www.geckoterminal.com/network/eth"  # adjust for BSC or others
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    tokens = []
    for row in soup.find_all("tr")[1:6]:
        token = {}
        try:
            token["name"] = row.find_all("td")[1].text.strip()
            token["symbol"] = row.find_all("td")[2].text.strip()
            token["volume"] = float(row.find_all("td")[4].text.replace('$','').replace(',',''))
            tokens.append(token)
        except:
            continue
    return tokens