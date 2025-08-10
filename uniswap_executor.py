from web3 import Web3
from eth_account import Account
import json
import yaml
import time
from utils import fetch_token_price_usd
from telegram_bot import send_telegram_message
from trade_logger import log_trade

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

w3 = Web3(Web3.HTTPProvider(config["infura_url"]))
wallet_address = Web3.to_checksum_address(config["wallet_address"])
private_key = config["private_key"]
slippage = config["slippage"]
gas_price = w3.to_wei("30", "gwei")

# Load ABI
with open("uniswap_router_abi.json", "r") as f:
    router_abi = json.load(f)

router_address = Web3.to_checksum_address("0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D")
router = w3.eth.contract(address=router_address, abi=router_abi)

WETH_ADDRESS = Web3.to_checksum_address("0xC02aaA39b223FE8D0A0E5C4F27eAD9083C756Cc2")

def buy_token(token_address, dry_run=False):
    token_address = Web3.to_checksum_address(token_address)
    price_usd = fetch_token_price_usd(token_address)

    eth_amount = config["trade_amount_usd"] / price_usd
    eth_amount_wei = w3.to_wei(eth_amount, "ether")

    amount_out_min = 0
    deadline = int(time.time()) + 600

    txn = router.functions.swapExactETHForTokens(
        amount_out_min,
        [WETH_ADDRESS, token_address],
        wallet_address,
        deadline
    ).build_transaction({
        "from": wallet_address,
        "value": eth_amount_wei,
        "gas": 300000,
        "gasPrice": gas_price,
        "nonce": w3.eth.get_transaction_count(wallet_address),
        "chainId": 1,
    })

    if dry_run:
        print("🚫 DRY RUN: Buy transaction built but not sent.")
        print(txn)
        return

    signed_txn = w3.eth.account.sign_transaction(txn, private_key=private_key)
    tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
    tx_link = f"https://etherscan.io/tx/{tx_hash.hex()}"

    message = (
        f"✅ *Buy Executed*\n"
        f"🔹 Token: `{token_address}`\n"
        f"💵 Price: ${price_usd:.6f}\n"
        f"🔗 [TX Link]({tx_link})"
    )
    send_telegram_message(message, markdown=True)

    with open("entry_price.txt", "w") as f:
        f.write(f"{token_address},{price_usd:.10f}")

    log_trade(token_address, "BUY", price_usd, tx_hash.hex())

    return tx_hash.hex()

def sell_token(token_address):
    from_token = Web3.to_checksum_address(token_address)
    token_contract = w3.eth.contract(address=from_token, abi=json.loads('[{"constant":true,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"type":"function"},{"constant":true,"inputs":[{"name":"owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"type":"function"},{"constant":false,"inputs":[{"name":"spender","type":"address"},{"name":"amount","type":"uint256"}],"name":"approve","outputs":[{"name":"","type":"bool"}],"type":"function"}]'))

    decimals = token_contract.functions.decimals().call()
    balance = token_contract.functions.balanceOf(wallet_address).call()

    try:
        # Approve Uniswap
        approve_txn = token_contract.functions.approve(router_address, balance).build_transaction({
            "from": wallet_address,
            "gas": 100000,
            "gasPrice": gas_price,
            "nonce": w3.eth.get_transaction_count(wallet_address),
            "chainId": 1,
        })

        signed_approve = w3.eth.account.sign_transaction(approve_txn, private_key=private_key)
        approve_hash = w3.eth.send_raw_transaction(signed_approve.rawTransaction)
        w3.eth.wait_for_transaction_receipt(approve_hash, timeout=60)

        # Sell
        amount_out_min = 0
        deadline = int(time.time()) + 600

        txn = router.functions.swapExactTokensForETH(
            balance,
            amount_out_min,
            [from_token, WETH_ADDRESS],
            wallet_address,
            deadline
        ).build_transaction({
            "from": wallet_address,
            "gas": 300000,
            "gasPrice": gas_price,
            "nonce": w3.eth.get_transaction_count(wallet_address),
            "chainId": 1,
        })

        signed_txn = w3.eth.account.sign_transaction(txn, private_key=private_key)
        tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)

        # Success!
        price_usd = fetch_token_price_usd(token_address)
        pnl = None
        try:
            with open("entry_price.txt", "r") as f:
                data = f.read().strip().split(",")
                if len(data) == 2 and data[0].lower() == token_address.lower():
                    entry_price = float(data[1])
                    pnl = ((price_usd - entry_price) / entry_price) * 100
        except:
            pass

        log_trade(token_address, "SELL", price_usd, tx_hash.hex(), pnl)

        tx_link = f"https://etherscan.io/tx/{tx_hash.hex()}"
        message = (
            f"✅ *Sell Executed*\n"
            f"🔹 Token: `{token_address}`\n"
            f"💵 Price: ${price_usd:.6f}\n"
            f"📈 PnL: {pnl:.2f}%\n"
            f"🔗 [TX Link]({tx_link})"
        )
        send_telegram_message(message, markdown=True)

        return tx_hash.hex()

    except Exception as e:
        # Honeypot detected
        message = (
            f"🔴 *Honeypot Detected!*\n"
            f"❌ Sell Failed for `{token_address}`\n"
            f"💣 Error: `{str(e)}`"
        )
        send_telegram_message(message, markdown=True)
        print(f"[!] Honeypot detected: {token_address} | Error: {e}")
        return None