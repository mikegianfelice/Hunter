import csv
import os
from datetime import datetime

LOG_FILE = "trade_history.csv"

def log_trade(token_address, action, price_usd, tx_hash=None, pnl_percent=None):
    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Timestamp",
                "Token",
                "Action",
                "Price (USD)",
                "PnL (%)",
                "TX Link"
            ])

        writer.writerow([
            datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            token_address,
            action,
            f"{price_usd:.6f}",
            f"{pnl_percent:.2f}%" if pnl_percent is not None else "",
            f"https://etherscan.io/tx/{tx_hash}" if tx_hash else ""
        ])