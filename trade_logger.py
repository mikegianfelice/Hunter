# trade_logger.py
import csv, os
from datetime import datetime, timezone

LOG_FILE = "trade_log.csv"

FIELDS = [
    "ts", "event", "symbol", "address",
    "entry_price", "exit_price",
    "pnl_pct", "hold_secs",
    "tp", "sl", "trail_drop",
    "volume24h", "liquidity",
    "sent_score", "sent_mentions",
    "reason"  # e.g., "buy", "tp", "sl", "trail", "manual"
]

def _open_writer():
    exists = os.path.isfile(LOG_FILE)
    f = open(LOG_FILE, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if not exists:
        w.writeheader()
    return f, w

def log_buy(token: dict, entry_price: float, tp: float, sl: float, trail_drop: float,
            sent_score: float = None, sent_mentions: int = None):
    f, w = _open_writer()
    row = dict(ts=_now(), event="BUY",
               symbol=token.get("symbol"), address=token.get("address"),
               entry_price=round(entry_price, 10), exit_price="",
               pnl_pct="", hold_secs="",
               tp=tp, sl=sl, trail_drop=trail_drop,
               volume24h=token.get("volume24h"), liquidity=token.get("liquidity"),
               sent_score=sent_score, sent_mentions=sent_mentions, reason="buy")
    w.writerow(row); f.close()

def log_sell(token: dict, entry_price: float, exit_price: float, reason: str,
             opened_at_iso: str = None, tp: float = None, sl: float = None, trail_drop: float = None,
             sent_score: float = None, sent_mentions: int = None):
    pnl = (exit_price - entry_price) / entry_price * 100 if entry_price else ""
    hold_secs = _hold_secs(opened_at_iso) if opened_at_iso else ""
    f, w = _open_writer()
    row = dict(ts=_now(), event="SELL",
               symbol=token.get("symbol"), address=token.get("address"),
               entry_price=round(entry_price, 10), exit_price=round(exit_price, 10),
               pnl_pct=round(pnl, 4) if pnl != "" else "",
               hold_secs=hold_secs,
               tp=tp, sl=sl, trail_drop=trail_drop,
               volume24h=token.get("volume24h"), liquidity=token.get("liquidity"),
               sent_score=sent_score, sent_mentions=sent_mentions, reason=reason)
    w.writerow(row); f.close()

def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def _hold_secs(opened_at_iso):
    try:
        opened = datetime.strptime(opened_at_iso, "%Y-%m-%d %H:%M:%S")
        return int((datetime.now(timezone.utc) - opened).total_seconds())
    except Exception:
        return ""