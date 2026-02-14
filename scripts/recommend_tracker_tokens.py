#!/usr/bin/env python3
"""
Price Tracker Token Recommender — daily log analysis.

Scans the last ~48 hours of practical_sustainable.log and identifies tokens
that are getting consistent buy/weak_buy recommendations but are NOT currently
in the minute price tracker. Also flags tracked tokens with zero activity.

Sends a formatted recommendation to Telegram.

Usage:
    python scripts/recommend_tracker_tokens.py            # send to Telegram
    python scripts/recommend_tracker_tokens.py --dry-run  # print to stdout only
"""

import os
import re
import sys
import subprocess
from collections import defaultdict
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.storage.minute_price_tracker import TRACKED_TOKENS
from src.monitoring.telegram_bot import send_telegram_message

LOG_FILE = PROJECT_ROOT / "scripts" / "practical_sustainable.log"
TAIL_LINES = 200_000  # ~48h of logs


def _tail_lines(path: Path, n: int) -> list[str]:
    """Read the last n lines of a file efficiently using tail."""
    try:
        result = subprocess.run(
            ["tail", "-n", str(n), str(path)],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout.splitlines()
    except Exception:
        with open(path, "r", errors="replace") as f:
            lines = f.readlines()
        return [l.rstrip("\n") for l in lines[-n:]]


# --- Regex patterns (compiled once) ---
RE_RECOMMENDATION = re.compile(
    r"recommendation_check: Token (\S+): action=(buy|weak_buy).*passes_ai_filters=True",
    re.IGNORECASE,
)
RE_AI_DOWNGRADED = re.compile(
    r"recommendation_check: Token (\S+): action=skip.*original=(buy|weak_buy).*passes_ai_filters=False",
    re.IGNORECASE,
)
RE_CONCENTRATION_PASS = re.compile(
    r"trading\.holder_concentration: Holder concentration for (\S+):.*below threshold",
    re.IGNORECASE,
)
RE_CONCENTRATION_BLOCK = re.compile(
    r"holder_concentration_blocked: Token (\S+) BLOCKED",
    re.IGNORECASE,
)
RE_LANE_SELECTED = re.compile(
    r"lane_selected:.*for (\S+)",
    re.IGNORECASE,
)
RE_NO_LANE = re.compile(
    r"(?:strategy\.lane\.no_lane|strategy\.buy\.no_lane):.*(?:for |qualifies for )(\S+)",
    re.IGNORECASE,
)
RE_CANDLE_BLOCK = re.compile(
    r"candle_validation\.(?:insufficient_candles|no_candles):.*Token (\S+) blocked",
    re.IGNORECASE,
)
RE_TRADE_EXECUTED = re.compile(
    r"trade_execution_start: Starting trade execution for (\S+)",
    re.IGNORECASE,
)
RE_TRADE_STATE_BLOCKED = re.compile(
    r"token_trade_state_blocked.*symbol.*?(\w+)",
    re.IGNORECASE,
)


def scan_logs(lines: list[str]) -> dict:
    """Parse log lines and collect per-token stats."""
    stats = defaultdict(lambda: {
        "recommendations": 0,
        "ai_downgraded": 0,
        "concentration_pass": 0,
        "concentration_block": 0,
        "lane_selected": 0,
        "no_lane": 0,
        "candle_block": 0,
        "trade_executed": 0,
        "trade_state_blocked": 0,
    })

    for line in lines:
        # Buy/weak_buy recommendations that pass AI filters
        m = RE_RECOMMENDATION.search(line)
        if m:
            stats[m.group(1)]["recommendations"] += 1
            continue

        # AI downgraded to skip
        m = RE_AI_DOWNGRADED.search(line)
        if m:
            stats[m.group(1)]["ai_downgraded"] += 1
            continue

        # Holder concentration pass
        m = RE_CONCENTRATION_PASS.search(line)
        if m:
            stats[m.group(1)]["concentration_pass"] += 1
            continue

        # Holder concentration block
        m = RE_CONCENTRATION_BLOCK.search(line)
        if m:
            stats[m.group(1)]["concentration_block"] += 1
            continue

        # Lane selected (entry approved)
        m = RE_LANE_SELECTED.search(line)
        if m:
            stats[m.group(1)]["lane_selected"] += 1
            continue

        # No lane
        m = RE_NO_LANE.search(line)
        if m:
            # Extract just the token name (strip trailing punctuation)
            token = m.group(1).rstrip(":,")
            stats[token]["no_lane"] += 1
            continue

        # Candle validation block
        m = RE_CANDLE_BLOCK.search(line)
        if m:
            stats[m.group(1)]["candle_block"] += 1
            continue

        # Actual trade execution
        m = RE_TRADE_EXECUTED.search(line)
        if m:
            # Token name may have trailing whitespace in log (e.g. "Fartcoin ")
            stats[m.group(1).strip()]["trade_executed"] += 1
            continue

        # Token trade state blocked
        m = RE_TRADE_STATE_BLOCKED.search(line)
        if m:
            stats[m.group(1)]["trade_state_blocked"] += 1

    return dict(stats)


def _is_tracked(symbol: str) -> bool:
    """Check if a symbol is already in the price tracker (case-insensitive)."""
    tracked_lower = {k.lower() for k in TRACKED_TOKENS.keys()}
    return symbol.lower().strip() in tracked_lower


def classify_tokens(stats: dict) -> dict:
    """Classify tokens into recommendation categories."""
    add_candidates = []      # Should add to tracker
    already_tracked = []     # Already tracked, with activity
    blocked_tokens = []      # Blocked by concentration (tracker won't help)
    remove_candidates = []   # Tracked but zero activity

    for sym, s in stats.items():
        rec = s["recommendations"]
        conc_pass = s["concentration_pass"]
        conc_block = s["concentration_block"]
        lane = s["lane_selected"]
        no_lane = s["no_lane"]
        candle = s["candle_block"]
        executed = s["trade_executed"]
        ai_down = s["ai_downgraded"]

        tracked = _is_tracked(sym)

        # Skip tokens with no buy recommendations at all
        if rec == 0 and ai_down == 0:
            continue

        # Tokens blocked by holder concentration more than half the time — tracker won't help
        total_conc_checks = conc_pass + conc_block
        if total_conc_checks > 0 and conc_block / total_conc_checks > 0.5:
            blocked_tokens.append((sym, s))
            continue

        # Tokens with too few concentration passes to be reliable (< 3 passes)
        if conc_block > 0 and conc_pass < 3:
            blocked_tokens.append((sym, s))
            continue

        if tracked:
            already_tracked.append((sym, s))
        else:
            # Not tracked — good candidate if:
            #   1. Strong signal: 5+ recs, reliably passes concentration (>= 3 passes)
            #   2. Blocked by candles or no-lane (price data would actually help)
            has_data_gap = candle > 0 or no_lane > 0
            if rec >= 5 and conc_pass >= 3 and has_data_gap:
                add_candidates.append((sym, s))
            elif rec >= 10 and conc_pass >= 5:
                # Very high signal volume even without obvious data gap
                add_candidates.append((sym, s))

    # Check tracked tokens with zero log activity
    tracked_lower = {k.lower() for k in TRACKED_TOKENS.keys()}
    active_lower = {sym.lower().strip() for sym in stats.keys()}
    for t in tracked_lower:
        if t not in active_lower:
            remove_candidates.append(t)

    return {
        "add": sorted(add_candidates, key=lambda x: -x[1]["recommendations"]),
        "tracked_active": sorted(already_tracked, key=lambda x: -x[1]["trade_executed"]),
        "blocked": sorted(blocked_tokens, key=lambda x: -x[1]["recommendations"]),
        "remove": sorted(remove_candidates),
    }


def _fmt_token(sym: str, s: dict) -> str:
    """Format a token stats line."""
    parts = []
    if s["recommendations"]:
        parts.append(f"{s['recommendations']} recs")
    if s["ai_downgraded"]:
        parts.append(f"{s['ai_downgraded']} AI-skip")
    if s["concentration_pass"]:
        parts.append(f"{s['concentration_pass']} conc-pass")
    if s["concentration_block"]:
        parts.append(f"{s['concentration_block']} conc-block")
    if s["lane_selected"]:
        parts.append(f"{s['lane_selected']} lane-ok")
    if s["no_lane"]:
        parts.append(f"{s['no_lane']} no-lane")
    if s["candle_block"]:
        parts.append(f"{s['candle_block']} candle-block")
    if s["trade_executed"]:
        parts.append(f"{s['trade_executed']} trades")
    if s["trade_state_blocked"]:
        parts.append(f"{s['trade_state_blocked']} cooldown-block")
    return f"  {sym} — {' | '.join(parts)}"


def format_message(classified: dict) -> str:
    """Build the Telegram message — only recommendations and removals."""
    add = classified["add"]
    remove = classified["remove"]

    # If nothing to report, say so
    if not add and not remove:
        return "\U0001F50D Tracker Recommendations (48h)\n\nNo changes recommended."

    lines = ["\U0001F50D Tracker Recommendations (48h)", ""]

    # ADD recommendations
    if add:
        lines.append(f"\U0001F195 ADD ({len(add)})")
        for sym, s in add:
            reason = []
            if s["candle_block"] > 0:
                reason.append("needs candle data")
            if s["no_lane"] > 0:
                reason.append("needs VWAP data")
            if s["recommendations"] >= 10:
                reason.append("high signal volume")
            tag = f" [{', '.join(reason)}]" if reason else ""
            lines.append(f"{_fmt_token(sym, s)}{tag}")
        lines.append("")

    # Remove candidates
    if remove:
        lines.append(f"\U0001F5D1 REMOVE ({len(remove)})")
        for sym in remove:
            lines.append(f"  {sym} — zero activity in 48h")
        lines.append("")

    total_tracked = len(TRACKED_TOKENS)
    lines.append(f"Tracking {total_tracked} tokens (~{total_tracked * 288:,} calls/day)")

    return "\n".join(lines)


def main():
    dry_run = "--dry-run" in sys.argv

    if not LOG_FILE.exists():
        print(f"Log file not found: {LOG_FILE}")
        sys.exit(1)

    print(f"Reading last {TAIL_LINES} lines from {LOG_FILE} ...")
    lines = _tail_lines(LOG_FILE, TAIL_LINES)
    print(f"Loaded {len(lines)} lines. Scanning for token activity ...")

    stats = scan_logs(lines)
    print(f"Found {len(stats)} unique tokens in logs.")

    classified = classify_tokens(stats)
    message = format_message(classified)

    print(message)

    if not dry_run:
        print("\nSending to Telegram ...")
        ok = send_telegram_message(message, deduplicate=False, message_type="recommendation")
        print("Sent!" if ok else "Failed to send.")
    else:
        print("\n(dry-run — not sending to Telegram)")


if __name__ == "__main__":
    main()
