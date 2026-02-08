#!/usr/bin/env python3
"""
Tracked Token Audit — 48h summary sent to Telegram.

Scans the last ~48 hours of practical_sustainable.log and counts
per-token: buy successes, already-held blocks, no-lane (RSI/momentum)
blocks, and AI hold/filter blocks. Sends a formatted summary to Telegram
highlighting tokens with zero buy opportunities.

Usage:
    python scripts/audit_tracked_tokens.py          # send to Telegram
    python scripts/audit_tracked_tokens.py --dry-run  # print to stdout only
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
# ~200k lines covers roughly 48h based on observed log growth
TAIL_LINES = 200_000


def _tail_lines(path: Path, n: int) -> list[str]:
    """Read the last n lines of a file efficiently using tail."""
    try:
        result = subprocess.run(
            ["tail", "-n", str(n), str(path)],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout.splitlines()
    except Exception:
        # Fallback: read entire file (slow but works)
        with open(path, "r", errors="replace") as f:
            lines = f.readlines()
        return [l.rstrip("\n") for l in lines[-n:]]


def audit(lines: list[str]) -> dict[str, dict[str, int]]:
    """Count events per tracked token across log lines."""
    symbols = list(TRACKED_TOKENS.keys())

    # Pre-compile one regex per metric (case-insensitive)
    patterns = {}
    for sym in symbols:
        s = re.escape(sym)
        patterns[sym] = {
            "buy_success": re.compile(
                rf"trade\.buy\.success:.*'symbol':\s*'{s}'|buy.*success.*{s}",
                re.IGNORECASE,
            ),
            "already_held": re.compile(
                rf"risk_gate_blocked.*'symbol':\s*'{s}'.*token_already_held|token_already_held.*{s}",
                re.IGNORECASE,
            ),
            "no_lane": re.compile(
                rf"no.lane.*{s}|No entry lane.*{s}", re.IGNORECASE
            ),
            "ai_hold": re.compile(
                rf"recommendation_check.*{s}.*passes_ai_filters=False|action=hold.*{s}.*passes_ai_filters=False",
                re.IGNORECASE,
            ),
        }

    counts: dict[str, dict[str, int]] = {
        sym: {"buy_success": 0, "already_held": 0, "no_lane": 0, "ai_hold": 0}
        for sym in symbols
    }

    for line in lines:
        line_lower = line.lower()
        for sym in symbols:
            if sym not in line_lower:
                continue
            for metric, rx in patterns[sym].items():
                if rx.search(line):
                    counts[sym][metric] += 1

    return counts


def _verdict(row: dict[str, int]) -> str:
    """Classify token health based on counts."""
    if row["buy_success"] >= 4:
        return "Active"
    if row["buy_success"] >= 1:
        return "Marginal"
    if row["no_lane"] > 0 or row["ai_hold"] > 0:
        return "REMOVE?"
    return "REMOVE?"


def _format_token_line(sym: str, row: dict[str, int]) -> str:
    """Format a single token line with key stats."""
    buys = row["buy_success"]
    held = row["already_held"]
    no_lane = row["no_lane"]
    ai_hold = row["ai_hold"]

    buy_str = f"{buys} buy" + ("s" if buys != 1 else "")

    # Show the most relevant blocking reason
    parts = [buy_str]
    if held > 0:
        parts.append(f"{held} held")
    if no_lane > 0:
        parts.append(f"{no_lane} no-lane")
    if ai_hold > 0:
        parts.append(f"{ai_hold} AI holds")

    return f"  {sym} — {' | '.join(parts)}"


def format_message(counts: dict[str, dict[str, int]]) -> str:
    """Build the Telegram message string with grouped sections."""
    # Classify tokens into groups
    active = []
    marginal = []
    remove = []

    for sym, row in counts.items():
        v = _verdict(row)
        if v == "Active":
            active.append((sym, row))
        elif v == "Marginal":
            marginal.append((sym, row))
        else:
            remove.append((sym, row))

    # Sort within each group by buy_success descending
    active.sort(key=lambda kv: -kv[1]["buy_success"])
    marginal.sort(key=lambda kv: -kv[1]["buy_success"])
    remove.sort(key=lambda kv: kv[0])  # alphabetical

    lines = [f"\U0001F4CA Tracked Token Audit (48h)", ""]

    if active:
        lines.append(f"\u2705 ACTIVE ({len(active)} tokens)")
        for sym, row in active:
            lines.append(_format_token_line(sym, row))
        lines.append("")

    if marginal:
        lines.append(f"\u26A0\uFE0F MARGINAL ({len(marginal)} tokens)")
        for sym, row in marginal:
            lines.append(_format_token_line(sym, row))
        lines.append("")

    if remove:
        lines.append(f"\U0001F6AB NO BUY OPPORTUNITIES ({len(remove)} tokens)")
        for sym, row in remove:
            lines.append(_format_token_line(sym, row))
        lines.append("")

    # Summary footer
    removal_names = [sym for sym, _ in remove]
    if removal_names:
        lines.append(f"Remove candidates: {', '.join(removal_names)}")
    else:
        lines.append("All tracked tokens had buy opportunities.")

    return "\n".join(lines)


def main():
    dry_run = "--dry-run" in sys.argv

    if not LOG_FILE.exists():
        print(f"Log file not found: {LOG_FILE}")
        sys.exit(1)

    print(f"Reading last {TAIL_LINES} lines from {LOG_FILE} ...")
    lines = _tail_lines(LOG_FILE, TAIL_LINES)
    print(f"Loaded {len(lines)} lines. Auditing {len(TRACKED_TOKENS)} tracked tokens ...")

    counts = audit(lines)
    message = format_message(counts)

    print(message)

    if not dry_run:
        print("\nSending to Telegram ...")
        ok = send_telegram_message(message, deduplicate=False, message_type="audit")
        print("Sent!" if ok else "Failed to send.")
    else:
        print("\n(dry-run — not sending to Telegram)")


if __name__ == "__main__":
    main()
