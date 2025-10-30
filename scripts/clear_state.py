# clear_state.py
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List

# Files we can safely clear between modes
STATE_JSONS: List[str] = [
    "open_positions.json",     # current positions
    "price_memory.json",       # last-seen prices for momentum
    "cooldown.json",           # token cooldowns
    "data/risk_state.json",         # daily risk counters
]
STATE_MISC: List[str] = [
    ".monitor_lock",
    "system/.monitor_heartbeat",
    "entry_price.txt",
    "blacklist.json",          # optional: comment this out if you want to keep blacklist
]
LOGS_SAFE_TO_ARCHIVE: List[str] = [
    "trending_tokens.csv",
    "trade_log.csv",
]

RUN_STATE = "system/.run_state.json"  # remembers last known mode so we clear only on changes

def _write_json(path: str, data):
    Path(path).write_text(json.dumps(data, indent=2))

def _read_json(path: str):
    try:
        return json.loads(Path(path).read_text() or "{}")
    except Exception:
        return {}

def _archive_file(p: Path, tag: str):
    if not p.exists():
        return
    folder = Path("archives")
    folder.mkdir(exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    dest = folder / f"{p.stem}.{tag}.{ts}{p.suffix}"
    try:
        p.rename(dest)
        print(f"📦 Archived {p.name} → {dest}")
    except Exception as e:
        print(f"⚠️ Could not archive {p}: {e}")

def _clear_json(path: str):
    p = Path(path)
    try:
        p.write_text("{}")
        print(f"🧹 Cleared {p}")
    except Exception as e:
        print(f"⚠️ Could not clear {p}: {e}")

def _remove_file(path: str):
    p = Path(path)
    try:
        if p.exists():
            p.unlink()
            print(f"🗑️ Removed {p}")
    except Exception as e:
        print(f"⚠️ Could not remove {p}: {e}")

def clear_for_live():
    """
    When switching from test_mode=True -> False:
      - Clear volatile JSON state
      - Remove locks/heartbeats
      - Archive CSV logs (don’t delete)
    """
    for f in STATE_JSONS:
        _clear_json(f)
    for f in STATE_MISC:
        _remove_file(f)
    for f in LOGS_SAFE_TO_ARCHIVE:
        _archive_file(Path(f), tag="prelive")

def clear_for_test():
    """
    Optional: when switching from live -> test, you can also clear/trim.
    We keep logs but still wipe volatile state to avoid ghost positions.
    """
    for f in STATE_JSONS:
        _clear_json(f)
    for f in STATE_MISC:
        _remove_file(f)

def remember_mode(is_test_mode: bool):
    _write_json(RUN_STATE, {"test_mode": bool(is_test_mode)})

def last_mode() -> bool:
    """
    Returns last known test_mode (True/False). Defaults to True if unknown.
    """
    data = _read_json(RUN_STATE)
    return bool(data.get("test_mode", True))

def ensure_mode_transition_clean(current_test_mode: bool, force_reset: bool = False):
    """
    Idempotent: call once on startup. If mode changed (or force_reset=True), clean appropriately.
    """
    prev = last_mode()
    if force_reset or (prev != current_test_mode):
        if not current_test_mode and (force_reset or prev is True):
            print("🧼 Detected switch to LIVE — clearing test state…")
            clear_for_live()
        elif current_test_mode and (force_reset or prev is False):
            print("🧼 Detected switch to TEST — clearing live state…")
            clear_for_test()
        remember_mode(current_test_mode)
    else:
        # still remember mode in case RUN_STATE was deleted
        remember_mode(current_test_mode)