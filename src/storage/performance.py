from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

_PERFORMANCE_JSON_PATH = Path("data/performance_data.json")
_LOCK = threading.Lock()


def set_json_path(path: str | Path) -> None:
    global _PERFORMANCE_JSON_PATH
    _PERFORMANCE_JSON_PATH = Path(path)


def load_performance_data() -> Dict[str, Any]:
    """Load performance data from JSON file"""
    if not _PERFORMANCE_JSON_PATH.exists():
        return {"trades": [], "daily_stats": {}}
    
    try:
        with _PERFORMANCE_JSON_PATH.open("r", encoding="utf-8") as fh:
            payload = json.load(fh) or {}
        return {
            "trades": payload.get("trades", []) or [],
            "daily_stats": payload.get("daily_stats", {}) or {},
        }
    except Exception:
        return {"trades": [], "daily_stats": {}}


def replace_performance_data(data: Dict[str, Any]) -> None:
    """Save performance data to JSON file"""
    snapshot = dict(data)
    snapshot.setdefault("last_updated", datetime.now().isoformat())
    _PERFORMANCE_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with _LOCK:
        with _PERFORMANCE_JSON_PATH.open("w", encoding="utf-8") as fh:
            json.dump(snapshot, fh, indent=2)
