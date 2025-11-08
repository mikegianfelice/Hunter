"""
Lightweight trade-intent idempotency store to avoid duplicate executions.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional, Tuple

from src.monitoring.structured_logger import log_info, log_warning
from src.storage.intents import (
    get_trade_intent,
    upsert_trade_intent,
    prune_trade_intents,
)

def _now() -> float:
    return time.time()


_DEFAULT_DUPLICATE_TTL_SECONDS = 900  # 15 minutes
_RETENTION_SECONDS = 24 * 3600        # keep history for 24h for auditing
_LOCK = threading.Lock()


@dataclass
class TradeIntent:
    chain: str
    side: str
    token_address: str
    symbol: str
    quantity: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    intent_id: str = ""
    created_at: float = field(default_factory=_now)

    def serialise(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload.setdefault("status", "registered")
        payload.setdefault("updated_at", self.created_at)
        return payload


def build_trade_intent(
    chain: str,
    side: str,
    token_address: str,
    symbol: str,
    quantity: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> TradeIntent:
    norm_chain = (chain or "").lower()
    norm_side = (side or "").lower()
    norm_token = (token_address or "").lower()
    symbol_key = (symbol or "").upper()
    snapshot = f"{norm_chain}|{norm_side}|{norm_token}|{quantity:.8f}|{symbol_key}"
    digest = hashlib.sha256(snapshot.encode()).hexdigest()
    return TradeIntent(
        chain=norm_chain,
        side=norm_side,
        token_address=norm_token,
        symbol=symbol_key,
        quantity=quantity,
        metadata=metadata or {},
        intent_id=digest,
    )


def register_trade_intent(intent: TradeIntent, duplicate_ttl_seconds: int = _DEFAULT_DUPLICATE_TTL_SECONDS) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Store a trade intent. Returns (registered, existing_entry_if_duplicate).
    """
    with _LOCK:
        prune_trade_intents(_now() - _RETENTION_SECONDS)
        existing = get_trade_intent(intent.intent_id)
        now = _now()
        if existing:
            age = now - existing.get("created_at", now)
            if age < duplicate_ttl_seconds and existing.get("status") in {"registered", "pending", "completed"}:
                log_warning(
                    "idempotency.intent.duplicate",
                    "Duplicate trade intent detected",
                    context={
                        "intent_id": intent.intent_id,
                        "chain": intent.chain,
                        "side": intent.side,
                        "token": intent.token_address,
                        "age_seconds": round(age, 2),
                        "existing_status": existing.get("status"),
                    },
                )
                return False, existing

        entry = intent.serialise()
        upsert_trade_intent(intent.intent_id, entry)
        log_info(
            "idempotency.intent.registered",
            "Registered trade intent",
            context={
                "intent_id": intent.intent_id,
                "chain": intent.chain,
                "side": intent.side,
                "token": intent.token_address,
            },
        )
        return True, existing


def _update_intent(intent_id: str, **updates: Any) -> None:
    with _LOCK:
        entry = get_trade_intent(intent_id)
        if not entry:
            return
        entry.update(updates)
        entry["updated_at"] = _now()
        upsert_trade_intent(intent_id, entry)


def mark_trade_intent_pending(intent_id: str) -> None:
    _update_intent(intent_id, status="pending")


def mark_trade_intent_completed(intent_id: str, tx_hash: Optional[str] = None) -> None:
    updates: Dict[str, Any] = {"status": "completed"}
    if tx_hash:
        updates["tx_hash"] = tx_hash
    _update_intent(intent_id, **updates)


def mark_trade_intent_failed(intent_id: str, reason: Optional[str] = None) -> None:
    updates: Dict[str, Any] = {"status": "failed"}
    if reason:
        updates["failure_reason"] = reason
    _update_intent(intent_id, **updates)


