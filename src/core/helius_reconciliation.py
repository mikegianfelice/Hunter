"""
Helius-powered reconciliation for open positions and PnL accuracy.

This module integrates with the performance tracker and open_positions.json to:

1. Verify that every recorded Solana position still has on-chain balance.
2. Close phantom positions (no balance) and clean up open_positions.json.
3. Derive real execution metrics (tokens received, actual spend, fees) from
   Helius transaction breakdowns so PnL is based on on-chain data.

It is intentionally conservative – if the Helius client or credentials are
missing we short-circuit without mutating bot state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from src.config.secrets import HELIUS_API_KEY, SOLANA_WALLET_ADDRESS
from src.core.performance_tracker import performance_tracker
from src.utils.helius_client import HeliusClient
from src.storage.positions import load_positions as load_positions_store, replace_positions

try:
    # Lazily imported – falls back to static SOL price if API fails.
    from src.utils.solana_transaction_analyzer import get_sol_price_usd as get_sol_price_usd
except Exception:  # pragma: no cover - defensive import
    def get_sol_price_usd() -> float:
        return 0.0


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OPEN_POSITIONS_FILE = DATA_DIR / "open_positions.json"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
BALANCE_EPSILON = 1e-9
TIMESTAMP_TOLERANCE_SECONDS = 900  # +/- 15 minutes for heuristic matching


def reconcile_positions_and_pnl(limit: int = 200) -> Dict[str, Any]:
    """
    Run the full Helius reconciliation process.

    Returns a summary dictionary with counts of the actions performed.
    """
    if not HELIUS_API_KEY or not SOLANA_WALLET_ADDRESS:
        return {
            "enabled": False,
            "reason": "Missing HELIUS_API_KEY or SOLANA_WALLET_ADDRESS secrets",
        }

    # Early exit if no Solana positions to reconcile
    solana_trades = [t for t in performance_tracker.trades 
                     if (t.get("chain") or "").lower() == "solana"]
    if not solana_trades:
        return {
            "enabled": True,
            "open_positions_closed": 0,
            "open_positions_verified": 0,
            "trades_updated": 0,
            "issues": [],
            "skipped": "No Solana positions to reconcile",
        }
    
    # Check rate limit before making API calls
    try:
        from src.utils.api_tracker import get_tracker
        tracker = get_tracker()
        helius_calls = tracker.get_count('helius')
        helius_max = 30000
        if helius_calls >= helius_max * 0.95:  # Stop if at 95% of limit
            return {
                "enabled": True,
                "open_positions_closed": 0,
                "open_positions_verified": 0,
                "trades_updated": 0,
                "issues": [],
                "skipped": f"Near Helius rate limit ({helius_calls}/{helius_max})",
            }
    except Exception:
        pass  # Continue if check fails

    client = HeliusClient(HELIUS_API_KEY)
    context = _HeliusContext(client, SOLANA_WALLET_ADDRESS, limit=limit)

    summary = {
        "enabled": True,
        "open_positions_closed": 0,
        "open_positions_verified": 0,
        "trades_updated": 0,
        "issues": [],
    }

    open_positions = load_positions_store()
    trades_changed = False
    positions_changed = False

    # ------------------------------------------------------------------ #
    # Step 1: Verify open positions against on-chain balances
    # ------------------------------------------------------------------ #
    balance_index = context.balance_index
    for trade in performance_tracker.trades:
        if (trade.get("chain") or "").lower() != "solana":
            continue

        mint = (trade.get("address") or "").lower()
        if not mint:
            continue

        balance = balance_index.get(mint, 0.0)
        trade_id = trade.get("id")
        status = trade.get("status", "open").lower()

        if status == "open":
            if balance <= BALANCE_EPSILON:
                # No tokens left – mark as manual close.
                entry_price = trade.get("entry_price", 0.0) or 0.0
                exit_price = entry_price  # Use entry price as fallback if we can't determine exit price
                
                # Try to get actual exit price from recent transactions
                exit_tx = context.find_matching_transaction(trade, mint, direction="sell")
                if exit_tx:
                    # Extract exit price from transaction if possible
                    transfers = exit_tx.get("tokenTransfers") or []
                    mint_lower = mint.lower()
                    usdc_received = _aggregate_token_amount(
                        transfers, SOLANA_WALLET_ADDRESS, USDC_MINT.lower(), incoming=True
                    )
                    tokens_sold = _aggregate_token_amount(
                        transfers, SOLANA_WALLET_ADDRESS, mint_lower, incoming=False
                    )
                    if tokens_sold > BALANCE_EPSILON and usdc_received > 0:
                        exit_price = usdc_received / tokens_sold
                
                performance_tracker.log_trade_exit(
                    trade_id,
                    exit_price,
                    0.0,
                    status="manual_close",
                )
                
                # Also log to trade_log.csv
                try:
                    from src.monitoring.monitor_position import log_trade
                    log_trade(mint, entry_price, exit_price, "reconciliation_close")
                except Exception as e:
                    print(f"⚠️ Failed to log trade to trade_log.csv: {e}")
                
                summary["open_positions_closed"] += 1
                trades_changed = True
                if _remove_position_for_trade(open_positions, trade_id, mint):
                    positions_changed = True
            else:
                summary["open_positions_verified"] += 1
        else:
            # Trade is closed but still has balance – flag for manual review.
            if balance > BALANCE_EPSILON:
                summary["issues"].append(
                    f"Closed trade {trade_id or mint} still holds {balance:.6f} tokens on-chain."
                )

    # ------------------------------------------------------------------ #
    # Step 2: Rebuild execution metrics using Helius transaction data
    # ------------------------------------------------------------------ #
    sol_price = get_sol_price_usd() or 0.0
    transactions = context.transactions
    for trade in performance_tracker.trades:
        if (trade.get("chain") or "").lower() != "solana":
            continue

        mint = (trade.get("address") or "").lower()
        if not mint:
            continue

        updated = False

        # Entry reconciliation ------------------------------------------------
        entry_sig = trade.get("buy_tx_hash")
        entry_tx = context.get_transaction(entry_sig) if entry_sig else None
        if entry_tx is None:
            entry_tx = context.find_matching_transaction(trade, mint, direction="buy")
            if entry_tx:
                entry_sig = entry_tx.get("signature")
                if entry_sig:
                    trade["buy_tx_hash"] = entry_sig
        if entry_tx:
            if _apply_entry_metrics(trade, entry_tx, SOLANA_WALLET_ADDRESS, sol_price):
                updated = True

        # Exit reconciliation -------------------------------------------------
        if trade.get("status", "open").lower() != "open":
            exit_sig = trade.get("sell_tx_hash")
            exit_tx = context.get_transaction(exit_sig) if exit_sig else None
            if exit_tx is None:
                exit_tx = context.find_matching_transaction(trade, mint, direction="sell")
                if exit_tx:
                    exit_sig = exit_tx.get("signature")
                    if exit_sig:
                        trade["sell_tx_hash"] = exit_sig
            if exit_tx:
                if _apply_exit_metrics(trade, exit_tx, SOLANA_WALLET_ADDRESS, sol_price):
                    updated = True

        if updated:
            trades_changed = True
            summary["trades_updated"] += 1

    # ------------------------------------------------------------------ #
    # Persist changes if needed
    # ------------------------------------------------------------------ #
    if trades_changed:
        performance_tracker.save_data()
    if positions_changed:
        replace_positions(open_positions)

    return summary


# =============================================================================
# Helper classes / functions
# =============================================================================


@dataclass
class _HeliusContext:
    client: HeliusClient
    wallet: str
    limit: int = 200

    def __post_init__(self) -> None:
        self.wallet = self.wallet.strip()
        self._transactions: Dict[str, Dict[str, Any]] = {}
        self._transactions_loaded: bool = False
        self._balances: Optional[Dict[str, float]] = None

    # Transactions ----------------------------------------------------- #
    @property
    def transactions(self) -> List[Dict[str, Any]]:
        if not self._transactions_loaded:
            txs = self.client.get_address_transactions(self.wallet, limit=self.limit)
            self._transactions = {
                tx.get("signature"): tx for tx in txs if tx.get("signature")
            }
            self._transactions_loaded = True
        return list(self._transactions.values())

    def get_transaction(self, signature: Optional[str]) -> Optional[Dict[str, Any]]:
        if not signature:
            return None
        if signature not in self._transactions:
            tx = self.client.get_transaction(signature)
            if tx:
                self._transactions[signature] = tx
        return self._transactions.get(signature)

    def find_matching_transaction(
        self, trade: Mapping[str, Any], mint: str, direction: str
    ) -> Optional[Dict[str, Any]]:
        """
        Heuristic matching: locate a transaction where the wallet receives
        (direction="buy") or sends (direction="sell") the given mint around the
        recorded entry/exit time.
        """
        target_time = _parse_iso_datetime(
            trade.get("entry_time") if direction == "buy" else trade.get("exit_time")
        )
        if target_time is None:
            # Fallback: allow entire fetched range
            target_time = datetime.now(timezone.utc)

        expected_value = None
        try:
            entry_price = float(trade.get("entry_price") or 0.0)
            position_size = float(trade.get("position_size_usd") or 0.0)
            if entry_price > 0 and position_size > 0:
                expected_value = position_size / entry_price
        except (TypeError, ValueError):
            expected_value = None

        mint_lower = mint.lower()
        wallet = self.wallet
        for tx in self.transactions:
            ts = tx.get("timestamp")
            if ts is None:
                continue
            tx_time = datetime.fromtimestamp(ts, tz=timezone.utc)
            if abs((tx_time - target_time).total_seconds()) > TIMESTAMP_TOLERANCE_SECONDS:
                continue

            transfers: List[Dict[str, Any]] = tx.get("tokenTransfers") or []
            if not transfers:
                continue

            if direction == "buy":
                amount = sum(
                    t.get("tokenAmount", 0.0)
                    for t in transfers
                    if (t.get("mint") or "").lower() == mint_lower
                    and t.get("toUserAccount") == wallet
                )
                if amount <= BALANCE_EPSILON:
                    continue
                if expected_value and not _approximately_equal(amount, expected_value, tolerance=0.35):
                    continue
                return tx
            else:
                amount = sum(
                    t.get("tokenAmount", 0.0)
                    for t in transfers
                    if (t.get("mint") or "").lower() == mint_lower
                    and t.get("fromUserAccount") == wallet
                )
                if amount <= BALANCE_EPSILON:
                    continue
                return tx
        return None

    # Balances --------------------------------------------------------- #
    @property
    def balance_index(self) -> Dict[str, float]:
        if self._balances is None:
            balances = self.client.get_address_balances(self.wallet)
            tokens = balances.get("tokens") or []
            index: Dict[str, float] = {}
            for token in tokens:
                mint = (token.get("mint") or "").lower()
                amount_raw = token.get("amount") or 0
                decimals = token.get("decimals") or 0
                if mint:
                    amount = float(amount_raw) / (10 ** decimals) if decimals else float(amount_raw)
                    index[mint] = amount
            self._balances = index
        return self._balances


# =============================================================================
# Metric application helpers
# =============================================================================


def _apply_entry_metrics(
    trade: Dict[str, Any],
    tx: Mapping[str, Any],
    wallet: str,
    sol_price: float,
) -> bool:
    """Populate entry-side metrics from a Helius transaction."""
    transfers = tx.get("tokenTransfers") or []
    if not transfers:
        return False

    mint_lower = (trade.get("address") or "").lower()
    token_in = _aggregate_token_amount(transfers, wallet, mint_lower, incoming=True)
    usdc_spent = _aggregate_token_amount(
        transfers, wallet, USDC_MINT.lower(), incoming=False
    )
    sol_spent, sol_received = _aggregate_native_transfers(tx, wallet)
    sol_fee = float(tx.get("fee", 0)) / 1_000_000_000

    updated = False
    if token_in > BALANCE_EPSILON:
        if trade.get("entry_tokens_received") != token_in:
            trade["entry_tokens_received"] = token_in
            updated = True
    if usdc_spent > 0:
        if trade.get("entry_amount_usd_actual") != usdc_spent:
            trade["entry_amount_usd_actual"] = usdc_spent
            updated = True
    total_sol_cost = sol_fee + max(sol_spent - sol_received, 0)
    if sol_price > 0 and total_sol_cost > 0:
        fee_usd = total_sol_cost * sol_price
        if trade.get("entry_gas_fee_usd") != fee_usd:
            trade["entry_gas_fee_usd"] = fee_usd
            updated = True
    return updated


def _apply_exit_metrics(
    trade: Dict[str, Any],
    tx: Mapping[str, Any],
    wallet: str,
    sol_price: float,
) -> bool:
    """Populate exit-side metrics and after-fee PnL."""
    transfers = tx.get("tokenTransfers") or []
    if not transfers:
        return False

    mint_lower = (trade.get("address") or "").lower()
    token_out = _aggregate_token_amount(transfers, wallet, mint_lower, incoming=False)
    usdc_in = _aggregate_token_amount(transfers, wallet, USDC_MINT.lower(), incoming=True)
    sol_spent, sol_received = _aggregate_native_transfers(tx, wallet)
    sol_fee = float(tx.get("fee", 0)) / 1_000_000_000

    updated = False
    if token_out > BALANCE_EPSILON:
        trade["exit_tokens_sold"] = token_out
        updated = True
    if usdc_in > 0:
        trade["actual_proceeds_usd"] = usdc_in
        updated = True

    total_sol_fee = sol_fee + max(sol_spent - sol_received, 0)
    exit_fee_usd = total_sol_fee * sol_price if sol_price > 0 and total_sol_fee > 0 else 0.0
    if exit_fee_usd > 0:
        trade["exit_gas_fee_usd"] = exit_fee_usd
        updated = True

    # Compute after-fee PnL if we have enough data
    entry_cost = float(trade.get("entry_amount_usd_actual") or trade.get("position_size_usd") or 0.0)
    entry_fees = float(trade.get("entry_gas_fee_usd") or 0.0)
    total_fees = entry_fees + exit_fee_usd
    proceeds = float(trade.get("actual_proceeds_usd") or 0.0)

    if entry_cost > 0 and proceeds > 0:
        pnl_after_fees_usd = proceeds - entry_cost - total_fees
        pnl_after_fees_percent = (pnl_after_fees_usd / entry_cost) * 100 if entry_cost else 0.0
        trade["total_fees_usd"] = total_fees
        trade["pnl_after_fees_usd"] = pnl_after_fees_usd
        trade["pnl_after_fees_percent"] = pnl_after_fees_percent
        updated = True

    return updated


def _aggregate_token_amount(
    transfers: Iterable[Mapping[str, Any]],
    wallet: str,
    mint_lower: str,
    *,
    incoming: bool,
) -> float:
    total = 0.0
    for transfer in transfers:
        if (transfer.get("mint") or "").lower() != mint_lower:
            continue
        if incoming and transfer.get("toUserAccount") == wallet:
            total += float(transfer.get("tokenAmount") or 0.0)
        elif not incoming and transfer.get("fromUserAccount") == wallet:
            total += float(transfer.get("tokenAmount") or 0.0)
    return total


def _aggregate_native_transfers(tx: Mapping[str, Any], wallet: str) -> Tuple[float, float]:
    transfers = tx.get("nativeTransfers") or []
    spent = 0.0
    received = 0.0
    for transfer in transfers:
        amount = float(transfer.get("amount") or 0.0) / 1_000_000_000
        if transfer.get("fromUserAccount") == wallet:
            spent += amount
        if transfer.get("toUserAccount") == wallet:
            received += amount
    return spent, received


# =============================================================================
# Utility functions
# =============================================================================


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _approximately_equal(a: float, b: float, *, tolerance: float = 0.15) -> bool:
    if a == 0 or b == 0:
        return False
    return abs(a - b) / max(abs(a), abs(b)) <= tolerance


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return default


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2)
    tmp.replace(path)


def _remove_position_for_trade(open_positions: Dict[str, Any], trade_id: Optional[str], mint: str) -> bool:
    """
    Remove the open position entry matching the trade ID (preferred) or mint.
    Returns True if a position was removed.
    """
    removed = False
    if trade_id:
        key_to_remove = None
        for key, value in open_positions.items():
            if isinstance(value, dict) and value.get("trade_id") == trade_id:
                key_to_remove = key
                break
        if key_to_remove:
            open_positions.pop(key_to_remove, None)
            removed = True

    if not removed:
        # Fallback: remove first entry whose address matches the mint
        mint_lower = mint.lower()
        key_to_remove = None
        for key, value in open_positions.items():
            if isinstance(value, dict):
                address = (value.get("address") or "").lower()
            else:
                address = key.lower()
            if mint_lower == address:
                key_to_remove = key
                break
        if key_to_remove:
            open_positions.pop(key_to_remove, None)
            removed = True
    return removed

