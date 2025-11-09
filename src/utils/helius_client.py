"""
Utility client for interacting with the Helius enhanced Solana APIs.

Provides lightweight wrappers around the address transaction history,
balance, and transaction detail endpoints with retry/backoff courtesy of the
existing HTTP utilities. The client intentionally keeps dependencies minimal
so it can be reused from both synchronous monitoring loops and tooling.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlencode

from .http_utils import get_json, post_json


class HeliusClient:
    """
    Thin wrapper around the Helius REST endpoints we need for reconciliation.

    All methods return Python data structures (lists/dicts) and swallow common
    network exceptions, logging via the underlying http_utils helpers. Callers
    should treat `None` as a hard failure (e.g., credentials missing or API
    unreachable) and an empty list/dict as "no data for this query".
    """

    DEFAULT_BASE_URL = "https://api.helius.xyz"

    def __init__(self, api_key: str, base_url: Optional[str] = None) -> None:
        if not api_key:
            raise ValueError("Helius API key is required to initialise HeliusClient")

        self.api_key = api_key
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def get_address_transactions(
        self,
        address: str,
        *,
        limit: int = 100,
        before: Optional[str] = None,
        until: Optional[str] = None,
        tx_types: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch recent transactions for a wallet address.

        Args:
            address: Base58 Solana wallet address.
            limit:   Maximum number of transactions (Helius allows up to 200).
            before:  Pagination cursor (signature) to fetch older transactions.
            until:   Optional signature to stop at (exclusive).
            tx_types: Iterable of transaction type filters (e.g. ["SWAP"]).
        """

        params: Dict[str, Any] = {"limit": max(1, min(limit, 200))}
        if before:
            params["before"] = before
        if until:
            params["until"] = until
        if tx_types:
            params["types[]"] = list(tx_types)

        url = self._build_url(f"/v0/addresses/{address}/transactions", params)
        data = get_json(url)
        if not data:
            return []

        if isinstance(data, list):
            return data
        # Some Helius responses wrap the list in {"transactions": [...]}
        if isinstance(data, dict) and "transactions" in data:
            txs = data.get("transactions") or []
            return txs if isinstance(txs, list) else []
        return []

    def get_address_balances(self, address: str) -> Dict[str, Any]:
        """
        Fetch token and SOL balances for an address.
        """
        url = self._build_url(f"/v0/addresses/{address}/balances")
        data = get_json(url)
        return data or {}

    def get_transactions_by_signature(self, signatures: Iterable[str]) -> List[Dict[str, Any]]:
        """
        Fetch detailed transaction breakdowns for the provided signatures.

        Helius accepts up to 100 signatures per POST.
        """
        sig_list = [sig for sig in signatures if sig]
        if not sig_list:
            return []

        # Chunk to respect Helius limits
        results: List[Dict[str, Any]] = []
        for chunk in self._chunked(sig_list, 100):
            url = self._build_url("/v0/transactions")
            payload = {"transactions": chunk}
            data = post_json(url, payload)
            if isinstance(data, list):
                results.extend(data)
            elif isinstance(data, dict) and "transactions" in data:
                # Some beta endpoints return {"transactions": [...]}
                txs = data.get("transactions") or []
                if isinstance(txs, list):
                    results.extend(txs)
        return results

    def get_transaction(self, signature: str) -> Optional[Dict[str, Any]]:
        """
        Convenience helper to fetch a single transaction by signature.
        """
        if not signature:
            return None
        txs = self.get_transactions_by_signature([signature])
        return txs[0] if txs else None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _build_url(self, path: str, params: Optional[Dict[str, Any]] = None) -> str:
        params = params.copy() if params else {}
        params["api-key"] = self.api_key
        query = urlencode(params, doseq=True)
        return f"{self.base_url.rstrip('/')}{path}?{query}"

    @staticmethod
    def _chunked(items: Iterable[str], chunk_size: int) -> Iterable[List[str]]:
        chunk: List[str] = []
        for item in items:
            chunk.append(item)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

