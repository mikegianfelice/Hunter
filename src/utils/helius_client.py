"""
Utility client for interacting with the Helius Solana JSON-RPC API.

The reconciliation logic expects "enhanced" style responses (tokenTransfers,
nativeTransfers, timestamps) that used to be provided by api.helius.dev.
This client now derives the same structures directly from the RPC responses
returned by https://mainnet.helius-rpc.com/.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .http_utils import post_json


BALANCE_EPSILON = 1e-12


class HeliusClient:
    """
    Thin wrapper around the subset of the Helius RPC surface needed for
    reconciliation. The class converts JSON-RPC responses into the shape that
    the rest of the trading code already understands.
    """

    DEFAULT_BASE_URL = "https://mainnet.helius-rpc.com"

    def __init__(self, api_key: str, base_url: Optional[str] = None) -> None:
        if not api_key:
            raise ValueError("Helius API key is required to initialise HeliusClient")

        base = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        if "api-key=" in base.lower():
            self.endpoint = base
        else:
            separator = "&" if "?" in base else "?"
            self.endpoint = f"{base}{separator}api-key={api_key}"

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
        tx_types: Optional[Iterable[str]] = None,  # retained for API compatibility
    ) -> List[Dict[str, Any]]:
        """
        Fetch recent transactions for a wallet address.

        Returns a list of dicts with keys:
            - signature
            - timestamp (Unix epoch seconds)
            - tokenTransfers (list)
            - nativeTransfers (list)
        """
        _ = tx_types  # retained for compatibility; RPC layer does not yet filter by type

        signature_infos = self._get_signatures_for_address(
            address,
            limit=max(1, min(limit, 200)),
            before=before,
            until=until,
        )
        signatures = [info.signature for info in signature_infos if info.signature]
        if not signatures:
            return []

        raw_txs = self._fetch_transactions(signatures)
        results: List[Dict[str, Any]] = []
        for info, raw in zip(signature_infos, raw_txs):
            normalised = self._normalise_transaction(raw, info)
            if normalised:
                results.append(normalised)
        return results

    def get_address_balances(self, address: str) -> Dict[str, Any]:
        """
        Fetch token/SOL balances for an address.

        Matches the structure produced by the previous REST client:
            {"tokens": [{"mint": "...", "amount": ..., "decimals": ...}, ...]}
        """
        token_accounts = self._get_token_accounts(address)
        tokens: List[Dict[str, Any]] = []
        for account in token_accounts:
            mint = (account.get("mint") or "").lower()
            owner = account.get("owner")
            raw_amount = account.get("rawAmount", 0.0)
            decimals = account.get("decimals", 0)
            if not mint or owner != address:
                # Filter out delegated/unknown owners to avoid leaking other wallets
                continue
            if not raw_amount:
                continue
            tokens.append(
                {
                    "mint": mint,
                    "amount": float(raw_amount),
                    "decimals": decimals,
                }
            )

        return {"tokens": tokens}

    def get_transactions_by_signature(self, signatures: Iterable[str]) -> List[Dict[str, Any]]:
        """
        Fetch detailed transaction breakdowns for the provided signatures.
        """
        sig_list = [sig for sig in signatures if sig]
        if not sig_list:
            return []

        raw_txs = self._fetch_transactions(sig_list)
        results = []
        for signature, raw in zip(sig_list, raw_txs):
            info = SignatureInfo(signature=signature, block_time=None, slot=None)
            normalised = self._normalise_transaction(raw, info)
            if normalised:
                results.append(normalised)
        return results

    def get_transaction(self, signature: str) -> Optional[Dict[str, Any]]:
        if not signature:
            return None
        txs = self.get_transactions_by_signature([signature])
        return txs[0] if txs else None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _rpc_request(self, method: str, params: Optional[Sequence[Any]] = None) -> Any:
        payload = {
            "jsonrpc": "2.0",
            "id": f"helius-client:{method}",
            "method": method,
            "params": params or [],
        }
        response = post_json(self.endpoint, payload)
        
        # Track RPC call using centralized tracker
        if response and isinstance(response, dict) and "result" in response:
            from src.utils.api_tracker import track_helius_call
            track_helius_call()
        
        if not isinstance(response, dict):
            return None
        if "error" in response:
            return None
        return response.get("result")

    def _get_signatures_for_address(
        self,
        address: str,
        *,
        limit: int,
        before: Optional[str],
        until: Optional[str],
    ) -> List["SignatureInfo"]:
        config: Dict[str, Any] = {"limit": limit, "commitment": "confirmed"}
        if before:
            config["before"] = before
        if until:
            config["until"] = until

        result = self._rpc_request("getSignaturesForAddress", [address, config]) or []
        infos: List[SignatureInfo] = []
        for entry in result:
            signature = entry.get("signature")
            block_time = entry.get("blockTime")
            slot = entry.get("slot")
            infos.append(SignatureInfo(signature=signature, block_time=block_time, slot=slot))
        return infos

    def _fetch_transactions(self, signatures: Sequence[str]) -> List[Optional[Dict[str, Any]]]:
        transactions: List[Optional[Dict[str, Any]]] = []
        for signature in signatures:
            tx = self._rpc_request(
                "getTransaction",
                [
                    signature,
                    {
                        "encoding": "jsonParsed",
                        "commitment": "confirmed",
                        "maxSupportedTransactionVersion": 0,
                    },
                ],
            )
            transactions.append(tx)
        return transactions

    def _get_token_accounts(self, owner: str) -> List[Dict[str, Any]]:
        result = self._rpc_request(
            "getTokenAccountsByOwner",
            [
                owner,
                {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
                {"encoding": "jsonParsed", "commitment": "confirmed"},
            ],
        )
        accounts = []
        if not result:
            return accounts
        value = result.get("value") if isinstance(result, dict) else []
        for entry in value:
            account_data = (entry.get("account") or {}).get("data") if isinstance(entry, dict) else None
            parsed = (account_data or {}).get("parsed") if isinstance(account_data, dict) else None
            info = (parsed or {}).get("info") if isinstance(parsed, dict) else {}
            token_amount = info.get("tokenAmount") or {}
            ui_amount = _coerce_ui_amount(token_amount)
            decimals = token_amount.get("decimals") or 0
            mint = info.get("mint")
            owner_field = info.get("owner")
            raw_amount = token_amount.get("amount")
            if raw_amount is not None:
                try:
                    raw_amount_value = int(raw_amount)
                except (TypeError, ValueError):
                    try:
                        raw_amount_value = int(float(raw_amount))
                    except (TypeError, ValueError):
                        raw_amount_value = int(ui_amount * (10 ** decimals))
            else:
                raw_amount_value = int(ui_amount * (10 ** decimals))
            accounts.append(
                {
                    "mint": mint,
                    "owner": owner_field,
                    "amount": ui_amount,
                    "rawAmount": raw_amount_value,
                    "decimals": decimals,
                    "tokenAccount": entry.get("pubkey"),
                }
            )
        return accounts

    def _normalise_transaction(
        self,
        raw: Optional[Dict[str, Any]],
        info: "SignatureInfo",
    ) -> Optional[Dict[str, Any]]:
        if not raw or not isinstance(raw, dict):
            return None

        signature = self._extract_signature(raw) or info.signature
        if not signature:
            return None

        block_time = raw.get("blockTime") or info.block_time
        meta = raw.get("meta") or {}
        transaction = raw.get("transaction") or {}
        account_keys = self._extract_account_keys(transaction)

        token_transfers = self._build_token_transfers(meta)
        native_transfers = self._build_native_transfers(meta, account_keys)

        return {
            "signature": signature,
            "timestamp": block_time,
            "tokenTransfers": token_transfers,
            "nativeTransfers": native_transfers,
        }

    @staticmethod
    def _extract_signature(raw: Dict[str, Any]) -> Optional[str]:
        transaction = raw.get("transaction") or {}
        signatures = transaction.get("signatures")
        if isinstance(signatures, list) and signatures:
            first = signatures[0]
            return first if isinstance(first, str) else None
        return None

    @staticmethod
    def _extract_account_keys(transaction: Dict[str, Any]) -> List[str]:
        message = transaction.get("message") or {}
        account_keys = message.get("accountKeys")

        if isinstance(account_keys, list):
            return [_resolve_pubkey(key) for key in account_keys if _resolve_pubkey(key)]

        if isinstance(account_keys, dict):
            keys: List[str] = []
            for section in ("staticAccountKeys", "accountKeys"):
                values = account_keys.get(section) or []
                for key in values:
                    pubkey = _resolve_pubkey(key)
                    if pubkey:
                        keys.append(pubkey)
            return keys

        return []

    @staticmethod
    def _build_token_transfers(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        transfers: List[Dict[str, Any]] = []

        pre_balances = _index_token_balances(meta.get("preTokenBalances") or [])
        post_balances = _index_token_balances(meta.get("postTokenBalances") or [])

        all_keys = set(pre_balances.keys()) | set(post_balances.keys())
        for key in all_keys:
            pre = pre_balances.get(key)
            post = post_balances.get(key)
            owner = (post or pre or {}).get("owner")
            mint = (post or pre or {}).get("mint")
            pre_amount = (pre or {}).get("amount", 0.0)
            post_amount = (post or {}).get("amount", 0.0)
            delta = post_amount - pre_amount
            if not mint or abs(delta) <= BALANCE_EPSILON:
                continue

            record: Dict[str, Any] = {
                "mint": mint.lower(),
                "tokenAmount": abs(delta),
            }
            if delta > 0:
                record["toUserAccount"] = owner
            else:
                record["fromUserAccount"] = owner
            transfers.append(record)
        return transfers

    @staticmethod
    def _build_native_transfers(meta: Dict[str, Any], account_keys: Sequence[str]) -> List[Dict[str, Any]]:
        transfers: List[Dict[str, Any]] = []
        pre_balances = meta.get("preBalances") or []
        post_balances = meta.get("postBalances") or []
        length = min(len(pre_balances), len(post_balances), len(account_keys))
        for idx in range(length):
            pre_amount = pre_balances[idx] or 0
            post_amount = post_balances[idx] or 0
            delta = post_amount - pre_amount
            if delta == 0:
                continue
            record: Dict[str, Any] = {"amount": abs(delta)}
            account = account_keys[idx]
            if delta > 0:
                record["toUserAccount"] = account
            else:
                record["fromUserAccount"] = account
            transfers.append(record)
        return transfers


@dataclass
class SignatureInfo:
    signature: Optional[str]
    block_time: Optional[int]
    slot: Optional[int]


def _resolve_pubkey(entry: Any) -> Optional[str]:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        value = entry.get("pubkey")
        return value if isinstance(value, str) else None
    return None


def _index_token_balances(entries: Iterable[Dict[str, Any]]) -> Dict[Tuple[int, str], Dict[str, Any]]:
    indexed: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for entry in entries:
        account_index = entry.get("accountIndex")
        mint = (entry.get("mint") or "").lower()
        if account_index is None or not mint:
            continue
        ui_amount = _coerce_ui_amount(entry.get("uiTokenAmount") or {})
        indexed[(account_index, mint)] = {
            "owner": entry.get("owner"),
            "mint": mint,
            "amount": ui_amount,
        }
    return indexed


def _coerce_ui_amount(value: Any) -> float:
    if isinstance(value, dict):
        ui_amount = value.get("uiAmount")
        if ui_amount is not None:
            try:
                return float(ui_amount)
            except (TypeError, ValueError):
                pass

        amount_raw = value.get("amount")
        decimals = value.get("decimals") or 0
        if amount_raw is not None:
            try:
                return float(amount_raw) / (10 ** decimals)
            except (TypeError, ValueError, ZeroDivisionError):
                return 0.0
    elif isinstance(value, (float, int)):
        return float(value)
    return 0.0

