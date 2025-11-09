"""
Utility helpers for working with the CoinGecko API.

These helpers normalize request URLs/params so we always provide the required
`vs_currency`/`vs_currencies` query parameter. CoinGecko responds with
`{"error":"Missing parameter vs_currency"}` whenever that parameter is absent
for price/market endpoints, so centralizing the guard prevents regressions.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

DEFAULT_FIAT = (os.getenv("COINGECKO_VS_CURRENCY") or "usd").strip().lower() or "usd"


def _needs_currency_guard(url: str) -> bool:
    return "api.coingecko.com" in url


def _pick_param_name(path: str) -> str:
    """
    Determine which CoinGecko query parameter should carry the quote currency.
    """
    if "simple/price" in path or "simple/token_price" in path:
        return "vs_currencies"
    return "vs_currency"


def ensure_vs_currency(
    url: str, params: Optional[Dict[str, str]] = None
) -> Tuple[str, Optional[Dict[str, str]]]:
    """
    Ensure the CoinGecko request includes a vs_currency/vs_currencies parameter.

    Returns the possibly-updated URL and params dictionary.
    """
    if not _needs_currency_guard(url):
        return url, params

    parsed = urlparse(url)
    param_name = _pick_param_name(parsed.path)

    alt_name = "vs_currencies" if param_name == "vs_currency" else "vs_currency"

    # Quickly exit (or normalise) if the parameter already exists in params.
    if params:
        if param_name in params:
            return url, params
        if alt_name in params:
            updated = dict(params)
            updated[param_name] = updated.pop(alt_name)
            return url, updated

    query = parse_qs(parsed.query, keep_blank_values=True)
    if param_name in query:
        return url, params
    if alt_name in query:
        query[param_name] = query.pop(alt_name)
        new_query = urlencode(query, doseq=True)
        updated_url = urlunparse(parsed._replace(query=new_query))
        return updated_url, params

    currency = DEFAULT_FIAT

    if params is not None:
        updated = dict(params)
        updated.setdefault(param_name, currency)
        return url, updated

    query[param_name] = [currency]
    new_query = urlencode(query, doseq=True)
    updated_url = urlunparse(parsed._replace(query=new_query))
    return updated_url, params


