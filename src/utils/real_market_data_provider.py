#!/usr/bin/env python3
"""
Real market data provider utilities.

This module centralises access to free, public market data sources so that the
rest of the codebase can work exclusively with real observations rather than
simulated or mock values.  The provider currently aggregates data from the
following endpoints:

* CoinGecko - spot prices, market data, community metrics
* CoinCap    - high resolution historical price/volume series
* DexScreener - decentralised exchange liquidity metrics (when a contract
  address is supplied)

Where certain metrics are unavailable from free APIs (e.g. historical holder
counts), the provider returns ``None`` instead of synthesising a value.  Any
downstream consumer must therefore handle missing data explicitly.

All requests include lightweight retry logic and in-memory caching to avoid
excessive rate-limit pressure on upstream services.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from src.utils.coingecko_helpers import ensure_vs_currency, DEFAULT_FIAT
from src.utils.api_tracker import track_coingecko_call, track_coincap_call

logger = logging.getLogger(__name__)


COINGECKO_PUBLIC_BASE = "https://api.coingecko.com/api/v3/"
COINCAP_BASE = "https://api.coincap.io/v2"
DEXSCREENER_BASE = "https://api.dexscreener.com/latest/dex"


DEFAULT_SYMBOL_MAP: Dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "ADA": "cardano",
    "MATIC": "matic-network",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "DOT": "polkadot",
    "DOGE": "dogecoin",
    "AVAX": "avalanche-2",
    "TRX": "tron",
    "LTC": "litecoin",
    "LINK": "chainlink",
    "UNI": "uniswap",
    "AAVE": "aave",
    "ARB": "arbitrum",
    "OP": "optimism",
    "ATOM": "cosmos",
    "FTM": "fantom",
}


def _to_unix_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


@dataclass
class DexMetrics:
    liquidity_usd: Optional[float] = None
    volume_24h: Optional[float] = None
    transactions_24h: Optional[int] = None
    price_usd: Optional[float] = None
    source_pair: Optional[str] = None


class RealMarketDataProvider:
    """Access real market data from public APIs."""

    def __init__(self) -> None:
        self._session = requests.Session()
        self._symbol_to_id: Dict[str, str] = DEFAULT_SYMBOL_MAP.copy()
        self._id_cache_expiry: Dict[str, float] = {}
        self._ohlcv_cache: Dict[Tuple[str, str, str], pd.DataFrame] = {}
        self._ohlcv_expiry: Dict[Tuple[str, str, str], float] = {}
        self._dex_cache: Dict[str, DexMetrics] = {}
        self._dex_expiry: Dict[str, float] = {}
        self._coingecko_cache: Dict[str, Dict[str, float]] = {}
        self._coingecko_expiry: Dict[str, float] = {}
        
        # Load CoinGecko API key from environment
        self._coingecko_api_key = (os.getenv("COINGECKO_API_KEY") or "").strip()
        self._coingecko_base = COINGECKO_PUBLIC_BASE

        # Simple TTLs – can be tuned as needed.
        self._default_ttl = 300.0  # 5 minutes
        self._long_ttl = 3600.0    # 1 hour

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_coingecko_id(self, symbol: str) -> Optional[str]:
        """Resolve a trading symbol to a CoinGecko asset id."""

        if not symbol:
            return None

        sym_upper = symbol.upper()
        cached = self._symbol_to_id.get(sym_upper)
        if cached:
            return cached

        try:
            query_url = f"{self._coingecko_base}search"
            response = self._request_json(query_url, params={"query": symbol})
            if not response:
                return None

            coins = response.get("coins", [])
            for coin in coins:
                if coin.get("symbol", "").upper() == sym_upper:
                    asset_id = coin.get("id")
                    if asset_id:
                        self._symbol_to_id[sym_upper] = asset_id
                        return asset_id

            if coins:
                # Fallback to the first result if symbol match not found
                asset_id = coins[0].get("id")
                if asset_id:
                    self._symbol_to_id[sym_upper] = asset_id
                    return asset_id
        except Exception as exc:
            logger.warning("Failed to resolve CoinGecko id for %s: %s", symbol, exc)

        return None

    def get_asset_snapshot(self, symbol: str) -> Optional[Dict[str, Optional[float]]]:
        """Fetch spot market metrics for a symbol."""

        asset_id = self.get_coingecko_id(symbol)
        if not asset_id:
            return None

        cache_key = asset_id
        if cache_key in self._coingecko_cache and time.time() < self._coingecko_expiry.get(cache_key, 0):
            return self._coingecko_cache[cache_key]

        url = f"{self._coingecko_base}coins/{asset_id}"
        try:
            data = self._request_json(
                url,
                params={
                    "localization": "false",
                    "tickers": "false",
                    "market_data": "true",
                    "community_data": "true",
                    "developer_data": "false",
                    "sparkline": "false",
                    "vs_currency": DEFAULT_FIAT,
                },
            )
            if not data:
                return None

            market_data = data.get("market_data", {})
            community_data = data.get("community_data", {})

            snapshot = {
                "price_usd": self._safe_float(market_data.get("current_price", {}).get("usd")),
                "market_cap_usd": self._safe_float(market_data.get("market_cap", {}).get("usd")),
                "volume_24h_usd": self._safe_float(market_data.get("total_volume", {}).get("usd")),
                "price_change_pct_24h": self._safe_float(market_data.get("price_change_percentage_24h")),
                "sentiment_votes_up": self._safe_float(data.get("sentiment_votes_up_percentage")),
                "sentiment_votes_down": self._safe_float(data.get("sentiment_votes_down_percentage")),
                "twitter_followers": self._safe_float(community_data.get("twitter_followers")),
                "reddit_subscribers": self._safe_float(community_data.get("reddit_subscribers")),
            }

            self._coingecko_cache[cache_key] = snapshot
            self._coingecko_expiry[cache_key] = time.time() + self._default_ttl
            return snapshot

        except Exception as exc:
            logger.warning("Failed to fetch CoinGecko snapshot for %s: %s", symbol, exc)
            return None

    def get_historical_ohlcv(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Historical hourly OHLCV series for the requested symbol."""

        cache_key = (symbol.upper(), start_date, end_date)
        if cache_key in self._ohlcv_cache and time.time() < self._ohlcv_expiry.get(cache_key, 0):
            return self._ohlcv_cache[cache_key]

        asset_id = self.get_coingecko_id(symbol)
        if not asset_id:
            logger.warning("Unable to resolve asset id for %s", symbol)
            return None

        try:
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
        except ValueError:
            logger.error("Invalid date format for historical range: %s - %s", start_date, end_date)
            return None

        params = {
            "interval": "h1",
            "start": _to_unix_ms(start_dt),
            "end": _to_unix_ms(end_dt),
        }

        url = f"{COINCAP_BASE}/assets/{asset_id}/history"
        data = self._request_json(url, params=params)
        if not data or "data" not in data:
            logger.warning("No historical data returned for %s", symbol)
            return None

        rows: List[Dict[str, float]] = []
        previous_close: Optional[float] = None

        for entry in data["data"]:
            try:
                ts = datetime.fromtimestamp(entry["time"] / 1000, tz=timezone.utc)
                close_price = float(entry.get("priceUsd")) if entry.get("priceUsd") else None
                if close_price is None:
                    continue

                open_price = previous_close if previous_close is not None else close_price
                high_price = max(open_price, close_price)
                low_price = min(open_price, close_price)
                volume = self._safe_float(entry.get("volumeUsd24Hr"))
                market_cap = self._safe_float(entry.get("marketCapUsd"))

                rows.append(
                    {
                        "timestamp": ts,
                        "open": open_price,
                        "high": high_price,
                        "low": low_price,
                        "close": close_price,
                        "volume": volume or 0.0,
                        "market_cap": market_cap or 0.0,
                    }
                )

                previous_close = close_price
            except Exception as exc:
                logger.debug("Skipping malformed historical entry for %s: %s", symbol, exc)
                continue

        if not rows:
            logger.warning("Empty historical series for %s", symbol)
            return None

        df = pd.DataFrame(rows)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

        self._ohlcv_cache[cache_key] = df
        self._ohlcv_expiry[cache_key] = time.time() + self._long_ttl
        return df

    def get_dex_metrics(self, contract_address: str) -> Optional[DexMetrics]:
        """Retrieve liquidity metrics for a specific on-chain contract."""

        if not contract_address:
            return None

        address = contract_address.lower()
        if address in self._dex_cache and time.time() < self._dex_expiry.get(address, 0):
            return self._dex_cache[address]

        url = f"{DEXSCREENER_BASE}/tokens/{address}"
        payload = self._request_json(url)
        if not payload:
            return None

        pairs = payload.get("pairs", [])
        if not pairs:
            return None

        # Choose the deepest pair by liquidity USD
        richest = max(
            pairs,
            key=lambda p: float(p.get("liquidity", {}).get("usd", 0) or 0),
        )

        liquidity_usd = self._safe_float(richest.get("liquidity", {}).get("usd"))
        volume_24h = self._safe_float(richest.get("volume", {}).get("h24"))
        txns = richest.get("txns", {}).get("h24")
        tx_count = None
        if isinstance(txns, dict):
            buys = txns.get("buys") or 0
            sells = txns.get("sells") or 0
            tx_count = int(buys) + int(sells)

        price_usd = self._safe_float(richest.get("priceUsd"))
        pair_address = richest.get("pairAddress")

        metrics = DexMetrics(
            liquidity_usd=liquidity_usd,
            volume_24h=volume_24h,
            transactions_24h=tx_count,
            price_usd=price_usd,
            source_pair=pair_address,
        )

        self._dex_cache[address] = metrics
        self._dex_expiry[address] = time.time() + self._default_ttl
        return metrics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _request_json(self, url: str, params: Optional[Dict[str, str]] = None) -> Optional[Dict]:
        """HTTP GET with retry and timeout handling."""
        
        # Add CoinGecko API key if available
        headers = {}
        if self._coingecko_api_key and "coingecko.com" in url:
            headers["x-cg-demo-api-key"] = self._coingecko_api_key

        url, params = ensure_vs_currency(url, params)

        attempt = 0
        while attempt < 3:
            try:
            response = self._session.get(url, params=params, headers=headers, timeout=20)
            if response.status_code == 200:
                # Track API calls
                if "coingecko.com" in url:
                    track_coingecko_call()
                elif "coincap.io" in url or "rest.coincap.io" in url:
                    track_coincap_call()
                return response.json()

                if response.status_code == 429:
                    sleep_for = 1 + attempt
                    logger.warning("Rate limited by %s, sleeping %ss", url, sleep_for)
                    time.sleep(sleep_for)
                    attempt += 1
                    continue

                logger.warning("Unexpected status %s from %s", response.status_code, url)
                return None

            except requests.RequestException as exc:
                sleep_for = (attempt + 1) * 1.5
                logger.debug("HTTP error fetching %s: %s", url, exc)
                time.sleep(sleep_for)
                attempt += 1

        logger.error("Exceeded retry budget fetching %s", url)
        return None

    @staticmethod
    def _safe_float(value: Optional[float]) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None


# Shared singleton instance
real_market_data_provider = RealMarketDataProvider()


