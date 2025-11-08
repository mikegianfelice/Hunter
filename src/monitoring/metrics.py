"""
Prometheus metrics instrumentation for the Hunter trading bot.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional

from prometheus_client import Counter, Gauge, Histogram, start_http_server

_metrics_server_lock = threading.Lock()
_metrics_server_started = False

# Core trade metrics
trade_attempts = Counter(
    "hunter_trade_attempts_total",
    "Total trade attempts grouped by chain and side",
    ["chain", "side"],
)

trade_success = Counter(
    "hunter_trade_success_total",
    "Total successful trades grouped by chain and side",
    ["chain", "side"],
)

trade_failures = Counter(
    "hunter_trade_failure_total",
    "Total failed trades grouped by chain, side, and reason",
    ["chain", "side", "reason"],
)

trade_latency_ms = Histogram(
    "hunter_trade_latency_ms",
    "Latency of trade execution in milliseconds",
    buckets=(50, 100, 250, 500, 1000, 2500, 5000, 10000),
)

slippage_bps = Histogram(
    "hunter_trade_slippage_bps",
    "Observed slippage in basis points for executed trades",
    buckets=(1, 5, 10, 20, 50, 100, 200, 500, 1000),
)

# Risk and PnL metrics
circuit_breaker_active = Gauge(
    "hunter_circuit_breaker_active",
    "Indicates if circuit breaker is currently active (1) or inactive (0)",
)

realized_pnl_usd = Gauge(
    "hunter_realized_pnl_usd", "Cumulative realized PnL (USD) for the current session"
)

unrealized_pnl_usd = Gauge(
    "hunter_unrealized_pnl_usd",
    "Aggregate unrealized PnL (USD) across open positions",
)

wallet_balance_usd = Gauge(
    "hunter_wallet_balance_usd",
    "Wallet balance in USD grouped by chain",
    ["chain"],
)

# Infrastructure metrics
rpc_errors = Counter(
    "hunter_rpc_errors_total",
    "RPC error count grouped by chain and error type",
    ["chain", "error_type"],
)

preflight_failures = Counter(
    "hunter_preflight_failures_total",
    "Preflight failure count grouped by component",
    ["component"],
)

health_checks = Counter(
    "hunter_health_checks_total",
    "Number of health/readiness checks served grouped by status",
    ["status"],
)

last_readiness_gauge = Gauge(
    "hunter_last_readiness_timestamp",
    "Timestamp of the last successful readiness check (unix seconds)",
)

session_start_time = Gauge(
    "hunter_session_start_timestamp",
    "Timestamp for when the bot session started (unix seconds)",
)


def init_metrics_server(
    port: int = 9100,
    host: str = "0.0.0.0",
) -> None:
    """
    Start the Prometheus metrics HTTP server (idempotent).
    """
    global _metrics_server_started

    with _metrics_server_lock:
        if _metrics_server_started:
            return

        start_http_server(port, addr=host)
        session_start_time.set_to_current_time()
        _metrics_server_started = True


def record_trade_attempt(chain: str, side: str) -> None:
    trade_attempts.labels(chain=chain, side=side).inc()


def record_trade_success(chain: str, side: str, latency_ms: Optional[float] = None, slippage_bps_value: Optional[float] = None) -> None:
    trade_success.labels(chain=chain, side=side).inc()
    if latency_ms is not None:
        trade_latency_ms.observe(latency_ms)
    if slippage_bps_value is not None:
        slippage_bps.observe(slippage_bps_value)


def record_trade_failure(chain: str, side: str, reason: str, latency_ms: Optional[float] = None) -> None:
    trade_failures.labels(chain=chain, side=side, reason=reason).inc()
    if latency_ms is not None:
        trade_latency_ms.observe(latency_ms)


def update_pnl(realized: Optional[float] = None, unrealized: Optional[float] = None) -> None:
    if realized is not None:
        realized_pnl_usd.set(realized)
    if unrealized is not None:
        unrealized_pnl_usd.set(unrealized)


def update_wallet_balance(balances: Dict[str, float]) -> None:
    for chain, value in balances.items():
        wallet_balance_usd.labels(chain=chain).set(value)


def set_circuit_breaker(active: bool) -> None:
    circuit_breaker_active.set(1 if active else 0)


def record_rpc_error(chain: str, error_type: str) -> None:
    rpc_errors.labels(chain=chain, error_type=error_type).inc()


def record_preflight_failure(component: str) -> None:
    preflight_failures.labels(component=component).inc()


def track_health_check(status: str, ready: bool) -> None:
    health_checks.labels(status=status).inc()
    if ready:
        last_readiness_gauge.set(time.time())

