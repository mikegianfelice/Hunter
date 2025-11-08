"""
Execution guardrail helpers (slippage, gas, simulation, retries).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from web3 import Web3

from src.config.config_validator import ExecutionConfig, get_execution_config
from src.monitoring.structured_logger import log_warning, log_info
from src.monitoring.metrics import record_rpc_error


def _get_policy() -> ExecutionConfig:
    config = get_execution_config()
    if config is None:
        return ExecutionConfig()
    return config


def _normalise_chain(chain: Optional[str]) -> str:
    return (chain or "").lower()


def slippage_limit_for_chain(chain: str) -> float:
    policy = _get_policy()
    norm = _normalise_chain(chain)
    return policy.max_slippage_percent_by_chain.get(norm, policy.max_slippage_percent)


def enforce_slippage_limit(chain: str, requested: float, context: Optional[Dict[str, object]] = None) -> float:
    limit = slippage_limit_for_chain(chain)
    if requested <= limit:
        return requested

    log_warning(
        "execution.slippage.clamped",
        "Requested slippage exceeded policy; clamping to limit",
        context={
            "chain": chain,
            "requested": requested,
            "limit": limit,
            **(context or {}),
        },
    )
    return limit


def gas_ceiling_for_chain(chain: str) -> float:
    policy = _get_policy()
    norm = _normalise_chain(chain)
    return policy.gas_ceiling_gwei_by_chain.get(norm, policy.gas_ceiling_gwei)


def priority_fee_limit_for_chain(chain: str) -> float:
    policy = _get_policy()
    norm = _normalise_chain(chain)
    return policy.max_priority_fee_gwei_by_chain.get(norm, policy.max_priority_fee_gwei)


def apply_gas_guardrails(chain: str, w3: Web3, max_fee_wei: int, max_priority_wei: int) -> Tuple[int, int]:
    policy = _get_policy()
    norm = _normalise_chain(chain)

    ceiling_gwei = policy.gas_ceiling_gwei_by_chain.get(norm, policy.gas_ceiling_gwei)
    ceiling_wei = int(w3.to_wei(ceiling_gwei, "gwei"))

    adjusted_max_fee = max_fee_wei
    if max_fee_wei > ceiling_wei:
        adjusted_max_fee = ceiling_wei
        log_warning(
            "execution.gas.clamped",
            "Clamped maxFeePerGas to policy ceiling",
            context={
                "chain": chain,
                "requested_wei": max_fee_wei,
                "clamped_wei": adjusted_max_fee,
                "ceiling_gwei": ceiling_gwei,
            },
        )

    prio_limit_gwei = policy.max_priority_fee_gwei_by_chain.get(norm, policy.max_priority_fee_gwei)
    prio_limit_wei = int(w3.to_wei(prio_limit_gwei, "gwei"))

    adjusted_priority = max_priority_wei
    if max_priority_wei > prio_limit_wei:
        adjusted_priority = prio_limit_wei
        log_warning(
            "execution.priority.clamped",
            "Clamped maxPriorityFeePerGas to policy limit",
            context={
                "chain": chain,
                "requested_wei": max_priority_wei,
                "clamped_wei": adjusted_priority,
                "priority_limit_gwei": prio_limit_gwei,
            },
        )

    if adjusted_priority > adjusted_max_fee:
        adjusted_priority = adjusted_max_fee
        log_warning(
            "execution.priority.adjusted",
            "Adjusted priority fee to not exceed max fee",
            context={
                "chain": chain,
                "max_fee_wei": adjusted_max_fee,
            },
        )

    return adjusted_max_fee, adjusted_priority


def retry_policy() -> Tuple[int, float]:
    policy = _get_policy()
    return policy.max_retries, policy.retry_backoff_seconds


def should_simulate_transaction(chain: str) -> bool:
    policy = _get_policy()
    return policy.enable_simulation


def simulate_evm_transaction(chain: str, w3: Web3, tx_dict: Dict[str, int]) -> Tuple[bool, Optional[str]]:
    """
    Perform an eth_call simulation for the given transaction.
    Returns (success, error_message).
    """
    if not should_simulate_transaction(chain):
        return True, None

    call_tx = dict(tx_dict)
    # Remove fields not supported by eth_call
    call_tx.pop("nonce", None)
    call_tx.pop("chainId", None)

    try:
        w3.eth.call(call_tx, block_identifier="pending")
        return True, None
    except Exception as exc:
        record_rpc_error(chain, "simulation")
        log_warning(
            "execution.simulation.failed",
            "Transaction simulation failed",
            context={
                "chain": chain,
                "error": str(exc),
            },
        )
        return False, str(exc)


