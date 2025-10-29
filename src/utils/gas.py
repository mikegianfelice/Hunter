# gas.py
"""
EIP-1559 gas suggester using on-chain data (eth_feeHistory) with guardrails.
No external API keys required.
"""

from web3 import Web3
from typing import Tuple

DEFAULTS = {
    "gas_blocks": 20,                 # how many recent blocks to sample
    "gas_reward_percentile": 50,      # percentile of priority fee to use (10/50/90)
    "gas_basefee_headroom": 1.25,     # multiply latest baseFee by this buffer
    "gas_priority_min_gwei": 1.0,     # floor for priority fee
    "gas_priority_max_gwei": 5.0,     # cap for priority fee (before ceiling)
    "gas_ceiling_gwei": 100.0,        # absolute cap for maxFeePerGas
    "gas_multiplier": 1.0,            # scale both fees up/down uniformly
    "gas_extra_priority_gwei": 0.0,   # little nudge to priority (e.g., +0.5)
}

def _gwei_to_wei(w3: Web3, g: float) -> int:
    return int(w3.to_wei(g, "gwei"))

def _wei_to_gwei(w3: Web3, w: int) -> float:
    return float(w3.from_wei(w, "gwei"))

def suggest_fees(
    w3: Web3,
    config: dict = None,
) -> Tuple[int, int]:
    """
    Returns (maxFeePerGas, maxPriorityFeePerGas) in WEI using feeHistory.
    Applies buffers, floors, ceilings, and a global multiplier.
    """
    cfg = dict(DEFAULTS)
    if config:
        cfg.update({k: config.get(k, cfg[k]) for k in cfg.keys()})

    blocks = int(cfg["gas_blocks"])
    percentile = int(cfg["gas_reward_percentile"])
    headroom = float(cfg["gas_basefee_headroom"])
    prio_min = float(cfg["gas_priority_min_gwei"])
    prio_max = float(cfg["gas_priority_max_gwei"])
    ceiling = float(cfg["gas_ceiling_gwei"])
    mult = float(cfg["gas_multiplier"])
    prio_nudge = float(cfg["gas_extra_priority_gwei"])

    try:
        # Sample recent blocks
        # returns baseFeePerGas list (n+1 items) and reward matrix (n x len(percentiles))
        fh = w3.eth.fee_history(blocks, "latest", [percentile])
        base_fees = fh["baseFeePerGas"]              # list of wei
        rewards = fh["reward"]                       # list of [priority_fee_at_percentile] per block
        latest_base = int(base_fees[-1])             # wei
        # median-ish priority fee from recent blocks at chosen percentile
        if rewards:
            # average priority fee across sampled blocks
            prios = [int(r[0]) for r in rewards if r and len(r) > 0]
            if prios:
                avg_prio_wei = sum(prios) // len(prios)
            else:
                avg_prio_wei = _gwei_to_wei(w3, prio_min)
        else:
            avg_prio_wei = _gwei_to_wei(w3, prio_min)

        # Apply floors/caps in gwei space for clarity
        avg_prio_g = _wei_to_gwei(w3, avg_prio_wei) + prio_nudge
        avg_prio_g = max(prio_min, min(prio_max, avg_prio_g))

        # Build EIP-1559 fees
        base_with_headroom = int(latest_base * headroom)
        priority_wei = _gwei_to_wei(w3, avg_prio_g)
        max_fee = base_with_headroom + 2 * priority_wei  # generous headroom vs spikes

        # Apply global multiplier
        max_fee = int(max_fee * mult)
        priority_wei = int(priority_wei * mult)

        # Apply absolute ceiling
        ceiling_wei = _gwei_to_wei(w3, ceiling)
        if max_fee > ceiling_wei:
            max_fee = ceiling_wei
        if priority_wei > max_fee:
            # ensure invariant: maxFee >= priorityFee
            priority_wei = max_fee // 2

        return int(max_fee), int(priority_wei)

    except Exception as e:
        # Fallback to node's gas_price as a floor (legacy), then convert
        try:
            gp = int(w3.eth.gas_price)
        except Exception:
            gp = _gwei_to_wei(w3, 20.0)
        # assume 20% of gp as priority, clamp
        prio_g = max(prio_min, min(prio_max, _wei_to_gwei(w3, int(gp * 0.2))))
        prio_wei = _gwei_to_wei(w3, prio_g)
        max_fee = int(gp * 2 + prio_wei)
        # ceiling
        ceiling_wei = _gwei_to_wei(w3, ceiling)
        if max_fee > ceiling_wei:
            max_fee = ceiling_wei
        return int(max_fee), int(prio_wei)