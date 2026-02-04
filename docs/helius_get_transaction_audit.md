# Helius GET_TRANSACTION Usage Audit

## Summary

Helius bills and rate-limits by **each JSON-RPC method invocation**. A single HTTP request that contains a batch of 100 `getTransaction` calls counts as **100** requests toward the 300,000/day limit. This audit identifies why ~300k `getTransaction` calls occur daily and proposes fixes to reduce usage.

---

## Root Causes (by impact)

### 1. **Swap indexer (largest)**

- **Where:** `src/indexing/swap_indexer.py`
- **Behavior:** Every `index_interval_seconds` (default 60s), for **each tracked token**:
  - `getSignaturesForAddress(pool, limit=100)` → 1 RPC call
  - `_get_transactions_batch(signatures)` → **up to 100 `getTransaction`** (in batches of `transaction_batch_size`, default 25 → 4 batches = 100 getTransaction)
- **Volume:** With **N** tracked tokens: **N × 100 getTransaction per minute**.
  - N=30 → 30 × 100 × 60 × 24 ≈ **4.3M getTransaction/day** (far over 300k).
  - N=10 → ~1.44M/day.
- **Additional cost:** When resolving pools, `_find_pools_for_token` tries **program-accounts path first** (Raydium + Orca):
  - `getSignaturesForAddress` (×2) + `_get_transactions_batch(50)` (×2) + many `get_account_info` per candidate pool.
  - DexScreener is only used as fallback, so new tokens trigger heavy RPC use.

### 2. **15m candle RPC fallback**

- **Where:** `src/utils/market_data_fetcher.py` → `_get_solana_candles_from_rpc`
- **Behavior:** When indexed swaps are insufficient (or cache miss), for each token:
  - Up to **15 pages** of `getSignaturesForAddress` (15 calls)
  - Then **batches of 50** signatures → `get_transactions_by_signature` → **~500 getTransaction** per token (batch_size 25 → 20 batches × 25).
- **Volume:** Capped by `max_tokens_per_cycle_for_candles` (15), rate limit threshold, and cache. Still significant when many new tokens need candles (e.g. 15 × 500 = 7,500 getTransaction per cycle).

### 3. **Tracker undercounting**

- **Where:** `src/utils/api_tracker.py`, `helius_client._fetch_transactions`, `swap_indexer._get_transactions_batch`
- **Behavior:** Code calls `track_helius_call()` **once per batch** (one increment per HTTP batch).
- **Effect:** With `transaction_batch_size: 25`, the tracker undercounts by **25×** vs Helius. So 300k getTransaction on Helius ≈ **12k** in our tracker, making usage look low until the limit is hit.

### 4. **Reconciliation and one-off verification**

- **Where:** `src/core/helius_reconciliation.py`, `monitor_position.py`, `jupiter_lib.py`
- **Behavior:** `get_address_transactions(limit=200)` (ideally 1× `getTransactionsForAddress`), plus `get_transaction(entry_sig)` / `get_transaction(exit_sig)` per trade.
- **Volume:** Small (order of hundreds per day unless many positions).

---

## Solutions Implemented / Recommended

### Implemented in code

1. **Tracker matches Helius**
   - `track_helius_call(count=N)` so each batch increments by **N** (number of `getTransaction` in that batch).
   - Our “helius” count now reflects **RPC method invocations** (aligned with Helius billing).

2. **Swap indexer: DexScreener first for pool discovery**
   - In `_find_pools_for_token`, try **DexScreener first**, then fall back to program-accounts (Raydium/Orca).
   - Avoids `getSignaturesForAddress` + `_get_transactions_batch(50)` + many `get_account_info` for every new token.

3. **Swap indexer: cap on tracked tokens**
   - New config `swap_indexer.max_tracked_tokens` (e.g. 20). When at cap, new tokens are not added (or LRU eviction).
   - Prevents unbounded growth of tokens and thus of getTransaction volume.

### Recommended config / ops

4. **Increase index interval**
   - Set `swap_indexer.index_interval_seconds` to **120** or **180** to cut incremental indexing calls (e.g. 2–3× fewer getTransaction from the indexer).

5. **Prefer indexed candles**
   - Ensure tokens are added to the swap indexer and given time to backfill so candle builds use **indexed swaps** instead of RPC. Avoid forcing RPC path for many tokens in parallel.

6. **Optional: lower tokens per candle cycle**
   - Reduce `helius_15m_candle_policy.max_tokens_per_cycle_for_candles` (e.g. from 15 to 8–10) to lower peak getTransaction from the RPC candle path.

---

## Call sites reference

| Location | Method | Count semantics |
|----------|--------|------------------|
| `helius_client._rpc_request` | Any single RPC (e.g. getSignaturesForAddress) | 1 per call |
| `helius_client._fetch_transactions` | getTransaction (batched) | **N per batch** (N = batch size) |
| `swap_indexer._get_transactions_batch` | getTransaction (batched) | **N per batch** |
| `swap_indexer` get_signatures_for_address | getSignaturesForAddress | 1 per call |
| `market_data_fetcher` (RPC candle path) | getSignaturesForAddress + get_transactions_by_signature | 1 per page + N per tx batch |
| `order_flow_defense_solana` | Helius DEX API (different product) | 1 per request |

After the tracker change, `data/api_call_tracker.json` → `helius` should approximate the number of Helius RPC method invocations (including getTransaction) for the current day.
