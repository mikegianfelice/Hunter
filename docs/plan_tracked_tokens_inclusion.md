# Plan: Always Include Tracked Tokens in Token Selection

**Status: Implemented** (see `src/execution/enhanced_async_trading.py`)

## Objective
Ensure tokens from `minute_price_tracker.TRACKED_TOKENS` are always included in the token selection stage, regardless of filters or discovery results.

## Current Flow
1. `run_enhanced_trading_cycle()` fetches tokens via `fetch_trending_tokens_async()` for each chain
2. Tokens are filtered via `_filter_tokens_early()` 
3. Tokens are limited by `max_tokens_for_candles` policy
4. Tokens are processed in batches via `_process_token_batch()`

## Implementation Plan

### Step 1: Create Function to Fetch Tracked Token Data
**Location**: `src/execution/enhanced_async_trading.py`

Create `async def _fetch_tracked_tokens(self, chain: str) -> List[Dict]`:
- Import `TRACKED_TOKENS` from `src.storage.minute_price_tracker`
- Filter tracked tokens by chain (most are Solana)
- Fetch current token data from DexScreener/Jupiter API for each tracked token
- Return list of token dicts in same format as `_fetch_real_trending_tokens()`
- Handle errors gracefully (if API fails, still include token with basic data)

**Key Points**:
- Use Jupiter API for Solana tokens (faster, more reliable)
- Use DexScreener API as fallback
- Include all tracked tokens even if API fetch fails (use cached/last known data)
- Mark tokens with `_is_tracked_token: True` flag for priority handling

### Step 2: Inject Tracked Tokens into Token List
**Location**: `src/execution/enhanced_async_trading.py` in `run_enhanced_trading_cycle()`

After line 2482 (after `all_tokens.extend(result)`):
- Call `_fetch_tracked_tokens()` for each supported chain
- Merge tracked tokens into `all_tokens` list
- Deduplicate by address (tracked tokens take priority if duplicate)
- Log how many tracked tokens were added

**Implementation**:
```python
# Fetch tracked tokens and inject into list
tracked_tokens_all = []
for chain in self.config.chains.supported_chains:
    tracked = await self._fetch_tracked_tokens(chain)
    tracked_tokens_all.extend(tracked)

# Merge tracked tokens (deduplicate by address, tracked tokens take priority)
seen_addresses = {t.get('address', '').lower() for t in all_tokens}
for tracked_token in tracked_tokens_all:
    addr = tracked_token.get('address', '').lower()
    if addr and addr not in seen_addresses:
        all_tokens.append(tracked_token)
        seen_addresses.add(addr)
    elif addr:
        # Replace existing token with tracked version (tracked tokens have priority)
        for i, token in enumerate(all_tokens):
            if token.get('address', '').lower() == addr:
                all_tokens[i] = tracked_token
                break

log_info("trading.tracked_tokens_injected", 
         f"Injected {len(tracked_tokens_all)} tracked tokens into selection pool")
```

### Step 3: Ensure Tracked Tokens Bypass Early Filters
**Location**: `src/execution/enhanced_async_trading.py` in `_filter_tokens_early()`

Modify `_filter_tokens_early()` to:
- Check for `_is_tracked_token: True` flag
- Always include tracked tokens regardless of volume/liquidity thresholds
- Log when tracked tokens bypass filters

**Implementation**:
```python
def _filter_tokens_early(self, tokens: List[Dict]) -> List[Dict]:
    """Apply early filters, but always include tracked tokens"""
    filtered = []
    tracked_bypassed = 0
    
    for token in tokens:
        # Always include tracked tokens
        if token.get('_is_tracked_token', False):
            filtered.append(token)
            tracked_bypassed += 1
            continue
        
        # Apply normal filters for non-tracked tokens
        # ... existing filter logic ...
    
    if tracked_bypassed > 0:
        log_info("trading.tracked_tokens_bypassed_filters",
                f"{tracked_bypassed} tracked tokens bypassed early filters")
    
    return filtered
```

### Step 4: Ensure Tracked Tokens Bypass Candle Limiter
**Location**: `src/execution/enhanced_async_trading.py` in `run_enhanced_trading_cycle()`

Modify the candle limiter logic (around line 2512):
- Separate tracked tokens from regular tokens
- Always include tracked tokens in the final list
- Apply limiter only to non-tracked tokens
- Merge tracked tokens back after limiting

**Implementation**:
```python
# Separate tracked tokens before applying limiter
tracked_tokens_list = [t for t in filtered_tokens if t.get('_is_tracked_token', False)]
regular_tokens = [t for t in filtered_tokens if not t.get('_is_tracked_token', False)]

# Apply limiter only to regular tokens
if len(regular_tokens) > max_tokens_for_candles:
    regular_tokens.sort(key=self._get_token_sort_key, reverse=True)
    regular_tokens = regular_tokens[:max_tokens_for_candles]

# Merge tracked tokens back (always included)
filtered_tokens = tracked_tokens_list + regular_tokens

log_info("trading.candle_limiter",
        f"Limited to {len(filtered_tokens)} tokens ({len(tracked_tokens_list)} tracked + {len(regular_tokens)} regular)")
```

### Step 5: Add Logging and Monitoring
**Location**: Throughout implementation

Add structured logging:
- When tracked tokens are fetched
- When tracked tokens are injected
- When tracked tokens bypass filters
- When tracked tokens are included in final batch

**Log Events**:
- `trading.tracked_tokens.fetched` - Tracked tokens fetched successfully
- `trading.tracked_tokens.injected` - Tracked tokens added to selection pool
- `trading.tracked_tokens.bypassed_filters` - Tracked tokens bypassed early filters
- `trading.tracked_tokens.included` - Tracked tokens included in processing batch

## Benefits
1. **Consistency**: Tracked tokens always analyzed, ensuring price tracking continuity
2. **Priority**: Tracked tokens get priority in selection and processing
3. **Reliability**: Even if discovery APIs fail, tracked tokens are still included
4. **Monitoring**: Clear logging shows when tracked tokens are included

## Edge Cases to Handle
1. **API Failures**: If Jupiter/DexScreener fails for tracked token, use cached data or minimal token dict
2. **Duplicate Tokens**: Tracked tokens replace regular tokens if same address found
3. **Chain Mismatch**: Only fetch tracked tokens for supported chains
4. **Empty TRACKED_TOKENS**: Handle gracefully if no tracked tokens configured

## Testing Considerations
1. Test with empty `TRACKED_TOKENS` dict
2. Test with tracked tokens that fail API fetch
3. Test deduplication logic (tracked token replaces regular token)
4. Test filter bypass logic
5. Test candle limiter with tracked tokens
6. Verify tracked tokens appear in final batch processing

## Files to Modify
1. `src/execution/enhanced_async_trading.py`
   - Add `_fetch_tracked_tokens()` method
   - Modify `run_enhanced_trading_cycle()` to inject tracked tokens
   - Modify `_filter_tokens_early()` to bypass filters for tracked tokens
   - Modify candle limiter logic to always include tracked tokens

## Configuration (Optional)
Consider adding config option:
```yaml
token_selection:
  always_include_tracked_tokens: true  # Default: true
  tracked_tokens_priority: true        # Give tracked tokens priority in sorting
```
