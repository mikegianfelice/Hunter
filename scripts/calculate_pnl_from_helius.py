#!/usr/bin/env python3
"""
Calculate PnL directly from Helius transaction data.
This is the ground truth - on-chain data only, no reliance on performance_data.json.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Cache file for efficient updates
CACHE_FILE = project_root / "data" / "helius_wallet_value_cache.json"

from src.config.secrets import HELIUS_API_KEY, SOLANA_WALLET_ADDRESS
from src.utils.helius_client import HeliusClient, BALANCE_EPSILON
from src.core.helius_reconciliation import (
    USDC_MINT,
    _aggregate_token_amount,
    _aggregate_native_transfers,
)

try:
    from src.utils.solana_transaction_analyzer import get_sol_price_usd
except Exception:
    def get_sol_price_usd() -> float:
        return 150.0  # Fallback


def load_helius_cache() -> Optional[Dict]:
    """Load cached Helius transaction data"""
    if not CACHE_FILE.exists():
        return None
    try:
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
            # Convert ISO strings back to datetime objects for last_update
            if 'last_update' in cache:
                cache['last_update'] = datetime.fromisoformat(cache['last_update'])
            return cache
    except Exception as e:
        print(f"⚠️  Error loading cache: {e}")
        return None


def save_helius_cache(cache_data: Dict):
    """Save cached Helius transaction data"""
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Convert datetime objects to ISO strings for JSON serialization
        cache_copy = cache_data.copy()
        if 'last_update' in cache_copy and isinstance(cache_copy['last_update'], datetime):
            cache_copy['last_update'] = cache_copy['last_update'].isoformat()
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache_copy, f, indent=2, default=str)
    except Exception as e:
        print(f"⚠️  Error saving cache: {e}")


def calculate_pnl_from_helius(start_date_str: str, end_date_str: str, initial_wallet: float = 1000.0):
    """
    Calculate PnL directly from Helius transactions for a date range.
    Matches buy and sell transactions to calculate realized PnL.
    """
    if not HELIUS_API_KEY or not SOLANA_WALLET_ADDRESS:
        print("❌ Missing HELIUS_API_KEY or SOLANA_WALLET_ADDRESS")
        return None
    
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    end_date = (datetime.strptime(end_date_str, '%Y-%m-%d') + timedelta(days=1)).replace(tzinfo=timezone.utc)
    
    print(f"\n{'='*70}")
    print(f"💰 CALCULATING PnL FROM HELIUS TRANSACTIONS")
    print(f"   Period: {start_date_str} to {end_date_str}")
    print(f"{'='*70}\n")
    
    # Track API calls
    from src.utils.api_tracker import get_tracker
    api_tracker = get_tracker()
    initial_call_count = api_tracker.get_count('helius')
    
    client = HeliusClient(HELIUS_API_KEY)
    wallet = SOLANA_WALLET_ADDRESS.lower()
    sol_price = get_sol_price_usd()
    
    # Fetch transactions (may need to paginate for older dates)
    print("📡 Fetching transactions from Helius...")
    all_transactions = []
    before_signature = None
    max_pages = 10  # Fetch up to 2000 transactions
    
    for page in range(max_pages):
        transactions = client.get_address_transactions(
            SOLANA_WALLET_ADDRESS,
            limit=200,
            before=before_signature,
        )
        
        if not transactions:
            break
        
        # Filter by date range
        page_txs = []
        for tx in transactions:
            ts = tx.get('timestamp')
            if not ts:
                continue
            tx_time = datetime.fromtimestamp(ts, tz=timezone.utc)
            if start_date <= tx_time < end_date:
                page_txs.append(tx)
            elif tx_time < start_date:
                # Gone too far back
                break
        
        all_transactions.extend(page_txs)
        
        if not page_txs or tx_time < start_date:
            break
        
        # Prepare for next page
        before_signature = transactions[-1].get('signature')
        if not before_signature:
            break
    
    print(f"   Found {len(all_transactions)} transactions in date range\n")
    
    # Track positions: token -> list of (buy_amount, buy_cost, buy_time, buy_tx)
    positions: Dict[str, List[Tuple[float, float, datetime, Dict]]] = defaultdict(list)
    
    # Track completed trades for PnL calculation
    completed_trades: List[Dict] = []
    
    # Track deposits and withdrawals
    deposits: List[Dict] = []
    withdrawals: List[Dict] = []
    
    # Process transactions chronologically
    all_transactions.sort(key=lambda x: x.get('timestamp', 0))
    
    for tx in all_transactions:
        transfers = tx.get('tokenTransfers', [])
        if not transfers:
            continue
        
        tx_time = datetime.fromtimestamp(tx.get('timestamp', 0), tz=timezone.utc)
        sol_fee = float(tx.get('fee', 0)) / 1_000_000_000
        sol_fee_usd = sol_fee * sol_price
        
        # Get all unique token mints in this transaction (excluding USDC)
        token_mints = set()
        for transfer in transfers:
            mint = (transfer.get('mint') or '').lower()
            if mint and mint != USDC_MINT.lower():
                token_mints.add(mint)
        
        # Process each token mint in the transaction
        for mint in token_mints:
            # Check if we're receiving this token (BUY)
            token_received = _aggregate_token_amount(transfers, SOLANA_WALLET_ADDRESS, mint, incoming=True)
            # Check if we're sending this token (SELL)
            token_sent = _aggregate_token_amount(transfers, SOLANA_WALLET_ADDRESS, mint, incoming=False)
            
            # Check USDC transfers
            usdc_spent = _aggregate_token_amount(transfers, SOLANA_WALLET_ADDRESS, USDC_MINT.lower(), incoming=False)
            usdc_received = _aggregate_token_amount(transfers, SOLANA_WALLET_ADDRESS, USDC_MINT.lower(), incoming=True)
            
            # BUY: Receiving tokens, sending USDC
            if token_received > BALANCE_EPSILON and usdc_spent > 0:
                # This is a buy
                buy_cost_usd = usdc_spent + sol_fee_usd
                positions[mint].append((token_received, buy_cost_usd, tx_time, tx))
                
            # SELL: Sending tokens, receiving USDC
            elif token_sent > BALANCE_EPSILON and usdc_received > 0:
                # This is a sell
                proceeds_usd = usdc_received - sol_fee_usd
                sell_amount = token_sent
                
                # Match with buys (FIFO)
                remaining_to_sell = sell_amount
                while remaining_to_sell > BALANCE_EPSILON and positions[mint]:
                    buy_amount, buy_cost, buy_time, buy_tx = positions[mint][0]
                    
                    if buy_amount <= remaining_to_sell:
                        # Full buy consumed
                        portion = 1.0
                        sold = buy_amount
                        positions[mint].pop(0)
                    else:
                        # Partial buy consumed
                        portion = remaining_to_sell / buy_amount
                        sold = remaining_to_sell
                        positions[mint][0] = (
                            buy_amount - remaining_to_sell,
                            buy_cost * (1 - portion),
                            buy_time,
                            buy_tx
                        )
                    
                    # Calculate PnL for this portion
                    portion_cost = buy_cost * portion
                    portion_proceeds = proceeds_usd * (sold / sell_amount)
                    pnl = portion_proceeds - portion_cost
                    
                    completed_trades.append({
                        'mint': mint,
                        'buy_time': buy_time,
                        'sell_time': tx_time,
                        'buy_amount': sold,
                        'sell_amount': sold,
                        'cost_usd': portion_cost,
                        'proceeds_usd': portion_proceeds,
                        'pnl_usd': pnl,
                        'buy_tx': buy_tx.get('signature'),
                        'sell_tx': tx.get('signature'),
                    })
                    
                    remaining_to_sell -= sold
        
        # Check for deposits/withdrawals (USDC transfers not part of buy/sell)
        # Only check if we haven't already processed this transaction as a buy/sell
        usdc_received_total = _aggregate_token_amount(transfers, SOLANA_WALLET_ADDRESS, USDC_MINT.lower(), incoming=True)
        usdc_sent_total = _aggregate_token_amount(transfers, SOLANA_WALLET_ADDRESS, USDC_MINT.lower(), incoming=False)
        
        # Check if this transaction involved token trading
        has_token_trade = False
        for mint in token_mints:
            token_received = _aggregate_token_amount(transfers, SOLANA_WALLET_ADDRESS, mint, incoming=True)
            token_sent = _aggregate_token_amount(transfers, SOLANA_WALLET_ADDRESS, mint, incoming=False)
            if token_received > BALANCE_EPSILON or token_sent > BALANCE_EPSILON:
                has_token_trade = True
                break
        
        # If receiving USDC but NOT trading tokens, it's a deposit
        if usdc_received_total > BALANCE_EPSILON and not has_token_trade:
            sol_fee = float(tx.get('fee', 0)) / 1_000_000_000
            sol_fee_usd = sol_fee * sol_price
            deposit_amount = usdc_received_total - sol_fee_usd  # Net after fees
            if deposit_amount > BALANCE_EPSILON:
                deposits.append({
                    'time': tx_time,
                    'amount': deposit_amount,
                    'type': 'deposit',
                    'tx': tx.get('signature')
                })
        
        # If sending USDC but NOT trading tokens, it's potentially a withdrawal
        # NOTE: Automatic withdrawal detection is DISABLED to prevent false positives
        # Small USDC transfers (<$10) are likely fees or other transfers, not withdrawals
        # If you need to record actual withdrawals, add them manually to the cache
        MIN_WITHDRAWAL_THRESHOLD = 10.0  # Minimum $10 to be considered a withdrawal
        if usdc_sent_total > MIN_WITHDRAWAL_THRESHOLD and not has_token_trade:
            sol_fee = float(tx.get('fee', 0)) / 1_000_000_000
            sol_fee_usd = sol_fee * sol_price
            withdrawal_amount = usdc_sent_total + sol_fee_usd  # Include fees in withdrawal
            if withdrawal_amount > MIN_WITHDRAWAL_THRESHOLD:
                # Log potential withdrawal for manual review (but don't auto-add)
                print(f"⚠️  Potential withdrawal detected: ${withdrawal_amount:.2f} at {tx_time} (tx: {tx.get('signature', 'N/A')[:20]}...)")
                print(f"   Automatic withdrawal detection is disabled. If this is a real withdrawal, add it manually to the cache.")
                # Don't automatically add - require manual confirmation to prevent false positives
                # withdrawals.append({
                #     'time': tx_time,
                #     'amount': withdrawal_amount,
                #     'type': 'withdrawal',
                #     'tx': tx.get('signature')
                # })
    
    # Calculate total PnL (only trading performance, not deposits/withdrawals)
    trading_pnl = sum(trade['pnl_usd'] for trade in completed_trades)
    total_deposits = sum(d['amount'] for d in deposits)
    total_withdrawals = sum(w['amount'] for w in withdrawals)
    total_pnl = trading_pnl  # PnL should only reflect trading performance
    
    # Calculate adjusted capital base for percentage return calculation
    adjusted_capital_base = initial_wallet + total_deposits - total_withdrawals
    pnl_percentage = (trading_pnl / adjusted_capital_base * 100) if adjusted_capital_base > 0 else 0
    
    print("💰 REALIZED PnL (from Helius transactions):")
    print("-" * 70)
    
    # Group by token for display
    trades_by_token = defaultdict(list)
    for trade in completed_trades:
        trades_by_token[trade['mint']].append(trade)
    
    for mint, token_trades in sorted(trades_by_token.items(), key=lambda x: x[1][0]['buy_time']):
        token_total = sum(t['pnl_usd'] for t in token_trades)
        first_trade = token_trades[0]
        buy_time = first_trade['buy_time'].strftime('%Y-%m-%d')
        sell_time = token_trades[-1]['sell_time'].strftime('%Y-%m-%d')
        
        print(f"  {mint[:8]}... | ${token_total:8.2f} | {len(token_trades)} trades | Buy: {buy_time} | Sell: {sell_time}")
    
    print(f"\n  Trading PnL: ${trading_pnl:.2f}")
    print(f"  Deposits: ${total_deposits:.2f} ({len(deposits)} transactions)")
    print(f"  Withdrawals: ${total_withdrawals:.2f} ({len(withdrawals)} transactions)")
    print(f"  Adjusted Capital Base: ${adjusted_capital_base:.2f} (initial + deposits - withdrawals)")
    print(f"  Total Realized PnL: ${total_pnl:.2f} ({pnl_percentage:+.2f}%)")
    print(f"  Number of Completed Trades: {len(completed_trades)}")
    
    # Summary (expected wallet includes deposits/withdrawals but PnL calculation doesn't)
    expected_wallet = initial_wallet + trading_pnl + total_deposits - total_withdrawals
    actual_wallet = 963.00  # User's actual wallet
    difference = expected_wallet - actual_wallet
    
    print(f"\n{'='*70}")
    print(f"📊 SUMMARY")
    print(f"{'='*70}")
    adjusted_capital_base = initial_wallet + total_deposits - total_withdrawals
    pnl_percentage = (trading_pnl / adjusted_capital_base * 100) if adjusted_capital_base > 0 else 0
    
    print(f"  Initial Wallet ({start_date_str}): ${initial_wallet:.2f}")
    print(f"  Trading PnL: ${trading_pnl:.2f}")
    print(f"  Deposits: ${total_deposits:.2f}")
    print(f"  Withdrawals: ${total_withdrawals:.2f}")
    print(f"  Adjusted Capital Base: ${adjusted_capital_base:.2f} (initial + deposits - withdrawals)")
    print(f"  Trading PnL: ${trading_pnl:.2f} ({pnl_percentage:+.2f}%)")
    print(f"  Expected Wallet ({end_date_str}): ${expected_wallet:.2f}")
    print(f"  Actual Wallet ({end_date_str}): ${actual_wallet:.2f}")
    print(f"  Difference: ${difference:.2f}")
    
    # Show API call count
    final_call_count = api_tracker.get_count('helius')
    calls_used = final_call_count - initial_call_count
    print(f"\n  📡 Helius API Calls Used: {calls_used:,} (out of 30,000 daily limit)")
    print(f"     Remaining today: {30_000 - final_call_count:,}")
    
    print(f"\n  📝 This uses ONLY on-chain Helius transaction data")
    print(f"     (matches buy/sell transactions using FIFO)")
    print(f"     (PnL reflects trading performance only, not deposits/withdrawals)")
    print(f"{'='*70}\n")
    
    return {
        'realized_pnl': total_pnl,
        'trading_pnl': trading_pnl,
        'total_deposits': total_deposits,
        'total_withdrawals': total_withdrawals,
        'deposits': deposits,
        'withdrawals': withdrawals,
        'expected_wallet': expected_wallet,
        'actual_wallet': actual_wallet,
        'difference': difference,
        'trades': completed_trades
    }


def calculate_wallet_value_over_time_from_helius(
    start_date_str: str, 
    end_date_str: str, 
    initial_wallet: float = 200.0,
    force_refresh: bool = False
):
    """
    Calculate wallet value over time from Helius transactions with caching.
    First run: Full historical pull
    Subsequent runs: Only fetch new transactions since last update
    
    Args:
        start_date_str: Start date for calculation (YYYY-MM-DD)
        end_date_str: End date for calculation (YYYY-MM-DD)
        initial_wallet: Initial wallet value
        force_refresh: If True, ignore cache and do full refresh
    """
    if not HELIUS_API_KEY or not SOLANA_WALLET_ADDRESS:
        print("❌ Missing HELIUS_API_KEY or SOLANA_WALLET_ADDRESS")
        return None
    
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    end_date = (datetime.strptime(end_date_str, '%Y-%m-%d') + timedelta(days=1)).replace(tzinfo=timezone.utc)
    
    print(f"\n{'='*70}")
    print(f"📈 CALCULATING WALLET VALUE OVER TIME FROM HELIUS")
    print(f"   Period: {start_date_str} to {end_date_str}")
    print(f"   Initial Wallet: ${initial_wallet:.2f}")
    print(f"{'='*70}\n")
    
    # Track API calls
    from src.utils.api_tracker import get_tracker
    api_tracker = get_tracker()
    initial_call_count = api_tracker.get_count('helius')
    
    # Load cache first to check if we can skip API calls
    cache = load_helius_cache() if not force_refresh else None
    use_cache = cache is not None
    
    # Check if cache is recent enough to skip API calls entirely
    # We should skip API calls only if BOTH:
    # 1. Cache was updated recently (< 1 hour ago)
    # 2. Latest trade in cache is also recent (< 1 hour ago)
    # This ensures we don't skip fetching when cache was updated but contains old data
    cache_is_recent = False
    if use_cache:
        last_update = cache['last_update']
        if isinstance(last_update, str):
            last_update = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
        cache_age = datetime.now(timezone.utc) - last_update
        
        # Check latest trade age
        cached_trades = cache.get('completed_trades', [])
        latest_trade_time = None
        if cached_trades:
            for trade in cached_trades:
                sell_time = trade.get('sell_time')
                if sell_time:
                    if isinstance(sell_time, str):
                        sell_time_dt = datetime.fromisoformat(sell_time.replace('Z', '+00:00'))
                    else:
                        sell_time_dt = datetime.fromtimestamp(sell_time, tz=timezone.utc)
                    if latest_trade_time is None or sell_time_dt > latest_trade_time:
                        latest_trade_time = sell_time_dt
        
        # Only skip API calls if cache is recent AND latest trade is recent
        cache_update_recent = cache_age < timedelta(hours=1)
        latest_trade_recent = latest_trade_time and (datetime.now(timezone.utc) - latest_trade_time) < timedelta(hours=1)
        cache_is_recent = cache_update_recent and latest_trade_recent
        
        if cache_is_recent:
            print(f"📦 Using recent cache (updated {cache_age.total_seconds()/60:.1f} minutes ago)")
            if latest_trade_time:
                trade_age = (datetime.now(timezone.utc) - latest_trade_time).total_seconds()/60
                print(f"   Latest trade: {trade_age:.1f} minutes ago")
            print("   Skipping API calls to avoid rate limits - using cached data only\n")
        elif cache_update_recent and not latest_trade_recent:
            print(f"📦 Cache was updated {cache_age.total_seconds()/60:.1f} minutes ago")
            if latest_trade_time:
                trade_age_hours = (datetime.now(timezone.utc) - latest_trade_time).total_seconds()/3600
                print(f"   BUT latest trade is {trade_age_hours:.1f} hours old")
            print("   Will fetch new transactions to check for completed trades...\n")
        else:
            print(f"📦 Cache exists but is {cache_age.total_seconds()/3600:.1f} hours old")
            print("   Will fetch new transactions...\n")
    
    # Add a brief delay before starting to avoid rate limit conflicts
    # The fetch loop will handle rate limits if they occur
    import time
    if not cache_is_recent:
        print("📡 Proceeding with transaction fetch...")
        print("   Rate limits will be handled automatically during fetch\n")
        time.sleep(2.0)  # Brief delay to avoid immediate rate limit conflicts
    
    # If using recent cache, skip API calls entirely
    if cache_is_recent:
        # Use cached data directly
        last_update = cache['last_update']
        if isinstance(last_update, str):
            last_update = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
        processed_signatures = set(cache.get('processed_signatures', []))
        cached_trades = cache.get('completed_trades', [])
        cached_deposits = cache.get('deposits', [])
        cached_withdrawals = cache.get('withdrawals', [])
        all_transactions = []  # No new transactions to fetch
        fetch_error = None
        fetch_start = None  # Not needed when using cache only
    elif use_cache:
        last_update = cache['last_update']
        if isinstance(last_update, str):
            last_update = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
        processed_signatures = set(cache.get('processed_signatures', []))
        cached_trades = cache.get('completed_trades', [])
        cached_deposits = cache.get('deposits', [])
        cached_withdrawals = cache.get('withdrawals', [])
        
        print(f"📦 Loaded cache from {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Cached transactions: {len(processed_signatures)}")
        print(f"   Cached trades: {len(cached_trades)}")
        print(f"   Fetching new transactions since last update...\n")
        
        # Determine fetch_start: use the latest trade date or last_update, whichever is earlier
        # This ensures we don't miss trades that happened between cache updates
        latest_trade_date = None
        if cached_trades:
            for trade in cached_trades:
                sell_time = trade.get('sell_time')
                if sell_time:
                    if isinstance(sell_time, str):
                        sell_time = datetime.fromisoformat(sell_time.replace('Z', '+00:00'))
                    if not latest_trade_date or sell_time > latest_trade_date:
                        latest_trade_date = sell_time
        
        # Use the earlier of: latest trade date or last_update (with 5 min buffer)
        if latest_trade_date:
            fetch_start = min(latest_trade_date - timedelta(minutes=5), last_update - timedelta(minutes=5))
        else:
            fetch_start = last_update - timedelta(minutes=5)  # 5 min buffer for safety
        
        client = HeliusClient(HELIUS_API_KEY)
        wallet = SOLANA_WALLET_ADDRESS.lower()
        sol_price = get_sol_price_usd()
        
        # Fetch transactions (only new ones if using cache)
        print("📡 Fetching transactions from Helius...")
        all_transactions = []
        before_signature = None
        # When using cache, reduce pages significantly to minimize API calls (was 10, now 3-5)
        # Only fetch a few pages since new transactions are recent
        max_pages = 3 if use_cache else 50
        fetch_error = None
    else:
        print("📦 No cache found - performing full historical pull\n")
        processed_signatures = set()
        cached_trades = []
        cached_deposits = []
        cached_withdrawals = []
        fetch_start = start_date
        
        client = HeliusClient(HELIUS_API_KEY)
        wallet = SOLANA_WALLET_ADDRESS.lower()
        sol_price = get_sol_price_usd()
        
        # Fetch transactions
        print("📡 Fetching transactions from Helius...")
        all_transactions = []
        before_signature = None
        max_pages = 50  # Full refresh needs more pages
        fetch_error = None
    
    # Only fetch if not using recent cache
    if not cache_is_recent:
        # Add delay between requests to avoid rate limits
        import time
        
        try:
            for page in range(max_pages):
                # Add delay between requests to respect rate limits (especially important when fetching many pages)
                # Increased delay when using cache (fewer pages needed) to be more conservative
                # Increased delays to be more conservative with Helius rate limits
                if page > 0:
                    delay = 5.0 if use_cache else 4.0  # Increased delays: 5s when using cache, 4s for full refresh
                    print(f"   Waiting {delay:.1f}s before fetching page {page + 1}/{max_pages}...")
                    time.sleep(delay)
                
                try:
                    transactions = client.get_address_transactions(
                        SOLANA_WALLET_ADDRESS,
                        limit=200,
                        before=before_signature,
                    )
                except Exception as e:
                    fetch_error = e
                    error_str = str(e).lower()
                    
                    # Check if it's a rate limit error (429)
                    # Check both HTTPError status code and error message
                    is_rate_limit = False
                    if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                        is_rate_limit = e.response.status_code == 429
                    else:
                        is_rate_limit = "429" in error_str or "rate limit" in error_str or "too many requests" in error_str
                    
                    if is_rate_limit:
                        # If we hit rate limit, use cached data if available
                        if use_cache:
                            print(f"\n⚠️ Rate limit (429) hit (page {page + 1}/{max_pages})")
                            print(f"   Helius API rate limit exceeded - retries exhausted")
                            if all_transactions:
                                print(f"   ✅ Using cached data + {len(all_transactions)} newly fetched transactions")
                            else:
                                print(f"   ✅ Using cached data only (no new transactions fetched)")
                            print(f"   💡 Tip: Wait a few minutes and run again, or use --force-refresh to retry\n")
                            break  # Stop fetching, use cache
                        elif page < max_pages - 1:
                            # Wait longer for rate limits before retrying (exponential backoff)
                            wait_time = min(90, 20 * (page + 1))  # Increased: up to 90 seconds, more conservative
                            print(f"\n⚠️ Rate limit (429) hit (page {page + 1}/{max_pages})")
                            print(f"   Waiting {wait_time}s before retry (exponential backoff)...")
                            print(f"   This helps avoid further rate limit hits\n")
                            time.sleep(wait_time)
                            continue  # Retry this page
                        else:
                            # Last page and rate limited - fail gracefully
                            print(f"\n⚠️ Rate limit (429) hit on final page after all retries")
                            print(f"   Cannot proceed without transaction data")
                            print(f"   💡 Tip: Wait a few minutes and run again\n")
                            raise
                    
                    print(f"⚠️ Error fetching transactions from Helius (page {page + 1}): {e}")
                    # If we have cached data, we can still proceed with cached data
                    if use_cache:
                        if all_transactions:
                            print("   Using cached data and previously fetched transactions...")
                        else:
                            print("   Using cached data only (no new transactions fetched)...")
                        break
                    # For first run without cache, we need at least some data
                    if not use_cache and page == 0:
                        print("❌ Cannot proceed without any transaction data")
                        raise
                    # For subsequent pages without cache, continue with what we have
                    break
                
                if not transactions:
                    break
                
                # Filter by date range and skip already processed transactions
                page_txs = []
                earliest_in_page = None
                new_tx_count = 0
                
                for tx in transactions:
                    ts = tx.get('timestamp')
                    if not ts:
                        continue
                    tx_time = datetime.fromtimestamp(ts, tz=timezone.utc)
                    tx_sig = tx.get('signature')
                    
                    # Skip if already processed (when using cache)
                    if use_cache and tx_sig and tx_sig in processed_signatures:
                        continue
                    
                    # Only fetch transactions since fetch_start (or in date range for first run)
                    if use_cache:
                        if tx_time >= fetch_start and tx_time < end_date:
                            page_txs.append(tx)
                            new_tx_count += 1
                    else:
                        if start_date <= tx_time < end_date:
                            page_txs.append(tx)
                    
                    if earliest_in_page is None or tx_time < earliest_in_page:
                        earliest_in_page = tx_time
                
                all_transactions.extend(page_txs)
                
                # Stop if we've gone too far back
                if use_cache:
                    # When using cache, stop if we've gone before fetch_start
                    if earliest_in_page and earliest_in_page < fetch_start:
                        break
                else:
                    # First run: stop if before start_date
                    if earliest_in_page and earliest_in_page < start_date:
                        break
                
                # If using cache and no new transactions found, we're done
                if use_cache and new_tx_count == 0 and page_txs:
                    break
                
                # Prepare for next page
                before_signature = transactions[-1].get('signature')
                if not before_signature:
                    break
        
        except Exception as e:
            # If we have cached data, we can still proceed even if we couldn't fetch new transactions
            if use_cache:
                if all_transactions:
                    print(f"⚠️ Error during transaction fetch, but continuing with {len(all_transactions)} cached/new transactions: {e}")
                else:
                    print(f"⚠️ Error fetching new transactions, but will use existing cached data: {e}")
                fetch_error = e
                # Continue processing with cached data - don't raise
            else:
                # No cached data available, this is a critical error
                print(f"❌ Critical error fetching transactions: {e}")
                raise
    
    # Print summary of what we're processing
    if cache_is_recent:
        print(f"   Using cached data only (no new transactions fetched)")
        print(f"   Cached transactions: {len(processed_signatures)}")
        print(f"   Cached trades: {len(cached_trades)}\n")
    else:
        print(f"   Found {len(all_transactions)} new transactions in date range\n")
    
    # Track positions: token -> list of (buy_amount, buy_cost, buy_time, buy_tx)
    positions: Dict[str, List[Tuple[float, float, datetime, Dict]]] = defaultdict(list)
    
    # Track completed trades for PnL calculation
    completed_trades: List[Dict] = []
    
    # Track deposits and withdrawals
    deposits: List[Dict] = []
    withdrawals: List[Dict] = []
    
    # Track new signatures for caching
    new_signatures = set()
    
    # Process transactions chronologically
    all_transactions.sort(key=lambda x: x.get('timestamp', 0))
    
    for tx in all_transactions:
        tx_sig = tx.get('signature')
        if tx_sig:
            new_signatures.add(tx_sig)  # Track signature for caching
        
        transfers = tx.get('tokenTransfers', [])
        if not transfers:
            continue
        
        tx_time = datetime.fromtimestamp(tx.get('timestamp', 0), tz=timezone.utc)
        sol_fee = float(tx.get('fee', 0)) / 1_000_000_000
        sol_fee_usd = sol_fee * sol_price
        
        # Get all unique token mints in this transaction (excluding USDC)
        token_mints = set()
        for transfer in transfers:
            mint = (transfer.get('mint') or '').lower()
            if mint and mint != USDC_MINT.lower():
                token_mints.add(mint)
        
        # Process each token mint in the transaction
        for mint in token_mints:
            # Check if we're receiving this token (BUY)
            token_received = _aggregate_token_amount(transfers, SOLANA_WALLET_ADDRESS, mint, incoming=True)
            # Check if we're sending this token (SELL)
            token_sent = _aggregate_token_amount(transfers, SOLANA_WALLET_ADDRESS, mint, incoming=False)
            
            # Check USDC transfers
            usdc_spent = _aggregate_token_amount(transfers, SOLANA_WALLET_ADDRESS, USDC_MINT.lower(), incoming=False)
            usdc_received = _aggregate_token_amount(transfers, SOLANA_WALLET_ADDRESS, USDC_MINT.lower(), incoming=True)
            
            # BUY: Receiving tokens, sending USDC
            if token_received > BALANCE_EPSILON and usdc_spent > 0:
                # This is a buy
                buy_cost_usd = usdc_spent + sol_fee_usd
                positions[mint].append((token_received, buy_cost_usd, tx_time, tx))
                
            # SELL: Sending tokens, receiving USDC
            elif token_sent > BALANCE_EPSILON and usdc_received > 0:
                # This is a sell
                proceeds_usd = usdc_received - sol_fee_usd
                sell_amount = token_sent
                
                # Match with buys (FIFO)
                remaining_to_sell = sell_amount
                while remaining_to_sell > BALANCE_EPSILON and positions[mint]:
                    buy_amount, buy_cost, buy_time, buy_tx = positions[mint][0]
                    
                    if buy_amount <= remaining_to_sell:
                        # Full buy consumed
                        portion = 1.0
                        sold = buy_amount
                        positions[mint].pop(0)
                    else:
                        # Partial buy consumed
                        portion = remaining_to_sell / buy_amount
                        sold = remaining_to_sell
                        positions[mint][0] = (
                            buy_amount - remaining_to_sell,
                            buy_cost * (1 - portion),
                            buy_time,
                            buy_tx
                        )
                    
                    # Calculate PnL for this portion
                    portion_cost = buy_cost * portion
                    portion_proceeds = proceeds_usd * (sold / sell_amount)
                    pnl = portion_proceeds - portion_cost
                    
                    completed_trades.append({
                        'mint': mint,
                        'buy_time': buy_time,
                        'sell_time': tx_time,
                        'buy_amount': sold,
                        'sell_amount': sold,
                        'cost_usd': portion_cost,
                        'proceeds_usd': portion_proceeds,
                        'pnl_usd': pnl,
                        'buy_tx': buy_tx.get('signature'),
                        'sell_tx': tx.get('signature'),
                    })
                    
                    remaining_to_sell -= sold
        
        # Check for deposits/withdrawals (USDC transfers not part of buy/sell)
        # Only check if we haven't already processed this transaction as a buy/sell
        usdc_received_total = _aggregate_token_amount(transfers, SOLANA_WALLET_ADDRESS, USDC_MINT.lower(), incoming=True)
        usdc_sent_total = _aggregate_token_amount(transfers, SOLANA_WALLET_ADDRESS, USDC_MINT.lower(), incoming=False)
        
        # Check if this transaction involved token trading
        has_token_trade = False
        for mint in token_mints:
            token_received = _aggregate_token_amount(transfers, SOLANA_WALLET_ADDRESS, mint, incoming=True)
            token_sent = _aggregate_token_amount(transfers, SOLANA_WALLET_ADDRESS, mint, incoming=False)
            if token_received > BALANCE_EPSILON or token_sent > BALANCE_EPSILON:
                has_token_trade = True
                break
        
        # If receiving USDC but NOT trading tokens, it's a deposit
        if usdc_received_total > BALANCE_EPSILON and not has_token_trade:
            sol_fee = float(tx.get('fee', 0)) / 1_000_000_000
            sol_fee_usd = sol_fee * sol_price
            deposit_amount = usdc_received_total - sol_fee_usd  # Net after fees
            if deposit_amount > BALANCE_EPSILON:
                deposits.append({
                    'time': tx_time,
                    'amount': deposit_amount,
                    'type': 'deposit',
                    'tx': tx.get('signature')
                })
        
        # If sending USDC but NOT trading tokens, it's potentially a withdrawal
        # NOTE: Automatic withdrawal detection is DISABLED to prevent false positives
        # Small USDC transfers (<$10) are likely fees or other transfers, not withdrawals
        # If you need to record actual withdrawals, add them manually to the cache
        MIN_WITHDRAWAL_THRESHOLD = 10.0  # Minimum $10 to be considered a withdrawal
        if usdc_sent_total > MIN_WITHDRAWAL_THRESHOLD and not has_token_trade:
            sol_fee = float(tx.get('fee', 0)) / 1_000_000_000
            sol_fee_usd = sol_fee * sol_price
            withdrawal_amount = usdc_sent_total + sol_fee_usd  # Include fees in withdrawal
            if withdrawal_amount > MIN_WITHDRAWAL_THRESHOLD:
                # Log potential withdrawal for manual review (but don't auto-add)
                print(f"⚠️  Potential withdrawal detected: ${withdrawal_amount:.2f} at {tx_time} (tx: {tx.get('signature', 'N/A')[:20]}...)")
                print(f"   Automatic withdrawal detection is disabled. If this is a real withdrawal, add it manually to the cache.")
                # Don't automatically add - require manual confirmation to prevent false positives
                # withdrawals.append({
                #     'time': tx_time,
                #     'amount': withdrawal_amount,
                #     'type': 'withdrawal',
                #     'tx': tx.get('signature')
                # })
    
    # Combine cached and new data
    if use_cache:
        # Merge new trades/deposits/withdrawals with cached ones
        all_trades = cached_trades + completed_trades
        all_deposits = cached_deposits + deposits
        all_withdrawals = cached_withdrawals + withdrawals
        all_processed_signatures = processed_signatures | new_signatures
    else:
        all_trades = completed_trades
        all_deposits = deposits
        all_withdrawals = withdrawals
        all_processed_signatures = new_signatures
    
    # Build wallet value over time from ALL events (cached + new)
    wallet_value_events = []
    current_wallet = initial_wallet
    
    # Add starting point
    wallet_value_events.append({
        'time': start_date,
        'wallet_value': initial_wallet,
        'type': 'start',
        'pnl': 0.0
    })
    
    # Combine all events (trades, deposits, withdrawals) and sort chronologically
    all_events = []
    
    # Add completed trades (convert sell_time if it's a string)
    for trade in all_trades:
        sell_time = trade['sell_time']
        if isinstance(sell_time, str):
            sell_time = datetime.fromisoformat(sell_time.replace('Z', '+00:00'))
        all_events.append({
            'time': sell_time,
            'type': 'sell',
            'pnl': trade['pnl_usd'],
            'amount': trade['pnl_usd'],
            'mint': trade['mint']
        })
    
    # Add deposits (convert time if it's a string)
    for deposit in all_deposits:
        deposit_time = deposit['time']
        if isinstance(deposit_time, str):
            deposit_time = datetime.fromisoformat(deposit_time.replace('Z', '+00:00'))
        all_events.append({
            'time': deposit_time,
            'type': 'deposit',
            'amount': deposit['amount'],
            'pnl': deposit['amount']  # Deposits increase wallet
        })
    
    # Add withdrawals (convert time if it's a string)
    for withdrawal in all_withdrawals:
        withdrawal_time = withdrawal['time']
        if isinstance(withdrawal_time, str):
            withdrawal_time = datetime.fromisoformat(withdrawal_time.replace('Z', '+00:00'))
        all_events.append({
            'time': withdrawal_time,
            'type': 'withdrawal',
            'amount': withdrawal['amount'],
            'pnl': -withdrawal['amount']  # Withdrawals decrease wallet
        })
    
    # Sort all events chronologically
    all_events.sort(key=lambda x: x['time'])
    
    # Process events in chronological order
    for event in all_events:
        if event['type'] == 'sell':
            current_wallet += event['pnl']
            wallet_value_events.append({
                'time': event['time'],
                'wallet_value': current_wallet,
                'type': 'sell',
                'pnl': event['pnl'],
                'mint': event.get('mint')
            })
        elif event['type'] == 'deposit':
            current_wallet += event['amount']
            wallet_value_events.append({
                'time': event['time'],
                'wallet_value': current_wallet,
                'type': 'deposit',
                'pnl': event['amount'],
                'amount': event['amount']
            })
        elif event['type'] == 'withdrawal':
            current_wallet -= event['amount']
            wallet_value_events.append({
                'time': event['time'],
                'wallet_value': current_wallet,
                'type': 'withdrawal',
                'pnl': -event['amount'],
                'amount': event['amount']
            })
    
    # Calculate statistics (using all_trades, all_deposits, all_withdrawals)
    total_deposits = sum(d['amount'] for d in all_deposits)
    total_withdrawals = sum(w['amount'] for w in all_withdrawals)
    trading_pnl = sum(t['pnl_usd'] for t in all_trades)
    total_pnl = trading_pnl  # PnL should only reflect trading performance, not deposits/withdrawals
    
    # Calculate adjusted capital base for percentage return calculation
    adjusted_capital_base = initial_wallet + total_deposits - total_withdrawals
    pnl_percentage = (trading_pnl / adjusted_capital_base * 100) if adjusted_capital_base > 0 else 0
    
    print("📊 Wallet Value Over Time Summary:")
    print("-" * 70)
    print(f"  Initial Wallet ({start_date_str}): ${initial_wallet:.2f}")
    print(f"  Final Wallet ({end_date_str}): ${current_wallet:.2f}")
    print(f"  Trading PnL: ${trading_pnl:.2f} ({pnl_percentage:+.2f}% on ${adjusted_capital_base:.2f} capital)")
    print(f"  Deposits: ${total_deposits:.2f} ({len(all_deposits)} transactions)")
    print(f"  Withdrawals: ${total_withdrawals:.2f} ({len(all_withdrawals)} transactions)")
    print(f"  Adjusted Capital Base: ${adjusted_capital_base:.2f} (initial + deposits - withdrawals)")
    print(f"  Total Completed Trades: {len(all_trades)}")
    print(f"  Time Points: {len(wallet_value_events)}")
    if use_cache:
        print(f"  New Transactions Processed: {len(new_signatures)}")
    
    # Show API call count
    final_call_count = api_tracker.get_count('helius')
    calls_used = final_call_count - initial_call_count
    print(f"\n  📡 Helius API Calls Used: {calls_used:,} (out of 30,000 daily limit)")
    print(f"     Remaining today: {30_000 - final_call_count:,}")
    
    # Show first few and last few events
    if wallet_value_events:
        print(f"\n  First event: {wallet_value_events[0]['time'].strftime('%Y-%m-%d %H:%M:%S')} - ${wallet_value_events[0]['wallet_value']:.2f}")
        if len(wallet_value_events) > 1:
            print(f"  Last event: {wallet_value_events[-1]['time'].strftime('%Y-%m-%d %H:%M:%S')} - ${wallet_value_events[-1]['wallet_value']:.2f}")
        
        # Show sample of events (first 5 and last 5)
        print(f"\n  Sample events (first 5):")
        for i, event in enumerate(wallet_value_events[:5]):
            pnl_str = f" (PnL: ${event['pnl']:+.2f})" if event['type'] == 'sell' else ""
            print(f"    {event['time'].strftime('%Y-%m-%d %H:%M:%S')}: ${event['wallet_value']:.2f}{pnl_str}")
        
        if len(wallet_value_events) > 10:
            print(f"  ... ({len(wallet_value_events) - 10} more events) ...")
            print(f"  Sample events (last 5):")
            for event in wallet_value_events[-5:]:
                pnl_str = f" (PnL: ${event['pnl']:+.2f})" if event['type'] == 'sell' else ""
                print(f"    {event['time'].strftime('%Y-%m-%d %H:%M:%S')}: ${event['wallet_value']:.2f}{pnl_str}")
    
    # Save updated cache
    print("\n💾 Saving cache...")
    save_helius_cache({
        'last_update': datetime.now(timezone.utc),
        'processed_signatures': list(all_processed_signatures),
        'completed_trades': all_trades,
        'deposits': all_deposits,
        'withdrawals': all_withdrawals,
        'initial_wallet': initial_wallet,
        'start_date': start_date_str
    })
    print(f"   ✅ Cache saved ({len(all_processed_signatures)} transactions)")
    
    print(f"\n{'='*70}\n")
    
    return {
        'time_points': [e['time'] for e in wallet_value_events],
        'wallet_values': [e['wallet_value'] for e in wallet_value_events],
        'events': wallet_value_events,
        'initial_wallet': initial_wallet,
        'final_wallet': current_wallet,
        'total_pnl': total_pnl,
        'trading_pnl': trading_pnl,
        'total_deposits': total_deposits,
        'total_withdrawals': total_withdrawals,
        'deposits': all_deposits,
        'withdrawals': all_withdrawals,
        'completed_trades': all_trades,
        'cache_used': use_cache,
        'new_transactions': len(new_signatures) if 'new_signatures' in locals() else len(all_transactions)
    }


if __name__ == '__main__':
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate PnL from Helius transactions')
    parser.add_argument('--wallet-value', action='store_true', 
                       help='Calculate wallet value over time (for charts)')
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--initial-wallet', type=float, default=None,
                       help='Initial wallet value')
    parser.add_argument('--force-refresh', action='store_true',
                       help='Force full refresh (ignore cache)')
    
    args = parser.parse_args()
    
    if args.wallet_value:
        # Wallet value calculation with caching
        start_date = args.start_date or "2025-11-17"
        end_date = args.end_date or "2025-12-30"
        initial = args.initial_wallet or 200.0
        
        result = calculate_wallet_value_over_time_from_helius(
            start_date, 
            end_date, 
            initial,
            force_refresh=args.force_refresh
        )
        if result:
            print("\n✅ Wallet value calculation complete!")
            print(f"   Generated {len(result['time_points'])} time points")
            if result.get('cache_used'):
                print(f"   Used cache: {result['new_transactions']} new transactions processed")
            else:
                print(f"   Full refresh: {result['new_transactions']} transactions processed")
    else:
        # Default: calculate PnL for Dec 24-30 period
        start_date = args.start_date or "2025-12-24"
        end_date = args.end_date or "2025-12-30"
        initial = args.initial_wallet or 1000.0
        calculate_pnl_from_helius(start_date, end_date, initial)
