#!/usr/bin/env python3
"""
Reconcile a sold position that wasn't properly removed from tracking.

Use this script when:
- A position was sold successfully (transaction confirmed on-chain)
- But the position wasn't removed from open_positions.json
- Or the trade status in performance_data.json wasn't updated to "sold"

This script:
1. Verifies the transaction was successful
2. Updates performance_data.json to mark trade as sold
3. Removes position from open_positions.json
4. Updates hunter_state.db

Example usage:
    python3 scripts/reconcile_sold_position.py <tx_hash> --token <token_address> --trade-id <trade_id>
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.utils.solana_transaction_analyzer import analyze_jupiter_transaction
from src.config.secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS
from src.storage.positions import load_positions, replace_positions, remove_position
from src.storage.performance import load_performance_data, replace_performance_data
from src.core.performance_tracker import performance_tracker
from src.utils.position_sync import create_position_key, resolve_token_address
from datetime import datetime

def reconcile_position(tx_hash: str, token_address: str = None, trade_id: str = None):
    """
    Reconcile a position that was sold but not properly removed.
    
    Args:
        tx_hash: Transaction hash of the sell
        token_address: Token address (optional, will try to find from trade_id)
        trade_id: Trade ID (optional, will try to find from token_address)
    """
    print(f"🔍 Reconciling position for transaction: {tx_hash}")
    
    # Step 1: Verify transaction was successful
    print("\n📊 Step 1: Verifying transaction...")
    try:
        tx_data = analyze_jupiter_transaction(
            SOLANA_RPC_URL,
            tx_hash,
            SOLANA_WALLET_ADDRESS,
            is_buy=False
        )
        
        if not tx_data.get('success'):
            print(f"❌ Transaction analysis failed: {tx_data.get('error', 'Unknown error')}")
            return False
        
        print(f"✅ Transaction verified as successful")
        print(f"   Gas fee: ${tx_data.get('gas_fee_usd', 0):.4f}")
        print(f"   Proceeds: ${tx_data.get('actual_proceeds_usd', 0):.4f}")
        print(f"   SOL received: {tx_data.get('sol_received', 0):.6f}")
    except Exception as e:
        print(f"⚠️ Could not verify transaction: {e}")
        print(f"   Proceeding with reconciliation anyway...")
    
    # Step 2: Find the trade in performance_data
    print("\n📊 Step 2: Finding trade in performance data...")
    perf_data = load_performance_data()
    trades = perf_data.get('trades', [])
    
    trade = None
    if trade_id:
        trade = next((t for t in trades if t.get('id') == trade_id), None)
        if trade:
            print(f"✅ Found trade by trade_id: {trade_id}")
    elif token_address:
        # Find by token address
        token_lower = token_address.lower()
        for t in trades:
            if t.get('address', '').lower() == token_lower and t.get('status') == 'open':
                trade = t
                trade_id = t.get('id')
                print(f"✅ Found open trade by address: {token_address}")
                break
    
    if not trade:
        print(f"⚠️ Could not find trade. Searching all trades...")
        # Try to find any trade with this token address
        if token_address:
            token_lower = token_address.lower()
            for t in trades:
                if t.get('address', '').lower() == token_lower:
                    trade = t
                    trade_id = t.get('id')
                    print(f"✅ Found trade (may be closed): {trade_id}")
                    break
    
    if not trade:
        print(f"❌ Could not find trade for token {token_address or 'unknown'}")
        print(f"   Available trades with status 'open':")
        for t in trades:
            if t.get('status') == 'open':
                print(f"   - {t.get('id')}: {t.get('symbol')} ({t.get('address', '')[:16]}...)")
        return False
    
    trade_id = trade.get('id')
    token_address = trade.get('address')
    entry_price = float(trade.get('entry_price', 0))
    position_size = float(trade.get('position_size_usd', 0))
    
    print(f"   Trade ID: {trade_id}")
    print(f"   Symbol: {trade.get('symbol')}")
    print(f"   Address: {token_address}")
    print(f"   Entry price: ${entry_price:.6f}")
    print(f"   Position size: ${position_size:.2f}")
    
    # Step 3: Update performance_data.json
    print("\n📊 Step 3: Updating performance data...")
    
    # Calculate PnL if we have exit data
    exit_price = entry_price  # Default to entry price if we can't determine
    if tx_data.get('actual_proceeds_usd') and position_size > 0:
        # Estimate exit price from proceeds
        tokens_sold = tx_data.get('token_transfers', [])
        for transfer in tokens_sold:
            if transfer.get('direction') == 'send':
                tokens_amount = transfer.get('amount', 0)
                if tokens_amount > 0:
                    exit_price = tx_data.get('actual_proceeds_usd', 0) / tokens_amount
                    break
    
    pnl_usd = (exit_price - entry_price) / entry_price * position_size if entry_price > 0 else 0
    
    # Update trade record
    trade['status'] = 'sold'
    trade['exit_time'] = datetime.now().isoformat()
    trade['exit_price'] = exit_price
    trade['pnl_usd'] = pnl_usd
    trade['pnl_percent'] = (pnl_usd / position_size * 100) if position_size > 0 else 0
    trade['sell_tx_hash'] = tx_hash
    
    # Add fee data if available
    if tx_data.get('exit_gas_fee_usd'):
        trade['exit_gas_fee_usd'] = tx_data.get('exit_gas_fee_usd')
    elif tx_data.get('gas_fee_usd'):
        trade['exit_gas_fee_usd'] = tx_data.get('gas_fee_usd')
    
    if tx_data.get('actual_proceeds_usd'):
        trade['actual_proceeds_usd'] = tx_data.get('actual_proceeds_usd')
    
    # Calculate total fees and after-fee PnL
    entry_fees = float(trade.get('entry_gas_fee_usd', 0) or 0)
    exit_fees = float(trade.get('exit_gas_fee_usd', 0) or 0)
    total_fees = entry_fees + exit_fees
    proceeds = float(trade.get('actual_proceeds_usd', 0) or 0)
    entry_cost = float(trade.get('entry_amount_usd_actual', 0) or trade.get('position_size_usd', 0) or 0)
    
    if entry_cost > 0 and proceeds > 0:
        trade['total_fees_usd'] = total_fees
        trade['pnl_after_fees_usd'] = proceeds - entry_cost - total_fees
        trade['pnl_after_fees_percent'] = (trade['pnl_after_fees_usd'] / entry_cost * 100) if entry_cost > 0 else 0
    
    replace_performance_data(perf_data)
    print(f"✅ Updated performance data for trade {trade_id}")
    
    # Step 4: Remove from open_positions.json and database
    print("\n📊 Step 4: Removing from open positions...")
    positions = load_positions()
    
    # Find position by trade_id or token address
    position_key_to_remove = None
    for key, pos_data in positions.items():
        if isinstance(pos_data, dict):
            # Check by trade_id
            if pos_data.get('trade_id') == trade_id:
                position_key_to_remove = key
                break
            # Check by address
            pos_address = resolve_token_address(key, pos_data)
            if pos_address.lower() == token_address.lower():
                position_key_to_remove = key
                break
        else:
            # Legacy format - check key directly
            if key.lower() == token_address.lower():
                position_key_to_remove = key
                break
    
    if position_key_to_remove:
        remove_position(position_key_to_remove)
        print(f"✅ Removed position: {position_key_to_remove}")
    else:
        print(f"⚠️ Position not found in open_positions (may already be removed)")
        # List current positions for debugging
        if positions:
            print(f"   Current positions:")
            for key in positions.keys():
                print(f"   - {key}")
    
    # Step 5: Update performance tracker
    print("\n📊 Step 5: Updating performance tracker...")
    try:
        # Use the performance tracker to update the trade
        performance_tracker.log_trade_exit(
            trade_id,
            exit_price,
            pnl_usd,
            status='sold',
            additional_data={
                'sell_tx_hash': tx_hash,
                'exit_gas_fee_usd': tx_data.get('gas_fee_usd', 0),
                'actual_proceeds_usd': tx_data.get('actual_proceeds_usd', 0),
            }
        )
        print(f"✅ Updated performance tracker")
    except Exception as e:
        print(f"⚠️ Error updating performance tracker: {e}")
    
    # Step 6: Log to trade_log.csv
    print("\n📊 Step 6: Logging to trade_log.csv...")
    try:
        from src.monitoring.monitor_position import log_trade
        pnl_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
        log_trade(token_address, entry_price, exit_price, f"reconciliation_sell_{pnl_pct:.2f}%")
        print(f"✅ Logged to trade_log.csv")
    except Exception as e:
        print(f"⚠️ Error logging to trade_log.csv: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Reconciliation complete!")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Reconcile a sold position')
    parser.add_argument('tx_hash', help='Transaction hash of the sell')
    parser.add_argument('--token', help='Token address (optional)')
    parser.add_argument('--trade-id', help='Trade ID (optional)')
    
    args = parser.parse_args()
    
    reconcile_position(args.tx_hash, args.token, args.trade_id)

