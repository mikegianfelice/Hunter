#!/usr/bin/env python3
"""
Prepare ML Training Data from trade_log.csv
Fetches historical indicators for each trade and creates training dataset
"""

import sys
import os
import csv
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.utils.market_data_fetcher import market_data_fetcher
from src.utils.technical_indicators import technical_indicators

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# File paths
TRADE_LOG_FILE = project_root / "data" / "trade_log.csv"
OUTPUT_FILE = project_root / "data" / "ml_training_data.json"
API_CALL_LOG = project_root / "data" / "ml_api_calls.log"


def determine_chain_id(token_address: str) -> str:
    """Determine chain ID based on token address format"""
    # Solana addresses are 43-44 characters (base58)
    # Ethereum/Base addresses are 42 characters starting with 0x
    if len(token_address) in [43, 44]:
        return "solana"
    elif len(token_address) == 42 and token_address.startswith("0x"):
        # Could be Ethereum or Base - check config or default to ethereum
        return "ethereum"  # Default, can be refined
    else:
        return "solana"  # Default to Solana


def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse timestamp from trade_log.csv format"""
    try:
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        # Try alternative format
        return datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S")


def fetch_historical_candles(token_address: str, entry_timestamp: datetime, chain_id: str, hours: int = 24) -> Optional[List[Dict]]:
    """
    Fetch candlestick data for historical entry timestamp using Solana RPC
    Queries blockchain directly for DEX swap transactions (no external API needed)
    """
    try:
        logger.info(f"Fetching candlestick data for {token_address[:8]}... at {entry_timestamp}")
        
        # Convert entry timestamp to Unix timestamp
        entry_ts = entry_timestamp.timestamp()
        
        # Fetch candlestick data using historical timestamp
        # Uses Solana RPC to query DEX swap transactions directly
        candles = market_data_fetcher.get_candlestick_data(
            token_address=token_address,
            chain_id=chain_id,
            hours=hours,
            force_fetch=True,  # Force fresh fetch for historical accuracy
            target_timestamp=entry_ts  # Pass historical timestamp!
        )
        
        if not candles:
            logger.warning(f"No candlestick data for {token_address[:8]}...")
            return None
        
        # RPC returns data up to entry timestamp, but verify
        # Filter to ensure all candles are before or at entry time
        filtered_candles = []
        for candle in candles:
            candle_time = candle.get('time', candle.get('timestamp', 0))
            if isinstance(candle_time, str):
                # Try to parse if it's a string
                try:
                    candle_time = datetime.fromisoformat(candle_time.replace('Z', '+00:00')).timestamp()
                except:
                    candle_time = 0
            
            # Only include candles up to entry timestamp
            if candle_time <= entry_ts:
                filtered_candles.append(candle)
        
        # Sort by timestamp (oldest first)
        filtered_candles.sort(key=lambda x: x.get('time', x.get('timestamp', 0)))
        
        if len(filtered_candles) < 10:
            logger.warning(f"Insufficient historical candles ({len(filtered_candles)}) for {token_address[:8]}...")
            return None
        
        logger.info(f"✅ Got {len(filtered_candles)} candles for {token_address[:8]}...")
        return filtered_candles
        
    except Exception as e:
        logger.error(f"Error fetching candles for {token_address[:8]}...: {e}", exc_info=True)
        return None


def extract_features_at_entry(candles: List[Dict], entry_price: float, entry_timestamp: datetime) -> Dict:
    """Extract features at entry time from candlestick data"""
    try:
        # Calculate all technical indicators
        indicators = technical_indicators.calculate_all_indicators(candles)
        
        # Calculate momentum features
        if len(candles) >= 2:
            # 5-minute momentum (if we have 5m candles, otherwise approximate)
            recent_prices = [c.get('close', 0) for c in candles[-12:]]  # Last 12 candles (~1 hour if 5m)
            if len(recent_prices) >= 2:
                momentum_5m = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if recent_prices[0] > 0 else 0.0
            else:
                momentum_5m = 0.0
            
            # 1-hour momentum
            if len(candles) >= 12:
                hour_ago_price = candles[-12].get('close', entry_price)
                momentum_1h = (entry_price - hour_ago_price) / hour_ago_price if hour_ago_price > 0 else 0.0
            else:
                momentum_1h = 0.0
            
            # 24-hour momentum
            if len(candles) >= 24:
                day_ago_price = candles[0].get('close', entry_price)
                momentum_24h = (entry_price - day_ago_price) / day_ago_price if day_ago_price > 0 else 0.0
            else:
                momentum_24h = 0.0
        else:
            momentum_5m = 0.0
            momentum_1h = 0.0
            momentum_24h = 0.0
        
        # Calculate volume features
        volumes = [c.get('volume', 0) for c in candles if c.get('volume', 0) > 0]
        if volumes:
            volume_24h = sum(volumes)
            volume_avg = sum(volumes) / len(volumes) if len(volumes) > 0 else 0.0
            volume_spike = volumes[-1] / volume_avg if volume_avg > 0 else 1.0
        else:
            volume_24h = 0.0
            volume_avg = 0.0
            volume_spike = 1.0
        
        # Extract time features
        hour_of_day = entry_timestamp.hour
        day_of_week = entry_timestamp.weekday()  # 0 = Monday, 6 = Sunday
        
        # Build feature vector
        features = {
            # Price features
            'entry_price': entry_price,
            'price_momentum_5m': momentum_5m,
            'price_momentum_1h': momentum_1h,
            'price_momentum_24h': momentum_24h,
            
            # Technical indicators
            'rsi': indicators.get('rsi', 50.0),
            'macd_histogram': indicators.get('macd', {}).get('histogram', 0.0),
            'macd_signal': indicators.get('macd', {}).get('signal', 0.0),
            'bollinger_position': indicators.get('bollinger_bands', {}).get('position', 0.5),
            'bollinger_width': indicators.get('bollinger_bands', {}).get('width', 0.0),
            'vwap_distance': indicators.get('vwap', {}).get('vwap_distance_pct', 0.0),
            'is_above_vwap': indicators.get('vwap', {}).get('is_above_vwap', False),
            
            # Volume features
            'volume_24h': volume_24h,
            'volume_avg': volume_avg,
            'volume_spike': volume_spike,
            
            # Price action
            'trend_strength': indicators.get('price_action', {}).get('trend_strength', 0.5),
            'volatility': indicators.get('price_action', {}).get('volatility', 0.5),
            'momentum': indicators.get('price_action', {}).get('momentum', 0.5),
            
            # Moving averages
            'ma_20': indicators.get('moving_avg_20', entry_price),
            'ma_50': indicators.get('moving_avg_50', entry_price),
            'price_vs_ma20': (entry_price - indicators.get('moving_avg_20', entry_price)) / indicators.get('moving_avg_20', entry_price) if indicators.get('moving_avg_20', 0) > 0 else 0.0,
            
            # Time features
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week,
            
            # Additional derived features
            'price_change_24h': indicators.get('price_change_24h', 0.0),
        }
        
        return features
        
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return {}


def load_trade_log() -> List[Dict]:
    """Load trades from trade_log.csv"""
    trades = []
    
    if not TRADE_LOG_FILE.exists():
        logger.error(f"Trade log file not found: {TRADE_LOG_FILE}")
        return trades
    
    try:
        with open(TRADE_LOG_FILE, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                trades.append({
                    'timestamp': parse_timestamp(row['timestamp']),
                    'token': row['token'],
                    'entry_price': float(row['entry_price']),
                    'exit_price': float(row['exit_price']),
                    'pnl_pct': float(row['pnl_pct']),
                    'reason': row['reason']
                })
        
        logger.info(f"Loaded {len(trades)} trades from trade_log.csv")
        return trades
        
    except Exception as e:
        logger.error(f"Error loading trade log: {e}")
        return []


def prepare_training_data():
    """Main function to prepare training data"""
    logger.info("=" * 60)
    logger.info("Starting ML Training Data Preparation")
    logger.info("=" * 60)
    
    # Load trades
    trades = load_trade_log()
    if not trades:
        logger.error("No trades found. Exiting.")
        return
    
    logger.info(f"Processing {len(trades)} trades...")
    logger.info(f"This will make ~{len(trades)} Helius API calls")
    logger.info("Press Ctrl+C to cancel, or wait 5 seconds to continue...")
    time.sleep(5)
    
    training_data = []
    api_call_count = 0
    successful_extractions = 0
    failed_extractions = 0
    
    # Log API calls
    api_log = []
    
    for i, trade in enumerate(trades, 1):
        try:
            logger.info(f"\n[{i}/{len(trades)}] Processing trade: {trade['token'][:8]}...")
            logger.info(f"  Entry: {trade['timestamp']} @ ${trade['entry_price']:.6f}")
            logger.info(f"  Exit: ${trade['exit_price']:.6f} ({trade['pnl_pct']:.2f}%)")
            logger.info(f"  Reason: {trade['reason']}")
            
            # Determine chain
            chain_id = determine_chain_id(trade['token'])
            logger.info(f"  Chain: {chain_id}")
            
            # Fetch historical candlestick data
            start_time = time.time()
            candles = fetch_historical_candles(
                token_address=trade['token'],
                entry_timestamp=trade['timestamp'],
                chain_id=chain_id,
                hours=24
            )
            fetch_time = time.time() - start_time
            
            api_call_count += 1
            api_log_entry = {
                'trade_index': i,
                'token': trade['token'],
                'timestamp': trade['timestamp'].isoformat(),
                'fetch_time_seconds': fetch_time,
                'candles_count': len(candles) if candles else 0,
                'success': candles is not None
            }
            api_log.append(api_log_entry)
            
            if not candles:
                logger.warning(f"  ⚠️  No candlestick data available for {trade['token'][:8]}...")
                logger.debug(f"  API call took {fetch_time:.3f}s, returned {api_log_entry['candles_count']} candles")
                failed_extractions += 1
                continue
            
            # Extract features
            features = extract_features_at_entry(
                candles=candles,
                entry_price=trade['entry_price'],
                entry_timestamp=trade['timestamp']
            )
            
            if not features:
                logger.warning(f"  ⚠️  Failed to extract features")
                failed_extractions += 1
                continue
            
            # Create training sample
            # Label: 1 = hit take profit, 0 = hit stop loss
            label = 1 if trade['pnl_pct'] > 0 else 0
            
            training_sample = {
                'trade_index': i,
                'token': trade['token'],
                'entry_timestamp': trade['timestamp'].isoformat(),
                'entry_price': trade['entry_price'],
                'exit_price': trade['exit_price'],
                'pnl_pct': trade['pnl_pct'],
                'reason': trade['reason'],
                'label': label,
                'features': features,
                'chain_id': chain_id
            }
            
            training_data.append(training_sample)
            successful_extractions += 1
            
            logger.info(f"  ✅ Successfully extracted {len(features)} features")
            logger.info(f"  Label: {label} ({'TP' if label == 1 else 'SL'})")
            
            # Small delay to be respectful to API
            time.sleep(0.1)
            
        except KeyboardInterrupt:
            logger.warning("\n⚠️  Interrupted by user. Saving progress...")
            break
        except Exception as e:
            logger.error(f"  ❌ Error processing trade: {e}")
            failed_extractions += 1
            continue
    
    # Save training data
    logger.info("\n" + "=" * 60)
    logger.info("Saving training data...")
    
    output_data = {
        'metadata': {
            'total_trades': len(trades),
            'successful_extractions': successful_extractions,
            'failed_extractions': failed_extractions,
            'api_calls_made': api_call_count,
            'created_at': datetime.now().isoformat(),
            'source_file': str(TRADE_LOG_FILE)
        },
        'training_data': training_data,
        'api_call_log': api_log
    }
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"✅ Saved training data to {OUTPUT_FILE}")
    logger.info(f"   - {successful_extractions} successful extractions")
    logger.info(f"   - {failed_extractions} failed extractions")
    logger.info(f"   - {api_call_count} API calls made")
    
    # Save API call log separately
    API_CALL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(API_CALL_LOG, 'w') as f:
        json.dump(api_log, f, indent=2)
    
    logger.info(f"✅ Saved API call log to {API_CALL_LOG}")
    
    # Print summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("Summary Statistics:")
    logger.info("=" * 60)
    
    if training_data:
        tp_count = sum(1 for t in training_data if t['label'] == 1)
        sl_count = sum(1 for t in training_data if t['label'] == 0)
        win_rate = (tp_count / len(training_data)) * 100 if training_data else 0
        
        logger.info(f"Total samples: {len(training_data)}")
        logger.info(f"Take Profit (TP): {tp_count} ({tp_count/len(training_data)*100:.1f}%)")
        logger.info(f"Stop Loss (SL): {sl_count} ({sl_count/len(training_data)*100:.1f}%)")
        logger.info(f"Win Rate: {win_rate:.1f}%")
        
        # Feature statistics
        if training_data:
            sample_features = training_data[0]['features']
            logger.info(f"\nFeatures extracted ({len(sample_features)}):")
            for feature_name in sorted(sample_features.keys()):
                logger.info(f"  - {feature_name}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Training data preparation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        prepare_training_data()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

