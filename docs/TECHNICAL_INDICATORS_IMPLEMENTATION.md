# Technical Indicators Implementation Guide

## Overview

This implementation adds enhanced technical indicators (RSI, MACD, Bollinger Bands) with Helius API integration for Solana candlestick data, optimized for API rate limits.

## Components Implemented

### 1. Technical Indicators Module (`src/utils/technical_indicators.py`)
- **RSI (Relative Strength Index)**: 14-period default
- **MACD**: Fast EMA (12), Slow EMA (26), Signal (9)
- **Bollinger Bands**: 20-period, 2 standard deviations
- **Volume Profile**: Point of Control (POC), Value Area High/Low
- **Price Action Patterns**: Trend strength, volatility, momentum, support/resistance

### 2. Enhanced Market Data Fetcher (`src/utils/market_data_fetcher.py`)
- **Helius Integration**: Uses Helius API (30k/day limit) for Solana candlestick data
- **Smart Caching**: 
  - Helius: 5-minute cache
  - CoinGecko: 1-hour cache
- **API Tracking**: Tracks calls separately for Helius and CoinGecko
- **Fallback Strategy**: Uses price_memory if Helius fails (no API calls)

### 3. Enhanced Price Predictor (`src/ai/ai_price_predictor.py`)
- **Quality Filtering**: Only fetches candlestick data for tokens with $100k+ volume/liquidity
- **Enhanced Features**: RSI, MACD, Bollinger Bands, volume profile, price action
- **Graceful Fallback**: Uses simple indicators if candlestick data unavailable

### 4. A/B Testing Framework (`src/utils/ab_testing.py`)
- **Weight Configurations**: Tests different feature weight combinations
- **Performance Tracking**: Records win rate, PnL per configuration
- **Best Config Selection**: Automatically identifies best performing variant

### 5. Monitoring Scripts
- **`scripts/check_api_usage.py`**: Monitor API call usage
- **`scripts/analyze_ab_tests.py`**: Analyze A/B test performance

## Configuration

Added to `config.yaml`:

```yaml
# Enhanced Technical Analysis
enable_enhanced_technical_indicators: true
technical_indicators:
  rsi_period: 14
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9
  bollinger_period: 20
  bollinger_std: 2
  volume_profile_bins: 20
  min_candles_required: 10

# API Rate Limiting
api_rate_limiting:
  helius_max_daily: 30000
  coingecko_max_daily: 330
  helius_cache_minutes: 5
  coingecko_cache_hours: 1
  enable_api_call_tracking: true
  warn_at_calls_remaining: 50

# Technical Indicators Settings
technical_indicators_settings:
  enable_enhanced_indicators: true
  only_fetch_for_quality_tokens: true
  min_volume_for_candlestick_fetch: 100000
  min_liquidity_for_candlestick_fetch: 100000
  fallback_to_simple_indicators: true

# A/B Testing
enable_ab_testing: false
ab_testing:
  min_trades_per_variant: 10
  evaluation_period_hours: 168
  auto_switch_to_best: false
```

## API Usage

### Expected Daily Usage:
- **Helius**: ~100-200 calls/day (well under 30k limit)
- **CoinGecko**: ~10-30 calls/day (well under 330 limit)

### Strategy:
1. **Solana tokens**: Use Helius API (30k/day limit - use freely!)
2. **Ethereum/Base tokens**: Use The Graph (free) first, CoinGecko as fallback
3. **Quality filtering**: Only fetch for tokens with $100k+ volume/liquidity
4. **Caching**: 5-minute cache for Helius, 1-hour for CoinGecko
5. **Fallback**: Use price_memory if APIs fail (no API calls)

## Usage

### Check API Usage:
```bash
python scripts/check_api_usage.py
```

### Analyze A/B Tests:
```bash
python scripts/analyze_ab_tests.py
```

## Files Created/Modified

### New Files:
- `src/utils/technical_indicators.py`
- `src/utils/ab_testing.py`
- `scripts/check_api_usage.py`
- `scripts/analyze_ab_tests.py`
- `docs/TECHNICAL_INDICATORS_IMPLEMENTATION.md`

### Modified Files:
- `src/utils/market_data_fetcher.py` - Added Helius integration
- `src/ai/ai_price_predictor.py` - Added enhanced technical indicators
- `config.yaml` - Added configuration sections

## Dependencies

All dependencies are already in `requirements.txt`:
- `pandas` - For technical indicator calculations
- `numpy` - For mathematical operations

## Testing

The implementation includes:
- Error handling and graceful fallbacks
- Logging for debugging
- Default values when data unavailable
- Quality filtering to minimize API calls

## Next Steps

1. **Enable A/B Testing**: Set `enable_ab_testing: true` in config.yaml
2. **Monitor API Usage**: Run `check_api_usage.py` regularly
3. **Review Performance**: Use `analyze_ab_tests.py` after sufficient trades
4. **Adjust Weights**: Modify weight configurations based on results

