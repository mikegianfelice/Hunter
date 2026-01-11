#!/usr/bin/env python3
"""
Unit tests for 15-minute candle quality validation with lenient mode support.
"""

import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.utils.market_data_fetcher import MarketDataFetcher


class TestCandleQualityValidation:
    """Test candle quality validation with strict and lenient modes"""
    
    def test_strict_reject_when_below_absolute_minimum(self):
        """Test that validation rejects when candles < 12 (absolute minimum)"""
        fetcher = MarketDataFetcher()
        
        # Create 11 candles (below absolute minimum of 12)
        base_time = int(time.time())
        candles = []
        for i in range(11):
            candles.append({
                'time': base_time - (11 - i) * 900,  # 15-minute intervals
                'open': 1.0,
                'high': 1.1,
                'low': 0.9,
                'close': 1.0,
                'volume': 100.0
            })
        
        metadata = {
            'swaps_processed': 50,
            'non_empty_candles': 11,
            'swaps_per_candle_avg': 4.5
        }
        
        hours = 6
        result = fetcher._validate_candle_quality(candles, metadata, hours)
        
        assert result is False, "Should reject when candles < absolute minimum (12)"
    
    def test_lenient_accept_with_12_candles(self):
        """Test that lenient mode accepts 12 candles with good quality metrics"""
        fetcher = MarketDataFetcher()
        
        # Create 12 candles (at absolute minimum, below strict requirement of 24 for 6h)
        base_time = int(time.time())
        candles = []
        for i in range(12):
            candles.append({
                'time': base_time - (12 - i) * 900,  # 15-minute intervals (3 hours coverage)
                'open': 1.0,
                'high': 1.1,
                'low': 0.9,
                'close': 1.0,
                'volume': 100.0
            })
        
        # Good quality metadata: enough swaps, density, coverage
        metadata = {
            'swaps_processed': 24,  # >= max(24, 12 * 1.5) = max(24, 18) = 24
            'non_empty_candles': 12,  # >= 12 * 0.5 = 6
            'swaps_per_candle_avg': 2.0  # >= 1.5
        }
        
        hours = 6
        result = fetcher._validate_candle_quality(candles, metadata, hours)
        
        assert result is True, "Lenient mode should accept 12 candles with good quality"
        
        # Verify time coverage (should be >= 3 hours)
        time_span = candles[-1]['time'] - candles[0]['time']
        time_span_hours = time_span / 3600.0
        assert time_span_hours >= 3.0, f"Time coverage {time_span_hours:.2f}h should be >= 3.0h"
    
    def test_lenient_reject_when_coverage_insufficient(self):
        """Test that lenient mode rejects when time coverage < 3 hours"""
        fetcher = MarketDataFetcher()
        
        # Create 12 candles but only 2 hours of coverage
        base_time = int(time.time())
        candles = []
        for i in range(12):
            candles.append({
                'time': base_time - (12 - i) * 600,  # 10-minute intervals (2 hours coverage)
                'open': 1.0,
                'high': 1.1,
                'low': 0.9,
                'close': 1.0,
                'volume': 100.0
            })
        
        metadata = {
            'swaps_processed': 30,
            'non_empty_candles': 12,
            'swaps_per_candle_avg': 2.5
        }
        
        hours = 6
        result = fetcher._validate_candle_quality(candles, metadata, hours)
        
        assert result is False, "Lenient mode should reject when coverage < 3 hours"
        
        # Verify time coverage is insufficient
        time_span = candles[-1]['time'] - candles[0]['time']
        time_span_hours = time_span / 3600.0
        assert time_span_hours < 3.0, f"Time coverage {time_span_hours:.2f}h should be < 3.0h for rejection"
    
    def test_lenient_reject_when_density_insufficient(self):
        """Test that lenient mode rejects when swap density < threshold"""
        fetcher = MarketDataFetcher()
        
        # Create 12 candles with good coverage but low density
        base_time = int(time.time())
        candles = []
        for i in range(12):
            candles.append({
                'time': base_time - (12 - i) * 900,  # 15-minute intervals (3 hours coverage)
                'open': 1.0,
                'high': 1.1,
                'low': 0.9,
                'close': 1.0,
                'volume': 100.0
            })
        
        metadata = {
            'swaps_processed': 24,
            'non_empty_candles': 12,
            'swaps_per_candle_avg': 1.0  # < 1.5 threshold
        }
        
        hours = 6
        result = fetcher._validate_candle_quality(candles, metadata, hours)
        
        assert result is False, "Lenient mode should reject when density < 1.5"
    
    def test_lenient_reject_when_non_empty_ratio_insufficient(self):
        """Test that lenient mode rejects when non-empty candles < 50% ratio"""
        fetcher = MarketDataFetcher()
        
        # Create 12 candles but only 5 are non-empty (< 50%)
        base_time = int(time.time())
        candles = []
        for i in range(12):
            volume = 100.0 if i < 5 else 0.0  # First 5 have volume
            candles.append({
                'time': base_time - (12 - i) * 900,
                'open': 1.0,
                'high': 1.1,
                'low': 0.9,
                'close': 1.0,
                'volume': volume
            })
        
        metadata = {
            'swaps_processed': 30,
            'non_empty_candles': 5,  # < 12 * 0.5 = 6
            'swaps_per_candle_avg': 2.5
        }
        
        hours = 6
        result = fetcher._validate_candle_quality(candles, metadata, hours)
        
        assert result is False, "Lenient mode should reject when non-empty < 50%"
    
    def test_strict_mode_full_validation(self):
        """Test that strict mode validates all thresholds correctly"""
        fetcher = MarketDataFetcher()
        
        # Create 24 candles (meets strict requirement for 6h)
        base_time = int(time.time())
        candles = []
        for i in range(24):
            candles.append({
                'time': base_time - (24 - i) * 900,  # 15-minute intervals
                'open': 1.0,
                'high': 1.1,
                'low': 0.9,
                'close': 1.0,
                'volume': 100.0
            })
        
        # All strict thresholds met
        metadata = {
            'swaps_processed': 50,  # >= 40
            'non_empty_candles': 24,  # >= 12
            'swaps_per_candle_avg': 2.0  # >= 1.5
        }
        
        hours = 6
        result = fetcher._validate_candle_quality(candles, metadata, hours)
        
        assert result is True, "Strict mode should accept when all thresholds met"
    
    def test_indexed_swaps_produces_15m_buckets(self):
        """Test that indexed swaps function outputs 15-minute buckets"""
        fetcher = MarketDataFetcher()
        
        base_time = int(time.time())
        swaps = [
            {'block_time': base_time, 'price_usd': 1.0, 'volume_usd': 100.0},
            {'block_time': base_time + 450, 'price_usd': 1.1, 'volume_usd': 200.0},  # Same 15m bucket
            {'block_time': base_time + 900, 'price_usd': 1.2, 'volume_usd': 300.0},  # Next 15m bucket
            {'block_time': base_time + 1800, 'price_usd': 1.3, 'volume_usd': 400.0},  # Next 15m bucket
        ]
        
        hours = 1
        start_time = base_time
        end_time = base_time + 3600
        
        candles = fetcher._process_swaps_to_candles_indexed(swaps, hours, start_time, end_time)
        
        # Should produce 3 candles (first two swaps in same 15m bucket)
        assert len(candles) == 3, f"Expected 3 candles, got {len(candles)}"
        
        # Verify buckets are 15-minute (900s) intervals
        bucket_times = [c['time'] for c in candles]
        expected_first_bucket = int((base_time // 900) * 900)
        assert bucket_times[0] == expected_first_bucket, "First bucket should align to 15-minute boundary"
        assert bucket_times[1] == expected_first_bucket + 900, "Second bucket should be 15 minutes later"
        assert bucket_times[2] == expected_first_bucket + 1800, "Third bucket should be 30 minutes later"
        
        # Verify OHLC updates correctly for multiple swaps in same interval
        first_candle = candles[0]
        assert first_candle['open'] == 1.0, "Open should be first swap price"
        assert first_candle['high'] == 1.1, "High should be max of swaps in bucket"
        assert first_candle['low'] == 1.0, "Low should be min of swaps in bucket"
        assert first_candle['close'] == 1.1, "Close should be last swap price in bucket"
        assert first_candle['volume'] == 300.0, "Volume should be sum of swaps in bucket"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
