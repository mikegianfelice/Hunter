#!/usr/bin/env python3
"""
Script to get AI score for a specific token symbol or by address
Usage: python get_token_score.py USELESS
       python get_token_score.py USOR USoRyaQjch6E18nCdDvWoRgTo6osQs9MUd8JXEsspWR
"""

import asyncio
import sys
import os
import json
import logging
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Enable INFO logging so we see quality score breakdown (e.g. ai.score_breakdown for USOR)
logging.basicConfig(level=logging.INFO, format="%(message)s")
for name in ("src.ai", "src.monitoring.structured_logger"):
    logging.getLogger(name).setLevel(logging.INFO)

from src.utils.http_utils import get_json
from src.ai.ai_integration_engine import analyze_token_ai, get_ai_engine

def _pair_to_token_data(pair: Dict, symbol: str) -> Dict[str, Any]:
    """Convert DexScreener pair to our token_data format."""
    volume_24h = float(pair.get('volume', {}).get('h24', 0) if isinstance(pair.get('volume'), dict) else pair.get('volume', 0) or 0)
    volume_1h = float(pair.get('volume', {}).get('h1', 0) if isinstance(pair.get('volume'), dict) else 0)
    volume_change_1h = None
    if volume_24h > 0 and volume_1h > 0:
        volume_change_1h = (volume_1h * 24) / volume_24h - 1
    elif volume_24h > 0:
        volume_change_1h = -1.0
    price_change = pair.get('priceChange', {}) or {}
    h24 = price_change.get('h24') if isinstance(price_change, dict) else None
    price_change_24h_pct = float(h24 or pair.get('priceChange24h', 0) or 0)
    price_change_24h = price_change_24h_pct / 100.0  # store as decimal
    return {
        "symbol": pair.get('baseToken', {}).get('symbol', symbol),
        "address": pair.get('baseToken', {}).get('address', ''),
        "chainId": (pair.get('chainId') or 'solana').lower(),
        "priceUsd": float(pair.get('priceUsd', 0)),
        "volume24h": volume_24h,
        "liquidity": float(pair.get('liquidity', {}).get('usd', 0) if isinstance(pair.get('liquidity'), dict) else pair.get('liquidity', 0) or 0),
        "marketCap": float(pair.get('fdv', 0)),
        "priceChange24h": price_change_24h,
        "priceChange5m": float(price_change.get('m5', 0) or 0) / 100.0 if price_change.get('m5') is not None else None,
        "priceChange1h": float(price_change.get('h1', 0) or 0) / 100.0 if price_change.get('h1') is not None else None,
        "volumeChange1h": volume_change_1h,
        "holders": 0,
        "transactions24h": 0,
        "social_mentions": 0,
        "news_sentiment": 0.5,
        "timestamp": pair.get('pairCreatedAt', ''),
        "technical_indicators": {},
        "on_chain_metrics": {},
    }

async def fetch_token_data_by_address(address: str) -> Optional[Dict[str, Any]]:
    """Fetch token data from DexScreener by token address."""
    print(f"🔍 Fetching data for address {address[:8]}... from DexScreener...")
    url = f"https://api.dexscreener.com/latest/dex/tokens/{address}"
    try:
        data = get_json(url)
        if not data or 'pairs' not in data or not data['pairs']:
            print("❌ No pairs found for this address")
            return None
        pairs = data['pairs']
        best_pair = max(pairs, key=lambda p: float(p.get('liquidity', {}).get('usd', 0) or 0))
        symbol = best_pair.get('baseToken', {}).get('symbol', 'UNKNOWN')
        token_data = _pair_to_token_data(best_pair, symbol)
        print(f"✅ Found: {token_data['symbol']} on {token_data['chainId']}")
        print(f"   Price: ${token_data['priceUsd']:.8f}")
        print(f"   Volume 24h: ${token_data['volume24h']:,.2f}")
        print(f"   Liquidity: ${token_data['liquidity']:,.2f}")
        print(f"   Price Change 24h: {token_data['priceChange24h']*100:.2f}%")
        return token_data
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

async def fetch_token_data(symbol: str, address: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetch token data from DexScreener by address or by symbol search."""
    if address:
        return await fetch_token_data_by_address(address)
    print(f"🔍 Fetching data for {symbol} from DexScreener...")
    url = f"https://api.dexscreener.com/latest/dex/search/?q={symbol}"
    
    try:
        data = get_json(url)
        if not data or 'pairs' not in data:
            print(f"❌ No data found for {symbol}")
            return None
        
        pairs = data.get('pairs', [])
        if not pairs:
            print(f"❌ No pairs found for {symbol}")
            return None
        
        # Find the best pair (prefer USDC/USDT pairs, highest liquidity)
        best_pair = None
        best_liquidity = 0
        
        for pair in pairs:
            if pair.get('symbol', '').upper() == symbol.upper():
                liq = float(pair.get('liquidity', {}).get('usd', 0) if isinstance(pair.get('liquidity'), dict) else pair.get('liquidity', 0))
                if liq > best_liquidity:
                    best_liquidity = liq
                    best_pair = pair
        
        if not best_pair:
            # Use first pair if no exact match
            best_pair = pairs[0]
        
        # Extract volume data for volume change calculation
        volume_24h = float(best_pair.get('volume', {}).get('h24', 0) if isinstance(best_pair.get('volume'), dict) else best_pair.get('volume24h', 0))
        volume_1h = float(best_pair.get('volume', {}).get('h1', 0) if isinstance(best_pair.get('volume'), dict) else 0)
        
        # Calculate volume change 1h as decimal (0.1 = 10% increase)
        # Compare current hour volume to 24h average hourly rate
        volume_change_1h = None
        if volume_24h > 0 and volume_1h > 0:
            # Calculate decimal change: e.g., 0.1 = 10% increase, -0.5 = 50% decrease
            volume_change_1h = (volume_1h * 24) / volume_24h - 1
        elif volume_24h > 0:
            # If h1 volume is 0 but h24 exists, it's a -1.0 change (100% decrease)
            volume_change_1h = -1.0
        
        # Convert DexScreener format to our token format
        token_data = {
            "symbol": best_pair.get('baseToken', {}).get('symbol', symbol),
            "address": best_pair.get('baseToken', {}).get('address', ''),
            "chainId": best_pair.get('chainId', 'ethereum').lower(),
            "priceUsd": float(best_pair.get('priceUsd', 0)),
            "volume24h": volume_24h,
            "liquidity": float(best_pair.get('liquidity', {}).get('usd', 0) if isinstance(best_pair.get('liquidity'), dict) else best_pair.get('liquidity', 0)),
            "marketCap": float(best_pair.get('fdv', 0)),
            "priceChange24h": float(best_pair.get('priceChange', {}).get('h24', 0) if isinstance(best_pair.get('priceChange'), dict) else best_pair.get('priceChange24h', 0)) / 100.0 if best_pair.get('priceChange', {}).get('h24') else 0.0,
            "priceChange5m": float(best_pair.get('priceChange', {}).get('m5', 0) if isinstance(best_pair.get('priceChange'), dict) else best_pair.get('priceChange5m', 0)) if best_pair.get('priceChange', {}).get('m5') else None,
            "priceChange1h": float(best_pair.get('priceChange', {}).get('h1', 0) if isinstance(best_pair.get('priceChange'), dict) else best_pair.get('priceChange1h', 0)) if best_pair.get('priceChange', {}).get('h1') else None,
            "volumeChange1h": volume_change_1h,  # Decimal format (0.1 = 10% increase)
            "holders": 0,  # Not available from DexScreener
            "transactions24h": 0,  # Not available from DexScreener
            "social_mentions": 0,  # Not available from DexScreener
            "news_sentiment": 0.5,  # Default neutral
            "sent_score": 0,  # Default
            "sent_mentions": 0,  # Default
            "timestamp": best_pair.get('pairCreatedAt', ''),
            "technical_indicators": {},
            "on_chain_metrics": {}
        }
        
        print(f"✅ Found token: {token_data['symbol']} on {token_data['chainId']}")
        print(f"   Price: ${token_data['priceUsd']:.8f}")
        print(f"   Volume 24h: ${token_data['volume24h']:,.2f}")
        print(f"   Liquidity: ${token_data['liquidity']:,.2f}")
        print(f"   Price Change 24h: {token_data['priceChange24h']*100:.2f}%")
        
        return token_data
        
    except Exception as e:
        print(f"❌ Error fetching token data: {e}")
        return None

async def get_token_score(symbol: str, address: Optional[str] = None):
    """Get AI score for a token (by symbol or by address)."""
    print(f"\n{'='*60}")
    print(f"Getting AI Score for {symbol}" + (f" (address: {address[:8]}...)" if address else ""))
    print(f"{'='*60}\n")
    
    # Fetch token data (by address or symbol)
    token_data = await fetch_token_data(symbol, address=address)
    if not token_data:
        print(f"❌ Could not fetch data for {symbol}")
        return
    
    # Initialize AI engine
    print("\n🤖 Initializing AI engine...")
    try:
        ai_engine = await get_ai_engine()
        print("✅ AI engine initialized")
    except Exception as e:
        print(f"❌ Error initializing AI engine: {e}")
        return
    
    # Analyze token
    print(f"\n🧠 Analyzing {symbol} with AI system...")
    try:
        result = await ai_engine.analyze_token(token_data)
        
        print(f"\n{'='*60}")
        print(f"AI SCORE RESULTS FOR {symbol.upper()}")
        print(f"{'='*60}\n")
        
        print(f"📊 Overall Score: {result.overall_score:.4f} ({result.overall_score*100:.2f}%)")
        print(f"🎯 Confidence: {result.confidence:.4f} ({result.confidence*100:.2f}%)")
        print(f"⏱️  Processing Time: {result.processing_time:.3f}s")
        
        print(f"\n📈 Recommendations:")
        rec = result.recommendations
        print(f"   Action: {rec.get('action', 'unknown')}")
        print(f"   Position Size: ${rec.get('position_size', 0):.2f}")
        print(f"   Take Profit: {rec.get('take_profit', 0)*100:.2f}%")
        print(f"   Stop Loss: {rec.get('stop_loss', 0)*100:.2f}%")
        if rec.get('reasoning'):
            print(f"   Reasoning: {', '.join(rec.get('reasoning', []))}")
        
        print(f"\n⚠️  Risk Assessment:")
        risk = result.risk_assessment
        print(f"   Risk Level: {risk.get('risk_level', 'unknown')}")
        print(f"   Risk Score: {risk.get('risk_score', 0):.4f}")
        if risk.get('risk_factors'):
            print(f"   Risk Factors: {', '.join(risk.get('risk_factors', []))}")
        
        print(f"\n💹 Market Analysis:")
        market = result.market_analysis
        print(f"   Market Health: {market.get('market_health', 'unknown')}")
        print(f"   Liquidity Score: {market.get('liquidity_score', 0):.4f}")
        print(f"   Volume Score: {market.get('volume_score', 0):.4f}")
        print(f"   Market Trend: {market.get('market_trend', 'unknown')}")
        
        print(f"\n📊 Sentiment Analysis:")
        sentiment = result.sentiment_analysis
        print(f"   Category: {sentiment.get('category', 'unknown')}")
        print(f"   Score: {sentiment.get('score', 0):.4f}")
        print(f"   Trend: {sentiment.get('trend', 'unknown')}")
        
        print(f"\n🔮 Price Prediction:")
        prediction = result.prediction_analysis
        print(f"   Movement Probability: {prediction.get('price_movement_probability', 0):.4f}")
        print(f"   Expected Return: {prediction.get('expected_return', 0)*100:.2f}%")
        print(f"   Risk Score: {prediction.get('risk_score', 0):.4f}")
        
        print(f"\n{'='*60}\n")
        
    except Exception as e:
        print(f"❌ Error analyzing token: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_token_score.py <SYMBOL> [ADDRESS]")
        print("Example: python get_token_score.py USELESS")
        print("Example: python get_token_score.py USOR USoRyaQjch6E18nCdDvWoRgTo6osQs9MUd8JXEsspWR")
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    address = sys.argv[2].strip() if len(sys.argv) > 2 else None
    asyncio.run(get_token_score(symbol, address=address))

