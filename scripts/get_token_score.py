#!/usr/bin/env python3
"""
Script to get AI score for a specific token symbol
Usage: python get_token_score.py USELESS
"""

import asyncio
import sys
import os
import json
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.http_utils import get_json
from src.ai.ai_integration_engine import analyze_token_ai, get_ai_engine

async def fetch_token_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch token data from DexScreener"""
    print(f"🔍 Fetching data for {symbol} from DexScreener...")
    
    # Try to find token by symbol
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
        
        # Convert DexScreener format to our token format
        token_data = {
            "symbol": best_pair.get('baseToken', {}).get('symbol', symbol),
            "address": best_pair.get('baseToken', {}).get('address', ''),
            "chainId": best_pair.get('chainId', 'ethereum').lower(),
            "priceUsd": float(best_pair.get('priceUsd', 0)),
            "volume24h": float(best_pair.get('volume', {}).get('h24', 0) if isinstance(best_pair.get('volume'), dict) else best_pair.get('volume24h', 0)),
            "liquidity": float(best_pair.get('liquidity', {}).get('usd', 0) if isinstance(best_pair.get('liquidity'), dict) else best_pair.get('liquidity', 0)),
            "marketCap": float(best_pair.get('fdv', 0)),
            "priceChange24h": float(best_pair.get('priceChange', {}).get('h24', 0) if isinstance(best_pair.get('priceChange'), dict) else best_pair.get('priceChange24h', 0)) / 100.0 if best_pair.get('priceChange', {}).get('h24') else 0.0,
            "priceChange5m": float(best_pair.get('priceChange', {}).get('m5', 0) if isinstance(best_pair.get('priceChange'), dict) else best_pair.get('priceChange5m', 0)) if best_pair.get('priceChange', {}).get('m5') else None,
            "priceChange1h": float(best_pair.get('priceChange', {}).get('h1', 0) if isinstance(best_pair.get('priceChange'), dict) else best_pair.get('priceChange1h', 0)) if best_pair.get('priceChange', {}).get('h1') else None,
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

async def get_token_score(symbol: str):
    """Get AI score for a token"""
    print(f"\n{'='*60}")
    print(f"Getting AI Score for {symbol}")
    print(f"{'='*60}\n")
    
    # Fetch token data
    token_data = await fetch_token_data(symbol)
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
        print("Usage: python get_token_score.py <SYMBOL>")
        print("Example: python get_token_score.py USELESS")
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    asyncio.run(get_token_score(symbol))

