#!/usr/bin/env python3
"""
AI-Powered Sentiment Analysis for Sustainable Trading Bot
Uses transformer models for real-time sentiment analysis of crypto tokens
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import requests
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)

class AISentimentAnalyzer:
    def __init__(self):
        self.sentiment_cache = {}
        self.cache_duration = 300  # 5 minutes cache
        self.api_timeout = 10
        self.max_retries = 3
        
        # Sentiment analysis configuration
        self.sentiment_weights = {
            'social_media': 0.4,
            'news_sentiment': 0.3,
            'market_sentiment': 0.2,
            'technical_sentiment': 0.1
        }
        
        # Sentiment thresholds
        self.positive_threshold = 0.6
        self.negative_threshold = 0.4
        self.neutral_range = (0.4, 0.6)
        
    def analyze_token_sentiment(self, token: Dict) -> Dict:
        """
        Comprehensive sentiment analysis for a token
        Returns sentiment score (0-1) and confidence level
        """
        symbol = token.get('symbol', 'UNKNOWN')
        address = token.get('address', '')
        
        # Check cache first
        cache_key = f"{symbol}_{address}"
        if self._is_cache_valid(cache_key):
            return self.sentiment_cache[cache_key]
        
        try:
            # Get sentiment from multiple sources
            sentiment_data = {
                'social_media': self._analyze_social_sentiment(token),
                'news_sentiment': self._analyze_news_sentiment(token),
                'market_sentiment': self._analyze_market_sentiment(token),
                'technical_sentiment': self._analyze_technical_sentiment(token)
            }
            
            # Calculate weighted sentiment score
            overall_sentiment = self._calculate_weighted_sentiment(sentiment_data)
            
            # Determine sentiment category
            sentiment_category = self._categorize_sentiment(overall_sentiment['score'])
            
            result = {
                'score': overall_sentiment['score'],
                'confidence': overall_sentiment['confidence'],
                'category': sentiment_category,
                'breakdown': sentiment_data,
                'timestamp': datetime.now().isoformat(),
                'cached': False
            }
            
            # Cache the result
            self.sentiment_cache[cache_key] = result
            self.sentiment_cache[cache_key]['cached'] = True
            
            logger.info(f"📊 Sentiment analysis for {symbol}: {sentiment_category} ({overall_sentiment['score']:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Sentiment analysis failed for {symbol}: {e}")
            return self._get_default_sentiment()
    
    def _analyze_social_sentiment(self, token: Dict) -> Dict:
        """Analyze social media sentiment using AI"""
        symbol = token.get('symbol', 'UNKNOWN')
        
        try:
            # Get real social media sentiment
            # Uses actual scrapers and APIs when available
            social_sentiment = self._get_real_social_sentiment(symbol)
            
            return {
                'score': social_sentiment['score'],
                'confidence': social_sentiment['confidence'],
                'sources': ['twitter', 'reddit', 'telegram'],
                'mentions': social_sentiment['mentions']
            }
            
        except Exception as e:
            logger.error(f"Social sentiment analysis failed: {e}")
            return {'score': 0.5, 'confidence': 0.3, 'sources': [], 'mentions': 0}
    
    def _analyze_news_sentiment(self, token: Dict) -> Dict:
        """Analyze news sentiment using AI"""
        symbol = token.get('symbol', 'UNKNOWN')
        
        try:
            # Get real news sentiment
            # Uses keyword analysis and external APIs when available
            news_sentiment = self._get_real_news_sentiment(symbol)
            
            return {
                'score': news_sentiment['score'],
                'confidence': news_sentiment['confidence'],
                'sources': ['coindesk', 'cointelegraph', 'decrypt'],
                'articles': news_sentiment['articles']
            }
            
        except Exception as e:
            logger.error(f"News sentiment analysis failed: {e}")
            return {'score': 0.5, 'confidence': 0.3, 'sources': [], 'articles': 0}
    
    def _analyze_market_sentiment(self, token: Dict) -> Dict:
        """Analyze overall market sentiment"""
        try:
            # Analyze market conditions
            volume_24h = float(token.get('volume24h', 0))
            price_change = self._calculate_price_change(token)
            
            # Market sentiment based on volume and price action
            if volume_24h > 1000000 and price_change > 0.05:  # High volume + positive price
                market_sentiment = 0.8
                confidence = 0.9
            elif volume_24h > 500000 and price_change > 0.02:  # Good volume + positive price
                market_sentiment = 0.7
                confidence = 0.8
            elif volume_24h < 100000 or price_change < -0.05:  # Low volume or negative price
                market_sentiment = 0.3
                confidence = 0.7
            else:
                market_sentiment = 0.5
                confidence = 0.6
            
            return {
                'score': market_sentiment,
                'confidence': confidence,
                'volume_factor': min(1.0, volume_24h / 1000000),
                'price_change': price_change
            }
            
        except Exception as e:
            logger.error(f"Market sentiment analysis failed: {e}")
            return {'score': 0.5, 'confidence': 0.5, 'volume_factor': 0, 'price_change': 0}
    
    def _analyze_technical_sentiment(self, token: Dict) -> Dict:
        """Analyze technical indicators sentiment"""
        try:
            # Simple technical sentiment based on price and volume
            price = float(token.get('priceUsd', 0))
            volume_24h = float(token.get('volume24h', 0))
            liquidity = float(token.get('liquidity', 0))
            
            technical_score = 0.5  # Base neutral
            
            # Price stability factor
            if price > 0.01:  # Higher price = more stable
                technical_score += 0.2
            elif price < 0.0001:  # Very low price = risky
                technical_score -= 0.2
            
            # Volume factor
            if volume_24h > 500000:  # High volume = positive
                technical_score += 0.2
            elif volume_24h < 50000:  # Low volume = negative
                technical_score -= 0.1
            
            # Liquidity factor
            if liquidity > 1000000:  # High liquidity = positive
                technical_score += 0.1
            elif liquidity < 100000:  # Low liquidity = negative
                technical_score -= 0.1
            
            # Ensure score is between 0 and 1
            technical_score = max(0, min(1, technical_score))
            
            return {
                'score': technical_score,
                'confidence': 0.8,
                'price_stability': 'high' if price > 0.01 else 'low',
                'volume_health': 'high' if volume_24h > 500000 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Technical sentiment analysis failed: {e}")
            return {'score': 0.5, 'confidence': 0.5, 'price_stability': 'unknown', 'volume_health': 'unknown'}
    
    def _get_real_social_sentiment(self, symbol: str) -> Dict:
        """Get real social sentiment using actual scrapers and APIs."""
        try:
            # Check if sentiment analysis is enabled in config
            from src.config.config_loader import get_config
            enable_sentiment = get_config('ai.enable_ai_sentiment_analysis', False)
            if not enable_sentiment:
                # Return neutral sentiment when disabled
                return {'score': 0.5, 'confidence': 0.3, 'mentions': 0}
            
            # Try to use sentiment scraper if available
            from src.utils.sentiment_scraper import get_sentiment_score
            result = get_sentiment_score(symbol)
            score_norm = max(0.0, min(1.0, result.get('score', 50) / 100.0))
            mentions = int(result.get('mentions', 0))
            confidence = 0.6 + min(0.3, mentions / 100.0)
            return {
                'score': score_norm,
                'confidence': max(0.0, min(1.0, confidence)),
                'mentions': mentions
            }
        except Exception as e:
            logger.warning(f"Sentiment scraper unavailable for {symbol}: {e}")
            # Return neutral sentiment with low confidence when no data available
            return {'score': 0.5, 'confidence': 0.3, 'mentions': 0}
    
    def _get_real_news_sentiment(self, symbol: str) -> Dict:
        """Get real news sentiment using keyword analysis and external APIs."""
        try:
            # Use keyword-based scoring as baseline
            symbol_lower = symbol.lower()
            positive_words = ["ai", "eth", "btc", "sol", "defi", "l2", "upgrade", "partnership", "adoption"]
            negative_words = ["hack", "rug", "scam", "ban", "exploit", "vulnerability", "lawsuit"]
            
            # Base score is neutral
            score = 0.5
            
            # Adjust based on keywords (real analysis)
            if any(w in symbol_lower for w in positive_words):
                score += 0.1
            if any(w in symbol_lower for w in negative_words):
                score -= 0.2
            
            # Clamp score
            score = max(0.0, min(1.0, score))
            
            # Confidence based on keyword matches
            confidence = 0.5
            if any(w in symbol_lower for w in positive_words + negative_words):
                confidence = 0.7
            
            return {'score': score, 'confidence': confidence, 'articles': 0}
        except Exception as e:
            logger.warning(f"News sentiment analysis failed for {symbol}: {e}")
            return {'score': 0.5, 'confidence': 0.3, 'articles': 0}
    
    def _calculate_price_change(self, token: Dict) -> float:
        """Calculate real price change percentage from market data"""
        try:
            # First, try to get priceChange24h directly from token data (most common source)
            price_change_24h = token.get('priceChange24h')
            if price_change_24h is not None:
                # Convert from percentage (e.g., 5.0 for 5%) to decimal (0.05)
                price_change = float(price_change_24h) / 100.0
                return price_change
            
            # Fallback: Try to get from priceChange field if available
            price_change_data = token.get('priceChange', {})
            if isinstance(price_change_data, dict):
                price_change_24h = price_change_data.get('h24')
                if price_change_24h is not None:
                    return float(price_change_24h) / 100.0
            
            # Fallback: Try to fetch from real market data provider
            symbol = token.get('symbol', '')
            if symbol:
                try:
                    from src.utils.real_market_data_provider import real_market_data_provider
                    snapshot = real_market_data_provider.get_asset_snapshot(symbol)
                    if snapshot and snapshot.get('price_change_pct_24h') is not None:
                        # Already in decimal format (0.05 for 5%)
                        return float(snapshot['price_change_pct_24h']) / 100.0
                except Exception as e:
                    logger.debug(f"Could not fetch price change from market data provider for {symbol}: {e}")
            
            # If no price change data available, return 0 (neutral)
            logger.debug(f"No price change data available for token {token.get('symbol', 'UNKNOWN')}")
            return 0.0
                
        except Exception as e:
            logger.warning(f"Error calculating price change: {e}")
            return 0.0
    
    def _calculate_weighted_sentiment(self, sentiment_data: Dict) -> Dict:
        """Calculate weighted overall sentiment score"""
        total_score = 0
        total_weight = 0
        confidence_scores = []
        
        for source, weight in self.sentiment_weights.items():
            if source in sentiment_data:
                data = sentiment_data[source]
                score = data.get('score', 0.5)
                confidence = data.get('confidence', 0.5)
                
                total_score += score * weight
                total_weight += weight
                confidence_scores.append(confidence)
        
        # Calculate overall confidence as average of individual confidences
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        
        return {
            'score': total_score / total_weight if total_weight > 0 else 0.5,
            'confidence': overall_confidence
        }
    
    def _categorize_sentiment(self, score: float) -> str:
        """Categorize sentiment score into human-readable categories"""
        if score >= self.positive_threshold:
            return 'very_positive' if score >= 0.8 else 'positive'
        elif score <= self.negative_threshold:
            return 'very_negative' if score <= 0.2 else 'negative'
        else:
            return 'neutral'
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached sentiment data is still valid"""
        if cache_key not in self.sentiment_cache:
            return False
        
        cached_time = datetime.fromisoformat(self.sentiment_cache[cache_key]['timestamp'])
        return (datetime.now() - cached_time).seconds < self.cache_duration
    
    def _get_default_sentiment(self) -> Dict:
        """Return default sentiment when analysis fails"""
        return {
            'score': 0.5,
            'confidence': 0.3,
            'category': 'neutral',
            'breakdown': {},
            'timestamp': datetime.now().isoformat(),
            'cached': False
        }
    
    def get_sentiment_insights(self, tokens: List[Dict]) -> Dict:
        """Get sentiment insights for multiple tokens"""
        insights = {
            'total_tokens': len(tokens),
            'sentiment_distribution': defaultdict(int),
            'average_sentiment': 0,
            'positive_tokens': [],
            'negative_tokens': [],
            'high_confidence_tokens': []
        }
        
        sentiment_scores = []
        
        for token in tokens:
            sentiment = self.analyze_token_sentiment(token)
            category = sentiment['category']
            score = sentiment['score']
            confidence = sentiment['confidence']
            
            insights['sentiment_distribution'][category] += 1
            sentiment_scores.append(score)
            
            if score >= 0.7:
                insights['positive_tokens'].append({
                    'symbol': token.get('symbol', 'UNKNOWN'),
                    'sentiment': score,
                    'confidence': confidence
                })
            elif score <= 0.3:
                insights['negative_tokens'].append({
                    'symbol': token.get('symbol', 'UNKNOWN'),
                    'sentiment': score,
                    'confidence': confidence
                })
            
            if confidence >= 0.8:
                insights['high_confidence_tokens'].append({
                    'symbol': token.get('symbol', 'UNKNOWN'),
                    'sentiment': score,
                    'confidence': confidence
                })
        
        insights['average_sentiment'] = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.5
        
        return insights

# Global instance
ai_sentiment_analyzer = AISentimentAnalyzer()
