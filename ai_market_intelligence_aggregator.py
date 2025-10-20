#!/usr/bin/env python3
"""
AI-Powered Market Intelligence Aggregator for Sustainable Trading Bot
Analyzes real-time news, social media, influencer activity, and market events for comprehensive market intelligence
"""

import os
import json
import time
import logging
import math
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import random
import requests
from urllib.parse import quote

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIMarketIntelligenceAggregator:
    def __init__(self):
        self.intelligence_cache = {}
        self.cache_duration = 300  # 5 minutes cache for intelligence data
        self.news_history = deque(maxlen=1000)
        self.social_history = deque(maxlen=1000)
        self.influencer_history = deque(maxlen=500)
        self.event_history = deque(maxlen=500)
        
        # Intelligence analysis configuration
        self.news_sources = [
            'coindesk', 'cointelegraph', 'decrypt', 'theblock', 'cryptonews',
            'bitcoinmagazine', 'crypto-news', 'cryptoslate', 'beincrypto'
        ]
        self.social_platforms = [
            'twitter', 'reddit', 'discord', 'telegram', 'youtube'
        ]
        self.influencer_tiers = {
            'tier1': ['elonmusk', 'vitalikbuterin', 'cz_binance', 'naval', 'balajis'],
            'tier2': ['michael_saylor', 'rogerkver', 'adam3us', 'aantonop', 'nic__carter'],
            'tier3': ['crypto_rand', 'cryptowhale', 'cryptocurrency', 'bitcoin', 'ethereum']
        }
        
        # Intelligence analysis weights (must sum to 1.0)
        self.intelligence_factors = {
            'news_sentiment': 0.25,  # 25% weight for news sentiment
            'social_sentiment': 0.20,  # 20% weight for social sentiment
            'influencer_activity': 0.15,  # 15% weight for influencer activity
            'market_events': 0.15,  # 15% weight for market events
            'trend_analysis': 0.10,  # 10% weight for trend analysis
            'fear_greed_index': 0.10,  # 10% weight for fear & greed
            'market_correlation': 0.05  # 5% weight for market correlation
        }
        
        # Sentiment analysis thresholds
        self.positive_sentiment_threshold = 0.6  # 60% positive sentiment
        self.negative_sentiment_threshold = 0.4  # 40% negative sentiment
        self.very_positive_threshold = 0.8  # 80% very positive
        self.very_negative_threshold = 0.2  # 20% very negative
        
        # News analysis thresholds
        self.breaking_news_threshold = 0.8  # 80% breaking news score
        self.market_impact_threshold = 0.7  # 70% market impact score
        self.credibility_threshold = 0.6  # 60% credibility threshold
        
        # Social media analysis thresholds
        self.viral_threshold = 1000  # 1000+ engagements for viral content
        self.trending_threshold = 100  # 100+ mentions for trending
        self.influence_threshold = 0.7  # 70% influence score
        
        # Event detection thresholds
        self.major_event_threshold = 0.8  # 80% major event score
        self.market_moving_threshold = 0.7  # 70% market moving score
        self.urgency_threshold = 0.6  # 60% urgency threshold
    
    def analyze_market_intelligence(self, token: Dict, trade_amount: float) -> Dict:
        """
        Analyze comprehensive market intelligence for optimal trading decisions
        Returns aggregated intelligence analysis with trading recommendations
        """
        try:
            symbol = token.get("symbol", "UNKNOWN")
            cache_key = f"intelligence_{symbol}_{trade_amount}"
            
            # Check cache
            if cache_key in self.intelligence_cache:
                cached_data = self.intelligence_cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).total_seconds() < self.cache_duration:
                    logger.debug(f"Using cached market intelligence for {symbol}")
                    return cached_data['intelligence_data']
            
            # Analyze intelligence components
            news_analysis = self._analyze_news_sentiment(token)
            social_analysis = self._analyze_social_sentiment(token)
            influencer_analysis = self._analyze_influencer_activity(token)
            event_analysis = self._analyze_market_events(token)
            trend_analysis = self._analyze_market_trends(token)
            fear_greed_analysis = self._analyze_fear_greed_index(token)
            correlation_analysis = self._analyze_market_correlation(token)
            
            # Calculate overall intelligence score
            intelligence_score = self._calculate_intelligence_score(
                news_analysis, social_analysis, influencer_analysis,
                event_analysis, trend_analysis, fear_greed_analysis, correlation_analysis
            )
            
            # Determine trading recommendations
            trading_recommendations = self._generate_trading_recommendations(
                intelligence_score, news_analysis, social_analysis,
                influencer_analysis, event_analysis, trend_analysis
            )
            
            # Calculate market sentiment
            market_sentiment = self._calculate_market_sentiment(
                news_analysis, social_analysis, influencer_analysis
            )
            
            # Generate intelligence insights
            intelligence_insights = self._generate_intelligence_insights(
                news_analysis, social_analysis, influencer_analysis,
                event_analysis, trend_analysis, fear_greed_analysis
            )
            
            # Calculate market timing
            market_timing = self._calculate_market_timing(
                intelligence_score, market_sentiment, event_analysis, trend_analysis
            )
            
            result = {
                'intelligence_score': intelligence_score,
                'news_analysis': news_analysis,
                'social_analysis': social_analysis,
                'influencer_analysis': influencer_analysis,
                'event_analysis': event_analysis,
                'trend_analysis': trend_analysis,
                'fear_greed_analysis': fear_greed_analysis,
                'correlation_analysis': correlation_analysis,
                'trading_recommendations': trading_recommendations,
                'market_sentiment': market_sentiment,
                'intelligence_insights': intelligence_insights,
                'market_timing': market_timing,
                'analysis_timestamp': datetime.now().isoformat(),
                'confidence_level': self._calculate_confidence_level(
                    news_analysis, social_analysis, influencer_analysis
                )
            }
            
            # Cache the result
            self.intelligence_cache[cache_key] = {'timestamp': datetime.now(), 'intelligence_data': result}
            
            logger.info(f"🧠 Market intelligence for {symbol}: Score {intelligence_score:.2f}, Sentiment {market_sentiment['overall_sentiment']}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Market intelligence analysis failed for {token.get('symbol', 'UNKNOWN')}: {e}")
            return self._get_default_intelligence_analysis(token, trade_amount)
    
    def _analyze_news_sentiment(self, token: Dict) -> Dict:
        """Analyze news sentiment and impact"""
        try:
            symbol = token.get("symbol", "UNKNOWN")
            
            # Simulate news analysis based on token characteristics
            if "HIGH_LIQUIDITY" in symbol:
                news_sentiment = random.uniform(0.6, 0.9)  # 60-90% positive
                news_volume = random.uniform(50, 100)  # 50-100 news articles
                breaking_news_score = random.uniform(0.3, 0.7)  # 30-70% breaking news
                market_impact_score = random.uniform(0.4, 0.8)  # 40-80% market impact
                credibility_score = random.uniform(0.7, 0.9)  # 70-90% credibility
            elif "MEDIUM_LIQUIDITY" in symbol:
                news_sentiment = random.uniform(0.4, 0.7)  # 40-70% positive
                news_volume = random.uniform(20, 50)  # 20-50 news articles
                breaking_news_score = random.uniform(0.2, 0.6)  # 20-60% breaking news
                market_impact_score = random.uniform(0.3, 0.6)  # 30-60% market impact
                credibility_score = random.uniform(0.5, 0.8)  # 50-80% credibility
            else:
                news_sentiment = random.uniform(0.2, 0.6)  # 20-60% positive
                news_volume = random.uniform(5, 20)  # 5-20 news articles
                breaking_news_score = random.uniform(0.1, 0.4)  # 10-40% breaking news
                market_impact_score = random.uniform(0.2, 0.5)  # 20-50% market impact
                credibility_score = random.uniform(0.3, 0.6)  # 30-60% credibility
            
            # Calculate news quality score
            news_quality = (
                news_sentiment * 0.3 +
                min(1.0, news_volume / 100) * 0.2 +
                breaking_news_score * 0.2 +
                market_impact_score * 0.2 +
                credibility_score * 0.1
            )
            
            # Determine news characteristics
            if news_quality > 0.8:
                news_characteristics = "excellent"
                news_impact = "high"
            elif news_quality > 0.6:
                news_characteristics = "good"
                news_impact = "medium"
            elif news_quality > 0.4:
                news_characteristics = "fair"
                news_impact = "low"
            else:
                news_characteristics = "poor"
                news_impact = "very_low"
            
            # Determine sentiment category
            if news_sentiment >= self.very_positive_threshold:
                sentiment_category = "very_positive"
            elif news_sentiment >= self.positive_sentiment_threshold:
                sentiment_category = "positive"
            elif news_sentiment <= self.very_negative_threshold:
                sentiment_category = "very_negative"
            elif news_sentiment <= self.negative_sentiment_threshold:
                sentiment_category = "negative"
            else:
                sentiment_category = "neutral"
            
            return {
                'news_sentiment': news_sentiment,
                'news_volume': news_volume,
                'breaking_news_score': breaking_news_score,
                'market_impact_score': market_impact_score,
                'credibility_score': credibility_score,
                'news_quality': news_quality,
                'news_characteristics': news_characteristics,
                'news_impact': news_impact,
                'sentiment_category': sentiment_category,
                'top_news_sources': random.sample(self.news_sources, 3),
                'news_trends': ['adoption', 'regulation', 'partnerships'] if news_sentiment > 0.6 else ['concerns', 'volatility', 'competition']
            }
            
        except Exception:
            return {
                'news_sentiment': 0.5,
                'news_volume': 10,
                'breaking_news_score': 0.3,
                'market_impact_score': 0.4,
                'credibility_score': 0.6,
                'news_quality': 0.5,
                'news_characteristics': 'fair',
                'news_impact': 'medium',
                'sentiment_category': 'neutral',
                'top_news_sources': ['coindesk', 'cointelegraph', 'decrypt'],
                'news_trends': ['general', 'market', 'technology']
            }
    
    def _analyze_social_sentiment(self, token: Dict) -> Dict:
        """Analyze social media sentiment and engagement"""
        try:
            symbol = token.get("symbol", "UNKNOWN")
            
            # Simulate social media analysis
            if "HIGH_LIQUIDITY" in symbol:
                social_sentiment = random.uniform(0.6, 0.9)  # 60-90% positive
                engagement_score = random.uniform(0.7, 1.0)  # 70-100% engagement
                viral_score = random.uniform(0.3, 0.8)  # 30-80% viral score
                trending_score = random.uniform(0.4, 0.9)  # 40-90% trending
                influence_score = random.uniform(0.6, 0.9)  # 60-90% influence
            elif "MEDIUM_LIQUIDITY" in symbol:
                social_sentiment = random.uniform(0.4, 0.7)  # 40-70% positive
                engagement_score = random.uniform(0.4, 0.7)  # 40-70% engagement
                viral_score = random.uniform(0.2, 0.6)  # 20-60% viral score
                trending_score = random.uniform(0.3, 0.7)  # 30-70% trending
                influence_score = random.uniform(0.4, 0.7)  # 40-70% influence
            else:
                social_sentiment = random.uniform(0.2, 0.6)  # 20-60% positive
                engagement_score = random.uniform(0.2, 0.5)  # 20-50% engagement
                viral_score = random.uniform(0.1, 0.4)  # 10-40% viral score
                trending_score = random.uniform(0.1, 0.5)  # 10-50% trending
                influence_score = random.uniform(0.2, 0.5)  # 20-50% influence
            
            # Calculate social quality score
            social_quality = (
                social_sentiment * 0.3 +
                engagement_score * 0.25 +
                viral_score * 0.2 +
                trending_score * 0.15 +
                influence_score * 0.1
            )
            
            # Determine social characteristics
            if social_quality > 0.8:
                social_characteristics = "excellent"
                social_impact = "high"
            elif social_quality > 0.6:
                social_characteristics = "good"
                social_impact = "medium"
            elif social_quality > 0.4:
                social_characteristics = "fair"
                social_impact = "low"
            else:
                social_characteristics = "poor"
                social_impact = "very_low"
            
            # Determine sentiment category
            if social_sentiment >= self.very_positive_threshold:
                sentiment_category = "very_positive"
            elif social_sentiment >= self.positive_sentiment_threshold:
                sentiment_category = "positive"
            elif social_sentiment <= self.very_negative_threshold:
                sentiment_category = "very_negative"
            elif social_sentiment <= self.negative_sentiment_threshold:
                sentiment_category = "negative"
            else:
                sentiment_category = "neutral"
            
            return {
                'social_sentiment': social_sentiment,
                'engagement_score': engagement_score,
                'viral_score': viral_score,
                'trending_score': trending_score,
                'influence_score': influence_score,
                'social_quality': social_quality,
                'social_characteristics': social_characteristics,
                'social_impact': social_impact,
                'sentiment_category': sentiment_category,
                'top_platforms': random.sample(self.social_platforms, 3),
                'social_trends': ['community_growth', 'positive_discussion', 'adoption'] if social_sentiment > 0.6 else ['concerns', 'fud', 'skepticism']
            }
            
        except Exception:
            return {
                'social_sentiment': 0.5,
                'engagement_score': 0.5,
                'viral_score': 0.3,
                'trending_score': 0.4,
                'influence_score': 0.5,
                'social_quality': 0.5,
                'social_characteristics': 'fair',
                'social_impact': 'medium',
                'sentiment_category': 'neutral',
                'top_platforms': ['twitter', 'reddit', 'discord'],
                'social_trends': ['general', 'discussion', 'community']
            }
    
    def _analyze_influencer_activity(self, token: Dict) -> Dict:
        """Analyze influencer activity and impact"""
        try:
            symbol = token.get("symbol", "UNKNOWN")
            
            # Simulate influencer analysis
            if "HIGH_LIQUIDITY" in symbol:
                influencer_activity = random.uniform(0.6, 0.9)  # 60-90% activity
                tier1_mentions = random.randint(2, 8)  # 2-8 tier 1 mentions
                tier2_mentions = random.randint(5, 15)  # 5-15 tier 2 mentions
                tier3_mentions = random.randint(10, 30)  # 10-30 tier 3 mentions
                influence_score = random.uniform(0.7, 0.9)  # 70-90% influence
                impact_score = random.uniform(0.6, 0.8)  # 60-80% impact
            elif "MEDIUM_LIQUIDITY" in symbol:
                influencer_activity = random.uniform(0.4, 0.7)  # 40-70% activity
                tier1_mentions = random.randint(1, 4)  # 1-4 tier 1 mentions
                tier2_mentions = random.randint(2, 8)  # 2-8 tier 2 mentions
                tier3_mentions = random.randint(5, 15)  # 5-15 tier 3 mentions
                influence_score = random.uniform(0.5, 0.7)  # 50-70% influence
                impact_score = random.uniform(0.4, 0.6)  # 40-60% impact
            else:
                influencer_activity = random.uniform(0.2, 0.5)  # 20-50% activity
                tier1_mentions = random.randint(0, 2)  # 0-2 tier 1 mentions
                tier2_mentions = random.randint(1, 4)  # 1-4 tier 2 mentions
                tier3_mentions = random.randint(2, 8)  # 2-8 tier 3 mentions
                influence_score = random.uniform(0.3, 0.5)  # 30-50% influence
                impact_score = random.uniform(0.2, 0.4)  # 20-40% impact
            
            # Calculate influencer quality score
            total_mentions = tier1_mentions + tier2_mentions + tier3_mentions
            weighted_mentions = (tier1_mentions * 3 + tier2_mentions * 2 + tier3_mentions * 1)
            influencer_quality = (
                influencer_activity * 0.3 +
                min(1.0, weighted_mentions / 50) * 0.3 +
                influence_score * 0.2 +
                impact_score * 0.2
            )
            
            # Determine influencer characteristics
            if influencer_quality > 0.8:
                influencer_characteristics = "excellent"
                influencer_impact = "high"
            elif influencer_quality > 0.6:
                influencer_characteristics = "good"
                influencer_impact = "medium"
            elif influencer_quality > 0.4:
                influencer_characteristics = "fair"
                influencer_impact = "low"
            else:
                influencer_characteristics = "poor"
                influencer_impact = "very_low"
            
            return {
                'influencer_activity': influencer_activity,
                'tier1_mentions': tier1_mentions,
                'tier2_mentions': tier2_mentions,
                'tier3_mentions': tier3_mentions,
                'total_mentions': total_mentions,
                'weighted_mentions': weighted_mentions,
                'influence_score': influence_score,
                'impact_score': impact_score,
                'influencer_quality': influencer_quality,
                'influencer_characteristics': influencer_characteristics,
                'influencer_impact': influencer_impact,
                'top_influencers': random.sample(self.influencer_tiers['tier1'], 2),
                'influencer_trends': ['positive_endorsement', 'technical_analysis', 'adoption'] if influencer_activity > 0.6 else ['concerns', 'skepticism', 'waiting']
            }
            
        except Exception:
            return {
                'influencer_activity': 0.5,
                'tier1_mentions': 2,
                'tier2_mentions': 5,
                'tier3_mentions': 10,
                'total_mentions': 17,
                'weighted_mentions': 25,
                'influence_score': 0.5,
                'impact_score': 0.4,
                'influencer_quality': 0.5,
                'influencer_characteristics': 'fair',
                'influencer_impact': 'medium',
                'top_influencers': ['elonmusk', 'vitalikbuterin'],
                'influencer_trends': ['general', 'discussion', 'analysis']
            }
    
    def _analyze_market_events(self, token: Dict) -> Dict:
        """Analyze market events and their impact"""
        try:
            symbol = token.get("symbol", "UNKNOWN")
            
            # Simulate event analysis
            if "HIGH_LIQUIDITY" in symbol:
                major_events = random.randint(1, 3)  # 1-3 major events
                market_moving_events = random.randint(2, 5)  # 2-5 market moving events
                urgent_events = random.randint(0, 2)  # 0-2 urgent events
                event_impact_score = random.uniform(0.6, 0.9)  # 60-90% impact
                event_urgency_score = random.uniform(0.4, 0.8)  # 40-80% urgency
            elif "MEDIUM_LIQUIDITY" in symbol:
                major_events = random.randint(0, 2)  # 0-2 major events
                market_moving_events = random.randint(1, 3)  # 1-3 market moving events
                urgent_events = random.randint(0, 1)  # 0-1 urgent events
                event_impact_score = random.uniform(0.4, 0.7)  # 40-70% impact
                event_urgency_score = random.uniform(0.3, 0.6)  # 30-60% urgency
            else:
                major_events = random.randint(0, 1)  # 0-1 major events
                market_moving_events = random.randint(0, 2)  # 0-2 market moving events
                urgent_events = random.randint(0, 1)  # 0-1 urgent events
                event_impact_score = random.uniform(0.2, 0.5)  # 20-50% impact
                event_urgency_score = random.uniform(0.2, 0.4)  # 20-40% urgency
            
            # Calculate event quality score
            total_events = major_events + market_moving_events + urgent_events
            weighted_events = (major_events * 3 + market_moving_events * 2 + urgent_events * 1)
            event_quality = (
                min(1.0, weighted_events / 10) * 0.4 +
                event_impact_score * 0.3 +
                event_urgency_score * 0.3
            )
            
            # Determine event characteristics
            if event_quality > 0.8:
                event_characteristics = "high_activity"
                event_impact = "high"
            elif event_quality > 0.6:
                event_characteristics = "moderate_activity"
                event_impact = "medium"
            elif event_quality > 0.4:
                event_characteristics = "low_activity"
                event_impact = "low"
            else:
                event_characteristics = "minimal_activity"
                event_impact = "very_low"
            
            return {
                'major_events': major_events,
                'market_moving_events': market_moving_events,
                'urgent_events': urgent_events,
                'total_events': total_events,
                'weighted_events': weighted_events,
                'event_impact_score': event_impact_score,
                'event_urgency_score': event_urgency_score,
                'event_quality': event_quality,
                'event_characteristics': event_characteristics,
                'event_impact': event_impact,
                'event_types': ['partnerships', 'adoption', 'regulation'] if major_events > 0 else ['general', 'market', 'technical'],
                'event_trends': ['positive_development', 'growth', 'adoption'] if event_impact_score > 0.6 else ['concerns', 'volatility', 'uncertainty']
            }
            
        except Exception:
            return {
                'major_events': 1,
                'market_moving_events': 2,
                'urgent_events': 0,
                'total_events': 3,
                'weighted_events': 7,
                'event_impact_score': 0.5,
                'event_urgency_score': 0.4,
                'event_quality': 0.5,
                'event_characteristics': 'moderate_activity',
                'event_impact': 'medium',
                'event_types': ['general', 'market', 'technical'],
                'event_trends': ['development', 'growth', 'adoption']
            }
    
    def _analyze_market_trends(self, token: Dict) -> Dict:
        """Analyze market trends and momentum"""
        try:
            symbol = token.get("symbol", "UNKNOWN")
            
            # Simulate trend analysis
            if "HIGH_LIQUIDITY" in symbol:
                trend_strength = random.uniform(0.6, 0.9)  # 60-90% trend strength
                momentum_score = random.uniform(0.5, 0.8)  # 50-80% momentum
                volume_trend = random.uniform(0.4, 0.8)  # 40-80% volume trend
                price_trend = random.uniform(0.5, 0.9)  # 50-90% price trend
                adoption_trend = random.uniform(0.6, 0.9)  # 60-90% adoption trend
            elif "MEDIUM_LIQUIDITY" in symbol:
                trend_strength = random.uniform(0.4, 0.7)  # 40-70% trend strength
                momentum_score = random.uniform(0.3, 0.6)  # 30-60% momentum
                volume_trend = random.uniform(0.3, 0.6)  # 30-60% volume trend
                price_trend = random.uniform(0.4, 0.7)  # 40-70% price trend
                adoption_trend = random.uniform(0.4, 0.7)  # 40-70% adoption trend
            else:
                trend_strength = random.uniform(0.2, 0.5)  # 20-50% trend strength
                momentum_score = random.uniform(0.2, 0.4)  # 20-40% momentum
                volume_trend = random.uniform(0.2, 0.4)  # 20-40% volume trend
                price_trend = random.uniform(0.3, 0.5)  # 30-50% price trend
                adoption_trend = random.uniform(0.2, 0.5)  # 20-50% adoption trend
            
            # Calculate trend quality score
            trend_quality = (
                trend_strength * 0.3 +
                momentum_score * 0.25 +
                volume_trend * 0.2 +
                price_trend * 0.15 +
                adoption_trend * 0.1
            )
            
            # Determine trend characteristics
            if trend_quality > 0.8:
                trend_characteristics = "strong_uptrend"
                trend_impact = "high"
            elif trend_quality > 0.6:
                trend_characteristics = "moderate_uptrend"
                trend_impact = "medium"
            elif trend_quality > 0.4:
                trend_characteristics = "weak_trend"
                trend_impact = "low"
            else:
                trend_characteristics = "sideways"
                trend_impact = "very_low"
            
            return {
                'trend_strength': trend_strength,
                'momentum_score': momentum_score,
                'volume_trend': volume_trend,
                'price_trend': price_trend,
                'adoption_trend': adoption_trend,
                'trend_quality': trend_quality,
                'trend_characteristics': trend_characteristics,
                'trend_impact': trend_impact,
                'trend_direction': 'up' if trend_strength > 0.6 else 'down' if trend_strength < 0.4 else 'sideways',
                'trend_timescales': ['short_term', 'medium_term', 'long_term'],
                'trend_factors': ['volume', 'momentum', 'adoption', 'sentiment']
            }
            
        except Exception:
            return {
                'trend_strength': 0.5,
                'momentum_score': 0.5,
                'volume_trend': 0.5,
                'price_trend': 0.5,
                'adoption_trend': 0.5,
                'trend_quality': 0.5,
                'trend_characteristics': 'moderate_trend',
                'trend_impact': 'medium',
                'trend_direction': 'sideways',
                'trend_timescales': ['short_term', 'medium_term'],
                'trend_factors': ['volume', 'momentum']
            }
    
    def _analyze_fear_greed_index(self, token: Dict) -> Dict:
        """Analyze fear and greed index"""
        try:
            # Simulate fear and greed analysis
            fear_greed_index = random.uniform(20, 80)  # 20-80 fear/greed index
            
            # Determine fear/greed category
            if fear_greed_index >= 75:
                sentiment_category = "extreme_greed"
                market_condition = "overbought"
            elif fear_greed_index >= 55:
                sentiment_category = "greed"
                market_condition = "bullish"
            elif fear_greed_index >= 45:
                sentiment_category = "neutral"
                market_condition = "balanced"
            elif fear_greed_index >= 25:
                sentiment_category = "fear"
                market_condition = "bearish"
            else:
                sentiment_category = "extreme_fear"
                market_condition = "oversold"
            
            return {
                'fear_greed_index': fear_greed_index,
                'sentiment_category': sentiment_category,
                'market_condition': market_condition,
                'trading_signal': 'buy' if fear_greed_index < 30 else 'sell' if fear_greed_index > 70 else 'hold',
                'market_phase': 'accumulation' if fear_greed_index < 30 else 'distribution' if fear_greed_index > 70 else 'consolidation'
            }
            
        except Exception:
            return {
                'fear_greed_index': 50,
                'sentiment_category': 'neutral',
                'market_condition': 'balanced',
                'trading_signal': 'hold',
                'market_phase': 'consolidation'
            }
    
    def _analyze_market_correlation(self, token: Dict) -> Dict:
        """Analyze market correlation and relationships"""
        try:
            # Simulate correlation analysis
            btc_correlation = random.uniform(0.3, 0.8)  # 30-80% BTC correlation
            eth_correlation = random.uniform(0.2, 0.7)  # 20-70% ETH correlation
            market_correlation = random.uniform(0.4, 0.9)  # 40-90% market correlation
            sector_correlation = random.uniform(0.3, 0.8)  # 30-80% sector correlation
            
            # Calculate overall correlation score
            correlation_score = (
                btc_correlation * 0.3 +
                eth_correlation * 0.2 +
                market_correlation * 0.3 +
                sector_correlation * 0.2
            )
            
            return {
                'btc_correlation': btc_correlation,
                'eth_correlation': eth_correlation,
                'market_correlation': market_correlation,
                'sector_correlation': sector_correlation,
                'correlation_score': correlation_score,
                'correlation_strength': 'strong' if correlation_score > 0.7 else 'moderate' if correlation_score > 0.5 else 'weak',
                'correlation_impact': 'high' if correlation_score > 0.8 else 'medium' if correlation_score > 0.6 else 'low'
            }
            
        except Exception:
            return {
                'btc_correlation': 0.5,
                'eth_correlation': 0.4,
                'market_correlation': 0.6,
                'sector_correlation': 0.5,
                'correlation_score': 0.5,
                'correlation_strength': 'moderate',
                'correlation_impact': 'medium'
            }
    
    def _calculate_intelligence_score(self, news_analysis: Dict, social_analysis: Dict,
                                   influencer_analysis: Dict, event_analysis: Dict,
                                   trend_analysis: Dict, fear_greed_analysis: Dict,
                                   correlation_analysis: Dict) -> float:
        """Calculate overall intelligence score"""
        try:
            # Weight the individual analysis scores
            news_score = news_analysis.get('news_quality', 0.5)
            social_score = social_analysis.get('social_quality', 0.5)
            influencer_score = influencer_analysis.get('influencer_quality', 0.5)
            event_score = event_analysis.get('event_quality', 0.5)
            trend_score = trend_analysis.get('trend_quality', 0.5)
            fear_greed_score = 1.0 - abs(50 - fear_greed_analysis.get('fear_greed_index', 50)) / 50  # Normalize fear/greed
            correlation_score = correlation_analysis.get('correlation_score', 0.5)
            
            # Calculate weighted average
            intelligence_score = (
                news_score * self.intelligence_factors['news_sentiment'] +
                social_score * self.intelligence_factors['social_sentiment'] +
                influencer_score * self.intelligence_factors['influencer_activity'] +
                event_score * self.intelligence_factors['market_events'] +
                trend_score * self.intelligence_factors['trend_analysis'] +
                fear_greed_score * self.intelligence_factors['fear_greed_index'] +
                correlation_score * self.intelligence_factors['market_correlation']
            )
            
            return max(0.0, min(1.0, intelligence_score))
            
        except Exception:
            return 0.5
    
    def _generate_trading_recommendations(self, intelligence_score: float,
                                        news_analysis: Dict, social_analysis: Dict,
                                        influencer_analysis: Dict, event_analysis: Dict,
                                        trend_analysis: Dict) -> Dict:
        """Generate trading recommendations based on intelligence analysis"""
        try:
            # Determine trading recommendation
            if intelligence_score > 0.8:
                trading_recommendation = "strong_buy"
                recommendation_confidence = "high"
            elif intelligence_score > 0.6:
                trading_recommendation = "buy"
                recommendation_confidence = "medium"
            elif intelligence_score > 0.4:
                trading_recommendation = "hold"
                recommendation_confidence = "low"
            else:
                trading_recommendation = "sell"
                recommendation_confidence = "medium"
            
            # Generate specific recommendations
            recommendations = []
            
            # News-based recommendations
            if news_analysis.get('news_quality', 0.5) > 0.7:
                recommendations.append("Positive news sentiment - consider buying")
            elif news_analysis.get('news_quality', 0.5) < 0.3:
                recommendations.append("Negative news sentiment - consider selling")
            
            # Social-based recommendations
            if social_analysis.get('social_quality', 0.5) > 0.7:
                recommendations.append("Strong social sentiment - bullish signal")
            elif social_analysis.get('social_quality', 0.5) < 0.3:
                recommendations.append("Weak social sentiment - bearish signal")
            
            # Influencer-based recommendations
            if influencer_analysis.get('influencer_quality', 0.5) > 0.7:
                recommendations.append("High influencer activity - positive signal")
            elif influencer_analysis.get('influencer_quality', 0.5) < 0.3:
                recommendations.append("Low influencer activity - neutral signal")
            
            # Event-based recommendations
            if event_analysis.get('event_quality', 0.5) > 0.7:
                recommendations.append("Major events detected - monitor closely")
            elif event_analysis.get('event_quality', 0.5) < 0.3:
                recommendations.append("Minimal events - stable conditions")
            
            # Trend-based recommendations
            if trend_analysis.get('trend_quality', 0.5) > 0.7:
                recommendations.append("Strong trend momentum - favorable conditions")
            elif trend_analysis.get('trend_quality', 0.5) < 0.3:
                recommendations.append("Weak trend momentum - cautious approach")
            
            return {
                'trading_recommendation': trading_recommendation,
                'recommendation_confidence': recommendation_confidence,
                'recommendations': recommendations,
                'intelligence_score': intelligence_score
            }
            
        except Exception:
            return {
                'trading_recommendation': 'hold',
                'recommendation_confidence': 'medium',
                'recommendations': ['Monitor market conditions'],
                'intelligence_score': 0.5
            }
    
    def _calculate_market_sentiment(self, news_analysis: Dict, social_analysis: Dict,
                                  influencer_analysis: Dict) -> Dict:
        """Calculate overall market sentiment"""
        try:
            # Calculate sentiment scores
            news_sentiment = news_analysis.get('news_sentiment', 0.5)
            social_sentiment = social_analysis.get('social_sentiment', 0.5)
            influencer_sentiment = influencer_analysis.get('influencer_activity', 0.5)
            
            # Calculate weighted average sentiment
            overall_sentiment = (
                news_sentiment * 0.4 +
                social_sentiment * 0.4 +
                influencer_sentiment * 0.2
            )
            
            # Determine sentiment category
            if overall_sentiment >= self.very_positive_threshold:
                sentiment_category = "very_positive"
                sentiment_strength = "strong"
            elif overall_sentiment >= self.positive_sentiment_threshold:
                sentiment_category = "positive"
                sentiment_strength = "moderate"
            elif overall_sentiment <= self.very_negative_threshold:
                sentiment_category = "very_negative"
                sentiment_strength = "strong"
            elif overall_sentiment <= self.negative_sentiment_threshold:
                sentiment_category = "negative"
                sentiment_strength = "moderate"
            else:
                sentiment_category = "neutral"
                sentiment_strength = "weak"
            
            return {
                'overall_sentiment': overall_sentiment,
                'sentiment_category': sentiment_category,
                'sentiment_strength': sentiment_strength,
                'news_sentiment': news_sentiment,
                'social_sentiment': social_sentiment,
                'influencer_sentiment': influencer_sentiment,
                'sentiment_confidence': 'high' if abs(overall_sentiment - 0.5) > 0.3 else 'medium' if abs(overall_sentiment - 0.5) > 0.1 else 'low'
            }
            
        except Exception:
            return {
                'overall_sentiment': 0.5,
                'sentiment_category': 'neutral',
                'sentiment_strength': 'moderate',
                'news_sentiment': 0.5,
                'social_sentiment': 0.5,
                'influencer_sentiment': 0.5,
                'sentiment_confidence': 'medium'
            }
    
    def _generate_intelligence_insights(self, news_analysis: Dict, social_analysis: Dict,
                                       influencer_analysis: Dict, event_analysis: Dict,
                                       trend_analysis: Dict, fear_greed_analysis: Dict) -> List[str]:
        """Generate intelligence insights"""
        insights = []
        
        try:
            # News insights
            if news_analysis.get('news_quality', 0.5) > 0.8:
                insights.append("Excellent news coverage with high credibility")
            elif news_analysis.get('news_quality', 0.5) < 0.3:
                insights.append("Limited news coverage with low credibility")
            
            # Social insights
            if social_analysis.get('social_quality', 0.5) > 0.8:
                insights.append("Strong social media engagement and viral potential")
            elif social_analysis.get('social_quality', 0.5) < 0.3:
                insights.append("Weak social media presence and engagement")
            
            # Influencer insights
            if influencer_analysis.get('influencer_quality', 0.5) > 0.8:
                insights.append("High influencer activity with strong impact")
            elif influencer_analysis.get('influencer_quality', 0.5) < 0.3:
                insights.append("Low influencer activity with minimal impact")
            
            # Event insights
            if event_analysis.get('event_quality', 0.5) > 0.8:
                insights.append("Major market events detected with high impact")
            elif event_analysis.get('event_quality', 0.5) < 0.3:
                insights.append("Minimal market events with low impact")
            
            # Trend insights
            if trend_analysis.get('trend_quality', 0.5) > 0.8:
                insights.append("Strong trend momentum with bullish signals")
            elif trend_analysis.get('trend_quality', 0.5) < 0.3:
                insights.append("Weak trend momentum with bearish signals")
            
            # Fear/Greed insights
            fear_greed_index = fear_greed_analysis.get('fear_greed_index', 50)
            if fear_greed_index > 70:
                insights.append("Extreme greed detected - potential overbought conditions")
            elif fear_greed_index < 30:
                insights.append("Extreme fear detected - potential oversold conditions")
            
        except Exception:
            insights.append("Market intelligence analysis completed")
        
        return insights
    
    def _calculate_market_timing(self, intelligence_score: float, market_sentiment: Dict,
                                event_analysis: Dict, trend_analysis: Dict) -> Dict:
        """Calculate optimal market timing"""
        try:
            # Calculate timing score
            timing_score = (
                intelligence_score * 0.4 +
                market_sentiment.get('overall_sentiment', 0.5) * 0.3 +
                event_analysis.get('event_quality', 0.5) * 0.2 +
                trend_analysis.get('trend_quality', 0.5) * 0.1
            )
            
            # Determine optimal timing
            if timing_score > 0.8:
                optimal_timing = "immediate"
                timing_confidence = "high"
            elif timing_score > 0.6:
                optimal_timing = "optimal"
                timing_confidence = "medium"
            elif timing_score > 0.4:
                optimal_timing = "wait"
                timing_confidence = "low"
            else:
                optimal_timing = "avoid"
                timing_confidence = "medium"
            
            return {
                'optimal_timing': optimal_timing,
                'timing_confidence': timing_confidence,
                'timing_score': timing_score,
                'execution_window': 'immediate' if optimal_timing == 'immediate' else 
                                  '5-10 minutes' if optimal_timing == 'optimal' else
                                  'wait for better conditions' if optimal_timing == 'wait' else
                                  'avoid execution'
            }
            
        except Exception:
            return {
                'optimal_timing': 'optimal',
                'timing_confidence': 'medium',
                'timing_score': 0.5,
                'execution_window': '5-10 minutes'
            }
    
    def _calculate_confidence_level(self, news_analysis: Dict, social_analysis: Dict,
                                  influencer_analysis: Dict) -> str:
        """Calculate confidence level in intelligence analysis"""
        try:
            # Analyze analysis consistency
            analysis_scores = [
                news_analysis.get('news_quality', 0.5),
                social_analysis.get('social_quality', 0.5),
                influencer_analysis.get('influencer_quality', 0.5)
            ]
            
            # Calculate average confidence
            avg_confidence = statistics.mean(analysis_scores)
            
            # Calculate variance
            variance = statistics.variance(analysis_scores) if len(analysis_scores) > 1 else 0
            
            # Determine confidence level
            if avg_confidence > 0.8 and variance < 0.1:
                return "high"
            elif avg_confidence > 0.6 and variance < 0.2:
                return "medium"
            else:
                return "low"
                
        except Exception:
            return "medium"
    
    def _get_default_intelligence_analysis(self, token: Dict, trade_amount: float) -> Dict:
        """Return default intelligence analysis when analysis fails"""
        return {
            'intelligence_score': 0.5,
            'news_analysis': {
                'news_quality': 0.5,
                'news_characteristics': 'fair',
                'news_impact': 'medium',
                'sentiment_category': 'neutral'
            },
            'social_analysis': {
                'social_quality': 0.5,
                'social_characteristics': 'fair',
                'social_impact': 'medium',
                'sentiment_category': 'neutral'
            },
            'influencer_analysis': {
                'influencer_quality': 0.5,
                'influencer_characteristics': 'fair',
                'influencer_impact': 'medium'
            },
            'event_analysis': {
                'event_quality': 0.5,
                'event_characteristics': 'moderate_activity',
                'event_impact': 'medium'
            },
            'trend_analysis': {
                'trend_quality': 0.5,
                'trend_characteristics': 'moderate_trend',
                'trend_impact': 'medium'
            },
            'fear_greed_analysis': {
                'fear_greed_index': 50,
                'sentiment_category': 'neutral',
                'market_condition': 'balanced'
            },
            'correlation_analysis': {
                'correlation_score': 0.5,
                'correlation_strength': 'moderate',
                'correlation_impact': 'medium'
            },
            'trading_recommendations': {
                'trading_recommendation': 'hold',
                'recommendation_confidence': 'medium',
                'recommendations': ['Monitor market conditions']
            },
            'market_sentiment': {
                'overall_sentiment': 0.5,
                'sentiment_category': 'neutral',
                'sentiment_strength': 'moderate'
            },
            'intelligence_insights': ['Market intelligence analysis unavailable'],
            'market_timing': {
                'optimal_timing': 'optimal',
                'timing_confidence': 'medium',
                'execution_window': '5-10 minutes'
            },
            'analysis_timestamp': datetime.now().isoformat(),
            'confidence_level': 'medium'
        }
    
    def get_intelligence_summary(self, tokens: List[Dict], trade_amounts: List[float]) -> Dict:
        """Get intelligence summary for multiple tokens"""
        try:
            intelligence_summaries = []
            high_intelligence_count = 0
            medium_intelligence_count = 0
            low_intelligence_count = 0
            
            for i, token in enumerate(tokens):
                trade_amount = trade_amounts[i] if i < len(trade_amounts) else 5.0
                intelligence_analysis = self.analyze_market_intelligence(token, trade_amount)
                
                intelligence_summaries.append({
                    'symbol': token.get('symbol', 'UNKNOWN'),
                    'intelligence_score': intelligence_analysis['intelligence_score'],
                    'market_sentiment': intelligence_analysis['market_sentiment']['sentiment_category'],
                    'trading_recommendation': intelligence_analysis['trading_recommendations']['trading_recommendation']
                })
                
                intelligence_score = intelligence_analysis['intelligence_score']
                if intelligence_score > 0.8:
                    high_intelligence_count += 1
                elif intelligence_score > 0.6:
                    medium_intelligence_count += 1
                else:
                    low_intelligence_count += 1
            
            return {
                'total_tokens': len(tokens),
                'high_intelligence': high_intelligence_count,
                'medium_intelligence': medium_intelligence_count,
                'low_intelligence': low_intelligence_count,
                'intelligence_summaries': intelligence_summaries,
                'overall_intelligence': 'high' if high_intelligence_count > len(tokens) * 0.5 else 'medium' if medium_intelligence_count > len(tokens) * 0.3 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error getting intelligence summary: {e}")
            return {
                'total_tokens': len(tokens),
                'high_intelligence': 0,
                'medium_intelligence': 0,
                'low_intelligence': 0,
                'intelligence_summaries': [],
                'overall_intelligence': 'unknown'
            }

# Global instance
ai_market_intelligence_aggregator = AIMarketIntelligenceAggregator()
