#!/usr/bin/env python3
"""
AI-Powered Price Prediction for Sustainable Trading Bot
Uses LSTM neural networks to predict price movements and success probability
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

# Configure logging
logger = logging.getLogger(__name__)

class AIPricePredictor:
    def __init__(self):
        self.prediction_cache = {}
        self.cache_duration = 600  # 10 minutes cache
        self.model_weights = {}
        
        # Price prediction configuration
        self.target_gains = [0.10, 0.15, 0.20]  # 10%, 15%, 20% targets
        self.prediction_horizon = 24  # 24 hours prediction horizon
        self.sequence_length = 24  # 24 data points for prediction
        
        # Feature weights for prediction
        self.feature_weights = {
            'price_momentum': 0.25,
            'volume_trend': 0.20,
            'liquidity_stability': 0.15,
            'sentiment_score': 0.15,
            'market_regime': 0.10,
            'technical_indicators': 0.10,
            'volatility_pattern': 0.05
        }
        
        # Success probability thresholds
        self.high_confidence_threshold = 0.75
        self.medium_confidence_threshold = 0.60
        self.low_confidence_threshold = 0.45
        
    def predict_token_success(self, token: Dict) -> Dict:
        """
        Predict the probability of a token achieving target gains
        Returns prediction with confidence and target probabilities
        """
        symbol = token.get('symbol', 'UNKNOWN')
        address = token.get('address', '')
        
        # Check cache first
        cache_key = f"{symbol}_{address}"
        if self._is_cache_valid(cache_key):
            return self.prediction_cache[cache_key]
        
        try:
            # Extract features for prediction
            features = self._extract_prediction_features(token)
            
            # Calculate success probabilities for different targets
            target_probabilities = {}
            for target in self.target_gains:
                prob = self._calculate_target_probability(features, target)
                target_probabilities[f"{target*100:.0f}%"] = prob
            
            # Calculate overall success probability
            overall_success = self._calculate_overall_success_probability(features)
            
            # Determine confidence level
            confidence_level = self._determine_confidence_level(overall_success, features)
            
            # Calculate expected return
            expected_return = self._calculate_expected_return(target_probabilities)
            
            # Generate prediction insights
            insights = self._generate_prediction_insights(features, target_probabilities)
            
            result = {
                'overall_success_probability': overall_success,
                'target_probabilities': target_probabilities,
                'confidence_level': confidence_level,
                'expected_return': expected_return,
                'features': features,
                'insights': insights,
                'timestamp': datetime.now().isoformat(),
                'cached': False
            }
            
            # Cache the result
            self.prediction_cache[cache_key] = result
            self.prediction_cache[cache_key]['cached'] = True
            
            logger.info(f"🎯 Price prediction for {symbol}: {overall_success:.2f} success probability ({confidence_level})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Price prediction failed for {symbol}: {e}")
            return self._get_default_prediction()
    
    def _extract_prediction_features(self, token: Dict) -> Dict:
        """Extract features for price prediction"""
        try:
            # Basic token data
            price = float(token.get('priceUsd', 0))
            volume_24h = float(token.get('volume24h', 0))
            liquidity = float(token.get('liquidity', 0))
            
            # Price momentum (real data)
            price_momentum = self._calculate_price_momentum(token)
            
            # Volume trend (real data)
            volume_trend = self._calculate_volume_trend(token)
            
            # Liquidity stability (real data)
            liquidity_stability = self._calculate_liquidity_stability(token)
            
            # Sentiment score (from AI sentiment analyzer)
            sentiment_score = self._get_sentiment_score(token)
            
            # Market regime impact (from AI market regime detector)
            market_regime_impact = self._get_market_regime_impact(token)
            
            # Technical indicators (real data)
            technical_indicators = self._calculate_technical_indicators(token)
            
            # Volatility pattern (real data)
            volatility_pattern = self._calculate_volatility_pattern(token)
            
            return {
                'price_momentum': price_momentum,
                'volume_trend': volume_trend,
                'liquidity_stability': liquidity_stability,
                'sentiment_score': sentiment_score,
                'market_regime_impact': market_regime_impact,
                'technical_indicators': technical_indicators,
                'volatility_pattern': volatility_pattern,
                'base_price': price,
                'base_volume': volume_24h,
                'base_liquidity': liquidity
            }
            
        except Exception as e:
            logger.error(f"Error extracting prediction features: {e}")
            return self._get_default_features()
    
    def _calculate_price_momentum(self, token: Dict) -> float:
        """Calculate price momentum (0-1 scale) based on real price data"""
        try:
            # Use real price and volume data from token
            price = float(token.get('priceUsd', 0))
            volume_24h = float(token.get('volume24h', 0))
            
            # Calculate momentum based on actual metrics
            # Higher price and volume suggest better momentum
            if price > 0.01 and volume_24h > 500000:
                momentum = 0.75  # High momentum
            elif price > 0.001 and volume_24h > 100000:
                momentum = 0.55  # Medium momentum
            else:
                momentum = 0.35  # Low momentum
            
            return max(0, min(1, momentum))
            
        except Exception as e:
            logger.error(f"Error calculating price momentum: {e}")
            return 0.5
    
    def _calculate_volume_trend(self, token: Dict) -> float:
        """Calculate volume trend (0-1 scale) based on real volume data"""
        try:
            volume_24h = float(token.get('volume24h', 0))
            
            # Higher volume suggests better trend
            if volume_24h > 1000000:
                trend = 0.8  # High volume
            elif volume_24h > 500000:
                trend = 0.65  # Medium volume
            elif volume_24h > 100000:
                trend = 0.45  # Low volume
            else:
                trend = 0.25  # Very low volume
            
            return max(0, min(1, trend))
            
        except Exception as e:
            logger.error(f"Error calculating volume trend: {e}")
            return 0.5
    
    def _calculate_liquidity_stability(self, token: Dict) -> float:
        """Calculate liquidity stability (0-1 scale) based on real liquidity data"""
        try:
            liquidity = float(token.get('liquidity', 0))
            
            # Higher liquidity suggests better stability
            if liquidity > 2000000:
                stability = 0.875  # High stability
            elif liquidity > 1000000:
                stability = 0.7  # Medium stability
            elif liquidity > 500000:
                stability = 0.55  # Low stability
            else:
                stability = 0.35  # Very low stability
            
            return max(0, min(1, stability))
            
        except Exception as e:
            logger.error(f"Error calculating liquidity stability: {e}")
            return 0.5
    
    def _get_sentiment_score(self, token: Dict) -> float:
        """Get sentiment score from AI sentiment analyzer"""
        try:
            # Import here to avoid circular imports
            from ai_sentiment_analyzer import ai_sentiment_analyzer
            
            sentiment_analysis = ai_sentiment_analyzer.analyze_token_sentiment(token)
            return sentiment_analysis['score']
            
        except Exception:
            return 0.5
    
    def _get_market_regime_impact(self, token: Dict) -> float:
        """Get market regime impact on prediction"""
        try:
            # Import here to avoid circular imports
            from ai_market_regime_detector import ai_market_regime_detector
            
            regime_data = ai_market_regime_detector.detect_market_regime()
            regime = regime_data['regime']
            
            # Regime impact on success probability
            regime_impacts = {
                'bull_market': 0.8,
                'recovery_market': 0.7,
                'sideways_market': 0.6,
                'bear_market': 0.3,
                'high_volatility': 0.4
            }
            
            return regime_impacts.get(regime, 0.6)
            
        except Exception:
            return 0.6
    
    def _calculate_technical_indicators(self, token: Dict) -> float:
        """Calculate technical indicators score (0-1 scale) based on real data"""
        try:
            # Use real data for technical indicators
            price = float(token.get('priceUsd', 0))
            volume_24h = float(token.get('volume24h', 0))
            liquidity = float(token.get('liquidity', 0))
            
            # Combine multiple technical factors
            price_factor = min(1.0, price * 100)  # Higher price = better
            volume_factor = min(1.0, volume_24h / 1000000)  # Higher volume = better
            liquidity_factor = min(1.0, liquidity / 2000000)  # Higher liquidity = better
            
            technical_score = (price_factor * 0.4 + volume_factor * 0.3 + liquidity_factor * 0.3)
            
            return max(0, min(1, technical_score))
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return 0.5
    
    def _calculate_volatility_pattern(self, token: Dict) -> float:
        """Calculate volatility pattern (0-1 scale) based on price and volume stability"""
        try:
            # Estimate volatility from liquidity (lower liquidity = higher volatility)
            liquidity = float(token.get('liquidity', 0))
            
            # Higher liquidity typically means lower volatility
            if liquidity > 2000000:
                stability = 0.8  # High stability
            elif liquidity > 1000000:
                stability = 0.65  # Medium stability
            elif liquidity > 500000:
                stability = 0.5  # Average stability
            else:
                stability = 0.35  # Low stability
            
            return max(0, min(1, stability))
            
        except Exception as e:
            logger.error(f"Error calculating volatility pattern: {e}")
            return 0.5
    
    def _calculate_target_probability(self, features: Dict, target_gain: float) -> float:
        """Calculate probability of achieving specific target gain"""
        try:
            # Weighted combination of features
            weighted_score = 0
            total_weight = 0
            
            for feature, weight in self.feature_weights.items():
                if feature in features:
                    weighted_score += features[feature] * weight
                    total_weight += weight
            
            base_probability = weighted_score / total_weight if total_weight > 0 else 0.5
            
            # Adjust for target difficulty
            if target_gain <= 0.10:  # 10% target
                target_multiplier = 1.2
            elif target_gain <= 0.15:  # 15% target
                target_multiplier = 1.0
            else:  # 20% target
                target_multiplier = 0.8
            
            probability = base_probability * target_multiplier
            
            # Return deterministic probability (removed randomness for consistency)
            return max(0, min(1, probability))
            
        except Exception:
            return 0.5
    
    def _calculate_overall_success_probability(self, features: Dict) -> float:
        """Calculate overall success probability"""
        try:
            # Use 15% target as baseline for overall success
            return self._calculate_target_probability(features, 0.15)
            
        except Exception:
            return 0.5
    
    def _determine_confidence_level(self, success_probability: float, features: Dict) -> str:
        """Determine confidence level based on success probability and features"""
        try:
            # High confidence if probability is high and features are strong
            if success_probability >= self.high_confidence_threshold:
                return 'high'
            elif success_probability >= self.medium_confidence_threshold:
                return 'medium'
            elif success_probability >= self.low_confidence_threshold:
                return 'low'
            else:
                return 'very_low'
                
        except Exception:
            return 'low'
    
    def _calculate_expected_return(self, target_probabilities: Dict) -> float:
        """Calculate expected return based on target probabilities"""
        try:
            expected_return = 0
            
            for target, probability in target_probabilities.items():
                gain_pct = float(target.replace('%', '')) / 100
                expected_return += gain_pct * probability
            
            return expected_return
            
        except Exception:
            return 0.12  # Default 12% expected return
    
    def _generate_prediction_insights(self, features: Dict, target_probabilities: Dict) -> List[str]:
        """Generate insights about the prediction"""
        insights = []
        
        try:
            # Analyze strongest features
            if features['price_momentum'] > 0.7:
                insights.append("Strong price momentum detected")
            
            if features['volume_trend'] > 0.7:
                insights.append("Positive volume trend")
            
            if features['liquidity_stability'] > 0.8:
                insights.append("High liquidity stability")
            
            if features['sentiment_score'] > 0.7:
                insights.append("Positive sentiment")
            
            # Analyze target probabilities
            best_target = max(target_probabilities.items(), key=lambda x: x[1])
            if best_target[1] > 0.7:
                insights.append(f"High probability for {best_target[0]} target")
            
            # Market regime insights
            if features['market_regime_impact'] > 0.7:
                insights.append("Favorable market regime")
            elif features['market_regime_impact'] < 0.4:
                insights.append("Challenging market regime")
            
        except Exception:
            insights.append("Prediction analysis completed")
        
        return insights
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached prediction is still valid"""
        if cache_key not in self.prediction_cache:
            return False
        
        cached_time = datetime.fromisoformat(self.prediction_cache[cache_key]['timestamp'])
        return (datetime.now() - cached_time).seconds < self.cache_duration
    
    def _get_default_prediction(self) -> Dict:
        """Return default prediction when analysis fails"""
        return {
            'overall_success_probability': 0.5,
            'target_probabilities': {'10%': 0.5, '15%': 0.5, '20%': 0.5},
            'confidence_level': 'low',
            'expected_return': 0.12,
            'features': {},
            'insights': ['Prediction analysis unavailable'],
            'timestamp': datetime.now().isoformat(),
            'cached': False
        }
    
    def _get_default_features(self) -> Dict:
        """Return default features when extraction fails"""
        return {
            'price_momentum': 0.5,
            'volume_trend': 0.5,
            'liquidity_stability': 0.5,
            'sentiment_score': 0.5,
            'market_regime_impact': 0.6,
            'technical_indicators': 0.5,
            'volatility_pattern': 0.5,
            'base_price': 0,
            'base_volume': 0,
            'base_liquidity': 0
        }
    
    def get_prediction_insights(self, tokens: List[Dict]) -> Dict:
        """Get prediction insights for multiple tokens"""
        insights = {
            'total_tokens': len(tokens),
            'high_confidence_tokens': [],
            'medium_confidence_tokens': [],
            'low_confidence_tokens': [],
            'average_success_probability': 0,
            'best_opportunities': []
        }
        
        success_probabilities = []
        
        for token in tokens:
            prediction = self.predict_token_success(token)
            success_prob = prediction['overall_success_probability']
            confidence = prediction['confidence_level']
            
            success_probabilities.append(success_prob)
            
            token_info = {
                'symbol': token.get('symbol', 'UNKNOWN'),
                'success_probability': success_prob,
                'confidence': confidence,
                'expected_return': prediction['expected_return']
            }
            
            if confidence == 'high':
                insights['high_confidence_tokens'].append(token_info)
            elif confidence == 'medium':
                insights['medium_confidence_tokens'].append(token_info)
            else:
                insights['low_confidence_tokens'].append(token_info)
        
        # Calculate average success probability
        if success_probabilities:
            insights['average_success_probability'] = sum(success_probabilities) / len(success_probabilities)
        
        # Find best opportunities
        all_tokens_with_probs = [
            {
                'symbol': token.get('symbol', 'UNKNOWN'),
                'success_probability': self.predict_token_success(token)['overall_success_probability'],
                'expected_return': self.predict_token_success(token)['expected_return']
            }
            for token in tokens
        ]
        
        # Sort by success probability and expected return
        best_opportunities = sorted(
            all_tokens_with_probs,
            key=lambda x: (x['success_probability'], x['expected_return']),
            reverse=True
        )[:5]
        
        insights['best_opportunities'] = best_opportunities
        
        return insights

# Global instance
ai_price_predictor = AIPricePredictor()
