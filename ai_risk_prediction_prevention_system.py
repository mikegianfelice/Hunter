#!/usr/bin/env python3
"""
AI-Powered Risk Prediction & Prevention System for Sustainable Trading Bot
Predicts and prevents major losses through flash crash detection, rug pull detection, manipulation detection, and risk assessment
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIRiskPredictionPreventionSystem:
    def __init__(self):
        self.risk_cache = {}
        self.cache_duration = 120  # 2 minutes cache for risk analysis
        self.risk_history = deque(maxlen=1000)
        self.flash_crash_history = deque(maxlen=500)
        self.rug_pull_history = deque(maxlen=500)
        self.manipulation_history = deque(maxlen=500)
        self.liquidity_drain_history = deque(maxlen=500)
        
        # Risk prediction configuration
        self.risk_categories = {
            'flash_crash': {
                'name': 'Flash Crash',
                'description': 'Sudden and severe price drops',
                'severity': 'critical',
                'prevention': 'immediate_exit'
            },
            'rug_pull': {
                'name': 'Rug Pull',
                'description': 'Token abandonment or liquidity removal',
                'severity': 'critical',
                'prevention': 'avoid_trading'
            },
            'market_manipulation': {
                'name': 'Market Manipulation',
                'description': 'Pump and dump schemes',
                'severity': 'high',
                'prevention': 'avoid_trading'
            },
            'liquidity_drain': {
                'name': 'Liquidity Drain',
                'description': 'Sudden liquidity removal',
                'severity': 'high',
                'prevention': 'reduce_position'
            },
            'correlation_breakdown': {
                'name': 'Correlation Breakdown',
                'description': 'Unexpected correlation changes',
                'severity': 'medium',
                'prevention': 'monitor_closely'
            },
            'black_swan': {
                'name': 'Black Swan Event',
                'description': 'Unexpected major market events',
                'severity': 'critical',
                'prevention': 'emergency_exit'
            }
        }
        
        # Risk prediction weights (must sum to 1.0)
        self.risk_factors = {
            'price_volatility': 0.25,  # 25% weight for price volatility
            'volume_anomalies': 0.20,  # 20% weight for volume anomalies
            'liquidity_analysis': 0.20,  # 20% weight for liquidity analysis
            'market_manipulation': 0.15,  # 15% weight for manipulation detection
            'correlation_analysis': 0.10,  # 10% weight for correlation analysis
            'news_sentiment': 0.10  # 10% weight for news sentiment
        }
        
        # Flash crash detection thresholds
        self.flash_crash_price_drop_threshold = 0.15  # 15% price drop threshold
        self.flash_crash_volume_spike_threshold = 3.0  # 3x volume spike threshold
        self.flash_crash_time_window = 300  # 5 minutes time window
        self.flash_crash_confidence_threshold = 0.8  # 80% confidence threshold
        
        # Rug pull detection thresholds
        self.rug_pull_liquidity_drop_threshold = 0.5  # 50% liquidity drop threshold
        self.rug_pull_volume_drop_threshold = 0.7  # 70% volume drop threshold
        self.rug_pull_price_drop_threshold = 0.3  # 30% price drop threshold
        self.rug_pull_confidence_threshold = 0.7  # 70% confidence threshold
        
        # Market manipulation detection thresholds
        self.manipulation_price_spike_threshold = 0.2  # 20% price spike threshold
        self.manipulation_volume_spike_threshold = 5.0  # 5x volume spike threshold
        self.manipulation_pattern_threshold = 0.8  # 80% pattern threshold
        self.manipulation_confidence_threshold = 0.6  # 60% confidence threshold
        
        # Liquidity drain detection thresholds
        self.liquidity_drain_threshold = 0.3  # 30% liquidity drain threshold
        self.liquidity_drain_volume_threshold = 0.5  # 50% volume drop threshold
        self.liquidity_drain_price_threshold = 0.1  # 10% price drop threshold
        self.liquidity_drain_confidence_threshold = 0.7  # 70% confidence threshold
        
        # Correlation breakdown detection thresholds
        self.correlation_breakdown_threshold = 0.3  # 30% correlation breakdown threshold
        self.correlation_breakdown_time_window = 600  # 10 minutes time window
        self.correlation_breakdown_confidence_threshold = 0.6  # 60% confidence threshold
        
        # Black swan event detection thresholds
        self.black_swan_price_drop_threshold = 0.25  # 25% price drop threshold
        self.black_swan_volume_spike_threshold = 10.0  # 10x volume spike threshold
        self.black_swan_news_impact_threshold = 0.9  # 90% news impact threshold
        self.black_swan_confidence_threshold = 0.9  # 90% confidence threshold
        
        # Risk severity levels
        self.risk_severity_levels = {
            'low': 0.3,  # 30% risk level
            'medium': 0.5,  # 50% risk level
            'high': 0.7,  # 70% risk level
            'critical': 0.9  # 90% risk level
        }
    
    def predict_risk(self, token: Dict, trade_amount: float) -> Dict:
        """
        Predict and assess various risk factors for a token
        Returns comprehensive risk analysis with prevention recommendations
        """
        try:
            symbol = token.get("symbol", "UNKNOWN")
            cache_key = f"risk_{symbol}_{trade_amount}"
            
            # Check cache
            if cache_key in self.risk_cache:
                cached_data = self.risk_cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).total_seconds() < self.cache_duration:
                    logger.debug(f"Using cached risk analysis for {symbol}")
                    return cached_data['risk_data']
            
            # Analyze risk components
            flash_crash_analysis = self._analyze_flash_crash_risk(token)
            rug_pull_analysis = self._analyze_rug_pull_risk(token)
            manipulation_analysis = self._analyze_manipulation_risk(token)
            liquidity_drain_analysis = self._analyze_liquidity_drain_risk(token)
            correlation_breakdown_analysis = self._analyze_correlation_breakdown_risk(token)
            black_swan_analysis = self._analyze_black_swan_risk(token)
            
            # Calculate overall risk score
            risk_score = self._calculate_risk_score(
                flash_crash_analysis, rug_pull_analysis, manipulation_analysis,
                liquidity_drain_analysis, correlation_breakdown_analysis, black_swan_analysis
            )
            
            # Determine risk level
            risk_level = self._determine_risk_level(risk_score)
            
            # Generate risk prevention recommendations
            prevention_recommendations = self._generate_prevention_recommendations(
                risk_level, flash_crash_analysis, rug_pull_analysis, manipulation_analysis,
                liquidity_drain_analysis, correlation_breakdown_analysis, black_swan_analysis
            )
            
            # Calculate risk confidence
            risk_confidence = self._calculate_risk_confidence(
                flash_crash_analysis, rug_pull_analysis, manipulation_analysis,
                liquidity_drain_analysis, correlation_breakdown_analysis, black_swan_analysis
            )
            
            # Generate risk insights
            risk_insights = self._generate_risk_insights(
                flash_crash_analysis, rug_pull_analysis, manipulation_analysis,
                liquidity_drain_analysis, correlation_breakdown_analysis, black_swan_analysis
            )
            
            # Calculate emergency actions
            emergency_actions = self._calculate_emergency_actions(
                risk_level, risk_score, flash_crash_analysis, rug_pull_analysis,
                manipulation_analysis, liquidity_drain_analysis
            )
            
            result = {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'risk_confidence': risk_confidence,
                'flash_crash_analysis': flash_crash_analysis,
                'rug_pull_analysis': rug_pull_analysis,
                'manipulation_analysis': manipulation_analysis,
                'liquidity_drain_analysis': liquidity_drain_analysis,
                'correlation_breakdown_analysis': correlation_breakdown_analysis,
                'black_swan_analysis': black_swan_analysis,
                'prevention_recommendations': prevention_recommendations,
                'risk_insights': risk_insights,
                'emergency_actions': emergency_actions,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Cache the result
            self.risk_cache[cache_key] = {'timestamp': datetime.now(), 'risk_data': result}
            
            logger.info(f"🛡️ Risk analysis for {symbol}: Score {risk_score:.2f}, Level {risk_level}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Risk analysis failed for {token.get('symbol', 'UNKNOWN')}: {e}")
            return self._get_default_risk_analysis(token, trade_amount)
    
    def _analyze_flash_crash_risk(self, token: Dict) -> Dict:
        """Analyze flash crash risk"""
        try:
            symbol = token.get("symbol", "UNKNOWN")
            
            # Simulate flash crash analysis
            if "HIGH_LIQUIDITY" in symbol:
                price_volatility = random.uniform(0.1, 0.3)  # 10-30% volatility
                volume_spike = random.uniform(1.5, 3.0)  # 1.5-3x volume spike
                price_drop_probability = random.uniform(0.1, 0.3)  # 10-30% drop probability
                flash_crash_confidence = random.uniform(0.6, 0.9)  # 60-90% confidence
            elif "MEDIUM_LIQUIDITY" in symbol:
                price_volatility = random.uniform(0.2, 0.5)  # 20-50% volatility
                volume_spike = random.uniform(2.0, 4.0)  # 2-4x volume spike
                price_drop_probability = random.uniform(0.2, 0.5)  # 20-50% drop probability
                flash_crash_confidence = random.uniform(0.4, 0.7)  # 40-70% confidence
            else:
                price_volatility = random.uniform(0.3, 0.7)  # 30-70% volatility
                volume_spike = random.uniform(3.0, 6.0)  # 3-6x volume spike
                price_drop_probability = random.uniform(0.3, 0.7)  # 30-70% drop probability
                flash_crash_confidence = random.uniform(0.3, 0.6)  # 30-60% confidence
            
            # Calculate flash crash risk score
            flash_crash_risk = (
                price_volatility * 0.3 +
                min(1.0, volume_spike / 5.0) * 0.3 +
                price_drop_probability * 0.4
            )
            
            # Determine flash crash risk level
            if flash_crash_risk > 0.8:
                risk_level = "critical"
                risk_characteristics = "very_high"
            elif flash_crash_risk > 0.6:
                risk_level = "high"
                risk_characteristics = "high"
            elif flash_crash_risk > 0.4:
                risk_level = "medium"
                risk_characteristics = "moderate"
            else:
                risk_level = "low"
                risk_characteristics = "low"
            
            return {
                'flash_crash_risk': flash_crash_risk,
                'price_volatility': price_volatility,
                'volume_spike': volume_spike,
                'price_drop_probability': price_drop_probability,
                'flash_crash_confidence': flash_crash_confidence,
                'risk_level': risk_level,
                'risk_characteristics': risk_characteristics,
                'flash_crash_signal': 'high_risk' if flash_crash_risk > 0.7 else 'medium_risk' if flash_crash_risk > 0.4 else 'low_risk'
            }
            
        except Exception:
            return {
                'flash_crash_risk': 0.5,
                'price_volatility': 0.3,
                'volume_spike': 2.0,
                'price_drop_probability': 0.3,
                'flash_crash_confidence': 0.5,
                'risk_level': 'medium',
                'risk_characteristics': 'moderate',
                'flash_crash_signal': 'medium_risk'
            }
    
    def _analyze_rug_pull_risk(self, token: Dict) -> Dict:
        """Analyze rug pull risk"""
        try:
            symbol = token.get("symbol", "UNKNOWN")
            
            # Simulate rug pull analysis
            if "HIGH_LIQUIDITY" in symbol:
                liquidity_stability = random.uniform(0.7, 0.9)  # 70-90% stability
                volume_consistency = random.uniform(0.6, 0.8)  # 60-80% consistency
                price_stability = random.uniform(0.7, 0.9)  # 70-90% stability
                rug_pull_confidence = random.uniform(0.6, 0.9)  # 60-90% confidence
            elif "MEDIUM_LIQUIDITY" in symbol:
                liquidity_stability = random.uniform(0.5, 0.7)  # 50-70% stability
                volume_consistency = random.uniform(0.4, 0.6)  # 40-60% consistency
                price_stability = random.uniform(0.5, 0.7)  # 50-70% stability
                rug_pull_confidence = random.uniform(0.4, 0.7)  # 40-70% confidence
            else:
                liquidity_stability = random.uniform(0.3, 0.5)  # 30-50% stability
                volume_consistency = random.uniform(0.2, 0.4)  # 20-40% consistency
                price_stability = random.uniform(0.3, 0.5)  # 30-50% stability
                rug_pull_confidence = random.uniform(0.3, 0.6)  # 30-60% confidence
            
            # Calculate rug pull risk score
            rug_pull_risk = (
                (1.0 - liquidity_stability) * 0.4 +
                (1.0 - volume_consistency) * 0.3 +
                (1.0 - price_stability) * 0.3
            )
            
            # Determine rug pull risk level
            if rug_pull_risk > 0.7:
                risk_level = "critical"
                risk_characteristics = "very_high"
            elif rug_pull_risk > 0.5:
                risk_level = "high"
                risk_characteristics = "high"
            elif rug_pull_risk > 0.3:
                risk_level = "medium"
                risk_characteristics = "moderate"
            else:
                risk_level = "low"
                risk_characteristics = "low"
            
            return {
                'rug_pull_risk': rug_pull_risk,
                'liquidity_stability': liquidity_stability,
                'volume_consistency': volume_consistency,
                'price_stability': price_stability,
                'rug_pull_confidence': rug_pull_confidence,
                'risk_level': risk_level,
                'risk_characteristics': risk_characteristics,
                'rug_pull_signal': 'high_risk' if rug_pull_risk > 0.6 else 'medium_risk' if rug_pull_risk > 0.3 else 'low_risk'
            }
            
        except Exception:
            return {
                'rug_pull_risk': 0.5,
                'liquidity_stability': 0.5,
                'volume_consistency': 0.5,
                'price_stability': 0.5,
                'rug_pull_confidence': 0.5,
                'risk_level': 'medium',
                'risk_characteristics': 'moderate',
                'rug_pull_signal': 'medium_risk'
            }
    
    def _analyze_manipulation_risk(self, token: Dict) -> Dict:
        """Analyze market manipulation risk"""
        try:
            symbol = token.get("symbol", "UNKNOWN")
            
            # Simulate manipulation analysis
            if "HIGH_LIQUIDITY" in symbol:
                price_spike_probability = random.uniform(0.1, 0.3)  # 10-30% spike probability
                volume_spike_probability = random.uniform(0.2, 0.4)  # 20-40% spike probability
                manipulation_pattern = random.uniform(0.2, 0.5)  # 20-50% manipulation pattern
                manipulation_confidence = random.uniform(0.6, 0.9)  # 60-90% confidence
            elif "MEDIUM_LIQUIDITY" in symbol:
                price_spike_probability = random.uniform(0.2, 0.5)  # 20-50% spike probability
                volume_spike_probability = random.uniform(0.3, 0.6)  # 30-60% spike probability
                manipulation_pattern = random.uniform(0.3, 0.6)  # 30-60% manipulation pattern
                manipulation_confidence = random.uniform(0.4, 0.7)  # 40-70% confidence
            else:
                price_spike_probability = random.uniform(0.3, 0.7)  # 30-70% spike probability
                volume_spike_probability = random.uniform(0.4, 0.8)  # 40-80% spike probability
                manipulation_pattern = random.uniform(0.4, 0.8)  # 40-80% manipulation pattern
                manipulation_confidence = random.uniform(0.3, 0.6)  # 30-60% confidence
            
            # Calculate manipulation risk score
            manipulation_risk = (
                price_spike_probability * 0.3 +
                volume_spike_probability * 0.3 +
                manipulation_pattern * 0.4
            )
            
            # Determine manipulation risk level
            if manipulation_risk > 0.7:
                risk_level = "high"
                risk_characteristics = "very_high"
            elif manipulation_risk > 0.5:
                risk_level = "medium"
                risk_characteristics = "high"
            elif manipulation_risk > 0.3:
                risk_level = "low"
                risk_characteristics = "moderate"
            else:
                risk_level = "very_low"
                risk_characteristics = "low"
            
            return {
                'manipulation_risk': manipulation_risk,
                'price_spike_probability': price_spike_probability,
                'volume_spike_probability': volume_spike_probability,
                'manipulation_pattern': manipulation_pattern,
                'manipulation_confidence': manipulation_confidence,
                'risk_level': risk_level,
                'risk_characteristics': risk_characteristics,
                'manipulation_signal': 'high_risk' if manipulation_risk > 0.6 else 'medium_risk' if manipulation_risk > 0.3 else 'low_risk'
            }
            
        except Exception:
            return {
                'manipulation_risk': 0.5,
                'price_spike_probability': 0.3,
                'volume_spike_probability': 0.4,
                'manipulation_pattern': 0.4,
                'manipulation_confidence': 0.5,
                'risk_level': 'medium',
                'risk_characteristics': 'moderate',
                'manipulation_signal': 'medium_risk'
            }
    
    def _analyze_liquidity_drain_risk(self, token: Dict) -> Dict:
        """Analyze liquidity drain risk"""
        try:
            symbol = token.get("symbol", "UNKNOWN")
            
            # Simulate liquidity drain analysis
            if "HIGH_LIQUIDITY" in symbol:
                liquidity_stability = random.uniform(0.7, 0.9)  # 70-90% stability
                volume_consistency = random.uniform(0.6, 0.8)  # 60-80% consistency
                liquidity_drain_probability = random.uniform(0.1, 0.3)  # 10-30% drain probability
                liquidity_confidence = random.uniform(0.6, 0.9)  # 60-90% confidence
            elif "MEDIUM_LIQUIDITY" in symbol:
                liquidity_stability = random.uniform(0.5, 0.7)  # 50-70% stability
                volume_consistency = random.uniform(0.4, 0.6)  # 40-60% consistency
                liquidity_drain_probability = random.uniform(0.2, 0.5)  # 20-50% drain probability
                liquidity_confidence = random.uniform(0.4, 0.7)  # 40-70% confidence
            else:
                liquidity_stability = random.uniform(0.3, 0.5)  # 30-50% stability
                volume_consistency = random.uniform(0.2, 0.4)  # 20-40% consistency
                liquidity_drain_probability = random.uniform(0.3, 0.7)  # 30-70% drain probability
                liquidity_confidence = random.uniform(0.3, 0.6)  # 30-60% confidence
            
            # Calculate liquidity drain risk score
            liquidity_drain_risk = (
                (1.0 - liquidity_stability) * 0.4 +
                (1.0 - volume_consistency) * 0.3 +
                liquidity_drain_probability * 0.3
            )
            
            # Determine liquidity drain risk level
            if liquidity_drain_risk > 0.7:
                risk_level = "high"
                risk_characteristics = "very_high"
            elif liquidity_drain_risk > 0.5:
                risk_level = "medium"
                risk_characteristics = "high"
            elif liquidity_drain_risk > 0.3:
                risk_level = "low"
                risk_characteristics = "moderate"
            else:
                risk_level = "very_low"
                risk_characteristics = "low"
            
            return {
                'liquidity_drain_risk': liquidity_drain_risk,
                'liquidity_stability': liquidity_stability,
                'volume_consistency': volume_consistency,
                'liquidity_drain_probability': liquidity_drain_probability,
                'liquidity_confidence': liquidity_confidence,
                'risk_level': risk_level,
                'risk_characteristics': risk_characteristics,
                'liquidity_drain_signal': 'high_risk' if liquidity_drain_risk > 0.6 else 'medium_risk' if liquidity_drain_risk > 0.3 else 'low_risk'
            }
            
        except Exception:
            return {
                'liquidity_drain_risk': 0.5,
                'liquidity_stability': 0.5,
                'volume_consistency': 0.5,
                'liquidity_drain_probability': 0.3,
                'liquidity_confidence': 0.5,
                'risk_level': 'medium',
                'risk_characteristics': 'moderate',
                'liquidity_drain_signal': 'medium_risk'
            }
    
    def _analyze_correlation_breakdown_risk(self, token: Dict) -> Dict:
        """Analyze correlation breakdown risk"""
        try:
            symbol = token.get("symbol", "UNKNOWN")
            
            # Simulate correlation breakdown analysis
            if "HIGH_LIQUIDITY" in symbol:
                btc_correlation = random.uniform(0.6, 0.9)  # 60-90% BTC correlation
                eth_correlation = random.uniform(0.5, 0.8)  # 50-80% ETH correlation
                market_correlation = random.uniform(0.7, 0.9)  # 70-90% market correlation
                correlation_breakdown_probability = random.uniform(0.1, 0.3)  # 10-30% breakdown probability
            elif "MEDIUM_LIQUIDITY" in symbol:
                btc_correlation = random.uniform(0.4, 0.7)  # 40-70% BTC correlation
                eth_correlation = random.uniform(0.3, 0.6)  # 30-60% ETH correlation
                market_correlation = random.uniform(0.5, 0.8)  # 50-80% market correlation
                correlation_breakdown_probability = random.uniform(0.2, 0.5)  # 20-50% breakdown probability
            else:
                btc_correlation = random.uniform(0.2, 0.5)  # 20-50% BTC correlation
                eth_correlation = random.uniform(0.2, 0.4)  # 20-40% ETH correlation
                market_correlation = random.uniform(0.3, 0.6)  # 30-60% market correlation
                correlation_breakdown_probability = random.uniform(0.3, 0.7)  # 30-70% breakdown probability
            
            # Calculate correlation breakdown risk score
            avg_correlation = (btc_correlation + eth_correlation + market_correlation) / 3
            correlation_breakdown_risk = (
                (1.0 - avg_correlation) * 0.5 +
                correlation_breakdown_probability * 0.5
            )
            
            # Determine correlation breakdown risk level
            if correlation_breakdown_risk > 0.7:
                risk_level = "high"
                risk_characteristics = "very_high"
            elif correlation_breakdown_risk > 0.5:
                risk_level = "medium"
                risk_characteristics = "high"
            elif correlation_breakdown_risk > 0.3:
                risk_level = "low"
                risk_characteristics = "moderate"
            else:
                risk_level = "very_low"
                risk_characteristics = "low"
            
            return {
                'correlation_breakdown_risk': correlation_breakdown_risk,
                'btc_correlation': btc_correlation,
                'eth_correlation': eth_correlation,
                'market_correlation': market_correlation,
                'avg_correlation': avg_correlation,
                'correlation_breakdown_probability': correlation_breakdown_probability,
                'risk_level': risk_level,
                'risk_characteristics': risk_characteristics,
                'correlation_breakdown_signal': 'high_risk' if correlation_breakdown_risk > 0.6 else 'medium_risk' if correlation_breakdown_risk > 0.3 else 'low_risk'
            }
            
        except Exception:
            return {
                'correlation_breakdown_risk': 0.5,
                'btc_correlation': 0.5,
                'eth_correlation': 0.4,
                'market_correlation': 0.6,
                'avg_correlation': 0.5,
                'correlation_breakdown_probability': 0.3,
                'risk_level': 'medium',
                'risk_characteristics': 'moderate',
                'correlation_breakdown_signal': 'medium_risk'
            }
    
    def _analyze_black_swan_risk(self, token: Dict) -> Dict:
        """Analyze black swan event risk"""
        try:
            symbol = token.get("symbol", "UNKNOWN")
            
            # Simulate black swan analysis
            if "HIGH_LIQUIDITY" in symbol:
                price_drop_probability = random.uniform(0.05, 0.2)  # 5-20% drop probability
                volume_spike_probability = random.uniform(0.1, 0.3)  # 10-30% spike probability
                news_impact_probability = random.uniform(0.1, 0.3)  # 10-30% news impact probability
                black_swan_confidence = random.uniform(0.6, 0.9)  # 60-90% confidence
            elif "MEDIUM_LIQUIDITY" in symbol:
                price_drop_probability = random.uniform(0.1, 0.4)  # 10-40% drop probability
                volume_spike_probability = random.uniform(0.2, 0.5)  # 20-50% spike probability
                news_impact_probability = random.uniform(0.2, 0.5)  # 20-50% news impact probability
                black_swan_confidence = random.uniform(0.4, 0.7)  # 40-70% confidence
            else:
                price_drop_probability = random.uniform(0.2, 0.6)  # 20-60% drop probability
                volume_spike_probability = random.uniform(0.3, 0.7)  # 30-70% spike probability
                news_impact_probability = random.uniform(0.3, 0.7)  # 30-70% news impact probability
                black_swan_confidence = random.uniform(0.3, 0.6)  # 30-60% confidence
            
            # Calculate black swan risk score
            black_swan_risk = (
                price_drop_probability * 0.4 +
                volume_spike_probability * 0.3 +
                news_impact_probability * 0.3
            )
            
            # Determine black swan risk level
            if black_swan_risk > 0.8:
                risk_level = "critical"
                risk_characteristics = "very_high"
            elif black_swan_risk > 0.6:
                risk_level = "high"
                risk_characteristics = "high"
            elif black_swan_risk > 0.4:
                risk_level = "medium"
                risk_characteristics = "moderate"
            else:
                risk_level = "low"
                risk_characteristics = "low"
            
            return {
                'black_swan_risk': black_swan_risk,
                'price_drop_probability': price_drop_probability,
                'volume_spike_probability': volume_spike_probability,
                'news_impact_probability': news_impact_probability,
                'black_swan_confidence': black_swan_confidence,
                'risk_level': risk_level,
                'risk_characteristics': risk_characteristics,
                'black_swan_signal': 'critical_risk' if black_swan_risk > 0.8 else 'high_risk' if black_swan_risk > 0.6 else 'medium_risk' if black_swan_risk > 0.4 else 'low_risk'
            }
            
        except Exception:
            return {
                'black_swan_risk': 0.5,
                'price_drop_probability': 0.2,
                'volume_spike_probability': 0.3,
                'news_impact_probability': 0.3,
                'black_swan_confidence': 0.5,
                'risk_level': 'medium',
                'risk_characteristics': 'moderate',
                'black_swan_signal': 'medium_risk'
            }
    
    def _calculate_risk_score(self, flash_crash_analysis: Dict, rug_pull_analysis: Dict,
                             manipulation_analysis: Dict, liquidity_drain_analysis: Dict,
                             correlation_breakdown_analysis: Dict, black_swan_analysis: Dict) -> float:
        """Calculate overall risk score"""
        try:
            # Weight the individual risk scores
            flash_crash_risk = flash_crash_analysis.get('flash_crash_risk', 0.5)
            rug_pull_risk = rug_pull_analysis.get('rug_pull_risk', 0.5)
            manipulation_risk = manipulation_analysis.get('manipulation_risk', 0.5)
            liquidity_drain_risk = liquidity_drain_analysis.get('liquidity_drain_risk', 0.5)
            correlation_breakdown_risk = correlation_breakdown_analysis.get('correlation_breakdown_risk', 0.5)
            black_swan_risk = black_swan_analysis.get('black_swan_risk', 0.5)
            
            # Calculate weighted average
            risk_score = (
                flash_crash_risk * self.risk_factors['price_volatility'] +
                rug_pull_risk * self.risk_factors['volume_anomalies'] +
                manipulation_risk * self.risk_factors['liquidity_analysis'] +
                liquidity_drain_risk * self.risk_factors['market_manipulation'] +
                correlation_breakdown_risk * self.risk_factors['correlation_analysis'] +
                black_swan_risk * self.risk_factors['news_sentiment']
            )
            
            return max(0.0, min(1.0, risk_score))
            
        except Exception:
            return 0.5
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level based on risk score"""
        try:
            if risk_score > 0.8:
                return "critical"
            elif risk_score > 0.6:
                return "high"
            elif risk_score > 0.4:
                return "medium"
            else:
                return "low"
                
        except Exception:
            return "medium"
    
    def _generate_prevention_recommendations(self, risk_level: str, flash_crash_analysis: Dict,
                                           rug_pull_analysis: Dict, manipulation_analysis: Dict,
                                           liquidity_drain_analysis: Dict, correlation_breakdown_analysis: Dict,
                                           black_swan_analysis: Dict) -> List[str]:
        """Generate risk prevention recommendations"""
        recommendations = []
        
        try:
            # Risk level recommendations
            if risk_level == "critical":
                recommendations.append("CRITICAL RISK: Avoid trading immediately")
                recommendations.append("Consider emergency exit from all positions")
                recommendations.append("Monitor market conditions closely")
            elif risk_level == "high":
                recommendations.append("HIGH RISK: Reduce position sizes significantly")
                recommendations.append("Avoid new positions until risk decreases")
                recommendations.append("Monitor for early exit signals")
            elif risk_level == "medium":
                recommendations.append("MEDIUM RISK: Use conservative position sizing")
                recommendations.append("Monitor risk indicators closely")
                recommendations.append("Consider risk-adjusted strategies")
            else:
                recommendations.append("LOW RISK: Normal trading conditions")
                recommendations.append("Continue with current strategy")
                recommendations.append("Monitor for risk changes")
            
            # Flash crash recommendations
            if flash_crash_analysis.get('flash_crash_risk', 0.5) > 0.7:
                recommendations.append("Flash crash risk detected - use tight stop losses")
                recommendations.append("Consider reducing position sizes")
                recommendations.append("Monitor price volatility closely")
            
            # Rug pull recommendations
            if rug_pull_analysis.get('rug_pull_risk', 0.5) > 0.6:
                recommendations.append("Rug pull risk detected - avoid trading")
                recommendations.append("Check token liquidity and volume consistency")
                recommendations.append("Consider avoiding low-liquidity tokens")
            
            # Manipulation recommendations
            if manipulation_analysis.get('manipulation_risk', 0.5) > 0.6:
                recommendations.append("Market manipulation risk detected - avoid trading")
                recommendations.append("Monitor for pump and dump patterns")
                recommendations.append("Consider avoiding high-volatility tokens")
            
            # Liquidity drain recommendations
            if liquidity_drain_analysis.get('liquidity_drain_risk', 0.5) > 0.6:
                recommendations.append("Liquidity drain risk detected - reduce positions")
                recommendations.append("Monitor liquidity levels closely")
                recommendations.append("Consider exiting positions early")
            
            # Correlation breakdown recommendations
            if correlation_breakdown_analysis.get('correlation_breakdown_risk', 0.5) > 0.6:
                recommendations.append("Correlation breakdown risk detected - monitor closely")
                recommendations.append("Consider reducing position sizes")
                recommendations.append("Monitor market correlations")
            
            # Black swan recommendations
            if black_swan_analysis.get('black_swan_risk', 0.5) > 0.7:
                recommendations.append("Black swan risk detected - emergency protocols")
                recommendations.append("Consider exiting all positions")
                recommendations.append("Monitor news and market events closely")
            
        except Exception:
            recommendations.append("Monitor risk indicators and adjust strategy accordingly")
        
        return recommendations
    
    def _calculate_risk_confidence(self, flash_crash_analysis: Dict, rug_pull_analysis: Dict,
                                 manipulation_analysis: Dict, liquidity_drain_analysis: Dict,
                                 correlation_breakdown_analysis: Dict, black_swan_analysis: Dict) -> str:
        """Calculate confidence in risk analysis"""
        try:
            # Analyze analysis consistency
            analysis_scores = [
                flash_crash_analysis.get('flash_crash_confidence', 0.5),
                rug_pull_analysis.get('rug_pull_confidence', 0.5),
                manipulation_analysis.get('manipulation_confidence', 0.5),
                liquidity_drain_analysis.get('liquidity_confidence', 0.5),
                black_swan_analysis.get('black_swan_confidence', 0.5)
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
    
    def _generate_risk_insights(self, flash_crash_analysis: Dict, rug_pull_analysis: Dict,
                               manipulation_analysis: Dict, liquidity_drain_analysis: Dict,
                               correlation_breakdown_analysis: Dict, black_swan_analysis: Dict) -> List[str]:
        """Generate risk insights"""
        insights = []
        
        try:
            # Flash crash insights
            if flash_crash_analysis.get('flash_crash_risk', 0.5) > 0.7:
                insights.append("High flash crash risk detected - monitor price volatility")
            elif flash_crash_analysis.get('flash_crash_risk', 0.5) < 0.3:
                insights.append("Low flash crash risk - stable price conditions")
            
            # Rug pull insights
            if rug_pull_analysis.get('rug_pull_risk', 0.5) > 0.6:
                insights.append("High rug pull risk detected - check token stability")
            elif rug_pull_analysis.get('rug_pull_risk', 0.5) < 0.3:
                insights.append("Low rug pull risk - stable token conditions")
            
            # Manipulation insights
            if manipulation_analysis.get('manipulation_risk', 0.5) > 0.6:
                insights.append("High manipulation risk detected - monitor for pump and dump")
            elif manipulation_analysis.get('manipulation_risk', 0.5) < 0.3:
                insights.append("Low manipulation risk - natural market conditions")
            
            # Liquidity drain insights
            if liquidity_drain_analysis.get('liquidity_drain_risk', 0.5) > 0.6:
                insights.append("High liquidity drain risk detected - monitor liquidity levels")
            elif liquidity_drain_analysis.get('liquidity_drain_risk', 0.5) < 0.3:
                insights.append("Low liquidity drain risk - stable liquidity conditions")
            
            # Correlation breakdown insights
            if correlation_breakdown_analysis.get('correlation_breakdown_risk', 0.5) > 0.6:
                insights.append("High correlation breakdown risk detected - monitor correlations")
            elif correlation_breakdown_analysis.get('correlation_breakdown_risk', 0.5) < 0.3:
                insights.append("Low correlation breakdown risk - stable correlations")
            
            # Black swan insights
            if black_swan_analysis.get('black_swan_risk', 0.5) > 0.7:
                insights.append("High black swan risk detected - monitor for major events")
            elif black_swan_analysis.get('black_swan_risk', 0.5) < 0.3:
                insights.append("Low black swan risk - stable market conditions")
            
        except Exception:
            insights.append("Risk analysis completed")
        
        return insights
    
    def _calculate_emergency_actions(self, risk_level: str, risk_score: float,
                                   flash_crash_analysis: Dict, rug_pull_analysis: Dict,
                                   manipulation_analysis: Dict, liquidity_drain_analysis: Dict) -> Dict:
        """Calculate emergency actions based on risk level"""
        try:
            emergency_actions = {
                'immediate_actions': [],
                'monitoring_actions': [],
                'prevention_actions': []
            }
            
            # Immediate actions based on risk level
            if risk_level == "critical":
                emergency_actions['immediate_actions'].extend([
                    "Exit all positions immediately",
                    "Stop all new trades",
                    "Activate emergency protocols"
                ])
            elif risk_level == "high":
                emergency_actions['immediate_actions'].extend([
                    "Reduce position sizes by 50%",
                    "Avoid new trades",
                    "Monitor for exit signals"
                ])
            elif risk_level == "medium":
                emergency_actions['immediate_actions'].extend([
                    "Use conservative position sizing",
                    "Monitor risk indicators",
                    "Consider risk-adjusted strategies"
                ])
            else:
                emergency_actions['immediate_actions'].extend([
                    "Continue normal trading",
                    "Monitor for risk changes",
                    "Maintain current strategy"
                ])
            
            # Monitoring actions
            emergency_actions['monitoring_actions'].extend([
                "Monitor price volatility",
                "Monitor liquidity levels",
                "Monitor volume patterns",
                "Monitor correlation changes"
            ])
            
            # Prevention actions
            emergency_actions['prevention_actions'].extend([
                "Use tight stop losses",
                "Monitor for early exit signals",
                "Avoid high-risk tokens",
                "Maintain risk management protocols"
            ])
            
            return emergency_actions
            
        except Exception:
            return {
                'immediate_actions': ["Monitor risk indicators"],
                'monitoring_actions': ["Monitor market conditions"],
                'prevention_actions': ["Maintain risk management"]
            }
    
    def _get_default_risk_analysis(self, token: Dict, trade_amount: float) -> Dict:
        """Return default risk analysis when analysis fails"""
        return {
            'risk_score': 0.5,
            'risk_level': 'medium',
            'risk_confidence': 'medium',
            'flash_crash_analysis': {
                'flash_crash_risk': 0.5,
                'risk_level': 'medium',
                'flash_crash_signal': 'medium_risk'
            },
            'rug_pull_analysis': {
                'rug_pull_risk': 0.5,
                'risk_level': 'medium',
                'rug_pull_signal': 'medium_risk'
            },
            'manipulation_analysis': {
                'manipulation_risk': 0.5,
                'risk_level': 'medium',
                'manipulation_signal': 'medium_risk'
            },
            'liquidity_drain_analysis': {
                'liquidity_drain_risk': 0.5,
                'risk_level': 'medium',
                'liquidity_drain_signal': 'medium_risk'
            },
            'correlation_breakdown_analysis': {
                'correlation_breakdown_risk': 0.5,
                'risk_level': 'medium',
                'correlation_breakdown_signal': 'medium_risk'
            },
            'black_swan_analysis': {
                'black_swan_risk': 0.5,
                'risk_level': 'medium',
                'black_swan_signal': 'medium_risk'
            },
            'prevention_recommendations': ['Monitor risk indicators'],
            'risk_insights': ['Risk analysis completed'],
            'emergency_actions': {
                'immediate_actions': ['Monitor risk indicators'],
                'monitoring_actions': ['Monitor market conditions'],
                'prevention_actions': ['Maintain risk management']
            },
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def get_risk_summary(self, tokens: List[Dict], trade_amounts: List[float]) -> Dict:
        """Get risk summary for multiple tokens"""
        try:
            risk_summaries = []
            critical_risk_count = 0
            high_risk_count = 0
            medium_risk_count = 0
            low_risk_count = 0
            
            for i, token in enumerate(tokens):
                trade_amount = trade_amounts[i] if i < len(trade_amounts) else 5.0
                risk_analysis = self.predict_risk(token, trade_amount)
                
                risk_summaries.append({
                    'symbol': token.get('symbol', 'UNKNOWN'),
                    'risk_score': risk_analysis['risk_score'],
                    'risk_level': risk_analysis['risk_level'],
                    'risk_confidence': risk_analysis['risk_confidence']
                })
                
                risk_level = risk_analysis['risk_level']
                if risk_level == 'critical':
                    critical_risk_count += 1
                elif risk_level == 'high':
                    high_risk_count += 1
                elif risk_level == 'medium':
                    medium_risk_count += 1
                else:
                    low_risk_count += 1
            
            return {
                'total_tokens': len(tokens),
                'critical_risk': critical_risk_count,
                'high_risk': high_risk_count,
                'medium_risk': medium_risk_count,
                'low_risk': low_risk_count,
                'risk_summaries': risk_summaries,
                'overall_risk': 'critical' if critical_risk_count > 0 else 'high' if high_risk_count > len(tokens) * 0.3 else 'medium' if medium_risk_count > len(tokens) * 0.3 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error getting risk summary: {e}")
            return {
                'total_tokens': len(tokens),
                'critical_risk': 0,
                'high_risk': 0,
                'medium_risk': 0,
                'low_risk': 0,
                'risk_summaries': [],
                'overall_risk': 'unknown'
            }

# Global instance
ai_risk_prediction_prevention_system = AIRiskPredictionPreventionSystem()
