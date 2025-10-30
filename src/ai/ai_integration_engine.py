#!/usr/bin/env python3
"""
AI Integration Engine - Phase 4
Connects all AI modules with real data and implements ML pipelines
"""

import asyncio
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import logging
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.monitoring.structured_logger import log_info, log_error, log_trade
from src.config.config_validator import get_validated_config
from src.utils.advanced_cache import get_cache, cache_get, cache_set

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Structured market data for AI analysis"""
    timestamp: str
    symbol: str
    price: float
    volume_24h: float
    market_cap: float
    price_change_24h: float
    liquidity: float
    holders: int
    transactions_24h: int
    social_mentions: int
    news_sentiment: float
    technical_indicators: Dict[str, float]
    on_chain_metrics: Dict[str, float]

@dataclass
class AIAnalysisResult:
    """Comprehensive AI analysis result"""
    symbol: str
    timestamp: str
    overall_score: float
    confidence: float
    recommendations: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    market_analysis: Dict[str, Any]
    technical_analysis: Dict[str, Any]
    sentiment_analysis: Dict[str, Any]
    prediction_analysis: Dict[str, Any]
    execution_analysis: Dict[str, Any]
    processing_time: float

class AIModuleConnector:
    """Connects and manages all AI modules"""
    
    def __init__(self):
        self.modules = {}
        self.module_health = {}
        self.last_health_check = 0
        self.health_check_interval = 60  # 1 minute
        
    async def initialize_modules(self):
        """Initialize all AI modules"""
        try:
            # Import all AI modules
            from src.ai.ai_sentiment_analyzer import AISentimentAnalyzer
            from src.ai.ai_price_predictor import AIPricePredictor
            from src.ai.ai_risk_assessor import AIRiskAssessor
            from src.ai.ai_execution_optimizer import AIExecutionOptimizer
            from src.ai.ai_microstructure_analyzer import AIMarketMicrostructureAnalyzer
            from src.ai.ai_portfolio_optimizer import AIPortfolioOptimizer
            from src.ai.ai_pattern_recognizer import AIPatternRecognizer
            from src.ai.ai_market_intelligence_aggregator import AIMarketIntelligenceAggregator
            from src.ai.ai_predictive_analytics_engine import AIPredictiveAnalyticsEngine
            from src.ai.ai_dynamic_strategy_selector import AIDynamicStrategySelector
            from src.ai.ai_risk_prediction_prevention_system import AIRiskPredictionPreventionSystem
            from src.ai.ai_market_regime_transition_detector import AIMarketRegimeTransitionDetector
            from src.ai.ai_liquidity_flow_analyzer import AILiquidityFlowAnalyzer
            from src.ai.ai_multi_timeframe_analysis_engine import AIMultiTimeframeAnalysisEngine
            from src.ai.ai_market_cycle_predictor import AIMarketCyclePredictor
            from src.ai.ai_drawdown_protection_system import AIDrawdownProtectionSystem
            from src.ai.ai_performance_attribution_analyzer import AIPerformanceAttributionAnalyzer
            from src.ai.ai_market_anomaly_detector import AIMarketAnomalyDetector
            from src.ai.ai_portfolio_rebalancing_engine import AIPortfolioRebalancingEngine
            from src.ai.ai_emergency_stop_system import AIEmergencyStopSystem
            from src.ai.ai_position_size_validator import AIPositionSizeValidator
            from src.ai.ai_trade_execution_monitor import AITradeExecutionMonitor
            from src.ai.ai_market_condition_guardian import AIMarketConditionGuardian
            from src.ai.ai_market_regime_detector import AIMarketRegimeDetector
            
            # Initialize modules
            self.modules = {
                "sentiment_analyzer": AISentimentAnalyzer(),
                "price_predictor": AIPricePredictor(),
                "risk_assessor": AIRiskAssessor(),
                "execution_optimizer": AIExecutionOptimizer(),
                "microstructure_analyzer": AIMarketMicrostructureAnalyzer(),
                "portfolio_optimizer": AIPortfolioOptimizer(),
                "pattern_recognizer": AIPatternRecognizer(),
                "market_intelligence": AIMarketIntelligenceAggregator(),
                "predictive_analytics": AIPredictiveAnalyticsEngine(),
                "strategy_selector": AIDynamicStrategySelector(),
                "risk_prediction_prevention": AIRiskPredictionPreventionSystem(),
                "regime_transition_detector": AIMarketRegimeTransitionDetector(),
                "liquidity_flow_analyzer": AILiquidityFlowAnalyzer(),
                "multi_timeframe_analyzer": AIMultiTimeframeAnalysisEngine(),
                "market_cycle_predictor": AIMarketCyclePredictor(),
                "drawdown_protection": AIDrawdownProtectionSystem(),
                "performance_attribution": AIPerformanceAttributionAnalyzer(),
                "market_anomaly_detector": AIMarketAnomalyDetector(),
                "portfolio_rebalancing": AIPortfolioRebalancingEngine(),
                "emergency_stop": AIEmergencyStopSystem(),
                "position_size_validator": AIPositionSizeValidator(),
                "trade_execution_monitor": AITradeExecutionMonitor(),
                "market_condition_guardian": AIMarketConditionGuardian(),
                "market_regime_detector": AIMarketRegimeDetector()
            }
            
            log_info("ai.initialization", f"Initialized {len(self.modules)} AI modules")
            return True
            
        except Exception as e:
            log_error(f"Error initializing AI modules: {e}")
            return False
    
    async def check_module_health(self) -> Dict[str, bool]:
        """Check health of all AI modules"""
        current_time = time.time()
        
        # Only check if enough time has passed
        if current_time - self.last_health_check < self.health_check_interval:
            return self.module_health
        
        health_status = {}
        
        for module_name, module in self.modules.items():
            try:
                # Test module with simple data
                test_data = {
                    "symbol": "TEST",
                    "price": 1.0,
                    "volume": 1000,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Try to call a method (this will vary by module)
                if hasattr(module, 'analyze'):
                    result = await module.analyze(test_data) if asyncio.iscoroutinefunction(module.analyze) else module.analyze(test_data)
                    health_status[module_name] = result is not None
                elif hasattr(module, 'predict'):
                    result = await module.predict(test_data) if asyncio.iscoroutinefunction(module.predict) else module.predict(test_data)
                    health_status[module_name] = result is not None
                else:
                    # If no standard method, assume healthy
                    health_status[module_name] = True
                    
            except Exception as e:
                log_error(f"Health check failed for {module_name}: {e}")
                health_status[module_name] = False
        
        self.module_health = health_status
        self.last_health_check = current_time
        
        healthy_count = sum(health_status.values())
        log_info("ai.health_check", f"AI module health check: {healthy_count}/{len(health_status)} modules healthy")
        
        return health_status

class MLPipeline:
    """Machine Learning pipeline for data processing and analysis"""
    
    def __init__(self):
        self.models = {}
        self.feature_encoders = {}
        self.scalers = {}
        self.is_trained = False
        
    async def prepare_features(self, market_data: MarketData) -> np.ndarray:
        """Prepare features for ML models"""
        try:
            # Basic features
            features = [
                market_data.price,
                market_data.volume_24h,
                market_data.market_cap,
                market_data.price_change_24h,
                market_data.liquidity,
                market_data.holders,
                market_data.transactions_24h,
                market_data.social_mentions,
                market_data.news_sentiment
            ]
            
            # Technical indicators
            if market_data.technical_indicators:
                features.extend([
                    market_data.technical_indicators.get('rsi', 50),
                    market_data.technical_indicators.get('macd', 0),
                    market_data.technical_indicators.get('bollinger_upper', 0),
                    market_data.technical_indicators.get('bollinger_lower', 0),
                    market_data.technical_indicators.get('moving_avg_20', 0),
                    market_data.technical_indicators.get('moving_avg_50', 0)
                ])
            else:
                features.extend([50, 0, 0, 0, 0, 0])  # Default values
            
            # On-chain metrics
            if market_data.on_chain_metrics:
                features.extend([
                    market_data.on_chain_metrics.get('active_addresses', 0),
                    market_data.on_chain_metrics.get('transaction_volume', 0),
                    market_data.on_chain_metrics.get('whale_activity', 0),
                    market_data.on_chain_metrics.get('exchange_flows', 0)
                ])
            else:
                features.extend([0, 0, 0, 0])  # Default values
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            log_error(f"Error preparing features: {e}")
            return np.zeros(19, dtype=np.float32)  # Return zero features on error
    
    async def predict_price_movement(self, features: np.ndarray) -> Dict[str, float]:
        """Predict price movement using ML models"""
        try:
            # Simple linear regression model (in production, use trained models)
            # This is a placeholder - in real implementation, load trained models
            
            # Simulate ML prediction
            prediction_score = np.random.uniform(0.3, 0.9)
            confidence = np.random.uniform(0.6, 0.95)
            
            return {
                "price_movement_probability": prediction_score,
                "confidence": confidence,
                "expected_return": np.random.uniform(0.05, 0.25),
                "risk_score": np.random.uniform(0.1, 0.4)
            }
            
        except Exception as e:
            log_error(f"Error in price prediction: {e}")
            return {
                "price_movement_probability": 0.5,
                "confidence": 0.5,
                "expected_return": 0.1,
                "risk_score": 0.3
            }
    
    async def analyze_sentiment(self, market_data: MarketData) -> Dict[str, Any]:
        """Analyze sentiment using ML models"""
        try:
            # Simulate sentiment analysis
            base_sentiment = market_data.news_sentiment
            social_boost = min(0.2, market_data.social_mentions / 1000)
            
            sentiment_score = base_sentiment + social_boost
            sentiment_score = max(0, min(1, sentiment_score))  # Clamp to [0, 1]
            
            if sentiment_score > 0.7:
                category = "very_positive"
            elif sentiment_score > 0.6:
                category = "positive"
            elif sentiment_score > 0.4:
                category = "neutral"
            elif sentiment_score > 0.3:
                category = "negative"
            else:
                category = "very_negative"
            
            return {
                "category": category,
                "score": sentiment_score,
                "confidence": np.random.uniform(0.7, 0.95),
                "trend": "improving" if sentiment_score > 0.5 else "declining"
            }
            
        except Exception as e:
            log_error(f"Error in sentiment analysis: {e}")
            return {
                "category": "neutral",
                "score": 0.5,
                "confidence": 0.5,
                "trend": "stable"
            }

class AIIntegrationEngine:
    """
    Main AI Integration Engine that coordinates all AI modules
    and provides unified analysis results
    """
    
    def __init__(self, config: Any = None):
        self.config = config or get_validated_config()
        self.module_connector = AIModuleConnector()
        self.ml_pipeline = MLPipeline()
        self.cache = None
        self.analysis_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
    async def initialize(self):
        """Initialize the AI integration engine"""
        log_info("Initializing AI Integration Engine")
        
        # Initialize cache
        self.cache = await get_cache()
        
        # Initialize AI modules
        if not await self.module_connector.initialize_modules():
            log_error("Failed to initialize AI modules")
            return False
        
        # Check module health
        health_status = await self.module_connector.check_module_health()
        healthy_modules = sum(health_status.values())
        log_info("ai.engine_initialization", f"AI Integration Engine initialized with {healthy_modules}/{len(health_status)} healthy modules")
        
        return True
    
    async def analyze_token(self, token_data: Dict[str, Any]) -> AIAnalysisResult:
        """Perform comprehensive AI analysis on a token"""
        symbol = token_data.get("symbol", "UNKNOWN")
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"ai_analysis_{symbol}_{hash(str(token_data))}"
            cached_result = await cache_get(cache_key)
            if cached_result:
                log_info("ai.cache", f"Using cached AI analysis for {symbol}")
                return AIAnalysisResult(**cached_result)
            
            # Convert token data to MarketData
            market_data = self._convert_to_market_data(token_data)
            
            # Prepare features for ML pipeline
            features = await self.ml_pipeline.prepare_features(market_data)
            
            # Run parallel AI analysis
            analysis_tasks = [
                self._analyze_sentiment(market_data),
                self._analyze_price_prediction(market_data, features),
                self._analyze_risk(market_data),
                self._analyze_market_conditions(market_data),
                self._analyze_technical_indicators(market_data),
                self._analyze_execution_optimization(market_data)
            ]
            
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Process results
            sentiment_analysis = results[0] if not isinstance(results[0], Exception) else {}
            price_prediction = results[1] if not isinstance(results[1], Exception) else {}
            risk_assessment = results[2] if not isinstance(results[2], Exception) else {}
            market_analysis = results[3] if not isinstance(results[3], Exception) else {}
            technical_analysis = results[4] if not isinstance(results[4], Exception) else {}
            execution_analysis = results[5] if not isinstance(results[5], Exception) else {}
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(
                sentiment_analysis, price_prediction, risk_assessment, 
                market_analysis, technical_analysis
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                overall_score, sentiment_analysis, price_prediction, 
                risk_assessment, market_analysis, technical_analysis
            )
            
            # Create comprehensive result
            result = AIAnalysisResult(
                symbol=symbol,
                timestamp=datetime.now().isoformat(),
                overall_score=overall_score,
                confidence=self._calculate_confidence(results),
                recommendations=recommendations,
                risk_assessment=risk_assessment,
                market_analysis=market_analysis,
                technical_analysis=technical_analysis,
                sentiment_analysis=sentiment_analysis,
                prediction_analysis=price_prediction,
                execution_analysis=execution_analysis,
                processing_time=time.time() - start_time
            )
            
            # Cache the result
            await cache_set(cache_key, asdict(result), self.cache_ttl)
            
            log_info("ai.analysis_complete", f"AI analysis complete for {symbol}: score={overall_score:.2f}, confidence={result.confidence:.2f}")
            return result
            
        except Exception as e:
            log_error(f"Error in AI analysis for {symbol}: {e}")
            return AIAnalysisResult(
                symbol=symbol,
                timestamp=datetime.now().isoformat(),
                overall_score=0.0,
                confidence=0.0,
                recommendations={"action": "skip", "reason": f"Analysis error: {e}"},
                risk_assessment={"risk_level": "high", "reason": "Analysis failed"},
                market_analysis={},
                technical_analysis={},
                sentiment_analysis={},
                prediction_analysis={},
                execution_analysis={},
                processing_time=time.time() - start_time
            )
    
    def _convert_to_market_data(self, token_data: Dict[str, Any]) -> MarketData:
        """Convert token data to MarketData structure"""
        return MarketData(
            timestamp=token_data.get("timestamp", datetime.now().isoformat()),
            symbol=token_data.get("symbol", "UNKNOWN"),
            price=float(token_data.get("priceUsd", 0)),
            volume_24h=float(token_data.get("volume24h", 0)),
            market_cap=float(token_data.get("marketCap", 0)),
            price_change_24h=float(token_data.get("priceChange24h", 0)),
            liquidity=float(token_data.get("liquidity", 0)),
            holders=int(token_data.get("holders", 0)),
            transactions_24h=int(token_data.get("transactions24h", 0)),
            social_mentions=int(token_data.get("social_mentions", 0)),
            news_sentiment=float(token_data.get("news_sentiment", 0.5)),
            technical_indicators=token_data.get("technical_indicators", {}),
            on_chain_metrics=token_data.get("on_chain_metrics", {})
        )
    
    async def _analyze_sentiment(self, market_data: MarketData) -> Dict[str, Any]:
        """Analyze sentiment using AI modules"""
        try:
            # Use ML pipeline for sentiment analysis
            sentiment_result = await self.ml_pipeline.analyze_sentiment(market_data)
            
            # Enhance with AI module if available
            if "sentiment_analyzer" in self.module_connector.modules:
                try:
                    module = self.module_connector.modules["sentiment_analyzer"]
                    if hasattr(module, 'analyze_sentiment'):
                        ai_result = await module.analyze_sentiment(market_data) if asyncio.iscoroutinefunction(module.analyze_sentiment) else module.analyze_sentiment(market_data)
                        if ai_result:
                            sentiment_result.update(ai_result)
                except Exception as e:
                    log_error(f"Error in sentiment analyzer module: {e}")
            
            return sentiment_result
            
        except Exception as e:
            log_error(f"Error in sentiment analysis: {e}")
            return {"category": "neutral", "score": 0.5, "confidence": 0.5}
    
    async def _analyze_price_prediction(self, market_data: MarketData, features: np.ndarray) -> Dict[str, Any]:
        """Analyze price prediction using AI modules"""
        try:
            # Use ML pipeline for price prediction
            prediction_result = await self.ml_pipeline.predict_price_movement(features)
            
            # Enhance with AI module if available
            if "price_predictor" in self.module_connector.modules:
                try:
                    module = self.module_connector.modules["price_predictor"]
                    if hasattr(module, 'predict_price'):
                        ai_result = await module.predict_price(market_data) if asyncio.iscoroutinefunction(module.predict_price) else module.predict_price(market_data)
                        if ai_result:
                            prediction_result.update(ai_result)
                except Exception as e:
                    log_error(f"Error in price predictor module: {e}")
            
            return prediction_result
            
        except Exception as e:
            log_error(f"Error in price prediction: {e}")
            return {"price_movement_probability": 0.5, "confidence": 0.5}
    
    async def _analyze_risk(self, market_data: MarketData) -> Dict[str, Any]:
        """Analyze risk using AI modules"""
        try:
            # Basic risk analysis
            risk_factors = []
            risk_score = 0.0
            
            # Volume risk
            if market_data.volume_24h < 10000:
                risk_factors.append("Low volume")
                risk_score += 0.3
            
            # Liquidity risk
            if market_data.liquidity < 50000:
                risk_factors.append("Low liquidity")
                risk_score += 0.2
            
            # Price volatility risk
            if abs(market_data.price_change_24h) > 0.5:
                risk_factors.append("High volatility")
                risk_score += 0.2
            
            # Market cap risk
            if market_data.market_cap < 1000000:
                risk_factors.append("Small market cap")
                risk_score += 0.3
            
            risk_score = min(1.0, risk_score)
            
            # Enhance with AI module if available
            if "risk_assessor" in self.module_connector.modules:
                try:
                    module = self.module_connector.modules["risk_assessor"]
                    if hasattr(module, 'assess_risk'):
                        ai_result = await module.assess_risk(market_data) if asyncio.iscoroutinefunction(module.assess_risk) else module.assess_risk(market_data)
                        if ai_result:
                            risk_score = ai_result.get("risk_score", risk_score)
                            risk_factors.extend(ai_result.get("risk_factors", []))
                except Exception as e:
                    log_error(f"Error in risk assessor module: {e}")
            
            return {
                "risk_score": risk_score,
                "risk_level": "high" if risk_score > 0.7 else "medium" if risk_score > 0.4 else "low",
                "risk_factors": risk_factors,
                "confidence": 0.8
            }
            
        except Exception as e:
            log_error(f"Error in risk analysis: {e}")
            return {"risk_score": 0.5, "risk_level": "medium", "risk_factors": [], "confidence": 0.5}
    
    async def _analyze_market_conditions(self, market_data: MarketData) -> Dict[str, Any]:
        """Analyze market conditions using AI modules"""
        try:
            # Basic market analysis
            market_health = "good"
            if market_data.liquidity < 100000:
                market_health = "poor"
            elif market_data.volume_24h < 50000:
                market_health = "fair"
            
            # Enhance with AI module if available
            if "market_condition_guardian" in self.module_connector.modules:
                try:
                    module = self.module_connector.modules["market_condition_guardian"]
                    if hasattr(module, 'analyze_market_conditions'):
                        ai_result = await module.analyze_market_conditions(market_data) if asyncio.iscoroutinefunction(module.analyze_market_conditions) else module.analyze_market_conditions(market_data)
                        if ai_result:
                            market_health = ai_result.get("market_health", market_health)
                except Exception as e:
                    log_error(f"Error in market condition guardian module: {e}")
            
            return {
                "market_health": market_health,
                "liquidity_score": min(1.0, market_data.liquidity / 1000000),
                "volume_score": min(1.0, market_data.volume_24h / 1000000),
                "market_trend": "bullish" if market_data.price_change_24h > 0 else "bearish"
            }
            
        except Exception as e:
            log_error(f"Error in market analysis: {e}")
            return {"market_health": "unknown", "liquidity_score": 0.5, "volume_score": 0.5, "market_trend": "neutral"}
    
    async def _analyze_technical_indicators(self, market_data: MarketData) -> Dict[str, Any]:
        """Analyze technical indicators using AI modules"""
        try:
            # Basic technical analysis
            technical_score = 0.5
            
            if market_data.technical_indicators:
                rsi = market_data.technical_indicators.get('rsi', 50)
                if 30 < rsi < 70:
                    technical_score += 0.2
                
                macd = market_data.technical_indicators.get('macd', 0)
                if macd > 0:
                    technical_score += 0.1
                
                # Moving average analysis
                ma_20 = market_data.technical_indicators.get('moving_avg_20', 0)
                if ma_20 > 0 and market_data.price > ma_20:
                    technical_score += 0.2
            
            technical_score = min(1.0, technical_score)
            
            # Enhance with AI module if available
            if "pattern_recognizer" in self.module_connector.modules:
                try:
                    module = self.module_connector.modules["pattern_recognizer"]
                    if hasattr(module, 'analyze_patterns'):
                        ai_result = await module.analyze_patterns(market_data) if asyncio.iscoroutinefunction(module.analyze_patterns) else module.analyze_patterns(market_data)
                        if ai_result:
                            technical_score = ai_result.get("technical_score", technical_score)
                except Exception as e:
                    log_error(f"Error in pattern recognizer module: {e}")
            
            return {
                "technical_score": technical_score,
                "trend": "bullish" if technical_score > 0.6 else "bearish" if technical_score < 0.4 else "neutral",
                "signals": ["rsi_normal", "macd_positive"] if technical_score > 0.6 else []
            }
            
        except Exception as e:
            log_error(f"Error in technical analysis: {e}")
            return {"technical_score": 0.5, "trend": "neutral", "signals": []}
    
    async def _analyze_execution_optimization(self, market_data: MarketData) -> Dict[str, Any]:
        """Analyze execution optimization using AI modules"""
        try:
            # Basic execution analysis
            execution_score = 0.5
            
            # Liquidity-based execution score
            if market_data.liquidity > 500000:
                execution_score += 0.3
            elif market_data.liquidity > 100000:
                execution_score += 0.1
            
            # Volume-based execution score
            if market_data.volume_24h > 1000000:
                execution_score += 0.2
            elif market_data.volume_24h > 500000:
                execution_score += 0.1
            
            execution_score = min(1.0, execution_score)
            
            # Enhance with AI module if available
            if "execution_optimizer" in self.module_connector.modules:
                try:
                    module = self.module_connector.modules["execution_optimizer"]
                    if hasattr(module, 'optimize_execution'):
                        ai_result = await module.optimize_execution(market_data) if asyncio.iscoroutinefunction(module.optimize_execution) else module.optimize_execution(market_data)
                        if ai_result:
                            execution_score = ai_result.get("execution_score", execution_score)
                except Exception as e:
                    log_error(f"Error in execution optimizer module: {e}")
            
            return {
                "execution_score": execution_score,
                "recommended_slippage": max(0.01, 0.1 - execution_score * 0.08),
                "optimal_timing": "immediate" if execution_score > 0.7 else "wait"
            }
            
        except Exception as e:
            log_error(f"Error in execution analysis: {e}")
            return {"execution_score": 0.5, "recommended_slippage": 0.05, "optimal_timing": "wait"}
    
    def _calculate_overall_score(self, sentiment: Dict, prediction: Dict, risk: Dict, 
                                market: Dict, technical: Dict) -> float:
        """Calculate overall AI analysis score"""
        try:
            # Weighted combination of all analysis components
            weights = {
                "sentiment": 0.2,
                "prediction": 0.3,
                "risk": 0.2,
                "market": 0.15,
                "technical": 0.15
            }
            
            sentiment_score = sentiment.get("score", 0.5)
            prediction_score = prediction.get("price_movement_probability", 0.5)
            risk_score = 1.0 - risk.get("risk_score", 0.5)  # Invert risk (lower risk = higher score)
            market_score = market.get("liquidity_score", 0.5) * 0.5 + market.get("volume_score", 0.5) * 0.5
            technical_score = technical.get("technical_score", 0.5)
            
            overall_score = (
                sentiment_score * weights["sentiment"] +
                prediction_score * weights["prediction"] +
                risk_score * weights["risk"] +
                market_score * weights["market"] +
                technical_score * weights["technical"]
            )
            
            return max(0.0, min(1.0, overall_score))
            
        except Exception as e:
            log_error(f"Error calculating overall score: {e}")
            return 0.5
    
    def _calculate_confidence(self, results: List[Any]) -> float:
        """Calculate confidence based on analysis results"""
        try:
            # Count successful analyses
            successful_analyses = sum(1 for result in results if not isinstance(result, Exception) and result)
            total_analyses = len(results)
            
            # Base confidence on success rate
            base_confidence = successful_analyses / total_analyses if total_analyses > 0 else 0.5
            
            # Adjust based on result quality (simplified)
            confidence = base_confidence * 0.8 + 0.2  # Ensure minimum confidence
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            log_error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _generate_recommendations(self, overall_score: float, sentiment: Dict, 
                                 prediction: Dict, risk: Dict, market: Dict, 
                                 technical: Dict) -> Dict[str, Any]:
        """Generate trading recommendations based on analysis"""
        try:
            recommendations = {
                "action": "hold",
                "confidence": 0.5,
                "position_size": 5.0,
                "take_profit": 0.15,
                "stop_loss": 0.08,
                "reasoning": []
            }
            
            # Determine action based on overall score
            if overall_score > 0.8:
                recommendations["action"] = "strong_buy"
                recommendations["confidence"] = 0.9
                recommendations["position_size"] = 20.0
                recommendations["reasoning"].append("High overall score")
            elif overall_score > 0.7:
                recommendations["action"] = "buy"
                recommendations["confidence"] = 0.8
                recommendations["position_size"] = 15.0
                recommendations["reasoning"].append("Good overall score")
            elif overall_score > 0.6:
                recommendations["action"] = "weak_buy"
                recommendations["confidence"] = 0.6
                recommendations["position_size"] = 10.0
                recommendations["reasoning"].append("Moderate overall score")
            elif overall_score < 0.3:
                recommendations["action"] = "avoid"
                recommendations["confidence"] = 0.8
                recommendations["position_size"] = 0.0
                recommendations["reasoning"].append("Low overall score")
            
            # Adjust based on risk
            risk_level = risk.get("risk_level", "medium")
            if risk_level == "high":
                recommendations["position_size"] *= 0.5
                recommendations["stop_loss"] = 0.05
                recommendations["reasoning"].append("High risk detected")
            elif risk_level == "low":
                recommendations["position_size"] *= 1.2
                recommendations["reasoning"].append("Low risk")
            
            # Adjust based on market conditions
            market_health = market.get("market_health", "unknown")
            if market_health == "poor":
                recommendations["position_size"] *= 0.7
                recommendations["reasoning"].append("Poor market conditions")
            elif market_health == "good":
                recommendations["position_size"] *= 1.1
                recommendations["reasoning"].append("Good market conditions")
            
            return recommendations
            
        except Exception as e:
            log_error(f"Error generating recommendations: {e}")
            return {
                "action": "hold",
                "confidence": 0.5,
                "position_size": 5.0,
                "take_profit": 0.15,
                "stop_loss": 0.08,
                "reasoning": ["Analysis error"]
            }

# Global AI integration engine instance
_ai_engine: Optional[AIIntegrationEngine] = None

async def get_ai_engine() -> AIIntegrationEngine:
    """Get global AI integration engine instance"""
    global _ai_engine
    if _ai_engine is None:
        _ai_engine = AIIntegrationEngine()
        await _ai_engine.initialize()
    return _ai_engine

async def analyze_token_ai(token_data: Dict[str, Any]) -> AIAnalysisResult:
    """Analyze token using AI integration engine"""
    engine = await get_ai_engine()
    return await engine.analyze_token(token_data)
