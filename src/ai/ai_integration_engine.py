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
import math
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
    # New comprehensive analysis sections
    market_context: Dict[str, Any] = None  # Regime, cycle, liquidity, anomaly
    predictive_analytics: Dict[str, Any] = None  # Predictive, price, microstructure
    risk_controls: Dict[str, Any] = None  # Drawdown, risk prediction, emergency stop
    portfolio_analysis: Dict[str, Any] = None  # Portfolio optimization, rebalancing
    execution_optimization: Dict[str, Any] = None  # Execution monitor, position validation
    processing_time: float = 0.0

class AIModuleConnector:
    """Connects and manages all AI modules"""
    
    def __init__(self):
        self.modules = {}
        self.module_health = {}
        self.last_health_check = 0
        self.health_check_interval = 60  # 1 minute
        
    async def initialize_modules(self):
        """Initialize all AI modules with individual error handling"""
        self.modules = {}
        failed_modules = []
        
        # Define module specifications
        module_specs = [
            ("sentiment_analyzer", "src.ai.ai_sentiment_analyzer", "AISentimentAnalyzer"),
            ("price_predictor", "src.ai.ai_price_predictor", "AIPricePredictor"),
            ("risk_assessor", "src.ai.ai_risk_assessor", "AIRiskAssessor"),
            ("execution_optimizer", "src.ai.ai_execution_optimizer", "AIExecutionOptimizer"),
            ("microstructure_analyzer", "src.ai.ai_microstructure_analyzer", "AIMarketMicrostructureAnalyzer"),
            ("portfolio_optimizer", "src.ai.ai_portfolio_optimizer", "AIPortfolioOptimizer"),
            ("pattern_recognizer", "src.ai.ai_pattern_recognizer", "AIPatternRecognizer"),
            ("market_intelligence", "src.ai.ai_market_intelligence_aggregator", "AIMarketIntelligenceAggregator"),
            ("predictive_analytics", "src.ai.ai_predictive_analytics_engine", "AIPredictiveAnalyticsEngine"),
            ("strategy_selector", "src.ai.ai_dynamic_strategy_selector", "AIDynamicStrategySelector"),
            ("risk_prediction_prevention", "src.ai.ai_risk_prediction_prevention_system", "AIRiskPredictionPreventionSystem"),
            ("regime_transition_detector", "src.ai.ai_market_regime_transition_detector", "AIMarketRegimeTransitionDetector"),
            ("liquidity_flow_analyzer", "src.ai.ai_liquidity_flow_analyzer", "AILiquidityFlowAnalyzer"),
            ("multi_timeframe_analyzer", "src.ai.ai_multi_timeframe_analysis_engine", "AIMultiTimeframeAnalysisEngine"),
            ("market_cycle_predictor", "src.ai.ai_market_cycle_predictor", "AIMarketCyclePredictor"),
            ("drawdown_protection", "src.ai.ai_drawdown_protection_system", "AIDrawdownProtectionSystem"),
            ("performance_attribution", "src.ai.ai_performance_attribution_analyzer", "AIPerformanceAttributionAnalyzer"),
            ("market_anomaly_detector", "src.ai.ai_market_anomaly_detector", "AIMarketAnomalyDetector"),
            ("portfolio_rebalancing", "src.ai.ai_portfolio_rebalancing_engine", "AIPortfolioRebalancingEngine"),
            ("emergency_stop", "src.ai.ai_emergency_stop_system", "AIEmergencyStopSystem"),
            ("position_size_validator", "src.ai.ai_position_size_validator", "AIPositionSizeValidator"),
            ("trade_execution_monitor", "src.ai.ai_trade_execution_monitor", "AITradeExecutionMonitor"),
            ("market_condition_guardian", "src.ai.ai_market_condition_guardian", "AIMarketConditionGuardian"),
            ("market_regime_detector", "src.ai.ai_market_regime_detector", "AIMarketRegimeDetector"),
        ]
        
        # Initialize each module individually
        for module_name, module_path, class_name in module_specs:
            try:
                module = __import__(module_path, fromlist=[class_name])
                module_class = getattr(module, class_name)
                self.modules[module_name] = module_class()
                log_info(f"ai.init.{module_name}", f"Successfully initialized {module_name}")
            except Exception as e:
                failed_modules.append(module_name)
                log_error(f"ai.init.{module_name}", f"Failed to initialize {module_name}: {e}")
        
        # Log summary
        if failed_modules:
            log_info("ai.initialization", f"Initialized {len(self.modules)}/{len(module_specs)} AI modules. Failed: {', '.join(failed_modules)}")
        else:
            log_info("ai.initialization", f"Successfully initialized all {len(self.modules)} AI modules")
        
        # Return True if at least some modules were initialized
        return len(self.modules) > 0
    
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
                log_error(f"ai.health_check.{module_name}", f"Health check failed for {module_name}: {e}")
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
        """Predict price movement using market data analysis"""
        try:
            # Use feature-based prediction instead of random values
            # Features are normalized market metrics
            
            if len(features) < 10:
                # Insufficient data
                return {
                    "price_movement_probability": 0.5,
                    "confidence": 0.3,
                    "expected_return": 0.0,
                    "risk_score": 0.5
                }
            
            # Extract key features (first 10 are basic metrics)
            price = features[0] if len(features) > 0 else 1.0
            volume = features[1] if len(features) > 1 else 0.0
            market_cap = features[2] if len(features) > 2 else 0.0
            price_change = features[3] if len(features) > 3 else 0.0
            liquidity = features[4] if len(features) > 4 else 0.0
            
            # Calculate prediction score based on real metrics
            prediction_score = 0.5  # Neutral baseline
            confidence = 0.5
            
            # Volume-based prediction
            if volume > 500000:
                prediction_score += 0.15
                confidence += 0.1
            elif volume > 100000:
                prediction_score += 0.05
                confidence += 0.05
            
            # Liquidity-based prediction
            if liquidity > 1000000:
                prediction_score += 0.1
                confidence += 0.1
            elif liquidity > 500000:
                prediction_score += 0.05
                confidence += 0.05
            
            # Price momentum
            if price_change > 0.05:  # 5% positive movement
                prediction_score += 0.1
            elif price_change < -0.05:
                prediction_score -= 0.1
            
            # Market cap consideration
            if market_cap > 10000000:  # $10M+
                confidence += 0.05
            
            # Clamp values
            prediction_score = max(0.0, min(1.0, prediction_score))
            confidence = max(0.3, min(0.95, confidence))
            
            # Calculate expected return based on volume and liquidity
            vol_liq_ratio = volume / liquidity if liquidity > 0 else 0
            expected_return = min(0.25, max(0.05, vol_liq_ratio * 0.5))
            
            # Risk score based on volatility indicators
            risk_score = 0.3  # Base risk
            if volume < 50000 or liquidity < 50000:
                risk_score += 0.3
            if abs(price_change) > 0.2:  # High volatility
                risk_score += 0.2
            risk_score = min(0.9, risk_score)
            
            return {
                "price_movement_probability": prediction_score,
                "confidence": confidence,
                "expected_return": expected_return,
                "risk_score": risk_score
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
        """Analyze sentiment using real market data"""
        try:
            # Base sentiment from news (real data)
            base_sentiment = market_data.news_sentiment
            
            # Social mentions boost (real metric)
            social_boost = min(0.2, market_data.social_mentions / 1000)
            
            # Volume and transaction activity sentiment
            tx_sentiment = 0.0
            if market_data.transactions_24h > 1000:
                tx_sentiment = 0.1
            elif market_data.transactions_24h > 500:
                tx_sentiment = 0.05
            
            # Holder growth sentiment
            holder_sentiment = 0.0
            if market_data.holders > 10000:
                holder_sentiment = 0.1
            elif market_data.holders > 5000:
                holder_sentiment = 0.05
            
            # Combine sentiments
            sentiment_score = base_sentiment + social_boost + tx_sentiment + holder_sentiment
            sentiment_score = max(0, min(1, sentiment_score))  # Clamp to [0, 1]
            
            # Categorize sentiment
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
            
            # Calculate confidence based on data quality
            confidence = 0.6  # Base confidence
            if market_data.social_mentions > 50:
                confidence += 0.1
            if market_data.transactions_24h > 500:
                confidence += 0.1
            if market_data.holders > 1000:
                confidence += 0.1
            confidence = min(0.95, confidence)
            
            # Determine trend based on activity
            trend = "improving" if sentiment_score > 0.5 else "declining"
            if abs(sentiment_score - 0.5) < 0.1:
                trend = "stable"
            
            return {
                "category": category,
                "score": sentiment_score,
                "confidence": confidence,
                "trend": trend
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
        log_info("ai.engine.init", "Initializing AI Integration Engine")
        
        # Initialize cache
        self.cache = get_cache()
        
        # Initialize AI modules
        if not await self.module_connector.initialize_modules():
            log_error("Failed to initialize AI modules")
            return False
        
        # Check module health
        health_status = await self.module_connector.check_module_health()
        healthy_modules = sum(health_status.values())
        log_info("ai.engine_initialization", f"AI Integration Engine initialized with {healthy_modules}/{len(health_status)} healthy modules")
        
        return True
    
    async def analyze_token(
        self,
        token_data: Dict[str, Any],
        regime_data: Optional[Dict[str, Any]] = None
    ) -> AIAnalysisResult:
        """Perform comprehensive AI analysis on a token with all modules"""
        symbol = token_data.get("symbol", "UNKNOWN")
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"ai_analysis_{symbol}_{hash(str(token_data))}"
            cached_result = cache_get(cache_key)
            if cached_result:
                log_info("ai.cache", f"Using cached AI analysis for {symbol}")
                return AIAnalysisResult(**cached_result)
            
            # Convert token data to MarketData
            market_data = self._convert_to_market_data(token_data)
            
            # Prepare features for ML pipeline
            features = await self.ml_pipeline.prepare_features(market_data)
            
            # Get trade amount for modules that need it
            trade_amount = float(token_data.get("position_size_usd", token_data.get("recommended_position_size", 10.0)))
            
            # Stage 1: Core Analysis (existing modules)
            core_tasks = [
                self._analyze_sentiment(market_data),
                self._analyze_price_prediction(market_data, features),
                self._analyze_risk(market_data),
                self._analyze_market_conditions(market_data),
                self._analyze_technical_indicators(market_data),
                self._analyze_execution_optimization(market_data)
            ]
            
            # Stage 2: Market Context Analysis (regime, cycle, liquidity, anomaly)
            market_context_tasks = [
                self._analyze_market_context(
                    token_data,
                    market_data,
                    trade_amount,
                    precomputed_regime=regime_data
                )
            ]
            
            # Stage 3: Predictive Analytics (predictive, price, microstructure)
            predictive_tasks = [
                self._analyze_predictive_analytics(token_data, market_data, trade_amount)
            ]
            
            # Stage 4: Risk Controls (drawdown, risk prediction, emergency stop)
            risk_control_tasks = [
                self._analyze_risk_controls(token_data, market_data, trade_amount)
            ]
            
            # Stage 5: Portfolio Analysis (optimization, rebalancing)
            portfolio_tasks = [
                self._analyze_portfolio(token_data, market_data, trade_amount)
            ]
            
            # Stage 6: Execution Optimization (monitor, position validation)
            execution_opt_tasks = [
                self._analyze_execution_optimization_full(token_data, market_data, trade_amount)
            ]
            
            # Run all analysis stages in parallel
            all_tasks = core_tasks + market_context_tasks + predictive_tasks + risk_control_tasks + portfolio_tasks + execution_opt_tasks
            results = await asyncio.gather(*all_tasks, return_exceptions=True)
            
            # Process core results
            sentiment_analysis = results[0] if not isinstance(results[0], Exception) else {}
            price_prediction = results[1] if not isinstance(results[1], Exception) else {}
            risk_assessment = results[2] if not isinstance(results[2], Exception) else {}
            market_analysis = results[3] if not isinstance(results[3], Exception) else {}
            technical_analysis = results[4] if not isinstance(results[4], Exception) else {}
            execution_analysis = results[5] if not isinstance(results[5], Exception) else {}
            
            # Process new stage results
            market_context = results[6] if not isinstance(results[6], Exception) else {}
            predictive_analytics = results[7] if not isinstance(results[7], Exception) else {}
            risk_controls = results[8] if not isinstance(results[8], Exception) else {}
            portfolio_analysis = results[9] if not isinstance(results[9], Exception) else {}
            execution_optimization = results[10] if not isinstance(results[10], Exception) else {}
            
            # Calculate overall score with all modules (pass market_data for Fix 1)
            overall_score = self._calculate_overall_score_comprehensive(
                sentiment_analysis, price_prediction, risk_assessment, 
                market_analysis, technical_analysis, market_context,
                predictive_analytics, risk_controls, portfolio_analysis,
                market_data=market_data, token_data=token_data
            )
            
            # Generate recommendations with all modules
            recommendations = self._generate_recommendations_comprehensive(
                overall_score, sentiment_analysis, price_prediction, 
                risk_assessment, market_analysis, technical_analysis,
                market_context, predictive_analytics, risk_controls, portfolio_analysis
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
                market_context=market_context,
                predictive_analytics=predictive_analytics,
                risk_controls=risk_controls,
                portfolio_analysis=portfolio_analysis,
                execution_optimization=execution_optimization,
                processing_time=time.time() - start_time
            )
            
            # Cache the result
            cache_set(cache_key, asdict(result), self.cache_ttl)
            
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
                market_context={},
                predictive_analytics={},
                risk_controls={},
                portfolio_analysis={},
                execution_optimization={},
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
            
            # Use logarithmic scaling for better differentiation (Fix 4)
            # Log scale: $10k = 0.3, $100k = 0.5, $1M = 0.8, $10M = 1.0
            
            # Liquidity score with logarithmic scaling
            if market_data.liquidity > 0:
                liquidity_score = min(1.0, 0.3 + (math.log10(max(market_data.liquidity, 10000) / 10000) * 0.2))
            else:
                liquidity_score = 0.1  # Very low, not 0.5 default
            
            # Volume score with logarithmic scaling
            if market_data.volume_24h > 0:
                volume_score = min(1.0, 0.3 + (math.log10(max(market_data.volume_24h, 10000) / 10000) * 0.2))
            else:
                volume_score = 0.1  # Very low, not 0.5 default
            
            return {
                "market_health": market_health,
                "liquidity_score": liquidity_score,
                "volume_score": volume_score,
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
    
    async def _analyze_market_context(
        self,
        token_data: Dict,
        market_data: MarketData,
        trade_amount: float,
        precomputed_regime: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze market context: regime, cycle, liquidity, anomaly"""
        try:
            context = {}
            
            # Market regime detection
            if precomputed_regime:
                context["regime"] = precomputed_regime
            elif "market_regime_detector" in self.module_connector.modules:
                try:
                    module = self.module_connector.modules["market_regime_detector"]
                    if hasattr(module, 'detect_market_regime'):
                        regime_result = module.detect_market_regime() if not asyncio.iscoroutinefunction(module.detect_market_regime) else await module.detect_market_regime()
                        context["regime"] = regime_result
                except Exception as e:
                    log_error(f"Error in market regime detector: {e}")
            
            # Market cycle prediction
            if "market_cycle_predictor" in self.module_connector.modules:
                try:
                    module = self.module_connector.modules["market_cycle_predictor"]
                    if hasattr(module, 'predict_market_cycle'):
                        cycle_result = module.predict_market_cycle(token_data, trade_amount) if not asyncio.iscoroutinefunction(module.predict_market_cycle) else await module.predict_market_cycle(token_data, trade_amount)
                        context["cycle"] = cycle_result
                except Exception as e:
                    log_error(f"Error in market cycle predictor: {e}")
            
            # Liquidity flow analysis
            if "liquidity_flow_analyzer" in self.module_connector.modules:
                try:
                    module = self.module_connector.modules["liquidity_flow_analyzer"]
                    if hasattr(module, 'analyze_liquidity_flow'):
                        liquidity_result = module.analyze_liquidity_flow(token_data, trade_amount) if not asyncio.iscoroutinefunction(module.analyze_liquidity_flow) else await module.analyze_liquidity_flow(token_data, trade_amount)
                        context["liquidity"] = liquidity_result
                except Exception as e:
                    log_error(f"Error in liquidity flow analyzer: {e}")
            
            # Market anomaly detection
            if "market_anomaly_detector" in self.module_connector.modules:
                try:
                    module = self.module_connector.modules["market_anomaly_detector"]
                    if hasattr(module, 'detect_market_anomalies'):
                        anomaly_result = module.detect_market_anomalies(token_data, {"timestamp": market_data.timestamp}, {}) if not asyncio.iscoroutinefunction(module.detect_market_anomalies) else await module.detect_market_anomalies(token_data, {"timestamp": market_data.timestamp}, {})
                        context["anomaly"] = anomaly_result
                except Exception as e:
                    log_error(f"Error in market anomaly detector: {e}")
            
            # Regime transition detection
            if "regime_transition_detector" in self.module_connector.modules:
                try:
                    module = self.module_connector.modules["regime_transition_detector"]
                    if hasattr(module, 'detect_regime_transition'):
                        transition_result = module.detect_regime_transition(token_data, trade_amount) if not asyncio.iscoroutinefunction(module.detect_regime_transition) else await module.detect_regime_transition(token_data, trade_amount)
                        context["regime_transition"] = transition_result
                except Exception as e:
                    log_error(f"Error in regime transition detector: {e}")
            
            return context
            
        except Exception as e:
            log_error(f"Error in market context analysis: {e}")
            return {}
    
    async def _analyze_predictive_analytics(self, token_data: Dict, market_data: MarketData, trade_amount: float) -> Dict[str, Any]:
        """Analyze predictive analytics: predictive engine, price predictor, microstructure"""
        try:
            analytics = {}
            
            # Predictive analytics engine
            if "predictive_analytics" in self.module_connector.modules:
                try:
                    module = self.module_connector.modules["predictive_analytics"]
                    if hasattr(module, 'predict_price_movement'):
                        pred_result = module.predict_price_movement(token_data, trade_amount) if not asyncio.iscoroutinefunction(module.predict_price_movement) else await module.predict_price_movement(token_data, trade_amount)
                        analytics["predictive"] = pred_result
                except Exception as e:
                    log_error(f"Error in predictive analytics engine: {e}")
            
            # Market microstructure analysis
            if "microstructure_analyzer" in self.module_connector.modules:
                try:
                    module = self.module_connector.modules["microstructure_analyzer"]
                    if hasattr(module, 'analyze_market_microstructure'):
                        micro_result = module.analyze_market_microstructure(token_data, trade_amount) if not asyncio.iscoroutinefunction(module.analyze_market_microstructure) else await module.analyze_market_microstructure(token_data, trade_amount)
                        analytics["microstructure"] = micro_result
                except Exception as e:
                    log_error(f"Error in microstructure analyzer: {e}")
            
            # Multi-timeframe analysis
            if "multi_timeframe_analyzer" in self.module_connector.modules:
                try:
                    module = self.module_connector.modules["multi_timeframe_analyzer"]
                    if hasattr(module, 'analyze_multi_timeframe'):
                        timeframe_result = module.analyze_multi_timeframe(token_data, trade_amount) if not asyncio.iscoroutinefunction(module.analyze_multi_timeframe) else await module.analyze_multi_timeframe(token_data, trade_amount)
                        analytics["multi_timeframe"] = timeframe_result
                except Exception as e:
                    log_error(f"Error in multi-timeframe analyzer: {e}")
            
            return analytics
            
        except Exception as e:
            log_error(f"Error in predictive analytics: {e}")
            return {}
    
    async def _analyze_risk_controls(self, token_data: Dict, market_data: MarketData, trade_amount: float) -> Dict[str, Any]:
        """Analyze risk controls: drawdown protection, risk prediction prevention, emergency stop"""
        try:
            controls = {}
            
            # Drawdown protection
            if "drawdown_protection" in self.module_connector.modules:
                try:
                    module = self.module_connector.modules["drawdown_protection"]
                    if hasattr(module, 'analyze_drawdown_protection'):
                        portfolio_data = {"total_value": trade_amount * 10, "initial_value": trade_amount * 10}
                        drawdown_result = module.analyze_drawdown_protection(portfolio_data, [], {}) if not asyncio.iscoroutinefunction(module.analyze_drawdown_protection) else await module.analyze_drawdown_protection(portfolio_data, [], {})
                        controls["drawdown"] = drawdown_result
                except Exception as e:
                    log_error(f"Error in drawdown protection: {e}")
            
            # Risk prediction prevention
            if "risk_prediction_prevention" in self.module_connector.modules:
                try:
                    module = self.module_connector.modules["risk_prediction_prevention"]
                    if hasattr(module, 'predict_risk'):
                        risk_pred_result = module.predict_risk(token_data, trade_amount) if not asyncio.iscoroutinefunction(module.predict_risk) else await module.predict_risk(token_data, trade_amount)
                        controls["risk_prediction"] = risk_pred_result
                except Exception as e:
                    log_error(f"Error in risk prediction prevention: {e}")
            
            # Emergency stop system
            if "emergency_stop" in self.module_connector.modules:
                try:
                    module = self.module_connector.modules["emergency_stop"]
                    if hasattr(module, 'check_emergency_conditions'):
                        portfolio_data = {"total_value": trade_amount * 10, "initial_value": trade_amount * 10, "timestamp": market_data.timestamp}
                        emergency_result = module.check_emergency_conditions(portfolio_data, [], {}, []) if not asyncio.iscoroutinefunction(module.check_emergency_conditions) else await module.check_emergency_conditions(portfolio_data, [], {}, [])
                        controls["emergency"] = emergency_result
                except Exception as e:
                    log_error(f"Error in emergency stop system: {e}")
            
            return controls
            
        except Exception as e:
            log_error(f"Error in risk controls analysis: {e}")
            return {}
    
    async def _analyze_portfolio(self, token_data: Dict, market_data: MarketData, trade_amount: float) -> Dict[str, Any]:
        """Analyze portfolio: optimization, rebalancing"""
        try:
            portfolio = {}
            
            # Portfolio optimizer
            if "portfolio_optimizer" in self.module_connector.modules:
                try:
                    module = self.module_connector.modules["portfolio_optimizer"]
                    if hasattr(module, 'optimize_portfolio'):
                        positions = [token_data] if token_data else []
                        opt_result = module.optimize_portfolio(positions, trade_amount * 10) if not asyncio.iscoroutinefunction(module.optimize_portfolio) else await module.optimize_portfolio(positions, trade_amount * 10)
                        portfolio["optimization"] = opt_result
                except Exception as e:
                    log_error(f"Error in portfolio optimizer: {e}")
            
            # Portfolio rebalancing
            if "portfolio_rebalancing" in self.module_connector.modules:
                try:
                    module = self.module_connector.modules["portfolio_rebalancing"]
                    if hasattr(module, 'optimize_portfolio_allocation'):
                        positions = [token_data] if token_data else []
                        rebalance_result = module.optimize_portfolio_allocation(positions, {}) if not asyncio.iscoroutinefunction(module.optimize_portfolio_allocation) else await module.optimize_portfolio_allocation(positions, {})
                        portfolio["rebalancing"] = rebalance_result
                except Exception as e:
                    log_error(f"Error in portfolio rebalancing: {e}")
            
            return portfolio
            
        except Exception as e:
            log_error(f"Error in portfolio analysis: {e}")
            return {}
    
    async def _analyze_execution_optimization_full(self, token_data: Dict, market_data: MarketData, trade_amount: float) -> Dict[str, Any]:
        """Analyze execution optimization: monitor, position validation"""
        try:
            optimization = {}
            
            # Trade execution monitor
            if "trade_execution_monitor" in self.module_connector.modules:
                try:
                    module = self.module_connector.modules["trade_execution_monitor"]
                    if hasattr(module, 'monitor_trade_execution'):
                        trade_data = {"symbol": token_data.get("symbol"), "amount": trade_amount}
                        market_conditions = {}  # Empty dict for market conditions
                        monitor_result = module.monitor_trade_execution(trade_data, [], market_conditions) if not asyncio.iscoroutinefunction(module.monitor_trade_execution) else await module.monitor_trade_execution(trade_data, [], market_conditions)
                        optimization["monitor"] = monitor_result
                except Exception as e:
                    log_error(f"Error in trade execution monitor: {e}")
            
            # Position size validator
            if "position_size_validator" in self.module_connector.modules:
                try:
                    module = self.module_connector.modules["position_size_validator"]
                    if hasattr(module, 'validate_position_size'):
                        wallet_balance = trade_amount * 20  # Estimate
                        validation_result = module.validate_position_size(token_data, trade_amount, wallet_balance, [], {}) if not asyncio.iscoroutinefunction(module.validate_position_size) else await module.validate_position_size(token_data, trade_amount, wallet_balance, [], {})
                        optimization["position_validation"] = validation_result
                except Exception as e:
                    log_error(f"Error in position size validator: {e}")
            
            return optimization
            
        except Exception as e:
            log_error(f"Error in execution optimization: {e}")
            return {}
    
    def _calculate_overall_score(self, sentiment: Dict, prediction: Dict, risk: Dict, 
                                market: Dict, technical: Dict, market_data: Optional[MarketData] = None) -> float:
        """Calculate overall AI analysis score with actual token metrics (Fix 1)"""
        try:
            # Weighted combination of all analysis components
            weights = {
                "sentiment": 0.2,
                "prediction": 0.3,
                "risk": 0.2,
                "market": 0.15,
                "technical": 0.15
            }
            
            # Fix 1: Use actual token metrics instead of defaults
            # Sentiment: Use actual sentiment or calculate from volume/price action
            sentiment_score = sentiment.get("score", 0.5)
            if sentiment_score == 0.5 and market_data:  # Likely defaulted
                # Calculate from price momentum
                price_change = getattr(market_data, 'price_change_24h', 0)
                sentiment_score = 0.5 + (price_change * 2)  # Scale price change to sentiment
                sentiment_score = max(0.0, min(1.0, sentiment_score))
            
            # Prediction: Use actual prediction or calculate from volume/liquidity
            prediction_score = prediction.get("price_movement_probability", 0.5)
            if prediction_score == 0.5 and market_data:  # Likely defaulted
                # Calculate from volume/liquidity ratio
                vol_liq_ratio = market_data.volume_24h / max(market_data.liquidity, 1)
                # Higher ratio = more trading activity = better prediction
                prediction_score = min(0.9, max(0.1, 0.5 + (vol_liq_ratio * 0.3)))
            
            # Risk: Calculate from actual metrics (aligned with config thresholds)
            risk_raw = risk.get("risk_score", 0.5)
            if risk_raw == 0.5 and market_data:  # Likely defaulted
                # Calculate risk from volume and liquidity (aligned with config.yaml trading thresholds)
                # Config: min_volume_24h_for_buy: 100000, min_liquidity_usd_for_buy: 100000
                if market_data.volume_24h < 100000 or market_data.liquidity < 100000:
                    risk_raw = 0.7  # High risk - below trading thresholds
                elif market_data.volume_24h > 2000000 and market_data.liquidity > 2000000:
                    risk_raw = 0.2  # Low risk - exceptional quality
                else:
                    risk_raw = 0.5  # Medium risk - meets trading thresholds
            risk_score = 1.0 - risk_raw  # Invert (lower risk = higher score)
            
            # Market: Use actual calculated scores, don't default
            liquidity_score = market.get("liquidity_score")
            volume_score = market.get("volume_score")
            if (liquidity_score is None or liquidity_score == 0.5) and market_data:
                # Calculate from actual liquidity
                liquidity_score = min(1.0, market_data.liquidity / 1000000)  # Normalize to $1M
            if (volume_score is None or volume_score == 0.5) and market_data:
                # Calculate from actual volume
                volume_score = min(1.0, market_data.volume_24h / 1000000)  # Normalize to $1M
            # Fallback to defaults if market_data not available
            if liquidity_score is None:
                liquidity_score = 0.5
            if volume_score is None:
                volume_score = 0.5
            market_score = (liquidity_score * 0.5) + (volume_score * 0.5)
            
            # Technical: Use actual technical or calculate from price action
            technical_score = technical.get("technical_score", 0.5)
            if technical_score == 0.5 and market_data:  # Likely defaulted
                # Calculate from price momentum and volatility
                price_change = abs(getattr(market_data, 'price_change_24h', 0))
                # Moderate positive momentum = good technical
                if 0.02 < price_change < 0.15:  # 2-15% movement
                    technical_score = 0.6
                elif price_change > 0.15:  # Too volatile
                    technical_score = 0.4
                else:
                    technical_score = 0.5
            
            overall_score = (
                sentiment_score * weights["sentiment"] +
                prediction_score * weights["prediction"] +
                risk_score * weights["risk"] +
                market_score * weights["market"] +
                technical_score * weights["technical"]
            )
            
            # Fix 2: Expand score range with dynamic scaling
            component_scores = [
                sentiment_score,
                prediction_score,
                risk_score,
                market_score,
                technical_score
            ]
            
            # Calculate variance to determine if scores are clustered
            score_variance = sum((s - overall_score) ** 2 for s in component_scores) / len(component_scores)
            
            # If scores are too clustered (low variance), apply expansion
            if score_variance < 0.01:  # Scores are very similar
                # Expand based on best component
                max_component = max(component_scores)
                min_component = min(component_scores)
                
                # Shift score toward the best component
                if max_component > 0.6:
                    # Good components exist, boost score
                    overall_score = overall_score + (max_component - overall_score) * 0.3
                elif min_component < 0.4:
                    # Poor components exist, reduce score
                    overall_score = overall_score - (overall_score - min_component) * 0.3
            
            # Fix 5: Add minimum quality threshold enforcement (aligned with config.yaml)
            if market_data:
                # Align with config.yaml trading thresholds
                # Config: min_volume_24h_for_buy: 100000, min_liquidity_usd_for_buy: 100000
                min_volume = 100000  # $100k - matches min_volume_24h_for_buy
                min_liquidity = 100000  # $100k - matches min_liquidity_usd_for_buy
                
                if market_data.volume_24h < min_volume or market_data.liquidity < min_liquidity:
                    # Penalize score for tokens below trading thresholds
                    quality_penalty = 0.2
                    overall_score = overall_score - quality_penalty
                    overall_score = max(0.0, overall_score)  # Don't go below 0
                
                # Bonus for exceptional quality tokens (top tier - above typical trades)
                # Based on actual trade data: trades are $2.6M-$9.9M, so $2M+ is top tier
                if market_data.volume_24h > 2000000 and market_data.liquidity > 2000000:  # $2M+ both
                    quality_bonus = 0.1
                    overall_score = overall_score + quality_bonus
                    overall_score = min(1.0, overall_score)  # Don't go above 1
            
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
    
    def _apply_performance_calibration(self, overall_score: float, token_data: Dict) -> float:
        """
        Apply performance-based calibration to quality score (Fix 3).
        Adjusts score based on historical performance of similar tokens.
        """
        try:
            # Load historical performance data
            from src.storage.performance import load_performance_data
            perf_data = load_performance_data()
            
            if not perf_data or not perf_data.get('trades'):
                return overall_score  # No historical data, return as-is
            
            # Find similar tokens (similar volume/liquidity)
            volume = float(token_data.get('volume24h', 0))
            liquidity = float(token_data.get('liquidity', 0))
            
            similar_trades = []
            for trade in perf_data['trades']:
                trade_vol = float(trade.get('volume_24h', 0))
                trade_liq = float(trade.get('liquidity', 0))
                
                # Similar if within 50% of volume/liquidity
                if volume > 0 and liquidity > 0:
                    vol_ratio = trade_vol / volume
                    liq_ratio = trade_liq / liquidity
                    if (0.5 <= vol_ratio <= 2.0 and 0.5 <= liq_ratio <= 2.0):
                        similar_trades.append(trade)
            
            if len(similar_trades) >= 3:  # Need at least 3 similar trades
                # Calculate average PnL for similar trades
                pnls = [t.get('pnl_percent', 0) for t in similar_trades if t.get('pnl_percent') is not None]
                if pnls:
                    avg_pnl = sum(pnls) / len(pnls)
                    
                    # Adjust score based on historical performance
                    # Positive PnL = boost score, negative PnL = reduce score
                    pnl_adjustment = (avg_pnl / 100) * 0.2  # Scale PnL to score adjustment
                    overall_score = overall_score + pnl_adjustment
                    overall_score = max(0.0, min(1.0, overall_score))
            
            return overall_score
            
        except Exception as e:
            log_error(f"Error in performance calibration: {e}")
            return overall_score
    
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
    
    def _calculate_overall_score_comprehensive(self, sentiment: Dict, prediction: Dict, risk: Dict,
                                             market: Dict, technical: Dict, market_context: Dict,
                                             predictive_analytics: Dict, risk_controls: Dict,
                                             portfolio_analysis: Dict, market_data: Optional[MarketData] = None,
                                             token_data: Optional[Dict] = None) -> float:
        """Calculate overall AI analysis score with all modules"""
        try:
            # Start with base score from core analysis (pass market_data for Fix 1)
            base_score = self._calculate_overall_score(sentiment, prediction, risk, market, technical, market_data)
            
            # Extract scores from new modules
            context_score = 0.5
            if market_context:
                # Regime impact
                regime = market_context.get("regime", {})
                if isinstance(regime, dict):
                    regime_type = regime.get("regime", "sideways_market")
                    if regime_type == "bull_market":
                        context_score += 0.1
                    elif regime_type == "bear_market":
                        context_score -= 0.15
                    elif regime_type == "high_volatility":
                        context_score -= 0.1
                
                # Liquidity impact
                liquidity = market_context.get("liquidity", {})
                if isinstance(liquidity, dict):
                    flow_score = liquidity.get("liquidity_flow_score", 0.5)
                    context_score = (context_score + flow_score) / 2
                
                # Anomaly impact
                anomaly = market_context.get("anomaly", {})
                if isinstance(anomaly, dict):
                    anomaly_score = anomaly.get("anomaly_score", 0.5)
                    if anomaly_score > 0.7:  # High anomaly = risk
                        context_score -= 0.1
            
            # Predictive analytics impact
            pred_score = 0.5
            if predictive_analytics:
                pred = predictive_analytics.get("predictive", {})
                if isinstance(pred, dict):
                    pred_score = pred.get("success_probability", 0.5)
                
                micro = predictive_analytics.get("microstructure", {})
                if isinstance(micro, dict):
                    micro_score = micro.get("microstructure_score", 0.5)
                    pred_score = (pred_score + micro_score) / 2
            
            # Risk controls impact (inverse - lower risk = higher score)
            risk_control_score = 0.5
            if risk_controls:
                drawdown = risk_controls.get("drawdown", {})
                if isinstance(drawdown, dict):
                    drawdown_risk = drawdown.get("drawdown_risk_score", 0.5)
                    risk_control_score = 1.0 - drawdown_risk
                
                emergency = risk_controls.get("emergency", {})
                if isinstance(emergency, dict):
                    emergency_level = emergency.get("emergency_level", "normal")
                    if emergency_level in ["urgent", "critical", "emergency"]:
                        risk_control_score -= 0.3
            
            # Portfolio analysis impact
            portfolio_score = 0.5
            if portfolio_analysis:
                opt = portfolio_analysis.get("optimization", {})
                if isinstance(opt, dict):
                    metrics = opt.get("portfolio_metrics", {})
                    if isinstance(metrics, dict):
                        sharpe = metrics.get("sharpe_ratio", 0.0)
                        portfolio_score = min(1.0, max(0.0, (sharpe + 1.0) / 2.0))
            
            # Weighted combination
            weights = {
                "base": 0.4,
                "context": 0.15,
                "predictive": 0.2,
                "risk_control": 0.15,
                "portfolio": 0.1
            }
            
            overall_score = (
                base_score * weights["base"] +
                context_score * weights["context"] +
                pred_score * weights["predictive"] +
                risk_control_score * weights["risk_control"] +
                portfolio_score * weights["portfolio"]
            )
            
            overall_score = max(0.0, min(1.0, overall_score))
            
            # Fix 3: Apply performance-based calibration
            if token_data:
                overall_score = self._apply_performance_calibration(overall_score, token_data)
            
            return max(0.0, min(1.0, overall_score))
            
        except Exception as e:
            log_error(f"Error calculating comprehensive overall score: {e}")
            return self._calculate_overall_score(sentiment, prediction, risk, market, technical, market_data)
    
    def _generate_recommendations_comprehensive(self, overall_score: float, sentiment: Dict,
                                              prediction: Dict, risk: Dict, market: Dict,
                                              technical: Dict, market_context: Dict,
                                              predictive_analytics: Dict, risk_controls: Dict,
                                              portfolio_analysis: Dict) -> Dict[str, Any]:
        """Generate comprehensive trading recommendations with all modules"""
        try:
            # Start with base recommendations
            recommendations = self._generate_recommendations(
                overall_score, sentiment, prediction, risk, market, technical
            )
            
            # Adjust based on market context
            if market_context:
                regime = market_context.get("regime", {})
                if isinstance(regime, dict):
                    regime_type = regime.get("regime", "sideways_market")
                    position_multiplier = regime.get("position_multiplier", 1.0)
                    recommendations["position_size"] *= position_multiplier
                    recommendations["reasoning"].append(f"Market regime: {regime_type}")
                
                liquidity = market_context.get("liquidity", {})
                if isinstance(liquidity, dict):
                    flow_type = liquidity.get("liquidity_flow_type", "stable")
                    if flow_type == "trap":
                        recommendations["action"] = "avoid"
                        recommendations["reasoning"].append("Liquidity trap detected")
                    elif flow_type == "increasing":
                        recommendations["position_size"] *= 1.1
                        recommendations["reasoning"].append("Increasing liquidity")
                
                anomaly = market_context.get("anomaly", {})
                if isinstance(anomaly, dict):
                    severity = anomaly.get("anomaly_severity", "minor")
                    if severity in ["major", "extreme"]:
                        recommendations["action"] = "avoid"
                        recommendations["reasoning"].append(f"Major anomaly detected: {severity}")
            
            # Adjust based on risk controls
            if risk_controls:
                emergency = risk_controls.get("emergency", {})
                if isinstance(emergency, dict):
                    emergency_level = emergency.get("emergency_level", "normal")
                    if emergency_level in ["urgent", "critical", "emergency"]:
                        recommendations["action"] = "avoid"
                        recommendations["position_size"] = 0.0
                        recommendations["reasoning"].append(f"Emergency stop: {emergency_level}")
                
                drawdown = risk_controls.get("drawdown", {})
                if isinstance(drawdown, dict):
                    drawdown_risk = drawdown.get("drawdown_risk_score", 0.5)
                    if drawdown_risk > 0.7:
                        recommendations["position_size"] *= 0.5
                        recommendations["stop_loss"] = 0.05
                        recommendations["reasoning"].append("High drawdown risk")
            
            # Adjust based on portfolio analysis
            if portfolio_analysis:
                opt = portfolio_analysis.get("optimization", {})
                if isinstance(opt, dict):
                    allocation = opt.get("optimized_allocation", {})
                    if isinstance(allocation, dict):
                        # Use portfolio-optimized position size if available
                        recommended_actions = opt.get("recommended_actions", [])
                        if recommended_actions:
                            recommendations["reasoning"].extend(recommended_actions[:2])
            
            # Adjust based on execution optimization
            if predictive_analytics:
                micro = predictive_analytics.get("microstructure", {})
                if isinstance(micro, dict):
                    exec_recs = micro.get("execution_recommendations", {}) or {}
                    timing_data = micro.get("optimal_timing", {}) or {}

                    optimal_timing = "wait"
                    if isinstance(timing_data, dict):
                        optimal_timing = timing_data.get("optimal_timing", optimal_timing)
                    if isinstance(exec_recs, dict):
                        optimal_timing = exec_recs.get("optimal_timing", optimal_timing)

                    execution_recommendation = exec_recs.get("execution_recommendation") if isinstance(exec_recs, dict) else None

                    # Normalize values for easier comparisons
                    if isinstance(optimal_timing, str):
                        optimal_timing = optimal_timing.lower()
                    if isinstance(execution_recommendation, str):
                        execution_recommendation = execution_recommendation.lower()

                    if optimal_timing in {"avoid", "avoid_execution"} or execution_recommendation in {"avoid_execution"}:
                        recommendations["action"] = "avoid"
                        recommendations["position_size"] = 0.0
                        recommendations["reasoning"].append("Microstructure timing: avoid execution")
                    elif optimal_timing in {"wait"} or execution_recommendation in {"execute_cautious"}:
                        recommendations["action"] = "hold"
                        recommendations["reasoning"].append("Microstructure timing suggests waiting")
                    elif execution_recommendation in {"execute_immediately", "execute_optimal"}:
                        # Reinforce buy signal if microstructure strongly supports execution
                        if recommendations["action"] in {"hold", "weak_buy"}:
                            recommendations["action"] = "buy"
                        recommendations["confidence"] = max(recommendations.get("confidence", 0.5), 0.75)
                        recommendations["reasoning"].append("Microstructure favors execution now")
            
            return recommendations
            
        except Exception as e:
            log_error(f"Error generating comprehensive recommendations: {e}")
            return self._generate_recommendations(overall_score, sentiment, prediction, risk, market, technical)

# Global AI integration engine instance
_ai_engine: Optional[AIIntegrationEngine] = None

async def get_ai_engine() -> AIIntegrationEngine:
    """Get global AI integration engine instance"""
    global _ai_engine
    if _ai_engine is None:
        _ai_engine = AIIntegrationEngine()
        await _ai_engine.initialize()
    return _ai_engine

async def analyze_token_ai(
    token_data: Dict[str, Any],
    regime_data: Optional[Dict[str, Any]] = None
) -> AIAnalysisResult:
    """Analyze token using AI integration engine"""
    engine = await get_ai_engine()
    return await engine.analyze_token(token_data, regime_data=regime_data)
