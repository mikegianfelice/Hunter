#!/usr/bin/env python3
"""
Structured Logging System for Trading Bot
Provides comprehensive logging with performance metrics and structured data
"""

import json
import logging
import time
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import defaultdict, deque

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: str
    level: str
    event: str
    message: str
    context: Dict[str, Any]
    performance_metrics: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None

class StructuredLogger:
    """
    Advanced structured logging system with performance tracking
    """
    
    def __init__(self, log_dir: str = "logs", max_file_size: int = 10 * 1024 * 1024):  # 10MB
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.max_file_size = max_file_size
        
        # Log files
        self.main_log = self.log_dir / "trading.log"
        self.performance_log = self.log_dir / "performance.log"
        self.error_log = self.log_dir / "errors.log"
        self.trade_log = self.log_dir / "trades.log"
        
        # Performance tracking
        self.performance_metrics = defaultdict(list)
        self.session_metrics = defaultdict(dict)
        self.current_session_id = None
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Setup log rotation
        self._setup_log_rotation()
    
    def _setup_log_rotation(self):
        """Setup log file rotation based on size"""
        for log_file in [self.main_log, self.performance_log, self.error_log, self.trade_log]:
            if log_file.exists() and log_file.stat().st_size > self.max_file_size:
                self._rotate_log_file(log_file)
    
    def _rotate_log_file(self, log_file: Path):
        """Rotate log file when it gets too large"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = log_file.parent / f"{log_file.stem}_{timestamp}.log"
            log_file.rename(backup_file)
        except Exception as e:
            print(f"Error rotating log file {log_file}: {e}")
    
    def start_session(self, session_id: str = None):
        """Start a new logging session"""
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        
        with self.lock:
            self.current_session_id = session_id
            self.session_metrics[session_id] = {
                'start_time': datetime.now().isoformat(),
                'log_count': 0,
                'error_count': 0,
                'trade_count': 0
            }
    
    def end_session(self):
        """End current logging session"""
        with self.lock:
            if self.current_session_id and self.current_session_id in self.session_metrics:
                self.session_metrics[self.current_session_id]['end_time'] = datetime.now().isoformat()
                self.current_session_id = None
    
    def _write_log_entry(self, entry: LogEntry, log_file: Path):
        """Write log entry to file"""
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(asdict(entry)) + '\n')
        except Exception as e:
            print(f"Error writing to log file {log_file}: {e}")
    
    def _create_log_entry(self, level: LogLevel, event: str, message: str, 
                         context: Dict[str, Any] = None, 
                         performance_metrics: Dict[str, Any] = None) -> LogEntry:
        """Create a structured log entry"""
        return LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level.value,
            event=event,
            message=message,
            context=context or {},
            performance_metrics=performance_metrics,
            session_id=self.current_session_id,
            trace_id=self._generate_trace_id()
        )
    
    def _generate_trace_id(self) -> str:
        """Generate a unique trace ID for request tracking"""
        return f"trace_{int(time.time() * 1000)}_{id(threading.current_thread())}"
    
    def _update_session_metrics(self, level: LogLevel, event: str):
        """Update session metrics"""
        if self.current_session_id and self.current_session_id in self.session_metrics:
            self.session_metrics[self.current_session_id]['log_count'] += 1
            if level in [LogLevel.ERROR, LogLevel.CRITICAL]:
                self.session_metrics[self.current_session_id]['error_count'] += 1
            if event.startswith('trade.'):
                self.session_metrics[self.current_session_id]['trade_count'] += 1
    
    def log(self, level: LogLevel, event: str, message: str, 
            context: Dict[str, Any] = None, 
            performance_metrics: Dict[str, Any] = None):
        """Log a message with structured data"""
        with self.lock:
            entry = self._create_log_entry(level, event, message, context, performance_metrics)
            
            # Write to main log
            self._write_log_entry(entry, self.main_log)
            
            # Write to specific logs based on event type
            if event.startswith('trade.'):
                self._write_log_entry(entry, self.trade_log)
            elif level in [LogLevel.ERROR, LogLevel.CRITICAL]:
                self._write_log_entry(entry, self.error_log)
            
            # Write performance metrics to performance log
            if performance_metrics:
                perf_entry = self._create_log_entry(
                    LogLevel.INFO, 
                    f"performance.{event}", 
                    "Performance metrics recorded",
                    performance_metrics
                )
                self._write_log_entry(perf_entry, self.performance_log)
            
            # Update session metrics
            self._update_session_metrics(level, event)
            
            # Also log to console for immediate feedback
            self._log_to_console(level, event, message, context)
    
    def _log_to_console(self, level: LogLevel, event: str, message: str, context: Dict[str, Any] = None):
        """Log to console with formatting"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        level_emoji = {
            LogLevel.DEBUG: "🔍",
            LogLevel.INFO: "ℹ️",
            LogLevel.WARNING: "⚠️",
            LogLevel.ERROR: "❌",
            LogLevel.CRITICAL: "🚨"
        }
        
        emoji = level_emoji.get(level, "📝")
        context_str = f" [{context}]" if context else ""
        print(f"{timestamp} {emoji} {event}: {message}{context_str}")
    
    def debug(self, event: str, message: str, context: Dict[str, Any] = None, 
              performance_metrics: Dict[str, Any] = None):
        """Log debug message"""
        self.log(LogLevel.DEBUG, event, message, context, performance_metrics)
    
    def info(self, event: str, message: str, context: Dict[str, Any] = None, 
             performance_metrics: Dict[str, Any] = None):
        """Log info message"""
        self.log(LogLevel.INFO, event, message, context, performance_metrics)
    
    def warning(self, event: str, message: str, context: Dict[str, Any] = None, 
                performance_metrics: Dict[str, Any] = None):
        """Log warning message"""
        self.log(LogLevel.WARNING, event, message, context, performance_metrics)
    
    def error(self, event: str, message: str, context: Dict[str, Any] = None, 
              performance_metrics: Dict[str, Any] = None):
        """Log error message"""
        self.log(LogLevel.ERROR, event, message, context, performance_metrics)
    
    def critical(self, event: str, message: str, context: Dict[str, Any] = None, 
                 performance_metrics: Dict[str, Any] = None):
        """Log critical message"""
        self.log(LogLevel.CRITICAL, event, message, context, performance_metrics)
    
    def log_trade(self, trade_type: str, symbol: str, amount: float, 
                  success: bool, profit_loss: float = 0.0, 
                  execution_time: float = 0.0, error: str = None):
        """Log trade-specific events"""
        context = {
            'symbol': symbol,
            'amount': amount,
            'success': success,
            'profit_loss': profit_loss,
            'execution_time_ms': execution_time
        }
        
        if error:
            context['error'] = error
        
        performance_metrics = {
            'execution_time_ms': execution_time,
            'profit_loss': profit_loss,
            'success_rate': 1.0 if success else 0.0
        }
        
        if success:
            self.info(f"trade.{trade_type}.success", 
                     f"Trade {trade_type} successful for {symbol}", 
                     context, performance_metrics)
        else:
            self.error(f"trade.{trade_type}.failure", 
                      f"Trade {trade_type} failed for {symbol}", 
                      context, performance_metrics)
    
    def log_performance(self, component: str, operation: str, 
                       execution_time: float, success: bool, 
                       additional_metrics: Dict[str, Any] = None):
        """Log performance metrics"""
        context = {
            'component': component,
            'operation': operation,
            'success': success
        }
        
        performance_metrics = {
            'execution_time_ms': execution_time,
            'success': success,
            'component': component,
            'operation': operation
        }
        
        if additional_metrics:
            performance_metrics.update(additional_metrics)
        
        level = LogLevel.INFO if success else LogLevel.WARNING
        self.log(level, f"performance.{component}.{operation}", 
                f"Performance: {component}.{operation} took {execution_time:.2f}ms", 
                context, performance_metrics)
    
    def log_ai_module(self, module_name: str, operation: str, 
                     execution_time: float, success: bool, 
                     result_data: Dict[str, Any] = None):
        """Log AI module operations"""
        context = {
            'module': module_name,
            'operation': operation,
            'success': success
        }
        
        performance_metrics = {
            'execution_time_ms': execution_time,
            'success': success,
            'module': module_name,
            'operation': operation
        }
        
        if result_data:
            performance_metrics.update(result_data)
        
        level = LogLevel.INFO if success else LogLevel.WARNING
        self.log(level, f"ai.{module_name}.{operation}", 
                f"AI Module: {module_name}.{operation} {'succeeded' if success else 'failed'}", 
                context, performance_metrics)
    
    def log_risk_assessment(self, risk_level: str, risk_score: float, 
                           approved: bool, reason: str, 
                           risk_factors: Dict[str, float]):
        """Log risk assessment results"""
        context = {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'approved': approved,
            'reason': reason,
            'risk_factors': risk_factors
        }
        
        performance_metrics = {
            'risk_score': risk_score,
            'approved': approved,
            'risk_level': risk_level
        }
        
        level = LogLevel.INFO if approved else LogLevel.WARNING
        self.log(level, "risk.assessment", 
                f"Risk Assessment: {risk_level} risk (score: {risk_score:.2f}) - {'APPROVED' if approved else 'REJECTED'}", 
                context, performance_metrics)
    
    def get_session_summary(self, session_id: str = None) -> Dict[str, Any]:
        """Get summary for a session"""
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id and session_id in self.session_metrics:
            return self.session_metrics[session_id].copy()
        
        return {}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all sessions"""
        with self.lock:
            total_logs = sum(session['log_count'] for session in self.session_metrics.values())
            total_errors = sum(session['error_count'] for session in self.session_metrics.values())
            total_trades = sum(session['trade_count'] for session in self.session_metrics.values())
            
            return {
                'total_sessions': len(self.session_metrics),
                'total_logs': total_logs,
                'total_errors': total_errors,
                'total_trades': total_trades,
                'error_rate': total_errors / max(1, total_logs),
                'current_session': self.current_session_id,
                'sessions': dict(self.session_metrics)
            }
    
    def export_logs(self, output_file: str, session_id: str = None, 
                   log_types: List[str] = None):
        """Export logs to a file"""
        if log_types is None:
            log_types = ['main', 'trades', 'performance', 'errors']
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'session_id': session_id or self.current_session_id,
            'logs': {}
        }
        
        log_files = {
            'main': self.main_log,
            'trades': self.trade_log,
            'performance': self.performance_log,
            'errors': self.error_log
        }
        
        for log_type in log_types:
            if log_type in log_files and log_files[log_type].exists():
                try:
                    with open(log_files[log_type], 'r') as f:
                        lines = f.readlines()
                        export_data['logs'][log_type] = [
                            json.loads(line.strip()) for line in lines
                        ]
                except Exception as e:
                    export_data['logs'][log_type] = f"Error reading {log_type} log: {e}"
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

# Global structured logger instance
structured_logger = StructuredLogger()

def get_structured_logger() -> StructuredLogger:
    """Get global structured logger instance"""
    return structured_logger

# Convenience functions
def log_info(event: str, message: Optional[str] = None, context: Dict[str, Any] = None, 
             performance_metrics: Dict[str, Any] = None, **kwargs):
    """Log info message (flexible: accepts event+message or event+kwargs as context)"""
    if kwargs:
        context = {**(context or {}), **kwargs}
    structured_logger.info(event, message or "", context, performance_metrics)

def log_error(event: str, message: Optional[str] = None, context: Dict[str, Any] = None, 
              performance_metrics: Dict[str, Any] = None, **kwargs):
    """Log error message (flexible: accepts event+message or event+kwargs as context)"""
    if kwargs:
        context = {**(context or {}), **kwargs}
    structured_logger.error(event, message or "", context, performance_metrics)

def log_warning(event: str, message: Optional[str] = None, context: Dict[str, Any] = None, 
                performance_metrics: Dict[str, Any] = None, **kwargs):
    """Log warning message (flexible: accepts event+message or event+kwargs as context)"""
    if kwargs:
        context = {**(context or {}), **kwargs}
    structured_logger.warning(event, message or "", context, performance_metrics)

def log_trade(trade_type: str, symbol: str, amount: float, 
              success: bool, profit_loss: float = 0.0, 
              execution_time: float = 0.0, error: str = None):
    """Log trade event"""
    structured_logger.log_trade(trade_type, symbol, amount, success, profit_loss, execution_time, error)

def log_performance(component: str, operation: str, 
                   execution_time: float, success: bool, 
                   additional_metrics: Dict[str, Any] = None):
    """Log performance metrics"""
    structured_logger.log_performance(component, operation, execution_time, success, additional_metrics)

def log_ai_module(module_name: str, operation: str, 
                 execution_time: float, success: bool, 
                 result_data: Dict[str, Any] = None):
    """Log AI module operation"""
    structured_logger.log_ai_module(module_name, operation, execution_time, success, result_data)

def log_risk_assessment(risk_level: str, risk_score: float, 
                       approved: bool, reason: str, 
                       risk_factors: Dict[str, float]):
    """Log risk assessment"""
    structured_logger.log_risk_assessment(risk_level, risk_score, approved, reason, risk_factors)

def start_logging_session(session_id: str = None):
    """Start a new logging session"""
    structured_logger.start_session(session_id)

def end_logging_session():
    """End current logging session"""
    structured_logger.end_session()

def get_logging_summary() -> Dict[str, Any]:
    """Get logging summary"""
    return structured_logger.get_performance_summary()
