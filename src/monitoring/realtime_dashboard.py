#!/usr/bin/env python3
"""
Real-time Performance Dashboard - Phase 3
Live monitoring dashboard with WebSocket support and real-time metrics
"""

import asyncio
import json
import time
import websockets
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import logging
import os
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.monitoring.structured_logger import log_info, log_error
from src.config.config_validator import get_validated_config
from src.core.centralized_risk_manager import get_risk_summary
from src.ai.ai_circuit_breaker import get_ai_module_status

logger = logging.getLogger(__name__)

@dataclass
class TradingMetrics:
    """Real-time trading metrics"""
    timestamp: str
    total_trades: int
    successful_trades: int
    failed_trades: int
    success_rate: float
    total_pnl: float
    avg_execution_time: float
    trades_per_hour: float
    health_score: int
    active_positions: int
    daily_pnl: float
    weekly_pnl: float
    monthly_pnl: float

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    active_connections: int
    cache_hit_rate: float
    error_rate: float

@dataclass
class AISystemMetrics:
    """AI system metrics"""
    timestamp: str
    overall_healthy: bool
    unhealthy_modules: List[str]
    total_modules: int
    avg_response_time: float
    total_requests: int
    error_rate: float

class RealTimeDashboard:
    """
    Real-time performance dashboard with:
    - WebSocket server for live updates
    - Historical data storage
    - Performance metrics collection
    - Alert system
    - Multiple client support
    """
    
    def __init__(self, config: Any = None):
        self.config = config or get_validated_config()
        self.websocket_server = None
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        
        # Metrics storage
        self.trading_metrics_history = deque(maxlen=1000)  # Last 1000 data points
        self.system_metrics_history = deque(maxlen=1000)
        self.ai_metrics_history = deque(maxlen=1000)
        
        # Real-time data
        self.current_trading_metrics: Optional[TradingMetrics] = None
        self.current_system_metrics: Optional[SystemMetrics] = None
        self.current_ai_metrics: Optional[AISystemMetrics] = None
        
        # Performance tracking
        self.performance_data = {
            "start_time": time.time(),
            "total_cycles": 0,
            "total_errors": 0,
            "peak_memory": 0,
            "peak_cpu": 0
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            "success_rate_min": 0.6,
            "health_score_min": 70,
            "error_rate_max": 0.1,
            "memory_usage_max": 0.8,
            "cpu_usage_max": 0.8
        }
        
        # Active alerts
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        
    async def start_websocket_server(self, host: str = "localhost", port: int = 8765):
        """Start WebSocket server for real-time updates"""
        try:
            self.websocket_server = await websockets.serve(
                self.handle_client, host, port
            )
            log_info("dashboard.start", f"Real-time dashboard WebSocket server started on {host}:{port}")
            return True
        except Exception as e:
            log_error(f"Failed to start WebSocket server: {e}")
            return False
    
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connections"""
        self.clients.add(websocket)
        client_ip = websocket.remote_address[0] if websocket.remote_address else "unknown"
        log_info("dashboard.client", f"Dashboard client connected from {client_ip}")
        
        try:
            # Send initial data
            await self.send_initial_data(websocket)
            
            # Keep connection alive and send updates
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    await self.send_error(websocket, "Invalid JSON message")
                except Exception as e:
                    log_error(f"Error handling client message: {e}")
                    await self.send_error(websocket, f"Server error: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            log_info("dashboard.client", f"Dashboard client disconnected from {client_ip}")
        except Exception as e:
            log_error(f"WebSocket error: {e}")
        finally:
            self.clients.discard(websocket)
    
    async def send_initial_data(self, websocket):
        """Send initial dashboard data to new client"""
        initial_data = {
            "type": "initial_data",
            "timestamp": datetime.now().isoformat(),
            "trading_metrics": asdict(self.current_trading_metrics) if self.current_trading_metrics else None,
            "system_metrics": asdict(self.current_system_metrics) if self.current_system_metrics else None,
            "ai_metrics": asdict(self.current_ai_metrics) if self.current_ai_metrics else None,
            "performance_data": self.performance_data,
            "active_alerts": self.active_alerts,
            "alert_thresholds": self.alert_thresholds
        }
        
        await self.send_to_client(websocket, initial_data)
    
    async def handle_client_message(self, websocket, data: Dict[str, Any]):
        """Handle messages from clients"""
        message_type = data.get("type")
        
        if message_type == "ping":
            await self.send_to_client(websocket, {"type": "pong", "timestamp": datetime.now().isoformat()})
        
        elif message_type == "get_metrics":
            await self.send_current_metrics(websocket)
        
        elif message_type == "get_history":
            time_range = data.get("time_range", "1h")
            await self.send_historical_data(websocket, time_range)
        
        elif message_type == "update_thresholds":
            new_thresholds = data.get("thresholds", {})
            self.alert_thresholds.update(new_thresholds)
            await self.send_to_client(websocket, {
                "type": "thresholds_updated",
                "thresholds": self.alert_thresholds
            })
        
        else:
            await self.send_error(websocket, f"Unknown message type: {message_type}")
    
    async def send_to_client(self, websocket, data: Dict[str, Any]):
        """Send data to a specific client"""
        try:
            await websocket.send(json.dumps(data))
        except websockets.exceptions.ConnectionClosed:
            self.clients.discard(websocket)
        except Exception as e:
            log_error(f"Error sending data to client: {e}")
    
    async def broadcast_to_all(self, data: Dict[str, Any]):
        """Broadcast data to all connected clients"""
        if not self.clients:
            return
        
        message = json.dumps(data)
        disconnected_clients = set()
        
        for client in self.clients.copy():
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                log_error(f"Error broadcasting to client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.clients -= disconnected_clients
    
    async def send_error(self, websocket, error_message: str):
        """Send error message to client"""
        await self.send_to_client(websocket, {
            "type": "error",
            "message": error_message,
            "timestamp": datetime.now().isoformat()
        })
    
    async def send_current_metrics(self, websocket):
        """Send current metrics to client"""
        data = {
            "type": "current_metrics",
            "timestamp": datetime.now().isoformat(),
            "trading_metrics": asdict(self.current_trading_metrics) if self.current_trading_metrics else None,
            "system_metrics": asdict(self.current_system_metrics) if self.current_system_metrics else None,
            "ai_metrics": asdict(self.current_ai_metrics) if self.current_ai_metrics else None
        }
        await self.send_to_client(websocket, data)
    
    async def send_historical_data(self, websocket, time_range: str):
        """Send historical data to client"""
        # Calculate time cutoff
        now = time.time()
        if time_range == "1h":
            cutoff = now - 3600
        elif time_range == "6h":
            cutoff = now - 21600
        elif time_range == "24h":
            cutoff = now - 86400
        else:
            cutoff = now - 3600  # Default to 1 hour
        
        # Filter historical data
        trading_data = [
            asdict(metrics) for metrics in self.trading_metrics_history
            if time.mktime(datetime.fromisoformat(metrics.timestamp).timetuple()) > cutoff
        ]
        
        system_data = [
            asdict(metrics) for metrics in self.system_metrics_history
            if time.mktime(datetime.fromisoformat(metrics.timestamp).timetuple()) > cutoff
        ]
        
        ai_data = [
            asdict(metrics) for metrics in self.ai_metrics_history
            if time.mktime(datetime.fromisoformat(metrics.timestamp).timetuple()) > cutoff
        ]
        
        data = {
            "type": "historical_data",
            "time_range": time_range,
            "trading_metrics": trading_data,
            "system_metrics": system_data,
            "ai_metrics": ai_data,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.send_to_client(websocket, data)
    
    async def collect_trading_metrics(self) -> TradingMetrics:
        """Collect current trading metrics"""
        try:
            # Get risk summary
            risk_summary = get_risk_summary()
            
            # Calculate time-based PnL (simplified)
            current_time = datetime.now()
            daily_pnl = risk_summary.get("daily_pnl_usd", 0.0)
            weekly_pnl = daily_pnl * 7  # Simplified calculation
            monthly_pnl = daily_pnl * 30  # Simplified calculation
            
            metrics = TradingMetrics(
                timestamp=current_time.isoformat(),
                total_trades=0,  # Would be populated from performance monitor
                successful_trades=0,
                failed_trades=0,
                success_rate=0.0,
                total_pnl=0.0,
                avg_execution_time=0.0,
                trades_per_hour=0.0,
                health_score=100,
                active_positions=risk_summary.get("open_positions_count", 0),
                daily_pnl=daily_pnl,
                weekly_pnl=weekly_pnl,
                monthly_pnl=monthly_pnl
            )
            
            self.current_trading_metrics = metrics
            self.trading_metrics_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            log_error(f"Error collecting trading metrics: {e}")
            return None
    
    async def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            import psutil
            
            # Get system metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O
            network_io = psutil.net_io_counters()
            
            metrics = SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                network_io={
                    "bytes_sent": network_io.bytes_sent,
                    "bytes_recv": network_io.bytes_recv,
                    "packets_sent": network_io.packets_sent,
                    "packets_recv": network_io.packets_recv
                },
                active_connections=len(psutil.net_connections()),
                cache_hit_rate=0.0,  # Would be populated from cache system
                error_rate=0.0  # Would be calculated from error logs
            )
            
            self.current_system_metrics = metrics
            self.system_metrics_history.append(metrics)
            
            # Update performance data
            self.performance_data["peak_memory"] = max(
                self.performance_data["peak_memory"], memory.percent
            )
            self.performance_data["peak_cpu"] = max(
                self.performance_data["peak_cpu"], cpu_usage
            )
            
            return metrics
            
        except Exception as e:
            log_error(f"Error collecting system metrics: {e}")
            return None
    
    async def collect_ai_metrics(self) -> AISystemMetrics:
        """Collect current AI system metrics"""
        try:
            # Get AI module status
            ai_status = get_ai_module_status()
            
            metrics = AISystemMetrics(
                timestamp=datetime.now().isoformat(),
                overall_healthy=ai_status.get("overall_healthy", False),
                unhealthy_modules=ai_status.get("unhealthy_modules", []),
                total_modules=len(ai_status.get("unhealthy_modules", [])) + 
                             (1 if ai_status.get("overall_healthy", False) else 0),
                avg_response_time=0.0,  # Would be calculated from AI module response times
                total_requests=0,  # Would be tracked by AI modules
                error_rate=0.0  # Would be calculated from AI module errors
            )
            
            self.current_ai_metrics = metrics
            self.ai_metrics_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            log_error(f"Error collecting AI metrics: {e}")
            return None
    
    async def check_alerts(self):
        """Check for alert conditions"""
        alerts = []
        
        # Check trading metrics alerts
        if self.current_trading_metrics:
            if self.current_trading_metrics.success_rate < self.alert_thresholds["success_rate_min"]:
                alerts.append({
                    "type": "trading",
                    "level": "warning",
                    "message": f"Success rate below threshold: {self.current_trading_metrics.success_rate:.2%}",
                    "value": self.current_trading_metrics.success_rate,
                    "threshold": self.alert_thresholds["success_rate_min"]
                })
            
            if self.current_trading_metrics.health_score < self.alert_thresholds["health_score_min"]:
                alerts.append({
                    "type": "trading",
                    "level": "critical",
                    "message": f"Health score below threshold: {self.current_trading_metrics.health_score}",
                    "value": self.current_trading_metrics.health_score,
                    "threshold": self.alert_thresholds["health_score_min"]
                })
        
        # Check system metrics alerts
        if self.current_system_metrics:
            if self.current_system_metrics.memory_usage > self.alert_thresholds["memory_usage_max"] * 100:
                alerts.append({
                    "type": "system",
                    "level": "warning",
                    "message": f"Memory usage above threshold: {self.current_system_metrics.memory_usage:.1f}%",
                    "value": self.current_system_metrics.memory_usage,
                    "threshold": self.alert_thresholds["memory_usage_max"] * 100
                })
            
            if self.current_system_metrics.cpu_usage > self.alert_thresholds["cpu_usage_max"] * 100:
                alerts.append({
                    "type": "system",
                    "level": "warning",
                    "message": f"CPU usage above threshold: {self.current_system_metrics.cpu_usage:.1f}%",
                    "value": self.current_system_metrics.cpu_usage,
                    "threshold": self.alert_thresholds["cpu_usage_max"] * 100
                })
        
        # Check AI metrics alerts
        if self.current_ai_metrics:
            if not self.current_ai_metrics.overall_healthy:
                alerts.append({
                    "type": "ai",
                    "level": "critical",
                    "message": f"AI modules unhealthy: {', '.join(self.current_ai_metrics.unhealthy_modules)}",
                    "value": len(self.current_ai_metrics.unhealthy_modules),
                    "threshold": 0
                })
        
        # Update active alerts
        for alert in alerts:
            alert_key = f"{alert['type']}_{alert['level']}_{alert['message']}"
            self.active_alerts[alert_key] = {
                **alert,
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat()
            }
        
        # Send alerts to clients
        if alerts:
            await self.broadcast_to_all({
                "type": "alerts",
                "alerts": alerts,
                "timestamp": datetime.now().isoformat()
            })
    
    async def run_metrics_collection(self):
        """Run continuous metrics collection"""
        log_info("Starting real-time metrics collection")
        
        while True:
            try:
                # Collect all metrics
                await asyncio.gather(
                    self.collect_trading_metrics(),
                    self.collect_system_metrics(),
                    self.collect_ai_metrics(),
                    return_exceptions=True
                )
                
                # Check for alerts
                await self.check_alerts()
                
                # Broadcast current metrics to all clients
                if self.clients:
                    await self.broadcast_to_all({
                        "type": "metrics_update",
                        "timestamp": datetime.now().isoformat(),
                        "trading_metrics": asdict(self.current_trading_metrics) if self.current_trading_metrics else None,
                        "system_metrics": asdict(self.current_system_metrics) if self.current_system_metrics else None,
                        "ai_metrics": asdict(self.current_ai_metrics) if self.current_ai_metrics else None
                    })
                
                # Update performance data
                self.performance_data["total_cycles"] += 1
                
                # Wait before next collection
                await asyncio.sleep(5)  # Collect every 5 seconds
                
            except Exception as e:
                log_error(f"Error in metrics collection: {e}")
                self.performance_data["total_errors"] += 1
                await asyncio.sleep(10)  # Wait longer on error
    
    async def start_dashboard(self, host: str = "localhost", port: int = 8765):
        """Start the complete dashboard system"""
        log_info("Starting real-time performance dashboard")
        
        # Start WebSocket server
        if not await self.start_websocket_server(host, port):
            return False
        
        # Start metrics collection
        metrics_task = asyncio.create_task(self.run_metrics_collection())
        
        try:
            # Keep running
            await asyncio.gather(
                self.websocket_server.wait_closed(),
                metrics_task
            )
        except KeyboardInterrupt:
            log_info("Dashboard stopped by user")
        except Exception as e:
            log_error(f"Dashboard error: {e}")
        finally:
            # Cleanup
            metrics_task.cancel()
            if self.websocket_server:
                self.websocket_server.close()
                await self.websocket_server.wait_closed()
            
            log_info("Dashboard shutdown complete")

async def start_realtime_dashboard(host: str = "localhost", port: int = 8765):
    """Start the real-time dashboard"""
    dashboard = RealTimeDashboard()
    await dashboard.start_dashboard(host, port)

if __name__ == "__main__":
    asyncio.run(start_realtime_dashboard())
