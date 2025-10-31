#!/usr/bin/env python3
"""
Production Deployment Manager - Phase 4
Complete production deployment system with health checks, monitoring, and auto-recovery
"""

import asyncio
import json
import time
import subprocess
import psutil
import signal
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
import yaml
from pathlib import Path
import docker
import requests

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.monitoring.structured_logger import log_info, log_error, log_warning
from src.config.config_validator import get_validated_config

logger = logging.getLogger(__name__)

@dataclass
class HealthCheck:
    """Health check result"""
    name: str
    status: str  # 'healthy', 'unhealthy', 'warning'
    message: str
    timestamp: str
    response_time: float
    details: Dict[str, Any]

@dataclass
class SystemStatus:
    """Overall system status"""
    timestamp: str
    overall_health: str
    health_checks: List[HealthCheck]
    system_metrics: Dict[str, Any]
    alerts: List[str]
    uptime: float
    version: str

class HealthChecker:
    """Comprehensive health checking system"""
    
    def __init__(self):
        self.checks = {}
        self.last_check = {}
        self.check_interval = 30  # 30 seconds
        self.timeout = 10  # 10 seconds per check
        
    async def register_check(self, name: str, check_func: callable, interval: int = 30):
        """Register a health check function"""
        self.checks[name] = {
            'function': check_func,
            'interval': interval,
            'last_run': 0
        }
        log_info("health_check.registered", f"Registered health check: {name}")
    
    async def run_all_checks(self) -> List[HealthCheck]:
        """Run all registered health checks"""
        results = []
        current_time = time.time()
        
        for name, check_info in self.checks.items():
            # Check if enough time has passed since last run
            if current_time - check_info['last_run'] < check_info['interval']:
                continue
            
            try:
                start_time = time.time()
                result = await asyncio.wait_for(
                    check_info['function'](),
                    timeout=self.timeout
                )
                response_time = time.time() - start_time
                
                health_check = HealthCheck(
                    name=name,
                    status=result.get('status', 'unhealthy'),
                    message=result.get('message', 'No message'),
                    timestamp=datetime.now().isoformat(),
                    response_time=response_time,
                    details=result.get('details', {})
                )
                
                results.append(health_check)
                check_info['last_run'] = current_time
                
            except asyncio.TimeoutError:
                health_check = HealthCheck(
                    name=name,
                    status='unhealthy',
                    message=f'Health check timed out after {self.timeout}s',
                    timestamp=datetime.now().isoformat(),
                    response_time=self.timeout,
                    details={}
                )
                results.append(health_check)
                
            except Exception as e:
                health_check = HealthCheck(
                    name=name,
                    status='unhealthy',
                    message=f'Health check error: {str(e)}',
                    timestamp=datetime.now().isoformat(),
                    response_time=0,
                    details={'error': str(e)}
                )
                results.append(health_check)
        
        return results
    
    async def check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and performance"""
        try:
            # Check if database files exist and are accessible
            db_files = ['data/risk_state.json', 'data/open_positions.json', 'data/performance_data.json']
            accessible_files = 0
            
            for db_file in db_files:
                if os.path.exists(db_file):
                    try:
                        with open(db_file, 'r') as f:
                            json.load(f)
                        accessible_files += 1
                    except Exception:
                        pass
            
            health_score = accessible_files / len(db_files)
            
            if health_score >= 0.8:
                status = 'healthy'
                message = f'Database accessible ({accessible_files}/{len(db_files)} files)'
            elif health_score >= 0.5:
                status = 'warning'
                message = f'Database partially accessible ({accessible_files}/{len(db_files)} files)'
            else:
                status = 'unhealthy'
                message = f'Database issues ({accessible_files}/{len(db_files)} files accessible)'
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'accessible_files': accessible_files,
                    'total_files': len(db_files),
                    'health_score': health_score
                }
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Database check failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    async def check_ai_modules_health(self) -> Dict[str, Any]:
        """Check AI modules health"""
        try:
            from src.ai.ai_circuit_breaker import check_ai_module_health
            
            health_data = check_ai_module_health()
            overall_healthy = health_data.get('overall_healthy', False)
            unhealthy_modules = health_data.get('unhealthy_modules', [])
            
            if overall_healthy:
                status = 'healthy'
                message = 'All AI modules healthy'
            elif len(unhealthy_modules) <= 2:
                status = 'warning'
                message = f'{len(unhealthy_modules)} AI modules unhealthy: {", ".join(unhealthy_modules)}'
            else:
                status = 'unhealthy'
                message = f'Multiple AI modules unhealthy: {", ".join(unhealthy_modules)}'
            
            return {
                'status': status,
                'message': message,
                'details': health_data
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'AI modules check failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    async def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Network connectivity
            network_ok = True
            try:
                requests.get('https://api.coingecko.com/api/v3/ping', timeout=5)
            except:
                network_ok = False
            
            # Determine overall status
            if cpu_percent > 90 or memory_percent > 90 or disk_percent > 90:
                status = 'unhealthy'
                message = f'High resource usage: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%, Disk {disk_percent:.1f}%'
            elif cpu_percent > 70 or memory_percent > 70 or disk_percent > 80:
                status = 'warning'
                message = f'Moderate resource usage: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%, Disk {disk_percent:.1f}%'
            else:
                status = 'healthy'
                message = f'Resources normal: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%, Disk {disk_percent:.1f}%'
            
            if not network_ok:
                status = 'warning' if status == 'healthy' else status
                message += ' (Network issues)'
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'disk_percent': disk_percent,
                    'network_ok': network_ok,
                    'available_memory_gb': memory.available / (1024**3),
                    'free_disk_gb': disk.free / (1024**3)
                }
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'System resources check failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    async def check_trading_system_health(self) -> Dict[str, Any]:
        """Check trading system specific health"""
        try:
            from src.core.centralized_risk_manager import get_risk_summary
            
            risk_summary = get_risk_summary()
            
            # Check if trading is paused
            is_paused = risk_summary.get('is_paused', False)
            losing_streak = risk_summary.get('losing_streak', 0)
            daily_pnl = risk_summary.get('daily_pnl_usd', 0)
            
            if is_paused:
                status = 'warning'
                message = 'Trading system is paused'
            elif losing_streak >= 3:
                status = 'warning'
                message = f'High losing streak: {losing_streak}'
            elif daily_pnl < -100:
                status = 'warning'
                message = f'Significant daily loss: ${daily_pnl:.2f}'
            else:
                status = 'healthy'
                message = 'Trading system operational'
            
            return {
                'status': status,
                'message': message,
                'details': risk_summary
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Trading system check failed: {str(e)}',
                'details': {'error': str(e)}
            }

class AutoRecovery:
    """Automatic recovery system for failed components"""
    
    def __init__(self):
        self.recovery_actions = {}
        self.max_recovery_attempts = 3
        self.recovery_cooldown = 300  # 5 minutes
        
    def register_recovery_action(self, check_name: str, recovery_func: callable):
        """Register a recovery action for a health check"""
        self.recovery_actions[check_name] = {
            'function': recovery_func,
            'attempts': 0,
            'last_attempt': 0
        }
        log_info("recovery.registered", f"Registered recovery action for: {check_name}")
    
    async def attempt_recovery(self, check_name: str, health_check: HealthCheck) -> bool:
        """Attempt to recover from a failed health check"""
        if check_name not in self.recovery_actions:
            return False
        
        recovery_info = self.recovery_actions[check_name]
        current_time = time.time()
        
        # Check cooldown
        if current_time - recovery_info['last_attempt'] < self.recovery_cooldown:
            return False
        
        # Check max attempts
        if recovery_info['attempts'] >= self.max_recovery_attempts:
            log_error(f"Max recovery attempts reached for {check_name}")
            return False
        
        try:
            log_info("recovery.attempt", f"Attempting recovery for {check_name} (attempt {recovery_info['attempts'] + 1})")
            
            success = await recovery_info['function'](health_check)
            
            recovery_info['attempts'] += 1
            recovery_info['last_attempt'] = current_time
            
            if success:
                log_info("recovery.success", f"Recovery successful for {check_name}")
                recovery_info['attempts'] = 0  # Reset on success
            else:
                log_warning(f"Recovery failed for {check_name}")
            
            return success
            
        except Exception as e:
            log_error(f"Recovery error for {check_name}: {e}")
            recovery_info['attempts'] += 1
            recovery_info['last_attempt'] = current_time
            return False
    
    async def recover_database(self, health_check: HealthCheck) -> bool:
        """Recover database issues"""
        try:
            # Try to recreate missing database files
            db_files = ['data/risk_state.json', 'data/open_positions.json', 'data/performance_data.json']
            
            os.makedirs('data', exist_ok=True)
            for db_file in db_files:
                if not os.path.exists(db_file):
                    with open(db_file, 'w') as f:
                        json.dump({}, f)
                    log_info("recovery.database", f"Created missing database file: {db_file}")
            
            return True
            
        except Exception as e:
            log_error(f"Database recovery failed: {e}")
            return False
    
    async def recover_ai_modules(self, health_check: HealthCheck) -> bool:
        """Recover AI modules"""
        try:
            # Reset circuit breakers
            from src.ai.ai_circuit_breaker import circuit_breaker_manager
            
            for module_name, circuit_breaker in circuit_breaker_manager.items():
                if circuit_breaker.state == "OPEN":
                    circuit_breaker.state = "CLOSED"
                    circuit_breaker.failure_count = 0
                    log_info("recovery.circuit_breaker", f"Reset circuit breaker for {module_name}")
            
            return True
            
        except Exception as e:
            log_error(f"AI modules recovery failed: {e}")
            return False
    
    async def recover_trading_system(self, health_check: HealthCheck) -> bool:
        """Recover trading system"""
        try:
            # Reset risk state if needed
            from src.core.centralized_risk_manager import centralized_risk_manager
            
            # Reset daily state
            centralized_risk_manager._reset_daily_state()
            
            # Clear any pauses
            centralized_risk_manager.risk_state["paused_until"] = 0
            
            log_info("Trading system recovery completed")
            return True
            
        except Exception as e:
            log_error(f"Trading system recovery failed: {e}")
            return False

class ProductionManager:
    """Main production management system"""
    
    def __init__(self, config: Any = None):
        self.config = config or get_validated_config()
        self.health_checker = HealthChecker()
        self.auto_recovery = AutoRecovery()
        self.start_time = time.time()
        self.version = "4.0.0"
        self.status_history = []
        self.max_status_history = 100
        
        # Initialize health checks
        # Note: This will be called asynchronously in the async context
        
        # Initialize recovery actions
        self._setup_recovery_actions()
    
    async def _setup_health_checks(self):
        """Setup all health checks"""
        await self.health_checker.register_check(
            "database", 
            self.health_checker.check_database_health,
            interval=60
        )
        
        await self.health_checker.register_check(
            "ai_modules",
            self.health_checker.check_ai_modules_health,
            interval=30
        )
        
        await self.health_checker.register_check(
            "system_resources",
            self.health_checker.check_system_resources,
            interval=30
        )
        
        await self.health_checker.register_check(
            "trading_system",
            self.health_checker.check_trading_system_health,
            interval=60
        )
    
    def _setup_recovery_actions(self):
        """Setup recovery actions"""
        self.auto_recovery.register_recovery_action(
            "database",
            self.auto_recovery.recover_database
        )
        
        self.auto_recovery.register_recovery_action(
            "ai_modules",
            self.auto_recovery.recover_ai_modules
        )
        
        self.auto_recovery.register_recovery_action(
            "trading_system",
            self.auto_recovery.recover_trading_system
        )
    
    async def get_system_status(self) -> SystemStatus:
        """Get comprehensive system status"""
        # Run all health checks
        health_checks = await self.health_checker.run_all_checks()
        
        # Determine overall health
        unhealthy_checks = [h for h in health_checks if h.status == 'unhealthy']
        warning_checks = [h for h in health_checks if h.status == 'warning']
        
        if unhealthy_checks:
            overall_health = 'unhealthy'
        elif warning_checks:
            overall_health = 'warning'
        else:
            overall_health = 'healthy'
        
        # Get system metrics
        system_metrics = {
            'uptime': time.time() - self.start_time,
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'process_count': len(psutil.pids())
        }
        
        # Generate alerts
        alerts = []
        for check in health_checks:
            if check.status == 'unhealthy':
                alerts.append(f"CRITICAL: {check.name} - {check.message}")
            elif check.status == 'warning':
                alerts.append(f"WARNING: {check.name} - {check.message}")
        
        status = SystemStatus(
            timestamp=datetime.now().isoformat(),
            overall_health=overall_health,
            health_checks=health_checks,
            system_metrics=system_metrics,
            alerts=alerts,
            uptime=time.time() - self.start_time,
            version=self.version
        )
        
        # Store in history
        self.status_history.append(status)
        if len(self.status_history) > self.max_status_history:
            self.status_history.pop(0)
        
        return status
    
    async def run_health_monitoring(self):
        """Run continuous health monitoring"""
        log_info("Starting production health monitoring")
        
        while True:
            try:
                # Get system status
                status = await self.get_system_status()
                
                # Log status
                if status.overall_health == 'unhealthy':
                    log_error(f"System unhealthy: {len(status.alerts)} alerts")
                elif status.overall_health == 'warning':
                    log_warning(f"System warning: {len(status.alerts)} alerts")
                else:
                    log_info("health.monitoring", f"System healthy: {len(status.health_checks)} checks passed")
                
                # Attempt recovery for unhealthy checks
                for health_check in status.health_checks:
                    if health_check.status == 'unhealthy':
                        await self.auto_recovery.attempt_recovery(health_check.name, health_check)
                
                # Wait before next check
                await asyncio.sleep(30)
                
            except Exception as e:
                log_error(f"Error in health monitoring: {e}")
                await asyncio.sleep(60)
    
    async def start_production_system(self):
        """Start the complete production system"""
        log_info("production.start", "Starting production trading system")
        
        try:
            # Setup health checks
            await self._setup_health_checks()
            
            # Start health monitoring
            health_task = asyncio.create_task(self.run_health_monitoring())
            
            # Start main trading system
            from src.execution.enhanced_async_trading import run_enhanced_async_trading
            trading_task = asyncio.create_task(run_enhanced_async_trading())
            
            # Start real-time dashboard
            from src.monitoring.realtime_dashboard import start_realtime_dashboard
            dashboard_task = asyncio.create_task(start_realtime_dashboard())
            
            # Wait for all tasks
            await asyncio.gather(
                health_task,
                trading_task,
                dashboard_task,
                return_exceptions=True
            )
            
        except KeyboardInterrupt:
            log_info("production.stop", "Production system stopped by user")
        except Exception as e:
            log_error(f"Production system error: {e}")
        finally:
            log_info("production.shutdown", "Production system shutdown complete")
    
    def generate_status_report(self) -> str:
        """Generate comprehensive status report"""
        if not self.status_history:
            return "No status data available"
        
        latest_status = self.status_history[-1]
        
        report = f"""
# Production System Status Report

**Generated:** {latest_status.timestamp}
**Overall Health:** {latest_status.overall_health.upper()}
**Uptime:** {latest_status.uptime:.1f} seconds
**Version:** {latest_status.version}

## Health Checks
"""
        
        for check in latest_status.health_checks:
            status_emoji = "✅" if check.status == "healthy" else "⚠️" if check.status == "warning" else "❌"
            report += f"- {status_emoji} **{check.name}**: {check.message} ({check.response_time:.2f}s)\n"
        
        report += f"""
## System Metrics
- **CPU Usage:** {latest_status.system_metrics['cpu_percent']:.1f}%
- **Memory Usage:** {latest_status.system_metrics['memory_percent']:.1f}%
- **Disk Usage:** {latest_status.system_metrics['disk_percent']:.1f}%
- **Process Count:** {latest_status.system_metrics['process_count']}

## Alerts
"""
        
        if latest_status.alerts:
            for alert in latest_status.alerts:
                report += f"- {alert}\n"
        else:
            report += "- No active alerts\n"
        
        return report

class DockerManager:
    """Docker container management for production deployment"""
    
    def __init__(self):
        try:
            self.client = docker.from_env()
            self.container_name = "hunter-trading-bot"
        except Exception as e:
            log_error(f"Docker not available: {e}")
            self.client = None
    
    def build_image(self, dockerfile_path: str = "Dockerfile") -> bool:
        """Build Docker image"""
        if not self.client:
            return False
        
        try:
            log_info("Building Docker image...")
            image, build_logs = self.client.images.build(
                path=".",
                tag="hunter-trading-bot:latest",
                dockerfile=dockerfile_path
            )
            
            log_info("Docker image built successfully")
            return True
            
        except Exception as e:
            log_error(f"Docker build failed: {e}")
            return False
    
    def run_container(self, environment_vars: Dict[str, str] = None) -> bool:
        """Run Docker container"""
        if not self.client:
            return False
        
        try:
            # Stop existing container if running
            try:
                existing_container = self.client.containers.get(self.container_name)
                if existing_container.status == "running":
                    existing_container.stop()
                    existing_container.remove()
            except:
                pass
            
            # Run new container
            container = self.client.containers.run(
                "hunter-trading-bot:latest",
                name=self.container_name,
                environment=environment_vars or {},
                detach=True,
                ports={'8765': 8765},  # Dashboard port
                volumes={
                    '/var/run/docker.sock': {'bind': '/var/run/docker.sock', 'mode': 'ro'}
                }
            )
            
            log_info("docker.container", f"Docker container started: {container.id}")
            return True
            
        except Exception as e:
            log_error(f"Docker run failed: {e}")
            return False
    
    def stop_container(self) -> bool:
        """Stop Docker container"""
        if not self.client:
            return False
        
        try:
            container = self.client.containers.get(self.container_name)
            container.stop()
            container.remove()
            log_info("Docker container stopped")
            return True
            
        except Exception as e:
            log_error(f"Docker stop failed: {e}")
            return False

async def start_production_system():
    """Start the complete production system"""
    manager = ProductionManager()
    await manager.start_production_system()

if __name__ == "__main__":
    asyncio.run(start_production_system())
