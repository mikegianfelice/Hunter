#!/usr/bin/env python3
"""
Position Monitor Watchdog
Monitors the position monitor process and automatically restarts it if it crashes.
"""

import os
import sys
import time
import json
import subprocess
import signal
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

MONITOR_SCRIPT = project_root / "src" / "monitoring" / "monitor_position.py"
PID_FILE = project_root / "data" / ".monitor_watchdog.pid"
MONITOR_PID_FILE = project_root / "data" / ".monitor_lock"
LOG_FILE = project_root / "logs" / "monitor_watchdog.log"
CHECK_INTERVAL = 60  # Check every 60 seconds
MAX_RESTART_ATTEMPTS = 5  # Maximum restart attempts within 5 minutes
RESTART_WINDOW = 300  # 5 minutes in seconds
RECONCILE_INTERVAL_SEC = 30 * 60  # 30 minutes

_running = True
_monitor_process = None
_restart_history = []  # Track restart times
_last_reconcile_ts = 0.0

def log_message(message: str):
    """Log message to file and console"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", buffering=1) as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Failed to write to log file: {e}")
    
    print(log_entry.strip())

def cleanup_old_restart_history():
    """Remove restart timestamps older than the restart window"""
    current_time = time.time()
    global _restart_history
    _restart_history = [t for t in _restart_history if current_time - t < RESTART_WINDOW]

def should_restart() -> bool:
    """Check if we should restart (not too many restarts recently)"""
    cleanup_old_restart_history()
    return len(_restart_history) < MAX_RESTART_ATTEMPTS

def launch_monitor() -> subprocess.Popen:
    """Launch the position monitor process"""
    if not MONITOR_SCRIPT.exists():
        log_message(f"❌ Monitor script not found at {MONITOR_SCRIPT}")
        return None
    
    try:
        # Add project root to PYTHONPATH
        env = os.environ.copy()
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{project_root}:{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = str(project_root)
        
        # Open log file in append mode
        log_dir = project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        monitor_log_file = log_dir / "position_monitor.log"
        
        log_file = open(monitor_log_file, "a", buffering=1)
        
        process = subprocess.Popen(
            [sys.executable, str(MONITOR_SCRIPT)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(project_root),
            start_new_session=True
        )
        
        log_message(f"✅ Launched position monitor (PID: {process.pid})")
        return process
    except Exception as e:
        log_message(f"❌ Failed to launch monitor: {e}")
        import traceback
        log_message(f"Traceback: {traceback.format_exc()}")
        return None

def is_monitor_running(process: subprocess.Popen) -> bool:
    """Check if monitor process is still running"""
    if process is None:
        return False
    
    # Check if process is still alive
    if process.poll() is not None:
        # Process has exited
        return False
    
    # Also check if PID file exists (monitor creates this as JSON)
    if MONITOR_PID_FILE.exists():
        try:
            # Try to read PID from lock file (it's JSON format)
            with open(MONITOR_PID_FILE, 'r') as f:
                data = json.load(f)
                pid = int(data.get("pid", -1))
                if pid > 0:
                    # Check if process with that PID exists
                    try:
                        os.kill(pid, 0)  # Signal 0 just checks if process exists
                        return True
                    except OSError:
                        # PID doesn't exist
                        return False
        except (json.JSONDecodeError, ValueError, KeyError, OSError):
            # Lock file corrupted or doesn't contain valid PID
            pass
    
    # If we can't verify via PID file, assume running if process object is alive
    return process.poll() is None

def save_watchdog_pid():
    """Save watchdog PID to file"""
    try:
        PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(PID_FILE, 'w') as f:
            f.write(str(os.getpid()))
    except Exception as e:
        log_message(f"⚠️ Failed to save watchdog PID: {e}")

def remove_watchdog_pid():
    """Remove watchdog PID file"""
    try:
        if PID_FILE.exists():
            PID_FILE.unlink()
    except Exception as e:
        log_message(f"⚠️ Failed to remove watchdog PID: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    global _running
    log_message(f"⚠️ Received signal {signum}, shutting down...")
    _running = False
    
    # Stop monitor process
    global _monitor_process
    if _monitor_process:
        try:
            log_message("🛑 Stopping position monitor...")
            _monitor_process.terminate()
            _monitor_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            log_message("⚠️ Monitor didn't stop gracefully, forcing...")
            _monitor_process.kill()
        except Exception as e:
            log_message(f"⚠️ Error stopping monitor: {e}")
    
    remove_watchdog_pid()
    sys.exit(0)

def is_watchdog_running() -> bool:
    """Check if another watchdog instance is already running"""
    if not PID_FILE.exists():
        return False
    
    try:
        with open(PID_FILE, 'r') as f:
            pid = int(f.read().strip())
        # Check if process exists
        try:
            os.kill(pid, 0)
            # Process exists - another watchdog is running
            return True
        except OSError:
            # PID doesn't exist - stale file
            return False
    except (ValueError, OSError):
        return False

def _maybe_reconcile_positions():
    """Periodically reconcile position sizes from on-chain data"""
    global _last_reconcile_ts
    now = time.time()
    
    if now - _last_reconcile_ts < RECONCILE_INTERVAL_SEC:
        return
    
    _last_reconcile_ts = now
    
    try:
        from src.utils.position_sync import reconcile_position_sizes
        
        log_message("🔄 Running periodic position size reconciliation...")
        stats = reconcile_position_sizes(
            threshold_pct=5.0,
            min_balance_threshold=1e-6,
            chains=None,  # All chains
            verify_balance=True,
            dry_run=False,
            verbose=False,
        )
        
        log_message(
            f"✅ Position reconciliation complete: "
            f"updated={stats['updated']} "
            f"closed={stats['closed']} "
            f"skipped={stats['skipped']} "
            f"errors={len(stats['errors'])}"
        )
        
        if stats["errors"]:
            for err in stats["errors"][:5]:  # Log first 5 errors
                log_message(f"⚠️  Reconciliation error: {err}")
            if len(stats["errors"]) > 5:
                log_message(f"⚠️  ... and {len(stats['errors']) - 5} more errors")
                
    except Exception as e:
        log_message(f"❌ Position reconciliation failed: {e}")
        import traceback
        log_message(f"Traceback: {traceback.format_exc()}")

def main():
    """Main watchdog loop"""
    global _running, _monitor_process
    
    # Check if another watchdog is already running
    if is_watchdog_running():
        log_message("⚠️ Another watchdog instance is already running. Exiting.")
        sys.exit(0)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Save PID
    save_watchdog_pid()
    
    log_message("=" * 60)
    log_message("🚀 Position Monitor Watchdog Starting")
    log_message("=" * 60)
    
    # Launch monitor initially
    _monitor_process = launch_monitor()
    if _monitor_process is None:
        log_message("❌ Failed to launch monitor initially, exiting")
        remove_watchdog_pid()
        sys.exit(1)
    
    consecutive_failures = 0
    
    try:
        while _running:
            time.sleep(CHECK_INTERVAL)
            
            # Periodically reconcile position sizes
            _maybe_reconcile_positions()
            
            if not is_monitor_running(_monitor_process):
                log_message("⚠️ Position monitor is not running!")
                
                # Check if we should restart
                if not should_restart():
                    log_message(f"❌ Too many restart attempts ({len(_restart_history)}) in the last {RESTART_WINDOW}s")
                    log_message("❌ Watchdog stopping to prevent restart loop")
                    break
                
                # Record restart attempt
                _restart_history.append(time.time())
                consecutive_failures += 1
                
                log_message(f"🔄 Restarting position monitor (attempt {len(_restart_history)}/{MAX_RESTART_ATTEMPTS})...")
                
                # Clean up old process
                if _monitor_process:
                    try:
                        _monitor_process.terminate()
                        _monitor_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        _monitor_process.kill()
                    except Exception:
                        pass
                
                # Launch new process
                _monitor_process = launch_monitor()
                
                if _monitor_process is None:
                    log_message("❌ Failed to restart monitor")
                    if consecutive_failures >= 3:
                        log_message("❌ Multiple consecutive restart failures, stopping watchdog")
                        break
                else:
                    consecutive_failures = 0
                    log_message("✅ Monitor restarted successfully")
            else:
                # Monitor is running fine
                if consecutive_failures > 0:
                    log_message("✅ Monitor recovered")
                    consecutive_failures = 0
    
    except KeyboardInterrupt:
        log_message("⚠️ Watchdog interrupted by user")
    except Exception as e:
        log_message(f"❌ Watchdog error: {e}")
        import traceback
        log_message(f"Traceback: {traceback.format_exc()}")
    finally:
        # Cleanup
        if _monitor_process:
            try:
                log_message("🛑 Stopping position monitor...")
                _monitor_process.terminate()
                _monitor_process.wait(timeout=10)
            except Exception:
                try:
                    _monitor_process.kill()
                except Exception:
                    pass
        
        remove_watchdog_pid()
        log_message("👋 Watchdog stopped")

if __name__ == "__main__":
    main()

