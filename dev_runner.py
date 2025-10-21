#!/usr/bin/env python3
"""
Development runner with auto-restart functionality
Automatically restarts the bot when Python files change
"""

import os
import sys
import time
import subprocess
import signal
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class BotRestartHandler(FileSystemEventHandler):
    def __init__(self, restart_callback):
        self.restart_callback = restart_callback
        self.last_restart = 0
        self.restart_delay = 2  # Wait 2 seconds between restarts to avoid rapid restarts
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        # Only restart on Python file changes
        if event.src_path.endswith('.py'):
            current_time = time.time()
            if current_time - self.last_restart > self.restart_delay:
                print(f"\n🔄 File changed: {event.src_path}")
                print("🔄 Restarting bot in 2 seconds...")
                self.last_restart = current_time
                time.sleep(2)
                self.restart_callback()

class DevRunner:
    def __init__(self):
        self.bot_process = None
        self.observer = None
        self.running = True
        
    def start_bot(self):
        """Start the bot process"""
        if self.bot_process:
            self.stop_bot()
        
        print("🚀 Starting trading bot...")
        try:
            self.bot_process = subprocess.Popen([
                sys.executable, 'main.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print("✅ Bot started successfully")
        except Exception as e:
            print(f"❌ Failed to start bot: {e}")
    
    def stop_bot(self):
        """Stop the bot process"""
        if self.bot_process:
            print("🛑 Stopping bot...")
            try:
                self.bot_process.terminate()
                self.bot_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("⚠️  Force killing bot...")
                self.bot_process.kill()
            except Exception as e:
                print(f"⚠️  Error stopping bot: {e}")
            finally:
                self.bot_process = None
                print("✅ Bot stopped")
    
    def restart_bot(self):
        """Restart the bot"""
        print("\n" + "="*50)
        print("🔄 RESTARTING BOT")
        print("="*50)
        self.stop_bot()
        time.sleep(1)
        self.start_bot()
        print("="*50)
    
    def start_watching(self):
        """Start watching for file changes"""
        print("👀 Watching for file changes...")
        print("📁 Watching directory: .")
        print("🐍 Will restart on .py file changes")
        print("⏹️  Press Ctrl+C to stop")
        print("-" * 50)
        
        # Start the bot initially
        self.start_bot()
        
        # Set up file watcher
        event_handler = BotRestartHandler(self.restart_bot)
        self.observer = Observer()
        self.observer.schedule(event_handler, '.', recursive=True)
        self.observer.start()
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            self.running = False
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
        self.stop_bot()
        print("✅ Development runner stopped")

def main():
    print("🤖 Trading Bot Development Runner")
    print("=" * 40)
    
    # Check if main.py exists
    if not os.path.exists('main.py'):
        print("❌ main.py not found in current directory")
        sys.exit(1)
    
    runner = DevRunner()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n🛑 Received interrupt signal")
        runner.running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        runner.start_watching()
    except Exception as e:
        print(f"❌ Error: {e}")
        runner.cleanup()
        sys.exit(1)

if __name__ == "__main__":
    main()
