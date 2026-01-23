#!/bin/bash

# Trading Bot Stopper
# Comprehensively stops all trading bot processes

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🛑 Stopping all trading bot processes...${NC}"

# Function to kill processes by pattern
kill_processes() {
    local pattern="$1"
    local description="$2"
    
    # Find PIDs of processes matching the pattern
    local pids=$(ps aux | grep -E "$pattern" | grep -v grep | awk '{print $2}' | tr '\n' ' ')
    
    if [ -n "$pids" ]; then
        echo -e "${YELLOW}🔍 Found $description processes: $pids${NC}"
        
        # Try graceful termination first
        echo -e "${BLUE}📤 Sending TERM signal to $description...${NC}"
        for pid in $pids; do
            kill -TERM "$pid" 2>/dev/null || true
        done
        
        # Wait longer for graceful shutdown
        sleep 3
        
        # Check if processes are still running
        local remaining_pids=$(ps aux | grep -E "$pattern" | grep -v grep | awk '{print $2}' | tr '\n' ' ')
        
        if [ -n "$remaining_pids" ]; then
            echo -e "${YELLOW}⚠️  Some $description processes still running, forcing kill...${NC}"
            # Try KILL signal multiple times with increasing wait times
            for attempt in 1 2 3; do
                for pid in $remaining_pids; do
                    # Check if process still exists and is not a zombie
                    if ps -p "$pid" > /dev/null 2>&1; then
                        local state=$(ps -p "$pid" -o state= 2>/dev/null | tr -d ' ')
                        if [ "$state" != "Z" ]; then  # Not a zombie
                            kill -KILL "$pid" 2>/dev/null || true
                        fi
                    fi
                done
                sleep 2
                # Re-check remaining PIDs
                remaining_pids=$(ps aux | grep -E "$pattern" | grep -v grep | awk '{print $2}' | tr '\n' ' ')
                if [ -z "$remaining_pids" ]; then
                    break
                fi
            done
        fi
        
        # Final check - filter out zombie processes
        local final_pids=""
        local all_final_pids=$(ps aux | grep -E "$pattern" | grep -v grep | awk '{print $2}' | tr '\n' ' ')
        for pid in $all_final_pids; do
            if ps -p "$pid" > /dev/null 2>&1; then
                local state=$(ps -p "$pid" -o state= 2>/dev/null | tr -d ' ')
                if [ "$state" != "Z" ]; then  # Not a zombie, it's still running
                    final_pids="$final_pids $pid"
                fi
            fi
        done
        final_pids=$(echo "$final_pids" | tr ' ' '\n' | grep -v '^$' | tr '\n' ' ')
        
        if [ -n "$final_pids" ]; then
            echo -e "${RED}❌ Failed to kill some $description processes: $final_pids${NC}"
            echo -e "${YELLOW}💡 Try manually: kill -9 $final_pids${NC}"
            return 1
        else
            echo -e "${GREEN}✅ Successfully stopped $description${NC}"
            return 0
        fi
    else
        echo -e "${BLUE}ℹ️  No $description processes found${NC}"
        return 0
    fi
}

# Stop scheduled chart update services (launchd)
echo -e "${BLUE}📅 Checking for scheduled chart update services...${NC}"
CHART_SERVICES=("com.hunter.update_charts" "com.hunter.sync_chart_data")
for service in "${CHART_SERVICES[@]}"; do
    if launchctl list "$service" >/dev/null 2>&1; then
        echo -e "${YELLOW}🔍 Found launchd service: $service${NC}"
        launchctl unload "$HOME/Library/LaunchAgents/${service}.plist" 2>/dev/null || true
        launchctl remove "$service" 2>/dev/null || true
        if launchctl list "$service" >/dev/null 2>&1; then
            echo -e "${RED}❌ Failed to stop $service${NC}"
        else
            echo -e "${GREEN}✅ Stopped $service${NC}"
        fi
    fi
done

# Stop cron chart update job
echo -e "${BLUE}⏰ Checking for cron chart update jobs...${NC}"
if crontab -l 2>/dev/null | grep -q "cron_chart_update.sh"; then
    echo -e "${YELLOW}🔍 Found cron chart update job${NC}"
    echo -e "${BLUE}ℹ️  Cron job found but not removed (run manually to remove):${NC}"
    echo -e "${YELLOW}   crontab -l | grep -v cron_chart_update.sh | crontab -${NC}"
    echo -e "${BLUE}   Or edit with: crontab -e${NC}"
else
    echo -e "${BLUE}ℹ️  No cron chart update jobs found${NC}"
fi

# Stop screen sessions first
if command -v screen &> /dev/null; then
    echo -e "${BLUE}📺 Checking for screen sessions...${NC}"
    if screen -list | grep -q "trading_bot"; then
        echo -e "${YELLOW}🔍 Found trading_bot screen session${NC}"
        screen -X -S trading_bot quit
        sleep 1
        
        if screen -list | grep -q "trading_bot"; then
            echo -e "${RED}❌ Failed to stop screen session${NC}"
        else
            echo -e "${GREEN}✅ Screen session stopped${NC}"
        fi
    else
        echo -e "${BLUE}ℹ️  No trading_bot screen session found${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  screen command not available${NC}"
fi

# Kill core python entrypoints
kill_processes "python.*main\.py" "Python trading bot"

# Kill detached monitor/watchdog processes
kill_processes "monitor_position\.py" "Position monitor"
kill_processes "monitor_watchdog\.py" "Monitor watchdog"

# Kill price tracker process
kill_processes "track_minute_prices\.py" "Minute price tracker"

# Kill async trading executors/workers
kill_processes "enhanced_async_trading\.py" "Enhanced async trading engine"
kill_processes "multi_chain_executor\.py" "Multi-chain executor"

# Catch-all: any hunter/trading named helpers
kill_processes ".*hunter.*" "Hunter-related processes"
kill_processes ".*trading.*" "Trading-related helper processes"

# Kill any processes writing to our log files
echo -e "${BLUE}📝 Checking for processes writing to log files...${NC}"
if command -v lsof &> /dev/null; then
    # Check practical_sustainable.log
    log_pids=$(lsof practical_sustainable.log 2>/dev/null | awk 'NR>1 {print $2}' | tr '\n' ' ')
    if [ -n "$log_pids" ]; then
        echo -e "${YELLOW}🔍 Found processes writing to practical_sustainable.log: $log_pids${NC}"
        kill -TERM $log_pids 2>/dev/null || true
        sleep 1
        kill -KILL $log_pids 2>/dev/null || true
    fi
    
    # Check hunter.log
    hunter_log_pids=$(lsof logs/hunter.log 2>/dev/null | awk 'NR>1 {print $2}' | tr '\n' ' ')
    if [ -n "$hunter_log_pids" ]; then
        echo -e "${YELLOW}🔍 Found processes writing to hunter.log: $hunter_log_pids${NC}"
        kill -TERM $hunter_log_pids 2>/dev/null || true
        sleep 1
        kill -KILL $hunter_log_pids 2>/dev/null || true
    fi
else
    echo -e "${YELLOW}⚠️  lsof command not available, skipping log file check${NC}"
fi

# Final verification
echo -e "${BLUE}🔍 Final verification...${NC}"
remaining_processes=$(ps aux | grep -E "(python.*main\.py|monitor_position\.py|monitor_watchdog\.py|track_minute_prices\.py|enhanced_async_trading\.py|multi_chain_executor\.py|.*hunter.*|.*trading.*)" | grep -v grep | wc -l)

if [ "$remaining_processes" -eq 0 ]; then
    echo -e "${GREEN}✅ All trading bot processes stopped successfully!${NC}"
    
    # Clean up monitor lock/heartbeat artifacts
    lock_file="data/.monitor_lock"
    heartbeat_file="data/.monitor_heartbeat"
    if [ -f "$lock_file" ] || [ -f "$heartbeat_file" ]; then
        echo -e "${BLUE}🧹 Removing monitor lock/heartbeat files...${NC}"
        rm -f "$lock_file" "$heartbeat_file"
    fi
    
    # Send Telegram notification
    echo -e "${BLUE}📨 Sending Telegram notification...${NC}"
    if command -v python3 &> /dev/null; then
        # Get the directory where this script is located
        SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
        # Change to the parent directory where main.py is located
        cd "$SCRIPT_DIR/.."
        
        python3 << 'PYTHON_SCRIPT' 2>/dev/null || echo -e "${YELLOW}⚠️  Could not send Telegram notification (Python/telegram_bot not available)${NC}"
import sys
import os

# We're already in the project root directory (cd'd there above)
project_root = os.getcwd()
sys.path.insert(0, project_root)

try:
    from src.monitoring.telegram_bot import send_telegram_message
    result = send_telegram_message(
        '🛑 Sustainable Trading Bot Stopped\n\nBot has been manually stopped via stop_bot.sh\n\nStatus: All processes terminated successfully',
        deduplicate=False,
        message_type="status",
        async_mode=False
    )
    if result:
        print('✅ Telegram notification sent successfully')
    else:
        print('⚠️  Failed to send Telegram notification')
except Exception as e:
    print(f'❌ Error sending Telegram notification: {e}')
    sys.exit(0)  # Don't fail the script if Telegram fails
PYTHON_SCRIPT
    else
        echo -e "${YELLOW}⚠️  Python3 not available, skipping Telegram notification${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}📍 Commands:${NC}"
    echo -e "  Restart bot: ${YELLOW}./launch_bot.sh${NC}"
    echo -e "  View logs:   ${YELLOW}tail -f practical_sustainable.log${NC}"
    echo -e "  Check status: ${YELLOW}ps aux | grep python${NC}"
    echo -e "  Check screen: ${YELLOW}screen -list${NC}"
    exit 0
else
    echo -e "${RED}❌ $remaining_processes trading bot processes still running${NC}"
    echo -e "${YELLOW}Remaining processes:${NC}"
    ps aux | grep -E "(python.*main\.py|monitor_position\.py|monitor_watchdog\.py|track_minute_prices\.py|enhanced_async_trading\.py|multi_chain_executor\.py|.*hunter.*|.*trading.*)" | grep -v grep
    
    # Try one more aggressive kill attempt
    echo -e "${YELLOW}🔄 Attempting final aggressive kill...${NC}"
    remaining_pids=$(ps aux | grep -E "(python.*main\.py|monitor_position\.py|monitor_watchdog\.py|track_minute_prices\.py|enhanced_async_trading\.py|multi_chain_executor\.py|.*hunter.*|.*trading.*)" | grep -v grep | awk '{print $2}' | tr '\n' ' ')
    if [ -n "$remaining_pids" ]; then
        for pid in $remaining_pids; do
            if ps -p "$pid" > /dev/null 2>&1; then
                state=$(ps -p "$pid" -o state= 2>/dev/null | tr -d ' ')
                if [ "$state" != "Z" ]; then  # Not a zombie
                    echo -e "${YELLOW}   Killing PID $pid (state: $state)...${NC}"
                    kill -9 "$pid" 2>/dev/null || true
                fi
            fi
        done
        sleep 2
        
        # Final check after aggressive kill
        final_remaining=$(ps aux | grep -E "(python.*main\.py|monitor_position\.py|monitor_watchdog\.py|track_minute_prices\.py|enhanced_async_trading\.py|multi_chain_executor\.py|.*hunter.*|.*trading.*)" | grep -v grep | wc -l)
        if [ "$final_remaining" -eq 0 ]; then
            echo -e "${GREEN}✅ All processes killed after aggressive attempt${NC}"
            exit 0
        fi
    fi
    
    echo -e "${YELLOW}Try manually: kill -9 \$(ps aux | grep -E '(python.*main\.py|monitor_position\.py)' | grep -v grep | awk '{print \$2}')${NC}"
    exit 1
fi
