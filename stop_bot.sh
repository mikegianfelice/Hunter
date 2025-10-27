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
        kill -TERM $pids 2>/dev/null || true
        
        # Wait a moment for graceful shutdown
        sleep 2
        
        # Check if processes are still running
        local remaining_pids=$(ps aux | grep -E "$pattern" | grep -v grep | awk '{print $2}' | tr '\n' ' ')
        
        if [ -n "$remaining_pids" ]; then
            echo -e "${YELLOW}⚠️  Some $description processes still running, forcing kill...${NC}"
            kill -KILL $remaining_pids 2>/dev/null || true
            sleep 1
        fi
        
        # Final check
        local final_pids=$(ps aux | grep -E "$pattern" | grep -v grep | awk '{print $2}' | tr '\n' ' ')
        if [ -n "$final_pids" ]; then
            echo -e "${RED}❌ Failed to kill some $description processes: $final_pids${NC}"
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

# Kill all Python processes related to trading
kill_processes "python.*main\.py" "Python trading bot"

# Kill any processes with 'hunter' in the name
kill_processes ".*hunter.*" "Hunter-related processes"

# Kill any processes with 'trading' in the name
kill_processes ".*trading.*" "Trading-related processes"

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
remaining_processes=$(ps aux | grep -E "(python.*main\.py|.*hunter.*|.*trading.*)" | grep -v grep | wc -l)

if [ "$remaining_processes" -eq 0 ]; then
    echo -e "${GREEN}✅ All trading bot processes stopped successfully!${NC}"
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
    ps aux | grep -E "(python.*main\.py|.*hunter.*|.*trading.*)" | grep -v grep
    echo -e "${YELLOW}Try manually: kill -9 \$(ps aux | grep 'python.*main.py' | grep -v grep | awk '{print \$2}')${NC}"
    exit 1
fi
