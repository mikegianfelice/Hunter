#!/bin/bash

# Price Tracker Stopper
# Stops the price tracker screen session and related processes

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🛑 Stopping price tracker...${NC}"

# Stop screen session
if command -v screen &> /dev/null; then
    if screen -list | grep -q "price_tracker"; then
        echo -e "${YELLOW}🔍 Found price_tracker screen session${NC}"
        screen -X -S price_tracker quit
        sleep 1
        
        if screen -list | grep -q "price_tracker"; then
            echo -e "${RED}❌ Failed to stop screen session${NC}"
        else
            echo -e "${GREEN}✅ Screen session stopped${NC}"
        fi
    else
        echo -e "${BLUE}ℹ️  No price_tracker screen session found${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  screen command not available${NC}"
fi

# Kill any price tracker processes
pids=$(ps aux | grep -E "python.*main\.py.*price-tracker" | grep -v grep | awk '{print $2}' | tr '\n' ' ')

if [ -n "$pids" ]; then
    echo -e "${YELLOW}🔍 Found price tracker processes: $pids${NC}"
    for pid in $pids; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 2
    
    # Force kill if still running
    remaining_pids=$(ps aux | grep -E "python.*main\.py.*price-tracker" | grep -v grep | awk '{print $2}' | tr '\n' ' ')
    if [ -n "$remaining_pids" ]; then
        echo -e "${YELLOW}⚠️  Some processes still running, forcing kill...${NC}"
        for pid in $remaining_pids; do
            kill -KILL "$pid" 2>/dev/null || true
        done
    fi
    echo -e "${GREEN}✅ Price tracker processes stopped${NC}"
else
    echo -e "${BLUE}ℹ️  No price tracker processes found${NC}"
fi

# Final verification
remaining=$(ps aux | grep -E "python.*main\.py.*price-tracker" | grep -v grep | wc -l)

if [ "$remaining" -eq 0 ]; then
    echo -e "${GREEN}✅ Price tracker stopped successfully!${NC}"
    echo ""
    echo -e "${BLUE}📍 Commands:${NC}"
    echo -e "  Restart tracker: ${YELLOW}./launch_price_tracker.sh${NC}"
    echo -e "  View logs:       ${YELLOW}tail -f scripts/price_tracker.log${NC}"
    exit 0
else
    echo -e "${RED}❌ $remaining price tracker processes still running${NC}"
    echo -e "${YELLOW}Remaining processes:${NC}"
    ps aux | grep -E "python.*main\.py.*price-tracker" | grep -v grep
    exit 1
fi
