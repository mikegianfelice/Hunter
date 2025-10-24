#!/bin/bash

# Trading Bot Status Checker
# Displays current status of the trading bot

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}📊 Trading Bot Status${NC}"
echo -e "${CYAN}═════════════════════${NC}"
echo ""

# Check if screen is installed
if ! command -v screen &> /dev/null; then
    echo -e "${RED}❌ screen is not installed${NC}"
    exit 1
fi

# Check if the trading_bot session exists
if screen -list | grep -q "trading_bot"; then
    echo -e "${GREEN}✅ Bot Status: RUNNING${NC}"
    echo ""
    echo -e "${BLUE}📍 Quick Actions:${NC}"
    echo -e "  View bot:     ${YELLOW}screen -r trading_bot${NC}"
    echo -e "  Stop bot:     ${YELLOW}./stop_bot.sh${NC}"
    echo -e "  View logs:    ${YELLOW}tail -f practical_sustainable.log${NC}"
    echo ""
    echo -e "${BLUE}📊 All screen sessions:${NC}"
    screen -list | tail -n +2
else
    echo -e "${RED}❌ Bot Status: NOT RUNNING${NC}"
    echo ""
    echo -e "${BLUE}📍 Quick Actions:${NC}"
    echo -e "  Start bot:    ${YELLOW}./launch_bot.sh${NC}"
    echo -e "  View logs:    ${YELLOW}tail -f practical_sustainable.log${NC}"
fi

echo ""
echo -e "${CYAN}═════════════════════${NC}"
