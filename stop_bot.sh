#!/bin/bash

# Trading Bot Stopper
# Gracefully stops the bot running in a screen session

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if screen is installed
if ! command -v screen &> /dev/null; then
    echo -e "${RED}❌ screen is not installed${NC}"
    exit 1
fi

# Check if the trading_bot session exists
if ! screen -list | grep -q "trading_bot"; then
    echo -e "${YELLOW}⚠️  No trading_bot session found${NC}"
    echo -e "${BLUE}📍 The bot is not currently running.${NC}"
    exit 0
fi

# Stop the bot
echo -e "${BLUE}🛑 Stopping trading bot...${NC}"
screen -X -S trading_bot quit

# Give it a moment to terminate
sleep 1

# Verify it was stopped
if screen -list | grep -q "trading_bot"; then
    echo -e "${RED}❌ Failed to stop trading bot${NC}"
    echo -e "${YELLOW}Try manually: screen -X -S trading_bot quit${NC}"
    exit 1
else
    echo -e "${GREEN}✅ Trading bot stopped successfully!${NC}"
    echo ""
    echo -e "${BLUE}📍 Commands:${NC}"
    echo -e "  Restart bot: ${YELLOW}./launch_bot.sh${NC}"
    echo -e "  View logs:   ${YELLOW}tail -f practical_sustainable.log${NC}"
    echo -e "  Check status: ${YELLOW}screen -list${NC}"
fi
