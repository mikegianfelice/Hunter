#!/bin/bash

# Trading Bot Launcher
# Starts the bot in a detached screen session that survives terminal/screen closing

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
    echo "Install it with: brew install screen"
    exit 1
fi

# Kill existing screen session if it exists
if screen -list | grep -q "trading_bot"; then
    echo -e "${YELLOW}⚠️  Existing trading_bot session found, terminating...${NC}"
    screen -X -S trading_bot quit
    sleep 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create and run bot in a detached screen session
echo -e "${BLUE}🚀 Launching trading bot in screen session...${NC}"
screen -S trading_bot -d -m python3 main.py

# Give it a moment to start
sleep 1

# Verify the session was created
if screen -list | grep -q "trading_bot"; then
    echo -e "${GREEN}✅ Trading bot launched successfully!${NC}"
    echo ""
    echo -e "${BLUE}📍 Commands:${NC}"
    echo -e "  View bot:  ${YELLOW}screen -r trading_bot${NC}"
    echo -e "  Detach:    ${YELLOW}Ctrl+A then D${NC}"
    echo -e "  Stop bot:  ${YELLOW}screen -X -S trading_bot quit${NC}"
    echo -e "  List all:  ${YELLOW}screen -list${NC}"
    echo ""
    echo -e "${GREEN}🎯 Bot is now running in the background!${NC}"
else
    echo -e "${RED}❌ Failed to launch trading bot${NC}"
    exit 1
fi
