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
# Change to the parent directory where main.py is located
cd "$SCRIPT_DIR/.."

# Clear Python bytecode cache to ensure fresh imports
echo -e "${BLUE}🧹 Clearing Python cache...${NC}"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true

# Create and run bot in a detached screen session
echo -e "${BLUE}🚀 Launching trading bot in screen session...${NC}"

# Build launch command: activate venv if present, then run the bot unbuffered and log output
# Use python3 (venv will have it, or use system python3)
LAUNCH_CMD='cd "'"$SCRIPT_DIR/.."'"; \
  if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi; \
  export PYTHONUNBUFFERED=1; \
  python3 main.py trading >> scripts/practical_sustainable.log 2>&1'

# Start within a login shell so venv activation works reliably
screen -S trading_bot -d -m bash -lc "$LAUNCH_CMD"

# Give it a moment to start
sleep 2

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
    # Show recent log output to help diagnose fast-exit failures (e.g., missing venv deps)
    if [ -f scripts/practical_sustainable.log ]; then
        echo -e "${YELLOW}Last 40 log lines:${NC}"
        tail -n 40 scripts/practical_sustainable.log || true
    fi
    exit 1
fi
