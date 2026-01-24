#!/bin/bash

# Price Tracker Launcher
# Starts the price tracker in a detached screen session that survives terminal/screen closing

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
if screen -list | grep -q "price_tracker"; then
    echo -e "${YELLOW}⚠️  Existing price_tracker session found, terminating...${NC}"
    screen -X -S price_tracker quit
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

# Create and run price tracker in a detached screen session
echo -e "${BLUE}📊 Launching price tracker in screen session...${NC}"

# Build launch command: activate venv if present, then run the price tracker unbuffered and log output
LAUNCH_CMD='cd "'"$SCRIPT_DIR/.."'"; \
  if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi; \
  export PYTHONUNBUFFERED=1; \
  python3 main.py price-tracker >> scripts/price_tracker.log 2>&1'

# Start within a login shell so venv activation works reliably
screen -S price_tracker -d -m bash -lc "$LAUNCH_CMD"

# Give it a moment to start
sleep 2

# Verify the session was created
if screen -list | grep -q "price_tracker"; then
    echo -e "${GREEN}✅ Price tracker launched successfully!${NC}"
    echo ""
    echo -e "${BLUE}📍 Commands:${NC}"
    echo -e "  View tracker:  ${YELLOW}screen -r price_tracker${NC}"
    echo -e "  Detach:        ${YELLOW}Ctrl+A then D${NC}"
    echo -e "  Stop tracker:  ${YELLOW}screen -X -S price_tracker quit${NC}"
    echo -e "  List all:      ${YELLOW}screen -list${NC}"
    echo ""
    echo -e "${GREEN}🎯 Price tracker is now running in the background!${NC}"
    echo -e "${BLUE}📊 Tracking tokens every 5 minutes (no trading)${NC}"
else
    echo -e "${RED}❌ Failed to launch price tracker${NC}"
    # Show recent log output to help diagnose fast-exit failures (e.g., missing venv deps)
    if [ -f scripts/price_tracker.log ]; then
        echo -e "${YELLOW}Last 40 log lines:${NC}"
        tail -n 40 scripts/price_tracker.log || true
    fi
    exit 1
fi
