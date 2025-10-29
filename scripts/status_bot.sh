#!/bin/bash

# Trading Bot Status Checker with Live Logs
# Displays current status and shows live log output

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Function to show status
show_status() {
    echo -e "${CYAN}📊 Trading Bot Status${NC}"
    echo -e "${CYAN}═════════════════════${NC}"
    echo ""

    # Check for running Python processes
    python_processes=$(ps aux | grep -E "python.*main\.py" | grep -v grep | wc -l)
    hunter_processes=$(ps aux | grep -E ".*hunter.*" | grep -v grep | wc -l)
    trading_processes=$(ps aux | grep -E ".*trading.*" | grep -v grep | wc -l)

    # Check if screen is installed and has trading_bot session
    screen_running=false
    if command -v screen &> /dev/null; then
        if screen -list | grep -q "trading_bot"; then
            screen_running=true
        fi
    fi

    # Determine overall status
    if [ "$python_processes" -gt 0 ] || [ "$hunter_processes" -gt 0 ] || [ "$trading_processes" -gt 0 ] || [ "$screen_running" = true ]; then
        echo -e "${GREEN}✅ Bot Status: RUNNING${NC}"
        
        # Show process details
        echo -e "${BLUE}📊 Process Details:${NC}"
        if [ "$python_processes" -gt 0 ]; then
            echo -e "  Python processes: ${GREEN}$python_processes${NC}"
            ps aux | grep -E "python.*main\.py" | grep -v grep | awk '{print "    PID: " $2 " | " $11 " " $12 " " $13}'
        fi
        if [ "$hunter_processes" -gt 0 ]; then
            echo -e "  Hunter processes: ${GREEN}$hunter_processes${NC}"
        fi
        if [ "$trading_processes" -gt 0 ]; then
            echo -e "  Trading processes: ${GREEN}$trading_processes${NC}"
        fi
        if [ "$screen_running" = true ]; then
            echo -e "  Screen session: ${GREEN}Active${NC}"
        fi
        
        echo ""
        echo -e "${BLUE}📍 Quick Actions:${NC}"
        echo -e "  View bot:     ${YELLOW}screen -r trading_bot${NC}"
        echo -e "  Stop bot:     ${YELLOW}./stop_bot.sh${NC}"
        echo -e "  View logs:    ${YELLOW}tail -f practical_sustainable.log${NC}"
        echo -e "  Live logs:    ${YELLOW}./status_bot.sh --logs${NC}"
        
        if [ "$screen_running" = true ]; then
            echo ""
            echo -e "${BLUE}📊 Screen sessions:${NC}"
            screen -list | tail -n +2
        fi
    else
        echo -e "${RED}❌ Bot Status: NOT RUNNING${NC}"
        echo ""
        echo -e "${BLUE}📍 Quick Actions:${NC}"
        echo -e "  Start bot:    ${YELLOW}./launch_bot.sh${NC}"
        echo -e "  View logs:    ${YELLOW}tail -f practical_sustainable.log${NC}"
        echo -e "  Live logs:    ${YELLOW}./status_bot.sh --logs${NC}"
    fi

    echo ""
    echo -e "${CYAN}═════════════════════${NC}"
}

# Function to show live logs
show_live_logs() {
    echo -e "${PURPLE}📋 Live Trading Bot Logs${NC}"
    echo -e "${PURPLE}════════════════════════${NC}"
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to exit live logs${NC}"
    echo -e "${BLUE}Showing logs from: practical_sustainable.log${NC}"
    echo ""
    
    # Check if log file exists
    if [ ! -f "practical_sustainable.log" ]; then
        echo -e "${RED}❌ Log file not found: practical_sustainable.log${NC}"
        echo -e "${YELLOW}The bot may not have been started yet.${NC}"
        exit 1
    fi
    
    # Show last 20 lines first, then follow
    echo -e "${CYAN}📄 Last 20 log entries:${NC}"
    echo -e "${CYAN}───────────────────────${NC}"
    tail -n 20 practical_sustainable.log
    echo ""
    echo -e "${CYAN}📄 Live log feed (Ctrl+C to exit):${NC}"
    echo -e "${CYAN}──────────────────────────────────${NC}"
    
    # Follow the log file with color highlighting
    tail -f practical_sustainable.log | while read line; do
        # Color code different types of log entries
        if echo "$line" | grep -q "✅\|SUCCESS\|success"; then
            echo -e "${GREEN}$line${NC}"
        elif echo "$line" | grep -q "❌\|ERROR\|error\|FAILED\|failed"; then
            echo -e "${RED}$line${NC}"
        elif echo "$line" | grep -q "⚠️\|WARNING\|warning"; then
            echo -e "${YELLOW}$line${NC}"
        elif echo "$line" | grep -q "🔍\|INFO\|info\|Processing\|Executing"; then
            echo -e "${BLUE}$line${NC}"
        elif echo "$line" | grep -q "💰\|💸\|📊\|📈\|📉"; then
            echo -e "${PURPLE}$line${NC}"
        else
            echo "$line"
        fi
    done
}

# Function to show help
show_help() {
    echo -e "${CYAN}Trading Bot Status Checker${NC}"
    echo -e "${CYAN}══════════════════════════${NC}"
    echo ""
    echo -e "${BLUE}Usage:${NC}"
    echo -e "  ./status_bot.sh          Show current status"
    echo -e "  ./status_bot.sh --logs   Show live logs"
    echo -e "  ./status_bot.sh --help   Show this help"
    echo ""
    echo -e "${BLUE}Features:${NC}"
    echo -e "  • Comprehensive process detection"
    echo -e "  • Live log monitoring with color coding"
    echo -e "  • Screen session status"
    echo -e "  • Quick action commands"
    echo ""
}

# Main script logic
case "${1:-}" in
    --logs)
        show_live_logs
        ;;
    --help|-h)
        show_help
        ;;
    *)
        show_status
        ;;
esac
