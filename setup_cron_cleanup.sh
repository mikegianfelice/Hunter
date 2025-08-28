#!/bin/bash
# Setup script for automated smart blacklist cleanup

echo "🔧 Setting up automated smart blacklist cleanup..."

# Get the current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEANUP_SCRIPT="$SCRIPT_DIR/run_cleanup.py"

# Make sure the cleanup script is executable
chmod +x "$CLEANUP_SCRIPT"

echo "📁 Cleanup script location: $CLEANUP_SCRIPT"

# Create a log directory if it doesn't exist
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "📁 Log directory: $LOG_DIR"

# Function to add cron job
add_cron_job() {
    local schedule="$1"
    local description="$2"
    
    # Create the cron command
    local cron_cmd="$schedule cd $SCRIPT_DIR && python3 $CLEANUP_SCRIPT >> $LOG_DIR/cleanup.log 2>&1"
    
    # Check if cron job already exists
    if crontab -l 2>/dev/null | grep -q "$CLEANUP_SCRIPT"; then
        echo "⚠️  Cron job already exists for $description"
    else
        # Add to crontab
        (crontab -l 2>/dev/null; echo "$cron_cmd") | crontab -
        echo "✅ Added cron job for $description"
    fi
}

echo ""
echo "🕐 Available cleanup schedules:"
echo "1. Every 6 hours (recommended)"
echo "2. Every 12 hours"
echo "3. Daily at 2 AM"
echo "4. Twice daily (6 AM and 6 PM)"
echo "5. Custom schedule"

read -p "Choose schedule (1-5): " choice

case $choice in
    1)
        add_cron_job "0 */6 * * *" "every 6 hours"
        ;;
    2)
        add_cron_job "0 */12 * * *" "every 12 hours"
        ;;
    3)
        add_cron_job "0 2 * * *" "daily at 2 AM"
        ;;
    4)
        add_cron_job "0 6,18 * * *" "twice daily (6 AM and 6 PM)"
        ;;
    5)
        echo "Enter custom cron schedule (e.g., '0 */8 * * *' for every 8 hours):"
        read -p "Schedule: " custom_schedule
        add_cron_job "$custom_schedule" "custom schedule"
        ;;
    *)
        echo "❌ Invalid choice. Using default (every 6 hours)."
        add_cron_job "0 */6 * * *" "every 6 hours"
        ;;
esac

echo ""
echo "📋 Current cron jobs:"
crontab -l 2>/dev/null | grep "$CLEANUP_SCRIPT" || echo "No cleanup cron jobs found"

echo ""
echo "📊 Monitoring:"
echo "• Logs will be saved to: $LOG_DIR/cleanup.log"
echo "• To view recent logs: tail -f $LOG_DIR/cleanup.log"
echo "• To remove cron jobs: crontab -e"

echo ""
echo "✅ Setup complete! Smart blacklist cleanup will run automatically."
