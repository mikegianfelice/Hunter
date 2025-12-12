#!/bin/bash
# Setup cron job to periodically sync performance data to GitHub
# This ensures files are committed/pushed regularly even when bot isn't trading

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SYNC_SCRIPT="$PROJECT_ROOT/scripts/sync_chart_data.sh"

# Make sure sync script is executable
chmod +x "$SYNC_SCRIPT"

# Get current crontab
CURRENT_CRON=$(crontab -l 2>/dev/null || echo "")

# Check if cron job already exists
if echo "$CURRENT_CRON" | grep -q "sync_chart_data.sh"; then
    echo "⚠️  Cron job for sync_chart_data.sh already exists!"
    echo ""
    echo "Current crontab:"
    crontab -l | grep -A 2 -B 2 "sync_chart_data.sh"
    echo ""
    read -p "Do you want to replace it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled. Existing cron job left unchanged."
        exit 0
    fi
    # Remove existing entry
    crontab -l 2>/dev/null | grep -v "sync_chart_data.sh" | crontab -
fi

# Default: sync every hour at minute 0 (e.g., 1:00, 2:00, 3:00...)
# You can change this frequency by modifying the cron expression
# Examples:
#   "0 * * * *"  - Every hour
#   "*/30 * * * *" - Every 30 minutes
#   "0 */2 * * *" - Every 2 hours
#   "0 9,17 * * *" - At 9 AM and 5 PM daily

CRON_SCHEDULE="0 * * * *"  # Every hour
CRON_COMMAND="cd $PROJECT_ROOT && $SYNC_SCRIPT >> $PROJECT_ROOT/logs/sync_chart_data.log 2>&1"

# Add new cron job
(crontab -l 2>/dev/null; echo "# Auto-sync performance data to GitHub - added $(date)"; echo "$CRON_SCHEDULE $CRON_COMMAND") | crontab -

echo "✅ Cron job installed successfully!"
echo ""
echo "Schedule: Every hour at minute 0"
echo "Command: $CRON_COMMAND"
echo ""
echo "To view your crontab: crontab -l"
echo "To remove this cron job: crontab -l | grep -v sync_chart_data.sh | crontab -"
echo ""
echo "The sync will run automatically in the background."

