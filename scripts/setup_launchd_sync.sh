#!/bin/bash
# Setup launchd job to periodically sync performance data to GitHub
# This is more reliable than cron on macOS, especially when system sleeps

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PLIST_FILE="$SCRIPT_DIR/com.hunter.sync_chart_data.plist"
LAUNCHD_DIR="$HOME/Library/LaunchAgents"
LAUNCHD_PLIST="$LAUNCHD_DIR/com.hunter.sync_chart_data.plist"

# Make sure sync script is executable
chmod +x "$PROJECT_ROOT/scripts/sync_chart_data.sh"

# Create LaunchAgents directory if it doesn't exist
mkdir -p "$LAUNCHD_DIR"

# Check if launchd job already exists
if [ -f "$LAUNCHD_PLIST" ]; then
    echo "⚠️  Launchd job already exists!"
    echo ""
    echo "Current job: $LAUNCHD_PLIST"
    echo ""
    read -p "Do you want to replace it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled. Existing job left unchanged."
        exit 0
    fi
    # Unload existing job
    echo "🔄 Unloading existing job..."
    launchctl unload "$LAUNCHD_PLIST" 2>/dev/null || true
fi

# Copy plist to LaunchAgents directory
echo "📋 Installing launchd job..."
cp "$PLIST_FILE" "$LAUNCHD_PLIST"

# Load the job
echo "🚀 Loading launchd job..."
launchctl load "$LAUNCHD_PLIST"

echo "✅ Launchd job installed successfully!"
echo ""
echo "Schedule: Every hour at minute 0"
echo "Plist file: $LAUNCHD_PLIST"
echo ""
echo "To view job status: launchctl list | grep sync_chart_data"
echo "To unload job: launchctl unload $LAUNCHD_PLIST"
echo "To reload job: launchctl unload $LAUNCHD_PLIST && launchctl load $LAUNCHD_PLIST"
echo ""
echo "The sync will run automatically in the background, even after system sleep."

