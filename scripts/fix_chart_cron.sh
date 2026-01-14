#!/bin/bash
# Fix and reload the chart update launchd service

PLIST_PATH="$HOME/Library/LaunchAgents/com.hunter.update_charts.plist"
PROJECT_ROOT="/Users/gianf/Hunter"

echo "🔧 Fixing chart update cron job..."
echo ""

# Check if plist exists
if [ ! -f "$PLIST_PATH" ]; then
    echo "❌ Plist file not found at $PLIST_PATH"
    echo "   Run: scripts/setup_chart_update_cron.sh first"
    exit 1
fi

# Unload service if it exists (ignore errors)
echo "📋 Unloading existing service (if any)..."
launchctl unload "$PLIST_PATH" 2>/dev/null
launchctl remove com.hunter.update_charts 2>/dev/null

# Remove stale lock file
echo "🧹 Cleaning up stale lock files..."
rm -f "$PROJECT_ROOT/.update_charts.lock"

# Try to bootstrap (macOS 10.11+)
echo "🚀 Loading launchd service..."
if launchctl bootstrap gui/$(id -u) "$PLIST_PATH" 2>&1; then
    echo "✅ Service bootstrapped successfully"
else
    # Fallback to load (older macOS)
    echo "⚠️  Bootstrap failed, trying load method..."
    if launchctl load "$PLIST_PATH" 2>&1; then
        echo "✅ Service loaded successfully"
    else
        echo "❌ Failed to load service"
        echo ""
        echo "Try running manually:"
        echo "  launchctl load $PLIST_PATH"
        exit 1
    fi
fi

# Verify it's loaded
echo ""
echo "📊 Verifying service status..."
if launchctl list com.hunter.update_charts >/dev/null 2>&1; then
    echo "✅ Service is loaded and active"
    launchctl list com.hunter.update_charts
else
    echo "⚠️  Service may not be loaded properly"
    echo "   Check logs: tail -f $PROJECT_ROOT/logs/update_charts.log"
fi

echo ""
echo "📝 To check service status:"
echo "   launchctl list com.hunter.update_charts"
echo ""
echo "📝 To view logs:"
echo "   tail -f $PROJECT_ROOT/logs/update_charts.log"
echo ""
echo "📝 To manually trigger:"
echo "   $PROJECT_ROOT/scripts/update_charts.sh"
echo ""
echo "📝 To unload service:"
echo "   launchctl unload $PLIST_PATH"
