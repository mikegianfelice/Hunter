#!/bin/bash
# Setup launchd job to periodically sync performance data to GitHub (macOS)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PLIST_SOURCE="$SCRIPT_DIR/com.hunter.sync_chart_data.plist"
LAUNCHD_DIR="$HOME/Library/LaunchAgents"
PLIST_DEST="$LAUNCHD_DIR/com.hunter.sync_chart_data.plist"

LABEL="com.hunter.sync_chart_data"
DOMAIN="gui/$(id -u)"

echo "📍 Project root: $PROJECT_ROOT"
echo "📄 Plist source: $PLIST_SOURCE"
echo "📄 Plist dest:   $PLIST_DEST"
echo ""

# Make sure scripts are executable
chmod +x "$PROJECT_ROOT/scripts/sync_chart_data.sh"
chmod +x "$PROJECT_ROOT/scripts/setup_launchd_sync.sh" || true

# Ensure logs dir exists (launchd won't create it)
mkdir -p "$PROJECT_ROOT/logs"

# Ensure LaunchAgents dir exists
mkdir -p "$LAUNCHD_DIR"

if [ ! -f "$PLIST_SOURCE" ]; then
  echo "❌ Missing plist at $PLIST_SOURCE"
  exit 1
fi

# Copy plist into LaunchAgents
echo "📋 Installing plist into LaunchAgents..."
cp "$PLIST_SOURCE" "$PLIST_DEST"

# Best-effort unload any existing job (ignore errors)
echo "🧹 Unloading any existing job (if present)..."
launchctl bootout "$DOMAIN" "$PLIST_DEST" 2>/dev/null || true

# Load job
echo "🚀 Bootstrapping job..."
launchctl bootstrap "$DOMAIN" "$PLIST_DEST"

echo "✅ Job installed!"
echo ""
echo "Schedule: every hour at minute 0 (and RunAtLoad=true)"
echo "Log file: $PROJECT_ROOT/logs/sync_chart_data.log"
echo ""
echo "Status:"
launchctl print "$DOMAIN/$LABEL" | head -n 40 || true
echo ""
echo "Tip: If you ever want to uninstall:"
echo "  launchctl bootout $DOMAIN $PLIST_DEST"
