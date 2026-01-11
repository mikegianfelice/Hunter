#!/bin/bash
# Setup cron job for chart updates (runs every hour, same as launchd)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CRON_WRAPPER="$PROJECT_ROOT/scripts/cron_chart_update.sh"

# Create wrapper script for cron (cron needs full path and environment)
cat > "$CRON_WRAPPER" << 'WRAPPER_EOF'
#!/bin/bash
# Wrapper script for chart updates - ensures proper environment for cron

export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export MPLCONFIGDIR="/Users/gianf/Hunter/.matplotlib"

cd /Users/gianf/Hunter || exit 1

# Activate venv if it exists
if [ -f "/Users/gianf/Hunter/.venv/bin/activate" ]; then
    source /Users/gianf/Hunter/.venv/bin/activate
fi

# Run the actual update script
exec /Users/gianf/Hunter/scripts/update_charts.sh
WRAPPER_EOF

chmod +x "$CRON_WRAPPER"
echo "✅ Created cron wrapper script: $CRON_WRAPPER"

# Get current crontab
echo "📋 Current crontab:"
crontab -l 2>/dev/null || echo "  (no existing crontab)"

# Check if the cron job already exists
if crontab -l 2>/dev/null | grep -q "cron_chart_update.sh"; then
    echo "⚠️  Chart update cron job already exists!"
    echo ""
    echo "To update it, run:"
    echo "  crontab -e"
    echo ""
    echo "And update the line to:"
    echo "  0 * * * * $CRON_WRAPPER >> $PROJECT_ROOT/logs/cron_chart_update.log 2>&1"
else
    # Add the cron job
    (crontab -l 2>/dev/null; echo "0 * * * * $CRON_WRAPPER >> $PROJECT_ROOT/logs/cron_chart_update.log 2>&1") | crontab -
    echo "✅ Added cron job to run every hour at minute 0"
    echo ""
    echo "Cron job added:"
    echo "  0 * * * * $CRON_WRAPPER"
    echo ""
    echo "This will run every hour (same frequency as launchd)"
fi

echo ""
echo "📊 To verify, run: crontab -l"
echo "📝 To edit, run: crontab -e"
echo "🗑️  To remove, run: crontab -l | grep -v cron_chart_update.sh | crontab -"
