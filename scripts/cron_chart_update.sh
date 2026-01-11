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
