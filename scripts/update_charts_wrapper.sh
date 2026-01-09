#!/bin/bash
# Wrapper script to align chart updates to hour boundaries
# This ensures the update runs exactly at :00 minutes (12:00, 1:00, 2:00, etc.)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
UPDATE_SCRIPT="$SCRIPT_DIR/update_charts.sh"

# Calculate seconds until next hour boundary
current_seconds=$(date +%s)
current_minute=$(date +%M)
current_second=$(date +%S)

# Calculate seconds into the current hour
seconds_into_hour=$((current_minute * 60 + current_second))

# Calculate seconds until next hour boundary
seconds_until_next_hour=$((3600 - seconds_into_hour))

# If we're already at the hour boundary (within 5 seconds), run immediately
# Otherwise, wait until the next hour
if [ $seconds_until_next_hour -lt 5 ]; then
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] At hour boundary, running chart update immediately..."
    exec "$UPDATE_SCRIPT"
else
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] Waiting $seconds_until_next_hour seconds until next hour boundary..."
    sleep $seconds_until_next_hour
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] Hour boundary reached, running chart update..."
    exec "$UPDATE_SCRIPT"
fi
