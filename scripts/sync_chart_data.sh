#!/bin/bash
# Sync performance data files to repository for chart generation
# This script commits the latest performance data so GitHub Actions can generate charts

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Ensure we're in the project root
cd "$PROJECT_ROOT" || {
    echo "❌ Error: Could not change to project root: $PROJECT_ROOT" >&2
    exit 1
}

# Check if data files exist
if [ ! -f "data/performance_data.json" ]; then
    echo "❌ Error: data/performance_data.json not found in $(pwd)" >&2
    exit 1
fi

if [ ! -f "data/trade_log.csv" ]; then
    echo "❌ Error: data/trade_log.csv not found in $(pwd)" >&2
    exit 1
fi

# Check if git is available
if ! command -v git &> /dev/null; then
    echo "❌ Error: git command not found" >&2
    exit 1
fi

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Error: Not in a git repository" >&2
    exit 1
fi

# Check if there are changes
if git diff --quiet data/performance_data.json data/trade_log.csv 2>/dev/null; then
    echo "✅ No changes to sync - data files are up to date"
    exit 0
fi

# Stage the data files
echo "📊 Staging performance data files..."
git add data/performance_data.json data/trade_log.csv

# Commit with timestamp
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
if ! git commit -m "📈 Sync performance data for chart generation [$TIMESTAMP]" 2>&1; then
    echo "⚠️  No changes to commit or commit failed"
    exit 0
fi

# Push to remote
echo "🚀 Pushing to remote..."
if ! git push 2>&1; then
    echo "❌ Error: Failed to push to remote" >&2
    echo "   This might be due to authentication issues or network problems" >&2
    exit 1
fi

echo "✅ Performance data synced successfully!"
echo "   GitHub Actions workflow will use this data on next run"

