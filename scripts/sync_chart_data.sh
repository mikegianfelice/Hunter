#!/bin/bash
# Sync performance data files to repository for chart generation
# This script commits the latest performance data so GitHub Actions can generate charts

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Check if data files exist
if [ ! -f "data/performance_data.json" ]; then
    echo "❌ Error: data/performance_data.json not found"
    exit 1
fi

if [ ! -f "data/trade_log.csv" ]; then
    echo "❌ Error: data/trade_log.csv not found"
    exit 1
fi

# Check if there are changes
if git diff --quiet data/performance_data.json data/trade_log.csv; then
    echo "✅ No changes to sync - data files are up to date"
    exit 0
fi

# Stage the data files
echo "📊 Staging performance data files..."
git add data/performance_data.json data/trade_log.csv

# Commit with timestamp
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
git commit -m "📈 Sync performance data for chart generation [$TIMESTAMP]" || {
    echo "⚠️  No changes to commit"
    exit 0
}

# Push to remote
echo "🚀 Pushing to remote..."
git push

echo "✅ Performance data synced successfully!"
echo "   GitHub Actions workflow will use this data on next run"

