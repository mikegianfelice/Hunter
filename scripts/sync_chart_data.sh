#!/bin/bash
# Sync performance data files to repository for chart generation
# This script commits the latest performance data so GitHub Actions can generate charts

# Don't exit on error - we want to clean up locks even if something fails
set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SYNC_LOCK_FILE="$PROJECT_ROOT/.sync_chart_data.lock"

# Cleanup function to remove lock file
cleanup() {
    if [ -f "$SYNC_LOCK_FILE" ]; then
        # Check if the lock is stale (older than 5 minutes)
        LOCK_AGE=$(($(date +%s) - $(stat -f %m "$SYNC_LOCK_FILE" 2>/dev/null || echo 0)))
        if [ "$LOCK_AGE" -gt 300 ]; then
            echo "⚠️  Removing stale lock file (age: ${LOCK_AGE}s)"
            rm -f "$SYNC_LOCK_FILE"
        fi
    fi
}

# Trap to ensure cleanup on exit
trap cleanup EXIT

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

# open_positions.json is optional - create empty if it doesn't exist
if [ ! -f "data/open_positions.json" ]; then
    echo "⚠️  data/open_positions.json not found, creating empty file..."
    echo "{}" > "data/open_positions.json"
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

# Prevent multiple sync processes from running simultaneously
if [ -f "$SYNC_LOCK_FILE" ]; then
    LOCK_PID=$(cat "$SYNC_LOCK_FILE" 2>/dev/null)
    # Check if the process is still running
    if ps -p "$LOCK_PID" > /dev/null 2>&1; then
        echo "⏳ Another sync process is already running (PID: $LOCK_PID)"
        exit 0
    else
        # Stale lock - remove it
        echo "⚠️  Removing stale lock file"
        rm -f "$SYNC_LOCK_FILE"
    fi
fi

# Create lock file
echo $$ > "$SYNC_LOCK_FILE"

# Log script execution start
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
echo "[$TIMESTAMP] Starting sync..."
echo "Sync process started with PID $$"

# Clean up stale git index lock if it exists
if [ -f ".git/index.lock" ]; then
    LOCK_AGE=$(($(date +%s) - $(stat -f %m ".git/index.lock" 2>/dev/null || echo 0)))
    if [ "$LOCK_AGE" -gt 300 ]; then
        echo "⚠️  Removing stale git index lock (age: ${LOCK_AGE}s)"
        rm -f ".git/index.lock"
    else
        echo "⏳ Git index is locked (another git process may be running)"
        rm -f "$SYNC_LOCK_FILE"
        exit 0
    fi
fi

# Fetch latest changes from remote
echo "📥 Fetching latest changes from remote..."
git fetch origin main > /dev/null 2>&1

# Check if there are local changes to commit (both staged and unstaged)
HAS_UNCOMMITTED=false
if ! git diff --quiet data/performance_data.json data/trade_log.csv data/open_positions.json 2>/dev/null || \
   ! git diff --cached --quiet data/performance_data.json data/trade_log.csv data/open_positions.json 2>/dev/null; then
    HAS_UNCOMMITTED=true
fi

# Check if we're behind or diverged from remote
LOCAL=$(git rev-parse HEAD 2>/dev/null)
REMOTE=$(git rev-parse origin/main 2>/dev/null 2>&1)

if [ -n "$REMOTE" ] && [ "$LOCAL" != "$REMOTE" ]; then
    # Check if we're behind or diverged
    if git merge-base --is-ancestor "$LOCAL" "$REMOTE" 2>/dev/null; then
        # We're behind - just pull
        echo "📥 Pulling latest changes (we're behind remote)..."
        if ! git pull origin main > /dev/null 2>&1; then
            echo "⚠️  Failed to pull changes, trying rebase..."
            git pull --rebase origin main > /dev/null 2>&1
        fi
    elif git merge-base --is-ancestor "$REMOTE" "$LOCAL" 2>/dev/null; then
        # We're ahead - can push directly (but will handle below)
        echo "✅ Local branch is ahead of remote"
    else
        # Branches have diverged - need to rebase
        echo "🔄 Branches have diverged, rebasing local commits..."
        if ! git pull --rebase origin main > /dev/null 2>&1; then
            echo "⚠️  Rebase failed, attempting merge..."
            git pull --no-rebase origin main > /dev/null 2>&1 || {
                echo "❌ Error: Failed to sync with remote branch" >&2
                rm -f "$SYNC_LOCK_FILE"
                exit 1
            }
        fi
    fi
fi

# If we have uncommitted changes, commit them
if [ "$HAS_UNCOMMITTED" = true ]; then
    # Stage the data files
    echo "📊 Staging performance data files..."
    git add data/performance_data.json data/trade_log.csv data/open_positions.json
    
    # Commit with timestamp
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    if ! git commit -m "📈 Sync performance data for chart generation [$TIMESTAMP]" 2>&1; then
        echo "⚠️  No changes to commit or commit failed"
        rm -f "$SYNC_LOCK_FILE"
        exit 0
    fi
else
    echo "✅ No uncommitted changes to data files"
fi

# Check if we have commits to push
LOCAL=$(git rev-parse HEAD 2>/dev/null)
REMOTE=$(git rev-parse origin/main 2>/dev/null 2>&1)

if [ -n "$REMOTE" ] && [ "$LOCAL" != "$REMOTE" ]; then
    # Push to remote
    echo "🚀 Pushing to remote..."
    if ! git push origin main 2>&1; then
        echo "❌ Error: Failed to push to remote" >&2
        echo "   This might be due to authentication issues or network problems" >&2
        rm -f "$SYNC_LOCK_FILE"
        exit 1
    fi
    echo "✅ Performance data synced successfully!"
    echo "   GitHub Actions workflow will use this data on next run"
else
    echo "✅ Everything is up to date - no push needed"
fi

# Remove lock file on success
rm -f "$SYNC_LOCK_FILE"
exit 0

