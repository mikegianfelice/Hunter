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
    # Always remove our lock file on exit
    if [ -f "$SYNC_LOCK_FILE" ]; then
        rm -f "$SYNC_LOCK_FILE"
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

# Prevent multiple sync processes from running simultaneously (strong lock)
# Prefer flock if available; otherwise fall back to atomic noclobber lockfile.
if command -v flock >/dev/null 2>&1; then
    # Open lockfile without truncating it, then acquire an exclusive lock (non-blocking)
    exec 200>>"$SYNC_LOCK_FILE"
    if ! flock -n 200; then
        echo "⏳ Another sync process is already running (lock held)"
        exit 0
    fi
    # Record PID for debugging (optional)
    echo "$$" 1>&200
else
    # Fallback: atomic lock via noclobber
    if ! ( set -o noclobber; echo "$$" > "$SYNC_LOCK_FILE" ) 2> /dev/null; then
        # Lock file already exists – try to detect if it's stale
        LOCK_PID=$(cat "$SYNC_LOCK_FILE" 2>/dev/null || echo "")
        if [ -n "$LOCK_PID" ] && ps -p "$LOCK_PID" > /dev/null 2>&1; then
            echo "⏳ Another sync process is already running (PID: $LOCK_PID)"
            exit 0
        fi

        echo "⚠️  Removing stale lock file"
        rm -f "$SYNC_LOCK_FILE"

        # Try once more to acquire the lock
        if ! ( set -o noclobber; echo "$$" > "$SYNC_LOCK_FILE" ) 2> /dev/null; then
            echo "⏳ Could not acquire sync lock; another process started concurrently"
            exit 0
        fi
    fi
fi

# Log script execution start
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
echo "[$TIMESTAMP] Starting sync..."
echo "Sync process started with PID $$"

# Configure GitHub authentication using token from .env file
echo "🔐 Configuring GitHub authentication..."
GITHUB_TOKEN=""
if command -v python3 &> /dev/null; then
    # Try to load token from .env files (check both system/.env and .env)
    GITHUB_TOKEN=$(python3 -c "
import os
import sys
try:
    from dotenv import load_dotenv
    # Try system/.env first, then .env in project root
    load_dotenv('$PROJECT_ROOT/system/.env')
    load_dotenv('$PROJECT_ROOT/.env')
    token = os.getenv('GITHUB_SSH_KEY', '') or os.getenv('GITHUB_TOKEN', '')
    if token:
        print(token)
except ImportError:
    # Fallback: try to read .env file manually if python-dotenv not available
    import re
    for env_file in ['$PROJECT_ROOT/system/.env', '$PROJECT_ROOT/.env']:
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    match = re.match(r'^\s*GITHUB_SSH_KEY\s*=\s*(.+)$', line)
                    if match:
                        print(match.group(1).strip().strip('\"').strip(\"'\"))
                        break
                    match = re.match(r'^\s*GITHUB_TOKEN\s*=\s*(.+)$', line)
                    if match:
                        print(match.group(1).strip().strip('\"').strip(\"'\"))
                        break
        except FileNotFoundError:
            continue
except Exception as e:
    pass
" 2>/dev/null)
fi

if [ -n "$GITHUB_TOKEN" ]; then
    echo "✅ GitHub token found, configuring remote URL..."
    git remote set-url origin "https://${GITHUB_TOKEN}@github.com/mikegianfelice/Hunter.git" 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ Remote URL configured with token authentication"
    else
        echo "⚠️  Warning: Failed to set remote URL with token, will try without"
    fi
else
    echo "⚠️  Warning: No GitHub token found in .env files (GITHUB_SSH_KEY or GITHUB_TOKEN)"
    echo "   Will attempt push with existing git credentials"
fi

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

# Fetch latest changes from remote (with improved error logging)
echo "📥 Fetching latest changes from remote..."
FETCH_OUTPUT=$(git fetch origin main 2>&1)
FETCH_EXIT_CODE=$?
if [ $FETCH_EXIT_CODE -ne 0 ]; then
    echo "⚠️  Git fetch had issues: $FETCH_OUTPUT"
    # Continue anyway - might be network issue or auth issue
fi

# Check if there are local changes to commit (both staged and unstaged)
HAS_UNCOMMITTED=false
if ! git diff --quiet data/performance_data.json data/trade_log.csv data/open_positions.json 2>/dev/null || \
   ! git diff --cached --quiet data/performance_data.json data/trade_log.csv data/open_positions.json 2>/dev/null; then
    HAS_UNCOMMITTED=true
fi

# Check if we're behind or diverged from remote and sync FIRST (before checking for changes)
LOCAL=$(git rev-parse HEAD 2>/dev/null)
REMOTE=$(git rev-parse origin/main 2>/dev/null 2>&1)

if [ -n "$REMOTE" ] && [ "$LOCAL" != "$REMOTE" ]; then
    # Check if we're behind or diverged
    if git merge-base --is-ancestor "$LOCAL" "$REMOTE" 2>/dev/null; then
        # We're behind - just pull
        echo "📥 Pulling latest changes (we're behind remote)..."
        PULL_OUTPUT=$(git pull origin main 2>&1)
        PULL_EXIT_CODE=$?
        if [ $PULL_EXIT_CODE -ne 0 ]; then
            echo "⚠️  Failed to pull changes: $PULL_OUTPUT"
            echo "   Trying rebase..."
            REBASE_OUTPUT=$(git pull --rebase origin main 2>&1)
            REBASE_EXIT_CODE=$?
            if [ $REBASE_EXIT_CODE -ne 0 ]; then
                echo "⚠️  Rebase also failed: $REBASE_OUTPUT"
            else
                echo "✅ Successfully rebased"
            fi
        else
            echo "✅ Successfully pulled latest changes"
        fi
    elif git merge-base --is-ancestor "$REMOTE" "$LOCAL" 2>/dev/null; then
        # We're ahead - can push directly (but will handle below)
        echo "✅ Local branch is ahead of remote"
    else
        # Branches have diverged - need to rebase
        echo "🔄 Branches have diverged, rebasing local commits..."
        REBASE_OUTPUT=$(git pull --rebase origin main 2>&1)
        REBASE_EXIT_CODE=$?
        if [ $REBASE_EXIT_CODE -ne 0 ]; then
            echo "⚠️  Rebase failed: $REBASE_OUTPUT"
            echo "   Attempting merge..."
            MERGE_OUTPUT=$(git pull --no-rebase origin main 2>&1)
            MERGE_EXIT_CODE=$?
            if [ $MERGE_EXIT_CODE -ne 0 ]; then
                echo "❌ Error: Failed to sync with remote branch: $MERGE_OUTPUT" >&2
                rm -f "$SYNC_LOCK_FILE"
                exit 1
            else
                echo "✅ Successfully merged remote changes"
            fi
        else
            echo "✅ Successfully rebased"
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
    PUSH_OUTPUT=$(git push origin main 2>&1)
    PUSH_EXIT_CODE=$?
    if [ $PUSH_EXIT_CODE -ne 0 ]; then
        echo "❌ Error: Failed to push to remote" >&2
        echo "   Error details: $PUSH_OUTPUT" >&2
        echo "   This might be due to authentication issues or network problems" >&2
        if [ -z "$GITHUB_TOKEN" ]; then
            echo "   💡 Tip: Add GITHUB_SSH_KEY or GITHUB_TOKEN to .env file for automated authentication" >&2
        fi
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

