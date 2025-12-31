#!/bin/bash
# Generate and push performance charts to GitHub
# This script generates charts locally using Helius data and pushes them to GitHub

# Don't exit on error - we want to clean up locks even if something fails
set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
UPDATE_LOCK_FILE="$PROJECT_ROOT/.update_charts.lock"

# Cleanup function to remove lock file
cleanup() {
    if [ -f "$UPDATE_LOCK_FILE" ]; then
        rm -f "$UPDATE_LOCK_FILE"
    fi
}

trap cleanup EXIT

# Ensure we're in the project root
cd "$PROJECT_ROOT" || {
    echo "❌ Error: Could not change to project root: $PROJECT_ROOT" >&2
    exit 1
}

# Activate virtual environment if it exists
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
    echo "✅ Activated virtual environment"
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

# Prevent multiple update processes from running simultaneously
if command -v flock >/dev/null 2>&1; then
    exec 200>>"$UPDATE_LOCK_FILE"
    if ! flock -n 200; then
        echo "⏳ Another chart update process is already running (lock held)"
        exit 0
    fi
    echo "$$" 1>&200
else
    if ! ( set -o noclobber; echo "$$" > "$UPDATE_LOCK_FILE" ) 2> /dev/null; then
        LOCK_PID=$(cat "$UPDATE_LOCK_FILE" 2>/dev/null || echo "")
        if [ -n "$LOCK_PID" ] && ps -p "$LOCK_PID" > /dev/null 2>&1; then
            echo "⏳ Another chart update process is already running (PID: $LOCK_PID)"
            exit 0
        fi
        rm -f "$UPDATE_LOCK_FILE"
        if ! ( set -o noclobber; echo "$$" > "$UPDATE_LOCK_FILE" ) 2> /dev/null; then
            echo "⏳ Could not acquire update lock"
            exit 0
        fi
    fi
fi

# Log script execution
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
echo "[$TIMESTAMP] Starting chart update..."

# Configure GitHub authentication
echo "🔐 Configuring GitHub authentication..."
GITHUB_TOKEN=""
if command -v python3 &> /dev/null; then
    GITHUB_TOKEN=$(python3 -c "
try:
    from dotenv import load_dotenv
    import os
    load_dotenv('$PROJECT_ROOT/system/.env')
    load_dotenv('$PROJECT_ROOT/.env')
    token = os.getenv('GITHUB_TOKEN') or os.getenv('GITHUB_SSH_KEY')
    if token:
        print(token)
except ImportError:
    import re
    for env_file in ['$PROJECT_ROOT/system/.env', '$PROJECT_ROOT/.env']:
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    match = re.match(r'^\s*GITHUB_TOKEN\s*=\s*(.+)$', line)
                    if match:
                        print(match.group(1).strip().strip('\"').strip(\"'\"))
                        break
                    match = re.match(r'^\s*GITHUB_SSH_KEY\s*=\s*(.+)$', line)
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
else
    echo "⚠️  Warning: No GitHub token found, will use existing git credentials"
fi

# Create docs directory if it doesn't exist
mkdir -p docs

# Generate charts using Helius-based calculation
echo "📊 Generating performance charts..."

# Use python from venv if available, otherwise use system python3
PYTHON_CMD="python3"
if [ -f "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON_CMD="$PROJECT_ROOT/.venv/bin/python"
fi

# Generate 30-day chart
$PYTHON_CMD scripts/plot_wallet_value.py --days 30 --type percentage --save docs/performance_chart.png 2>&1
CHART30_EXIT=$?

# Generate 90-day chart
$PYTHON_CMD scripts/plot_wallet_value.py --days 90 --type percentage --save docs/performance_90d.png 2>&1
CHART90_EXIT=$?

# Generate 180-day chart
$PYTHON_CMD scripts/plot_wallet_value.py --days 180 --type percentage --save docs/performance_180d.png 2>&1
CHART180_EXIT=$?

# Check if any charts were generated successfully
if [ $CHART30_EXIT -ne 0 ] && [ $CHART90_EXIT -ne 0 ] && [ $CHART180_EXIT -ne 0 ]; then
    echo "⚠️  Warning: All chart generation attempts failed"
    # Don't exit - might still want to push if charts exist from previous run
fi

# Pull latest changes first (to avoid conflicts)
echo "📥 Fetching latest changes from remote..."
git fetch origin main 2>&1

LOCAL=$(git rev-parse HEAD 2>/dev/null)
REMOTE=$(git rev-parse origin/main 2>/dev/null 2>&1)

if [ -n "$REMOTE" ] && [ "$LOCAL" != "$REMOTE" ]; then
    if git merge-base --is-ancestor "$LOCAL" "$REMOTE" 2>/dev/null; then
        echo "📥 Pulling latest changes..."
        git pull origin main 2>&1 || git pull --rebase origin main 2>&1
    elif git merge-base --is-ancestor "$REMOTE" "$LOCAL" 2>/dev/null; then
        echo "✅ Local branch is ahead of remote"
    else
        echo "🔄 Branches have diverged, rebasing..."
        git pull --rebase origin main 2>&1 || git pull --no-rebase origin main 2>&1
    fi
fi

# Stage chart files
echo "📊 Staging chart files..."
git add docs/performance_chart.png docs/performance_90d.png docs/performance_180d.png 2>/dev/null

# Also stage cache file if it exists (helps with incremental updates)
if [ -f "data/helius_wallet_value_cache.json" ]; then
    git add data/helius_wallet_value_cache.json 2>/dev/null
fi

# Check if there are changes to commit
if git diff --cached --quiet; then
    echo "✅ No chart changes to commit"
    exit 0
fi

# Commit with timestamp
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
if git commit -m "📊 Update performance charts [$TIMESTAMP]" 2>&1; then
    echo "✅ Charts committed successfully"
else
    echo "⚠️  Failed to commit charts"
    exit 1
fi

# Push to remote
echo "🚀 Pushing charts to remote..."
PUSH_OUTPUT=$(git push origin main 2>&1)
PUSH_EXIT_CODE=$?

if [ $PUSH_EXIT_CODE -eq 0 ]; then
    echo "✅ Charts pushed successfully"
    exit 0
else
    echo "❌ Error: Failed to push charts" >&2
    echo "   Error details: $PUSH_OUTPUT" >&2
    exit 1
fi
