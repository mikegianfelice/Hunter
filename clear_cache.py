#!/usr/bin/env python3
"""
Clear Python cache helper script
Use this during development when you want to ensure the bot uses the latest code
"""

import os
import shutil
import sys

def clear_python_cache():
    """Clear all Python cache files"""
    cleared_files = 0
    cleared_dirs = 0
    
    print("🧹 Clearing Python cache...")
    
    # Remove all .pyc files
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.pyc'):
                try:
                    os.remove(os.path.join(root, file))
                    cleared_files += 1
                except Exception as e:
                    print(f"⚠️ Could not remove {file}: {e}")
    
    # Remove all __pycache__ directories
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs:
            if dir_name == '__pycache__':
                try:
                    shutil.rmtree(os.path.join(root, dir_name))
                    cleared_dirs += 1
                except Exception as e:
                    print(f"⚠️ Could not remove {dir_name}: {e}")
    
    print(f"✅ Cache cleared: {cleared_files} .pyc files, {cleared_dirs} __pycache__ directories")
    return cleared_files + cleared_dirs

def check_cache_status():
    """Check if cache needs clearing"""
    pyc_files = 0
    pycache_dirs = 0
    
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.pyc'):
                pyc_files += 1
        for dir_name in dirs:
            if dir_name == '__pycache__':
                pycache_dirs += 1
    
    print(f"📊 Cache status: {pyc_files} .pyc files, {pycache_dirs} __pycache__ directories")
    return pyc_files + pycache_dirs

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check_cache_status()
    else:
        clear_python_cache()
        print("\n💡 Tip: Run 'python3 clear_cache.py check' to see cache status")
