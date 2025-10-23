#!/usr/bin/env python3
"""
Git wrapper script that uses GitHub token from .env file
"""

import os
import subprocess
from dotenv import load_dotenv

def git_push():
    """Push changes to GitHub using token from .env"""
    # Load environment variables
    load_dotenv()
    
    github_token = os.getenv('GITHUB_SSH_KEY')  # Using the same env var name
    if not github_token:
        print("❌ GITHUB_SSH_KEY not found in .env file")
        return False
    
    try:
        # Change remote URL to use token authentication
        repo_url = f"https://{github_token}@github.com/mikegianfelice/Hunter.git"
        
        # Set the remote URL with token
        set_url_cmd = ['git', 'remote', 'set-url', 'origin', repo_url]
        result = subprocess.run(set_url_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Failed to set remote URL: {result.stderr}")
            return False
        
        print("✅ Remote URL updated with token authentication")
        
        # Push to GitHub
        push_cmd = ['git', 'push', 'origin', 'main']
        result = subprocess.run(push_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Successfully pushed to GitHub!")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"❌ Push failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error during push: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "push":
        git_push()
    else:
        print("Usage: python3 git_with_ssh.py push")
        print("This script will use the SSH key from your .env file to push to GitHub")
