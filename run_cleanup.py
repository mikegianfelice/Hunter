#!/usr/bin/env python3
"""
Standalone script to run smart blacklist cleanup
Can be run manually or scheduled via cron
"""

import sys
import os
import time
from datetime import datetime

# Add current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print(f"🧹 Smart Blacklist Cleanup - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # Import and run smart blacklist cleanup
        from smart_blacklist_cleaner import clean_delisted_tokens
        
        print("🔍 Starting delisted token verification...")
        result = clean_delisted_tokens()
        
        if result:
            removed_count = result.get("removed_count", 0)
            remaining_count = result.get("remaining_count", 0)
            original_count = removed_count + remaining_count
            
            print(f"\n📊 Cleanup Summary:")
            print(f"  • Original delisted tokens: {original_count}")
            print(f"  • Tokens reactivated: {removed_count}")
            print(f"  • Still delisted: {remaining_count}")
            print(f"  • Cleanup ratio: {removed_count/original_count*100:.1f}%" if original_count > 0 else "  • Cleanup ratio: 0%")
            
            if removed_count > 0:
                print(f"\n✅ Successfully reactivated {removed_count} tokens!")
                print("🎯 Bot should now have more trading opportunities.")
            else:
                print(f"\nℹ️ No tokens needed reactivation.")
                
        else:
            print("❌ Cleanup failed!")
            return 1
            
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        return 1
    
    print(f"\n✅ Cleanup completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
