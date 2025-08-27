#!/usr/bin/env python3
"""
Smart blacklist cleaner - selectively removes tokens based on risk analysis
"""

import json
import os
from datetime import datetime, timedelta

def analyze_blacklist():
    """Analyze the current blacklist and suggest safe removals"""
    
    if not os.path.exists("delisted_tokens.json"):
        print("❌ No delisted_tokens.json found")
        return
    
    with open("delisted_tokens.json", "r") as f:
        data = json.load(f)
    
    delisted_tokens = data.get("delisted_tokens", [])
    failure_counts = data.get("failure_counts", {})
    
    print(f"📊 Blacklist Analysis:")
    print(f"• Total delisted tokens: {len(delisted_tokens)}")
    print(f"• Tokens with failure counts: {len(failure_counts)}")
    
    # Categorize tokens
    ethereum_tokens = []
    solana_tokens = []
    high_failure_tokens = []
    
    for token in delisted_tokens:
        if token.startswith("0x"):
            ethereum_tokens.append(token)
        else:
            solana_tokens.append(token)
        
        if failure_counts.get(token, 0) >= 3:
            high_failure_tokens.append(token)
    
    print(f"• Ethereum tokens: {len(ethereum_tokens)}")
    print(f"• Solana tokens: {len(solana_tokens)}")
    print(f"• High failure tokens (≥3 failures): {len(high_failure_tokens)}")
    
    # Suggest safe removals
    safe_to_remove = []
    risky_to_keep = []
    
    for token in delisted_tokens:
        failure_count = failure_counts.get(token, 0)
        
        # Keep tokens with high failure counts
        if failure_count >= 3:
            risky_to_keep.append(token)
            continue
        
        # Keep tokens that caused 100% losses (from trade log)
        if token.lower() in [
            "0x2f97d022a31b07dd3d4187f9c0acedd5cc92246e",
            "0xd72ef7f9003b4d5da0f7d264408ab78ca47b17f0", 
            "0xca068090819d424affb9f51b35c96c949523367e",
            "0x47e323b5effbc7a16a288118f79fa6c723709fbb",
            "0xb80b2ee07b991f78cf324e23dd0304674314f4fa"
        ]:
            risky_to_keep.append(token)
            continue
        
        # Consider safe to remove: low failure count, no 100% loss history
        safe_to_remove.append(token)
    
    print(f"\n🔍 Recommendations:")
    print(f"• Safe to remove: {len(safe_to_remove)} tokens")
    print(f"• Risky to keep: {len(risky_to_keep)} tokens")
    
    return safe_to_remove, risky_to_keep

def selective_clean(safe_tokens_to_remove):
    """Selectively remove only the safer tokens from blacklist"""
    
    if not safe_tokens_to_remove:
        print("✅ No safe tokens to remove")
        return
    
    with open("delisted_tokens.json", "r") as f:
        data = json.load(f)
    
    delisted_tokens = data.get("delisted_tokens", [])
    failure_counts = data.get("failure_counts", {})
    
    # Remove safe tokens
    original_count = len(delisted_tokens)
    delisted_tokens = [t for t in delisted_tokens if t not in safe_tokens_to_remove]
    
    # Update data
    data["delisted_tokens"] = delisted_tokens
    data["removed_count"] = original_count - len(delisted_tokens)
    data["remaining_count"] = len(delisted_tokens)
    data["quick_cleaned_at"] = datetime.now().isoformat()
    
    # Save updated data
    with open("delisted_tokens.json", "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Removed {len(safe_tokens_to_remove)} safer tokens from blacklist")
    print(f"📊 Remaining: {len(delisted_tokens)} tokens")

def main():
    print("🧠 Smart Blacklist Cleaner")
    print("=" * 40)
    
    safe_to_remove, risky_to_keep = analyze_blacklist()
    
    if not safe_to_remove:
        print("\n❌ No safe tokens to remove. All blacklisted tokens appear to be high-risk.")
        return
    
    print(f"\n🔓 Safe tokens to remove (low risk):")
    for i, token in enumerate(safe_to_remove[:10]):  # Show first 10
        print(f"  {i+1}. {token}")
    if len(safe_to_remove) > 10:
        print(f"  ... and {len(safe_to_remove) - 10} more")
    
    print(f"\n⚠️  Risky tokens to keep:")
    for i, token in enumerate(risky_to_keep[:5]):  # Show first 5
        print(f"  {i+1}. {token}")
    if len(risky_to_keep) > 5:
        print(f"  ... and {len(risky_to_keep) - 5} more")
    
    response = input(f"\n🤔 Remove {len(safe_to_remove)} safer tokens? (y/N): ").strip().lower()
    
    if response == 'y':
        selective_clean(safe_to_remove)
        print("\n✅ Smart cleanup completed! Bot should now have more trading opportunities.")
    else:
        print("\n❌ Cleanup cancelled.")

if __name__ == "__main__":
    main()
