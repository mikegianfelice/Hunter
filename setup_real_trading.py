#!/usr/bin/env python3
"""
Setup script for real trading configuration
"""

import os
import sys

def create_env_template():
    """Create .env template file"""
    env_content = """# Trading Bot Secrets Configuration
# Fill in your actual values for real trading

# Ethereum Configuration
WALLET_ADDRESS=0x0000000000000000000000000000000000000000
PRIVATE_KEY=0x0000000000000000000000000000000000000000000000000000000000000000
INFURA_URL=https://mainnet.infura.io/v3/YOUR_INFURA_KEY

# Solana Configuration  
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
SOLANA_WALLET_ADDRESS=0000000000000000000000000000000000000000000000000000000000000000
SOLANA_PRIVATE_KEY=000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

# Base Chain Configuration
BASE_RPC_URL=https://mainnet.base.org

# Telegram Notifications (Optional)
TELEGRAM_BOT_TOKEN=YOUR_TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID=YOUR_TELEGRAM_CHAT_ID
"""
    
    # Create system directory if it doesn't exist
    os.makedirs("system", exist_ok=True)
    
    # Write .env template
    with open("system/.env.template", "w") as f:
        f.write(env_content)
    
    print("✅ Created system/.env.template")
    print("\n📋 Next steps:")
    print("1. Copy system/.env.template to system/.env")
    print("2. Fill in your actual wallet addresses and private keys")
    print("3. Get an Infura API key from https://infura.io")
    print("4. Run: python scripts/setup_secrets.py migrate")
    print("\n⚠️  WARNING: Never commit your .env file to git!")

def check_current_status():
    """Check current trading status"""
    print("🔍 Checking current trading configuration...")
    
    # Check if .env exists
    if os.path.exists("system/.env"):
        print("✅ Found system/.env file")
    else:
        print("❌ No system/.env file found")
    
    # Check if secrets are loaded
    try:
        from src.config.secrets import WALLET_ADDRESS, SOLANA_WALLET_ADDRESS
        if WALLET_ADDRESS and WALLET_ADDRESS != "0x0000000000000000000000000000000000000000":
            print("✅ Ethereum wallet configured")
        else:
            print("❌ Ethereum wallet not configured")
            
        if SOLANA_WALLET_ADDRESS and SOLANA_WALLET_ADDRESS != "0000000000000000000000000000000000000000000000000000000000000000":
            print("✅ Solana wallet configured")
        else:
            print("❌ Solana wallet not configured")
    except Exception as e:
        print(f"❌ Error loading secrets: {e}")

def main():
    print("🚀 Real Trading Setup")
    print("=" * 30)
    
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check_current_status()
    else:
        create_env_template()

if __name__ == "__main__":
    main()
