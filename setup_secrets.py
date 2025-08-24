#!/usr/bin/env python3
"""
Setup script for secure secrets management
Migrates from .env files to secure backends
"""

import os
import sys
from secrets_manager import setup_secrets_interactive, get_secrets_manager

def migrate_from_env():
    """Migrate secrets from .env file to secure backend"""
    print("🔄 Migrating from .env to secure secrets backend")
    print("=" * 50)
    
    # Check if .env exists
    if not os.path.exists(".env"):
        print("❌ No .env file found")
        return False
    
    # Load .env file
    from dotenv import load_dotenv
    load_dotenv()
    
    # Collect secrets from environment
    secrets = {}
    env_vars = [
        "WALLET_ADDRESS", "PRIVATE_KEY", "INFURA_URL", 
        "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "SOLANA_RPC_URL",
        "SOLANA_WALLET_ADDRESS", "SOLANA_PRIVATE_KEY", "BASE_RPC_URL"
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            secrets[var] = value
    
    if not secrets:
        print("❌ No secrets found in .env file")
        return False
    
    print(f"📋 Found {len(secrets)} secrets in .env file")
    
    # Ask user for backend preference
    print("\nSelect secrets backend:")
    print("1. AWS Secrets Manager (recommended for production)")
    print("2. Environment variables")
    print("3. Encrypted local files")
    
    choice = input("Enter choice (1-3): ").strip()
    
    backends = {
        "1": "aws",
        "2": "env", 
        "3": "local"
    }
    
    backend = backends.get(choice, "aws")
    
    # Store secrets
    manager = get_secrets_manager(backend=backend)
    
    if backend == "aws":
        success = manager.set_secret("trading_bot_secrets", secrets)
    elif backend == "env":
        print("\n📋 Add these environment variables to your shell:")
        for key, value in secrets.items():
            print(f"export TRADING_BOT_SECRETS_{key}='{value}'")
        success = True
    else:  # local
        success = manager.set_secret("trading_bot_secrets", secrets)
    
    if success:
        print(f"\n✅ Successfully migrated secrets to {backend} backend")
        
        # Ask if user wants to remove .env file
        if backend != "env":  # Don't remove if using env backend
            remove_env = input("\n🗑️ Remove .env file? (y/N): ").strip().lower()
            if remove_env == 'y':
                try:
                    os.remove(".env")
                    print("✅ .env file removed")
                except Exception as e:
                    print(f"⚠️ Could not remove .env file: {e}")
        
        return True
    else:
        print("\n❌ Failed to migrate secrets")
        return False

def main():
    """Main setup function"""
    print("🔐 Trading Bot Secrets Setup")
    print("=" * 40)
    
    if len(sys.argv) > 1 and sys.argv[1] == "migrate":
        # Migration mode
        return migrate_from_env()
    else:
        # Interactive setup
        return setup_secrets_interactive()

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Test the bot: python main.py")
        print("2. Check secrets are working: python -c 'from secrets import validate_secrets; validate_secrets()'")
    else:
        print("\n❌ Setup failed")
        sys.exit(1)
