# secrets_manager.py
import os
import json
import base64
from typing import Dict, Optional, Any
from cryptography.fernet import Fernet
# Lazy import boto3/botocore only when AWS backend is used
# This prevents permission errors when AWS is not configured

class SecretsManager:
    """Secure secrets management with multiple backends"""
    
    def __init__(self, backend: str = "aws", region: str = "us-east-1"):
        self.backend = backend
        self.region = region
        self.fernet = None
        self._init_encryption()
    
    def _init_encryption(self):
        """Initialize encryption for local storage"""
        try:
            # Try to load existing key or create new one
            key_file = ".secret_key"
            if os.path.exists(key_file):
                with open(key_file, "rb") as f:
                    key = f.read()
            else:
                key = Fernet.generate_key()
                with open(key_file, "wb") as f:
                    f.write(key)
            self.fernet = Fernet(key)
        except Exception as e:
            print(f"⚠️ Encryption initialization failed: {e}")
            self.fernet = None
    
    def get_secret(self, secret_name: str) -> Optional[Dict[str, Any]]:
        """Get secret from configured backend"""
        if self.backend == "aws":
            return self._get_aws_secret(secret_name)
        elif self.backend == "env":
            return self._get_env_secret(secret_name)
        elif self.backend == "local":
            return self._get_local_secret(secret_name)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def set_secret(self, secret_name: str, secret_data: Dict[str, Any]) -> bool:
        """Set secret in configured backend"""
        if self.backend == "aws":
            return self._set_aws_secret(secret_name, secret_data)
        elif self.backend == "env":
            return self._set_env_secret(secret_name, secret_data)
        elif self.backend == "local":
            return self._set_local_secret(secret_name, secret_data)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _get_aws_secret(self, secret_name: str) -> Optional[Dict[str, Any]]:
        """Get secret from AWS Secrets Manager"""
        try:
            # Lazy import boto3 only when AWS backend is actually used
            # This prevents permission errors when AWS is not configured
            try:
                import boto3
                from botocore.exceptions import ClientError
            except (ImportError, PermissionError, OSError) as import_error:
                print(f"⚠️ Cannot import boto3/botocore (AWS not configured or permission denied): {import_error}")
                print(f"💡 Tip: Set SECRETS_BACKEND=env or SECRETS_BACKEND=local to use non-AWS backends")
                return None
            
            session = boto3.session.Session()
            client = session.client(
                service_name='secretsmanager',
                region_name=self.region
            )
            
            response = client.get_secret_value(SecretId=secret_name)
            if 'SecretString' in response:
                return json.loads(response['SecretString'])
            else:
                # Handle binary secrets
                decoded_binary_secret = base64.b64decode(response['SecretBinary'])
                return json.loads(decoded_binary_secret)
                
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ResourceNotFoundException':
                print(f"⚠️ Secret {secret_name} not found in AWS Secrets Manager")
            elif error_code == 'InvalidRequestException':
                print(f"⚠️ Invalid request for secret {secret_name}")
            elif error_code == 'InvalidParameterException':
                print(f"⚠️ Invalid parameter for secret {secret_name}")
            else:
                print(f"⚠️ AWS Secrets Manager error: {e}")
            return None
        except Exception as e:
            print(f"❌ Error accessing AWS Secrets Manager: {e}")
            return None
    
    def _set_aws_secret(self, secret_name: str, secret_data: Dict[str, Any]) -> bool:
        """Set secret in AWS Secrets Manager"""
        try:
            # Lazy import boto3 only when AWS backend is actually used
            # This prevents permission errors when AWS is not configured
            try:
                import boto3
                from botocore.exceptions import ClientError
            except (ImportError, PermissionError, OSError) as import_error:
                print(f"⚠️ Cannot import boto3/botocore (AWS not configured or permission denied): {import_error}")
                print(f"💡 Tip: Set SECRETS_BACKEND=env or SECRETS_BACKEND=local to use non-AWS backends")
                return False
            
            session = boto3.session.Session()
            client = session.client(
                service_name='secretsmanager',
                region_name=self.region
            )
            
            secret_string = json.dumps(secret_data)
            
            try:
                # Try to update existing secret
                client.update_secret(
                    SecretId=secret_name,
                    SecretString=secret_string
                )
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    # Create new secret
                    client.create_secret(
                        Name=secret_name,
                        SecretString=secret_string,
                        Description=f"Trading bot secrets for {secret_name}"
                    )
                else:
                    raise
            
            print(f"✅ Secret {secret_name} stored in AWS Secrets Manager")
            return True
            
        except Exception as e:
            print(f"❌ Error storing secret in AWS Secrets Manager: {e}")
            return False
    
    def _get_env_secret(self, secret_name: str) -> Optional[Dict[str, Any]]:
        """Get secret from environment variables"""
        try:
            # Try to get from environment variables
            env_prefix = f"TRADING_BOT_{secret_name.upper()}"
            
            # Check if we have the main secret as JSON
            secret_json = os.getenv(f"{env_prefix}_JSON")
            if secret_json:
                return json.loads(secret_json)
            
            # Fallback to individual environment variables
            secret_data = {}
            expected_keys = [
                "PRIVATE_KEY", "WALLET_ADDRESS", "SOLANA_PRIVATE_KEY",
                "SOLANA_WALLET_ADDRESS", "INFURA_URL", "BASE_RPC_URL",
                "SOLANA_RPC_URL", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
                "HELIUS_API_KEY"
            ]
            
            for key in expected_keys:
                value = os.getenv(f"{env_prefix}_{key}")
                if value:
                    secret_data[key] = value
            
            return secret_data if secret_data else None
            
        except Exception as e:
            print(f"❌ Error reading environment secrets: {e}")
            return None
    
    def _set_env_secret(self, secret_name: str, secret_data: Dict[str, Any]) -> bool:
        """Set secret in environment variables (for reference)"""
        print(f"ℹ️ To set environment secrets for {secret_name}, add to your environment:")
        for key, value in secret_data.items():
            print(f"export TRADING_BOT_{secret_name.upper()}_{key}='{value}'")
        return True
    
    def _get_local_secret(self, secret_name: str) -> Optional[Dict[str, Any]]:
        """Get secret from encrypted local file"""
        try:
            secret_file = f".secrets_{secret_name}.enc"
            if not os.path.exists(secret_file):
                return None
            
            with open(secret_file, "rb") as f:
                encrypted_data = f.read()
            
            if self.fernet:
                decrypted_data = self.fernet.decrypt(encrypted_data)
                return json.loads(decrypted_data.decode())
            else:
                print("⚠️ Encryption not available for local secrets")
                return None
                
        except Exception as e:
            print(f"❌ Error reading local secret {secret_name}: {e}")
            return None
    
    def _set_local_secret(self, secret_name: str, secret_data: Dict[str, Any]) -> bool:
        """Set secret in encrypted local file"""
        try:
            if not self.fernet:
                print("⚠️ Encryption not available for local secrets")
                return False
            
            secret_file = f".secrets_{secret_name}.enc"
            encrypted_data = self.fernet.encrypt(json.dumps(secret_data).encode())
            
            with open(secret_file, "wb") as f:
                f.write(encrypted_data)
            
            # Set restrictive permissions
            os.chmod(secret_file, 0o600)
            print(f"✅ Secret {secret_name} stored in encrypted local file")
            return True
            
        except Exception as e:
            print(f"❌ Error storing local secret {secret_name}: {e}")
            return False

# Global secrets manager instance
_secrets_manager = None

def get_secrets_manager(backend: str = None, region: str = None) -> SecretsManager:
    """Get or create secrets manager instance"""
    global _secrets_manager
    
    if _secrets_manager is None:
        # Determine backend from environment or config
        # Default to "env" instead of "aws" to avoid boto3 import issues
        if backend is None:
            backend = os.getenv("SECRETS_BACKEND", "env")
        
        if region is None:
            region = os.getenv("AWS_REGION", "us-east-1")
        
        _secrets_manager = SecretsManager(backend=backend, region=region)
    
    return _secrets_manager

def get_secret(secret_name: str) -> Optional[Dict[str, Any]]:
    """Get secret from configured backend"""
    manager = get_secrets_manager()
    return manager.get_secret(secret_name)

def set_secret(secret_name: str, secret_data: Dict[str, Any]) -> bool:
    """Set secret in configured backend"""
    manager = get_secrets_manager()
    return manager.set_secret(secret_name, secret_data)

# Convenience functions for common secrets
def get_wallet_secrets() -> Optional[Dict[str, Any]]:
    """Get wallet-related secrets"""
    return get_secret("wallet")

def get_rpc_secrets() -> Optional[Dict[str, Any]]:
    """Get RPC endpoint secrets"""
    return get_secret("rpc")

def get_telegram_secrets() -> Optional[Dict[str, Any]]:
    """Get Telegram bot secrets"""
    return get_secret("telegram")

def setup_secrets_interactive():
    """Interactive setup for secrets"""
    print("🔐 Trading Bot Secrets Setup")
    print("=" * 40)
    
    # Determine backend
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
    
    # Collect secrets
    secrets = {}
    
    print("\n📝 Enter your secrets (press Enter to skip):")
    
    # Wallet secrets
    print("\n🔑 Wallet Configuration:")
    secrets["PRIVATE_KEY"] = input("Ethereum Private Key: ").strip()
    secrets["WALLET_ADDRESS"] = input("Ethereum Wallet Address: ").strip()
    secrets["SOLANA_PRIVATE_KEY"] = input("Solana Private Key: ").strip()
    secrets["SOLANA_WALLET_ADDRESS"] = input("Solana Wallet Address: ").strip()
    
    # RPC secrets
    print("\n🌐 RPC Configuration:")
    secrets["INFURA_URL"] = input("Infura URL: ").strip()
    secrets["BASE_RPC_URL"] = input("Base RPC URL: ").strip()
    secrets["SOLANA_RPC_URL"] = input("Solana RPC URL: ").strip()
    
    # Telegram secrets
    print("\n📱 Telegram Configuration:")
    secrets["TELEGRAM_BOT_TOKEN"] = input("Telegram Bot Token: ").strip()
    secrets["TELEGRAM_CHAT_ID"] = input("Telegram Chat ID: ").strip()
    
    # Remove empty values
    secrets = {k: v for k, v in secrets.items() if v}
    
    if not secrets:
        print("⚠️ No secrets provided")
        return False
    
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
        print(f"\n✅ Secrets stored successfully using {backend} backend")
        return True
    else:
        print("\n❌ Failed to store secrets")
        return False

if __name__ == "__main__":
    setup_secrets_interactive()
