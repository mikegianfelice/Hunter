import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables first
PROJECT_ROOT = Path(__file__).resolve().parents[2]
system_env_path = PROJECT_ROOT / "system" / ".env"
if system_env_path.exists():
    try:
        load_dotenv(system_env_path)
    except PermissionError:
        print(f"⚠️ Unable to read {system_env_path}, skipping.")
load_dotenv()

from .secrets_manager import get_secret, get_secrets_manager

# Helper constants/utilities
_DEFAULT_SOLANA_RPC = "https://api.mainnet-beta.solana.com"


def _derive_solana_rpc(explicit_url: str, helius_key: str) -> str:
    """
    Determine which Solana RPC endpoint to use.
    Priority:
      1. Explicit URL if it is set and not the public default (user override)
      2. Helius endpoint when API key is available
      3. Public Solana RPC fallback
    """
    explicit_url = (explicit_url or "").strip()
    helius_key = (helius_key or "").strip()

    if explicit_url and explicit_url != _DEFAULT_SOLANA_RPC:
        return explicit_url

    if helius_key:
        return f"https://mainnet.helius-rpc.com/?api-key={helius_key}"

    return explicit_url or _DEFAULT_SOLANA_RPC


# Try to get secrets from secure backend first
secrets = get_secret("trading_bot_secrets")

if secrets:
    # Use secrets from secure backend
    WALLET_ADDRESS = secrets.get("WALLET_ADDRESS")
    PRIVATE_KEY = secrets.get("PRIVATE_KEY")
    INFURA_URL = secrets.get("INFURA_URL")
    GRAPH_API_KEY = secrets.get("GRAPH_API_KEY")
    UNISWAP_V3_DEPLOYMENT_ID = secrets.get("UNISWAP_V3_DEPLOYMENT_ID")
    TELEGRAM_BOT_TOKEN = secrets.get("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = secrets.get("TELEGRAM_CHAT_ID")
    SOLANA_WALLET_ADDRESS = secrets.get("SOLANA_WALLET_ADDRESS")
    SOLANA_PRIVATE_KEY = secrets.get("SOLANA_PRIVATE_KEY")
    BASE_RPC_URL = secrets.get("BASE_RPC_URL", "https://mainnet.base.org")
    HELIUS_API_KEY = secrets.get("HELIUS_API_KEY")
    SOLANA_RPC_URL = _derive_solana_rpc(secrets.get("SOLANA_RPC_URL"), HELIUS_API_KEY)
else:
    # Fallback to environment variables (for backward compatibility)
    
    WALLET_ADDRESS = os.getenv("WALLET_ADDRESS")
    PRIVATE_KEY = os.getenv("PRIVATE_KEY")
    INFURA_URL = os.getenv("INFURA_URL")
    GRAPH_API_KEY = os.getenv("GRAPH_API_KEY")
    UNISWAP_V3_DEPLOYMENT_ID = os.getenv("UNISWAP_V3_DEPLOYMENT_ID")
    

    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    SOLANA_WALLET_ADDRESS = os.getenv("SOLANA_WALLET_ADDRESS")
    SOLANA_PRIVATE_KEY = os.getenv("SOLANA_PRIVATE_KEY")
    BASE_RPC_URL = os.getenv("BASE_RPC_URL", "https://mainnet.base.org")
    HELIUS_API_KEY = os.getenv("HELIUS_API_KEY")
    SOLANA_RPC_URL = _derive_solana_rpc(os.getenv("SOLANA_RPC_URL"), HELIUS_API_KEY)

# Validate required secrets
def validate_secrets():
    """Validate that required secrets are available"""
    required_secrets = {
        "WALLET_ADDRESS": WALLET_ADDRESS,
        "PRIVATE_KEY": PRIVATE_KEY,
        "INFURA_URL": INFURA_URL,
    }
    
    missing = [key for key, value in required_secrets.items() if not value]
    
    if missing:
        print("❌ Missing required secrets:")
        for secret in missing:
            print(f"   - {secret}")
        print("\n💡 Run 'python secrets_manager.py' to set up secrets securely")
        return False
    
    # Check Solana secrets (optional but recommended for Solana trading)
    solana_secrets = {
        "SOLANA_RPC_URL": SOLANA_RPC_URL,
        "SOLANA_WALLET_ADDRESS": SOLANA_WALLET_ADDRESS,
        "SOLANA_PRIVATE_KEY": SOLANA_PRIVATE_KEY,
    }
    
    missing_solana = [key for key, value in solana_secrets.items() if not value]
    
    if missing_solana:
        print("⚠️  Missing Solana secrets (Solana trading will not work):")
        for secret in missing_solana:
            print(f"   - {secret}")
        print("💡 Add Solana secrets to enable Solana chain trading\n")
    
    return True