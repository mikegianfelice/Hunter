import base64
import struct
from typing import Tuple, Optional, List
from solana.rpc.api import Client
from solana.transaction import Transaction, TransactionInstruction
from solana.keypair import Keypair
from solana.publickey import PublicKey
from solana.rpc.commitment import Commitment
import requests

from secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY

# Solana client
solana_client = Client(SOLANA_RPC_URL)

# Raydium program IDs
RAYDIUM_AMM_PROGRAM_ID = PublicKey("675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8")
RAYDIUM_SWAP_PROGRAM_ID = PublicKey("9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM")
SERUM_PROGRAM_ID = PublicKey("9xQeWvG816bUx9EPjHmaT23yvVM2ZWbrrpZb9PusVFin")

# Common token addresses
WSOL_MINT = PublicKey("So11111111111111111111111111111111111111112")
USDC_MINT = PublicKey("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")

def get_associated_token_address(wallet: PublicKey, mint: PublicKey) -> PublicKey:
    """Get the associated token account address"""
    return PublicKey.find_program_address(
        [
            bytes(wallet),
            bytes(PublicKey("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")),
            bytes(mint)
        ],
        PublicKey("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
    )[0]

def get_or_create_associated_token_account(
    wallet: PublicKey,
    mint: PublicKey,
    payer: Keypair
) -> PublicKey:
    """Get or create associated token account"""
    ata = get_associated_token_address(wallet, mint)
    
    # Check if account exists
    account_info = solana_client.get_account_info(ata)
    if account_info.value:
        return ata
    
    # Create account if it doesn't exist
    create_ata_ix = create_associated_token_account_instruction(
        payer=payer.public_key,
        owner=wallet,
        mint=mint
    )
    
    tx = Transaction()
    tx.add(create_ata_ix)
    
    result = solana_client.send_transaction(tx, payer)
    if result.value:
        return ata
    
    raise Exception("Failed to create associated token account")

def create_associated_token_account_instruction(
    payer: PublicKey,
    owner: PublicKey,
    mint: PublicKey
) -> TransactionInstruction:
    """Create instruction to create associated token account"""
    ata = get_associated_token_address(owner, mint)
    
    return TransactionInstruction(
        program_id=PublicKey("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"),
        data=b"",
        keys=[
            {"pubkey": payer, "is_signer": True, "is_writable": True},
            {"pubkey": ata, "is_signer": False, "is_writable": True},
            {"pubkey": owner, "is_signer": False, "is_writable": False},
            {"pubkey": mint, "is_signer": False, "is_writable": False},
            {"pubkey": PublicKey("11111111111111111111111111111111"), "is_signer": False, "is_writable": False},
            {"pubkey": PublicKey("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"), "is_signer": False, "is_writable": False},
        ]
    )

def get_raydium_pool_info(token_mint: str) -> Optional[dict]:
    """Get Raydium pool information for a token"""
    try:
        url = f"https://api.raydium.io/v2/main/pool/{token_mint}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"⚠️ Error getting pool info: {e}")
        return None

def build_raydium_swap_instruction(
    pool_info: dict,
    user_wallet: PublicKey,
    user_token_account: PublicKey,
    amount_in: int,
    minimum_amount_out: int
) -> TransactionInstruction:
    """Build Raydium swap instruction"""
    
    # Extract pool accounts from pool info
    pool_id = PublicKey(pool_info["id"])
    amm_id = PublicKey(pool_info["ammId"])
    amm_authority = PublicKey(pool_info["ammAuthority"])
    pool_coin_token_account = PublicKey(pool_info["baseVault"])
    pool_pc_token_account = PublicKey(pool_info["quoteVault"])
    serum_program_id = PublicKey(pool_info["serumProgramId"])
    serum_market = PublicKey(pool_info["serumMarket"])
    serum_coin_vault = PublicKey(pool_info["serumBaseVault"])
    serum_pc_vault = PublicKey(pool_info["serumQuoteVault"])
    serum_vault_signer = PublicKey(pool_info["serumVaultSigner"])
    
    # Build instruction data
    # Raydium swap instruction layout:
    # 1 byte: instruction (2 for swap)
    # 8 bytes: amount_in (u64)
    # 8 bytes: minimum_amount_out (u64)
    instruction_data = struct.pack("<BQQ", 2, amount_in, minimum_amount_out)
    
    return TransactionInstruction(
        program_id=RAYDIUM_AMM_PROGRAM_ID,
        data=instruction_data,
        keys=[
            {"pubkey": amm_id, "is_signer": False, "is_writable": True},
            {"pubkey": amm_authority, "is_signer": False, "is_writable": False},
            {"pubkey": user_wallet, "is_signer": True, "is_writable": False},
            {"pubkey": user_token_account, "is_signer": False, "is_writable": True},
            {"pubkey": pool_coin_token_account, "is_signer": False, "is_writable": True},
            {"pubkey": pool_pc_token_account, "is_signer": False, "is_writable": True},
            {"pubkey": serum_program_id, "is_signer": False, "is_writable": False},
            {"pubkey": serum_market, "is_signer": False, "is_writable": True},
            {"pubkey": serum_coin_vault, "is_signer": False, "is_writable": True},
            {"pubkey": serum_pc_vault, "is_signer": False, "is_writable": True},
            {"pubkey": serum_vault_signer, "is_signer": False, "is_writable": False},
        ]
    )

def execute_raydium_swap(
    token_mint: str,
    usd_amount: float,
    slippage_percent: float = 1.0
) -> Tuple[str, bool]:
    """
    Execute a real Raydium swap
    
    Args:
        token_mint: Token mint address
        usd_amount: Amount to spend in USD
        slippage_percent: Slippage tolerance (default 1%)
    
    Returns:
        (transaction_hash, success)
    """
    try:
        print(f"🚀 Executing real Raydium swap")
        
        # Get wallet keypair
        if not SOLANA_PRIVATE_KEY:
            print("❌ Solana private key not configured")
            return None, False
            
        # Decode private key
        try:
            private_key_bytes = base64.b58decode(SOLANA_PRIVATE_KEY)
            keypair = Keypair.from_secret_key(private_key_bytes)
        except Exception as e:
            print(f"❌ Error decoding private key: {e}")
            return None, False
        
        # Get pool information
        pool_info = get_raydium_pool_info(token_mint)
        if not pool_info:
            print(f"❌ Could not get pool info for token {token_mint}")
            return None, False
        
        print(f"📊 Pool found: {pool_info['name']}")
        
        # Get token price
        token_price = float(pool_info.get("price", 0))
        if token_price <= 0:
            print(f"❌ Invalid token price: ${token_price}")
            return None, False
        
        print(f"💰 Token price: ${token_price}")
        
        # Calculate amounts
        token_amount = int((usd_amount / token_price) * 1e9)  # Assuming 9 decimals
        minimum_amount_out = int(token_amount * (1 - slippage_percent / 100))
        
        print(f"📈 Buying {token_amount / 1e9:.6f} tokens")
        print(f"📉 Minimum out: {minimum_amount_out / 1e9:.6f} tokens")
        
        # Get or create token account
        token_mint_pubkey = PublicKey(token_mint)
        user_token_account = get_or_create_associated_token_account(
            keypair.public_key,
            token_mint_pubkey,
            keypair
        )
        
        print(f"🏦 Token account: {user_token_account}")
        
        # Build swap instruction
        swap_ix = build_raydium_swap_instruction(
            pool_info,
            keypair.public_key,
            user_token_account,
            int(usd_amount * 1e6),  # USDC has 6 decimals
            minimum_amount_out
        )
        
        # Create and send transaction
        transaction = Transaction()
        transaction.add(swap_ix)
        
        print(f"🔄 Sending transaction...")
        
        # Get recent blockhash
        recent_blockhash = solana_client.get_recent_blockhash()
        if not recent_blockhash.value:
            print("❌ Could not get recent blockhash")
            return None, False
        
        transaction.recent_blockhash = recent_blockhash.value.blockhash
        
        # Sign and send transaction
        transaction.sign(keypair)
        
        result = solana_client.send_transaction(transaction, keypair)
        if not result.value:
            print("❌ Transaction failed")
            return None, False
        
        tx_hash = result.value
        print(f"✅ Transaction successful: {tx_hash}")
        
        return tx_hash, True
        
    except Exception as e:
        print(f"❌ Error executing Raydium swap: {e}")
        return None, False
