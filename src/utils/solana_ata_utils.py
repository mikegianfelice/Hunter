#!/usr/bin/env python3
"""
Solana Associated Token Account (ATA) Utilities

Provides functions to calculate Associated Token Account addresses
for Solana SPL tokens. ATA addresses are deterministic and can be
calculated without RPC calls.
"""

from typing import Optional
from solders.pubkey import Pubkey


# SPL Token Program ID
SPL_TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")

# Associated Token Program ID (used for ATA derivation)
ASSOCIATED_TOKEN_PROGRAM_ID = Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")


def get_associated_token_address(owner: str, mint: str) -> Optional[str]:
    """
    Calculate the Associated Token Account (ATA) address for a wallet and token mint.
    
    The ATA is a Program Derived Address (PDA) that is deterministically derived from:
    - Owner wallet address
    - Token mint address
    - Associated Token Program ID
    
    Args:
        owner: Wallet address (base58 string or Pubkey)
        mint: Token mint address (base58 string or Pubkey)
    
    Returns:
        Base58-encoded ATA address string, or None if calculation fails
    
    Example:
        >>> owner = "YourWalletAddress..."
        >>> usdc_mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        >>> ata = get_associated_token_address(owner, usdc_mint)
        >>> print(ata)  # ATA address for USDC
    """
    try:
        # Convert string addresses to Pubkey objects
        if isinstance(owner, str):
            owner_pubkey = Pubkey.from_string(owner)
        else:
            owner_pubkey = owner
        
        if isinstance(mint, str):
            mint_pubkey = Pubkey.from_string(mint)
        else:
            mint_pubkey = mint
        
        # ATA derivation seeds (in order):
        # 1. Owner public key (32 bytes)
        # 2. Token Program ID (32 bytes)
        # 3. Mint public key (32 bytes)
        
        # Create seeds array - each seed must be bytes
        # Pubkey.__bytes__() returns the 32-byte representation
        seeds = [
            bytes(owner_pubkey),
            bytes(SPL_TOKEN_PROGRAM_ID),
            bytes(mint_pubkey)
        ]
        
        # Find Program Derived Address (PDA)
        # The bump seed is automatically found by trying all 256 possibilities
        # Returns (pubkey, bump_seed)
        ata_pubkey, bump_seed = Pubkey.find_program_address(seeds, ASSOCIATED_TOKEN_PROGRAM_ID)
        
        # Return base58-encoded address
        return str(ata_pubkey)
        
    except Exception as e:
        print(f"⚠️ Error calculating ATA address: {e}")
        return None


def get_ata_address_for_token(wallet_address: str, token_mint: str) -> Optional[str]:
    """
    Convenience wrapper for get_associated_token_address.
    
    Args:
        wallet_address: Wallet address as base58 string
        token_mint: Token mint address as base58 string
    
    Returns:
        ATA address as base58 string, or None on error
    """
    return get_associated_token_address(wallet_address, token_mint)


def validate_ata_address(address: str) -> bool:
    """
    Validate that an address is a valid Solana public key.
    
    Args:
        address: Base58-encoded address string
    
    Returns:
        True if address is valid, False otherwise
    """
    try:
        Pubkey.from_string(address)
        return True
    except Exception:
        return False

