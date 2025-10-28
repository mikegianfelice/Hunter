import re


EVM_ADDRESS_REGEX = re.compile(r"^0x[a-fA-F0-9]{40}$")

# Base58 alphabet used by Solana
_BASE58_ALPHABET = set("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")


def is_evm_address(address: str) -> bool:
    """
    Return True if address looks like a valid EVM address (Ethereum/Base):
    - Starts with 0x and has 40 hex chars after (total length 42)
    """
    if not isinstance(address, str):
        return False
    return bool(EVM_ADDRESS_REGEX.match(address))


def looks_like_evm_hex(address: str) -> bool:
    """
    Return True if address starts with 0x and is hex, regardless of length.
    Useful to distinguish EVM-like values that are not valid 20-byte addresses
    (e.g., pair addresses or tx hashes with 32 bytes / 64 hex chars).
    """
    if not isinstance(address, str) or not address.startswith("0x"):
        return False
    try:
        int(address[2:], 16)
        return True
    except Exception:
        return False


def is_solana_address(address: str) -> bool:
    """
    Proper validation for Solana addresses:
    - Does not start with 0x
    - Only contains Base58 characters
    - Typical length between 32 and 44
    - Can be properly decoded with base58
    """
    if not isinstance(address, str) or not address:
        return False
    if address.startswith("0x"):
        return False
    if not (32 <= len(address) <= 44):
        return False
    
    # Check if all characters are valid Base58
    if not all(ch in _BASE58_ALPHABET for ch in address):
        return False
    
    # Try to decode with base58 to ensure it's valid
    try:
        import base58
        decoded = base58.b58decode(address)
        # Solana addresses should decode to 32 bytes
        return len(decoded) == 32
    except Exception:
        return False


def detect_chain_from_address(address: str) -> str:
    """
    Return a coarse chain classification from address format:
    - "evm" for Ethereum-like (valid 20-byte EVM addresses)
    - "solana" for Base58 Solana addresses
    - "unknown" otherwise
    Note: Cannot distinguish Base vs Ethereum from address alone; both are EVM.
    """
    if is_evm_address(address):
        return "evm"
    if is_solana_address(address):
        return "solana"
    return "unknown"


def normalize_evm_address(address: str) -> str:
    """
    Best-effort normalization to checksum address when Web3 is available.
    Falls back to the original string if Web3 is not importable or conversion fails.
    """
    try:
        from web3 import Web3  # type: ignore
        return Web3.to_checksum_address(address) if address else address
    except Exception:
        return address


def validate_chain_address_match(address: str, declared_chain: str) -> tuple[bool, str, str]:
    """
    Validate that the address format matches the declared chain.
    
    Returns:
        (is_valid: bool, corrected_chain: str, error_message: str)
    """
    if not address or not declared_chain:
        return False, declared_chain, "Missing address or chain"
    
    detected_chain = detect_chain_from_address(address)
    declared_chain_lower = declared_chain.lower()
    
    # Handle EVM chains (ethereum, base, etc.)
    if detected_chain == "evm":
        if declared_chain_lower in ("ethereum", "base", "arbitrum", "optimism", "polygon", "bsc"):
            return True, declared_chain_lower, ""
        else:
            # For EVM addresses, we can correct to a supported EVM chain
            return True, "ethereum", f"Corrected chain from {declared_chain_lower} to ethereum (EVM address)"
    
    # Handle Solana
    elif detected_chain == "solana":
        if declared_chain_lower == "solana":
            return True, "solana", ""
        else:
            return True, "solana", f"Corrected chain from {declared_chain_lower} to solana based on address format"
    
    # Handle unknown address format
    else:
        return False, declared_chain_lower, f"Unknown address format: {address[:12]}..."


def get_supported_chains_for_address(address: str) -> list[str]:
    """
    Get list of supported chains that this address format could work on.
    """
    detected_chain = detect_chain_from_address(address)
    
    if detected_chain == "evm":
        return ["ethereum", "base", "arbitrum", "optimism", "polygon", "bsc"]
    elif detected_chain == "solana":
        return ["solana"]
    else:
        return []


