# Solana Trading Implementation

This document describes the complete Solana trading implementation for the crypto trading bot, including real Raydium DEX integration.

## Overview

The Solana implementation provides:
- ✅ Real Solana wallet integration
- ✅ Raydium DEX trading (buy/sell tokens)
- ✅ Automatic token account creation
- ✅ Price fetching from Raydium API
- ✅ Pool discovery and liquidity checking
- ✅ Swap quotes and slippage protection
- ✅ Transaction building and signing
- ✅ Test mode support

## Features

### 🔐 Wallet Integration
- Secure private key management
- Automatic wallet initialization
- Balance checking (SOL and tokens)
- Transaction signing

### 💱 Raydium DEX Trading
- Real token swaps on Raydium
- USDC as quote currency
- Automatic slippage protection (2% default)
- Pool discovery and validation
- Swap quotes before execution

### 🏦 Token Account Management
- Automatic associated token account creation
- Token balance tracking
- Support for any SPL token

### 📊 Price & Market Data
- Real-time token prices from Raydium API
- Pool liquidity information
- Market depth analysis

## Configuration

### Environment Variables

Add these to your `.env` file:

```bash
# Solana Configuration
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
SOLANA_WALLET_ADDRESS=your_wallet_address_here
SOLANA_PRIVATE_KEY=your_base58_private_key_here
```

### Config.yaml Settings

```yaml
# Multi-Chain Settings
enable_multi_chain: true
supported_chains:
  - "ethereum"
  - "base"
  - "solana"       # Enabled - uses Phantom wallet

# Modes
test_mode: false   # Set to true for simulation only
```

## Usage

### Basic Trading

```python
from solana_executor import buy_token_solana, sell_token_solana

# Buy tokens
tx_hash, success = buy_token_solana(
    token_address="token_mint_address",
    usd_amount=10.0,  # $10 USD
    symbol="TOKEN",
    test_mode=False  # Set to True for simulation
)

# Sell tokens
tx_hash, success = sell_token_solana(
    token_address="token_mint_address", 
    token_amount=100.0,  # 100 tokens
    symbol="TOKEN",
    test_mode=False
)
```

### Advanced Usage

```python
from solana_executor import get_solana_executor

# Get executor instance
executor = get_solana_executor()

# Check balances
sol_balance = executor.get_solana_balance()
token_balance = executor.get_token_balance("token_mint_address")

# Get prices
price = executor.get_token_price_usd("token_mint_address")

# Find pools
pool = executor.find_pool_for_token("token_mint_address")

# Get swap quotes
quote = executor.get_swap_quote(pool, 1000000, is_buy=True)  # 1 USDC
```

## Architecture

### Core Components

1. **SolanaExecutor Class**
   - Main trading engine
   - Wallet management
   - RPC client integration
   - Transaction building

2. **PoolInfo Dataclass**
   - Raydium pool data structure
   - Pool metadata and state
   - Liquidity information

3. **Trading Functions**
   - `buy_token_solana()` - Purchase tokens
   - `sell_token_solana()` - Sell tokens
   - `get_solana_balance()` - Check balances
   - `get_token_price_usd()` - Get prices

### Transaction Flow

1. **Pool Discovery**
   - Fetch all Raydium pools
   - Find pool for token/USDC pair
   - Validate pool liquidity

2. **Quote Generation**
   - Get swap quote from Raydium API
   - Calculate expected output
   - Apply slippage protection

3. **Account Preparation**
   - Check for existing token accounts
   - Create associated token accounts if needed
   - Validate account balances

4. **Transaction Building**
   - Build Raydium swap instruction
   - Add all required account metas
   - Set transaction parameters

5. **Execution**
   - Sign transaction with wallet
   - Submit to Solana network
   - Wait for confirmation

## Testing

### Run Test Suite

```bash
python test_solana.py
```

This will test:
- ✅ Environment configuration
- ✅ Wallet initialization
- ✅ Balance checking
- ✅ Price fetching
- ✅ Pool discovery
- ✅ Swap quotes
- ✅ Simulation mode

### Manual Testing

1. **Test Mode First**
   ```python
   # Always test in simulation mode first
   tx_hash, success = buy_token_solana(token_address, 1.0, "TEST", test_mode=True)
   ```

2. **Small Amounts**
   ```python
   # Start with small amounts for real trading
   tx_hash, success = buy_token_solana(token_address, 0.1, "TEST", test_mode=False)
   ```

## Security Considerations

### Private Key Management
- Store private keys securely in environment variables
- Never commit private keys to version control
- Use hardware wallets for large amounts

### Transaction Safety
- Always verify transaction details before signing
- Use slippage protection to prevent MEV attacks
- Monitor transaction confirmations

### Error Handling
- Implement proper error handling for failed transactions
- Retry logic for network issues
- Balance validation before trades

## Troubleshooting

### Common Issues

1. **"SOLANA_PRIVATE_KEY not found"**
   - Ensure private key is set in .env file
   - Check key format (should be base58 encoded)

2. **"No Raydium pool found"**
   - Token may not have liquidity on Raydium
   - Check if token is listed on Raydium
   - Verify token mint address

3. **"Failed to create token account"**
   - Insufficient SOL for transaction fees
   - Network congestion
   - Invalid token mint address

4. **"Swap quote failed"**
   - Insufficient liquidity in pool
   - Token price too volatile
   - Pool temporarily unavailable

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Optimization

### Caching
- Pool data is cached to reduce API calls
- Price data cached for 15 minutes
- Token account lookups cached

### Batch Operations
- Multiple swaps can be batched in single transaction
- Reduced gas costs for multiple operations

### Connection Pooling
- RPC connection reuse
- Automatic retry on connection failures

## Future Enhancements

### Planned Features
- [ ] Jupiter aggregator integration
- [ ] MEV protection
- [ ] Advanced order types (limit orders)
- [ ] Portfolio tracking
- [ ] Automated rebalancing

### Performance Improvements
- [ ] Async transaction processing
- [ ] WebSocket price feeds
- [ ] Local transaction simulation
- [ ] Optimized instruction building

## Support

For issues or questions:
1. Check the troubleshooting section
2. Run the test suite
3. Review transaction logs
4. Verify environment configuration

## License

This implementation is part of the crypto trading bot project and follows the same license terms.
