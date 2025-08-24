# 🚀 Crypto Trading Bot 100X

A sophisticated automated cryptocurrency trading bot designed for high-frequency trading across **multiple blockchain networks**. This bot uses advanced strategies including sentiment analysis, trend detection, and risk management to identify and execute profitable trades on Ethereum, Solana, Base, Polygon, BSC, Arbitrum, Optimism, and PulseChain.

## ⚠️ **IMPORTANT DISCLAIMER**

**This bot is for educational and research purposes only. Cryptocurrency trading involves significant risk of loss. Never invest more than you can afford to lose. The developers are not responsible for any financial losses incurred from using this bot.**

## 🎯 Features

- **🌐 Multi-Chain Trading** - Trade on 8+ blockchain networks (Ethereum, Solana, Base, Polygon, BSC, Arbitrum, Optimism, PulseChain)
- **🔄 Real-time Token Scanning** - Monitors trending tokens across multiple platforms and chains
- **📊 Sentiment Analysis** - Analyzes social media sentiment from Reddit and Nitter (Ethereum-focused)
- **⚡ High-Speed Execution** - Optimized for quick entry and exit strategies
- **🛡️ Advanced Risk Management** - Built-in position sizing, stop-loss mechanisms, and duplicate token prevention
- **💰 Wallet Balance Protection** - Automatic balance checking with gas fee buffer to prevent failed transactions
- **🚫 Duplicate Token Prevention** - Prevents buying the same token multiple times to avoid concentration risk
- **📱 Telegram Notifications** - Real-time alerts for trades and bot status
- **📈 Performance Tracking** - Comprehensive logging and analytics
- **🔧 Configurable Strategy** - Easily adjustable parameters via config.yaml
- **🔍 Token Safety Checks** - TokenSniffer integration for Ethereum tokens
- **🚫 Promotional Content Filtering** - Automatically filters out spam and promotional tokens
- **🚨 Delisting Detection** - Automatically detects and handles delisted tokens
- **🛡️ Pre-Buy Delisting Check** - Prevents buying tokens that are already delisted or inactive
- **📊 Position Monitoring** - Real-time position tracking with automatic sell triggers
- **🔄 Multi-Chain Price Fetching** - Chain-specific price monitoring for accurate PnL calculation
- **🌞 Solana Raydium Integration** - Real trading on Solana with automatic token account creation
- **🏊 Pool Discovery** - Automatic Raydium pool finding and liquidity validation
- **💱 Swap Quotes** - Pre-trade quote generation with slippage protection

## 🌐 Supported Blockchains

| Chain | Native Token | DEX | Status | Features |
|-------|-------------|-----|--------|----------|
| **Ethereum** | ETH | Uniswap V2/V3 | ✅ Full Support | Real trading, sentiment analysis, TokenSniffer |
| **Base** | ETH | Uniswap | ✅ Full Support | Real trading, gas optimization |
| **Solana** | SOL | Raydium | ✅ Full Support | Real trading, ATA creation, pool discovery |
| **Polygon** | MATIC | Uniswap | 🔧 Available | Basic support |
| **BSC** | BNB | PancakeSwap | 🔧 Available | Basic support |
| **Arbitrum** | ETH | Uniswap | 🔧 Available | Basic support |
| **Optimism** | ETH | Uniswap | 🔧 Available | Basic support |
| **PulseChain** | PLS | PulseX | 🔧 Available | Basic support |

## 📋 Prerequisites

- **Python 3.8+**
- **Web3.py** for blockchain interactions
- **Solana SDK** for Solana blockchain interactions
- **Cryptocurrency wallets** with native tokens for gas fees:
  - **MetaMask** for Ethereum and Base
  - **Phantom** for Solana
- **Private keys** for wallet signing
- **Telegram Bot Token** (optional, for notifications)
- **Infura API key** (for Ethereum and EVM chains)

## 🔧 Recent Updates & Fixes

### Latest Improvements (v2.1)
- **🔐 Enhanced Secrets Management**: Fixed AWS Secrets Manager integration with proper fallback to `.env` files
- **💰 Wallet Address Validation**: Added automatic checksum address conversion to prevent Web3 errors
- **🚫 AWS Error Elimination**: Configured default secrets backend to avoid AWS credential errors
- **🛡️ Improved Risk Management**: Enhanced wallet balance checking across multiple chains
- **📊 Better Error Handling**: Cleaner startup process with proper state management

### Known Issues Resolved
- ✅ Fixed `ModuleNotFoundError: No module named 'cryptography'` 
- ✅ Fixed `web3.py only accepts checksum addresses` error
- ✅ Eliminated AWS Secrets Manager credential errors
- ✅ Improved multi-chain wallet balance detection

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/mikegianfelice/CRYPTO_TRADING_BOT_100X.git
cd CRYPTO_TRADING_BOT_100X
```

### 2. Install Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

**⚠️ Important: Solana Dependencies**
The bot requires specific versions of Solana packages for proper functionality:
- `solana==0.27.0` (includes required `keypair` module)
- `solders>=0.9.0` (compatible version)

If you encounter `ModuleNotFoundError: No module named 'solana.keypair'`, run:
```bash
pip uninstall solana -y
pip install solana==0.27.0
```

### 3. Configure the Bot

#### 🔐 Secure Secrets Management

The bot supports multiple secure secrets backends to protect your wallet credentials. **For most users, we recommend using the `.env` file approach (Option C) as it's simpler and doesn't require AWS setup.**

**Option A: AWS Secrets Manager (Production)**
```bash
# Install AWS CLI and configure credentials
aws configure

# Run interactive setup
python setup_secrets.py
```

**Option B: Environment Variables**
```bash
# Set environment variables
export TRADING_BOT_SECRETS_PRIVATE_KEY="your_ethereum_private_key"
export TRADING_BOT_SECRETS_WALLET_ADDRESS="your_ethereum_wallet_address"
export TRADING_BOT_SECRETS_SOLANA_PRIVATE_KEY="your_solana_private_key"
export TRADING_BOT_SECRETS_SOLANA_WALLET_ADDRESS="your_solana_wallet_address"
export TRADING_BOT_SECRETS_INFURA_URL="https://mainnet.infura.io/v3/your_key"
export TRADING_BOT_SECRETS_BASE_RPC_URL="https://mainnet.base.org"
export TRADING_BOT_SECRETS_SOLANA_RPC_URL="https://api.mainnet-beta.solana.com"
export TRADING_BOT_SECRETS_TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
export TRADING_BOT_SECRETS_TELEGRAM_CHAT_ID="your_chat_id"
```

**Option C: .env File (Recommended for most users)**
```bash
# Copy the example file
cp .env.example .env

# Edit .env file with your credentials
nano .env
```

**Configure Secrets Backend**
Add this line to your `.env` file to avoid AWS errors:
```env
SECRETS_BACKEND=env
```

#### 🔄 Migrate from .env (Legacy)

If you have an existing `.env` file, migrate to secure storage:
```bash
python setup_secrets.py migrate
```

#### 📝 .env Configuration (Recommended)

Create your `.env` file with the following structure:

```env
# Ethereum/Base Wallet Configuration (MetaMask)
PRIVATE_KEY=your_ethereum_private_key_here
WALLET_ADDRESS=your_ethereum_wallet_address_here

# Solana Wallet Configuration (Phantom)
SOLANA_WALLET_ADDRESS=your_phantom_wallet_address_here
SOLANA_PRIVATE_KEY=your_phantom_private_key_here

# Blockchain RPC URLs
INFURA_URL=https://mainnet.infura.io/v3/your_infura_key
BASE_RPC_URL=https://mainnet.base.org
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com

# Telegram Configuration (Optional)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Secrets Backend Configuration (Important!)
SECRETS_BACKEND=env
```

**⚠️ Security Note**: Keep your `.env` file secure and never commit it to version control. The `.env` file is already in `.gitignore` to prevent accidental commits.

#### Configure Trading Parameters
Edit `config.yaml` to customize your trading strategy:

```yaml
# Multi-Chain Configuration
enable_multi_chain: true
supported_chains:
  - ethereum
  - base
  - solana
  # - polygon
  # - bsc
  # - arbitrum
  # - optimism
  # - pulsechain

# Trading Configuration
test_mode: false              # Set to true for simulation only
trade_amount_usd: 5.0         # Amount per trade in USD
slippage: 0.02                # Slippage tolerance (2%)

# Strategy Parameters
min_volume_24h: 3000          # Minimum 24h volume in USD
min_liquidity: 3000           # Minimum liquidity in USD
min_momentum_pct: 0.005       # Minimum price momentum (0.5%)
enable_pre_buy_delisting_check: true  # Check if token is delisted before buying
fastpath_volume: 50000        # Fast-path volume threshold
fastpath_liquidity: 25000     # Fast-path liquidity threshold
fastpath_sentiment: 40        # Fast-path sentiment threshold

# Risk Management
max_daily_loss: 0.05          # Maximum daily loss (5%)
stop_loss_percentage: 0.15    # Stop loss percentage (15%)
take_profit_percentage: 0.3   # Take profit percentage (30%)
cooldown_period: 300          # Cooldown between trades (seconds)
max_concurrent_positions: 5   # Maximum open positions at a time
daily_loss_limit_usd: 50      # Daily loss cap before circuit breaker
max_losing_streak: 3          # Consecutive losing trades before pause
circuit_breaker_minutes: 60   # Pause duration if circuit breaker triggered
per_trade_max_usd: 25         # Max USD per trade
min_wallet_balance_buffer: 0.01 # Keep 1% of balance for gas fees

# Position Monitoring
take_profit: 0.5              # Take profit at 50% gain
stop_loss: 0.25               # Stop loss at 25% loss
trailing_stop_percent: 0.1    # 10% trailing stop

# Token Filtering
enforce_keywords: false       # Enable/disable keyword filtering
trusted_tokens: []            # List of trusted token addresses
```

### 4. Test Mode (Recommended First Step)
Before running with real money, test the bot in simulation mode:

```yaml
# In config.yaml
test_mode: true               # Set to true for simulation only
```

### 5. Test Solana Implementation (Optional)
Test the Solana trading implementation:

```bash
python test_solana.py
```

This will verify:
- ✅ Environment configuration
- ✅ Wallet initialization
- ✅ Balance checking
- ✅ Price fetching
- ✅ Pool discovery
- ✅ Swap quotes
- ✅ Simulation mode

### 6. Run the Bot
```bash
python main.py
```

## 📊 Configuration Options

### Multi-Chain Strategy
- **Three-Network Support**: Ethereum, Base, and Solana trading with full implementation
- **Chain-Specific Requirements**: Different volume/liquidity thresholds per chain
- **Sentiment Analysis**: Ethereum-focused (skipped for other chains)
- **TokenSniffer Integration**: Ethereum token safety checks
- **DEX-Specific Execution**: Optimized for each blockchain's DEX (Uniswap, Raydium)
- **Wallet Integration**: MetaMask for Ethereum/Base, Phantom for Solana
- **Multi-Chain Price Fetching**: Chain-specific price monitoring for accurate PnL
- **Solana Features**: Raydium DEX integration, automatic token account creation, pool discovery

### Trading Strategy
- **Trending Detection**: Monitors social media and DEX activity across chains
- **Sentiment Analysis**: Analyzes Reddit and Twitter sentiment (Ethereum)
- **Liquidity Analysis**: Ensures sufficient liquidity for trades
- **Volume Analysis**: Tracks trading volume patterns
- **Promotional Filtering**: Automatically filters spam and promotional content

### Risk Management
- **Position Sizing**: Automatic position size calculation
- **Stop Loss**: Configurable stop-loss levels
- **Take Profit**: Automatic profit-taking
- **Daily Limits**: Maximum daily loss protection
- **Cooldown Periods**: Prevents overtrading
- **Multi-Chain Limits**: Separate limits per blockchain
- **Duplicate Token Prevention**: Prevents buying the same token multiple times
- **Wallet Balance Protection**: Checks available funds before trading with gas fee buffer
- **Circuit Breaker**: Automatic pause after consecutive losses
- **Concurrent Position Limits**: Maximum number of open positions
- **Cross-Chain Safety**: Prevents trading on unsupported networks
- **Delisting Detection**: Automatically detects and handles delisted tokens
- **Pre-Buy Delisting Check**: Prevents buying tokens that are already delisted or inactive

### Position Monitoring
- **Real-time Monitoring**: Continuous position tracking every 30 seconds
- **Automatic Sell Triggers**: Take profit, stop loss, and trailing stop
- **Delisting Detection**: Identifies delisted tokens after 5 consecutive price fetch failures
- **PnL Calculation**: Real-time profit/loss tracking
- **Telegram Alerts**: Instant notifications for all position events
- **Trade Logging**: Comprehensive CSV logging with reason codes

### Gas Optimization
- **Dynamic Gas Pricing**: Adjusts gas prices based on network conditions
- **Gas Limit Management**: Optimizes transaction costs
- **Failed Transaction Handling**: Automatic retry mechanisms
- **Chain-Specific Optimization**: Different strategies per blockchain

## 📱 Telegram Integration

The bot can send real-time notifications via Telegram:

1. **Create a Telegram Bot**:
   - Message @BotFather on Telegram
   - Create a new bot and get the token
   - Add the token to your `.env` file

2. **Get Your Chat ID**:
   - Message your bot
   - Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
   - Find your chat_id in the response

3. **Configure Notifications**:
   ```yaml
   # In config.yaml
   telegram_enabled: true
   notify_on_trade: true
   notify_on_error: true
   ```

### Telegram Notifications Include:
- **Trade Executions**: Buy/sell confirmations with transaction hashes
- **Position Updates**: Take profit, stop loss, and trailing stop triggers
- **Delisting Alerts**: Notifications when tokens are delisted with loss amounts
- **Error Alerts**: Important errors and warnings
- **Daily Summaries**: Performance updates and statistics

## 📈 Monitoring and Analytics

### Dashboard
Open `trading_bot_dashboard.html` in your browser to view:
- Real-time trading performance
- Position tracking across chains
- Profit/loss analysis
- Trading history

### Logs
The bot creates detailed logs in:
- `archives/` - Trading logs and data
- Console output - Real-time status updates
- `trending_tokens.csv` - Token discovery history
- `trade_log.csv` - Detailed trade history with reason codes
- `delisted_tokens.json` - Tracking of delisted tokens

## 🔧 Advanced Configuration

### Custom Strategies
You can modify `strategy.py` to implement custom trading strategies:

```python
def custom_strategy(token_data):
    # Your custom logic here
    return buy_signal, sell_signal
```

### Risk Management
Adjust risk parameters in `risk_manager.py`:

```python
# Custom risk rules
def custom_risk_check(position):
    # Your risk assessment logic
    return risk_score
```

### Multi-Chain Configuration
Configure chain-specific settings in `multi_chain_executor.py`:

```python
CHAIN_CONFIGS = {
    "ethereum": {
        "rpc_url": INFURA_URL,
        "native_token": "ETH",
        "dex": "uniswap",
        # ... more config
    }
}
```

### Position Monitoring
Customize position monitoring in `monitor_position.py`:

```python
# Adjust delisting detection sensitivity
def _detect_delisted_token(token_address: str, consecutive_failures: int) -> bool:
    # Customize failure threshold
    return consecutive_failures >= 5  # Default: 5 failures
```

## 🛠️ Troubleshooting

### Common Issues

1. **"Insufficient funds"**
   - Ensure your wallet has enough native tokens for gas fees
   - Check your private key is correct
   - Verify RPC endpoints are accessible
   - The bot now automatically checks wallet balance before trading
   - Consider reducing trade amount if balance is low

2. **"SOLANA_PRIVATE_KEY not found"**
   - Ensure Solana private key is set in .env file
   - Check key format (should be base58 encoded)
   - Verify SOLANA_RPC_URL is accessible
   - For production trading, use a paid RPC provider

3. **"No Raydium pool found"**
   - Token may not have liquidity on Raydium
   - Check if token is listed on Raydium
   - Verify token mint address is correct
   - Some tokens may only be available on other DEXs

2. **"Transaction failed"**
   - Increase gas limit in config.yaml
   - Check network congestion
   - Verify token contract is valid
   - Ensure sufficient native tokens for gas

3. **"No trending tokens found"**
   - Check internet connection
   - Verify API endpoints are accessible
   - Adjust trending thresholds
   - Check if tokens are being filtered as promotional

4. **"Chain not supported"**
   - Verify chain is in supported_chains list
   - Check chain configuration in multi_chain_executor.py
   - Ensure RPC URL is correct

5. **"Token already held"**
   - The bot prevents buying the same token multiple times
   - This is a safety feature to avoid concentration risk
   - Wait for the position to close before buying again

6. **"Insufficient balance"**
   - The bot checks wallet balance before trading
   - Includes gas fee buffer to prevent failed transactions
   - Consider reducing trade amount or adding more funds

7. **"Token delisted"**
   - The bot automatically detects delisted tokens
   - Sends Telegram alert with loss amount
   - Removes from active monitoring
   - Logs as "delisted" trade with 100% loss

8. **"Price fetch failed"**
   - Check RPC endpoint connectivity
   - Verify token address is correct
   - May indicate token is delisted or has no liquidity
   - Bot will automatically handle after 5 consecutive failures

### Debug Mode
Enable debug logging in `config.yaml`:
```yaml
debug_mode: true
log_level: DEBUG
```

## 📚 File Structure

```
crypto_trading_bot_100x/
├── main.py                    # Main entry point
├── config.yaml               # Configuration file
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (create this)
├── multi_chain_executor.py  # Multi-chain trade execution
├── strategy.py              # Trading strategy implementation
├── token_scraper.py         # Token discovery across chains
├── sentiment_scraper.py     # Sentiment analysis
├── risk_manager.py          # Risk management
├── monitor_position.py      # Position monitoring and sell triggers
├── telegram_bot.py          # Telegram notifications
├── token_sniffer.py         # Token safety checks
├── solana_executor.py       # Solana blockchain interactions (NEW!)
├── test_solana.py           # Solana implementation tests (NEW!)
├── SOLANA_IMPLEMENTATION.md # Solana documentation (NEW!)
├── secrets_manager.py       # Secure secrets management (NEW!)
├── setup_secrets.py         # Secrets setup and migration (NEW!)
├── utils.py                 # Utility functions
├── uniswap_router_abi.json  # Uniswap contract ABI
├── trending_tokens.csv      # Token discovery history
├── price_memory.json        # Price history for momentum
├── open_positions.json      # Current positions
├── delisted_tokens.json     # Delisting tracking
├── trade_log.csv            # Detailed trade history
└── blacklist.json           # Blacklisted tokens
```

## 🛡️ Safety Features

### Multi-Network Support
- **Three-Network Trading**: Ethereum, Base, and Solana
- **Wallet Integration**: MetaMask for ETH/Base, Phantom for Solana
- **Cross-Chain Safety**: Prevents trading on unsupported networks
- **Network-Specific Balance Checking**: Verifies funds on each network

### Duplicate Token Prevention
- **Automatic Detection**: The bot checks if a token is already in your open positions
- **Concentration Risk Mitigation**: Prevents over-exposure to single tokens
- **Smart Filtering**: Only allows one position per unique token address

### Wallet Balance Protection
- **Real-time Balance Checking**: Verifies available funds before each trade
- **Gas Fee Buffer**: Automatically reserves funds for transaction fees
- **Configurable Buffer**: Set `min_wallet_balance_buffer` to control reserve amount
- **Failed Transaction Prevention**: Avoids costly failed transactions

### Advanced Risk Controls
- **Circuit Breaker**: Automatic pause after consecutive losses
- **Daily Loss Limits**: Configurable daily loss caps
- **Position Limits**: Maximum concurrent positions
- **Trade Size Limits**: Per-trade maximum amounts

### Delisting Detection
- **Automatic Detection**: Identifies delisted tokens after 5 consecutive price fetch failures
- **Loss Tracking**: Records 100% loss for delisted tokens
- **Telegram Alerts**: Immediate notification when tokens are delisted
- **Position Cleanup**: Removes delisted tokens from active monitoring
- **Trade Logging**: Logs delisted trades with reason code

### Pre-Buy Delisting Check
- **Pre-Purchase Validation**: Checks if tokens are delisted before buying
- **Chain-Specific Logic**: Different validation rules for Ethereum vs Solana tokens
- **Conservative Approach**: Skips tokens that can't be verified
- **DexScreener Protection**: Prevents buying tokens with stale/inactive data
- **Configurable**: Can be enabled/disabled via `enable_pre_buy_delisting_check`
- **Dependency Validation**: Enhanced error handling for missing Solana dependencies
- **Graceful Degradation**: Provides clear error messages and installation instructions when dependencies are missing

### Position Monitoring
- **Real-time Tracking**: Monitors all positions every 30 seconds
- **Automatic Sell Triggers**: Take profit, stop loss, and trailing stop
- **Multi-chain Price Fetching**: Chain-specific price monitoring
- **PnL Calculation**: Real-time profit/loss tracking
- **Telegram Integration**: Instant notifications for all events

## 🔒 Security Best Practices

### 🔐 Secrets Management
1. **Use secure secrets backends** - AWS Secrets Manager, encrypted local files, or environment variables
2. **Never store private keys in plain text** - Avoid `.env` files for production
3. **Rotate credentials regularly** - Update private keys and API tokens periodically
4. **Use dedicated trading wallets** - Separate from main wallets
5. **Limit wallet permissions** - Only grant necessary permissions

### 🛡️ Trading Security
6. **Start with small amounts** - Test with minimal funds first
7. **Monitor the bot regularly** - Check for unusual activity
8. **Set appropriate wallet balance buffers** - Configure `min_wallet_balance_buffer` to reserve funds for gas
9. **Monitor position concentration** - The bot prevents duplicate buys, but monitor overall portfolio diversity
10. **Understand delisting risks** - Meme tokens can be delisted quickly, resulting in 100% loss
11. **Monitor position alerts** - Pay attention to Telegram notifications for position updates

### 🔧 Infrastructure Security
12. **Regularly update dependencies** - Keep all packages updated
13. **Test on testnets first** - Validate functionality before mainnet
14. **Use hardware wallets for large amounts** - Consider hardware wallets for significant funds
15. **Secure wallet setup** - Use separate wallets for different networks (MetaMask for ETH/Base, Phantom for Solana)
16. **Backup secrets securely** - Store backup credentials in secure locations
17. **Monitor access logs** - Track who has access to your secrets

## 🔧 Troubleshooting

### Common Issues

#### 1. Solana Import Errors
**Error**: `ModuleNotFoundError: No module named 'solana.keypair'`

**Solution**: Install the correct Solana version:
```bash
pip uninstall solana -y
pip install solana==0.27.0
```

**Why**: Newer versions of the Solana package (0.30+) removed the `keypair` module, which is required for the bot's pre-buy delisting checks.

#### 2. Delisted Token Still Bought
**Issue**: Token appears "delisted" in logs but gets bought anyway

**Explanation**: 
- "Delisted" messages in discovery logs indicate tokens with zero liquidity, not actual delisting
- If pre-buy delisting check fails due to missing dependencies, the bot defaults to allowing the trade
- Check if Solana dependencies are properly installed

**Solution**: Ensure Solana dependencies are correct and pre-buy checks are working

#### 3. Pre-Buy Delisting Check Failures
**Error**: `⚠️ Pre-buy check failed for [TOKEN]: No module named 'solana.keypair'`

**Solution**: Fix Solana dependencies as shown above

**Impact**: Without proper delisting checks, the bot may buy tokens that are actually delisted or inactive

#### 4. SSL Warnings
**Warning**: `NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+`

**Solution**: This is a warning and doesn't affect functionality. For production, consider updating OpenSSL or using a different Python environment.

#### 5. Secrets Management Issues
**Error**: `❌ Missing required secrets`

**Solution**: 
```bash
# Set up secrets interactively
python setup_secrets.py

# Or migrate from existing .env
python setup_secrets.py migrate
```

**Error**: `❌ Error accessing AWS Secrets Manager`

**Solution**: 
- Ensure AWS CLI is configured: `aws configure`
- Check IAM permissions for Secrets Manager
- Verify AWS region matches your configuration

**Error**: `⚠️ Encryption not available for local secrets`

**Solution**: Install cryptography package:
```bash
pip install cryptography
```

### Dependency Issues

#### Solana Package Version Conflicts
The bot requires specific Solana package versions:
- ✅ `solana==0.27.0` (includes `keypair` module)
- ❌ `solana>=0.30.0` (missing `keypair` module)

#### Installation Verification
Test Solana dependencies:
```bash
python -c "import solana.keypair; print('✅ Solana keypair module working')"
```

### Performance Issues

#### High Gas Fees
- Use Base network for lower fees
- Adjust `min_wallet_balance_buffer` to ensure sufficient gas
- Monitor gas prices before trading

#### Slow Token Discovery
- Check RPC endpoint connectivity
- Verify API rate limits
- Consider using paid RPC providers for production

## 📞 Support

For issues and questions:
- Check the troubleshooting section above
- Review the configuration options
- Test in simulation mode first
- Start with small position sizes
- Verify chain-specific requirements
- Ensure wallet configurations are correct for each network
- Check RPC endpoint connectivity for all chains
- Monitor Telegram alerts for position updates
- Understand delisting risks and detection

### Solana-Specific Support
- **Test Solana Implementation**: Run `python test_solana.py`
- **Check Solana Documentation**: See `SOLANA_IMPLEMENTATION.md`
- **Verify RPC Endpoint**: Ensure SOLANA_RPC_URL is accessible
- **Wallet Setup**: Use Phantom wallet for Solana
- **Token Account Creation**: Bot automatically creates associated token accounts
- **Pool Discovery**: Bot finds Raydium pools automatically

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Risk Warning

**Cryptocurrency trading is highly risky and volatile. This bot is provided as-is without any guarantees. Always:**
- Test thoroughly in simulation mode
- Start with small amounts
- Monitor performance closely
- Never invest more than you can afford to lose
- Understand the risks involved
- Be aware of cross-chain transaction risks
- Understand that meme tokens can be delisted quickly
- Monitor position alerts and delisting notifications

---

**Happy Multi-Chain Trading! 🚀📈🌐**
