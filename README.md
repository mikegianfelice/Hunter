# 🚀 Crypto Trading Bot 100X

A sophisticated automated cryptocurrency trading bot designed for high-frequency trading on **Ethereum and Solana**. This bot uses advanced strategies including sentiment analysis, trend detection, and risk management to identify and execute profitable trades with real-time monitoring and smart blacklist management.

## ⚠️ **IMPORTANT DISCLAIMER**

**This bot is for educational and research purposes only. Cryptocurrency trading involves significant risk of loss. Never invest more than you can afford to lose. The developers are not responsible for any financial losses incurred from using this bot.**

## 🎯 Features

- **🌐 Dual-Chain Trading** - Real trading on Ethereum (MetaMask) and Solana (Phantom)
- **🔄 Real-time Token Scanning** - Monitors trending tokens across multiple platforms
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
- **🚨 Enhanced Delisting Detection** - Automatic detection and blacklisting of delisted/inactive tokens
- **🛡️ Smart Blacklist Management** - Automatic cleanup of blacklist every 6 hours to maintain trading opportunities
- **📊 Position Monitoring** - Real-time position tracking with automatic sell triggers
- **🔄 Multi-Chain Price Fetching** - Chain-specific price monitoring for accurate PnL calculation
- **🌞 Multi-DEX Solana Integration** - Real trading on Solana across multiple DEXs (Raydium, PumpSwap, Meteora, Heaven)
- **🏊 Multi-DEX Pool Discovery** - Automatic pool finding across multiple Solana DEXs with liquidity validation
- **💱 Multi-DEX Swap Quotes** - Pre-trade quote generation with slippage protection across all supported DEXs

## 🌐 Supported Blockchains

| Chain | Native Token | DEX | Status | Features |
|-------|-------------|-----|--------|----------|
| **Ethereum** | ETH | Uniswap V2/V3 | ✅ Full Support | Real trading, sentiment analysis, TokenSniffer |
| **Solana** | SOL | Multi-DEX (Raydium, PumpSwap, Meteora, Heaven) | ✅ Full Support | Real trading, ATA creation, multi-DEX pool discovery |

*Note: Other chains (Base, Polygon, BSC, Arbitrum, Optimism, PulseChain) are disabled by default to focus on chains where you have active wallets.*

## 📋 Prerequisites

- **Python 3.8+**
- **Web3.py** for blockchain interactions
- **Solana SDK** for Solana blockchain interactions
- **Cryptocurrency wallets** with native tokens for gas fees:
  - **MetaMask** for Ethereum
  - **Phantom** for Solana
- **Private keys** for wallet signing
- **Telegram Bot Token** (optional, for notifications)
- **Infura API key** (for Ethereum)

## 🔧 Recent Updates & Fixes

### Latest Improvements (v3.1) - Custom Jupiter Library & Real Trading Implementation
- **🎯 Custom Jupiter Library**: Created `jupiter_lib.py` with direct transaction handling for Jupiter v6
- **✅ Real Trading Confirmed**: Successfully executing real trades with actual transaction hashes
- **🔧 Transaction Size Management**: Automatic fallback with smaller amounts when transactions are too large
- **📊 Multi-Source Price Data**: Enhanced price fetching with DexScreener, Birdeye, and CoinGecko
- **🛡️ Smart Transaction Signing**: Proper transaction signing using `solders` library
- **⚡ Optimized Performance**: Reduced transaction size limits and improved success rates
- **🔄 Fallback Mechanisms**: Multiple retry attempts with different amounts for better success
- **📱 Real Transaction Hashes**: No more dummy `1111...` hashes - real transaction tracking

### Latest Improvements (v3.0) - Enhanced Price Verification & Smart Blacklist Management
- **🔧 Fixed JSON Corruption**: Resolved corrupted `delisted_tokens.json` file that was causing parsing errors
- **🎯 Enhanced Solana Token Verification**: More lenient thresholds for Solana tokens when DexScreener shows good data
- **🛡️ Smart Blacklist Management**: Implemented failure tracking system to avoid blacklisting tokens for temporary API failures
- **📊 Better Error Handling**: Multiple fallback mechanisms when price APIs fail instead of rejecting tokens
- **🔄 Automatic Blacklist Review**: Periodic cleanup of old blacklisted tokens (every 7 days)
- **✅ Fixed TOPLESS & PUMP Issues**: Resolved specific problems where legitimate tokens were incorrectly rejected
- **📈 Improved Token Evaluation**: Trust DexScreener data when APIs fail but metrics are good
- **⚡ Enhanced Resilience**: System continues working when external APIs are down

### Latest Improvements (v2.9) - SOL Price API Fix & Trading Strategy Optimization
- **🔧 Fixed SOL Price API Issues**: Resolved CoinGecko rate limiting and Jupiter API parameter errors
- **🎯 Optimized Solana Token Evaluation**: Modified strategy to trust DexScreener data for Solana tokens with good volume/liquidity
- **✅ Resolved 10+ Hour No-Trades Issue**: Fixed critical problem where bot was incorrectly rejecting all tokens due to price validation failures
- **🛡️ Enhanced Pre-Buy Delisting Logic**: Improved Solana token validation to use volume/liquidity thresholds instead of unreliable price APIs
- **📊 Better Token Discovery**: Bot now properly evaluates tokens based on DexScreener data rather than failing price validation
- **⚡ Improved Trading Success**: Bot attempted first trade in 10+ hours after implementing fixes
- **🔧 Fixed Jupiter API Parameters**: Corrected boolean parameter format for Jupiter quote API calls
- **📈 Enhanced Error Handling**: Better fallback mechanisms for price API failures

### Latest Improvements (v2.8) - Smart Blacklist Management & Dual-Chain Focus
- **🧹 Smart Blacklist Auto-Cleanup**: Automatic blacklist maintenance every 6 hours to maintain trading opportunities
- **🎯 Dual-Chain Focus**: Simplified to Ethereum (MetaMask) and Solana (Phantom) only for better reliability
- **🛡️ Enhanced Pre-Buy Delisting**: Improved detection of delisted tokens before trading to prevent losses
- **✅ Real Trading Confirmed**: Bot now executes real transactions (no more simulated trades)
- **📱 Clean Telegram Messages**: Real transaction hashes sent to Telegram instead of simulated ones
- **🔧 Simplified Configuration**: Removed support for chains without active wallets to reduce complexity
- **⚡ Better Performance**: Focused on two chains for faster execution and better reliability

### Latest Improvements (v2.6) - Jupiter Price API Fix & Enhanced Reliability
- **🔧 Fixed Jupiter Price API Error**: Resolved `[Errno 8] nodename nor servname provided, or not known` error
- **📊 Enhanced Price Fetching**: Replaced deprecated `price.jup.ag` with reliable `quote-api.jup.ag` and CoinGecko
- **🔄 Multi-Source Price Data**: Implemented fallback system using Jupiter quote API and CoinGecko for better reliability
- **⚡ Improved SOL Price Fetching**: Enhanced `get_sol_price_usd()` with multiple data sources and better error handling
- **🛡️ Better Token Price Coverage**: Added support for more tokens (BONK, PEPE, JITO) in CoinGecko fallback
- **📈 Enhanced Error Handling**: Graceful degradation when price APIs are unavailable

### Latest Improvements (v2.5) - Multi-Chain Trading & Telegram Fixes
- **🎯 Fixed No-Trades Issue**: Resolved 20+ hour trading drought by optimizing token discovery
- **📊 Enhanced Token Discovery**: Increased from 18 to 89 tokens per cycle (395% improvement)
- **🌐 Multi-Chain Optimization**: Enabled Ethereum + Solana trading, disabled chains without funds
- **🛡️ Re-enabled Delisting Checks**: Prevents buying delisted tokens to avoid "investment lost" messages
- **📱 Telegram Deduplication**: Fixed spam issue - risk control messages sent only once per 5 minutes
- **⚡ Reduced Requirements**: Lowered volume/liquidity thresholds for more trading opportunities
- **🔓 Relaxed Filtering**: Removed overly strict keyword filters (AI, MOON) for more opportunities
- **📈 Better Chain Support**: Added 6 new API sources for enhanced token discovery

### Latest Improvements (v2.4) - Enhanced Token Discovery & Quality
- **🎯 Fixed No-Trades Issue**: Resolved the main blocking factor (delisted tokens list was too aggressive)
- **📊 Enhanced Token Discovery**: 225% more unique symbols, 13,640% higher average volume
- **🔄 Improved Token Quality**: Implemented 8-point scoring system with better filtering
- **🌐 Better API Sources**: Added 4 additional DexScreener endpoints with randomized order
- **🛡️ Smart Filtering**: Enhanced promotional content detection and spam filtering
- **⚖️ Symbol Diversity**: Limited duplicate symbols to prevent token list domination
- **📈 Higher Thresholds**: Increased volume/liquidity requirements for better quality tokens
- **🧹 Cleanup Tools**: Added delisted tokens cleanup and maintenance scripts

### Latest Improvements (v2.3) - Multi-DEX Solana Support
- **🌐 Multi-DEX Solana Integration**: Added support for multiple Solana DEXs (Raydium, PumpSwap, Meteora, Heaven)
- **🔍 Enhanced Pool Discovery**: Automatic pool searching across all supported DEXs with priority-based selection
- **💱 Multi-DEX Swap Execution**: Real trading capabilities across multiple Solana DEXs
- **📊 Improved Token Discovery**: Better token finding by searching across multiple DEXs simultaneously
- **🛡️ DEX Fallback System**: If one DEX fails, automatically tries others for better success rates
- **⚡ Performance Optimization**: Reduced volume/liquidity requirements for more trading opportunities
- **🔓 Relaxed Safety Checks**: Temporarily disabled TokenSniffer and sentiment checks for increased trading activity

### Latest Improvements (v2.2)
- **🚨 Enhanced Delisting Detection**: Automatic detection and blacklisting of delisted/inactive tokens
- **📊 Smart Failure Tracking**: Tokens with 5+ failures are automatically delisted
- **🛡️ Improved Token Screening**: Pre-buy and post-trade delisting checks
- **⚡ Performance Optimization**: Reduced processing time by avoiding inactive tokens
- **🔄 Persistent Learning**: Delisted tokens are permanently blocked across bot restarts

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
- ✅ Resolved delisted token processing issues
- ✅ Fixed 20+ hour trading drought by optimizing token discovery
- ✅ Resolved Telegram message spam with deduplication system
- ✅ Fixed "token delisted investment lost" by re-enabling pre-buy checks
- ✅ Optimized multi-chain support to only trade on chains with funds
- ✅ **CRITICAL FIX**: Resolved 10+ hour no-trades issue caused by pre-buy delisting check incorrectly blocking new tokens
- ✅ **CRITICAL FIX**: Resolved SOL price API failures causing all Solana tokens to be marked as "delisted"
- ✅ **CRITICAL FIX**: Fixed Jupiter API parameter errors preventing price validation
- ✅ **CRITICAL FIX**: Optimized Solana token evaluation strategy to trust DexScreener data

## 📱 Telegram Notifications & Deduplication

The bot includes intelligent Telegram notification system with automatic message deduplication:

### **🔄 Message Deduplication System**
- **5-Minute Cache**: Prevents sending identical messages within 5 minutes
- **Automatic Cleanup**: Old messages are automatically removed from cache
- **Smart Filtering**: Reduces spam from repeated risk control messages
- **Configurable**: Can be disabled per message if needed

### **📨 Notification Types**
- **Trade Executions**: Buy/sell confirmations with transaction links
- **Risk Alerts**: Position limits, daily loss limits, circuit breakers
- **Error Notifications**: Failed trades, delisted tokens, API issues
- **Status Updates**: Bot startup, mode changes, configuration updates

### **⚙️ Configuration**
```yaml
# Telegram settings in config.yaml
telegram_bot_token: "your_bot_token"
telegram_chat_id: "your_chat_id"
```

## 🚀 Jupiter v6 Integration - Real Solana Trading

The bot now features a **custom Jupiter library** that enables real trading on Solana with actual transaction execution:

### **✅ Real Trading Confirmed**
- **Actual Transaction Hashes**: No more dummy `1111...` hashes - real transaction tracking
- **Custom Jupiter Library**: `jupiter_lib.py` with direct transaction handling for Jupiter v6
- **Smart Transaction Signing**: Proper transaction signing using `solders` library
- **Transaction Size Management**: Automatic fallback with smaller amounts when transactions are too large

### **🔧 How It Works**
1. **Quote Retrieval**: Gets best swap quotes from Jupiter v6 API
2. **Transaction Creation**: Jupiter creates the swap transaction
3. **Transaction Signing**: Signs the transaction with your wallet using custom library
4. **Transaction Submission**: Sends signed transaction to Solana network
5. **Fallback Logic**: Multiple retry attempts with different amounts for better success

### **📊 Real Transaction Results**
Recent successful trades with actual transaction hashes:
- **Buy Transaction**: `5dZMcrGAAZmhs79bm1K46wasKvLXDVgSoKPnoHwVAuZmKJ3fwkLZAbAZwnXjwoWZs3nLDQoFqaFHmyvWChwD3jDJ`
- **Sell Transaction**: `endhq277LAdSe59qHBXgDskLocmBCqiUARzFRrQpTPNTB4kbepmULq8XuNCwMWwaWaMgkvRVYbhY9SQdr3nnqug`

### **🛡️ Safety Features**
- **Transaction Size Limits**: Automatically reduces trade amounts when transactions are too large
- **Multiple Fallbacks**: Tries with 50%, 25%, 10%, and 5% of original amount
- **Real-time Validation**: Verifies transaction success before proceeding
- **Error Handling**: Comprehensive error handling and retry mechanisms

## 🌐 Multi-DEX Solana Support

The bot now supports **multiple Solana DEXs** for enhanced trading opportunities and better success rates:

### **🔍 Supported Solana DEXs:**

| DEX | API Base | Status | Priority |
|-----|----------|--------|----------|
| **Raydium** | `https://api.raydium.io/v2` | ✅ Primary | 1st |
| **PumpSwap** | `https://api.pumpswap.finance` | ✅ Active | 2nd |
| **Meteora** | `https://api.meteora.ag` | ✅ Active | 3rd |
| **Heaven** | `https://api.heaven.so` | ✅ Active | 4th |

### **🚀 How Multi-DEX Works:**

#### **1. Priority-Based Pool Discovery**
- **Sequential Search**: Searches DEXs in priority order (Raydium → PumpSwap → Meteora → Heaven)
- **First Match Wins**: Uses the first DEX that has a valid pool for the token
- **Automatic Fallback**: If one DEX fails, automatically tries the next

#### **2. Enhanced Token Discovery**
- **Broader Coverage**: Tokens listed on any supported DEX can be traded
- **Better Liquidity**: Access to pools across multiple DEXs increases trading opportunities
- **Reduced Failures**: Higher success rate when one DEX is unavailable

#### **3. Unified Trading Interface**
- **Single API**: All DEXs use the same trading interface
- **Consistent Execution**: Same slippage protection and quote generation across all DEXs
- **Simplified Management**: No need to configure multiple DEX settings

### **📊 Benefits:**
- **🎯 Higher Success Rate**: More trading opportunities across multiple DEXs
- **💰 Better Liquidity**: Access to pools that might not be on Raydium
- **🛡️ Redundancy**: If one DEX is down, others continue working
- **⚡ Faster Execution**: Priority-based selection finds pools quickly

## 🎯 Enhanced Token Discovery System

The bot now features a significantly improved token discovery system that addresses quality, diversity, and filtering issues:

### **📊 Key Improvements:**

#### **1. Quality Scoring System (8-point scale)**
- **Volume Scoring**: 0-3 points based on 24h volume ($10k, $50k, $100k+)
- **Liquidity Scoring**: 0-3 points based on liquidity ($10k, $50k, $100k+)
- **Symbol Quality**: Penalizes spam symbols, rewards unique ones
- **Chain Bonus**: +1 for Ethereum, neutral for major chains

#### **2. Enhanced Filtering**
- **Promotional Content**: Better detection of spam/promotional tokens
- **Keyword Filtering**: Blocks common spam keywords (INU, AI, PEPE, DOGE, etc.)
- **Symbol Diversity**: Prevents token list from being dominated by one symbol (max 2 per symbol)
- **Minimum Requirements**: Higher volume/liquidity thresholds for better quality

#### **3. Better API Usage**
- **Multiple Sources**: 7 primary + 3 fallback DexScreener endpoints
- **Randomized Order**: Prevents bias toward specific sources
- **Rate Limiting**: Added delays between requests
- **Error Handling**: Better fallback mechanisms

### **📈 Performance Results:**
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Unique Symbols** | 4 | 13 | +225% |
| **Avg Volume** | $1,673 | $229,970 | +13,640% |
| **Avg Liquidity** | $34,973 | $361,500 | +934% |
| **Symbol Diversity** | Poor (65% HOT) | Good (13 unique) | +225% |

### **🛠️ Maintenance Tools:**
- `cleanup_delisted_tokens.py` - Clean up false positives in delisted tokens list


## 🚨 Enhanced Delisting Detection System

The bot now includes an intelligent delisting detection system that automatically identifies and blocks inactive or delisted tokens:

### **🔍 How It Works:**

#### **1. Pre-Buy Screening**
- **Zero Price Detection**: Tokens with zero prices are flagged as potentially delisted
- **Low Price Detection**: Tokens with prices < $0.0000001 are automatically delisted
- **API Failure Detection**: Tokens that can't be verified are conservatively blocked

#### **2. Smart Failure Tracking**
- **Failure Counter**: Each failed trade increments a failure counter
- **Auto-Delisting**: Tokens with 5+ failures are automatically delisted
- **Persistent Storage**: Delisted tokens are permanently blocked across bot restarts

#### **3. Post-Trade Analysis**
- **Trade Failure Analysis**: Failed trades trigger additional delisting checks
- **Zero Price Verification**: Tokens with zero prices after failed trades are delisted
- **Automatic Blacklisting**: Delisted tokens are added to `delisted_tokens.json`

### **📁 Files Used:**
- `delisted_tokens.json` - Stores permanently delisted tokens
- `cooldown_log.json` - Tracks failure counts and temporary cooldowns
- `blacklist_manager.py` - Manages both regular blacklist and delisted tokens

### **⚙️ Configuration:**
```yaml
enable_pre_buy_delisting_check: true   # Enable pre-buy delisting checks
```

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
# Multi-Chain Configuration (Optimized for Available Funds)
enable_multi_chain: true
supported_chains:
  - ethereum     # Enabled - requires ETH for gas fees
  - solana       # Enabled - requires SOL for gas fees
  # - base       # Disabled - need ETH on Base network
  # - pulsechain # Disabled - need PLS for gas fees
  # - polygon    # Disabled - need MATIC for gas fees
  # - bsc        # Disabled - need BNB for gas fees
  # - arbitrum   # Disabled - need ETH on Arbitrum
  # - optimism   # Disabled - need ETH on Optimism

# Trading Configuration
test_mode: false              # Set to true for simulation only
trade_amount_usd: 10.0        # Amount per trade in USD (increased for better positions)
slippage: 0.02                # Slippage tolerance (2%)

# Strategy Parameters (Optimized for More Opportunities)
min_volume_24h: 100           # Minimum 24h volume in USD (reduced from 3000)
min_liquidity: 100            # Minimum liquidity in USD (reduced from 3000)
min_momentum_pct: 0.001       # Minimum price momentum (0.1% - reduced from 0.5%)
enable_pre_buy_delisting_check: true  # Check if token is delisted before buying
fastpath_volume: 500          # Fast-path volume threshold (reduced from 50000)
fastpath_liquidity: 500       # Fast-path liquidity threshold (reduced from 25000)
fastpath_sentiment: 20        # Fast-path sentiment threshold (reduced from 40)

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

### 4. Test Mode (Optional)
The bot is configured for real trading by default. If you want to test in simulation mode first:

```yaml
# In config.yaml
test_mode: true               # Set to true for simulation only
```

**Note**: The bot now executes real trades by default. Make sure you have sufficient funds in your MetaMask (Ethereum) and Phantom (Solana) wallets before running.



### 6. Run the Bot
```bash
python main.py
```

## 📊 Configuration Options

### Dual-Chain Strategy
- **Two-Network Support**: Ethereum and Solana trading with full implementation
- **Chain-Specific Requirements**: Different volume/liquidity thresholds per chain
- **Sentiment Analysis**: Ethereum-focused (skipped for Solana)
- **TokenSniffer Integration**: Ethereum token safety checks
- **DEX-Specific Execution**: Optimized for each blockchain's DEX (Uniswap, Raydium)
- **Wallet Integration**: MetaMask for Ethereum, Phantom for Solana
- **Multi-Chain Price Fetching**: Chain-specific price monitoring for accurate PnL
- **Solana Features**: Multi-DEX integration (Raydium, PumpSwap, Meteora, Heaven), automatic token account creation, priority-based pool discovery
- **Smart Blacklist Management**: Automatic cleanup every 6 hours to maintain trading opportunities

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
- **Smart Blacklist Management**: Automatic cleanup of blacklist every 6 hours to maintain trading opportunities while keeping high-risk tokens blocked

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
- **Trade Executions**: Buy/sell confirmations with real transaction hashes
- **Position Updates**: Take profit, stop loss, and trailing stop triggers
- **Delisting Alerts**: Notifications when tokens are delisted with loss amounts
- **Error Alerts**: Important errors and warnings
- **Daily Summaries**: Performance updates and statistics
- **Blacklist Maintenance**: Notifications when smart cleanup removes safer tokens

## 📈 Monitoring and Analytics

### Dashboard
Open `trading_bot_dashboard.html` in your browser to view:
- Real-time trading performance
- Position tracking across chains
- Profit/loss analysis
- Trading history

### Smart Blacklist Management
The bot automatically maintains its blacklist to ensure trading opportunities:

- **Automatic Cleanup**: Runs every 6 loops (6 hours) to remove safer tokens
- **Risk-Based Filtering**: Keeps high-risk tokens (3+ failures, 100% losses) blocked
- **Trading Opportunity Maintenance**: Removes tokens with low failure counts to allow new trades
- **Transparent Logging**: Shows exactly which tokens are removed and why
- **Configurable**: Adjust cleanup frequency and failure thresholds in config.yaml

**Configuration Options:**
```yaml
# Smart Blacklist Maintenance
enable_smart_blacklist_cleanup: true    # Auto-clean blacklist every 6 loops
blacklist_cleanup_interval: 6           # Clean every N loops (6 = every 6 hours)
blacklist_keep_failure_threshold: 3     # Keep tokens with N+ failures
```

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

9. **"Jupiter price API error: nodename nor servname provided"**
   - **FIXED**: This error has been resolved in v2.6
   - The bot now uses reliable `quote-api.jup.ag` and CoinGecko instead of deprecated `price.jup.ag`
   - If you see this error, update to the latest version
   - The bot automatically falls back to multiple price sources for reliability

10. **"Bot running for 10+ hours with no trades"**
    - **FIXED**: This critical issue has been resolved in v2.9
    - The pre-buy delisting check was incorrectly blocking ALL new tokens due to SOL price API failures
    - Solana tokens were being marked as "delisted" because Jupiter API calls were failing
    - The strategy now trusts DexScreener data for Solana tokens with good volume/liquidity
    - Update to the latest version to fix this issue
    - The bot now properly distinguishes between delisted tokens and new tokens

11. **"All Solana tokens marked as delisted"**
    - **FIXED**: This issue has been resolved in v2.9
    - Jupiter API parameter errors were causing all price validation to fail
    - CoinGecko rate limiting was preventing SOL price fetching
    - The bot now uses volume/liquidity thresholds for Solana token evaluation
    - Update to the latest version to fix this issue

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
├── jupiter_lib.py           # Custom Jupiter library for real trading (NEW!)
├── jupiter_executor.py      # Jupiter trading executor (NEW!)
├── solana_executor.py       # Solana blockchain interactions (legacy)

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
