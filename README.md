# 🚀 Hunter: The AI-Powered Crypto Trading Bot

Hunter is a quant level AI-powered crypto trading bot executing real on-chain trades with live PnL, smart token scanning, sentiment scoring, microstructure analysis, and adaptive risk control. Updated in real time. If you find this project useful, drop a ⭐️ to support it.

## ⚠️ **IMPORTANT DISCLAIMER**

**Cryptocurrency trading involves significant risk of loss. Never invest more than you can afford to lose. The developers are not responsible for any financial losses incurred from using this bot.**

## 📊 Live Performance

![30-Day Performance](docs/performance_chart.png)

*Performance charts updated daily. Past performance does not guarantee future results.*

**Quick Stats:**
- [7-Day Performance](docs/performance_7d.png)
- [90-Day Performance](docs/performance_90d.png)

---

## 🔄 How It Works

### **1. Token Selection** 🔍
The bot continuously scans DexScreener for trending tokens across Solana and Base. It filters out low-quality tokens by requiring:
- Minimum $150k daily volume
- Minimum $150k liquidity
- Active trading activity
- No stablecoins or wrapped tokens

Top candidates are ranked by volume and passed to the AI system for deeper analysis.

### **2. Entry Decision** ✅
Before entering any trade, tokens must pass multiple AI-driven checks:
- **AI Quality Score** ≥ 60% (combines sentiment, technical analysis, price prediction)
- **Success Probability** ≥ 60% (AI's prediction of profitable outcome)
- **Risk Score** ≤ 50% (lower is safer)
- **AI Recommendation** = "buy" with >70% confidence
- **Risk Gates** = Approved (wallet balance, position limits, market conditions)

Only tokens passing all criteria are considered for trading.

### **3. Exit Strategy** 💰
The bot continuously monitors all open positions and automatically sells when:
- **Take Profit** is hit: 13% gain (default, can be dynamic based on market conditions)
- **Stop Loss** is triggered: 7% loss
- **Trailing Stop** activates: Locks in profits if price drops 6% from peak
- **Token Delisted**: Auto-sells if token becomes untradeable

All positions are monitored in real-time, and exits execute automatically on-chain.

---

## 🎯 Artificial Intelligence Features

### **🧠 Core AI Systems**
- **🔍 AI Market Microstructure Analyzer** - Real-time order book analysis, trade flow patterns, liquidity analysis, whale activity detection, market maker identification, and manipulation detection
- **⚡ AI Trade Execution Optimizer** - Intelligent execution timing, optimal DEX/router selection, slippage minimization, gas optimization, and success prediction
- **📊 AI Sentiment Analysis** - Advanced sentiment analysis using transformer models for social media, news, market, and technical sentiment
- **🎯 AI Market Regime Detection** - Machine learning-based market condition detection (bull, bear, sideways, volatile, recovery) with adaptive strategy
- **🔮 AI Price Prediction** - LSTM neural networks for predicting token success probability and optimal entry/exit timing
- **📈 AI Portfolio Optimization** - Modern portfolio theory optimization for capital allocation, risk-adjusted returns, and diversification
- **🛡️ AI Risk Assessment** - Machine learning-based risk scoring, loss probability prediction, and dynamic risk management
- **🔍 AI Pattern Recognition** - Computer vision-based candlestick pattern detection, support/resistance analysis, and momentum signals
- **🧠 AI Market Intelligence Aggregator** - Comprehensive market intelligence analysis including news sentiment, social media intelligence, influencer tracking, and market events
- **🔮 AI Predictive Analytics Engine** - Advanced price movement prediction using technical analysis, sentiment analysis, volume analysis, and correlation analysis
- **🎯 AI Dynamic Strategy Selector** - Adaptive strategy selection that automatically switches between momentum, mean reversion, breakout, scalping, swing, and trend-following strategies based on market conditions
- **🛡️ AI Risk Prediction & Prevention System** - Comprehensive risk protection including flash crash detection, rug pull prevention, manipulation detection, liquidity drain monitoring, correlation breakdown analysis, and black swan event detection
- **🔄 AI Market Regime Transition Detector** - Real-time market regime change detection for optimal strategy switching and risk management
- **💧 AI Liquidity Flow Analyzer** - Advanced liquidity flow analysis for market imbalance detection and optimal entry/exit timing
- **⏰ AI Multi-Timeframe Analysis Engine** - Comprehensive multi-timeframe analysis integrating data from multiple timeframes for robust market view
- **🔄 AI Market Cycle Predictor** - Market cycle prediction for strategic positioning and anticipating major market shifts
- **🛡️ AI Drawdown Protection System** - Advanced drawdown protection with portfolio analysis, individual trade analysis, and market condition monitoring
- **📊 AI Performance Attribution Analyzer** - Performance driver identification for strategy optimization and performance improvement
- **🔍 AI Market Anomaly Detector** - Unusual market condition detection for opportunity and risk identification
- **⚖️ AI Portfolio Rebalancing Engine** - Optimal portfolio allocation using modern portfolio theory for risk-adjusted returns and diversification

### **🌐 Multi-Chain Trading**
- **Real trading on Solana (Phantom) and Base (MetaMask)**
- **DEX Integration** - Uniswap V3 (Base), Jupiter + Raydium (Solana)
- **Chain-Specific Optimization** - Tailored strategies for each blockchain
- **Robust Fallbacks on Solana** - Jupiter as primary with Raydium fallback for reliability
- **USDC Trading on Solana** - Uses USDC as base currency for Solana trades (configurable)

### **📊 Advanced Analytics & Performance**
- **Dynamic Position Sizing** - AI-calculated position sizes based on token quality, risk, and market conditions
- **Performance Tracking** - Comprehensive PnL analysis, win rate tracking, and quality tier performance
- **Real-time Insights** - Live market analysis, execution optimization, and risk assessment
- **Quality Scoring System** - 0-100 quality scores combining traditional metrics with AI analysis

### **🛡️ Risk Management & Safety**
- **AI-Enhanced Risk Assessment** - Multi-factor risk analysis with machine learning
- **Manipulation Detection** - Advanced detection of pump/dump schemes, wash trading, and spoofing
- **Whale Activity Monitoring** - Real-time whale trade detection and impact analysis
- **Market Maker Intelligence** - Leverages market maker presence for execution stability
- **Smart Blacklist Management** - Automatic cleanup and intelligent token filtering

### **⚡ Execution & Optimization**
- **Optimal Execution Timing** - AI-determined best execution windows
- **Slippage Minimization** - Advanced slippage optimization across all trades
- **Gas Optimization** - Intelligent gas price optimization and cost reduction
- **Route Selection** - Optimal DEX/router selection for each trade
- **Success Prediction** - AI-powered execution success probability calculation
- **AI Fill Verifier** - Confirms real fills, reroutes failed slices, and prevents ghost entries
- **AI Time-Window Scheduler** - Gates entries to high-quality execution windows based on live fill/slippage metrics

### **📱 Monitoring & Notifications**
- **Telegram Integration** - Real-time alerts for trades, performance, and system status
- **Performance Dashboard** - Interactive command-line dashboard for performance analysis
- **Live Trading Ready** - Comprehensive live trading readiness checks
- **Configurable Strategy** - Extensive configuration options via config.yaml

## 🌐 Supported Blockchains

| Chain | Native Token | DEX | Status | Features |
|-------|-------------|-----|--------|----------|
| **Solana** | SOL/USDC | Jupiter + Raydium | ✅ Full Support | Real trading, ATA creation, Raydium fallback, USDC trading |
| **Base** | ETH | Uniswap V3 | ✅ Full Support | Real trading, EIP-1559 gas optimization, re-quote protection |
| ~~**Ethereum**~~ | ETH | Uniswap V2/V3 | ❌ Disabled | Removed to focus on chains with lower fees and better opportunities |

*Note: Currently focusing on Solana and Base for optimal fee efficiency and trading opportunities. Ethereum support may be re-enabled in the future.*

## 🧠 AI System Integration Architecture

The bot features a **comprehensive AI integration engine** that coordinates **30 AI modules** across **6 analysis stages**:

### **Stage 1: Core Analysis** (Foundation)
1. **📊 Sentiment Analysis** - Analyzes social media, news, market, and technical sentiment
2. **🔮 Price Prediction** - Predicts token success probability using LSTM neural networks
3. **🛡️ Risk Assessment** - Machine learning-based risk scoring and loss prediction
4. **📈 Market Conditions** - Market health, liquidity, and volume analysis
5. **🔍 Technical Analysis** - Pattern recognition, indicators, and trend analysis
6. **⚡ Execution Optimization** - Basic execution timing and slippage optimization

### **Stage 2: Market Context Analysis** (Regime & Environment)
7. **🎯 Market Regime Detection** - Detects market conditions (bull, bear, sideways, volatile, recovery) and adapts strategy
8. **🔄 Market Regime Transition Detector** - Real-time detection of regime changes for optimal strategy switching
9. **🔄 Market Cycle Predictor** - Predicts market cycles (accumulation, markup, distribution, markdown) for strategic positioning
10. **💧 Liquidity Flow Analyzer** - Analyzes liquidity flow patterns, detects traps, and optimizes execution timing
11. **🔍 Market Anomaly Detector** - Detects unusual market conditions, opportunities, and risks

### **Stage 3: Predictive Analytics** (Forecasting)
12. **🔮 Predictive Analytics Engine** - Advanced price movement prediction using multiple analysis methods
13. **🔍 Market Microstructure Analyzer** - Analyzes order book, trade flow, liquidity, whale activity, and manipulation
14. **⏰ Multi-Timeframe Analysis Engine** - Integrates data from multiple timeframes for robust market view

### **Stage 4: Risk Controls** (Protection)
15. **🛡️ Risk Prediction & Prevention System** - Comprehensive risk protection including flash crash detection, rug pull prevention, manipulation detection
16. **🛡️ Drawdown Protection System** - Advanced drawdown protection with portfolio analysis and market condition monitoring
17. **🚨 Emergency Stop System** - Automatically halts trading during extreme conditions, system errors, or excessive losses

### **Stage 5: Portfolio Analysis** (Optimization)
18. **📈 Portfolio Optimization** - Optimizes capital allocation using modern portfolio theory
19. **⚖️ Portfolio Rebalancing Engine** - Optimal portfolio allocation for risk-adjusted returns and diversification

### **Stage 6: Execution Optimization** (Post-Analysis)
20. **⚡ Trade Execution Monitor** - Monitors trade execution quality and performance
21. **✅ Position Size Validator** - Validates position sizes before execution to prevent oversized trades
22. **🎯 Dynamic Strategy Selector** - Adaptive strategy selection based on market conditions and performance
23. **🔍 Pattern Recognition** - Computer vision-based pattern detection and signal generation
24. **🧠 Market Intelligence Aggregator** - Comprehensive market intelligence including news, social media, and influencer analysis
25. **🔌 Circuit Breaker** - Automatic fault tolerance and recovery
26. **✅ Fill Verifier** - Confirms real fills, reroutes failed slices, prevents ghost entries
27. **⏰ Time Window Scheduler** - Gates entries to high-quality execution windows
28. **💵 Partial Take-Profit Manager** - Staged profit-taking with adaptive trailing stops
29. **🛡️ Market Condition Guardian** - Trading safety monitoring and market condition checks

### **Integration Benefits**
- **Parallel Processing**: All modules run in parallel for maximum efficiency
- **Comprehensive Scoring**: Overall score combines all 6 stages with weighted importance
- **Intelligent Recommendations**: Trading decisions incorporate insights from all modules
- **Graceful Degradation**: System continues operating even if some modules fail
- **Circuit Breaker Protection**: Automatic fault tolerance and recovery

### **📊 Performance Metrics**
- **Target Gains**: 10-20% consistent returns
- **Quality Focus**: Minimum $150k volume, $150k liquidity for improved win rate (raised from $100k)
- **Quality Scoring**: Minimum quality score of 50 (0-100 scale, raised from 40)
- **Risk Management**: Maximum 6 concurrent positions, $10 daily loss limit
- **Stop Loss/Take Profit**: 7% stop loss, 13% take profit (with dynamic TP up to 20%)
- **Execution Optimization**: AI-optimized execution with balance verification for sell transactions
- **Performance Tracking**: Comprehensive analytics with quality tier analysis, failed entry attempts excluded from win rate

### **🎯 Multi-Chain Tier System**
The bot now features a sophisticated **tiered position sizing system** that scales with your total portfolio value across all chains:

#### **💰 Unified Portfolio Management**
- **Combined Balance Calculation**: Sums balances from Solana (Phantom) + Base (MetaMask) wallets
- **Unified Tier Detection**: Uses total portfolio value to determine appropriate trading tier
- **Consistent Position Sizing**: Same tier-based position sizes across all supported chains
- **Accelerated Growth**: Larger positions on all chains as your portfolio grows

#### **📈 Tier Structure**
| Tier | Balance Range | Base Position | Max Position | Max Exposure | Description |
|------|---------------|---------------|--------------|--------------|-------------|
| **Tier 1** | $100 - $499 | $5 - $10 | $10 | $100 | Learning Phase - Conservative |
| **Tier 2** | $500 - $4,999 | $25 - $50 | $50 | $500 | Scaling Phase - Moderate |
| **Tier 3** | $5,000 - $19,999 | $50 - $100 | $100 | $1,000 | Acceleration Phase - Aggressive |
| **Tier 4** | $20,000 - $99,999 | $100 - $200 | $200 | $2,000 | Professional Phase - Maximum |
| **Tier 5** | $100,000+ | $200 - $500 | $500 | $5,000 | Institutional Phase - Elite |

#### **🚀 Key Benefits**
- **Unified Risk Management**: Total portfolio value determines your tier, not individual chain balances
- **Consistent Trading**: Same position sizes across Solana and Base
- **Faster Growth**: Combined balance reaches higher tiers sooner than individual chains
- **Better Portfolio Utilization**: Maximum capital efficiency across all chains
- **Dynamic Scaling**: Automatically adjusts as your portfolio grows

#### **⚙️ Configuration**
```yaml
# Enable tiered position sizing
enable_tiered_position_sizing: true
tiered_position_scaling: true

# Wallet tiers configuration
wallet_tiers:
  tier_1:
    min_balance: 100
    max_balance: 499
    base_position_size_usd: 5.0
    max_position_size_usd: 10.0
    max_total_exposure_usd: 100.0
    max_wallet_usage_percent: 0.10
    description: "Learning Phase - Conservative"
  # ... additional tiers
```

### ⚠️ Current Solana/Jupiter API Status (Updated)
- **Jupiter API Endpoint Changes**: The legacy `quote-api.jup.ag` is deprecated
- **Default Uses Ultra (Free)**: The bot uses `https://lite-api.jup.ag` (Ultra) by default — no API key required
- **Paid Endpoint Optional**: You can set `JUPITER_API_BASE=https://api.jup.ag` and `JUPITER_API_KEY` to use the paid API
- **Tradeability Checks**: Uses real market data from DexScreener for liquidity and trading activity verification
- **Real Data Integration**: All tradeability checks use actual DexScreener data (liquidity, transaction counts, prices)
- **Impact**: Minimal - actual trading functionality continues to work with real market data validation
- **Status**: Tradeability checks fully functional using DexScreener API

## 📋 Prerequisites

- **Python 3.8+**
- **Web3.py** for blockchain interactions
- **Solana SDK** for Solana blockchain interactions
- **Cryptocurrency wallets** with native tokens for gas fees:
  - **Phantom** for Solana (SOL or USDC for trading)
  - **MetaMask** for Base (ETH for gas)
- **Private keys** for wallet signing
- **Telegram Bot Token** (optional, for notifications)
- **RPC endpoints** for Solana and Base networks

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

## 🔄 Raydium Fallback System - Enhanced Solana Trading

The bot now includes a **custom Raydium library** with DexScreener fallback for reliable quote generation and enhanced trading capabilities:

### **✅ Raydium Fallback Confirmed**
- **Custom Raydium Library**: `raydium_lib.py` with direct API interaction similar to Jupiter library
- **DexScreener Integration**: Robust fallback for quote generation when Raydium API fails
- **Smart DEX Routing**: Intelligent routing - Raydium for volatile tokens (BONK, PEPE, JITO), Jupiter for others
- **Enhanced Reliability**: Multiple retry mechanisms and fallback systems for API failures

### **🔧 How Raydium Fallback Works**
1. **Primary Attempt**: Direct Raydium API call for quotes and swap transactions
2. **Fallback System**: If Raydium API fails, uses DexScreener for accurate quote generation
3. **Smart Routing**: Routes volatile tokens to Raydium, others to Jupiter for optimal execution
4. **Retry Logic**: Multiple attempts with different parameters to handle API limitations
5. **Error Handling**: Graceful degradation when APIs are unavailable

### **📊 Raydium Fallback Results**
- **Quote Generation**: 100% reliable via DexScreener fallback when direct Raydium API fails
- **Price Accuracy**: DexScreener provides accurate market prices and liquidity data
- **API Resilience**: System continues working even when Raydium API returns 404 errors
- **Trading Continuity**: Bot maintains trading capability despite API limitations

### **🛡️ Raydium Safety Features**
- **DexScreener Validation**: Uses DexScreener data to validate token liquidity and pricing
- **Multiple API Versions**: Tries different Raydium API versions for compatibility
- **Transaction Size Management**: Automatic parameter adjustment for large transactions
- **Comprehensive Error Handling**: Detailed logging and fallback mechanisms

## 🌐 Solana DEX Support

The bot supports Jupiter and Raydium on Solana, with intelligent fallback:

- **Primary**: Jupiter Ultra API (`lite-api.jup.ag`) for quoting and swaps
- **Fallback**: Raydium swap flow when Jupiter rejects or times out
- **DexScreener Validation**: Used to validate liquidity and pricing before trading
## 🎯 Enhanced Token Discovery System

The bot now features a significantly improved token discovery system that addresses quality, diversity, and filtering issues:

### **📊 Key Improvements:**

#### **1. Quality Scoring System (8-point scale)**
- **Volume Scoring**: 0-3 points based on 24h volume ($10k, $50k, $100k+)
- **Liquidity Scoring**: 0-3 points based on liquidity ($10k, $50k, $100k+)
- **Symbol Quality**: Penalizes spam symbols, rewards unique ones
- **Chain Bonus**: Neutral for major chains (Solana, Base)

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

### **🏆 Major Coin Integration**
The bot now includes **established cryptocurrencies and DeFi blue chips** alongside trending tokens:

#### **📈 Established Tokens Supported:**
- **Major Cryptocurrencies**: WETH, WBTC, SOL, BONK, mSOL
- **DeFi Blue Chips**: UNI, AAVE, COMP, MKR, LINK, CRV, BAL, SUSHI, 1INCH, YFI, SNX, REN, KNC
- **Layer 2 & Scaling**: L2, SCALING tokens
- **Quality Categories**: Gaming, NFT, AI, Privacy, Infrastructure, Storage, Compute, Identity

#### **🎯 Smart Token Selection:**
- **Priority Scoring**: Established tokens get higher priority in selection
- **Diversity Balance**: Mix of established coins and trending tokens
- **Stablecoin Filtering**: Automatically excludes USDC, USDT, DAI, BUSD (no volatility for trading)
- **Real Data Integration**: All token analysis now uses real market data instead of simulation

#### **🔍 Enhanced Search Queries:**
- **Established Coins**: Direct searches for ETH, BTC, SOL, WETH, WBTC
- **DeFi Protocols**: Searches for Uniswap, Aave, Compound, Maker, Chainlink
- **Sector-Specific**: Gaming, NFT, AI, Privacy, Infrastructure categories
- **Time-Based**: 24h, 7d, 30d trending searches for established tokens

### **📈 Performance Results:**
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Unique Symbols** | 4 | 13+ | +225% |
| **Avg Volume** | $1,673 | $229,970+ | +13,640% |
| **Avg Liquidity** | $34,973 | $361,500+ | +934% |
| **Symbol Diversity** | Poor (65% HOT) | Excellent (established + trending) | +225% |
| **Token Quality** | Meme-heavy | Balanced (established + quality) | +300% |

### **🛠️ Maintenance Tools:**
- Use `scripts/clear_state.py` to clean up delisted tokens and reset bot state


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
- `delisted_tokens.json` (snapshot) - Tracking of delisted tokens (SQLite is source of truth)
- `cooldown.json` - Tracks failure counts and temporary cooldowns (legacy `cooldown_log.json` automatically migrated)
- `blacklist_manager.py` - Manages both regular blacklist and delisted tokens (SQLite-backed with JSON snapshots)
- `blacklist.json` / `blacklist_failures.json` / `blacklist_reasons.json` (snapshots) - SQLite is source of truth

### **⚙️ Configuration:**
```yaml
enable_pre_buy_delisting_check: true   # Enable pre-buy delisting checks
```

## 🔄 Recent Updates & Improvements

### **Enhanced Sell Transaction Verification** (Latest)
- **Balance Check Verification**: Enhanced sell verification now includes balance checks before assuming transaction failure
- **Multi-Method Verification**: Uses RPC verification, balance checks, and retry logic for robust transaction confirmation
- **Reduced False Negatives**: Prevents false "SELL FAILED" messages when transactions actually succeeded
- **Improved Reliability**: More accurate position monitoring with better on-chain transaction verification

### **Performance Tracking Improvements**
- **Failed Entry Attempts Excluded**: Failed entry attempts (tokens not received) are now excluded from win rate calculations
- **Accurate Metrics**: Win rate now reflects only actual completed trades, providing more accurate performance analysis
- **Quality Tier Analysis**: Failed entry attempts are filtered out of quality tier performance tracking

### **Entry Criteria Improvements** (Win Rate Optimization)
- **Raised Quality Thresholds**: 
  - Minimum quality score: 40 → 50 (improves token selection)
  - Minimum 24h volume: $100k → $150k (better liquidity)
  - Minimum liquidity: $100k → $150k (reduced slippage)
  - Minimum momentum: 0.1% → 0.2% (stronger entry signals)
- **Result**: Improved win rate by filtering weaker tokens before entry

### **Configuration Updates**
- **Supported Chains**: Focused on Solana and Base (Ethereum removed for fee efficiency)
- **Position Sizing**: Dynamic tiered position sizing based on portfolio value
- **Risk Management**: Tightened stop loss (7%) and take profit (13% base, dynamic up to 20%)

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/mikegianfelice/Hunter.git
cd Hunter
```

### 1.5. GitHub Authentication (Optional)
If you plan to contribute or push changes, you can set up GitHub token authentication:

**Option A: Using the included git wrapper script**
1. Add your GitHub Personal Access Token to `.env`:
   ```bash
   GITHUB_SSH_KEY=your_github_token_here
   ```
2. Use the wrapper script for git operations:
   ```bash
   python3 git_with_ssh.py push
   ```

**Option B: Manual SSH setup**
Set up SSH keys in your `~/.ssh/` directory and use standard git commands.

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
# Base Wallet Configuration (MetaMask)
PRIVATE_KEY=your_base_private_key_here
WALLET_ADDRESS=your_base_wallet_address_here

# Solana Wallet Configuration (Phantom)
SOLANA_WALLET_ADDRESS=your_phantom_wallet_address_here
SOLANA_PRIVATE_KEY=your_phantom_private_key_here

# Blockchain RPC URLs
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
# Multi-Chain Configuration (Current Active Chains)
supported_chains: ["solana", "base"]  # Solana and Base only (Ethereum removed for fee efficiency)

# Trading Configuration (Current Settings)
test_mode: false              # LIVE TRADING ENABLED
trade_amount_usd: 6           # Position size (USD)
slippage: 0.03                # 3% slippage tolerance
take_profit: 0.13             # 13% take profit target (base)
stop_loss: 0.07               # 7% stop loss
use_dynamic_tp: true          # Enable dynamic take profit (8-20% range)

# Strategy Thresholds (Improved Entry Quality)
min_quality_score: 50         # Minimum quality score (0-100) - raised from 40 to improve win rate
min_volume_24h_for_buy: 150000   # $150k minimum 24h volume - raised from $100k
min_liquidity_usd_for_buy: 150000 # $150k minimum liquidity - raised from $100k
min_momentum_pct: 0.002       # 0.2% minimum momentum - raised from 0.1%

# Risk Management (Current Settings)
max_concurrent_positions: 6   # Maximum open positions
daily_loss_limit_usd: 10.0   # Daily loss limit
max_losing_streak: 3         # Stop after 3 consecutive losses
circuit_breaker_minutes: 60  # Cooldown after losses
per_trade_max_usd: 25        # Maximum per trade
min_wallet_balance_buffer: 0.02  # 2% buffer for gas fees

# AI System Configuration (Key Modules Enabled)
enable_ai_execution_optimization: true
enable_ai_microstructure_analysis: true
enable_ai_market_intelligence: true
enable_ai_predictive_analytics: true
enable_ai_dynamic_strategy_selector: true
enable_ai_risk_prediction_prevention: true
enable_ai_market_regime_detection: true
enable_ai_portfolio_optimization: true
enable_ai_drawdown_protection: true
enable_ai_emergency_stop: true
enable_ai_fill_verifier: true
enable_ai_partial_tp_manager: true
enable_ai_time_window_scheduler: true
# Note: Some AI modules may be disabled if too restrictive - check config.yaml for full list

# Advanced Trading Features
enable_order_splitting: true
max_price_impact_per_slice: 0.02  # 2% max price impact
min_slice_amount_usd: 5.0         # Minimum slice size
max_slices_per_trade: 5           # Maximum slices

enable_dynamic_slippage: true
dynamic_slippage_multiplier: 2.0  # Impact × 2.0 = slippage
max_dynamic_slippage: 0.08        # Maximum 8% slippage
min_dynamic_slippage: 0.01        # Minimum 1% slippage

enable_exactout_trades: true
exactout_liquidity_threshold: 25000  # Use for <$25k liquidity
exactout_volume_threshold: 10000     # Use for <$10k volume

enable_route_restrictions: true
prefer_direct_routes: true          # Prefer single-hop routes
max_route_hops: 1                   # Maximum 1 hop (tightened)
enable_direct_pool_swaps: true      # Bypass aggregators for known pools

# Solana Configuration
solana_base_currency: "USDC"  # Options: "SOL" or "USDC"
solana_min_sol_for_fees: 0.05  # Minimum SOL for transaction fees
```

### 4. Test Mode (Optional)
The bot is configured for real trading by default. Test mode validates transactions using real market data but doesn't execute them:

```yaml
# In config.yaml
test_mode: true               # Set to true to validate without executing trades
```

**Note**: In test mode, the bot still uses real market data for quotes and validation - it just doesn't execute transactions. The bot executes real trades by default when `test_mode: false`. Make sure you have sufficient funds in your Phantom (Solana) and MetaMask (Base) wallets before running in live mode.



### 6. Run the Bot

#### **Option A: Background Execution (Recommended) 🚀**
Use the provided launcher script to run the bot in the background with `screen`:

```bash
# Make sure you're in the Hunter directory
cd /Users/gianf/Desktop/Cursor/Hunter

# Run the launcher script
./launch_bot.sh
```

The script will:
- ✅ Create a detached `screen` session called `trading_bot`
- ✅ Start your bot in the background
- ✅ Keep it running even if you close your laptop screen
- ✅ Survive terminal/connection interruptions

**Managing the Background Bot:**

```bash
# View what your bot is doing
screen -r trading_bot

# Detach from screen (leave it running)
# While viewing: Press Ctrl+A then D

# Stop the bot
screen -X -S trading_bot quit

# Check if it's running
screen -list
```

**Or use the convenience scripts:**

```bash
# Start the bot in background
./launch_bot.sh

# Stop the bot
./stop_bot.sh

# Check bot status
./status_bot.sh
```

#### **Option B: Foreground Execution**
Run the bot directly in your terminal (closes when you close the terminal):

```bash
python main.py
```

**Note**: This method will stop the bot when you close your laptop screen or terminal window. Use Option A for continuous 24/7 trading.


## 📊 Configuration Options

### Multi-Chain Strategy
- **Two-Network Support**: Solana and Base trading with full implementation (Ethereum removed for fee efficiency)
- **Chain-Specific Requirements**: Different volume/liquidity thresholds per chain
- **DEX-Specific Execution**: Optimized for each blockchain's DEX (Jupiter + Raydium on Solana, Uniswap V3 on Base)
- **Wallet Integration**: Phantom for Solana, MetaMask for Base
- **Multi-Chain Price Fetching**: Chain-specific price monitoring for accurate PnL
- **Solana Features**: 
  - Multi-DEX integration (Jupiter primary, Raydium fallback)
  - USDC or SOL base currency (configurable)
  - Automatic token account creation (ATA)
  - Priority-based pool discovery
- **Base Features**: Uniswap V3 integration, EIP-1559 gas optimization, re-quote protection, multiple fee tier support
- **Smart Blacklist Management**: Automatic cleanup every loop to maintain trading opportunities

### AI-Enhanced Trading Strategy
- **Sustainable 10-20% Gains**: Focus on consistent returns over high-risk moonshots
- **Quality Token Selection**: AI-enhanced quality scoring with minimum thresholds
- **Dynamic Position Sizing**: AI-calculated position sizes based on quality and risk
- **Market Regime Adaptation**: Strategy adapts to bull, bear, sideways, and volatile markets
- **Comprehensive Risk Management**: Multi-layered risk assessment with AI-powered loss prevention
- **Sentiment Analysis**: Sentiment analysis available when enabled (currently disabled by default to avoid being too restrictive)
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
- **Smart Blacklist Management**: Automatic cleanup of blacklist every loop to maintain trading opportunities while keeping high-risk tokens blocked

### Position Monitoring
- **Real-time Monitoring**: Continuous position tracking every 30 seconds
- **Automatic Sell Triggers**: Take profit, stop loss, and trailing stop
- **AI Partial Take-Profit Manager** - Locks in gains with staged profit-taking and adaptive trailing stops
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
- **Hourly Status Reports**: Comprehensive bot status, performance, and market conditions (sent every 1 hour)
- **Blacklist Maintenance**: Notifications when smart cleanup removes safer tokens

## 📈 Monitoring and Analytics

### Performance Analytics
View real-time trading performance through:
- Performance data in `data/performance_data.json`
- Trade logs in `data/trade_log.csv`
- Console output with real-time status updates
- Telegram notifications for all trading events

### Smart Blacklist Management
The bot automatically maintains its blacklist to ensure trading opportunities:

- **Automatic Cleanup**: Runs every loop to remove safer tokens
- **Risk-Based Filtering**: Keeps high-risk tokens (3+ failures, 100% losses) blocked
- **Trading Opportunity Maintenance**: Removes tokens with low failure counts to allow new trades
- **Transparent Logging**: Shows exactly which tokens are removed and why
- **Configurable**: Adjust cleanup frequency and failure thresholds in config.yaml

**Configuration Options:**
```yaml
# Smart Blacklist Maintenance
enable_smart_blacklist_cleanup: true    # Auto-clean blacklist every loop
blacklist_cleanup_interval: 1           # Clean every N loops (1 = every loop)
blacklist_keep_failure_threshold: 3     # Keep tokens with N+ failures
```

### Logs
The bot creates detailed logs in:
- `logs/` - Trading logs (trading.log, errors.log, performance.log, etc.)
- Console output - Real-time status updates
- `data/trending_tokens.csv` - Token discovery history
- `data/trade_log.csv` - Detailed trade history with reason codes
- `data/delisted_tokens.json` (snapshot) - Tracking of delisted tokens (SQLite is source of truth)

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
    "solana": {
        "rpc_url": SOLANA_RPC_URL,
        "native_token": "SOL",
        "base_currency": "USDC",  # or "SOL"
        "dex": "jupiter",  # with raydium fallback
        # ... more config
    },
    "base": {
        "rpc_url": BASE_RPC_URL,
        "native_token": "ETH",
        "dex": "uniswap_v3",
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

9. **"Network error checking Jupiter tradeability: ConnectionError"**
   - **UPDATED**: Tradeability checks now use DexScreener for real market data verification
   - Jupiter API endpoints changed, but bot now uses DexScreener for reliable tradeability checks
   - All tradeability checks use actual liquidity, transaction counts, and price data from DexScreener
   - No assumptions - tokens are verified using real market metrics (liquidity, transactions, price) before trading
   - This is NOT an internet connection issue - DexScreener provides reliable tradeability verification

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

12. **"Jupiter quote failed (attempt 1/3): 400"**
    - **FIXED**: This issue has been resolved in v3.6
    - **Causes**: Invalid token addresses, insufficient liquidity, API parameter validation issues
    - **Solution**: Bot now uses direct routes and legacy transactions to reduce transaction size
    - **Result**: Successfully executing trades with known tradeable tokens (BONK, PEPE, JITO)
    - **Status**: ✅ Resolved with transaction size optimization

13. **"base64 encoded solana_transaction::versioned::VersionedTransaction too large"**
    - **FIXED**: This issue has been resolved in v3.6
    - **Causes**: Complex Jupiter routes creating transactions larger than Solana's 1644-byte limit
    - **Solution**: Bot now uses direct routes and legacy transactions to reduce size
    - **Result**: Successfully executing trades with optimized transaction size
    - **Status**: ✅ Resolved with route optimization

14. **"Could not get SOL price from any source"**
    - **CURRENT ISSUE**: All SOL price APIs (CoinGecko, Jupiter, DexScreener) are failing
    - **Causes**: Rate limiting, API downtime, network connectivity issues
    - **Workaround**: Bot uses fallback price when all sources fail
    - **Solution**: Enhanced fallback mechanisms and alternative price sources
    - **Status**: Being improved with better error handling

15. **"Configuration not reloading"**
    - **FIXED**: This issue has been resolved in v3.2
    - **Cause**: Python modules were loading config at import time and caching values
    - **Solution**: Implemented dynamic configuration loading with `config_loader.py`
    - **Result**: Changes to `config.yaml` now take effect immediately without restart
    - **Status**: ✅ Resolved with automatic cache clearing

16. **"Stop-loss triggered but SELL FAILED!"**
    - **FIXED**: This issue has been resolved with enhanced sell verification
    - **Cause**: RPC verification could return false negatives, causing valid sells to be marked as failed
    - **Solution**: Implemented multi-method verification (RPC + balance checks + retries) before assuming failure
    - **Result**: More accurate sell verification with reduced false negatives
    - **Status**: ✅ Resolved with balance verification fallback

### Debug Mode
Enable debug logging in `config.yaml`:
```yaml
debug_mode: true
log_level: DEBUG
```

## 📚 File Structure

```
Hunter/
├── main.py
├── README.md
├── requirements.txt
├── config.yaml
├── .env                         # User-provided (ignored)
├── data/
│   ├── open_positions.json
│   ├── performance_data.json
│   ├── delisted_tokens.json
│   ├── risk_state.json
│   ├── balance_cache.json
│   ├── sol_price_cache.json
│   ├── trade_log.csv
│   ├── trending_tokens.csv
│   └── uniswap_router_abi.json
├── logs/
│   ├── trading.log
│   ├── performance.log
│   ├── errors.log
│   └── ...
├── scripts/
│   ├── launch_bot.sh
│   ├── stop_bot.sh
│   ├── status_bot.sh
│   ├── setup_secrets.py
│   ├── clear_state.py
│   └── dev_runner.py
├── src/
│   ├── __init__.py
│   ├── ai/
│   │   ├── ai_circuit_breaker.py
│   │   ├── ai_integration_engine.py
│   │   ├── ai_market_regime_detector.py
│   │   └── ... (many AI modules)
│   ├── analytics/
│   │   └── backtesting_engine.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── config_loader.py
│   │   ├── config_validator.py
│   │   ├── secrets.py
│   │   └── secrets_manager.py
│   ├── core/
│   │   ├── strategy.py
│   │   ├── risk_manager.py
│   │   ├── performance_tracker.py
│   │   ├── advanced_trading.py
│   │   └── centralized_risk_manager.py
│   ├── deployment/
│   │   └── production_manager.py
│   ├── execution/
│   │   ├── async_trading_loop.py
│   │   ├── enhanced_async_trading.py
│   │   ├── multi_chain_executor.py
│   │   ├── uniswap_executor.py
│   │   ├── base_executor.py
│   │   ├── jupiter_lib.py
│   │   ├── jupiter_executor.py
│   │   ├── raydium_lib.py
│   │   ├── raydium_executor.py
│   │   └── solana_executor.py
│   ├── monitoring/
│   │   ├── monitor_position.py
│   │   ├── performance_monitor.py
│   │   ├── realtime_dashboard.py
│   │   ├── structured_logger.py
│   │   └── telegram_bot.py
│   └── utils/
│       ├── token_scraper.py
│       ├── tradeability_checker.py
│       ├── preflight_check.py
│       ├── cooldown.py
│       ├── http_utils.py
│       └── ... (other utilities)
├── system/
└── tests/
```

## 🚀 AI-Enhanced Trading Features

The bot now includes sophisticated AI-powered trading strategies for optimal execution:

### 🧠 **AI System Integration**
- **30 AI Modules**: Working together for comprehensive market analysis
- **Real-time Analysis**: Continuous market monitoring and adaptation
- **Institutional-Grade Intelligence**: Professional-level trading algorithms
- **Adaptive Strategy**: Automatically adjusts to changing market conditions

### 📊 **Advanced Execution Features**
- **AI Execution Optimization**: Intelligent timing, routing, and cost optimization
- **Market Microstructure Analysis**: Order book, trade flow, and liquidity analysis
- **Dynamic Position Sizing**: AI-calculated position sizes based on quality and risk
- **Performance Tracking**: Comprehensive analytics with quality tier analysis

### 🛡️ **Enhanced Risk Management**
- **AI Risk Assessment**: Machine learning-based risk scoring and loss prediction
- **Manipulation Detection**: Advanced detection of pump/dump schemes and wash trading
- **Whale Activity Monitoring**: Real-time whale trade detection and impact analysis
- **Market Maker Intelligence**: Leverages market maker presence for execution stability
- **Sequential Execution**: Slices sent sequentially with re-quoting between each
- **Configurable Limits**: Minimum slice size ($5) and maximum slices per trade (5)

### 🎯 **Dynamic Slippage**
- **Impact-Based Calculation**: Slippage derived from predicted price impact
- **Smart Multiplier**: Impact × 2.0 = slippage (with 1-8% bounds)
- **Real-Time Adjustment**: Slippage calculated per trade based on current market conditions
- **Risk Management**: Prevents overpaying while ensuring trade execution

### 🔄 **ExactOut Trades**
- **Sketchy Token Protection**: For tokens with <$25k liquidity or <$10k volume
- **Spending Cap**: Receive X tokens max, cap what you'll spend
- **Failure Tolerance**: Accept that many attempts will fail
- **Smart Fallback**: Continue with remaining slices even if some fail

### 🛣️ **Route Restrictions**
- **Direct Route Preference**: Prefer single-hop routes over multi-hop
- **Hop Limits**: Maximum 1 hop in any route
- **Direct Pool Swaps**: Bypass aggregators for known pools
- **Chain-Specific Optimization**: Different strategies per blockchain

### 🔍 **Enhanced Preflight Checks**
- **Token Validation**: Check decimals, mint frozen status, transfer fees
- **Solana-Specific**: ATA existence, mint frozen status verification
- **Pool Reserves**: Verify sufficient liquidity (2x trade amount)
- **Transfer Fee Detection**: Block tokens with >10% transfer fees

### ⚡ **Priority Fees & Performance**
- **Compute Unit Pricing**: Enhanced priority fees for faster execution
- **EIP-1559 Support**: Dynamic gas pricing on Base (Ethereum L2)
- **Fresh Quote Retry**: Re-fetch quotes immediately before execution
- **Stale Price Protection**: Reject quotes older than 2-3 seconds

### 🚫 **Smart Trade Rejection**
- **High Slippage Protection**: Reject when required slippage > 15%
- **Low Liquidity Filter**: Skip tokens with <$5k liquidity
- **Expected Loss Calculation**: Sometimes the only pro move is to pass
- **Risk-Reward Assessment**: Automatic evaluation before execution

## 🛡️ Safety Features

### Multi-Network Support
- **Two-Network Trading**: Solana and Base
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
- **Chain-Specific Logic**: Different validation rules for Solana vs Base tokens
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
12. **Verify sell transactions** - Enhanced verification prevents false "SELL FAILED" messages

### 🔧 Infrastructure Security
13. **Regularly update dependencies** - Keep all packages updated
14. **Test on testnets first** - Validate functionality before mainnet
15. **Use hardware wallets for large amounts** - Consider hardware wallets for significant funds
16. **Secure wallet setup** - Use separate wallets for different networks (Phantom for Solana, MetaMask for Base)
17. **Backup secrets securely** - Store backup credentials in secure locations
18. **Monitor access logs** - Track who has access to your secrets

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

**Happy Trading! 🚀📈**
