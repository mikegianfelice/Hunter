# 🚀 Crypto Trading Bot 100X

A sophisticated automated cryptocurrency trading bot designed for high-frequency trading on Uniswap V3. This bot uses advanced strategies including sentiment analysis, trend detection, and risk management to identify and execute profitable trades.

## ⚠️ **IMPORTANT DISCLAIMER**

**This bot is for educational and research purposes only. Cryptocurrency trading involves significant risk of loss. Never invest more than you can afford to lose. The developers are not responsible for any financial losses incurred from using this bot.**

## 🎯 Features

- **🔄 Real-time Token Scanning** - Monitors trending tokens across multiple platforms
- **📊 Sentiment Analysis** - Analyzes social media sentiment from Reddit and Nitter
- **⚡ High-Speed Execution** - Optimized for quick entry and exit strategies
- **🛡️ Risk Management** - Built-in position sizing and stop-loss mechanisms
- **📱 Telegram Notifications** - Real-time alerts for trades and bot status
- **📈 Performance Tracking** - Comprehensive logging and analytics
- **🔧 Configurable Strategy** - Easily adjustable parameters via config.yaml

## 📋 Prerequisites

- **Python 3.8+**
- **Web3.py** for blockchain interactions
- **Ethereum wallet** with ETH for gas fees
- **Private key** for wallet signing
- **Telegram Bot Token** (optional, for notifications)

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

### 3. Configure the Bot

#### Create Environment File
```bash
cp .env.example .env
```

#### Edit `.env` file with your credentials:
```env
# Wallet Configuration
PRIVATE_KEY=your_private_key_here
WALLET_ADDRESS=your_wallet_address_here

# Telegram Configuration (Optional)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# API Keys (Optional)
ETHERSCAN_API_KEY=your_etherscan_api_key
```

#### Configure Trading Parameters
Edit `config.yaml` to customize your trading strategy:

```yaml
# Trading Configuration
max_position_size: 0.1          # Maximum position size in ETH
min_liquidity: 10000            # Minimum liquidity required
max_slippage: 5                 # Maximum slippage percentage
gas_limit: 500000               # Gas limit for transactions

# Strategy Parameters
trending_threshold: 10          # Minimum trending score
sentiment_threshold: 0.6        # Minimum sentiment score
cooldown_period: 300            # Cooldown between trades (seconds)

# Risk Management
max_daily_loss: 0.05            # Maximum daily loss (5%)
stop_loss_percentage: 0.15      # Stop loss percentage (15%)
take_profit_percentage: 0.3     # Take profit percentage (30%)
```

### 4. Test Mode (Recommended First Step)
Before running with real money, test the bot in simulation mode:

```yaml
# In config.yaml
test_mode: true                 # Set to true for simulation only
```

### 5. Run the Bot
```bash
python main.py
```

## 📊 Configuration Options

### Trading Strategy
- **Trending Detection**: Monitors social media and DEX activity
- **Sentiment Analysis**: Analyzes Reddit and Twitter sentiment
- **Liquidity Analysis**: Ensures sufficient liquidity for trades
- **Volume Analysis**: Tracks trading volume patterns

### Risk Management
- **Position Sizing**: Automatic position size calculation
- **Stop Loss**: Configurable stop-loss levels
- **Take Profit**: Automatic profit-taking
- **Daily Limits**: Maximum daily loss protection
- **Cooldown Periods**: Prevents overtrading

### Gas Optimization
- **Dynamic Gas Pricing**: Adjusts gas prices based on network conditions
- **Gas Limit Management**: Optimizes transaction costs
- **Failed Transaction Handling**: Automatic retry mechanisms

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

## 📈 Monitoring and Analytics

### Dashboard
Open `trading_bot_dashboard.html` in your browser to view:
- Real-time trading performance
- Position tracking
- Profit/loss analysis
- Trading history

### Logs
The bot creates detailed logs in:
- `archives/` - Trading logs and data
- Console output - Real-time status updates

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

## 🛠️ Troubleshooting

### Common Issues

1. **"Insufficient funds"**
   - Ensure your wallet has enough ETH for gas fees
   - Check your private key is correct

2. **"Transaction failed"**
   - Increase gas limit in config.yaml
   - Check network congestion
   - Verify token contract is valid

3. **"No trending tokens found"**
   - Check internet connection
   - Verify API endpoints are accessible
   - Adjust trending thresholds

### Debug Mode
Enable debug logging in `config.yaml`:
```yaml
debug_mode: true
log_level: DEBUG
```

## 📚 File Structure

```
crypto_trading_bot_100x/
├── main.py                 # Main entry point
├── config.yaml            # Configuration file
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
├── executor.py           # Trade execution logic
├── strategy.py           # Trading strategy implementation
├── token_scraper.py      # Token discovery
├── sentiment_scraper.py  # Sentiment analysis
├── risk_manager.py       # Risk management
├── telegram_bot.py       # Telegram notifications
├── utils.py              # Utility functions
└── uniswap_router_abi.json # Uniswap contract ABI
```

## 🔒 Security Best Practices

1. **Never share your private key**
2. **Use a dedicated trading wallet**
3. **Start with small amounts**
4. **Monitor the bot regularly**
5. **Keep your `.env` file secure**
6. **Regularly update dependencies**

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review the configuration options
- Test in simulation mode first
- Start with small position sizes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Risk Warning

**Cryptocurrency trading is highly risky and volatile. This bot is provided as-is without any guarantees. Always:**
- Test thoroughly in simulation mode
- Start with small amounts
- Monitor performance closely
- Never invest more than you can afford to lose
- Understand the risks involved

---

**Happy Trading! 🚀📈**
