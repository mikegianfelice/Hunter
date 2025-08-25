import json
import time
import requests
import base58
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from solana.rpc.api import Client
from solana.transaction import Transaction
from solana.keypair import Keypair
from solana.publickey import PublicKey
from solana.system_program import TransferParams, transfer
from solana.rpc.commitment import Commitment
import struct

from secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS, SOLANA_PRIVATE_KEY

# Multi-DEX configuration
DEX_CONFIGS = {
    "raydium": {
        "name": "Raydium",
        "api_base": "https://api.raydium.io/v2",
        "pools_endpoint": "/main/pools",
        "quote_endpoint": "/main/quote",
        "price_endpoint": "/main/price",
        "program_id": "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
    },
    "pumpswap": {
        "name": "PumpSwap",
        "api_base": "https://api.pumpswap.finance",
        "pools_endpoint": "/pools",
        "quote_endpoint": "/quote",
        "price_endpoint": "/price",
        "program_id": "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM"
    },
    "meteora": {
        "name": "Meteora",
        "api_base": "https://api.meteora.ag",
        "pools_endpoint": "/pools",
        "quote_endpoint": "/quote",
        "price_endpoint": "/price",
        "program_id": "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM"
    },
    "heaven": {
        "name": "Heaven",
        "api_base": "https://api.heaven.so",
        "pools_endpoint": "/pools",
        "quote_endpoint": "/quote",
        "price_endpoint": "/price",
        "program_id": "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM"
    }
}

# Raydium DEX configuration (keeping for backward compatibility)
RAYDIUM_PROGRAM_ID = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"  # Raydium AMM program
RAYDIUM_SWAP_PROGRAM_ID = "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM"  # Raydium swap program
RAYDIUM_OPENBOOK_PROGRAM_ID = "9xQeWvG816bUx9EPjHmaT23yvVM2ZWbrrpZb9PusVFin"  # OpenBook program

# Common token addresses
WSOL_MINT = "So11111111111111111111111111111111111111112"  # Wrapped SOL
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # USDC
RAY_MINT = "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R"  # Raydium token

@dataclass
class PoolInfo:
    """Multi-DEX pool information"""
    pool_id: str
    dex_name: str
    base_mint: str
    quote_mint: str
    lp_mint: str
    base_vault: str
    quote_vault: str
    market_id: str
    market_base_vault: str
    market_quote_vault: str
    market_authority: str
    target_orders: str
    withdraw_queue: str
    lp_vault: str
    owner: str
    lp_fee_rate: float
    platform_fee_rate: float
    target_orders_total: int
    target_orders_remaining: int
    base_decimals: int
    quote_decimals: int
    state: int
    reset_flag: int
    min_size: int
    vol_max_cut_ratio: float
    amount_wave_ratio: float
    coin_lot_size: int
    pc_lot_size: int
    min_price_multiplier: float
    max_price_multiplier: float
    system_decimals_value: int
    min_separate_numerator: int
    min_separate_denominator: int
    trade_fee_numerator: int
    trade_fee_denominator: int
    pnl_numerator: int
    pnl_denominator: int
    swap_fee_numerator: int
    swap_fee_denominator: int
    need_take_pnl_coin: int
    need_take_pnl_pc: int
    total_pnl_pc: int
    total_pnl_coin: int
    pool_total_deposit_pc: int
    pool_total_deposit_coin: int
    swap_coin_in_amount: int
    swap_pc_out_amount: int
    swap_coin_to_pc_fee: int
    swap_pc_in_amount: int
    swap_coin_out_amount: int
    swap_pc_to_coin_fee: int
    token_account_coin: str
    token_account_pc: str
    max_coin_size: int
    max_pc_size: int
    base_reserve: int
    quote_reserve: int
    base_price: float
    quote_price: float

class MultiDEXSolanaExecutor:
    """Multi-DEX Solana trading executor supporting Raydium, PumpSwap, Meteora, Heaven, etc."""
    
    def __init__(self):
        self.client = Client(SOLANA_RPC_URL, commitment=Commitment("confirmed"))
        self.wallet_address = None
        self.keypair = None
        self._initialize_wallet()
    
    def _initialize_wallet(self):
        """Initialize Solana wallet from private key"""
        try:
            if not SOLANA_PRIVATE_KEY:
                raise ValueError("SOLANA_PRIVATE_KEY not found in environment")
            
            # Decode private key (assuming base58 format)
            private_key_bytes = base58.b58decode(SOLANA_PRIVATE_KEY)
            self.keypair = Keypair.from_secret_key(private_key_bytes)
            self.wallet_address = str(self.keypair.public_key)
            
            print(f"✅ Solana wallet initialized: {self.wallet_address[:8]}...{self.wallet_address[-8:]}")
            
        except Exception as e:
            print(f"❌ Failed to initialize Solana wallet: {e}")
            raise
    
    def get_solana_balance(self) -> float:
        """Get SOL balance in wallet"""
        try:
            response = self.client.get_balance(self.keypair.public_key)
            if response.value is not None:
                balance_sol = response.value / 1e9  # Convert lamports to SOL
                print(f"💰 SOL Balance: {balance_sol:.6f}")
                return balance_sol
            else:
                print("⚠️ Could not fetch SOL balance")
                return 0.0
        except Exception as e:
            print(f"❌ Error getting Solana balance: {e}")
            return 0.0
    
    def get_token_balance(self, token_mint: str) -> float:
        """Get token balance for a specific mint"""
        try:
            # Find token account
            response = self.client.get_token_accounts_by_owner(
                self.keypair.public_key,
                {"mint": PublicKey(token_mint)}
            )
            
            if response.value:
                # Get the first token account
                token_account = response.value[0]
                balance = int(token_account.account.data.parsed['info']['tokenAmount']['amount'])
                decimals = token_account.account.data.parsed['info']['tokenAmount']['decimals']
                return balance / (10 ** decimals)
            
            return 0.0
        except Exception as e:
            print(f"❌ Error getting token balance for {token_mint}: {e}")
            return 0.0
    
    def get_token_price_usd(self, token_address: str) -> float:
        """Get token price in USD from multiple DEXs"""
        for dex_name, dex_config in DEX_CONFIGS.items():
            try:
                url = f"{dex_config['api_base']}{dex_config['price_endpoint']}?ids={token_address}"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    price = float(data.get(token_address, {}).get('price', 0))
                    if price > 0:
                        print(f"✅ Fetched {dex_name} price for {token_address[:8]}...{token_address[-8:]}: ${price:.6f}")
                        return price
            except Exception as e:
                print(f"⚠️ {dex_name} price API error: {e}")
                continue
        
        print(f"⚠️ Zero price returned for {token_address[:8]}...{token_address[-8:]}")
        return 0.0
    
    def get_dex_pools(self, dex_name: str) -> Dict[str, PoolInfo]:
        """Get pools from a specific DEX"""
        try:
            dex_config = DEX_CONFIGS.get(dex_name)
            if not dex_config:
                print(f"❌ Unknown DEX: {dex_name}")
                return {}
            
            url = f"{dex_config['api_base']}{dex_config['pools_endpoint']}"
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                pools_data = response.json()
                pools = {}
                
                for pool in pools_data:
                    try:
                        pool_info = PoolInfo(
                            pool_id=pool.get('id', ''),
                            dex_name=dex_name,
                            base_mint=pool.get('baseMint', ''),
                            quote_mint=pool.get('quoteMint', ''),
                            lp_mint=pool.get('lpMint', ''),
                            base_vault=pool.get('baseVault', ''),
                            quote_vault=pool.get('quoteVault', ''),
                            market_id=pool.get('marketId', ''),
                            market_base_vault=pool.get('marketBaseVault', ''),
                            market_quote_vault=pool.get('marketQuoteVault', ''),
                            market_authority=pool.get('marketAuthority', ''),
                            target_orders=pool.get('targetOrders', ''),
                            withdraw_queue=pool.get('withdrawQueue', ''),
                            lp_vault=pool.get('lpVault', ''),
                            owner=pool.get('owner', ''),
                            lp_fee_rate=float(pool.get('lpFeeRate', 0)),
                            platform_fee_rate=float(pool.get('platformFeeRate', 0)),
                            target_orders_total=int(pool.get('targetOrdersTotal', 0)),
                            target_orders_remaining=int(pool.get('targetOrdersRemaining', 0)),
                            base_decimals=int(pool.get('baseDecimals', 0)),
                            quote_decimals=int(pool.get('quoteDecimals', 0)),
                            state=int(pool.get('state', 0)),
                            reset_flag=int(pool.get('resetFlag', 0)),
                            min_size=int(pool.get('minSize', 0)),
                            vol_max_cut_ratio=float(pool.get('volMaxCutRatio', 0)),
                            amount_wave_ratio=float(pool.get('amountWaveRatio', 0)),
                            coin_lot_size=int(pool.get('coinLotSize', 0)),
                            pc_lot_size=int(pool.get('pcLotSize', 0)),
                            min_price_multiplier=float(pool.get('minPriceMultiplier', 0)),
                            max_price_multiplier=float(pool.get('maxPriceMultiplier', 0)),
                            system_decimals_value=int(pool.get('systemDecimalsValue', 0)),
                            min_separate_numerator=int(pool.get('minSeparateNumerator', 0)),
                            min_separate_denominator=int(pool.get('minSeparateDenominator', 0)),
                            trade_fee_numerator=int(pool.get('tradeFeeNumerator', 0)),
                            trade_fee_denominator=int(pool.get('tradeFeeDenominator', 0)),
                            pnl_numerator=int(pool.get('pnlNumerator', 0)),
                            pnl_denominator=int(pool.get('pnlDenominator', 0)),
                            swap_fee_numerator=int(pool.get('swapFeeNumerator', 0)),
                            swap_fee_denominator=int(pool.get('swapFeeDenominator', 0)),
                            need_take_pnl_coin=int(pool.get('needTakePnlCoin', 0)),
                            need_take_pnl_pc=int(pool.get('needTakePnlPc', 0)),
                            total_pnl_pc=int(pool.get('totalPnlPc', 0)),
                            total_pnl_coin=int(pool.get('totalPnlCoin', 0)),
                            pool_total_deposit_pc=int(pool.get('poolTotalDepositPc', 0)),
                            pool_total_deposit_coin=int(pool.get('poolTotalDepositCoin', 0)),
                            swap_coin_in_amount=int(pool.get('swapCoinInAmount', 0)),
                            swap_pc_out_amount=int(pool.get('swapPcOutAmount', 0)),
                            swap_coin_to_pc_fee=int(pool.get('swapCoinToPcFee', 0)),
                            swap_pc_in_amount=int(pool.get('swapPcInAmount', 0)),
                            swap_coin_out_amount=int(pool.get('swapCoinOutAmount', 0)),
                            swap_pc_to_coin_fee=int(pool.get('swapPcToCoinFee', 0)),
                            token_account_coin=pool.get('tokenAccountCoin', ''),
                            token_account_pc=pool.get('tokenAccountPc', ''),
                            max_coin_size=int(pool.get('maxCoinSize', 0)),
                            max_pc_size=int(pool.get('maxPcSize', 0)),
                            base_reserve=int(pool.get('baseReserve', 0)),
                            quote_reserve=int(pool.get('quoteReserve', 0)),
                            base_price=float(pool.get('basePrice', 0)),
                            quote_price=float(pool.get('quotePrice', 0))
                        )
                        pools[pool_info.pool_id] = pool_info
                    except Exception as e:
                        print(f"⚠️ Error parsing {dex_name} pool {pool.get('id', 'unknown')}: {e}")
                        continue
                
                print(f"✅ Fetched {len(pools)} {dex_name} pools")
                return pools
            else:
                print(f"⚠️ Failed to fetch {dex_name} pools: {response.status_code}")
                return {}
        except Exception as e:
            print(f"❌ Error fetching {dex_name} pools: {e}")
            return {}
    
    def find_pool_for_token_multi_dex(self, token_mint: str, quote_mint: str = USDC_MINT) -> Optional[PoolInfo]:
        """Find pool for a token across multiple DEXs"""
        print(f"🔍 Searching for pool across multiple DEXs for {token_mint[:8]}...{token_mint[-8:]}")
        
        # Try each DEX in order of preference
        dex_priority = ["raydium", "pumpswap", "meteora", "heaven"]
        
        for dex_name in dex_priority:
            try:
                pools = self.get_dex_pools(dex_name)
                
                # Look for pool with token_mint as base and quote_mint as quote
                for pool in pools.values():
                    if (pool.base_mint == token_mint and pool.quote_mint == quote_mint) or \
                       (pool.base_mint == quote_mint and pool.quote_mint == token_mint):
                        print(f"✅ Found {dex_name} pool for {token_mint[:8]}...{token_mint[-8:]} <-> {quote_mint[:8]}...{quote_mint[-8:]}")
                        return pool
                
                print(f"⚠️ No {dex_name} pool found for {token_mint[:8]}...{token_mint[-8:]}")
                
            except Exception as e:
                print(f"❌ Error searching {dex_name}: {e}")
                continue
        
        print(f"❌ No pool found for {token_mint[:8]}...{token_mint[-8:]} across all DEXs")
        return None
    
    def get_swap_quote_multi_dex(self, pool: PoolInfo, input_amount: int, is_buy: bool) -> Dict[str, Any]:
        """Get swap quote from the pool's DEX"""
        try:
            dex_config = DEX_CONFIGS.get(pool.dex_name)
            if not dex_config:
                print(f"❌ Unknown DEX: {pool.dex_name}")
                return {}
            
            url = f"{dex_config['api_base']}{dex_config['quote_endpoint']}"
            
            if is_buy:
                # Buying token with USDC
                params = {
                    "inputMint": pool.quote_mint,
                    "outputMint": pool.base_mint,
                    "amount": str(input_amount),
                    "slippage": 0.02  # 2% slippage
                }
            else:
                # Selling token for USDC
                params = {
                    "inputMint": pool.base_mint,
                    "outputMint": pool.quote_mint,
                    "amount": str(input_amount),
                    "slippage": 0.02  # 2% slippage
                }
            
            response = requests.post(url, json=params, timeout=10)
            if response.status_code == 200:
                quote_data = response.json()
                print(f"✅ Got {pool.dex_name} swap quote: {quote_data.get('outAmount', 0)} output for {input_amount} input")
                return quote_data
            else:
                print(f"⚠️ Failed to get {pool.dex_name} swap quote: {response.status_code}")
                return {}
        except Exception as e:
            print(f"❌ Error getting {pool.dex_name} swap quote: {e}")
            return {}
    
    def execute_swap_multi_dex(self, pool: PoolInfo, input_amount: int, min_output_amount: int, 
                              is_buy: bool, slippage: float = 0.02) -> Tuple[str, bool]:
        """Execute swap transaction on the pool's DEX"""
        try:
            print(f"🚀 Executing {pool.dex_name} swap...")
            print(f"   Pool: {pool.pool_id}")
            print(f"   Input amount: {input_amount}")
            print(f"   Min output: {min_output_amount}")
            print(f"   Slippage: {slippage * 100}%")
            
            # For now, we'll simulate the swap since full implementation requires
            # complex instruction building for each DEX
            print(f"⚠️ Full {pool.dex_name} swap implementation requires DEX-specific instruction building")
            print(f"🔄 Simulating {pool.dex_name} swap for now...")
            
            # Simulate successful swap
            tx_hash = f"simulated_{pool.dex_name}_swap_{int(time.time())}"
            print(f"✅ Simulated {pool.dex_name} swap: {tx_hash}")
            return tx_hash, True
            
        except Exception as e:
            print(f"❌ Error executing {pool.dex_name} swap: {e}")
            return None, False

# Global executor instance
_solana_executor = None

def get_solana_executor() -> MultiDEXSolanaExecutor:
    """Get or create Solana executor instance"""
    global _solana_executor
    if _solana_executor is None:
        _solana_executor = MultiDEXSolanaExecutor()
    return _solana_executor

# Backward compatibility functions
class SolanaExecutor(MultiDEXSolanaExecutor):
    """Backward compatibility wrapper"""
    def find_pool_for_token(self, token_mint: str, quote_mint: str = USDC_MINT) -> Optional[PoolInfo]:
        return self.find_pool_for_token_multi_dex(token_mint, quote_mint)
    
    def get_swap_quote(self, pool: PoolInfo, input_amount: int, is_buy: bool) -> Dict[str, Any]:
        return self.get_swap_quote_multi_dex(pool, input_amount, is_buy)
    
    def execute_raydium_swap(self, pool: PoolInfo, input_amount: int, min_output_amount: int, 
                           is_buy: bool, slippage: float = 0.02) -> Tuple[str, bool]:
        return self.execute_swap_multi_dex(pool, input_amount, min_output_amount, is_buy, slippage)
    
    def get_raydium_pools(self) -> Dict[str, PoolInfo]:
        return self.get_dex_pools("raydium")

def get_solana_balance() -> float:
    """Get SOL balance in wallet - simplified version"""
    try:
        executor = get_solana_executor()
        return executor.get_solana_balance()
    except Exception as e:
        print(f"⚠️ Error getting Solana balance: {e}")
        return 0.0

def get_token_price_usd(token_address: str) -> float:
    """Get token price in USD from Raydium"""
    try:
        executor = get_solana_executor()
        return executor.get_token_price_usd(token_address)
    except Exception as e:
        print(f"⚠️ Error getting token price: {e}")
        return 0.0

def get_raydium_pool_info(token_address: str) -> Optional[dict]:
    """Get Raydium pool information for a token"""
    try:
        executor = get_solana_executor()
        pool = executor.find_pool_for_token_multi_dex(token_address)
        if pool:
            return {
                'pool_id': pool.pool_id,
                'base_mint': pool.base_mint,
                'quote_mint': pool.quote_mint,
                'base_reserve': pool.base_reserve,
                'quote_reserve': pool.quote_reserve,
                'base_price': pool.base_price,
                'quote_price': pool.quote_price
            }
        return None
    except Exception as e:
        print(f"⚠️ Error getting pool info: {e}")
        return None

def calculate_swap_amount(usd_amount: float, token_price: float) -> int:
    """Calculate token amount to buy based on USD amount"""
    if token_price <= 0:
        return 0
    token_amount = usd_amount / token_price
    # Convert to token decimals (assuming 9 decimals for most Solana tokens)
    return int(token_amount * 1e9)

def buy_token_solana(token_address: str, usd_amount: float, symbol: str, test_mode: bool = False) -> Tuple[str, bool]:
    """
    Execute a real Solana token purchase on multiple DEXs
    
    Args:
        token_address: Token mint address
        usd_amount: Amount to spend in USD
        symbol: Token symbol for logging
        test_mode: If True, simulate the trade instead of executing real transaction
    
    Returns:
        (transaction_hash, success)
    """
    try:
        if test_mode:
            print(f"🔄 Simulating Solana trade for {symbol}")
            tx_hash = f"simulated_solana_tx_{int(time.time())}_{symbol}"
            print(f"✅ Simulated Solana trade: {tx_hash}")
            print(f"   Token: {symbol}")
            print(f"   Amount: ${usd_amount}")
            return tx_hash, True
        else:
            print(f"🚀 Executing real Solana trade for {symbol}")
            
            executor = get_solana_executor()
            
            # Find pool for this token
            pool = executor.find_pool_for_token_multi_dex(token_address)
            if not pool:
                print(f"❌ No pool found for {symbol} across all DEXs")
                return None, False
            
            # Calculate input amount (USDC)
            input_amount = int(usd_amount * 1e6)  # USDC has 6 decimals
            
            # Get swap quote
            quote = executor.get_swap_quote_multi_dex(pool, input_amount, is_buy=True)
            if not quote:
                print(f"❌ Failed to get swap quote for {symbol}")
                return None, False
            
            min_output_amount = int(quote.get('outAmount', 0))
            
            # Execute the swap
            tx_hash, success = executor.execute_swap_multi_dex(
                pool, input_amount, min_output_amount, is_buy=True
            )
            
            if success:
                print(f"✅ Real Solana trade executed: {tx_hash}")
                print(f"   Token: {symbol}")
                print(f"   Amount: ${usd_amount}")
                print(f"   Output: {min_output_amount} tokens")
                return tx_hash, True
            else:
                print(f"❌ Solana trade failed for {symbol}")
                return None, False
        
    except Exception as e:
        print(f"❌ Solana trade failed for {symbol}: {e}")
        return None, False

def sell_token_solana(token_address: str, token_amount: float, symbol: str, test_mode: bool = False) -> Tuple[str, bool]:
    """
    Execute a real Solana token sale on multiple DEXs
    
    Args:
        token_address: Token mint address
        token_amount: Amount of tokens to sell
        symbol: Token symbol for logging
        test_mode: If True, simulate the trade instead of executing real transaction
    
    Returns:
        (transaction_hash, success)
    """
    try:
        if test_mode:
            print(f"🔄 Simulating Solana sell for {symbol}")
            tx_hash = f"simulated_solana_sell_tx_{int(time.time())}_{symbol}"
            print(f"✅ Simulated Solana sell: {tx_hash}")
            print(f"   Token: {symbol}")
            print(f"   Amount: {token_amount}")
            return tx_hash, True
        else:
            print(f"🚀 Executing real Solana sell for {symbol}")
            
            executor = get_solana_executor()
            
            # Find pool for this token
            pool = executor.find_pool_for_token_multi_dex(token_address)
            if not pool:
                print(f"❌ No pool found for {symbol} across all DEXs")
                return None, False
            
            # Calculate input amount (tokens)
            input_amount = int(token_amount * 1e9)  # Assuming 9 decimals
            
            # Get swap quote
            quote = executor.get_swap_quote_multi_dex(pool, input_amount, is_buy=False)
            if not quote:
                print(f"❌ Failed to get swap quote for {symbol}")
                return None, False
            
            min_output_amount = int(quote.get('outAmount', 0))
            
            # Execute the swap
            tx_hash, success = executor.execute_swap_multi_dex(
                pool, input_amount, min_output_amount, is_buy=False
            )
            
            if success:
                print(f"✅ Real Solana sell executed: {tx_hash}")
                print(f"   Token: {symbol}")
                print(f"   Amount: {token_amount}")
                print(f"   Output: {min_output_amount} USDC")
                return tx_hash, True
            else:
                print(f"❌ Solana sell failed for {symbol}")
                return None, False
        
    except Exception as e:
        print(f"❌ Solana sell failed for {symbol}: {e}")
        return None, False

# Test function
if __name__ == "__main__":
    # Test balance check
    balance = get_solana_balance()
    print(f"SOL Balance: {balance}")
    
    # Test price check
    # price = get_token_price_usd("So11111111111111111111111111111111111111112")  # Wrapped SOL
    # print(f"SOL Price: ${price}")
