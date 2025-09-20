#!/usr/bin/env python3
"""
Wallet Balance Checker - Compare actual wallet holdings against bot's tracked positions
"""

import json
import requests
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from secrets import SOLANA_RPC_URL, SOLANA_WALLET_ADDRESS
from config_loader import get_config

@dataclass
class TokenBalance:
    mint: str
    symbol: str
    amount: float
    decimals: int
    value_usd: float = 0.0

class WalletBalanceChecker:
    def __init__(self):
        self.wallet_address = SOLANA_WALLET_ADDRESS
        self.rpc_url = SOLANA_RPC_URL
        
    def get_solana_token_balances(self) -> List[TokenBalance]:
        """Get all token balances from Solana wallet"""
        try:
            # Get token accounts for the wallet
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTokenAccountsByOwner",
                "params": [
                    self.wallet_address,
                    {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
                    {"encoding": "jsonParsed"}
                ]
            }
            
            response = requests.post(self.rpc_url, json=payload, timeout=15)
            if response.status_code != 200:
                print(f"❌ RPC request failed: {response.status_code}")
                return []
                
            data = response.json()
            if "error" in data:
                print(f"❌ RPC error: {data['error']}")
                return []
                
            token_accounts = data.get("result", {}).get("value", [])
            balances = []
            
            for account in token_accounts:
                try:
                    parsed_data = account.get("account", {}).get("data", {}).get("parsed", {})
                    info = parsed_data.get("info", {})
                    
                    mint = info.get("mint", "")
                    amount = float(info.get("tokenAmount", {}).get("amount", 0))
                    decimals = int(info.get("tokenAmount", {}).get("decimals", 0))
                    
                    # Convert to human readable amount
                    human_amount = amount / (10 ** decimals) if decimals > 0 else amount
                    
                    # Skip zero balances
                    if human_amount <= 0:
                        continue
                        
                    # Get token symbol and price
                    symbol = self._get_token_symbol(mint)
                    value_usd = self._get_token_price_usd(mint, human_amount)
                    
                    balances.append(TokenBalance(
                        mint=mint,
                        symbol=symbol,
                        amount=human_amount,
                        decimals=decimals,
                        value_usd=value_usd
                    ))
                    
                except Exception as e:
                    print(f"⚠️ Error parsing token account: {e}")
                    continue
                    
            return balances
            
        except Exception as e:
            print(f"❌ Error getting token balances: {e}")
            return []
    
    def _get_token_symbol(self, mint: str) -> str:
        """Get token symbol from mint address with caching"""
        # Only keep the most essential hardcoded mappings (SOL, USDC, USDT)
        # These are system tokens that rarely change
        essential_tokens = {
            "So11111111111111111111111111111111111111112": "SOL",
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": "USDC", 
            "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB": "USDT",
        }
        
        if mint in essential_tokens:
            return essential_tokens[mint]
        
        # For all other tokens, use dynamic lookup with caching
        return self._get_token_symbol_dynamic(mint)
    
    def _get_token_symbol_dynamic(self, mint: str) -> str:
        """Dynamically fetch token symbol from DexScreener with caching"""
        # Check cache first (you could implement file-based caching here)
        # For now, always fetch from API but log the lookup
        
        try:
            url = f"https://api.dexscreener.com/latest/dex/tokens/{mint}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                pairs = data.get("pairs", [])
                if pairs:
                    symbol = pairs[0].get("baseToken", {}).get("symbol", "UNKNOWN")
                    print(f"🔍 Dynamic lookup: {mint[:8]}...{mint[-8:]} = {symbol}")
                    return symbol
        except Exception as e:
            print(f"⚠️ Dynamic lookup failed for {mint[:8]}...{mint[-8:]}: {e}")
            
        return f"UNKNOWN_{mint[:8]}"
    
    def _get_token_price_usd(self, mint: str, amount: float) -> float:
        """Get USD value of token amount"""
        try:
            # Try DexScreener first
            url = f"https://api.dexscreener.com/latest/dex/tokens/{mint}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                pairs = data.get("pairs", [])
                if pairs:
                    price = float(pairs[0].get("priceUsd", 0))
                    return amount * price
        except Exception:
            pass
            
        return 0.0
    
    def load_open_positions(self) -> Dict:
        """Load bot's tracked open positions"""
        try:
            with open("open_positions.json", "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading open positions: {e}")
            return {}
    
    def compare_wallet_vs_positions(self) -> Tuple[List[TokenBalance], List[str], List[str]]:
        """
        Compare actual wallet holdings vs bot's tracked positions
        Returns: (actual_holdings, missing_from_wallet, extra_in_wallet)
        """
        print("🔍 Checking wallet balances vs tracked positions...")
        
        # Get actual wallet balances
        wallet_balances = self.get_solana_token_balances()
        print(f"📊 Found {len(wallet_balances)} tokens in wallet")
        
        # Get tracked positions
        tracked_positions = self.load_open_positions()
        print(f"📋 Bot tracking {len(tracked_positions)} positions")
        
        # Find tokens that are tracked but missing from wallet
        missing_from_wallet = []
        wallet_mints = {balance.mint for balance in wallet_balances}
        
        for mint, position_data in tracked_positions.items():
            if mint not in wallet_mints:
                symbol = position_data.get("symbol", "UNKNOWN")
                missing_from_wallet.append(f"{symbol} ({mint[:8]}...{mint[-8:]})")
                print(f"❌ Missing from wallet: {symbol} ({mint[:8]}...{mint[-8:]})")
        
        # Find tokens that are in wallet but not tracked
        extra_in_wallet = []
        tracked_mints = set(tracked_positions.keys())
        
        for balance in wallet_balances:
            if balance.mint not in tracked_mints:
                extra_in_wallet.append(f"{balance.symbol} ({balance.amount:.6f} tokens, ${balance.value_usd:.2f})")
                print(f"➕ Extra in wallet: {balance.symbol} ({balance.amount:.6f} tokens, ${balance.value_usd:.2f})")
        
        return wallet_balances, missing_from_wallet, extra_in_wallet
    
    def cleanup_phantom_positions(self, dry_run: bool = True) -> List[str]:
        """Remove positions that don't exist in wallet"""
        wallet_balances, missing_from_wallet, _ = self.compare_wallet_vs_positions()
        
        if not missing_from_wallet:
            print("✅ No phantom positions found")
            return []
        
        wallet_mints = {balance.mint for balance in wallet_balances}
        tracked_positions = self.load_open_positions()
        
        phantom_positions = []
        cleaned_positions = {}
        
        for mint, position_data in tracked_positions.items():
            if mint not in wallet_mints:
                symbol = position_data.get("symbol", "UNKNOWN")
                phantom_positions.append(f"{symbol} ({mint[:8]}...{mint[-8:]})")
                print(f"🧹 Would remove phantom position: {symbol}")
            else:
                cleaned_positions[mint] = position_data
        
        if not dry_run and phantom_positions:
            # Save cleaned positions
            with open("open_positions.json", "w") as f:
                json.dump(cleaned_positions, f, indent=2)
            print(f"✅ Removed {len(phantom_positions)} phantom positions")
        elif dry_run:
            print(f"🔍 Dry run: Would remove {len(phantom_positions)} phantom positions")
        
        return phantom_positions

def main():
    """Main function to run wallet balance check"""
    checker = WalletBalanceChecker()
    
    print("🔍 Wallet Balance Checker")
    print("=" * 50)
    
    # Compare wallet vs positions
    wallet_balances, missing, extra = checker.compare_wallet_vs_positions()
    
    print("\n📊 Wallet Holdings:")
    total_value = 0
    for balance in wallet_balances:
        print(f"   • {balance.symbol}: {balance.amount:.6f} tokens (${balance.value_usd:.2f})")
        total_value += balance.value_usd
    
    print(f"\n💰 Total Wallet Value: ${total_value:.2f}")
    
    if missing:
        print(f"\n❌ Missing from wallet ({len(missing)}):")
        for item in missing:
            print(f"   • {item}")
    
    if extra:
        print(f"\n➕ Extra in wallet ({len(extra)}):")
        for item in extra:
            print(f"   • {item}")
    
    # Offer to clean up phantom positions
    if missing:
        print(f"\n🧹 Found {len(missing)} phantom positions")
        response = input("Clean up phantom positions? (y/N): ").strip().lower()
        if response == 'y':
            checker.cleanup_phantom_positions(dry_run=False)
        else:
            print("🔍 Dry run - showing what would be cleaned:")
            checker.cleanup_phantom_positions(dry_run=True)
    
    print("\n✅ Wallet balance check complete!")

if __name__ == "__main__":
    main()
