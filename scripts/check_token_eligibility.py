#!/usr/bin/env python3
"""
Check if a token passes all validation tests for trading eligibility
"""
import sys
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.http_utils import get_json
from src.utils.tradeability_checker import check_jupiter_tradeability, check_raydium_tradeability, is_token_tradeable
from src.utils.token_scraper import (
    is_valid_token_data,
    calculate_token_score,
    is_promotional_content,
    EXCLUDED_KEYWORDS,
    ENFORCE_KEYWORDS
)
from src.core.advanced_trading import AdvancedTrading
from src.utils.blacklist_manager import is_blacklisted

def check_token_eligibility(token_address: str):
    """Comprehensive check of token eligibility"""
    print(f"\n{'='*80}")
    print(f"🔍 Checking Token Eligibility: {token_address}")
    print(f"{'='*80}\n")
    
    results = {
        "token_address": token_address,
        "passed": True,
        "failed_checks": [],
        "passed_checks": [],
        "stats": {},
        "score": 0
    }
    
    # Step 1: Fetch token data from DexScreener
    print("📊 Step 1: Fetching token data from DexScreener...")
    try:
        url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
        data = get_json(url, timeout=10)
        
        if not data or not data.get("pairs"):
            print("❌ No pairs found on DexScreener")
            results["passed"] = False
            results["failed_checks"].append("No DexScreener pairs found")
            return results
        
        pairs = data["pairs"]
        # Get the pair with highest liquidity
        richest_pair = max(pairs, key=lambda p: float((p.get("liquidity") or {}).get("usd") or 0))
        
        base_token = richest_pair.get("baseToken", {})
        symbol = base_token.get("symbol", "UNKNOWN")
        name = base_token.get("name", "")
        chain_id = richest_pair.get("chainId", "").lower()
        
        price_usd = float(richest_pair.get("priceUsd") or 0)
        volume24h = float((richest_pair.get("volume") or {}).get("h24") or 0)
        liquidity_usd = float((richest_pair.get("liquidity") or {}).get("usd") or 0)
        market_cap = float(richest_pair.get("marketCap") or richest_pair.get("fdv") or 0)
        
        txns_24h = richest_pair.get("txns", {}).get("h24", {})
        tx_count_24h = int(txns_24h.get("buys", 0)) + int(txns_24h.get("sells", 0))
        
        pair_created_at = richest_pair.get("pairCreatedAt")
        pair_age_hours = None
        if pair_created_at:
            age_ms = int(pair_created_at)
            current_ms = int(datetime.now().timestamp() * 1000)
            pair_age_hours = (current_ms - age_ms) / (1000 * 3600)
        
        price_change_24h = float((richest_pair.get("priceChange") or {}).get("h24") or 0)
        
        results["stats"] = {
            "symbol": symbol,
            "name": name,
            "chain_id": chain_id,
            "price_usd": price_usd,
            "volume24h": volume24h,
            "liquidity_usd": liquidity_usd,
            "market_cap": market_cap,
            "tx_count_24h": tx_count_24h,
            "pair_age_hours": pair_age_hours,
            "price_change_24h": price_change_24h,
            "dex": richest_pair.get("dexId", ""),
            "pair_address": richest_pair.get("pairAddress", "")
        }
        
        print(f"✅ Token Data Retrieved:")
        print(f"   Symbol: {symbol} ({name})")
        print(f"   Chain: {chain_id.upper()}")
        print(f"   Price: ${price_usd:.6f}")
        print(f"   Volume 24h: ${volume24h:,.2f}")
        print(f"   Liquidity: ${liquidity_usd:,.2f}")
        print(f"   Market Cap: ${market_cap:,.2f}")
        print(f"   Transactions 24h: {tx_count_24h:,}")
        print(f"   Price Change 24h: {price_change_24h:+.2f}%")
        if pair_age_hours:
            print(f"   Pair Age: {pair_age_hours:.1f} hours")
        
    except Exception as e:
        print(f"❌ Error fetching token data: {e}")
        results["passed"] = False
        results["failed_checks"].append(f"Data fetch error: {str(e)}")
        return results
    
    # Step 2: Check blacklist
    print(f"\n🚫 Step 2: Checking blacklist...")
    if is_blacklisted(token_address):
        print("❌ Token is blacklisted")
        results["passed"] = False
        results["failed_checks"].append("Token is blacklisted")
    else:
        print("✅ Token is not blacklisted")
        results["passed_checks"].append("Not blacklisted")
    
    # Step 3: Check promotional content
    print(f"\n📝 Step 3: Checking promotional content filter...")
    if is_promotional_content(symbol, name):
        print("❌ Token flagged as promotional/spam content")
        results["passed"] = False
        results["failed_checks"].append("Promotional content detected")
    else:
        print("✅ Token passes promotional content check")
        results["passed_checks"].append("Not promotional")
    
    # Step 4: Check excluded keywords
    print(f"\n🔤 Step 4: Checking excluded keywords...")
    symbol_upper = symbol.upper()
    blocked = any(k in symbol_upper for k in EXCLUDED_KEYWORDS)
    if blocked and ENFORCE_KEYWORDS:
        print(f"❌ Token symbol contains excluded keyword")
        results["passed"] = False
        results["failed_checks"].append(f"Excluded keyword in symbol: {symbol}")
    else:
        print("✅ Token passes keyword filter")
        results["passed_checks"].append("No excluded keywords")
    
    # Step 5: Check valid token data
    print(f"\n✅ Step 5: Validating token data...")
    if not is_valid_token_data(symbol, token_address, volume24h, liquidity_usd):
        print("❌ Token data validation failed")
        print(f"   Volume check: ${volume24h:,.2f} >= $3,000: {volume24h >= 3000}")
        print(f"   Liquidity check: ${liquidity_usd:,.2f} >= $8,000: {liquidity_usd >= 8000}")
        results["passed"] = False
        results["failed_checks"].append("Token data validation failed")
    else:
        print("✅ Token data is valid")
        print(f"   ✓ Volume: ${volume24h:,.2f} (min: $3,000)")
        print(f"   ✓ Liquidity: ${liquidity_usd:,.2f} (min: $8,000)")
        results["passed_checks"].append("Valid token data")
    
    # Step 6: Check price threshold
    print(f"\n💰 Step 6: Checking price threshold...")
    if price_usd < 0.0000001:
        print(f"❌ Price too low: ${price_usd:.10f}")
        results["passed"] = False
        results["failed_checks"].append("Price too low (suspicious)")
    else:
        print(f"✅ Price acceptable: ${price_usd:.6f}")
        results["passed_checks"].append("Price threshold passed")
    
    # Step 7: Check volume/liquidity ratio
    print(f"\n📈 Step 7: Checking volume/liquidity ratio...")
    if liquidity_usd > 0:
        vol_liq_ratio = volume24h / liquidity_usd
        max_ratio = 10.0  # From config
        min_ratio = 0.05
        
        if vol_liq_ratio > max_ratio:
            print(f"❌ Volume/liquidity ratio too high: {vol_liq_ratio:.2f}x (max: {max_ratio}x)")
            print(f"   Possible wash trading/manipulation")
            results["passed"] = False
            results["failed_checks"].append(f"Volume/liquidity ratio too high: {vol_liq_ratio:.2f}x")
        elif vol_liq_ratio < min_ratio:
            print(f"❌ Volume/liquidity ratio too low: {vol_liq_ratio:.2f}x (min: {min_ratio}x)")
            results["passed"] = False
            results["failed_checks"].append(f"Volume/liquidity ratio too low: {vol_liq_ratio:.2f}x")
        else:
            print(f"✅ Volume/liquidity ratio acceptable: {vol_liq_ratio:.2f}x")
            results["passed_checks"].append(f"Volume/liquidity ratio OK: {vol_liq_ratio:.2f}x")
    else:
        print("⚠️ Cannot check ratio (no liquidity)")
    
    # Step 8: Calculate token score
    print(f"\n🎯 Step 8: Calculating token score...")
    score = calculate_token_score(symbol, volume24h, liquidity_usd, chain_id, token_address)
    results["score"] = score
    print(f"   Token Score: {score}/8")
    if score >= 1:
        print("✅ Score meets minimum threshold (≥1)")
        results["passed_checks"].append(f"Score: {score}/8")
    else:
        print("❌ Score too low (minimum: 1)")
        results["passed"] = False
        results["failed_checks"].append(f"Score too low: {score}/8")
    
    # Step 9: Check tradeability (Jupiter/Raydium)
    print(f"\n🔄 Step 9: Checking tradeability...")
    token_data = {
        "address": token_address,
        "symbol": symbol,
        "chainId": chain_id,
        "priceUsd": price_usd,
        "volume24h": volume24h,
        "liquidity": liquidity_usd
    }
    
    is_tradeable, reason = is_token_tradeable(token_data)
    if is_tradeable:
        print(f"✅ Token is tradeable: {reason}")
        results["passed_checks"].append(f"Tradeable: {reason}")
    else:
        print(f"❌ Token is not tradeable: {reason}")
        results["passed"] = False
        results["failed_checks"].append(f"Not tradeable: {reason}")
    
    # Additional Jupiter/Raydium specific checks
    if chain_id == "solana":
        print(f"\n   Checking Jupiter tradeability...")
        jupiter_ok = check_jupiter_tradeability(token_address, chain_id)
        print(f"   Jupiter: {'✅ Tradeable' if jupiter_ok else '❌ Not tradeable'}")
        
        print(f"\n   Checking Raydium tradeability...")
        raydium_ok = check_raydium_tradeability(token_address, chain_id)
        print(f"   Raydium: {'✅ Tradeable' if raydium_ok else '❌ Not tradeable'}")
    
    # Step 10: Preflight checks
    print(f"\n✈️ Step 10: Running preflight checks...")
    try:
        advanced_trading = AdvancedTrading()
        
        # Create token dict for preflight
        preflight_token = {
            "symbol": symbol,
            "address": token_address,
            "chainId": chain_id,
            "priceUsd": price_usd,
            "liquidity_usd": liquidity_usd,
            "market_cap": market_cap
        }
        
        if pair_age_hours:
            created_time = datetime.now(timezone.utc) - timedelta(hours=pair_age_hours)
            preflight_token["created_at"] = created_time.isoformat()
        
        # Use default trade amount for check
        trade_amount = 100.0  # $100 default
        preflight_passed, preflight_reason = advanced_trading.enhanced_preflight_check(preflight_token, trade_amount)
        
        if preflight_passed:
            print(f"✅ Preflight check passed: {preflight_reason}")
            results["passed_checks"].append(f"Preflight: {preflight_reason}")
        else:
            print(f"❌ Preflight check failed: {preflight_reason}")
            results["passed"] = False
            results["failed_checks"].append(f"Preflight failed: {preflight_reason}")
    except Exception as e:
        print(f"⚠️ Preflight check error: {e}")
        results["failed_checks"].append(f"Preflight error: {str(e)}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📋 SUMMARY")
    print(f"{'='*80}")
    print(f"Token: {symbol} ({token_address})")
    print(f"Chain: {chain_id.upper()}")
    print(f"Score: {score}/8")
    print(f"\n✅ Passed Checks: {len(results['passed_checks'])}")
    for check in results["passed_checks"]:
        print(f"   ✓ {check}")
    
    print(f"\n❌ Failed Checks: {len(results['failed_checks'])}")
    for check in results["failed_checks"]:
        print(f"   ✗ {check}")
    
    print(f"\n{'='*80}")
    if results["passed"]:
        print("🎉 RESULT: TOKEN PASSES ALL CHECKS - ELIGIBLE FOR TRADING")
    else:
        print("🚫 RESULT: TOKEN FAILS SOME CHECKS - NOT ELIGIBLE")
    print(f"{'='*80}\n")
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_token_eligibility.py <token_address>")
        sys.exit(1)
    
    token_address = sys.argv[1]
    results = check_token_eligibility(token_address)
    
    # Save results to JSON
    output_file = f"data/token_check_{token_address[:8]}.json"
    Path("data").mkdir(exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"📄 Results saved to: {output_file}")
