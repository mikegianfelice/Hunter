#!/usr/bin/env python3
"""
Network Diagnostics - Test connectivity to trading APIs
"""

import requests
import socket
import time
from typing import Dict, List, Tuple

def test_dns_resolution(hostname: str) -> Tuple[bool, str]:
    """Test if a hostname can be resolved"""
    try:
        ip_address = socket.gethostbyname(hostname)
        return True, ip_address
    except socket.gaierror as e:
        return False, f"DNS resolution failed: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def test_tcp_connection(hostname: str, port: int = 443, timeout: float = 5.0) -> Tuple[bool, str]:
    """Test if we can establish a TCP connection"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((hostname, port))
        sock.close()
        
        if result == 0:
            return True, f"Successfully connected to {hostname}:{port}"
        else:
            return False, f"Connection failed with error code: {result}"
    except Exception as e:
        return False, f"Error: {e}"

def test_https_request(url: str, timeout: float = 10.0) -> Tuple[bool, str, int]:
    """Test if we can make an HTTPS request"""
    try:
        start_time = time.time()
        response = requests.get(url, timeout=timeout)
        elapsed = time.time() - start_time
        
        return True, f"Success (status: {response.status_code}, time: {elapsed:.2f}s)", response.status_code
    except requests.exceptions.ConnectionError as e:
        return False, f"Connection error: {e}", 0
    except requests.exceptions.Timeout as e:
        return False, f"Timeout error: {e}", 0
    except requests.exceptions.RequestException as e:
        return False, f"Request error: {e}", 0
    except Exception as e:
        return False, f"Error: {e}", 0

def test_jupiter_api() -> Dict[str, any]:
    """Test connectivity to Jupiter API
    
    NOTE: Jupiter API endpoint changed from quote-api.jup.ag to api.jup.ag
    """
    hostname = "api.jup.ag"  # Changed from quote-api.jup.ag
    port = 443
    
    print(f"\n{'='*60}")
    print(f"Testing Jupiter API ({hostname})")
    print(f"{'='*60}")
    
    results = {}
    
    # Test 1: DNS Resolution
    print(f"\n1. Testing DNS resolution for {hostname}...")
    dns_ok, dns_msg = test_dns_resolution(hostname)
    results['dns'] = {'success': dns_ok, 'message': dns_msg}
    
    if dns_ok:
        print(f"   ✅ {dns_msg}")
    else:
        print(f"   ❌ {dns_msg}")
        return results
    
    # Test 2: TCP Connection
    print(f"\n2. Testing TCP connection to {hostname}:{port}...")
    tcp_ok, tcp_msg = test_tcp_connection(hostname, port)
    results['tcp'] = {'success': tcp_ok, 'message': tcp_msg}
    
    if tcp_ok:
        print(f"   ✅ {tcp_msg}")
    else:
        print(f"   ❌ {tcp_msg}")
        return results
    
    # Test 3: HTTPS Request (simple health check)
    print(f"\n3. Testing HTTPS request to Jupiter API...")
    test_url = "https://api.jup.ag/v6/quote?inputMint=So11111111111111111111111111111111111111112&outputMint=EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v&amount=1000000000&slippageBps=50"
    https_ok, https_msg, status_code = test_https_request(test_url)
    results['https'] = {'success': https_ok, 'message': https_msg, 'status_code': status_code}
    
    if https_ok:
        print(f"   ✅ {https_msg}")
    else:
        print(f"   ❌ {https_msg}")
    
    return results

def test_raydium_api() -> Dict[str, any]:
    """Test connectivity to Raydium API"""
    hostname = "api.raydium.io"
    port = 443
    
    print(f"\n{'='*60}")
    print(f"Testing Raydium API ({hostname})")
    print(f"{'='*60}")
    
    results = {}
    
    # Test 1: DNS Resolution
    print(f"\n1. Testing DNS resolution for {hostname}...")
    dns_ok, dns_msg = test_dns_resolution(hostname)
    results['dns'] = {'success': dns_ok, 'message': dns_msg}
    
    if dns_ok:
        print(f"   ✅ {dns_msg}")
    else:
        print(f"   ❌ {dns_msg}")
        return results
    
    # Test 2: TCP Connection
    print(f"\n2. Testing TCP connection to {hostname}:{port}...")
    tcp_ok, tcp_msg = test_tcp_connection(hostname, port)
    results['tcp'] = {'success': tcp_ok, 'message': tcp_msg}
    
    if tcp_ok:
        print(f"   ✅ {tcp_msg}")
    else:
        print(f"   ❌ {tcp_msg}")
        return results
    
    # Test 3: HTTPS Request
    print(f"\n3. Testing HTTPS request to Raydium API...")
    test_url = "https://api.raydium.io/v2/main/price?ids=So11111111111111111111111111111111111111112"
    https_ok, https_msg, status_code = test_https_request(test_url)
    results['https'] = {'success': https_ok, 'message': https_msg, 'status_code': status_code}
    
    if https_ok:
        print(f"   ✅ {https_msg}")
    else:
        print(f"   ❌ {https_msg}")
    
    return results

def test_general_connectivity() -> Dict[str, any]:
    """Test general internet connectivity"""
    print(f"\n{'='*60}")
    print(f"Testing General Internet Connectivity")
    print(f"{'='*60}")
    
    results = {}
    
    # Test Google DNS
    print(f"\n1. Testing DNS resolution for google.com...")
    dns_ok, dns_msg = test_dns_resolution("google.com")
    results['google_dns'] = {'success': dns_ok, 'message': dns_msg}
    
    if dns_ok:
        print(f"   ✅ {dns_msg}")
    else:
        print(f"   ❌ {dns_msg}")
    
    # Test HTTPS to Google
    print(f"\n2. Testing HTTPS request to google.com...")
    https_ok, https_msg, status_code = test_https_request("https://www.google.com", timeout=5.0)
    results['google_https'] = {'success': https_ok, 'message': https_msg, 'status_code': status_code}
    
    if https_ok:
        print(f"   ✅ {https_msg}")
    else:
        print(f"   ❌ {https_msg}")
    
    return results

def run_all_diagnostics() -> Dict[str, any]:
    """Run all network diagnostics"""
    print(f"\n{'#'*60}")
    print(f"# Network Diagnostics for Trading Bot")
    print(f"{'#'*60}")
    
    results = {
        'general': test_general_connectivity(),
        'jupiter': test_jupiter_api(),
        'raydium': test_raydium_api()
    }
    
    print(f"\n{'='*60}")
    print(f"Diagnostics Summary")
    print(f"{'='*60}")
    
    # Check overall status
    general_ok = results['general'].get('google_dns', {}).get('success', False)
    jupiter_dns_ok = results['jupiter'].get('dns', {}).get('success', False)
    jupiter_https_ok = results['jupiter'].get('https', {}).get('success', False)
    raydium_dns_ok = results['raydium'].get('dns', {}).get('success', False)
    raydium_https_ok = results['raydium'].get('https', {}).get('success', False)
    
    print(f"\nGeneral Internet: {'✅ OK' if general_ok else '❌ FAIL'}")
    print(f"Jupiter API DNS:  {'✅ OK' if jupiter_dns_ok else '❌ FAIL'}")
    print(f"Jupiter API HTTP: {'✅ OK' if jupiter_https_ok else '❌ FAIL'}")
    print(f"Raydium API DNS:  {'✅ OK' if raydium_dns_ok else '❌ FAIL'}")
    print(f"Raydium API HTTP: {'✅ OK' if raydium_https_ok else '❌ FAIL'}")
    
    if not general_ok:
        print(f"\n⚠️  WARNING: General internet connectivity issue detected!")
        print(f"   Please check your internet connection and DNS settings.")
    elif not jupiter_dns_ok or not raydium_dns_ok:
        print(f"\n⚠️  WARNING: DNS resolution issue detected for trading APIs!")
        print(f"   This could be a temporary DNS issue or ISP blocking.")
        print(f"   Consider using alternative DNS servers (e.g., 8.8.8.8, 1.1.1.1)")
    elif not jupiter_https_ok or not raydium_https_ok:
        print(f"\n⚠️  WARNING: Cannot reach trading APIs!")
        print(f"   This could be due to firewall, network restrictions, or API outage.")
    else:
        print(f"\n✅ All connectivity checks passed!")
    
    return results

if __name__ == "__main__":
    run_all_diagnostics()

