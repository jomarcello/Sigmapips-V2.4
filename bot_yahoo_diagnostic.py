#!/usr/bin/env python3
import sys
import os
import platform
import datetime
import requests
import logging
import importlib
import json
import socket

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('bot_diagnostic')

def log_system_info():
    """Log system information"""
    logger.info("=== SYSTEM INFORMATION ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Machine: {platform.machine()}")
    logger.info(f"Processor: {platform.processor()}")
    logger.info(f"Node: {platform.node()}")
    
    # Get environment variables (exclude sensitive ones)
    env_vars = {k: v for k, v in os.environ.items() 
               if not any(sensitive in k.lower() for sensitive in 
                         ['key', 'token', 'secret', 'pass', 'auth'])}
    logger.info(f"Environment variables (non-sensitive): {json.dumps(env_vars, indent=2)}")
    
    # Current working directory
    logger.info(f"Current working directory: {os.getcwd()}")
    
    # Network info
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        logger.info(f"Hostname: {hostname}")
        logger.info(f"Local IP: {local_ip}")
    except Exception as e:
        logger.error(f"Error getting network info: {str(e)}")

def log_package_versions():
    """Log versions of relevant packages"""
    logger.info("=== PACKAGE VERSIONS ===")
    packages = ['yfinance', 'pandas', 'numpy', 'requests', 'urllib3', 'tenacity']
    
    for package in packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            logger.info(f"{package}: {version}")
        except ImportError:
            logger.warning(f"{package}: Not installed")

def check_network_connectivity():
    """Check network connectivity to key services"""
    logger.info("=== NETWORK CONNECTIVITY ===")
    urls = [
        'https://finance.yahoo.com',
        'https://query1.finance.yahoo.com',
        'https://query2.finance.yahoo.com',
        'https://www.google.com',
        'https://api.ipify.org'
    ]
    
    for url in urls:
        try:
            start_time = datetime.datetime.now()
            response = requests.get(url, timeout=10, 
                                   headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/90.0.4430.212 Safari/537.36'})
            elapsed = (datetime.datetime.now() - start_time).total_seconds()
            
            logger.info(f"{url}: Status {response.status_code}, Response time: {elapsed:.2f}s")
            
            if url == 'https://api.ipify.org':
                logger.info(f"External IP: {response.text}")
                
        except Exception as e:
            logger.error(f"{url}: Connection error: {str(e)}")

def check_rate_limits():
    """Check if we're being rate limited by Yahoo Finance"""
    logger.info("=== RATE LIMIT TESTS ===")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/90.0.4430.212 Safari/537.36'
    }
    
    # Test Yahoo Finance crumb endpoint
    try:
        response = requests.get("https://query1.finance.yahoo.com/v1/test/getcrumb", 
                               headers=headers, timeout=10)
        logger.info(f"Yahoo Finance crumb test: Status {response.status_code}")
        logger.info(f"Response: {response.text}")
        logger.info(f"Headers: {dict(response.headers)}")
    except Exception as e:
        logger.error(f"Error testing Yahoo Finance crumb: {str(e)}")
    
    # Make multiple requests to see if we get rate limited
    logger.info("Making multiple requests to test rate limiting...")
    for i in range(5):
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/AAPL?interval=1d&range=1d"
            response = requests.get(url, headers=headers, timeout=10)
            logger.info(f"Request {i+1}: Status {response.status_code}")
            
            # Check if response contains rate limit indicators
            response_text = response.text.lower()
            if "rate limit" in response_text or "too many requests" in response_text:
                logger.error(f"Rate limiting detected in response text: {response.text[:100]}...")
        except Exception as e:
            logger.error(f"Error in request {i+1}: {str(e)}")
            
        # Short pause between requests
        import time
        time.sleep(2)

def main():
    logger.info("Starting bot environment diagnostics")
    
    log_system_info()
    log_package_versions()
    check_network_connectivity()
    check_rate_limits()
    
    logger.info("Diagnostics completed")

if __name__ == "__main__":
    main() 