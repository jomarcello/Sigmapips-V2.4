#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
import time
import asyncio
import logging
import requests
import random
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('yahoo_test')

# Display yfinance version
logger.info(f"yfinance version: {yf.__version__}")

def setup_session():
    """Create a session with appropriate headers"""
    session = requests.Session()
    
    # User agent rotation
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:88.0) Gecko/20100101 Firefox/88.0'
    ]
    
    # Choose a random user agent
    user_agent = random.choice(user_agents)
    logger.info(f"Using User-Agent: {user_agent}")
    
    session.headers.update({
        'User-Agent': user_agent,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive'
    })
    
    return session

async def test_download_method(symbol):
    """Test the yf.download method"""
    logger.info(f"Testing yf.download for {symbol}")
    session = setup_session()
    
    # Set timeframe
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    try:
        logger.info(f"Downloading data for {symbol} from {start_date.date()} to {end_date.date()}")
        df = yf.download(
            tickers=symbol,
            start=start_date.date(),
            end=end_date.date(),
            interval="1h",
            progress=False,
            session=session,
            timeout=20,
            ignore_tz=True
        )
        
        if df is None or df.empty:
            logger.warning(f"Download returned empty DataFrame for {symbol}")
        else:
            logger.info(f"Download successful for {symbol}, got {len(df)} rows")
            logger.info(f"First few rows: {df.head(3)}")
            return df
            
    except Exception as e:
        logger.error(f"Error downloading data: {str(e)}")
        if "429" in str(e) or "too many requests" in str(e).lower():
            logger.error("Rate limit error detected")
        return None
        
    return None

async def test_ticker_method(symbol):
    """Test the Ticker.history method"""
    logger.info(f"Testing Ticker.history for {symbol}")
    session = setup_session()
    
    # Set timeframe
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    try:
        ticker = yf.Ticker(symbol, session=session)
        logger.info(f"Ticker object created for {symbol}")
        
        # First try to get basic info
        try:
            info = ticker.info
            logger.info(f"Basic info for {symbol}: {list(info.keys())[:5]} (showing first 5 keys)")
        except Exception as info_e:
            logger.error(f"Error getting ticker info: {str(info_e)}")
        
        # Try to get history
        df = ticker.history(
            start=start_date,
            end=end_date,
            interval="1h",
            prepost=False,
            auto_adjust=False
        )
        
        if df is None or df.empty:
            logger.warning(f"Ticker.history returned empty DataFrame for {symbol}")
        else:
            logger.info(f"Ticker.history successful for {symbol}, got {len(df)} rows")
            logger.info(f"Columns: {df.columns.tolist()}")
            logger.info(f"First few rows: {df.head(3)}")
            return df
            
    except Exception as e:
        logger.error(f"Error using Ticker.history: {str(e)}")
        if "429" in str(e) or "too many requests" in str(e).lower():
            logger.error("Rate limit error detected")
        return None
        
    return None

async def test_alternative_oil_symbols():
    """Test various oil-related symbols to find alternatives to CL=F"""
    oil_symbols = [
        "CL=F",      # WTI Crude Oil Futures
        "BZ=F",      # Brent Crude Oil Futures
        "QM=F",      # E-mini Crude Oil Futures
        "XOP",       # SPDR S&P Oil & Gas Exploration & Production ETF
        "USO",       # United States Oil Fund
        "UCO",       # ProShares Ultra Bloomberg Crude Oil
        "OIL",       # iPath Pure Beta Crude Oil ETN
        "XLE",       # Energy Select Sector SPDR Fund
        "XOM",       # Exxon Mobil
        "CVX"        # Chevron
    ]
    
    results = {}
    
    for symbol in oil_symbols:
        logger.info(f"Testing alternative symbol: {symbol}")
        
        # Wait between requests to avoid rate limiting
        await asyncio.sleep(5)
        
        # Try download method
        df = await test_download_method(symbol)
        
        if df is not None and not df.empty:
            results[symbol] = {
                "success": True,
                "method": "download",
                "rows": len(df)
            }
            continue
        
        # If download failed, try ticker method
        await asyncio.sleep(5)  # Wait between attempts
        
        df = await test_ticker_method(symbol)
        
        if df is not None and not df.empty:
            results[symbol] = {
                "success": True,
                "method": "ticker",
                "rows": len(df)
            }
        else:
            results[symbol] = {
                "success": False
            }
    
    logger.info("Results of alternative symbol testing:")
    for symbol, result in results.items():
        if result["success"]:
            logger.info(f"✓ {symbol}: Success using {result['method']} method, got {result['rows']} rows")
        else:
            logger.info(f"✗ {symbol}: Failed with both methods")
    
    return results

async def test_network_environment():
    """Test network environment and configuration"""
    logger.info("Testing network environment:")
    
    # Check for proxies
    proxies = {
        "http": requests.utils.getproxies().get("http", "None"),
        "https": requests.utils.getproxies().get("https", "None")
    }
    logger.info(f"Detected proxies: {proxies}")
    
    # Check IP
    try:
        response = requests.get("https://api.ipify.org")
        if response.status_code == 200:
            ip = response.text
            logger.info(f"External IP address: {ip}")
            
            # Do a rate limit check
            remaining = response.headers.get("X-RateLimit-Remaining", "unknown")
            logger.info(f"Rate limit remaining: {remaining}")
        else:
            logger.warning(f"Failed to get IP address, status code: {response.status_code}")
    except Exception as e:
        logger.error(f"Error getting IP address: {str(e)}")
    
    # Check Yahoo Finance connectivity
    try:
        response = requests.get("https://finance.yahoo.com/", 
                               headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/51.0.2704.103"})
        logger.info(f"Yahoo Finance connectivity: {response.status_code}")
    except Exception as e:
        logger.error(f"Error connecting to Yahoo Finance: {str(e)}")

async def main():
    """Main test function"""
    logger.info("Starting Yahoo Finance API tests")
    
    # Test environment
    await test_network_environment()
    
    # Test main symbol
    symbol = "CL=F"
    logger.info(f"Testing main symbol: {symbol}")
    
    # Try download method
    df1 = await test_download_method(symbol)
    
    # Wait to avoid rate limiting
    logger.info("Waiting 10 seconds before next test...")
    await asyncio.sleep(10)
    
    # Try ticker method
    df2 = await test_ticker_method(symbol)
    
    # Wait before alternatives
    logger.info("Waiting 10 seconds before testing alternatives...")
    await asyncio.sleep(10)
    
    # Test alternatives
    await test_alternative_oil_symbols()
    
    logger.info("All tests completed")

if __name__ == "__main__":
    asyncio.run(main()) 