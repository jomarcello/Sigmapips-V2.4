import asyncio
import logging
import sys
from datetime import datetime, timedelta
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Import the YahooFinanceProvider
from yfinance_provider import YahooFinanceProvider

async def test_oil_data():
    """Test retrieving oil data from Yahoo Finance"""
    logging.info("Starting oil data test")
    
    # Test USOIL specifically
    symbol = "USOIL"
    timeframe = "1h"
    
    logging.info(f"Testing {symbol} on {timeframe}")
    result = await YahooFinanceProvider.get_market_data(symbol, timeframe, 50)
    
    if result is not None and isinstance(result, tuple) and len(result) == 2:
        df, indicators = result
        if df is not None and not df.empty:
            logging.info(f"✅ Got {len(df)} rows for {symbol} on {timeframe}")
            logging.info(f"First row: {df.iloc[0]}")
            logging.info(f"Last row: {df.iloc[-1]}")
            logging.info(f"Date range: {df.index[0]} to {df.index[-1]}")
            
            # Check indicators dictionary
            if indicators:
                logging.info(f"Indicators: {indicators}")
        else:
            logging.error(f"❌ Failed to get data for {symbol} on {timeframe}")
    else:
        logging.error(f"❌ Failed to get data for {symbol} on {timeframe}")
    
    # Try each ticker in the alternative list directly
    direct_tickers = ["CL=F", "BNO", "UCO", "OIL", "XLE"]
    
    for ticker in direct_tickers:
        logging.info(f"Testing direct ticker: {ticker}")
        
        # Try to get data
        result = await YahooFinanceProvider.get_market_data(ticker, timeframe, 50)
        
        if result is not None and isinstance(result, tuple) and len(result) == 2:
            df, indicators = result
            if df is not None and not df.empty:
                logging.info(f"✅ Direct ticker {ticker}: Got {len(df)} rows")
                logging.info(f"Date range: {df.index[0]} to {df.index[-1]}")
            else:
                logging.error(f"❌ Direct ticker {ticker}: No data returned")
        else:
            logging.error(f"❌ Direct ticker {ticker}: No data returned")

if __name__ == "__main__":
    asyncio.run(test_oil_data()) 