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
    
    # Test future date fix
    current_date = datetime.now()
    future_date = current_date + timedelta(days=10)
    
    logging.info(f"Testing _fix_future_dates with future date: {future_date}")
    fixed_start, fixed_end = YahooFinanceProvider._fix_future_dates(current_date, future_date)
    logging.info(f"Fixed dates: start={fixed_start}, end={fixed_end}")
    
    # Test all oil symbols
    oil_symbols = ["XTIUSD", "WTIUSD", "USOIL", "XBRUSD"]
    
    for symbol in oil_symbols:
        logging.info(f"Testing {symbol}")
        
        # Test all timeframes
        for timeframe in ["1h", "4h", "1d"]:
            logging.info(f"Getting {timeframe} data for {symbol}")
            df = await YahooFinanceProvider.get_market_data(symbol, timeframe, 50)
            
            if df is not None and not df.empty:
                logging.info(f"✅ Got {len(df)} rows for {symbol} on {timeframe}")
                logging.info(f"First row: {df.iloc[0]}")
                logging.info(f"Last row: {df.iloc[-1]}")
                logging.info(f"Date range: {df.index[0]} to {df.index[-1]}")
                
                # Check indicators
                if hasattr(df, 'indicators'):
                    logging.info(f"Indicators: {df.indicators}")
            else:
                logging.error(f"❌ Failed to get data for {symbol} on {timeframe}")
    
    # Try individual tickers directly
    direct_tickers = ["CL=F", "BNO", "UCO", "OIL", "XLE", "BZ=F"]
    
    for ticker in direct_tickers:
        logging.info(f"Testing direct ticker: {ticker}")
        formatted_symbol = ticker  # No need to format as it's already in Yahoo format
        
        # Calculate dates
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=30)
        
        # Try to get data
        try:
            df = await YahooFinanceProvider._download_data(
                formatted_symbol, 
                start_date,
                end_date,
                "1d",
                timeout=60,
                original_symbol=ticker
            )
            
            if df is not None and not df.empty:
                logging.info(f"✅ Direct ticker {ticker}: Got {len(df)} rows")
                logging.info(f"Date range: {df.index[0]} to {df.index[-1]}")
            else:
                logging.error(f"❌ Direct ticker {ticker}: No data returned")
        except Exception as e:
            logging.error(f"❌ Direct ticker {ticker}: Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_oil_data()) 