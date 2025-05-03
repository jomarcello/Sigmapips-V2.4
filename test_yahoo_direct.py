import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Ensure trading_bot package is in the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import Yahoo Finance provider
from trading_bot.services.chart_service.yfinance_provider import YahooFinanceProvider

async def test_yahoo_finance():
    """Test Yahoo Finance data retrieval directly"""
    logger.info("Starting Yahoo Finance direct test...")
    
    # Create instance of the provider
    yahoo_provider = YahooFinanceProvider()
    
    # Instruments to test with corresponding Yahoo Finance symbols
    test_instruments = {
        "EURUSD": "EURUSD=X",
        "US500": "^GSPC",
        "XTIUSD": "CL=F",
        "XAUUSD": "GC=F"
    }
    
    # Test get_market_data method which is the main entry point
    logger.info("\nTesting get_market_data method...")
    for instrument, yahoo_symbol in test_instruments.items():
        try:
            logger.info(f"Getting market data for {instrument} (Yahoo symbol: {yahoo_symbol})")
            result = YahooFinanceProvider.get_market_data(instrument)
            
            # Since get_market_data returns a tuple and not a coroutine (no await needed)
            if isinstance(result, tuple) and len(result) == 2:
                data, metadata = result
                
                if data is not None and not data.empty:
                    logger.info(f"Successfully retrieved market data for {instrument}: {len(data)} rows")
                    logger.info(f"Data sample:\n{data.head(3)}")
                    logger.info(f"Metadata: {metadata}")
                else:
                    logger.error(f"Failed to retrieve market data for {instrument}")
            else:
                logger.error(f"Unexpected result format for {instrument}: {result}")
        
        except Exception as e:
            logger.error(f"Error getting market data for {instrument}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(test_yahoo_finance()) 