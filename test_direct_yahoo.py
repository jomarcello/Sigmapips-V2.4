import asyncio
import os
import sys
import logging
import pandas as pd
from datetime import datetime
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Set pandas display options for better output
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)

# Import the providers we want to test
from trading_bot.services.chart_service.direct_yahoo_provider import DirectYahooProvider
from trading_bot.services.chart_service.yfinance_provider import YahooFinanceProvider

# Test instruments
TEST_INSTRUMENTS = [
    "EURUSD",  # Major forex pair
    "US500",   # S&P 500 index
    "XTIUSD",  # Oil
    "XAUUSD"   # Gold
]

async def test_get_market_data():
    """Test the get_market_data method for different instruments"""
    logger.info("\nTesting get_market_data method...")
    
    for instrument in TEST_INSTRUMENTS:
        logger.info(f"\nTesting instrument: {instrument}")
        
        # Get Yahoo Finance symbol for reference
        yahoo_symbol = DirectYahooProvider._format_symbol(instrument)
        logger.info(f"Yahoo Finance symbol: {yahoo_symbol}")
        
        try:
            # Get market data using DirectYahooProvider
            result = await DirectYahooProvider.get_market_data(instrument)
            
            if result:
                df, info = result
                
                # Check if we got data
                if df is not None and not df.empty:
                    logger.info(f"Successfully received data for {instrument}")
                    logger.info(f"Data shape: {df.shape}")
                    logger.info(f"First 3 rows:\n{df.head(3)}")
                    
                    # Check metadata
                    if info:
                        logger.info(f"Metadata: {info}")
                else:
                    logger.error(f"No data received for {instrument}")
            else:
                logger.error(f"Empty result for {instrument}")
                
        except Exception as e:
            logger.error(f"Error getting market data for {instrument}: {str(e)}")
            
        # Add a small delay to avoid rate limiting
        await asyncio.sleep(2)

async def test_chart_generation():
    """Test chart generation for different instruments"""
    logger.info("\nTesting chart generation...")
    
    for instrument in TEST_INSTRUMENTS:
        logger.info(f"\nGenerating chart for {instrument}")
        
        try:
            # Generate chart using DirectYahooProvider
            chart_data = DirectYahooProvider.get_chart(instrument, timeframe="1h")
            
            if chart_data:
                # Save chart to file for inspection
                chart_filename = f"test_direct_{instrument.lower()}.png"
                with open(chart_filename, "wb") as f:
                    f.write(chart_data)
                    
                file_size = os.path.getsize(chart_filename)
                logger.info(f"Chart generated successfully and saved as {chart_filename} ({file_size} bytes)")
            else:
                logger.error(f"Failed to generate chart for {instrument}")
                
        except Exception as e:
            logger.error(f"Error generating chart for {instrument}: {str(e)}")
            
        # Add a small delay to avoid rate limiting
        await asyncio.sleep(2)

async def compare_with_yahoo_finance_provider():
    """Compare DirectYahooProvider with YahooFinanceProvider"""
    logger.info("\nComparing DirectYahooProvider with YahooFinanceProvider...")
    
    instrument = "EURUSD"  # Use a single instrument for comparison
    
    try:
        # Test DirectYahooProvider
        start_time_direct = time.time()
        direct_result = await DirectYahooProvider.get_market_data(instrument)
        direct_time = time.time() - start_time_direct
        
        # Test YahooFinanceProvider
        start_time_yahoo = time.time()
        yahoo_result = await YahooFinanceProvider.get_market_data(instrument)
        yahoo_time = time.time() - start_time_yahoo
        
        logger.info(f"DirectYahooProvider time: {direct_time:.2f}s")
        logger.info(f"YahooFinanceProvider time: {yahoo_time:.2f}s")
        
        # Compare data shapes
        if direct_result and yahoo_result:
            direct_df, direct_info = direct_result
            yahoo_df, yahoo_info = yahoo_result
            
            if direct_df is not None and yahoo_df is not None:
                logger.info(f"DirectYahooProvider data shape: {direct_df.shape}")
                logger.info(f"YahooFinanceProvider data shape: {yahoo_df.shape}")
                
                # Compare the last few rows
                logger.info(f"DirectYahooProvider last row: {direct_df.iloc[-1]}")
                logger.info(f"YahooFinanceProvider last row: {yahoo_df.iloc[-1]}")
                
    except Exception as e:
        logger.error(f"Error during comparison: {str(e)}")

async def main():
    """Main test function"""
    logger.info("Starting DirectYahooProvider tests...")
    
    # Set logging for tests
    logging.getLogger('trading_bot').setLevel(logging.INFO)
    
    # Test get_market_data
    await test_get_market_data()
    
    # Test chart generation
    await test_chart_generation()
    
    # Compare with YahooFinanceProvider
    await compare_with_yahoo_finance_provider()
    
    logger.info("DirectYahooProvider tests completed!")

if __name__ == "__main__":
    asyncio.run(main()) 