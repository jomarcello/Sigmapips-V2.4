import asyncio
import logging
import pandas as pd
import os
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Import the DirectYahooProvider class
from trading_bot.services.chart_service.direct_yahoo_provider import DirectYahooProvider

# Test commodity instruments
COMMODITY_INSTRUMENTS = [
    "XAUUSD",  # Gold
    "XTIUSD",  # Oil
    "XAGUSD",  # Silver
    "XBRUSD"   # Brent Oil
]

async def test_commodity_symbols():
    """Test the updated commodity symbol mappings"""
    logger.info("Testing updated commodity symbol mappings")
    
    for instrument in COMMODITY_INSTRUMENTS:
        # Get the Yahoo Finance symbol
        yahoo_symbol = DirectYahooProvider._format_symbol(instrument)
        logger.info(f"Instrument: {instrument} -> Yahoo Finance symbol: {yahoo_symbol}")
        
        # Try to download data
        logger.info(f"Downloading data for {instrument} using symbol {yahoo_symbol}")
        
        try:
            data = await DirectYahooProvider._download_data(
                yahoo_symbol,
                interval="1d",  # Using daily data for more reliable results
                period="1mo"    # Using 1 month period
            )
            
            if data is not None and not data.empty:
                logger.info(f"✅ Successfully downloaded data for {instrument} with shape {data.shape}")
                logger.info(f"First row: {data.iloc[0]}")
                logger.info(f"Last row: {data.iloc[-1]}")
            else:
                logger.error(f"❌ Failed to download data for {instrument} - returned empty DataFrame")
                
        except Exception as e:
            logger.error(f"❌ Error downloading data for {instrument}: {str(e)}")
            
        # Add a small delay between requests
        await asyncio.sleep(2)

async def test_chart_generation():
    """Test chart generation for commodity instruments"""
    logger.info("\nTesting chart generation for commodities...")
    
    for instrument in COMMODITY_INSTRUMENTS:
        logger.info(f"Generating chart for {instrument}")
        
        try:
            # Generate chart
            chart_data = DirectYahooProvider.get_chart(instrument, timeframe="1d")  # Daily timeframe for reliability
            
            if chart_data:
                # Save chart to file for inspection
                chart_filename = f"test_commodity_{instrument.lower()}.png"
                with open(chart_filename, "wb") as f:
                    f.write(chart_data)
                    
                file_size = os.path.getsize(chart_filename)
                logger.info(f"✅ Chart for {instrument} generated successfully and saved as {chart_filename} ({file_size} bytes)")
            else:
                logger.error(f"❌ Failed to generate chart for {instrument}")
                
        except Exception as e:
            logger.error(f"❌ Error generating chart for {instrument}: {str(e)}")
            
        # Add a small delay between requests
        await asyncio.sleep(2)

async def test_market_data():
    """Test the get_market_data method for commodity instruments"""
    logger.info("\nTesting get_market_data method for commodities...")
    
    for instrument in COMMODITY_INSTRUMENTS:
        logger.info(f"Getting market data for {instrument}")
        
        try:
            data, info = await DirectYahooProvider.get_market_data(instrument)
            
            if data is not None and not data.empty:
                logger.info(f"✅ Successfully got market data for {instrument} with shape {data.shape}")
                logger.info(f"First row: {data.iloc[0]}")
                if info:
                    logger.info(f"Market info data: {info}")
            else:
                logger.error(f"❌ Failed to get market data for {instrument}")
                
        except Exception as e:
            logger.error(f"❌ Error getting market data for {instrument}: {str(e)}")
            
        # Add a small delay between requests
        await asyncio.sleep(2)

async def main():
    """Main function that runs all tests"""
    logger.info("Starting commodity symbol mapping tests")
    
    # Print mapping for verification
    for instrument in COMMODITY_INSTRUMENTS:
        symbol = DirectYahooProvider._format_symbol(instrument)
        logger.info(f"Mapping check: {instrument} -> {symbol}")
    
    # Test the symbol mappings
    await test_commodity_symbols()
    
    # Test the get_market_data method
    await test_market_data()
    
    # Test chart generation
    await test_chart_generation()
    
    logger.info("Commodity symbol mapping tests complete")

if __name__ == "__main__":
    asyncio.run(main()) 