import asyncio
import logging
import os
import pandas as pd
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Import DirectYahooProvider
from trading_bot.services.chart_service.direct_yahoo_provider import DirectYahooProvider

# Test commodity instruments
COMMODITY_INSTRUMENTS = [
    "XAUUSD",  # Gold
    "XTIUSD",  # Oil
    "XAGUSD",  # Silver
    "XBRUSD"   # Brent Oil
]

async def test_chart_generation():
    """Test chart generation for commodities with updated symbols"""
    logger.info("Testing chart generation for commodities with the DirectYahooProvider")
    
    for instrument in COMMODITY_INSTRUMENTS:
        logger.info(f"Generating chart for {instrument}")
        
        try:
            # Get the Yahoo Finance symbol
            yahoo_symbol = DirectYahooProvider._format_symbol(instrument)
            logger.info(f"Using symbol: {yahoo_symbol} for {instrument}")
            
            # Generate chart using DirectYahooProvider's get_chart method
            chart_data = DirectYahooProvider.get_chart(instrument, timeframe="1d")
            
            if chart_data and len(chart_data) > 1000:
                # Save chart to file
                chart_filename = f"chart_{instrument.lower()}.png"
                with open(chart_filename, "wb") as f:
                    f.write(chart_data)
                
                file_size = os.path.getsize(chart_filename)
                logger.info(f"✅ Chart for {instrument} generated successfully and saved as {chart_filename} ({file_size} bytes)")
            else:
                logger.error(f"❌ Failed to generate chart for {instrument} or chart too small: {len(chart_data) if chart_data else 0} bytes")
        
        except Exception as e:
            logger.error(f"❌ Error generating chart for {instrument}: {str(e)}")
        
        # Add a small delay between requests
        await asyncio.sleep(2)

async def test_get_market_data():
    """Test get_market_data for commodities"""
    logger.info("\nTesting get_market_data for commodities")
    
    for instrument in COMMODITY_INSTRUMENTS:
        logger.info(f"Getting market data for {instrument}")
        
        try:
            # Call get_market_data
            data, info = await DirectYahooProvider.get_market_data(instrument)
            
            if data is not None and not data.empty:
                logger.info(f"✅ Successfully got market data for {instrument} with shape {data.shape}")
                logger.info(f"First row: {data.iloc[0]}")
                logger.info(f"Last row: {data.iloc[-1]}")
                
                if info:
                    logger.info(f"Metadata: {info}")
            else:
                logger.error(f"❌ Failed to get market data for {instrument}")
        
        except Exception as e:
            logger.error(f"❌ Error getting market data for {instrument}: {str(e)}")
        
        # Add a small delay between requests
        await asyncio.sleep(2)

async def main():
    """Main function to run tests"""
    logger.info("Starting commodity chart tests with DirectYahooProvider")
    
    # Test chart generation
    await test_chart_generation()
    
    # Test get_market_data
    await test_get_market_data()
    
    logger.info("Commodity chart tests completed")

if __name__ == "__main__":
    asyncio.run(main()) 