import asyncio
import logging
import sys
import os
import time
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

# Import ChartService
from trading_bot.services.chart_service.chart import ChartService

async def test_yahoo_chart():
    """Test chart generation using Yahoo Finance data"""
    logger.info("Starting Yahoo Finance chart test...")
    
    # Initialize ChartService
    try:
        chart_service = ChartService()
        await chart_service.initialize()
        logger.info("ChartService initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ChartService: {str(e)}")
        logger.error(traceback.format_exc())
        return
    
    # Instruments to test - start with just one from each category to test
    test_instruments = {
        "Forex": ["EURUSD"],
        "Indices": ["US500"],
        "Commodities": ["XAUUSD"]
    }
    
    timeframe = "1h"
    
    # Test chart generation for each instrument
    for category, instruments in test_instruments.items():
        logger.info(f"\n--- Testing Chart Generation for Category: {category} ---")
        
        for instrument in instruments:
            logger.info(f"Generating chart for {instrument} ({timeframe})")
            
            try:
                # Get technical analysis
                logger.info(f"Getting technical analysis for {instrument}")
                analysis = await chart_service.get_technical_analysis(instrument, timeframe)
                
                if analysis and "not available" not in analysis:
                    logger.info(f"Successfully retrieved technical analysis for {instrument}")
                    logger.info(f"Analysis preview (first 300 chars): {analysis[:300]}...")
                else:
                    logger.error(f"Failed to retrieve valid analysis for {instrument}")
                    if analysis:
                        logger.error(f"Analysis error message: {analysis}")
                
                # Get chart image
                logger.info(f"Getting chart image for {instrument}")
                start_time = time.time()
                chart_bytes = await chart_service.get_chart(instrument, timeframe)
                end_time = time.time()
                
                logger.info(f"Chart generation took {end_time - start_time:.2f} seconds")
                
                if chart_bytes and len(chart_bytes) > 1000:
                    logger.info(f"Successfully retrieved chart for {instrument}: {len(chart_bytes)} bytes")
                    
                    # Save chart image for inspection
                    output_file = f"test_chart_{instrument.lower()}.png"
                    with open(output_file, "wb") as f:
                        f.write(chart_bytes)
                    logger.info(f"Chart image saved to {output_file}")
                else:
                    if chart_bytes:
                        logger.error(f"Chart image for {instrument} is too small: {len(chart_bytes)} bytes")
                    else:
                        logger.error(f"Failed to retrieve chart for {instrument}: No data returned")
            
            except Exception as e:
                logger.error(f"Error processing {instrument}: {str(e)}")
                logger.error(traceback.format_exc())
            
            # Add delay between requests to avoid rate limiting
            await asyncio.sleep(2)
    
    # Clean up resources
    try:
        await chart_service.cleanup()
        logger.info("ChartService cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during ChartService cleanup: {str(e)}")
    
    logger.info("Test completed")

if __name__ == "__main__":
    asyncio.run(test_yahoo_chart()) 