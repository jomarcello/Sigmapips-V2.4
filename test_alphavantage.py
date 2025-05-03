import asyncio
import logging
import sys
import os
from datetime import datetime
import pandas as pd

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

# Import Alpha Vantage provider
from trading_bot.services.chart_service.alphavantage_provider import AlphaVantageProvider

async def test_alphavantage():
    """Test Alpha Vantage data retrieval directly"""
    logger.info("Starting Alpha Vantage direct test...")
    
    # Create instance of the provider
    alpha_provider = AlphaVantageProvider()
    
    # Instruments to test with corresponding Alpha Vantage symbols
    test_instruments = {
        "EURUSD": "EUR/USD",
        "US500": "SPX",
        "XTIUSD": "CL",
        "XAUUSD": "GOLD"
    }
    
    # Test get_market_data method which is the main entry point
    logger.info("\nTesting get_market_data method...")
    for instrument, av_symbol in test_instruments.items():
        logger.info(f"\nTesting instrument: {instrument} (Alpha Vantage symbol: {av_symbol})")
        
        try:
            # Get market data
            df, metadata = await alpha_provider.get_market_data(instrument)
            
            if df is not None and not df.empty:
                # Show dataframe info
                logger.info(f"Successfully retrieved data:")
                logger.info(f"Dataframe shape: {df.shape}")
                logger.info(f"Dataframe columns: {df.columns.tolist()}")
                logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
                
                # Show a few rows
                logger.info("\nLatest data points:")
                logger.info(df.tail(5).to_string())
                
                # Show metadata
                logger.info("\nMetadata:")
                for key, value in metadata.items():
                    logger.info(f"{key}: {value}")
            else:
                logger.error(f"No data received for {instrument}")
                
        except Exception as e:
            logger.error(f"Error testing {instrument}: {str(e)}")
    
    # Now test the chart generation
    logger.info("\nTesting chart generation...")
    for instrument in test_instruments.keys():
        logger.info(f"\nGenerating chart for {instrument}")
        
        try:
            # Generate chart
            chart_bytes = alpha_provider.get_chart(instrument, timeframe="1h")
            
            # Save chart to file for inspection
            if chart_bytes:
                output_file = f"test_alpha_{instrument.lower()}.png"
                with open(output_file, "wb") as f:
                    f.write(chart_bytes)
                logger.info(f"Chart saved to {output_file} ({len(chart_bytes)} bytes)")
            else:
                logger.error(f"Failed to generate chart for {instrument}")
                
        except Exception as e:
            logger.error(f"Error generating chart for {instrument}: {str(e)}")

if __name__ == "__main__":
    # Run the async test
    asyncio.run(test_alphavantage()) 