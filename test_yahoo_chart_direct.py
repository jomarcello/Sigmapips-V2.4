import logging
import sys
import os

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

# Import YahooFinanceProvider directly
from trading_bot.services.chart_service.yfinance_provider import YahooFinanceProvider

def test_yahoo_chart_direct():
    """Test YahooFinanceProvider.get_chart directly without using ChartService"""
    logger.info("Starting direct Yahoo Finance chart test...")
    
    # Instruments to test
    test_instruments = [
        "EURUSD",  # Forex
        "US500",   # Index (S&P 500)
        "XAUUSD"   # Commodity (Gold)
    ]
    
    timeframe = "1h"
    
    # Test chart generation for each instrument
    for instrument in test_instruments:
        logger.info(f"Generating chart for {instrument} ({timeframe})")
        
        try:
            # Get chart directly from YahooFinanceProvider
            chart_bytes = YahooFinanceProvider.get_chart(instrument, timeframe, fullscreen=True)
            
            if chart_bytes and len(chart_bytes) > 1000:
                logger.info(f"Successfully generated chart for {instrument}: {len(chart_bytes)} bytes")
                
                # Save chart image for inspection
                output_file = f"test_chart_direct_{instrument.lower()}.png"
                with open(output_file, "wb") as f:
                    f.write(chart_bytes)
                logger.info(f"Chart image saved to {output_file}")
            else:
                if chart_bytes:
                    logger.error(f"Chart image for {instrument} is too small: {len(chart_bytes)} bytes")
                else:
                    logger.error(f"Failed to generate chart for {instrument}: No data returned")
        
        except Exception as e:
            logger.error(f"Error generating chart for {instrument}: {str(e)}")
            logger.error(f"Traceback: {logging.traceback.format_exc()}")

if __name__ == "__main__":
    test_yahoo_chart_direct() 