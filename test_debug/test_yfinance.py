import sys
import os
import logging

# Debug logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
logger.info(f"Added {parent_dir} to sys.path")

# Import the YahooFinanceProvider
try:
    from trading_bot.services.chart_service.yfinance_provider import YahooFinanceProvider
    logger.info("Successfully imported YahooFinanceProvider")
    
    # Test the _format_symbol method
    symbols = ["XAUUSD", "XAGUSD", "EURUSD", "XTIUSD", "US30"]
    for symbol in symbols:
        formatted = YahooFinanceProvider._format_symbol(symbol)
        logger.info(f"Symbol {symbol} formatted to: {formatted}")
        
    # Inspect the method
    import inspect
    logger.info("\nMethod source:")
    logger.info(inspect.getsource(YahooFinanceProvider._format_symbol))
    
except Exception as e:
    logger.error(f"Error importing YahooFinanceProvider: {str(e)}")
    import traceback
    logger.error(traceback.format_exc()) 