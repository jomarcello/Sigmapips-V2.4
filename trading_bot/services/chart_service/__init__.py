"""
Chart Service module
"""
import logging

# Initialize logging
logger = logging.getLogger(__name__)
logger.info("Chart Service module loaded")

# Import providers with try/except to handle import errors
try:
    from .binance_provider import BinanceProvider
    logger.info("BinanceProvider imported successfully")
except ImportError:
    logger.warning("Failed to import BinanceProvider, will be unavailable")
    # Define a placeholder class to avoid errors
    class BinanceProvider:
        """Placeholder for BinanceProvider when actual module is unavailable"""
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("BinanceProvider is not available")

try:
    from .tradingview_provider import TradingViewProvider
    logger.info("TradingViewProvider imported successfully")
except ImportError:
    logger.warning("Failed to import TradingViewProvider, will be unavailable")
    # Define a placeholder class to avoid errors
    class TradingViewProvider:
        """Placeholder for TradingViewProvider when actual module is unavailable"""
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("TradingViewProvider is not available")

# Instead of direct import, define a getter function to delay the import
def get_chart_service():
    try:
        from .chart import ChartService
        return ChartService
    except ImportError as e:
        logger.error(f"Failed to import ChartService: {str(e)}")
        raise



