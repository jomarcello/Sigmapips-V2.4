"""
Chart Service module
"""
import logging

# Import providers directly in __init__ 
from .binance_provider import BinanceProvider
from .tradingview_provider import TradingViewProvider

# Instead of direct import, define a getter function to delay the import
def get_chart_service():
    from .chart import ChartService
    return ChartService

# Initialize logging
logger = logging.getLogger(__name__)
logger.info("Chart Service module loaded")



