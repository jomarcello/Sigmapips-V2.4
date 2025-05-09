"""
Chart Service module
"""
import logging

# Instead of direct import, define a getter function to delay the import
def get_chart_service():
    from .chart import ChartService
    return ChartService

from .binance_provider import BinanceProvider
from .tradingview_provider import TradingViewProvider

# Initialize logging
logger = logging.getLogger(__name__)
logger.info("Chart Service module loaded")



