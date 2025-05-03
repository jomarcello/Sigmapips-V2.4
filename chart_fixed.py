import json
import random
import logging
import traceback
import warnings
from typing import Optional, Dict, Any, List, Union, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

class ChartService:
    def __init__(self):
        # Initialize your chart service here
        pass

    # Other methods...

    async def get_technical_analysis(self, instrument: str, timeframe: str = "1h") -> str:
        """
        Get technical analysis for an instrument based on provider data.
        """
        logger.info(f"Getting technical analysis for {instrument} on {timeframe} timeframe")
        
        # Normalize instrument name
        instrument = self._normalize_instrument_name(instrument)
        logger.info(f"Normalized instrument name: {instrument}")
        
        # Detect the market type
        market_type = await self._detect_market_type(instrument)
        logger.info(f"Detected market type: {market_type}")
        
        # For crypto, we use Binance; for others, we prefer Yahoo Finance
        is_crypto = market_type == "crypto"
        
        # Set up the appropriate providers
        binance_provider = None
        yahoo_provider = None
        alltick_provider = None
        
        try:
            if is_crypto:
                # Only initialize Binance for crypto
                from trading_bot.services.chart_service.binance_provider import BinanceProvider
                binance_provider = BinanceProvider()
        except Exception as e:
            logger.error(f"Error in provider setup for technical analysis: {str(e)}")
            return await self._generate_default_analysis(instrument, timeframe)
        
        # Main provider initialization and data fetching block
        try:
            # First try to load YahooFinance if needed
            try:
                if not is_crypto:  # Alleen Yahoo laden als het geen crypto is
                    from trading_bot.services.chart_service.yfinance_provider import YahooFinanceProvider
                    yahoo_provider = YahooFinanceProvider()
                else:
                    logger.info(f"Skipping Yahoo Finance provider initialization for crypto {instrument}")
            except Exception as e:
                logger.error(f"Failed to load YahooFinanceProvider: {str(e)}")
             
            # Rest of the method...
        except Exception as e:
            logger.error(f"Error during technical analysis: {str(e)}")
            return await self._generate_default_analysis(instrument, timeframe)

    async def _generate_default_analysis(self, instrument: str, timeframe: str) -> str:
        """Generate a fallback analysis when the API fails"""
        try:
            # Default values
            current_price = 0.0
            # Rest of the method...
        except Exception as e:
            logger.error(f"Error in default analysis: {str(e)}")
            return f"Unable to generate analysis for {instrument}. Please try again later." 