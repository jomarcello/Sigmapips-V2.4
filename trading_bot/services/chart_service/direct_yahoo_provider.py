import logging
import traceback
import asyncio
import os
import json
from typing import Optional, Dict, Any, Tuple
import time
import pandas as pd
from datetime import datetime, timedelta
import random
import numpy as np
from cachetools import TTLCache
import matplotlib.pyplot as plt
from io import BytesIO
import yfinance as yf

logger = logging.getLogger(__name__)

# Extra logging at module startup
logger.info("=== Initializing Direct Yahoo Finance Provider ===")

# --- Cache Configuration ---
# Cache for raw downloaded data (symbol, interval) -> DataFrame
# Cache for 30 minutes (1800 seconds)
data_download_cache = TTLCache(maxsize=100, ttl=1800) 
# Cache for processed market data (symbol, timeframe, limit) -> DataFrame with indicators
market_data_cache = TTLCache(maxsize=100, ttl=1800)

class DirectYahooProvider:
    """Provider class for Yahoo Finance using yfinance library directly"""
    
    # Cache data to minimize API calls
    _cache = {}
    _cache_timeout = 3600  # Cache timeout in seconds (1 hour)
    _last_api_call = 0
    _min_delay_between_calls = 5  # Verhoogd van 1 naar 5 seconden
    
    # 429 tracking
    _429_count = 0
    _429_last_time = 0
    _max_429_count = 3  # Maximum number of 429 errors before extended backoff

    @staticmethod
    async def _wait_for_rate_limit():
        """Enhanced rate limiting for API calls with 429 backoff logic"""
        current_time = time.time()
        delay = DirectYahooProvider._min_delay_between_calls
        
        # Check if we've been experiencing 429 errors recently
        if DirectYahooProvider._429_count > 0:
            # If recent 429 error (within last 30 minutes)
            if current_time - DirectYahooProvider._429_last_time < 1800:
                # Apply exponential backoff based on 429 count
                backoff_multiplier = min(2 ** DirectYahooProvider._429_count, 32)  # Cap at 32x
                delay = DirectYahooProvider._min_delay_between_calls * backoff_multiplier
                logger.warning(f"[YFinance] Using 429 backoff delay of {delay:.2f}s (429 count: {DirectYahooProvider._429_count})")
                
                # Add random jitter to avoid thundering herd
                delay += random.uniform(1, 5)
            else:
                # Reset 429 count if no recent 429s
                DirectYahooProvider._429_count = 0
        
        # Standard rate limiting
        if DirectYahooProvider._last_api_call > 0:
            time_since_last_call = current_time - DirectYahooProvider._last_api_call
            if time_since_last_call < delay:
                wait_time = delay - time_since_last_call + random.uniform(0.5, 2.0)  # Added jitter
                logger.info(f"[YFinance] Rate limiting: Waiting {wait_time:.2f} seconds before next call")
                await asyncio.sleep(wait_time)
                
        DirectYahooProvider._last_api_call = time.time()

    @staticmethod
    async def _download_data(symbol: str, interval: str = "1d", period: str = "1mo") -> pd.DataFrame:
        """Download data using yfinance library with caching."""
        logger.info(f"[YFinance] Starting download for {symbol} with interval={interval}, period={period}")
        
        # --- Caching Logic ---
        # Cache key represents the actual request made
        cache_key = (symbol, interval, period)

        if cache_key in data_download_cache:
            logger.info(f"[YFinance Cache] HIT for download: Key={cache_key}")
            df = data_download_cache[cache_key]
            if df is not None and not df.empty:
                return df.copy() # Return a copy to prevent mutation
            else:
                logger.warning(f"[YFinance Cache] Invalid cached data for {symbol} (empty or None). Removing from cache.")
                del data_download_cache[cache_key]
                
        logger.info(f"[YFinance Cache] MISS for download: Key={cache_key}")
        # --- End Caching Logic ---

        # Wait for rate limit
        await DirectYahooProvider._wait_for_rate_limit()
        
        # Define function to execute in the thread pool
        def download():
            try:
                logger.info(f"[YFinance] Executing download for {symbol} with interval={interval}, period={period}")
                
                # Use yfinance to download data
                ticker = yf.Ticker(symbol)
                
                # Get historical data
                df = ticker.history(period=period, interval=interval)
                
                if df is None or df.empty:
                    logger.error(f"[YFinance] No data returned for {symbol}")
                    return None
                
                # Rename columns to match our expected format
                df = df.rename(columns={
                    'Open': 'Open',
                    'High': 'High', 
                    'Low': 'Low',
                    'Close': 'Close',
                    'Volume': 'Volume'
                })
                
                # Drop unused columns
                for col in df.columns:
                    if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        df = df.drop(col, axis=1)
                
                logger.info(f"[YFinance] Successfully downloaded data for {symbol} with shape {df.shape}")
                
                # Cache the result
                data_download_cache[cache_key] = df.copy()
                
                return df
                
            except yf.exceptions.YFRateLimitError as rate_error:
                # Track 429 error for backoff algorithm
                DirectYahooProvider._429_count += 1
                DirectYahooProvider._429_last_time = time.time()
                
                logger.error(f"[YFinance] Rate limit (429) hit for {symbol}: {str(rate_error)}")
                logger.warning(f"[YFinance] Increasing backoff - 429 count now at {DirectYahooProvider._429_count}")
                return None
                
            except Exception as e:
                logger.error(f"[YFinance] Error downloading data: {str(e)}")
                traceback.print_exc()
                return None
        
        # Execute download in thread pool
        loop = asyncio.get_running_loop()
        df = await loop.run_in_executor(None, download)
        
        if df is None or df.empty:
            logger.error(f"[YFinance] No data returned or error for {symbol}")
            return None
            
        return df

    @staticmethod
    def _validate_and_clean_data(df: pd.DataFrame, instrument: str = None) -> pd.DataFrame:
        """Validate and clean the data"""
        try:
            # Check if DataFrame is valid
            if df is None or df.empty:
                logger.error(f"[YFinance] Invalid DataFrame for {instrument}: None or empty")
                return None
                
            # Ensure the DataFrame has the required columns
            required_cols = ['Open', 'High', 'Low', 'Close']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"[YFinance] Missing required columns for {instrument}. Available: {df.columns.tolist()}")
                return None
                
            # Remove duplicate indices
            if df.index.duplicated().sum() > 0:
                df = df[~df.index.duplicated(keep='last')]
                
            # Remove NaN values
            df = df.dropna()
            
            # Ensure index is sorted
            df = df.sort_index()
            
            logger.info(f"[YFinance] Validation successful, shape after cleaning: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"[YFinance] Error validating data: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    @staticmethod
    def _map_timeframe_to_interval(timeframe: str) -> Optional[str]:
        """Map timeframe to Yahoo Finance interval"""
        timeframe_map = {
            "M1": "1m",
            "M5": "5m",
            "M15": "15m",
            "M30": "30m",
            "H1": "1h",
            "H4": "4h",  # yfinance supports 4h
            "D1": "1d",
            "W1": "1wk",
            "MN1": "1mo",
            # Alternative format
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
            "1w": "1wk",
            "1mo": "1mo",
        }
        
        return timeframe_map.get(timeframe)

    @staticmethod
    def _map_timeframe_to_period(timeframe: str) -> str:
        """Map timeframe to a suitable period to fetch enough data"""
        timeframe_map = {
            "M1": "1d",     # For 1-minute data, fetch 1 day
            "M5": "5d",     # For 5-minute data, fetch 5 days
            "M15": "7d",    # For 15-minute data, fetch 7 days
            "M30": "7d",    # For 30-minute data, fetch 7 days
            "H1": "60d",    # For 1-hour data, fetch 60 days
            "H4": "120d",   # For 4-hour data, fetch 120 days
            "D1": "1y",     # For daily data, fetch 1 year
            "W1": "2y",     # For weekly data, fetch 2 years
            "MN1": "5y",    # For monthly data, fetch 5 years
            # Alternative format
            "1m": "1d",
            "5m": "5d",
            "15m": "7d",
            "30m": "7d",
            "1h": "60d",
            "4h": "120d",
            "1d": "1y",
            "1w": "2y",
            "1mo": "5y",
        }
        
        return timeframe_map.get(timeframe, "1mo")  # Default to 1 month if not found

    @staticmethod
    async def get_market_data(symbol: str, timeframe: str = "1h", limit: int = 100) -> Optional[Tuple[pd.DataFrame, Dict]]:
        """
        Fetches market data from Yahoo Finance, validates it, and calculates indicators.
        
        Args:
            symbol: The instrument symbol to fetch data for
            timeframe: The timeframe for the data (e.g., 1h, 4h, 1d)
            limit: The maximum number of data points to return
            
        Returns:
            A tuple: (DataFrame with indicators, analysis_info dictionary)
        """
        try:
            # Periodieke reset van 429 counter als het te lang geleden is (>3 uur)
            current_time = time.time()
            if current_time - DirectYahooProvider._429_last_time > 10800:  # 3 uur in seconden
                if DirectYahooProvider._429_count > 0:
                    logger.info(f"[YFinance] Resetting 429 counter (was {DirectYahooProvider._429_count}) after 3 hours")
                    DirectYahooProvider._429_count = 0
            
            logger.info(f"[YFinance] Fetching market data for {symbol} (timeframe: {timeframe}, limit: {limit})")
        except Exception as e:
            logger.error(f"[YFinance] Error in initial check: {e}")
            
        # Use the provided timeframe instead of a fixed one
        fixed_timeframe = timeframe
        
        # Generate cache key based on symbol, timeframe and limit
        cache_key = (symbol, fixed_timeframe, limit)
 
        if cache_key in market_data_cache:
            logger.info(f"[YFinance Cache] HIT for market data: {symbol} timeframe {fixed_timeframe} limit {limit}")
            cached_result = market_data_cache[cache_key]
            # Check if the cached result is None or a tuple before unpacking
            if cached_result is None:
                logger.warning(f"[YFinance Cache] Cached value was None for {symbol}")
                return None, None
            # Ensure we have a valid tuple
            if isinstance(cached_result, tuple) and len(cached_result) == 2:
                cached_df, cached_info = cached_result
                return cached_df.copy(), cached_info.copy() # Return copies
            else:
                logger.warning(f"[YFinance Cache] Invalid cached format for {symbol}, expected tuple, got {type(cached_result)}")
                # Remove invalid format from cache
                del market_data_cache[cache_key]
                # Continue with fetching new data
                
        logger.info(f"[YFinance Cache] MISS for market data: {symbol} timeframe {fixed_timeframe} limit {limit}")

        logger.info(f"[YFinance] Getting market data for {symbol} on timeframe {fixed_timeframe}")
        df = None
        analysis_info = {}

        try:
            # 1. Format symbol for Yahoo Finance
            yahoo_symbol = DirectYahooProvider._format_symbol(symbol)
            interval = DirectYahooProvider._map_timeframe_to_interval(fixed_timeframe)
            period = DirectYahooProvider._map_timeframe_to_period(fixed_timeframe)

            if not interval:
                 logger.error(f"[YFinance] Could not map timeframe '{fixed_timeframe}' to Yahoo Finance interval.")
                 return None, None
                 
            # Download data using yfinance
            df = await DirectYahooProvider._download_data(
                yahoo_symbol, 
                interval=interval,
                period=period
            )
                
            if df is None or df.empty:
                logger.warning(f"[YFinance] No data returned for {symbol} ({yahoo_symbol}) after download attempt.")
                market_data_cache[cache_key] = None # Cache None result
                return None, None
                
            # Log success and data shape before validation
            logger.info(f"[YFinance] Successfully downloaded data for {symbol} with shape {df.shape}")
            
            # Validate and clean the data
            df = DirectYahooProvider._validate_and_clean_data(df, symbol)
            if df is None:
                logger.error(f"[YFinance] Data validation failed for {symbol}")
                market_data_cache[cache_key] = None
                return None, None
            
            # Ensure we have enough data for indicators
            if len(df) < limit:
                 logger.warning(f"[YFinance] Insufficient data after cleaning for {symbol} (got {len(df)}, needed ~{limit}). Indicators might be inaccurate.")
                 # Proceed but log warning
            
            # Calculate indicators
            df_with_indicators = df.copy()
            indicators = {}
            
            try:
                # Ensure required columns exist
                required_cols = ['Open', 'High', 'Low', 'Close']
                if not all(col in df_with_indicators.columns for col in required_cols):
                     logger.error(f"[YFinance] Missing required columns {required_cols} for indicator calculation in {symbol}. Skipping indicators.")
                else:
                     # Safely access last row data
                     last_row = df_with_indicators.iloc[-1]
                     indicators = {
                          'open': float(last_row['Open']),
                          'high': float(last_row['High']),
                          'low': float(last_row['Low']),
                          'close': float(last_row['Close']),
                          'volume': float(last_row['Volume']) if 'Volume' in df_with_indicators.columns and pd.notna(last_row['Volume']) else 0
                     }

                     # Calculate Technical Indicators using pandas
                     logger.info(f"[YFinance] Calculating indicators for {symbol} manually using pandas")
                     try:
                         close = df_with_indicators['Close']

                         # EMA
                         df_with_indicators['EMA_20'] = close.ewm(span=20, adjust=False).mean()
                         df_with_indicators['EMA_50'] = close.ewm(span=50, adjust=False).mean()
                         df_with_indicators['EMA_200'] = close.ewm(span=200, adjust=False).mean()

                         # RSI
                         delta = close.diff()
                         gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                         loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                         rs = gain / loss
                         df_with_indicators['RSI_14'] = 100 - (100 / (1 + rs))

                         # MACD
                         ema_12 = close.ewm(span=12, adjust=False).mean()
                         ema_26 = close.ewm(span=26, adjust=False).mean()
                         df_with_indicators['MACD_12_26_9'] = ema_12 - ema_26
                         df_with_indicators['MACDs_12_26_9'] = df_with_indicators['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
                         df_with_indicators['MACDh_12_26_9'] = df_with_indicators['MACD_12_26_9'] - df_with_indicators['MACDs_12_26_9']
                         logger.info(f"[YFinance] Indicators calculated. DataFrame columns: {df_with_indicators.columns.tolist()}")

                     except Exception as ta_error:
                         logger.error(f"[YFinance] Error calculating technical indicators for {symbol}: {ta_error}", exc_info=True)
                         # Return dataframe without indicators if calculation fails

            except Exception as indicator_e:
                 logger.error(f"[YFinance] Error calculating indicators for {symbol}: {indicator_e}")
                 # Continue without indicators or with partial indicators if possible

            # Limit the number of candles after calculations
            df_limited = df_with_indicators.iloc[-limit:]
            
            # Prepare result and cache
            result_df = df_limited.copy()
            
            # Cache the final result tuple (DataFrame, indicators_dict)
            market_data_cache[cache_key] = (result_df.copy(), indicators.copy())
            
            # Log the shape being returned
            logger.info(f"[YFinance] Returning market data for {symbol} with shape {result_df.shape}")

            return result_df, indicators
            
        except Exception as e:
            logger.error(f"[YFinance] Error processing market data for {symbol}: {str(e)}")
            logger.error(traceback.format_exc())
            return None, None

    @staticmethod
    def _get_instrument_precision(instrument: str) -> int:
        """Get the appropriate decimal precision for an instrument"""
        instrument = instrument.upper().replace("/", "")
        
        # JPY pairs use 3 decimal places
        if instrument.endswith("JPY") or "JPY" in instrument:
            return 3
            
        # Most forex pairs use 5 decimal places
        if len(instrument) == 6 and all(c.isalpha() for c in instrument):
            return 5
            
        # Crypto typically uses 2 decimal places for major coins, more for smaller ones
        if any(crypto in instrument for crypto in ["BTC", "ETH", "LTC", "XRP"]):
            return 2
            
        # Gold typically uses 2 decimal places
        if instrument in ["XAUUSD", "GC=F"]:
            return 2
            
        # Silver typically uses 3 decimal places
        if instrument in ["XAGUSD", "SI=F"]:
            return 3
            
        # Indices typically use 2 decimal places
        if any(index in instrument for index in ["US30", "US500", "US100", "UK100", "DE40", "JP225"]):
            return 2
            
        # Default to 4 decimal places as a safe value
        return 4
    
    @staticmethod
    def _format_symbol(instrument: str) -> str:
        """Format the instrument for Yahoo Finance API"""
        try:
            logger.info(f"[YFinance] Formatting symbol: {instrument}")
            
            # Verwijder '=' tekens als die al aanwezig zijn om dubbele formatting te voorkomen
            if '=' in instrument:
                clean_instrument = instrument.split('=')[0]
                logger.info(f"[YFinance] Removed existing '=' from symbol: {instrument} -> {clean_instrument}")
                instrument = clean_instrument
                
            # Special cases for indices
            indices_map = {
                "US30": "^DJI",    # Dow Jones Industrial Average
                "US500": "^GSPC",  # S&P 500
                "US100": "^NDX",   # Nasdaq 100
                "DE40": "^GDAXI",  # German DAX
                "UK100": "^FTSE",  # FTSE 100
                "JP225": "^N225",  # Nikkei 225
            }
            
            if instrument in indices_map:
                result = indices_map[instrument]
                logger.info(f"[YFinance] Mapped index: {instrument} -> {result}")
                return result
            
            # Commodities with futures contracts
            commodities_map = {
                "XAUUSD": "GC=F",  # Gold futures 
                "XTIUSD": "CL=F",  # WTI Crude Oil futures
                "USOIL": "CL=F",   # WTI Crude Oil futures (alternative name)
                "XBRUSD": "BZ=F",  # Brent Oil futures
                "XAGUSD": "SI=F",  # Silver futures
            }
            
            if instrument in commodities_map:
                result = commodities_map[instrument]
                logger.info(f"[YFinance] Mapped commodity: {instrument} -> {result}")
                return result
            
            # Cryptocurrencies
            crypto_map = {
                "BTCUSD": "BTC-USD",
                "ETHUSD": "ETH-USD",
            }
            
            if instrument in crypto_map:
                result = crypto_map[instrument]
                logger.info(f"[YFinance] Mapped crypto: {instrument} -> {result}")
                return result
            
            # Forex pairs need to be formatted
            if len(instrument) == 6 and all(c.isalpha() for c in instrument):
                result = f"{instrument}=X"
                logger.info(f"[YFinance] Formatted forex pair: {instrument} -> {result}")
                return result
            
            # Als het instrument al een '=' bevat, return het origineel
            if '=' in instrument:
                logger.info(f"[YFinance] Symbol already contains '=', using as is: {instrument}")
                return instrument
                
            # Default: return as is with logging
            logger.info(f"[YFinance] Using symbol as is: {instrument}")
            return instrument
        except Exception as e:
            logger.error(f"[YFinance] Error formatting symbol {instrument}: {str(e)}")
            # Return original instrument in case of error
            return instrument

    @staticmethod
    def get_chart(instrument: str, timeframe: str = "1h", fullscreen: bool = False) -> Optional[bytes]:
        """Generate a chart for the given instrument using matplotlib."""
        try:
            # Map the instrument to Yahoo Finance symbol
            yahoo_symbol = DirectYahooProvider._format_symbol(instrument)
            
            # Map timeframe to Yahoo Finance interval and period
            interval = DirectYahooProvider._map_timeframe_to_interval(timeframe)
            period = DirectYahooProvider._map_timeframe_to_period(timeframe)
            
            if not interval:
                logger.error(f"Invalid timeframe: {timeframe}")
                return None
            
            logger.info(f"Getting data for chart: {instrument} ({yahoo_symbol}) with interval={interval}, period={period}")
            
            try:
                # Use yfinance to get the data
                ticker = yf.Ticker(yahoo_symbol)
                df = ticker.history(period=period, interval=interval)
                
                if df is None or df.empty:
                    logger.error(f"No data returned for {instrument} ({yahoo_symbol})")
                    return None
                
                # Ensure we have the expected columns
                required_cols = ['Open', 'High', 'Low', 'Close']
                if not all(col in df.columns for col in required_cols):
                    logger.error(f"Missing required columns for {instrument} ({yahoo_symbol}). Available: {df.columns.tolist()}")
                    return None
            
            except Exception as download_e:
                logger.error(f"Error downloading data for chart: {str(download_e)}")
                return None
            
            # Limit to the last 100 candles for better visualization
            data = df.tail(100)
            
            # Generate a nice looking chart with matplotlib
            logger.info(f"Generating matplotlib chart for {instrument} with {len(data)} rows")
            
            # Create figure with a nice size
            figsize = (12, 8) if fullscreen else (10, 6)
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot the close price
            data['Close'].plot(ax=ax, color='blue', linewidth=1.5)
            
            # Add some EMAs
            data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
            data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
            data['EMA20'].plot(ax=ax, color='orange', linewidth=1.0, label='EMA 20')
            data['EMA50'].plot(ax=ax, color='red', linewidth=1.0, label='EMA 50')
            
            # Set title and labels
            ax.set_title(f"{instrument} - {timeframe} Chart", fontsize=14)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price', fontsize=12)
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3)
            
            # Add legend
            ax.legend()
            
            # Format x-axis dates better
            fig.autofmt_xdate()
            
            # Add current price annotation at the last point
            last_price = float(data['Close'].iloc[-1])
            last_date = data.index[-1]
            price_precision = DirectYahooProvider._get_instrument_precision(instrument)
            price_format = f'{{:.{price_precision}f}}'
            price_text = price_format.format(last_price)
            
            ax.annotate(price_text,
                        xy=(last_date, last_price),
                        xytext=(0, 10),
                        textcoords='offset points',
                        fontsize=10,
                        backgroundcolor='white',
                        bbox=dict(boxstyle='round,pad=0.3', edgecolor='gray', facecolor='white', alpha=0.7))
            
            # Make it look nice
            plt.tight_layout()
            
            # Save to BytesIO buffer
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100)
            plt.close(fig)
            
            # Get the image data
            buf.seek(0)
            image_data = buf.getvalue()
            
            logger.info(f"Chart generated successfully for {instrument}: {len(image_data)} bytes")
            return image_data
            
        except Exception as e:
            logger.error(f"Error generating chart for {instrument}: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    @staticmethod
    def get_available_markets():
        """Get list of common markets/tickers from predefined list"""
        try:
            # Return a simple predefined list of common markets
            markets = {
                "forex": [
                    {"symbol": "EURUSD", "name": "Euro/US Dollar"},
                    {"symbol": "GBPUSD", "name": "British Pound/US Dollar"},
                    {"symbol": "USDJPY", "name": "US Dollar/Japanese Yen"},
                    {"symbol": "USDCHF", "name": "US Dollar/Swiss Franc"},
                    {"symbol": "AUDUSD", "name": "Australian Dollar/US Dollar"},
                    {"symbol": "NZDUSD", "name": "New Zealand Dollar/US Dollar"}
                ],
                "indices": [
                    {"symbol": "US30", "name": "Dow Jones Industrial Average"},
                    {"symbol": "US500", "name": "S&P 500"},
                    {"symbol": "US100", "name": "Nasdaq 100"},
                    {"symbol": "DE40", "name": "German DAX"},
                    {"symbol": "UK100", "name": "FTSE 100"},
                    {"symbol": "JP225", "name": "Nikkei 225"}
                ],
                "commodities": [
                    {"symbol": "XAUUSD", "name": "Gold"},
                    {"symbol": "XAGUSD", "name": "Silver"},
                    {"symbol": "XTIUSD", "name": "WTI Crude Oil"},
                    {"symbol": "XBRUSD", "name": "Brent Crude Oil"}
                ],
                "crypto": [
                    {"symbol": "BTCUSD", "name": "Bitcoin/US Dollar"},
                    {"symbol": "ETHUSD", "name": "Ethereum/US Dollar"}
                ]
            }
            
            return {"data": markets}
        
        except Exception as e:
            logger.error(f"Error getting available markets: {str(e)}")
            return None 