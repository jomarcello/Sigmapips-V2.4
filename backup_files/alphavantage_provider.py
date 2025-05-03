import logging
import traceback
import asyncio
import os
from typing import Optional, Dict, Any, Tuple
import time
import pandas as pd
from datetime import datetime, timedelta
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tenacity import retry, stop_after_attempt, wait_exponential
import numpy as np
from cachetools import TTLCache
import matplotlib.pyplot as plt
import mplfinance as mpf
from io import BytesIO
import hashlib

logger = logging.getLogger(__name__)

# Extra logging bij het starten van de module
logger.info("=== Initializing Alpha Vantage Provider ===")

# API Key for Alpha Vantage
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "NLZRCK6C9B8RC8JP")

# --- Cache Configuration ---
# Cache for raw downloaded data (symbol, interval) -> DataFrame
# Cache for 30 minutes (1800 seconds)
data_download_cache = TTLCache(maxsize=100, ttl=1800) 
# Cache for processed market data (symbol, timeframe, limit) -> DataFrame with indicators
market_data_cache = TTLCache(maxsize=100, ttl=1800)

class AlphaVantageProvider:
    """Provider class for Alpha Vantage API integration"""
    
    # Cache data to minimize API calls
    _cache = {}
    _cache_timeout = 3600  # Cache timeout in seconds (1 hour)
    _last_api_call = 0
    _min_delay_between_calls = 15  # Alpha Vantage has a limit of 5 API requests per minute for free tier
    _429_backoff_time = 60  # Increased from 30 to 60 seconds
    _session = None
    
    # Track 429 errors
    _429_count = 0
    _429_last_time = 0
    _max_429_count = 3  # Maximum number of 429 errors before extended backoff

    @staticmethod
    def _get_session():
        """Get or create a requests session with retry logic"""
        if AlphaVantageProvider._session is None:
            session = requests.Session()
            
            # Use a single consistent modern browser user agent
            user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            
            retries = Retry(
                total=5,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "POST", "OPTIONS"]
            )
            adapter = HTTPAdapter(max_retries=retries, pool_maxsize=10)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            # Set the consistent user agent
            session.headers.update({
                'User-Agent': user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Pragma': 'no-cache',
                'Cache-Control': 'no-cache',
            })
            
            # For Railway environment: try to use proxies if available
            if os.environ.get('ENVIRONMENT') == 'production' or os.environ.get('RAILWAY_ENVIRONMENT') is not None:
                try:
                    # Check if HTTP_PROXY or HTTPS_PROXY environment variables are set
                    http_proxy = os.environ.get('HTTP_PROXY')
                    https_proxy = os.environ.get('HTTPS_PROXY')
                    
                    if http_proxy or https_proxy:
                        proxies = {}
                        if http_proxy:
                            proxies['http'] = http_proxy
                        if https_proxy:
                            proxies['https'] = https_proxy
                            
                        session.proxies.update(proxies)
                        logger.info(f"Using proxy settings for Alpha Vantage requests: {proxies}")
                except Exception as e:
                    logger.error(f"Error setting up proxies: {str(e)}")
            
            AlphaVantageProvider._session = session
        return AlphaVantageProvider._session
    
    @staticmethod
    async def _wait_for_rate_limit():
        """Wait if we've hit the rate limit with adaptive backoff for 429 errors"""
        current_time = time.time()
        delay = AlphaVantageProvider._min_delay_between_calls
        
        # Check if we've been experiencing 429 errors recently
        if AlphaVantageProvider._429_count > 0:
            # If recent 429 error (within last 30 minutes)
            if current_time - AlphaVantageProvider._429_last_time < 1800:  # 30 minutes
                # Apply exponential backoff based on 429 count
                backoff_multiplier = min(2 ** AlphaVantageProvider._429_count, 32)  # Cap at 32x
                delay = AlphaVantageProvider._min_delay_between_calls * backoff_multiplier
                logger.warning(f"[Alpha] Using 429 backoff delay of {delay:.2f}s (429 count: {AlphaVantageProvider._429_count})")
                
                # Add random jitter to avoid thundering herd
                delay += random.uniform(1, 5)
            else:
                # Reset 429 count if no recent 429s
                AlphaVantageProvider._429_count = 0
        
        # Standard rate limiting
        if AlphaVantageProvider._last_api_call > 0:
            time_since_last_call = current_time - AlphaVantageProvider._last_api_call
            if time_since_last_call < delay:
                wait_time = delay - time_since_last_call + random.uniform(0.5, 2.0)  # Increased jitter
                logger.info(f"[Alpha] Rate limiting: Waiting {wait_time:.2f} seconds before next call")
                await asyncio.sleep(wait_time)
                
        AlphaVantageProvider._last_api_call = time.time()

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30), # Adjusted retry wait
        reraise=True
    )
    async def _download_data(symbol: str, interval: str = None, outputsize: str = "compact") -> pd.DataFrame:
        """Download data using Alpha Vantage API with retry logic and caching."""
        logger.info(f"[Alpha] Starting download for {symbol} with interval={interval}, outputsize={outputsize}")
        
        # --- Caching Logic ---
        # Cache key represents the actual request made
        cache_key = (symbol, interval, outputsize)

        if cache_key in data_download_cache:
            logger.info(f"[Alpha Cache] HIT for download: Key={cache_key}")
            df = data_download_cache[cache_key]
            if df is not None and not df.empty:
                return df.copy() # Return a copy to prevent mutation
            else:
                logger.warning(f"[Alpha Cache] Invalid cached data for {symbol} (empty or None). Removing from cache.")
                del data_download_cache[cache_key]
                
        logger.info(f"[Alpha Cache] MISS for download: Key={cache_key}")
        # --- End Caching Logic ---

        # Wait for rate limit
        await AlphaVantageProvider._wait_for_rate_limit()
        
        # Ensure session exists
        session = AlphaVantageProvider._get_session()
        
        # Define function to execute in the thread pool
        def download():
            try:
                logger.info(f"[Alpha] Executing download for {symbol} with interval={interval}")
                
                # Determine which API function to use based on interval and symbol
                api_symbol = symbol
                
                # For commodities, use ETF equivalents
                if symbol == "GOLD" or symbol == "XAU/USD":
                    api_symbol = "GLD" # GLD ETF for Gold
                elif symbol == "CL" or symbol == "XTI/USD":
                    api_symbol = "USO" # USO ETF for Oil
                elif symbol == "SILVER" or symbol == "XAG/USD":
                    api_symbol = "SLV" # SLV ETF for Silver
                elif symbol == "BRENT" or symbol == "XBR/USD":
                    api_symbol = "BNO" # BNO ETF for Brent Oil
                
                # For forex pairs, use the FX API
                is_forex = "/" in symbol and len(symbol.split("/")) == 2
                
                if is_forex:
                    # Handle forex pairs with the FX_DAILY or FX_INTRADAY API
                    from_currency, to_currency = symbol.split("/")
                    
                    if interval and interval.endswith('min'):
                        # Use intraday FX data
                        function = "FX_INTRADAY"
                        url = "https://www.alphavantage.co/query"
                        params = {
                            "function": function,
                            "from_symbol": from_currency,
                            "to_symbol": to_currency,
                            "interval": interval,
                            "apikey": ALPHA_VANTAGE_API_KEY,
                            "outputsize": outputsize,
                            "datatype": "json"
                        }
                    else:
                        # Use daily FX data
                        function = "FX_DAILY"
                        url = "https://www.alphavantage.co/query"
                        params = {
                            "function": function,
                            "from_symbol": from_currency,
                            "to_symbol": to_currency,
                            "apikey": ALPHA_VANTAGE_API_KEY,
                            "outputsize": outputsize,
                            "datatype": "json"
                        }
                elif interval and interval.endswith('min'):
                    # Use intraday for minute-based intervals
                    function = "TIME_SERIES_INTRADAY"
                    url = "https://www.alphavantage.co/query"
                    params = {
                        "function": function,
                        "symbol": api_symbol,
                        "interval": interval,
                        "apikey": ALPHA_VANTAGE_API_KEY,
                        "outputsize": outputsize,
                        "datatype": "json"
                    }
                else:
                    # Use daily data for other cases
                    function = "TIME_SERIES_DAILY"
                    url = "https://www.alphavantage.co/query"
                    params = {
                        "function": function,
                        "symbol": api_symbol,
                        "apikey": ALPHA_VANTAGE_API_KEY,
                        "outputsize": outputsize,
                        "datatype": "json"
                    }
                
                # Log the API request parameters
                logger.info(f"[Alpha] Making API request with parameters: {params}")
                
                # Execute the request
                response = session.get(url, params=params)
                
                # Check for HTTP errors
                response.raise_for_status()
                
                # Parse JSON response
                data = response.json()
                
                # Check for error messages
                if "Error Message" in data:
                    raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")
                
                if "Information" in data and "Note" in data:
                    logger.warning(f"[Alpha] API limit information: {data['Information']}, Note: {data['Note']}")
                
                # Log keys in response for debugging
                logger.info(f"[Alpha] Response keys: {data.keys()}")
                
                # Extract the time series data based on function
                time_series_key = None
                if function == "TIME_SERIES_INTRADAY":
                    time_series_key = f"Time Series ({interval})"
                elif function == "TIME_SERIES_DAILY":
                    time_series_key = "Time Series (Daily)"
                elif function == "FX_DAILY":
                    time_series_key = "Time Series FX (Daily)"
                elif function == "FX_INTRADAY":
                    time_series_key = f"Time Series FX ({interval})"
                
                if time_series_key not in data:
                    logger.error(f"[Alpha] Could not find time series data in response. Keys: {data.keys()}")
                    return None
                
                # Convert to DataFrame
                time_series = data[time_series_key]
                df = pd.DataFrame.from_dict(time_series, orient='index')
                
                # Rename columns to match expected format
                column_rename_map = {
                    "1. open": "Open",
                    "2. high": "High",
                    "3. low": "Low",
                    "4. close": "Close",
                    "5. volume": "Volume"
                }
                
                # FX data doesn't have volume
                if function.startswith("FX_"):
                    column_rename_map = {
                        "1. open": "Open",
                        "2. high": "High",
                        "3. low": "Low",
                        "4. close": "Close"
                    }
                
                # Rename columns
                df = df.rename(columns=column_rename_map)
                
                # Convert string values to float
                for col in ["Open", "High", "Low", "Close"]:
                    if col in df.columns:
                        df[col] = df[col].astype(float)
                
                # Handle volume
                if "Volume" in df.columns:
                    df["Volume"] = df["Volume"].astype(float)
                else:
                    # For FX data or other data without volume, add dummy column
                    df["Volume"] = 0
                
                # Convert index to datetime and sort
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                
                logger.info(f"[Alpha] Successfully downloaded data for {symbol} with shape {df.shape}")
                
                # Cache the result
                data_download_cache[cache_key] = df.copy()
                
                return df
                
            except Exception as e:
                logger.error(f"[Alpha] Error downloading data: {str(e)}")
                traceback.print_exc()
                return None
        
        # Execute download in thread pool (since requests is blocking)
        loop = asyncio.get_running_loop()
        df = await loop.run_in_executor(None, download)
        
        if df is None or df.empty:
            logger.error(f"[Alpha] No data returned or error for {symbol}")
            return None
            
        return df

    @staticmethod
    def _validate_and_clean_data(df: pd.DataFrame, instrument: str = None) -> pd.DataFrame:
        """Validate and clean the data"""
        try:
            # Check if DataFrame is valid
            if df is None or df.empty:
                logger.error(f"[Alpha] Invalid DataFrame for {instrument}: None or empty")
                return None
                
            # Ensure the DataFrame has the required columns
            required_cols = ['Open', 'High', 'Low', 'Close']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"[Alpha] Missing required columns for {instrument}. Available: {df.columns.tolist()}")
                return None
                
            # Remove duplicate indices
            if df.index.duplicated().sum() > 0:
                df = df[~df.index.duplicated(keep='last')]
                
            # Remove NaN values
            df = df.dropna()
            
            # Ensure index is sorted
            df = df.sort_index()
            
            logger.info(f"[Alpha] Validation successful, shape after cleaning: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"[Alpha] Error validating data: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    @staticmethod
    def _map_timeframe_to_alpha_vantage(timeframe: str) -> Optional[str]:
        """Map timeframe to Alpha Vantage interval"""
        timeframe_map = {
            "M1": "1min",
            "M5": "5min",
            "M15": "15min",
            "M30": "30min",
            "H1": "60min",
            "H4": None,  # Alpha Vantage doesn't have 4h directly, will require resampling
            "D1": "daily",
            "W1": "weekly",
            "MN1": "monthly",
            # Alternative format
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "60min",
            "4h": None,  # Alpha Vantage doesn't have 4h, will require resampling
            "1d": "daily",
            "1w": "weekly",
            "1mo": "monthly",
        }
        
        return timeframe_map.get(timeframe)

    @staticmethod
    async def get_market_data(symbol: str, limit: int = 100) -> Optional[Tuple[pd.DataFrame, Dict]]:
        """
        Fetches market data from Alpha Vantage API for a FIXED timeframe (H1), validates it, and calculates indicators.
        Returns a tuple: (DataFrame with indicators, analysis_info dictionary)
        """
        try:
            logger.info(f"[Alpha] Fetching market data for {symbol} (limit: {limit})")
        except Exception as e:
            logger.error(f"[Alpha] Error in initial check: {e}")
            
        # Using fixed timeframe of H1 (1 hour)
        fixed_timeframe = "H1" 
        
        # Generate cache key based on symbol, timeframe and limit
        cache_key = (symbol, fixed_timeframe, limit)
 
        if cache_key in market_data_cache:
            logger.info(f"[Alpha Cache] HIT for market data: {symbol} timeframe {fixed_timeframe} limit {limit}")
            cached_result = market_data_cache[cache_key]
            # Check if the cached result is None or a tuple before unpacking
            if cached_result is None:
                logger.warning(f"[Alpha Cache] Cached value was None for {symbol}")
                return None, None
            # Ensure we have a valid tuple
            if isinstance(cached_result, tuple) and len(cached_result) == 2:
                cached_df, cached_info = cached_result
                return cached_df.copy(), cached_info.copy() # Return copies
            else:
                logger.warning(f"[Alpha Cache] Invalid cached format for {symbol}, expected tuple, got {type(cached_result)}")
                # Remove invalid format from cache
                del market_data_cache[cache_key]
                # Continue with fetching new data
                
        logger.info(f"[Alpha Cache] MISS for market data: {symbol} timeframe {fixed_timeframe} limit {limit}")

        logger.info(f"[Alpha] Getting market data for {symbol} on fixed {fixed_timeframe} timeframe")
        df = None
        analysis_info = {}

        try:
            # 1. Format symbol and map timeframe
            formatted_symbol = AlphaVantageProvider._format_symbol(symbol, is_crypto=False, is_commodity=False)
            av_interval = AlphaVantageProvider._map_timeframe_to_alpha_vantage(fixed_timeframe)

            if not av_interval:
                 logger.error(f"[Alpha] Could not map fixed timeframe '{fixed_timeframe}' to Alpha Vantage interval.")
                 return None, None
                 
            # For the 1-hour timeframe, we need to use the intraday data with 60min interval
            if av_interval == "60min":
                outputsize = "full" if limit > 100 else "compact"
                df = await AlphaVantageProvider._download_data(
                    formatted_symbol, 
                    interval=av_interval,
                    outputsize=outputsize
                )
            else:
                # For daily data
                outputsize = "full" if limit > 100 else "compact"
                df = await AlphaVantageProvider._download_data(
                    formatted_symbol,
                    outputsize=outputsize
                )
                
            if df is None or df.empty:
                logger.warning(f"[Alpha] No data returned for {symbol} ({formatted_symbol}) after download attempt.")
                market_data_cache[cache_key] = None # Cache None result
                return None, None
                
            # Log success and data shape before validation
            logger.info(f"[Alpha] Successfully downloaded data for {symbol} with shape {df.shape}")
            
            # Validate and clean the data
            df = AlphaVantageProvider._validate_and_clean_data(df, symbol)
            if df is None:
                logger.error(f"[Alpha] Data validation failed for {symbol}")
                market_data_cache[cache_key] = None
                return None, None
                
            # For 4h timeframe, resample from 1h
            if fixed_timeframe == "4h" and av_interval == "60min":
                logger.info(f"[Alpha] Resampling 1h data to 4h for {symbol}")
                try:
                    # Ensure index is datetime before resampling
                    if not isinstance(df.index, pd.DatetimeIndex):
                         df.index = pd.to_datetime(df.index)
                         
                    # Ensure timezone information exists (UTC is common) for resampling
                    if df.index.tz is None:
                       df = df.tz_localize('UTC')
                    else:
                       df = df.tz_convert('UTC') # Convert to UTC if needed

                    # Define resampling logic
                    resample_logic = {
                        'Open': 'first',
                        'High': 'max',
                        'Low': 'min',
                        'Close': 'last',
                        'Volume': 'sum'
                    }
                    # Filter out columns not present in df
                    resample_logic = {k: v for k, v in resample_logic.items() if k in df.columns}

                    df_resampled = df.resample('4H', label='right', closed='right').agg(resample_logic)
                    df_resampled.dropna(inplace=True) # Drop rows where any value is NaN
                    
                    if df_resampled.empty:
                         logger.warning(f"[Alpha] Resampling to 4h resulted in empty DataFrame for {symbol}. Using 1h data instead.")
                         # Stick with df (1h) if resampling fails
                    else:
                         df = df_resampled # Use the resampled data
                         logger.info(f"[Alpha] Successfully resampled to 4h with shape {df.shape}")
                         
                except Exception as resample_e:
                    logger.error(f"[Alpha] Error resampling to 4h: {str(resample_e)}")
                    # Continue with 1h data (df) if resampling fails
            
            # Ensure we have enough data for indicators
            if len(df) < limit:
                 logger.warning(f"[Alpha] Insufficient data after cleaning/resampling for {symbol} (got {len(df)}, needed ~{limit}). Indicators might be inaccurate.")
                 # Proceed but log warning
            
            # Calculate indicators
            df_with_indicators = df.copy()
            indicators = {}
            
            try:
                # Ensure required columns exist
                required_cols = ['Open', 'High', 'Low', 'Close']
                if not all(col in df_with_indicators.columns for col in required_cols):
                     logger.error(f"[Alpha] Missing required columns {required_cols} for indicator calculation in {symbol}. Skipping indicators.")
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
                     logger.info(f"[Alpha] Calculating indicators for {symbol} manually using pandas")
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
                         logger.info(f"[Alpha] Indicators calculated. DataFrame columns: {df_with_indicators.columns.tolist()}")

                     except Exception as ta_error:
                         logger.error(f"[Alpha] Error calculating technical indicators for {symbol}: {ta_error}", exc_info=True)
                         # Return dataframe without indicators if calculation fails

            except Exception as indicator_e:
                 logger.error(f"[Alpha] Error calculating indicators for {symbol}: {indicator_e}")
                 # Continue without indicators or with partial indicators if possible

            # Limit the number of candles after calculations
            df_limited = df_with_indicators.iloc[-limit:]
            
            # Prepare result and cache
            result_df = df_limited.copy()
            
            # Cache the final result tuple (DataFrame, indicators_dict)
            market_data_cache[cache_key] = (result_df.copy(), indicators.copy())
            
            # Log the shape being returned
            logger.info(f"[Alpha] Returning market data for {symbol} with shape {result_df.shape}")

            return result_df, indicators
            
        except Exception as e:
            logger.error(f"[Alpha] Error processing market data for {symbol}: {str(e)}")
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
    def _format_symbol(instrument: str, is_crypto: bool, is_commodity: bool) -> str:
        """Format the instrument for Alpha Vantage API"""
        # Special cases for indices
        if instrument == "US30":
            return "DJI" # Dow Jones Industrial Average
        elif instrument == "US500":
            return "SPX" # S&P 500
        elif instrument == "US100":
            return "NDX" # Nasdaq 100
        elif instrument == "DE40":
            return "DAX" # German DAX
        elif instrument == "UK100":
            return "FTSE" # FTSE 100
        elif instrument == "JP225":
            return "NI225" # Nikkei 225
        
        # Forex pairs need to be formatted differently for Alpha Vantage (e.g. EUR/USD)
        if len(instrument) == 6 and not is_crypto and not is_commodity:
            if instrument == "EURUSD":
                return "EUR/USD"
            elif instrument == "GBPUSD":
                return "GBP/USD"
            elif instrument == "USDJPY":
                return "USD/JPY"
            else:
                # For other forex pairs, insert a slash in the middle
                return f"{instrument[:3]}/{instrument[3:]}"
        
        # Commodities
        if instrument == "XAUUSD":  # Gold
            return "GOLD" # Alpha Vantage symbol for Gold
        elif instrument == "XTIUSD" or instrument == "USOIL":  # Oil
            return "CL" # Alpha Vantage symbol for Crude Oil
        elif instrument == "XBRUSD":  # Brent oil
            return "BRENT" # Alpha Vantage symbol for Brent Oil
        elif instrument == "XAGUSD":  # Silver
            return "SILVER" # Alpha Vantage symbol for Silver
        
        # Cryptocurrencies
        if instrument == "BTCUSD":
            return "BTC/USD"
        elif instrument == "ETHUSD":
            return "ETH/USD"
        
        # Default: return as is
        return instrument

    @staticmethod
    def get_chart(instrument: str, timeframe: str = "1h", fullscreen: bool = False) -> Optional[bytes]:
        """Generate a chart for the given instrument using matplotlib."""
        try:
            # Map the instrument to Alpha Vantage symbol
            av_symbol = AlphaVantageProvider._format_symbol(instrument, 
                                                          is_crypto=instrument.startswith('BTC') or instrument.startswith('ETH'),
                                                          is_commodity=instrument.startswith('XAU') or instrument.startswith('XTI'))
            
            # Map timeframe to Alpha Vantage interval
            interval = AlphaVantageProvider._map_timeframe_to_alpha_vantage(timeframe)
            if not interval:
                logger.error(f"Invalid timeframe: {timeframe}")
                return None
            
            # Create session for API request
            session = requests.Session()
            
            # For commodities, use ETF equivalents
            api_symbol = av_symbol
            if av_symbol == "GOLD" or av_symbol == "XAU/USD":
                api_symbol = "GLD" # GLD ETF for Gold
            elif av_symbol == "CL" or av_symbol == "XTI/USD":
                api_symbol = "USO" # USO ETF for Oil
            elif av_symbol == "SILVER" or av_symbol == "XAG/USD":
                api_symbol = "SLV" # SLV ETF for Silver
            elif av_symbol == "BRENT" or av_symbol == "XBR/USD":
                api_symbol = "BNO" # BNO ETF for Brent Oil
            
            # For the Alpha Vantage API
            if interval == "60min":
                # Use intraday API
                url = "https://www.alphavantage.co/query"
                params = {
                    "function": "TIME_SERIES_INTRADAY",
                    "symbol": api_symbol,
                    "interval": interval,
                    "outputsize": "full",
                    "apikey": ALPHA_VANTAGE_API_KEY
                }
            else:
                # Use daily API
                url = "https://www.alphavantage.co/query"
                params = {
                    "function": "TIME_SERIES_DAILY",
                    "symbol": api_symbol,
                    "outputsize": "full",
                    "apikey": ALPHA_VANTAGE_API_KEY
                }
                
            logger.info(f"Getting data for chart: {api_symbol} with interval={interval}")
            
            # Get the data
            response = requests.get(url, params=params)
            data = response.json()
            
            # Extract the time series data
            time_series_key = None
            if interval == "60min":
                time_series_key = f"Time Series (60min)"
            else:
                time_series_key = "Time Series (Daily)"
            
            if time_series_key not in data:
                logger.error(f"No data returned for {api_symbol}")
                return None
                
            # Convert to DataFrame
            time_series = data[time_series_key]
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Rename columns
            df = df.rename(columns={
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. volume": "Volume"
            })
            
            # Convert string values to float
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df[col] = df[col].astype(float)
            
            # Convert index to datetime and sort
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            if df.empty:
                logger.error(f"No data returned for {api_symbol}")
                return None
                
            # Limit to the last 'num_candles' for better visualization
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
            last_price = float(data['Close'].iloc[-1].item())
            last_date = data.index[-1]
            price_text = f'${last_price:.4f}'
            
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
    def _generate_random_chart(instrument: str, timeframe: str = "1h") -> bytes:
        """Generate a random chart for demonstration when real data is unavailable"""
        try:
            logger.info(f"Generating random chart data for {instrument}")
            
            # Set a seed based on the instrument name for reproducibility
            seed = int(hashlib.md5(instrument.encode()).hexdigest(), 16) % 10000
            np.random.seed(seed)
            
            # Generate dates
            periods = 100
            end_date = datetime.now()
            
            if timeframe == "1h":
                dates = pd.date_range(end=end_date, periods=periods, freq='H')
            elif timeframe == "4h":
                dates = pd.date_range(end=end_date, periods=periods, freq='4H')
            elif timeframe == "1d":
                dates = pd.date_range(end=end_date, periods=periods, freq='D')
            else:
                dates = pd.date_range(end=end_date, periods=periods, freq='D')
            
            # Generate price with random walk
            base_price = 0
            
            # Set base price depending on the instrument
            if instrument == "EURUSD":
                base_price = 1.08
            elif instrument == "GBPUSD":
                base_price = 1.26
            elif instrument == "USDJPY":
                base_price = 150.0
            elif instrument == "XAUUSD":
                base_price = 2000.0
            elif instrument == "XTIUSD" or instrument == "USOIL":
                base_price = 75.0
            elif instrument == "BTCUSD":
                base_price = 60000.0
            elif instrument == "ETHUSD":
                base_price = 3000.0
            elif instrument == "US500":
                base_price = 5000.0
            elif instrument == "US100":
                base_price = 17000.0
            elif instrument == "US30":
                base_price = 38000.0
            else:
                # Default
                base_price = 100.0
            
            # Generate random walk
            random_walk = np.random.normal(0, base_price * 0.01, size=periods)
            random_walk = np.cumsum(random_walk)
            
            # Create price series
            prices = base_price + random_walk
            
            # Create OHLCV data
            ohlcv_data = []
            for i, date in enumerate(dates):
                price = prices[i]
                daily_volatility = base_price * 0.005  # 0.5% daily volatility
                
                open_price = price - daily_volatility * np.random.randn()
                high_price = max(price, open_price) + daily_volatility * abs(np.random.randn())
                low_price = min(price, open_price) - daily_volatility * abs(np.random.randn())
                close_price = price
                volume = np.random.randint(1000, 10000)
                
                ohlcv_data.append({
                    'Open': open_price,
                    'High': high_price,
                    'Low': low_price,
                    'Close': close_price,
                    'Volume': volume
                })
                
            # Create DataFrame
            df = pd.DataFrame(ohlcv_data, index=dates)
            
            # Generate chart
            logger.info(f"Creating random chart visualization for {instrument}")
            
            # Create figure with a nice size
            figsize = (10, 6)
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot the close price
            df['Close'].plot(ax=ax, color='blue', linewidth=1.5)
            
            # Add some EMAs
            df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
            df['EMA20'].plot(ax=ax, color='orange', linewidth=1.0, label='EMA 20')
            df['EMA50'].plot(ax=ax, color='red', linewidth=1.0, label='EMA 50')
            
            # Set title and labels
            ax.set_title(f"{instrument} - {timeframe} Chart (Demo Data)", fontsize=14)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price', fontsize=12)
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3)
            
            # Add legend
            ax.legend()
            
            # Format x-axis dates better
            fig.autofmt_xdate()
            
            # Add current price annotation at the last point
            last_price = float(df['Close'].iloc[-1])
            last_date = df.index[-1]
            price_precision = AlphaVantageProvider._get_instrument_precision(instrument)
            price_format = f'${{:.{price_precision}f}}'
            price_text = price_format.format(last_price)
            
            ax.annotate(price_text,
                        xy=(last_date, last_price),
                        xytext=(0, 10),
                        textcoords='offset points',
                        fontsize=10,
                        backgroundcolor='white',
                        bbox=dict(boxstyle='round,pad=0.3', edgecolor='gray', facecolor='white', alpha=0.7))
            
            # Add watermark
            plt.figtext(0.5, 0.01, "Demo Data - Not Real Market Prices", 
                        ha="center", fontsize=8, color="gray")
            
            # Make it look nice
            plt.tight_layout()
            
            # Save to BytesIO buffer
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100)
            plt.close(fig)
            
            # Get the image data
            buf.seek(0)
            image_data = buf.getvalue()
            
            logger.info(f"Random chart generated for {instrument}")
            return image_data
            
        except Exception as e:
            logger.error(f"Error generating random chart for {instrument}: {str(e)}")
            logger.error(traceback.format_exc())
            # Return an empty image if we can't generate a chart
            return b''

    @staticmethod
    def _get_reliable_date():
        """Get a date for reference purposes"""
        return datetime.now() 