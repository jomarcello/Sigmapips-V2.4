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
import yfinance as yf
import numpy as np
from cachetools import TTLCache
import matplotlib.pyplot as plt
import mplfinance as mpf
from io import BytesIO

logger = logging.getLogger(__name__)

# Extra logging bij het starten van de module
logger.info("=== Initializing Yahoo Finance Provider ===")
logger.info(f"yfinance version: {yf.__version__ if hasattr(yf, '__version__') else 'unknown'}")

# --- Cache Configuration ---
# Cache for raw downloaded data (symbol, interval) -> DataFrame
# Cache for 30 minutes (1800 seconds)
data_download_cache = TTLCache(maxsize=100, ttl=1800) 
# Cache for processed market data (symbol, timeframe, limit) -> DataFrame with indicators
market_data_cache = TTLCache(maxsize=100, ttl=1800)

class YahooFinanceProvider:
    """Provider class for Yahoo Finance API integration"""
    
    # Cache data to minimize API calls
    _cache = {}
    _cache_timeout = 3600  # Cache timeout in seconds (1 hour)
    _last_api_call = 0
    _min_delay_between_calls = 5  # Increased from 2 to 5 seconds
    _429_backoff_time = 60  # Increased from 30 to 60 seconds
    _session = None
    
    # Track 429 errors
    _429_count = 0
    _429_last_time = 0
    _max_429_count = 3  # Maximum number of 429 errors before extended backoff

    @staticmethod
    def _get_session():
        """Get or create a requests session with retry logic"""
        if YahooFinanceProvider._session is None:
            session = requests.Session()
            
            # Use a single consistent modern browser user agent instead of rotating
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
                        logger.info(f"Using proxy settings for Yahoo Finance requests: {proxies}")
                except Exception as e:
                    logger.error(f"Error setting up proxies: {str(e)}")
            
            YahooFinanceProvider._session = session
        return YahooFinanceProvider._session
    
    @staticmethod
    async def _wait_for_rate_limit():
        """Wait if we've hit the rate limit with adaptive backoff for 429 errors"""
        current_time = time.time()
        delay = YahooFinanceProvider._min_delay_between_calls
        
        # Check if we've been experiencing 429 errors recently
        if YahooFinanceProvider._429_count > 0:
            # If recent 429 error (within last 30 minutes)
            if current_time - YahooFinanceProvider._429_last_time < 1800:  # 30 minutes instead of 5
                # Apply exponential backoff based on 429 count
                backoff_multiplier = min(2 ** YahooFinanceProvider._429_count, 32)  # Cap at 32x instead of 16x
                delay = YahooFinanceProvider._min_delay_between_calls * backoff_multiplier
                logger.warning(f"[Yahoo] Using 429 backoff delay of {delay:.2f}s (429 count: {YahooFinanceProvider._429_count})")
                
                # Add random jitter to avoid thundering herd
                delay += random.uniform(1, 5)
            else:
                # Reset 429 count if no recent 429s
                YahooFinanceProvider._429_count = 0
        
        # Standard rate limiting
        if YahooFinanceProvider._last_api_call > 0:
            time_since_last_call = current_time - YahooFinanceProvider._last_api_call
            if time_since_last_call < delay:
                wait_time = delay - time_since_last_call + random.uniform(0.5, 2.0)  # Increased jitter
                logger.info(f"[Yahoo] Rate limiting: Waiting {wait_time:.2f} seconds before next call")
                await asyncio.sleep(wait_time)
                
        YahooFinanceProvider._last_api_call = time.time()

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30), # Adjusted retry wait
        reraise=True
    )
    async def _download_data(symbol: str, start_date: datetime = None, end_date: datetime = None, interval: str = None, timeout: int = 30, original_symbol: str = None, period: str = None) -> pd.DataFrame:
        """Download data using yfinance with retry logic and caching."""
        logger.info(f"[Yahoo] Starting download for {symbol} with interval={interval}, period={period}")
        
        # --- Caching Logic ---
        # Cache key should ideally represent the actual request made
        if period:
            # Use period and interval for intraday caching
            cache_key = (symbol, interval, period)
        elif start_date and end_date:
            # Use start/end date and interval for daily+ caching
            cache_key = (symbol, interval, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        else:
            # Fallback cache key if parameters are weird (should not happen often)
            logger.warning("Could not determine proper cache key, using fallback.")
            # Use today's date for cache key
            current_date = datetime.now().strftime('%Y-%m-%d')
            cache_key = (symbol, interval, current_date)
            logger.info(f"[Yahoo] Using current date for cache key: {current_date}")

        if cache_key in data_download_cache:
            logger.info(f"[Yahoo Cache] HIT for download: Key={cache_key}")
            df = data_download_cache[cache_key]
            if df is not None and not df.empty:
                return df.copy() # Return a copy to prevent mutation
            else:
                logger.warning(f"[Yahoo Cache] Invalid cached data for {symbol} (empty or None). Removing from cache.")
                del data_download_cache[cache_key]
                
        logger.info(f"[Yahoo Cache] MISS for download: Key={cache_key}")
        # --- End Caching Logic ---

        # Ensure valid dates or period
        now = YahooFinanceProvider._get_reliable_date()
        
        # If we're using period, make sure it's valid
        if period:
            logger.info(f"[Yahoo] Using period parameter: {period} for {symbol}")
        # If using dates, validate them
        else:
            if end_date is None:
                end_date = now
                logger.info(f"[Yahoo] No end_date provided, using current date: {end_date.date()}")
                
            if start_date is None:
                if interval == '1h':
                    # For hourly data, get ~7 days by default
                    start_date = end_date - timedelta(days=7)
                elif interval == '1d':
                    # For daily data, get ~100 days by default
                    start_date = end_date - timedelta(days=100)
                else:
                    # Default: 30 days
                    start_date = end_date - timedelta(days=30)
                logger.info(f"[Yahoo] No start_date provided, using {start_date.date()} (based on interval {interval})")
                
            # Make sure dates are valid
            if end_date > now:
                logger.warning(f"[Yahoo] End date {end_date.date()} is in the future, using current date")
                end_date = now
                
            if start_date > now or start_date > end_date:
                logger.warning(f"[Yahoo] Start date {start_date.date()} is invalid, adjusting")
                start_date = end_date - timedelta(days=30)
        
        logger.info(f"[Yahoo] Attempting direct download method with yf.download for {symbol} (Interval: {interval}, Period: {period}, Start: {start_date.date() if start_date else 'N/A'}, End: {end_date.date() if end_date else 'N/A'})")
        
        # Ensure session exists
        session = YahooFinanceProvider._get_session()
        
        # Function to perform the download (runs in executor)
        def download():
            try:
                # Add explicit logging
                logger.info(f"[Yahoo] Executing download for {symbol}, format: {'period' if period else 'start/end'}")
                
                # Check if set_tz_session_for_downloading exists (handle different yfinance versions)
                if hasattr(yf.multi, 'set_tz_session_for_downloading'):
                     yf.multi.set_tz_session_for_downloading(session)
                else:
                     logger.warning("[Yahoo] Function set_tz_session_for_downloading not available in this yfinance version")

                # Construct download arguments
                download_kwargs = {
                    'tickers': symbol,
                    'progress': False,
                    'session': session,
                    'timeout': timeout,
                    'ignore_tz': False,
                    'threads': False,  # Disable multi-threading to avoid issues
                    'proxy': None,  # Let the session handle proxy settings
                    'actions': False,  # No need for dividends/splits
                    'auto_adjust': True,  # Auto-adjust data
                    'prepost': False,  # No pre/post market data for forex
                    'rounding': True,  # Round values to appropriate precision
                }
                
                # Voeg het interval toe als het niet None is
                if interval:
                    download_kwargs['interval'] = interval
                
                # Use period OR start/end, not both - prefer period
                if period:
                     download_kwargs['period'] = period
                     logger.info(f"[Yahoo] Using period={period} for download")
                elif start_date is not None and end_date is not None:
                     download_kwargs['start'] = start_date
                     download_kwargs['end'] = end_date
                     logger.info(f"[Yahoo] Using start={start_date.date()} and end={end_date.date()} for download")
                else:
                     # Fallback naar een veilige periode als er geen parameters zijn
                     download_kwargs['period'] = '7d'  # Standaard 7 dagen
                     logger.info(f"[Yahoo] Using fallback period=7d for download")

                # Download data - explicit logging
                logger.info(f"[Yahoo] Executing yf.download with params: interval={interval}, {'period='+period if period else f'start={start_date.date()}, end={end_date.date()}'}")
                df = yf.download(**download_kwargs)
                
                # Log result summary
                if df is not None and not df.empty:
                    logger.info(f"[Yahoo] Download successful, got {len(df)} rows for {symbol}")
                    logger.info(f"[Yahoo] Data range: {df.index[0]} to {df.index[-1]}")
                    logger.info(f"[Yahoo] Columns: {df.columns.tolist()}")
                else:
                    logger.warning(f"[Yahoo] Download returned empty DataFrame for {symbol}")
                    
                return df
                
            except Exception as e:
                 logger.error(f"[Yahoo] Error during yf.download for {symbol}: {str(e)}")
                 # Check for 429 error specifically
                 if "429" in str(e) or "too many requests" in str(e).lower():
                     # Update 429 tracking
                     YahooFinanceProvider._429_count += 1
                     YahooFinanceProvider._429_last_time = time.time()
                     logger.warning(f"[Yahoo] 429 Too Many Requests detected (count: {YahooFinanceProvider._429_count})")
                     # Add extra wait for 429
                     time.sleep(random.uniform(1.0, 3.0))  # Add immediate delay
                 # Add more specific error checks if needed (e.g., connection errors)
                 if "No data found" in str(e) or "symbol may be delisted" in str(e):
                     logger.warning(f"[Yahoo] No data found for {symbol} in range {start_date} to {end_date}")
                     return pd.DataFrame() # Return empty DataFrame on no data
                 raise # Reraise other exceptions for tenacity

        # Run the download in a separate thread to avoid blocking asyncio event loop
        try:
             # Use default executor (ThreadPoolExecutor) but get the loop more carefully
             try:
                 loop = asyncio.get_running_loop()
             except RuntimeError:
                 # If there's no running loop
                 loop = asyncio.new_event_loop()
                 asyncio.set_event_loop(loop)
             
             # Run download in executor
             df = await loop.run_in_executor(None, download)
        except Exception as e:
             logger.error(f"[Yahoo] Download failed for {symbol} after retries: {e}")
             df = None # Ensure df is None on failure

        if df is not None and not df.empty:
             logger.info(f"[Yahoo] Direct download successful for {symbol}, got {len(df)} rows")
             # --- Cache Update ---
             data_download_cache[cache_key] = df.copy() # Store a copy in cache
             # --- End Cache Update ---
        elif df is not None and df.empty:
             logger.warning(f"[Yahoo] Download returned empty DataFrame for {symbol}")
             # Cache the empty result too, to avoid repeated failed attempts for a short period
             data_download_cache[cache_key] = df.copy()
        else:
             logger.warning(f"[Yahoo] Download returned None for {symbol}")
             # Optionally cache None or handle differently if needed

        return df
    
    @staticmethod
    def _validate_and_clean_data(df: pd.DataFrame, instrument: str = None) -> pd.DataFrame:
        """
        Validate and clean the market data
        
        Args:
            df: DataFrame with market data
            instrument: The instrument symbol to determine appropriate decimal precision
        """
        if df is None or df.empty:
            logger.warning(f"[Yahoo] Download for {symbol} returned empty DataFrame")
            market_data_cache[cache_key] = None
            return None, None
                    
        # Log success and data shape before validation
        logger.info(f"[Yahoo] Successfully downloaded data for {symbol} with shape {df.shape}")
                
        # Validate and clean the data - basic validation
        try:
            # Check for MultiIndex kolommen (gebruikelijk bij Yahoo Finance)
            if isinstance(df.columns, pd.MultiIndex):
                logger.info(f"[Yahoo] Converting MultiIndex columns to standard format")
                # Maak een nieuwe DataFrame met de juiste kolomnamen
                standard_df = pd.DataFrame()
                # Pak de eerste level 1 waarde (meestal de ticker symbol)
                ticker_col = df.columns.get_level_values(1)[0]
                
                # Kopieer de belangrijke kolommen
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if (col, ticker_col) in df.columns:
                        standard_df[col] = df[(col, ticker_col)]
                        
                if not standard_df.empty:
                    df = standard_df
                else:
                    logger.error(f"[Yahoo] Could not convert MultiIndex, columns: {df.columns}")
                    market_data_cache[cache_key] = None
                    return None, None
                    
            # Zorg dat de vereiste kolommen aanwezig zijn
            required_cols = ['Open', 'High', 'Low', 'Close']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"[Yahoo] Missing required columns for {symbol}. Available: {df.columns.tolist()}")
                market_data_cache[cache_key] = None
                return None, None
                
            # Verwijder duplicate indices en NaN waarden
            if df.index.duplicated().sum() > 0:
                df = df[~df.index.duplicated(keep='last')]
                
            # Verwijder NaN waarden
            df = df.dropna()
            
            logger.info(f"[Yahoo] Validation successful, shape after cleaning: {df.shape}")
        except Exception as validate_e:
            logger.error(f"[Yahoo] Error validating data: {validate_e}")
            logger.error(traceback.format_exc())
            market_data_cache[cache_key] = None
            return None, None

    @staticmethod
    def _map_timeframe_to_yfinance(timeframe: str) -> Optional[str]:
        """Maps internal timeframe notation (e.g., H1, M15) to yfinance interval."""
        mapping = {
            # Intraday
            'M1': '1m',
            'M5': '5m',
            'M15': '15m',
            'M30': '30m',
            'H1': '1h', # Correct mapping for 1 hour
            'H4': '4h', # yfinance does not support 4h, needs resampling from 1h or 1d?
                       # For now, let's try requesting 1h and maybe resample later if needed
                       # Or fallback to 1d if 1h fails?
                       # Simplest for now: map H4 to 1d, but log warning
            # Daily/Weekly/Monthly
            'D1': '1d',
            'W1': '1wk',
            'MN1': '1mo'
        }
        # Case-insensitive matching
        input_tf_upper = timeframe.upper()

        if input_tf_upper == 'H4':
             logger.warning("yfinance does not directly support 4h interval. Mapping H4 to '1d'. Consider resampling 1h data if needed.")
             return '1d' # Fallback H4 to 1d for now

        # Try direct mapping first (e.g., '1h' -> '1h')
        if timeframe in mapping.values():
             return timeframe

        # Try mapping from internal notation (e.g., 'H1' -> '1h')
        yf_interval = mapping.get(input_tf_upper)

        if not yf_interval:
            logger.warning(f"Unsupported timeframe '{timeframe}' for Yahoo Finance. Defaulting to '1d'.")
            return '1d' # Default to daily if mapping fails
        
        # Additional check for valid intraday intervals (max 60 days history)
        # yfinance intraday intervals < 1d are limited to 60 days of history
        # Longer intervals like 1h might be limited to 730 days.
        # We might need to adjust start_date based on interval.
        return yf_interval

    @staticmethod
    def _calculate_period_for_interval(interval: str, limit: int) -> str:
         """Calculate appropriate yfinance period based on interval and limit."""
         # Simple estimation: try to get slightly more than needed
         days_needed = 1 # Default
         try:
             if 'm' in interval or 'h' in interval:
                 minutes = 0
                 if 'm' in interval:
                     minutes = int(interval.replace('m', ''))
                 elif 'h' in interval:
                     minutes = int(interval.replace('h', '')) * 60
                 
                 if minutes > 0:
                     # Estimate days needed, adding buffer
                     days_needed = max(1, int((minutes * limit / (24 * 60)) * 1.5) + 2) # Buffer added
                 else: # Default for 1h or invalid intraday
                      days_needed = max(10, int(limit / 24 * 1.5) + 2) # e.g. 300/24 * 1.5 + 2 = ~21 days for 1h

                 # Intraday data limits for yfinance
                 # 1m: max 7 days
                 # <1h: max 60 days
                 # 1h: max 730 days
                 if interval == '1m': days_needed = min(days_needed, 7)
                 elif 'm' in interval: days_needed = min(days_needed, 60)
                 elif interval == '1h': days_needed = min(days_needed, 729) # Limit to 729 days for 1h

             elif 'd' in interval:
                  days_needed = int(limit * 1.5) + 5 # Need ~455 days for 300 limit
             elif 'wk' in interval:
                  days_needed = int(limit * 7 * 1.5) + 30
             elif 'mo' in interval:
                  days_needed = int(limit * 30 * 1.5) + 90
             else: # Default case
                  days_needed = 60

         except Exception as e:
             logger.warning(f"Error calculating days needed for interval {interval}: {e}. Defaulting to 60d.")
             days_needed = 60

         # Ensure minimum period
         days_needed = max(days_needed, 2)
         # Return integer days for start/end calculation, or string period for yfinance period param
         # Correction: Let's return the yfinance compatible period string directly
         if 'm' in interval or 'h' in interval:
              period_str = f"{days_needed}d"
              logger.info(f"Calculated yfinance download period '{period_str}' for interval '{interval}' and limit {limit}")
              return period_str
         else: # For daily or longer, just return the number of days needed for start_date calculation
              logger.info(f"Calculated days needed {days_needed} for interval '{interval}' and limit {limit} (for start_date)")
              return str(days_needed) # Return as string for consistency before ValueError check

    @staticmethod
    async def get_market_data(symbol: str, limit: int = 100) -> Optional[Tuple[pd.DataFrame, Dict]]:
        """
        Fetches market data from Yahoo Finance for a FIXED timeframe (H1), validates it, and calculates indicators.
        Returns a tuple: (DataFrame with indicators, analysis_info dictionary)
        """
        # Verifieer systeemdatum, maar gebruik gewoon vaste periodes
        try:
            logger.info(f"[Yahoo] Fetching market data for {symbol} (limit: {limit})")
        except Exception as date_check_e:
            logger.error(f"[Yahoo] Error in initial check: {date_check_e}")
            
        # <<< FIXED TIMEFRAME >>>
        fixed_timeframe = "H1" 
        # <<< END FIXED TIMEFRAME >>>

        # Genereer een cache key die niet afhankelijk is van datum
        # We gebruiken alleen symbol, timeframe en limit (geen datums)
        cache_key = (symbol, fixed_timeframe, limit)
 
        if cache_key in market_data_cache:
            logger.info(f"[Yahoo Cache] HIT for market data: {symbol} timeframe {fixed_timeframe} limit {limit}")
            cached_result = market_data_cache[cache_key]
            # Check if the cached result is None or a tuple before unpacking
            if cached_result is None:
                logger.warning(f"[Yahoo Cache] Cached value was None for {symbol}")
                return None, None
            # Ensure we have a valid tuple
            if isinstance(cached_result, tuple) and len(cached_result) == 2:
                cached_df, cached_info = cached_result
                return cached_df.copy(), cached_info.copy() # Return copies
            else:
                logger.warning(f"[Yahoo Cache] Invalid cached format for {symbol}, expected tuple, got {type(cached_result)}")
                # Remove invalid format from cache
                del market_data_cache[cache_key]
                # Continue with fetching new data
        logger.info(f"[Yahoo Cache] MISS for market data: {symbol} timeframe {fixed_timeframe} limit {limit}")

        logger.info(f"[Yahoo] Getting market data for {symbol} on fixed {fixed_timeframe} timeframe") # Log fixed timeframe
        df = None
        analysis_info = {}

        try:
            # 1. Format symbol and map timeframe
            formatted_symbol = YahooFinanceProvider._format_symbol(symbol, is_crypto=False, is_commodity=False) # Assume not crypto/commodity unless detected
            yf_interval = YahooFinanceProvider._map_timeframe_to_yfinance(fixed_timeframe) # Use fixed_timeframe

            if not yf_interval:
                 logger.error(f"[Yahoo] Could not map fixed timeframe '{fixed_timeframe}' to yfinance interval.")
                 return None, None

            # 2. ALLEEN period gebruiken, geen datums (veel betrouwbaarder)
            # Bereken de juiste periode voor het interval
            if yf_interval == '1h':
                # Voor 1h data, gebruik 7d (ongeveer 168 uur aan data)
                yf_period = '7d'
                logger.info(f"[Yahoo] Using fixed period '{yf_period}' for interval '{yf_interval}'")
            elif yf_interval == '1d':
                # Voor dagelijkse data, gebruik 6mo (ongeveer 180 dagen)
                yf_period = '6mo'
                logger.info(f"[Yahoo] Using fixed period '{yf_period}' for interval '{yf_interval}'")
            else:
                # Fallback
                yf_period = '1mo'
                logger.info(f"[Yahoo] Using fallback period '{yf_period}' for interval '{yf_interval}'")

            # Wait for rate limit
            await YahooFinanceProvider._wait_for_rate_limit()
            
            try:
                # Download the data from Yahoo Finance using the cached downloader
                # ALLEEN PERIOD meegeven, geen start_date/end_date!
                df = await YahooFinanceProvider._download_data(
                    formatted_symbol,
                    interval=yf_interval,
                    period=yf_period,
                    timeout=30,
                    original_symbol=symbol
                )
                
                if df is None or df.empty:
                    logger.warning(f"[Yahoo] No data returned for {symbol} ({formatted_symbol}) after download attempt.")
                    market_data_cache[cache_key] = None # Cache None result
                    return None, None # Return tuple
                    
                # Log success and data shape before validation
                logger.info(f"[Yahoo] Successfully downloaded data for {symbol} with shape {df.shape}")
                
                # Validate and clean the data
                # Hier gebruiken we onze vereenvoudigde validatiefunctie
                try:
                    # Check for MultiIndex kolommen (gebruikelijk bij Yahoo Finance)
                    if isinstance(df.columns, pd.MultiIndex):
                        logger.info(f"[Yahoo] Converting MultiIndex columns to standard format")
                        # Maak een nieuwe DataFrame met de juiste kolomnamen
                        standard_df = pd.DataFrame()
                        # Pak de eerste level 1 waarde (meestal de ticker symbol)
                        ticker_col = df.columns.get_level_values(1)[0]
                        
                        # Kopieer de belangrijke kolommen
                        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                            if (col, ticker_col) in df.columns:
                                standard_df[col] = df[(col, ticker_col)]
                                
                        if not standard_df.empty:
                            df = standard_df
                        else:
                            logger.error(f"[Yahoo] Could not convert MultiIndex, columns: {df.columns}")
                            market_data_cache[cache_key] = None
                            return None, None
                            
                    # Zorg dat de vereiste kolommen aanwezig zijn
                    required_cols = ['Open', 'High', 'Low', 'Close']
                    if not all(col in df.columns for col in required_cols):
                        logger.error(f"[Yahoo] Missing required columns for {symbol}. Available: {df.columns.tolist()}")
                        market_data_cache[cache_key] = None
                        return None, None
                        
                    # Verwijder duplicate indices en NaN waarden
                    if df.index.duplicated().sum() > 0:
                        df = df[~df.index.duplicated(keep='last')]
                        
                    # Verwijder NaN waarden
                    df = df.dropna()
                    
                    logger.info(f"[Yahoo] Validation successful, shape after cleaning: {df.shape}")
                except Exception as validate_e:
                    logger.error(f"[Yahoo] Error validating data: {validate_e}")
                    logger.error(traceback.format_exc())
                    market_data_cache[cache_key] = None
                    return None, None

                # For 4h timeframe, resample from 1h
                if fixed_timeframe == "4h" and yf_interval == "1h": # Ensure we fetched 1h data
                    logger.info(f"[Yahoo] Resampling 1h data to 4h for {symbol}")
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
                        df_resampled.dropna(inplace=True) # Drop rows where any value is NaN (often first row after resample)
                        
                        if df_resampled.empty:
                             logger.warning(f"[Yahoo] Resampling to 4h resulted in empty DataFrame for {symbol}. Using 1h data instead.")
                             # Stick with df (1h) if resampling fails
                        else:
                             df = df_resampled # Use the resampled data
                             logger.info(f"[Yahoo] Successfully resampled to 4h with shape {df.shape}")
                             
                    except Exception as resample_e:
                        logger.error(f"[Yahoo] Error resampling to 4h: {str(resample_e)}")
                        # Continue with 1h data (df) if resampling fails
                
                # Ensure we have enough data *before* limiting for indicators
                if len(df) < limit: # Check if we have enough historical data
                     logger.warning(f"[Yahoo] Insufficient data after cleaning/resampling for {symbol} (got {len(df)}, needed ~{limit}). Indicators might be inaccurate.")
                     # Potentially return None or handle differently if strict data requirement
                     # For now, proceed but log warning.

                # --- Calculate indicators BEFORE limiting ---
                df_with_indicators = df.copy() # Work on a copy
                indicators = {}
                
                try:
                    # Ensure required columns exist
                    required_cols = ['Open', 'High', 'Low', 'Close']
                    if not all(col in df_with_indicators.columns for col in required_cols):
                         logger.error(f"[Yahoo] Missing required columns {required_cols} for indicator calculation in {symbol}. Skipping indicators.")
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

                         # 4. Calculate Technical Indicators using plain pandas
                         logger.info(f"[Yahoo] Calculating indicators for {symbol} manually using pandas")
                         try:
                             close = df_with_indicators['Close'] # Use validated column name

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
                             df_with_indicators['MACDh_12_26_9'] = df_with_indicators['MACD_12_26_9'] - df_with_indicators['MACDs_12_26_9'] # Histogram
                             logger.info(f"[Yahoo] Indicators calculated. DataFrame columns: {df_with_indicators.columns.tolist()}")

                         except Exception as ta_error:
                             logger.error(f"[Yahoo] Error calculating technical indicators for {symbol}: {ta_error}", exc_info=True)
                             # Return dataframe without indicators if calculation fails

                except Exception as indicator_e:
                     logger.error(f"[Yahoo] Error calculating indicators for {symbol}: {indicator_e}")
                     # Continue without indicators or with partial indicators if possible
                # --- End Indicator Calculation ---

                # --- Limit the number of candles AFTER calculations ---
                df_limited = df_with_indicators.iloc[-limit:]
                
                # --- Prepare result and cache --- 
                # Return the limited DataFrame AND the separate indicators dict
                result_df = df_limited.copy()
                # REMOVED: result_df.indicators = indicators # Avoid UserWarning by not setting attribute (FIXED WARNING)
                
                # Cache the final result tuple (DataFrame, indicators_dict)
                market_data_cache[cache_key] = (result_df.copy(), indicators.copy()) # Cache copies
                # Log the shape being returned
                logger.info(f"[Yahoo] Returning market data for {symbol} with shape {result_df.shape}")

                return result_df, indicators # Return tuple
                
            except Exception as download_e:
                logger.error(f"[Yahoo] Error processing market data for {symbol}: {str(download_e)}")
                if isinstance(download_e, KeyError) and 'Open' in str(download_e):
                     logger.error(f"[Yahoo] Likely issue with column names after download for {symbol}. Raw columns: {df.columns if 'df' in locals() and df is not None else 'N/A'}")
                
                # Special handling for 429 errors
                if "429" in str(download_e) or "too many requests" in str(download_e).lower():
                    logger.warning(f"[Yahoo] 429 Too Many Requests detected in get_market_data. Adding to 429 counter.")
                    YahooFinanceProvider._429_count += 1
                    YahooFinanceProvider._429_last_time = time.time()
                    # Add a longer delay for main flow 429s to allow rate limiting to recover
                    await asyncio.sleep(YahooFinanceProvider._429_backoff_time * min(YahooFinanceProvider._429_count, 3))
                
                logger.error(traceback.format_exc()) # Log full traceback for download errors
                market_data_cache[cache_key] = None # Cache None result on error
                raise download_e
                
        except Exception as e:
            logger.error(f"[Yahoo] Unexpected error in get_market_data for {symbol}: {str(e)}")
            logger.error(traceback.format_exc()) # Log full traceback for unexpected errors
            # Ensure None is cached on unexpected error before returning
            market_data_cache[cache_key] = None 
            return None, None # Return tuple

    @staticmethod
    async def get_stock_info(symbol: str) -> Optional[Dict]:
        """Get detailed information about a stock"""
        try:
            formatted_symbol = YahooFinanceProvider._format_symbol(symbol)
            
            # Wait for rate limit
            await YahooFinanceProvider._wait_for_rate_limit()
            
            # Get loop more carefully
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # If there's no running loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Get stock info with yfinance
            def get_info():
                try:
                    ticker = yf.Ticker(formatted_symbol)
                    info = ticker.info
                    return info
                except Exception as e:
                    logger.error(f"Error getting stock info: {str(e)}")
                    raise e
            
            info = await loop.run_in_executor(None, get_info)
            return info
            
        except Exception as e:
            logger.error(f"Error getting stock info from Yahoo Finance: {str(e)}")
            return None
    
    @staticmethod
    def _get_instrument_precision(instrument: str) -> int:
        """Get the appropriate decimal precision for an instrument
        
        Args:
            instrument: The trading instrument symbol
            
        Returns:
            int: Number of decimal places to use
        """
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
        """Format the instrument for Yahoo Finance API"""
        # Special cases
        if instrument == "EURUSD":
            return "EURUSD=X"
        elif instrument == "GBPUSD":
            return "GBPUSD=X"
        elif instrument == "USDJPY":
            return "USDJPY=X"
        elif instrument == "XAUUSD":  # Gold
            return "GC=F"
        elif instrument == "XTIUSD" or instrument == "USOIL":  # Oil
            return "CL=F"
        elif instrument == "XBRUSD":  # Brent oil
            return "BZ=F"
        elif instrument == "XAGUSD":  # Silver
            return "SI=F"
        elif instrument == "US30":
            return "^DJI"  # Dow Jones
        elif instrument == "US500":
            return "^GSPC"  # S&P 500
        elif instrument == "US100":
            return "^NDX"  # Nasdaq 100
        elif instrument == "DE40":
            return "^GDAXI"  # DAX
        elif instrument == "UK100":
            return "^FTSE"  # FTSE 100
        elif instrument == "JP225":
            return "^N225"  # Nikkei 225
        elif instrument == "AU200":
            return "^AXJO"  # ASX 200
        elif instrument == "BTCUSD":
            return "BTC-USD"
        elif instrument == "ETHUSD":
            return "ETH-USD"
        
        # Add suffix for forex pairs
        if len(instrument) == 6 and not is_crypto and not is_commodity:
            return f"{instrument}=X"
        
        # Default: return as is
        return instrument 

    @staticmethod
    def get_chart(instrument: str, timeframe: str = "1h", fullscreen: bool = False) -> Optional[bytes]:
        """Generate a chart for the given instrument using matplotlib.
        
        Args:
            instrument: The instrument symbol (e.g., EURUSD, XAUUSD)
            timeframe: The timeframe for the chart (e.g., 1h, 4h, 1d)
            fullscreen: Whether to generate a larger chart
            
        Returns:
            Bytes of the PNG image, or None if generation fails
        """
        try:
            # Map the instrument to Yahoo Finance symbol
            yahoo_symbol = YahooFinanceProvider._format_symbol(instrument, 
                                                              is_crypto=instrument.startswith('BTC') or instrument.startswith('ETH'),
                                                              is_commodity=instrument.startswith('XAU') or instrument.startswith('XTI'))
            
            # Map timeframe to Yahoo Finance interval
            interval = YahooFinanceProvider._map_timeframe_to_yfinance(timeframe)
            if not interval:
                logger.error(f"Invalid timeframe: {timeframe}")
                return None
            
            # Gebruik een vaste period parameter op basis van timeframe
            # Dit vermijdt problemen met datums
            if timeframe == "1h" or interval == "1h":
                period = "7d"  # Een week voor uurgegevens
            elif timeframe == "4h" or interval == "4h":
                period = "30d"  # Een maand voor 4-uursgegevens
            elif timeframe == "1d" or interval == "1d":
                period = "6mo"  # Zes maanden voor dagelijkse gegevens
            else:
                period = "1mo"  # Standaard een maand
                
            logger.info(f"Getting data for chart: {yahoo_symbol} with interval={interval} and period={period}")
            
            # Get the data met period in plaats van start/end date
            data = yf.download(
                yahoo_symbol,
                interval=interval,
                period=period,
                progress=False,
                actions=False,
                auto_adjust=True,
                threads=False
            )
            
            if data.empty:
                logger.error(f"No data returned for {yahoo_symbol}")
                return None
                
            # Limit to the last 'num_candles' for better visualization
            data = data.tail(100)
            
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
            last_price = float(data['Close'].iloc[-1].item())  # Convert to scalar first then float
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
    def _get_reliable_date():
        """Get a date for reference purposes"""
        return datetime.now()
