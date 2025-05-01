import logging
import traceback
import asyncio
import os
from typing import Optional, Dict, Any
import time
import pandas as pd
from datetime import datetime, timedelta
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError, wait_fixed
import yfinance as yf
import functools
import numpy as np
import pytz
import re

logger = logging.getLogger(__name__)

# Retry strategy for general errors
retry_general = dict(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)

# Specific retry strategy for rate limit errors
retry_rate_limit = dict(
    stop=stop_after_attempt(4),  # Allow more attempts for rate limits
    wait=wait_fixed(30),  # Use wait_fixed instead of wait_base
    reraise=True
)

# Function to check if an exception is a rate limit error
def is_rate_limit_error(exception):
    """Check if an exception is a rate limit error based on its message"""
    error_msg = str(exception).lower()
    rate_limit_phrases = [
        "rate limit", 
        "too many requests", 
        "429", 
        "rate limited", 
        "try after a while"
    ]
    return any(phrase in error_msg for phrase in rate_limit_phrases)

# List of user agents to rotate
user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0'
]

class YahooFinanceProvider:
    """Provider class for Yahoo Finance API integration with simplified error handling"""
    
    # Keep track of last API call time to manage rate limiting
    _last_api_call = 0
    _min_delay_between_calls = 2  # Start with 2 seconds minimum delay
    _session = None
    
    # Add rate limit tracking
    _rate_limit_hits = 0
    _rate_limit_last_hit = 0
    _rate_limit_backoff = 5  # Start with 5 seconds, will increase on consecutive hits
    
    # List of user agents to rotate
    _user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
        'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1'
    ]
    
    # Add domains that may need to be whitelisted
    _yahoo_domains = [
        "fc.yahoo.com",
        "query2.finance.yahoo.com", 
        "query1.finance.yahoo.com"
    ]

    @staticmethod
    def _get_session():
        """Get or create a requests session with retry logic"""
        if YahooFinanceProvider._session is None:
            session = requests.Session()
            
            # Configure retry strategy
            retries = Retry(
                total=3, 
                backoff_factor=1.0,
                status_forcelist=[429, 500, 502, 503, 504]
            )
            adapter = HTTPAdapter(max_retries=retries)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            # Use a random user agent
            user_agent = random.choice(YahooFinanceProvider._user_agents)
            
            session.headers.update({
                'User-Agent': user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Cache-Control': 'max-age=0',
                'Connection': 'keep-alive'
            })
            
            # Try to get initial cookies
            try:
                session.get('https://finance.yahoo.com/', timeout=10)
            except Exception as e:
                logger.warning(f"[Yahoo] Failed to get initial cookies: {str(e)}")
            
            YahooFinanceProvider._session = session
        
        # Rotate user agents on each call
        YahooFinanceProvider._session.headers.update({
            'User-Agent': random.choice(YahooFinanceProvider._user_agents)
        })
        
        return YahooFinanceProvider._session

    @staticmethod
    def _wait_for_rate_limit():
        """Ensures a minimum delay between consecutive API calls."""
        now = time.time()
        time_since_last_call = now - YahooFinanceProvider._last_api_call
        
        # Check if we've had recent rate limit hits
        time_since_rate_limit = now - YahooFinanceProvider._rate_limit_last_hit
        
        # Determine the wait time based on rate limit history
        wait_time = YahooFinanceProvider._min_delay_between_calls
        
        # If we've had rate limit hits recently, use the more aggressive backoff
        if YahooFinanceProvider._rate_limit_hits > 0 and time_since_rate_limit < 300:  # Within last 5 minutes
            wait_time = max(wait_time, YahooFinanceProvider._rate_limit_backoff)
            logger.info(f"[Yahoo] Using rate limit backoff: {wait_time:.2f} seconds (hits: {YahooFinanceProvider._rate_limit_hits})")
            
            # Consider resetting the session if we're using aggressive backoff
            YahooFinanceProvider._reset_session_if_needed()
        
        # If we need to wait, do so
        if time_since_last_call < wait_time:
            actual_wait = wait_time - time_since_last_call
            logger.info(f"[Yahoo] Rate limit: waiting for {actual_wait:.2f} seconds before next call.")
            time.sleep(actual_wait)
        
        # Add some random jitter to avoid synchronized requests
        jitter = random.uniform(0.5, 2.0)
        time.sleep(jitter)
        
        # Update last call time
        YahooFinanceProvider._last_api_call = time.time()
    
    @staticmethod
    def _ensure_valid_dates(start_date, end_date):
        """Ensure dates are valid (in the past) and properly formatted"""
        # Get current date
        current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
        
        # Ensure end_date is not in the future
        if end_date >= current_date:
            end_date = current_date - timedelta(days=1)  # Yesterday
        
        # Ensure start_date is not in the future and before end_date
        if start_date >= current_date or start_date >= end_date:
            start_date = end_date - timedelta(days=30)  # 30 days before end_date
        
        return start_date, end_date

    @staticmethod
    def _format_symbol(symbol: str) -> str:
        """Format symbol for Yahoo Finance API"""
        symbol = symbol.upper()
        
        # Handle commodities
        commodities_map = {
            "XAUUSD": "GC=F",   # Gold futures
            "XAGUSD": "SI=F",   # Silver futures
            "XTIUSD": "CL=F",   # WTI Crude futures (primary)
            "WTIUSD": "CL=F",   # WTI Crude futures (alias)
            "USOIL": "CL=F",    # WTI Crude futures (alias)
            "XBRUSD": "BZ=F",   # Brent Crude oil futures
            "XPDUSD": "PA=F",   # Palladium futures
            "XPTUSD": "PL=F",   # Platinum futures
            "NATGAS": "NG=F",   # Natural Gas futures
            "COPPER": "HG=F"    # Copper futures
        }
        
        if symbol in commodities_map:
            return commodities_map[symbol]
        
        # Handle indices
        indices_map = {
            "US30": "^DJI",      # Dow Jones
            "US500": "^GSPC",    # S&P 500
            "US100": "^NDX",     # Nasdaq 100
            "UK100": "^FTSE",    # FTSE 100
            "DE40": "^GDAXI",    # DAX
            "JP225": "^N225",    # Nikkei 225
            "AU200": "^AXJO",    # ASX 200
            "EU50": "^STOXX50E", # Euro Stoxx 50
            "FR40": "^FCHI",     # CAC 40
            "HK50": "^HSI"       # Hang Seng
        }
        
        if symbol in indices_map:
            return indices_map[symbol]
        
        # Handle forex pairs (e.g., EURUSD -> EUR=X)
        if len(symbol) == 6 and all(c.isalpha() for c in symbol):
            base = symbol[:3]
            quote = symbol[3:]
            return f"{base}{quote}=X"
        
        return symbol

    @staticmethod
    def _validate_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Perform basic validation and cleaning of market data"""
        if df is None or df.empty:
            return df
        
        try:
            # Remove any duplicate indices
            if df.index.duplicated().sum() > 0:
                df = df[~df.index.duplicated(keep='last')]
            
            # Forward fill a small number of missing values
            df = df.ffill(limit=2)
            
            # Remove rows with remaining NaN values
            df = df.dropna()
            
            # Validate price relationships
            valid_rows = (
                (df['High'] >= df['Low']) & 
                (df['High'] >= df['Open']) & 
                (df['High'] >= df['Close']) &
                (df['Low'] <= df['Open']) & 
                (df['Low'] <= df['Close'])
            )
            
            df = df[valid_rows]
            
            # Validate Volume if it exists
            if 'Volume' in df.columns:
                df = df[df['Volume'] >= 0]
            
            return df
        
        except Exception as e:
            logger.error(f"[Yahoo] Error in data validation: {str(e)}")
            return df
    
    @staticmethod
    def _try_download_with_ticker(ticker_symbol, start_date, end_date, interval, timeout=30):
        """Try to download data using the Ticker approach"""
        logger.info(f"[Yahoo] Trying ticker.history for {ticker_symbol}")
        
        try:
            session = YahooFinanceProvider._get_session()
            ticker = yf.Ticker(ticker_symbol, session=session)
            
            df = ticker.history(
                start=start_date.date(),
                end=end_date.date(),
                interval=interval,
                prepost=False,
                auto_adjust=True,
                rounding=True
            )
            
            if df is not None and not df.empty:
                logger.info(f"[Yahoo] Ticker.history successful for {ticker_symbol}, got {len(df)} rows")
                # Reset consecutive rate limit count on success
                if YahooFinanceProvider._rate_limit_hits > 0:
                    YahooFinanceProvider._rate_limit_hits = max(0, YahooFinanceProvider._rate_limit_hits - 1)
                return df
            else:
                logger.warning(f"[Yahoo] Ticker.history returned empty DataFrame for {ticker_symbol}")
                return None
                
        except Exception as e:
            error_msg = str(e).lower()
            logger.error(f"[Yahoo] Ticker.history exception for {ticker_symbol}: {str(e)}")
            
            # Check for connectivity issues due to ad blockers or firewalls
            if YahooFinanceProvider._is_connectivity_error(e):
                domains_list = ", ".join(YahooFinanceProvider._yahoo_domains)
                logger.error(f"[Yahoo] Connection error detected! This may be due to ad-blocking software or firewall. "
                             f"Try whitelisting these domains: {domains_list}")
                time.sleep(2)  # Brief pause
            # Check for rate limit errors
            elif YahooFinanceProvider._is_rate_limit_error(e):
                YahooFinanceProvider._handle_rate_limit_error()
                logger.error(f"[Yahoo] Rate limit detected during ticker.history for {ticker_symbol}")
                
                # Add an additional sleep on rate limit error
                time.sleep(5)
            
            return None
    
    @staticmethod
    def _try_download_with_download(ticker_symbol, start_date, end_date, interval, timeout=30):
        """Try to download data using the yf.download approach"""
        logger.info(f"[Yahoo] Trying yf.download for {ticker_symbol}")
        
        try:
            session = YahooFinanceProvider._get_session()
            
            df = yf.download(
                tickers=ticker_symbol,
                start=start_date.date(),
                end=end_date.date(),
                interval=interval,
                progress=False,
                session=session,
                timeout=timeout,
                ignore_tz=True,
                prepost=False,
                threads=False,
                rounding=True
            )
            
            if df is not None and not df.empty:
                logger.info(f"[Yahoo] yf.download successful for {ticker_symbol}, got {len(df)} rows")
                # Reset consecutive rate limit count on success
                if YahooFinanceProvider._rate_limit_hits > 0:
                    YahooFinanceProvider._rate_limit_hits = max(0, YahooFinanceProvider._rate_limit_hits - 1)
                return df
            else:
                logger.warning(f"[Yahoo] yf.download returned empty DataFrame for {ticker_symbol}")
                return None
                
        except Exception as e:
            error_msg = str(e).lower()
            logger.error(f"[Yahoo] yf.download exception for {ticker_symbol}: {str(e)}")
            
            # Check for connectivity issues due to ad blockers or firewalls
            if YahooFinanceProvider._is_connectivity_error(e):
                domains_list = ", ".join(YahooFinanceProvider._yahoo_domains)
                logger.error(f"[Yahoo] Connection error detected! This may be due to ad-blocking software or firewall. "
                             f"Try whitelisting these domains: {domains_list}")
                time.sleep(2)  # Brief pause
            # Check for rate limit errors
            elif YahooFinanceProvider._is_rate_limit_error(e):
                YahooFinanceProvider._handle_rate_limit_error()
                logger.error(f"[Yahoo] Rate limit detected during yf.download for {ticker_symbol}")
                
                # Add an additional sleep on rate limit error
                time.sleep(5)
                
            return None

    @staticmethod
    def _download_market_data(symbol, formatted_symbol, timeframe, start_date, end_date, interval):
        """Download market data with multiple attempts and methods"""
        # Wait for rate limits
        YahooFinanceProvider._wait_for_rate_limit()
        
        # First attempt: try yf.download
        df = YahooFinanceProvider._try_download_with_download(
            formatted_symbol, start_date, end_date, interval, timeout=45
        )
        
        # If download failed, wait and try Ticker approach
        if df is None or df.empty:
            # Add delay before second attempt
            delay = 3
            if YahooFinanceProvider._rate_limit_hits > 0:
                # Use longer delay if we've had rate limit issues
                delay = min(10 * YahooFinanceProvider._rate_limit_hits, 60)
                logger.info(f"[Yahoo] Using longer delay ({delay}s) before retry due to rate limiting")
                
            time.sleep(delay)
            YahooFinanceProvider._wait_for_rate_limit()
            
            df = YahooFinanceProvider._try_download_with_ticker(
                formatted_symbol, start_date, end_date, interval
            )
        
        # For commodities, we might need to try alternative symbols
        if (df is None or df.empty) and symbol in ["USOIL", "XTIUSD", "WTIUSD"]:
            # Try alternative symbols for oil
            alternatives = ["CL=F", "USO", "BNO"]
            
            for alt in alternatives:
                if alt == formatted_symbol:
                    continue  # Skip if it's the same as what we already tried
                
                logger.info(f"[Yahoo] Trying alternative symbol {alt} for {symbol}")
                
                # Add delay before attempt with alternative
                delay = 5
                if YahooFinanceProvider._rate_limit_hits > 0:
                    # Use longer delay if we've had rate limit issues
                    delay = min(15 * YahooFinanceProvider._rate_limit_hits, 120)
                    logger.info(f"[Yahoo] Using longer delay ({delay}s) before alternative symbol due to rate limiting")
                    
                time.sleep(delay)
                YahooFinanceProvider._wait_for_rate_limit()
                
                df = YahooFinanceProvider._try_download_with_download(
                    alt, start_date, end_date, interval, timeout=30
                )
                
                if df is not None and not df.empty:
                    logger.info(f"[Yahoo] Successfully got data using alternative symbol {alt}")
                    break
            
            return df
            
    @classmethod
    def get_market_data(cls, symbol, timeframe, limit=None, **kwargs):
        """
        Get market data for the specified symbol, timeframe and limit
        """
        try:
            # Wait if needed due to rate limiting
            YahooFinanceProvider._wait_for_rate_limit()
            
            # Map the timeframe to the Yahoo interval
            interval = YahooFinanceProvider._map_timeframe_to_interval(timeframe)
            
            # Get the Yahoo symbol
            yahoo_symbol = YahooFinanceProvider._map_symbol_to_yahoo(symbol)
            
            # Calculate the start and end dates based on the requested timeframe
            end_date = datetime.now()
            
            if limit:
                # Calculate start date based on timeframe and requested limit
                start_date = YahooFinanceProvider._calculate_start_date(end_date, timeframe, limit)
            else:
                # Use a default start date if no limit is provided
                start_date = end_date - timedelta(days=365)
            
            # Try to download the data using both methods
            df = None
            connectivity_error = False
            
            # Try with Ticker.history first
            df = YahooFinanceProvider._try_download_with_ticker(yahoo_symbol, start_date, end_date, interval)
            
            # If that failed, try with yf.download
            if df is None or df.empty:
                df = YahooFinanceProvider._try_download_with_download(yahoo_symbol, start_date, end_date, interval)
            
            # If still no data, log error and return None
            if df is None or df.empty:
                logger.error(f"[Yahoo] Failed to get data for {yahoo_symbol} with timeframe {timeframe}")
                return None
            
            # Process the data for our needs
            processed_df = YahooFinanceProvider._process_dataframe(df, yahoo_symbol)
            
            # Ensure the limit is applied
            if limit and len(processed_df) > limit:
                processed_df = processed_df.tail(limit)
            
            # Extract indicators at the same time
            indicators = None
            if not processed_df.empty:
                indicators = YahooFinanceProvider._extract_indicators_from_dataframe(processed_df)
            
            return processed_df, indicators
            
        except Exception as e:
            # Handle connectivity errors specifically
            if YahooFinanceProvider._is_connectivity_error(e):
                domains_list = ", ".join(YahooFinanceProvider._yahoo_domains)
                error_message = (
                    f"[Yahoo] Connection error detected when trying to fetch data for {symbol}. "
                    f"This is often caused by ad-blocking software or firewalls blocking Yahoo Finance domains. "
                    f"Please try whitelisting these domains in your ad blocker, hosts file, or firewall: "
                    f"{domains_list}"
                )
                logger.error(error_message)
                # Return a special error that the UI can display to help users
                return None, {"error": "connectivity", "message": error_message}
            # Handle rate limiting specifically
            elif YahooFinanceProvider._is_rate_limit_error(e):
                YahooFinanceProvider._handle_rate_limit_error()
                logger.error(f"[Yahoo] Rate limit detected during data fetch for {symbol}")
                return None, {"error": "rate_limit"}
            else:
                logger.exception(f"[Yahoo] Error fetching market data for {symbol} with timeframe {timeframe}: {str(e)}")
                return None, {"error": "unknown", "message": str(e)}
    
    @staticmethod
    def get_stock_info(symbol: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a stock"""
        try:
            formatted_symbol = YahooFinanceProvider._format_symbol(symbol)
            
            # Wait for rate limit
            YahooFinanceProvider._wait_for_rate_limit()
            
            try:
                ticker = yf.Ticker(formatted_symbol)
                info = ticker.info
                return info
            except Exception as e:
                logger.error(f"[Yahoo] Error getting stock info: {str(e)}")
                
                # Check for connectivity issues
                if YahooFinanceProvider._is_connectivity_error(e):
                    domains_list = ", ".join(YahooFinanceProvider._yahoo_domains)
                    logger.error(f"[Yahoo] Connection error detected when getting stock info. "
                                 f"This may be due to ad-blocking software or firewall. "
                                 f"Try whitelisting these domains: {domains_list}")
                # Check for rate limit errors
                elif YahooFinanceProvider._is_rate_limit_error(e):
                    YahooFinanceProvider._handle_rate_limit_error()
                    logger.error(f"[Yahoo] Rate limit detected during get_stock_info for {symbol}")
                
                raise
            
        except Exception as e:
            logger.error(f"[Yahoo] Error getting stock info: {str(e)}")
            return None

    @staticmethod
    def _handle_rate_limit_error():
        """Update rate limit tracking when a 429 error is detected"""
        now = time.time()
        YahooFinanceProvider._rate_limit_last_hit = now
        YahooFinanceProvider._rate_limit_hits += 1
        
        # Exponentially increase backoff time based on consecutive hits
        # Cap at 5 minutes (300 seconds)
        YahooFinanceProvider._rate_limit_backoff = min(
            YahooFinanceProvider._rate_limit_backoff * 2,
            300
        )
        
        # Also increase the minimum delay between calls
        YahooFinanceProvider._min_delay_between_calls = min(
            YahooFinanceProvider._min_delay_between_calls + 1,
            10
        )
        
        logger.warning(f"[Yahoo] Rate limit hit detected. Consecutive hits: {YahooFinanceProvider._rate_limit_hits}")
        logger.warning(f"[Yahoo] New backoff time: {YahooFinanceProvider._rate_limit_backoff} seconds")
        
        # Reset session if we've hit too many rate limits
        YahooFinanceProvider._reset_session_if_needed()

    @staticmethod
    def _reset_session_if_needed():
        """Reset the session if we've hit too many rate limits"""
        if YahooFinanceProvider._rate_limit_hits >= 5:
            logger.warning("[Yahoo] Too many rate limit hits, resetting session")
            YahooFinanceProvider._session = None
            # Create a new session
            _ = YahooFinanceProvider._get_session()
            # Reset the rate limit counter partially
            YahooFinanceProvider._rate_limit_hits = max(1, YahooFinanceProvider._rate_limit_hits // 2)
            return True
        return False

    @staticmethod
    def _is_rate_limit_error(exception):
        """Check if an exception is a rate limit error based on its message"""
        error_msg = str(exception).lower()
        rate_limit_phrases = [
            "rate limit", 
            "too many requests", 
            "429", 
            "rate limited", 
            "try after a while"
        ]
        return any(phrase in error_msg for phrase in rate_limit_phrases)

    @staticmethod
    def _is_connectivity_error(exception):
        """Check if the exception is related to connectivity issues with Yahoo domains"""
        error_msg = str(exception).lower()
        
        # Check for common connection errors that might be due to ad blockers/firewalls
        connection_error_phrases = [
            "newconnectionerror",
            "failed to establish a new connection",
            "connectionerror",
            "proxyerror",
            "connection refused",
            "winerror 10049",
            "max retries exceeded",
            "unable to connect"
        ]
        
        # Check for Yahoo domains that might be blocked
        domains_mentioned = any(domain.lower() in error_msg for domain in YahooFinanceProvider._yahoo_domains)
        
        # Return True if both a connection error and Yahoo domain are mentioned
        return any(phrase in error_msg for phrase in connection_error_phrases) and domains_mentioned

    @staticmethod
    def _map_timeframe_to_interval(timeframe):
        """Map the trading bot timeframe to Yahoo Finance interval"""
        timeframe_mapping = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "1h",  # We'll resample 1h data to 4h
            "1d": "1d",
            "1wk": "1wk",
            "1mo": "1mo"
        }
        return timeframe_mapping.get(timeframe, "1d")  # Default to daily if unknown

    @staticmethod
    def _map_symbol_to_yahoo(symbol):
        """Map a trading bot symbol to a Yahoo Finance symbol"""
        # Handle common mappings
        symbol_mapping = {
            "XAUUSD": "GC=F",  # Gold futures
            "XAGUSD": "SI=F",  # Silver futures
            "USOIL": "CL=F",   # Crude oil futures
            "US30": "^DJI",    # Dow Jones Industrial Average
            "SPX500": "^GSPC", # S&P 500
            "NAS100": "^NDX",  # NASDAQ 100
            "VIX": "^VIX",     # Volatility Index
            "US500": "^GSPC",  # S&P 500 (alternative name)
            "DE30": "^GDAXI",  # DAX
            "UK100": "^FTSE",  # FTSE 100
            "JP225": "^N225",  # Nikkei 225
        }
        
        # Check direct mapping first
        if symbol in symbol_mapping:
            return symbol_mapping[symbol]
        
        # Handle forex pairs (e.g., EURUSD -> EURUSD=X)
        if re.match(r'^[A-Z]{6}$', symbol):
            return f"{symbol[:3]}{symbol[3:]}=X"
        
        # Handle indices that may need a caret
        if symbol.startswith("US") or symbol.startswith("EU") or symbol.startswith("UK"):
            if not symbol.startswith("^"):
                return f"^{symbol}"
        
        # Default case: return as is
        return symbol

    @staticmethod
    def _calculate_start_date(end_date, timeframe, limit):
        """Calculate the start date based on the timeframe and limit"""
        if timeframe == "1m":
            # Yahoo only provides 7 days of 1m data
            days = min(7, limit / 24 / 60)
            return end_date - timedelta(days=days + 1)
        elif timeframe == "5m":
            # Yahoo provides 60 days of 5m data
            days = min(60, limit * 5 / 24 / 60)
            return end_date - timedelta(days=days + 1)
        elif timeframe == "15m":
            # Yahoo provides 60 days of 15m data
            days = min(60, limit * 15 / 24 / 60)
            return end_date - timedelta(days=days + 1)
        elif timeframe == "30m":
            # Yahoo provides 60 days of 30m data
            days = min(60, limit * 30 / 24 / 60)
            return end_date - timedelta(days=days + 1)
        elif timeframe == "1h":
            # Yahoo provides 730 days of 1h data
            days = min(730, limit / 24)
            return end_date - timedelta(days=days + 5)  # Add some buffer
        elif timeframe == "4h":
            # For 4h data (which is resampled from 1h), we need 4x the hours
            days = min(730, limit * 4 / 24)
            return end_date - timedelta(days=days + 5)
        elif timeframe == "1d":
            # For daily data, just add days plus some buffer for indicators
            days = limit + 50  # Extra days for calculating indicators
            return end_date - timedelta(days=days)
        elif timeframe == "1wk":
            # For weekly data
            weeks = limit
            return end_date - timedelta(weeks=weeks + 10)
        elif timeframe == "1mo":
            # For monthly data
            days = limit * 30 + 60  # Approximate
            return end_date - timedelta(days=days)
        else:
            # Default fallback - 1 year
            return end_date - timedelta(days=365)

    @staticmethod
    def _process_dataframe(df, symbol):
        """Process the dataframe to ensure it's in the correct format"""
        if df is None or df.empty:
            return pd.DataFrame()
        
        try:
            # Handle potential MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                logger.info(f"[Yahoo] Processing MultiIndex DataFrame for {symbol}")
                
                # Get column level names
                level_names = df.columns.names
                logger.debug(f"[Yahoo] MultiIndex levels: {level_names}")
                
                # For data from yf.download with a single ticker, the structure is typically:
                # first level: Open, High, Low, Close, etc.
                # second level: ticker name
                
                # Create a new DataFrame with standard column names
                if len(df.columns.levels) >= 2:
                    # Get all unique column types from first level
                    col_types = df.columns.levels[0].tolist()
                    
                    # Create new flat DataFrame
                    new_df = pd.DataFrame(index=df.index)
                    
                    for col_type in col_types:
                        # Get the values for this column type (e.g., all 'Open' values)
                        if (col_type, symbol) in df.columns:
                            new_df[col_type] = df[(col_type, symbol)]
                        # Try using the first ticker if symbol specific column not found
                        elif len(df.columns.levels[1]) > 0:
                            ticker = df.columns.levels[1][0]
                            if (col_type, ticker) in df.columns:
                                new_df[col_type] = df[(col_type, ticker)]
                    
                    # Use the new DataFrame
                    df = new_df
                    logger.info(f"[Yahoo] Successfully flattened MultiIndex DataFrame to {list(df.columns)}")
            
            # Ensure standard column names
            standard_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
            actual_columns = set(df.columns)
            
            # Check if we need to rename columns (case insensitive)
            rename_map = {}
            for std_col in standard_columns:
                for actual_col in actual_columns:
                    if std_col.lower() == actual_col.lower() and std_col != actual_col:
                        rename_map[actual_col] = std_col
            
            # Apply renaming if needed
            if rename_map:
                df = df.rename(columns=rename_map)
                logger.info(f"[Yahoo] Renamed columns: {rename_map}")
            
            # Handle 4h timeframe resampling if needed
            if getattr(df, '_timeframe', None) == '4h' or getattr(df, 'interval', None) == '4h':
                logger.info(f"[Yahoo] Resampling 1h data to 4h for {symbol}")
                try:
                    # Create a dict of column mappings for aggregation
                    agg_dict = {
                        'Open': 'first',
                        'High': 'max',
                        'Low': 'min',
                        'Close': 'last'
                    }
                    if 'Volume' in df.columns:
                        agg_dict['Volume'] = 'sum'
                        
                    # Resample to 4h
                    df = df.resample('4h').agg(agg_dict)
                    df.dropna(inplace=True)
                    logger.info(f"[Yahoo] Successfully resampled to 4h with shape {df.shape}")
                except Exception as e:
                    logger.error(f"[Yahoo] Error resampling to 4h: {str(e)}")
                    # Continue with 1h data if resampling fails
            
            # Ensure the DataFrame has a proper datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                    logger.info(f"[Yahoo] Converted index to DatetimeIndex")
                except Exception as e:
                    logger.error(f"[Yahoo] Error converting index to datetime: {str(e)}")
            
            # Sort the DataFrame by date (ascending)
            df = df.sort_index()
            
            # Drop any NaN rows
            df = df.dropna()
            
            return df
        except Exception as e:
            logger.error(f"[Yahoo] Error processing DataFrame: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    @staticmethod
    def _extract_indicators_from_dataframe(df):
        """Extract indicators from the DataFrame's last row"""
        if df is None or df.empty:
            return None
        
        try:
            # Create indicators dictionary with basic OHLCV data
            indicators = {
                'open': float(df['Open'].iloc[-1]),
                'high': float(df['High'].iloc[-1]),
                'low': float(df['Low'].iloc[-1]),
                'close': float(df['Close'].iloc[-1]),
                'volume': float(df['Volume'].iloc[-1]) if 'Volume' in df.columns else 0
            }
            
            # Add EMAs if we have enough data
            if len(df) >= 20:
                df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
                indicators['EMA20'] = float(df['EMA20'].iloc[-1])
            
            if len(df) >= 50:
                df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
                indicators['EMA50'] = float(df['EMA50'].iloc[-1])
                
            if len(df) >= 200:
                df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
                indicators['EMA200'] = float(df['EMA200'].iloc[-1])
            
            # Calculate RSI
            if len(df) >= 14:
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                df['RSI'] = 100 - (100 / (1 + rs))
                indicators['RSI'] = float(df['RSI'].iloc[-1])
            
            # Calculate MACD
            if len(df) >= 26:
                df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
                df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = df['EMA12'] - df['EMA26']
                df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                indicators['MACD'] = float(df['MACD'].iloc[-1])
                indicators['MACD_signal'] = float(df['MACD_signal'].iloc[-1])
                indicators['MACD_hist'] = indicators['MACD'] - indicators['MACD_signal']
            
            return indicators
            
        except Exception as e:
            logger.error(f"[Yahoo] Error extracting indicators: {str(e)}")
            logger.error(traceback.format_exc())
            return None
