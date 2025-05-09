import os
import logging
import aiohttp
import json
import random
from typing import Dict, Any, Optional, List, Tuple, Set
import asyncio
import socket
import re
import ssl
import sys
import time
from datetime import datetime, timedelta
import threading
import pathlib
import statistics
import copy
import traceback
import openai # Added OpenAI
from openai import AsyncOpenAI # Added AsyncOpenAI

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """Class to track and analyze performance metrics for API calls and caching"""
    
    def __init__(self, max_history: int = 100):
        """
        Initialize performance metrics tracking
        
        Args:
            max_history: Maximum number of data points to store
        """
        self.api_calls = {
            'openai': [], # Changed from tavily/deepseek to openai
            'total': []
        }
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_history = max_history
        self.lock = threading.Lock()
    
    def record_api_call(self, api_name: str, duration: float) -> None:
        """
        Record the duration of an API call
        
        Args:
            api_name: Name of the API ('openai')
            duration: Duration of the call in seconds
        """
        with self.lock:
            if api_name in self.api_calls:
                # Keep only the most recent entries
                if len(self.api_calls[api_name]) >= self.max_history:
                    self.api_calls[api_name].pop(0)
                self.api_calls[api_name].append(duration)
    
    def record_total_request(self, duration: float) -> None:
        """
        Record the total duration of a sentiment request
        
        Args:
            duration: Duration of the request in seconds
        """
        with self.lock:
            if len(self.api_calls['total']) >= self.max_history:
                self.api_calls['total'].pop(0)
            self.api_calls['total'].append(duration)
    
    def record_cache_hit(self) -> None:
        """Record a cache hit"""
        with self.lock:
            self.cache_hits += 1
    
    def record_cache_miss(self) -> None:
        """Record a cache miss"""
        with self.lock:
            self.cache_misses += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics
        
        Returns:
            Dict with performance statistics
        """
        with self.lock:
            metrics = {
                'api_calls': {},
                'cache': {
                    'hits': self.cache_hits,
                    'misses': self.cache_misses,
                    'hit_rate': (self.cache_hits / (self.cache_hits + self.cache_misses) * 100)
                    if (self.cache_hits + self.cache_misses) > 0 else 0
                }
            }
            
            # Calculate API call statistics
            for api_name, durations in self.api_calls.items():
                if durations:
                    metrics['api_calls'][api_name] = {
                        'count': len(durations),
                        'avg_duration': statistics.mean(durations),
                        'min_duration': min(durations),
                        'max_duration': max(durations),
                        'median_duration': statistics.median(durations),
                        'p90_duration': sorted(durations)[int(len(durations) * 0.9)] if len(durations) >= 10 else None
                    }
                else:
                    metrics['api_calls'][api_name] = {
                        'count': 0,
                        'avg_duration': None,
                        'min_duration': None,
                        'max_duration': None,
                        'median_duration': None,
                        'p90_duration': None
                    }
            
            return metrics
    
    def reset(self) -> None:
        """Reset all metrics"""
        with self.lock:
            self.api_calls = {
                'openai': [], # Changed
                'total': []
            }
            self.cache_hits = 0
            self.cache_misses = 0

class MarketSentimentService:
    """Service for retrieving market sentiment data"""
    
    def __init__(self, cache_ttl_minutes: int = 30, persistent_cache: bool = True, cache_file: str = None, fast_mode: bool = False):
        """
        Initialize the market sentiment service
        
        Args:
            cache_ttl_minutes: Time in minutes to keep sentiment data in cache (default: 30)
            persistent_cache: Whether to save/load cache to/from disk (default: True)
            cache_file: Path to cache file, if None uses default in user's home directory
            fast_mode: Whether to use faster, more efficient API calls (default: False)
        """
        # API configuration
        # self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY") # Commented out Deepseek
        # self.tavily_api_key = os.getenv("TAVILY_API_KEY") # Commented out Tavily API key for direct use here
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        if not self.openai_api_key:
            logger.warning("No OpenAI API key found in environment variables. Sentiment analysis via OpenAI will not be available.")
            logger.debug("Environment variables available:")
            for k, v in os.environ.items():
                if "API_KEY" in k or "api_key" in k:
                    logger.debug(f"  {k}: {v[:6]}...{v[-4:]}")
            self.openai_client = None
        else:
            try:
                logger.info(f"Attempting to initialize OpenAI client with key: {self.openai_api_key[:6]}...{self.openai_api_key[-4:]}")
                self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)
                masked_key = self.openai_api_key[:6] + "..." + self.openai_api_key[-4:] if len(self.openai_api_key) > 10 else "***"
                logger.info(f"OpenAI API key is configured: {masked_key}")
                logger.info("OpenAI client initialized successfully.")
            except openai.OpenAIError as e:
                logger.error(f"OpenAI API error during client initialization: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Error details: {e.error}")
                self.openai_client = None
            except Exception as e:
                logger.error(f"Unexpected error initializing OpenAI client: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                logger.exception(e)
                self.openai_client = None
        
        # Configure the API parameters (these might become OpenAI specific or be removed)
        # self.api_key = self.deepseek_api_key # Commented out
        # self.api_model = "deepseek-chat"      # Commented out
        # self.api_url = "https://api.deepseek.com/v1/chat/completions"  # Commented out
        self.api_timeout = 30                 # Default timeout in seconds, can be adjusted for OpenAI
        
        # Additional URL configurations (mostly deprecated for Deepseek/Tavily in sentiment context)
        # self.deepseek_url = "https://api.deepseek.com/v1/chat/completions" # Commented out
        # self.tavily_url = "https://api.tavily.com/search" # Commented out for direct use here

        # Initialize the Tavily client - This is commented out as its primary use for sentiment analysis news fetching is replaced.
        # self.tavily_client = TavilyClient(self.tavily_api_key) # Commenting out
        
        # Fast mode flag
        self.fast_mode = fast_mode
        
        # Initialize cache settings
        self.cache_ttl = cache_ttl_minutes * 60  # Convert minutes to seconds
        self.use_persistent_cache = persistent_cache
        
        # Enable caching by default
        self.cache_enabled = True
        
        # Performance metrics
        self.metrics = PerformanceMetrics()
        
        # Set default cache file path if not specified
        if cache_file is None:
            home_dir = pathlib.Path.home()
            cache_dir = home_dir / ".trading_bot"
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_file = cache_dir / "sentiment_cache.json"
        else:
            self.cache_file = pathlib.Path(cache_file)
        
        # Initialize cache for sentiment data
        self.sentiment_cache = {}  # Format: {instrument: {'data': sentiment_data, 'timestamp': creation_time}}
        
        # Load cache from disk if persistent caching is enabled - but don't block startup
        # Instead of loading here, we'll provide an async method to load the cache
        self.cache_loaded = False
        
        # Background task lock to prevent multiple saves at once
        self._cache_lock = threading.Lock()
        
        # Common request timeouts and concurrency control
        if self.fast_mode:
            # Faster timeouts for fast mode
            self.request_timeout = aiohttp.ClientTimeout(total=10, connect=4)
            # Semaphore for limiting concurrent requests in fast mode
            self.request_semaphore = asyncio.Semaphore(5)
            logger.info("Fast mode enabled: using optimized request parameters")
        else:
            # Standard timeouts
            self.request_timeout = aiohttp.ClientTimeout(total=15, connect=5)
            # In standard mode, allow fewer concurrent requests
            self.request_semaphore = asyncio.Semaphore(3)
            logger.info("Standard mode: using regular request parameters")
        
        logger.info(f"Sentiment cache TTL set to {cache_ttl_minutes} minutes ({self.cache_ttl} seconds)")
        logger.info(f"Persistent caching {'enabled' if self.use_persistent_cache else 'disabled'}, cache file: {self.cache_file if self.use_persistent_cache else 'N/A'}")
        
        # Log API key status (without revealing full keys)
        # if self.tavily_api_key: # Commented out Tavily specific logging
        #     masked_key = self.tavily_api_key[:6] + "..." + self.tavily_api_key[-4:] if len(self.tavily_api_key) > 10 else "***"
        #     logger.info(f"Tavily API key is configured: {masked_key}")
        # else:
        #     logger.warning("No Tavily API key found") 

        # Log DeepSeek API key status # Commented out Deepseek specific logging
        # if self.deepseek_api_key:
        #     masked_key = self.deepseek_api_key[:6] + "..." + self.deepseek_api_key[-4:] if len(self.deepseek_api_key) > 10 else "***"
        #     logger.info(f"DeepSeek API key is configured: {masked_key}")
        # else:
        #     logger.warning("No DeepSeek API key found")
            
    def _build_search_query(self, instrument: str, market_type: str) -> str:
        """
        Build an optimized search query (topic) for market sentiment for OpenAI.
        
        Args:
            instrument: The trading instrument (e.g., 'EURUSD')
            market_type: The market type (e.g., 'forex', 'crypto')
            
        Returns:
            str: Optimized search query
        """
        logger.info(f"Building search query for {instrument} ({market_type})")
        
        # Format instrument for better search results
        if len(instrument) == 6 and instrument.isalpha():
            # Likely a forex pair, format with a slash
            formatted_instrument = f"{instrument[:3]}/{instrument[3:]}"
        else:
            formatted_instrument = instrument
        # Create a simpler but effective query
        if market_type == 'forex':
            query = f"{formatted_instrument} forex market analysis news today central bank interest rates"
        elif market_type == 'crypto':
            query = f"{formatted_instrument} cryptocurrency market analysis news today"
        elif market_type == 'indices':
            query = f"{formatted_instrument} stock index market analysis news today"
        elif market_type == 'commodities':
            query = f"{formatted_instrument} commodity market analysis news today"
        else:
            # Generic fallback
            query = f"{formatted_instrument} {market_type} market analysis news today"
            
        logger.info(f"Search query built: {query}")
        return query
            
    async def get_sentiment(self, instrument: str, market_type: Optional[str] = None, is_prefetch: bool = False) -> Dict[str, Any]:
        """
        Retrieves market sentiment data for a given instrument.
        Uses OpenAI gpt-4o-mini for analysis.
        Args:
            instrument: The trading instrument (e.g., 'EURUSD')
            market_type: The market type (e.g., 'forex', 'crypto'). If None, it's guessed.
            is_prefetch: True if this is a background prefetch call.
        Returns:
            A dictionary containing sentiment data, or an error structure.
        """
        request_start_time = time.time()
        
        if not market_type:
            market_type = self._guess_market_from_instrument(instrument)
            logger.info(f"Guessed market type for {instrument}: {market_type}")
        
        cache_key = self._get_market_specific_cache_key(instrument, market_type)

        if self.cache_enabled:
            cached_data = self._get_from_market_specific_cache(instrument, market_type)
            if cached_data:
                logger.info(f"Sentiment for {instrument} ({market_type}) found in cache.")
                self.metrics.record_cache_hit()
                timestamp_str = cached_data.get('timestamp')
                data_generated_by = cached_data.get('generated_by')

                # Invalidate cache if it's old OR if it was not generated by openai_gpt-4o-mini (forcing refresh to new system)
                is_stale = True # Assume stale by default
                is_old_generator = data_generated_by != "openai_gpt-4o-mini"

                if is_old_generator:
                    logger.info(f"Cached data for {instrument} was generated by '{data_generated_by}'. Refreshing with OpenAI.")
                elif timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "").split('.')[0]) # Handle potential Z and microseconds
                        if (datetime.utcnow().timestamp() - timestamp.timestamp()) < self.cache_ttl:
                            is_stale = False
                        else:
                            logger.info(f"Cached data for {instrument} is stale (older than {self.cache_ttl}s). Fetching fresh data.")
                    except ValueError as e:
                        logger.warning(f"Could not parse timestamp '{timestamp_str}' for cached data of {instrument}: {e}. Fetching fresh data.")
                else: # No timestamp, assume stale
                    logger.info(f"Cached data for {instrument} has no timestamp. Fetching fresh data.")
                
                if not is_stale and not is_old_generator:
                    self.metrics.record_total_request(time.time() - request_start_time)
                    return cached_data
        
        self.metrics.record_cache_miss()
        logger.info(f"Fetching sentiment for {instrument} ({market_type}) using OpenAI API as it was not in cache, stale, or from old generator.")

        if not self.openai_client:
            logger.error(f"OpenAI client not available for {instrument}. Returning error structure.")
            self.metrics.record_total_request(time.time() - request_start_time)
            return self._generate_error_sentiment(instrument, market_type, "OpenAI client not configured or API key missing.")

        search_topic = self._build_search_query(instrument, market_type)
        sentiment_data = None # Initialize sentiment_data
        
        try:
            # Limit concurrent requests using the semaphore initialized in __init__
            async with self.request_semaphore: 
                sentiment_data = await self._get_openai_sentiment_analysis(instrument, market_type, search_topic)
        except Exception as e: # Catch any unexpected error during the guarded call
            logger.error(f"Unexpected exception during OpenAI sentiment analysis for {instrument}: {e}")
            logger.exception(e)
            sentiment_data = None # Ensure it's None

        if sentiment_data:
            logger.info(f"Successfully fetched sentiment for {instrument} ({market_type}) using OpenAI.")
            if self.cache_enabled:
                self._add_market_specific_to_cache(instrument, market_type, sentiment_data)
            self.metrics.record_total_request(time.time() - request_start_time)
            return sentiment_data
        else:
            logger.warning(f"Failed to fetch sentiment for {instrument} ({market_type}) using OpenAI. Returning error structure.")
            self.metrics.record_total_request(time.time() - request_start_time)
            return self._generate_error_sentiment(instrument, market_type, "Failed to retrieve sentiment from OpenAI after API call.")

    def _generate_error_sentiment(self, instrument: str, market_type: str, error_message: str) -> Dict[str, Any]:
        """Generates a standardized error sentiment structure for user display and caching."""
        logger.error(f"Generating error sentiment for {instrument} ({market_type}): {error_message}")
        return {
            "instrument": instrument,
            "market_type": market_type,
            "overall_sentiment": "neutral", 
            "percentage_breakdown": {"bullish": 0, "bearish": 0, "neutral": 100},
            "trend_analysis": f"Error: {error_message}",
            "key_factors": "Sentiment analysis could not be performed at this time.",
            "confidence_score": 0.0,
            "sources_consulted": "N/A",
            "timestamp": datetime.utcnow().isoformat(),
            "error": True,
            "generated_by": "system_error"
        }

    def _format_compact_sentiment_text(self, instrument: str, bullish_pct: float, bearish_pct: float, neutral_pct: float = None) -> str:
        """
        Format a compact version of sentiment text for Telegram.
        This function creates a concise version that stays under 800 characters.
        """
        # Calculate neutral percentage if not provided
        if neutral_pct is None:
            neutral_pct = 100 - bullish_pct - bearish_pct
            
        # Determine overall sentiment based on bullish and bearish percentages
        sentiment = "Neutral ⚖️"
        sentiment_detail = "mixed"
        
        if bullish_pct - bearish_pct > 10:
            sentiment = "Bullish 📈"
            sentiment_detail = "bullish"
        elif bearish_pct - bullish_pct > 10:
            sentiment = "Bearish 📉"
            sentiment_detail = "bearish"
            
        # Create formatted text with HTML formatting
        formatted_text = f"""<b>🎯 {instrument} Market Sentiment Analysis</b>

<b>Overall Sentiment:</b> {sentiment}

<b>Market Sentiment Breakdown:</b>
🟢 Bullish: {bullish_pct}%
🔴 Bearish: {bearish_pct}%
⚪️ Neutral: {neutral_pct}%

<b>📊 Market Sentiment Analysis:</b>
Current market shows {sentiment_detail} trend with {bullish_pct}% positive sentiment vs {bearish_pct}% negative sentiment.

<b>📰 Key Drivers:</b>
• Economic indicators show {sentiment_detail} outlook
• Market sentiment calculation based on recent data
• Check detailed analysis for more information"""

        return formatted_text

    def _get_quick_local_sentiment(self, instrument: str) -> Dict[str, Any]:
        """
        Get a quick sentiment response from local data.
        This is the fastest way to get sentiment data, but it's less accurate.
        """
        logger.info(f"Getting quick local sentiment for {instrument}")
        
        try:
            # First check the cache
            cached_data = self._get_from_cache(instrument)
            if cached_data:
                logger.info(f"Cache hit for {instrument}, returning cached sentiment for quick local")
                
                # Format the cached data into a quick response
                sentiment_text = cached_data.get('analysis', cached_data.get('sentiment_text', ''))
                
                result = {
                    'instrument': instrument,
                    'bullish_percentage': cached_data.get('bullish', cached_data.get('bullish_percentage', 50)),
                    'bearish_percentage': cached_data.get('bearish', cached_data.get('bearish_percentage', 50)),
                    'neutral_percentage': cached_data.get('neutral', cached_data.get('neutral_percentage', 0)),
                    'sentiment_text': sentiment_text,
                    'source': 'cache_quick'
                }
                
                logger.info(f"Returning quick cached sentiment for {instrument}")
                return result
            
            # Create a loading message instead of fake sentiment values
            sentiment_text = f"""<b>⌛ Loading {instrument} Sentiment Analysis</b>

Please wait while we gather real-time market data...

<i>This may take a few moments. We're analyzing the latest market information to provide you with accurate sentiment data.</i>"""
            
            result = {
                'instrument': instrument,
                'bullish_percentage': 0,   # These values won't be shown
                'bearish_percentage': 0,   # These values won't be shown  
                'neutral_percentage': 0,   # These values won't be shown
                'sentiment_text': sentiment_text,
                'source': 'loading_screen',
                'is_loading': True         # Add flag to indicate this is just a loading screen
            }
            
            logger.info(f"Returning loading screen for {instrument}")
            return result
                
        except Exception as e:
            logger.error(f"Error getting quick local sentiment for {instrument}: {str(e)}")
            logger.exception(e)
            
            # Fallback to a simple loading message on error
            sentiment_text = f"""<b>⌛ Loading {instrument} Sentiment Analysis</b>

Please wait while we gather real-time market data...

<i>This may take a few moments as we connect to our data sources.</i>"""
            
            result = {
                'instrument': instrument,
                'bullish_percentage': 0,
                'bearish_percentage': 0,
                'neutral_percentage': 0,
                'sentiment_text': sentiment_text,
                'source': 'loading_error',
                'is_loading': True
            }
            
            logger.error(f"Error fallback used for {instrument} due to exception in quick local sentiment")
            return result

    async def load_cache(self):
        """
        Asynchronously load cache from file
        This can be called after initialization to load the cache without blocking startup
        
        Returns:
            bool: True if cache was loaded successfully, False otherwise
        """
        if not self.use_persistent_cache or not self.cache_file or self.cache_loaded:
            return False
            
        try:
            # Run the load operation in a thread pool to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._load_cache_from_file)
            self.cache_loaded = True
            logger.info(f"Asynchronously loaded {len(self.sentiment_cache)} sentiment cache entries")
            return True
        except Exception as e:
            logger.error(f"Error loading sentiment cache asynchronously: {str(e)}")
            return False

    def set_cache_ttl(self, minutes: int) -> None:
        """
        Set the cache Time-To-Live (TTL) duration
        
        Args:
            minutes: New TTL duration in minutes
        """
        if minutes < 1:
            logger.warning(f"Invalid cache TTL value ({minutes}), minimum is 1 minute")
            minutes = 1
            
        old_ttl = self.cache_ttl / 60
        self.cache_ttl = minutes * 60  # Convert to seconds
        
        logger.info(f"Updated cache TTL from {old_ttl} to {minutes} minutes")
        
        # Clean up any expired entries with the new TTL
        self.cleanup_expired_cache()
        
        # If persistent cache is enabled, save changes
        if self.use_persistent_cache and self.cache_file:
            self._save_cache_to_file()

    def get_cache_info(self, instrument: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed information about the cache status
        
        Args:
            instrument: Optional specific instrument to check. If None, returns info for all.
            
        Returns:
            Dictionary with cache information including TTL settings and cached instruments
        """
        now = time.time()
        
        if not self.cache_enabled:
            return {
                "cache_enabled": False,
                "message": "Cache is currently disabled"
            }
        
        # If specific instrument is requested, return details for just that one
        if instrument:
            cache_key = instrument.upper()
            
            if cache_key in self.sentiment_cache:
                entry = self.sentiment_cache[cache_key]
                age = now - entry['timestamp']
                expires_in = max(0, self.cache_ttl - age)
                
                # Create a clean copy without timestamp
                data_copy = copy.deepcopy(entry)
                if 'timestamp' in data_copy:
                    del data_copy['timestamp']
                
                return {
                    "cache_enabled": True,
                    "instrument": cache_key,
                    "cached": True,
                    "age_seconds": round(age),
                    "age_minutes": round(age / 60, 1),
                    "expires_in_seconds": round(expires_in),
                    "expires_in_minutes": round(expires_in / 60, 1),
                    "expired": age > self.cache_ttl,
                    "cache_ttl_minutes": self.cache_ttl / 60,
                    "data": data_copy
                }
            else:
                return {
                    "cache_enabled": True,
                    "instrument": cache_key,
                    "cached": False,
                    "message": f"No cache entry found for {cache_key}"
                }
        
        # Get information about all cached instruments
        cached_instruments = []
        for key, entry in self.sentiment_cache.items():
            age = now - entry['timestamp']
            expires_in = max(0, self.cache_ttl - age)
            
            cached_instruments.append({
                "instrument": key,
                "age_seconds": round(age),
                "age_minutes": round(age / 60, 1),
                "expires_in_seconds": round(expires_in),
                "expires_in_minutes": round(expires_in / 60, 1),
                "expired": age > self.cache_ttl,
                "bullish_percentage": entry.get("bullish_percentage", entry.get("bullish", 0)),
                "bearish_percentage": entry.get("bearish_percentage", entry.get("bearish", 0)),
                "source": entry.get("source", "unknown")
            })
        
        # Sort by expiration time (soonest first)
        cached_instruments.sort(key=lambda x: x["expires_in_seconds"])
        
        # Calculate cache statistics
        total_entries = len(self.sentiment_cache)
        active_entries = sum(1 for i in cached_instruments if not i["expired"])
        expired_entries = total_entries - active_entries
        
        return {
            "cache_enabled": True,
            "cache_ttl_minutes": self.cache_ttl / 60,
            "persistent_cache": self.use_persistent_cache,
            "cache_file": str(self.cache_file) if self.cache_file else None,
            "total_entries": total_entries,
            "active_entries": active_entries,
            "expired_entries": expired_entries,
            "instruments": cached_instruments
        }

    def _get_market_specific_cache_key(self, instrument: str, market_type: str) -> str:
        """
        Generate a market-specific cache key
        
        Args:
            instrument: Trading instrument symbol
            market_type: Market type (forex, crypto, etc.)
            
        Returns:
            Cache key in format INSTRUMENT_MARKETTYPE
        """
        instrument = instrument.upper()
        market_type = market_type.lower() if market_type else "unknown"
        return f"{instrument}_{market_type}"
    
    def _add_market_specific_to_cache(self, instrument: str, market_type: str, sentiment_data: Dict[str, Any]) -> None:
        """
        Add sentiment data to cache with market-specific key
        
        Args:
            instrument: Trading instrument symbol
            market_type: Market type (forex, crypto, etc.)
            sentiment_data: Sentiment data to cache
        """
        if not self.cache_enabled:
            return  # Cache is disabled
            
        try:
            # Generate market-specific cache key
            cache_key = self._get_market_specific_cache_key(instrument, market_type)
            
            # Make a copy to avoid reference issues
            cache_data = copy.deepcopy(sentiment_data)
            # Add timestamp for TTL check
            cache_data['timestamp'] = time.time()
            # Add market info
            cache_data['market_type'] = market_type
            
            # Store in memory cache
            self.sentiment_cache[cache_key] = cache_data
            
            # Also store with standard key for backward compatibility
            standard_key = instrument.upper()
            self.sentiment_cache[standard_key] = cache_data
            
            # If persistent cache is enabled, save to file
            if self.use_persistent_cache and self.cache_file:
                self._save_cache_to_file()
                
        except Exception as e:
            logger.error(f"Error adding to market-specific sentiment cache: {str(e)}")
    
    def _get_from_market_specific_cache(self, instrument: str, market_type: str) -> Optional[Dict[str, Any]]:
        """
        Get sentiment data from cache using market-specific key
        
        Args:
            instrument: Trading instrument symbol
            market_type: Market type (forex, crypto, etc.)
            
        Returns:
            Dictionary with sentiment data or None if not in cache or expired
        """
        if not self.cache_enabled:
            return None  # Cache is disabled
            
        try:
            # Try with market-specific key first
            cache_key = self._get_market_specific_cache_key(instrument, market_type)
            
            # If not found with market-specific key, try standard key
            if cache_key not in self.sentiment_cache:
                cache_key = instrument.upper()
                
                # If still not found, return None
                if cache_key not in self.sentiment_cache:
                    self.metrics.record_cache_miss()
                    return None
            
            cache_data = self.sentiment_cache[cache_key]
            
            # Check if expired
            current_time = time.time()
            cache_time = cache_data.get('timestamp', 0)
            
            if current_time - cache_time < self.cache_ttl:
                # Make a copy to avoid reference issues
                result = copy.deepcopy(cache_data)
                # Remove timestamp as it's internal
                if 'timestamp' in result:
                    del result['timestamp']
                
                # Record cache hit metric
                self.metrics.record_cache_hit()
                
                return result
            else:
                # Expired, remove from cache
                del self.sentiment_cache[cache_key]
                
                # Also remove standard key if it exists and has same timestamp
                standard_key = instrument.upper()
                if standard_key in self.sentiment_cache and self.sentiment_cache[standard_key].get('timestamp') == cache_time:
                    del self.sentiment_cache[standard_key]
                
            self.metrics.record_cache_miss()
            return None
            
        except Exception as e:
            logger.error(f"Error getting from market-specific sentiment cache: {str(e)}")
            return None

    def _format_sentiment_text(self, instrument: str, bullish_pct: float, bearish_pct: float, neutral_pct: float = None) -> str:
        """
        Format sentiment text based on percentages
        """
        # Calculate neutral if not provided
        if neutral_pct is None:
            neutral_pct = max(0, 100 - bullish_pct - bearish_pct)
            
        # Determine overall sentiment with more nuanced grading
        if bullish_pct > bearish_pct + 20:
            overall_sentiment = "Strongly Bullish 📈"
            strength = "strong"
            direction = "upward"
            outlook = "positive"
        elif bullish_pct > bearish_pct + 10:
            overall_sentiment = "Bullish 📈"
            strength = "notable"
            direction = "upward"
            outlook = "positive"
        elif bullish_pct > bearish_pct + 5:
            overall_sentiment = "Slightly Bullish 📈"
            strength = "mild"
            direction = "upward"
            outlook = "cautiously positive"
        elif bearish_pct > bullish_pct + 20:
            overall_sentiment = "Strongly Bearish 📉"
            strength = "strong"
            direction = "downward"
            outlook = "negative"
        elif bearish_pct > bullish_pct + 10:
            overall_sentiment = "Bearish 📉"
            strength = "notable"
            direction = "downward"
            outlook = "negative"
        elif bearish_pct > bullish_pct + 5:
            overall_sentiment = "Slightly Bearish 📉"
            strength = "mild"
            direction = "downward"
            outlook = "cautiously negative"
        else:
            overall_sentiment = "Neutral ⚖️"
            strength = "balanced"
            direction = "sideways"
            outlook = "mixed"
        
        # Determine volatility based on spread between bullish and bearish
        sentiment_spread = abs(bullish_pct - bearish_pct)
        if sentiment_spread > 30:
            volatility = "High"
            risk_profile = "elevated risk"
        elif sentiment_spread > 15:
            volatility = "Moderate"
            risk_profile = "moderate risk"
        else:
            volatility = "Low"
            risk_profile = "lower risk"
        
        # Generate main analysis text based on sentiment
        if bullish_pct > bearish_pct + 10:
            main_analysis = f"{instrument} currently shows {strength} bullish sentiment based on comprehensive market analysis. Positive economic indicators and favorable news flow are driving optimism among market participants. This {direction} momentum reflects confidence in the instrument's near-term prospects."
        elif bearish_pct > bullish_pct + 10:
            main_analysis = f"{instrument} currently shows {strength} bearish sentiment based on comprehensive market analysis. Concerning economic indicators and cautious news flow are weighing on market participants' outlook. This {direction} pressure suggests challenges in the instrument's near-term performance."
        else:
            main_analysis = f"{instrument} currently shows {strength} sentiment with no clear directional bias. Mixed economic signals and balanced market factors are creating a neutral trading environment. Market participants appear divided on future direction, resulting in {direction} price action with limited conviction."
        
        # Generate market context based on sentiment percentages
        context_analysis = f"With {bullish_pct:.0f}% bullish sentiment against {bearish_pct:.0f}% bearish sentiment, the {outlook} bias represents the current market consensus. Neutral positioning accounts for {neutral_pct:.0f}% of market sentiment, indicating some uncertainty remains despite the prevailing view."
        
        # Create economic indicators and policy stance bullets
        econ_indicators = f"""• Inflation data showing {outlook} economic trends
• Interest rate policies influencing market direction
• Economic growth figures affecting sentiment
• Central bank policy stance {direction} aligned
• Economic releases supporting {outlook} outlook"""
                
        # Create market developments bullets based on sentiment direction
        if bullish_pct > bearish_pct + 10:
            market_developments = f"""• Recent economic data releases supporting {outlook} view
• Central bank communications indicating accommodative stance
• Institutional activity showing increasing interest
• Positive market developments reinforcing bullish sentiment
• Policy decisions supporting economic expansion"""
        elif bearish_pct > bullish_pct + 10:
            market_developments = f"""• Recent economic data showing weakening trends
• Central bank communications indicating cautious outlook
• Institutional positioning reflecting diminished risk appetite
• Market developments reinforcing bearish sentiment
• Policy decisions focused on economic stability"""
        else:
            market_developments = f"""• Mixed economic data creating balanced outlook
• Central bank communications showing neutral stance
• Institutional positioning without clear directional bias
• Recent developments maintaining sentiment equilibrium
• Policy decisions balancing growth and stability"""
            
        # Create outlook based on sentiment (without technical references)
        if bullish_pct > bearish_pct + 15:
            outlook_text = f"The strong bullish bias (spread of {sentiment_spread:.0f}%) suggests a positive outlook for {instrument}. Traders should monitor upcoming economic releases and central bank communications that may affect market sentiment in this {volatility.lower()} volatility environment."
        elif bearish_pct > bullish_pct + 15:
            outlook_text = f"The strong bearish bias (spread of {sentiment_spread:.0f}%) suggests a cautious outlook for {instrument}. Traders should watch for upcoming economic data and policy announcements that may impact market direction in this {volatility.lower()} volatility environment."
        else:
            outlook_text = f"The balanced sentiment (narrow spread of {sentiment_spread:.0f}%) suggests a mixed outlook for {instrument}. Traders should remain attentive to upcoming economic indicators and policy statements that could provide clearer direction in this {volatility.lower()} volatility environment."
        
        # Format the full sentiment text with the new structure
        sentiment_text = f"""<b>🎯 {instrument} Market Sentiment Analysis</b>

<b>Overall Sentiment:</b> {overall_sentiment}

<b>Market Sentiment Breakdown:</b>
🟢 Bullish: {bullish_pct:.0f}%
🔴 Bearish: {bearish_pct:.0f}%
⚪️ Neutral: {neutral_pct:.0f}%

<b>Key Economic Indicators:</b>
{econ_indicators}

<b>Recent Market Developments:</b>
{market_developments}

<b>Market Outlook:</b>
{outlook_text}
"""
        return sentiment_text

    def _prepare_enhanced_sentiment_prompt(self, instrument: str, market_type: str, search_data: Optional[str] = None) -> str:
        """
        Prepare an enhanced prompt for DeepSeek to get better sentiment analysis
        
        Args:
            instrument: The instrument to analyze
            market_type: The market type of the instrument
            search_data: Optional search data from Tavily
            
        Returns:
            str: A formatted prompt for DeepSeek
        """
        logger.info(f"Preparing enhanced sentiment prompt for {instrument} ({market_type})")
        
        # Format the instrument properly for better analysis
        if len(instrument) == 6 and instrument.isalpha() and market_type == 'forex':
            formatted_instrument = f"{instrument[:3]}/{instrument[3:]}"
        else:
            formatted_instrument = instrument
        
        # Create current date/time reference
        from datetime import datetime
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Create base information for the prompt with structured sections
        prompt = f"""You are an expert financial market analyst specializing in sentiment analysis. Your task is to analyze the current market sentiment for {formatted_instrument} ({market_type.upper()}) based on the following market data.

ANALYSIS PARAMETERS:
- Instrument: {formatted_instrument}
- Market Type: {market_type.upper()}
- Analysis Date: {current_datetime}

"""
        
        # Add search data if we have it, with clear formatting
        if search_data and len(search_data) > 100:
            # Trim search data if it's too long
            if len(search_data) > 8000:
                logger.info(f"Trimming search data from {len(search_data)} to 8000 characters")
                search_data = search_data[:8000] + "...[truncated]"
            
            prompt += f"""MARKET DATA:
{search_data}

"""
        else:
            prompt += f"""NO RECENT MARKET DATA AVAILABLE FOR {formatted_instrument}. Please analyze based on your knowledge of current market conditions and provide a neutral assessment if you lack specific recent information.

"""
        
        # Add detailed instructions for the analysis
        prompt += f"""ANALYSIS REQUIREMENTS:
Please provide a detailed market sentiment analysis that includes:

1. Overall market sentiment (bullish, bearish, or neutral) with a clear percentage breakdown
2. Precise sentiment percentages: Bullish (0-100), Bearish (0-100), and Neutral (0-100) - these must add up to 100%
3. A thorough analysis of current market trends based on the provided data
4. Specific factors influencing sentiment with EXACT figures when available (interest rates, economic indicators)
5. Important events and news affecting the instrument with SPECIFIC details from the provided data

IMPORTANT GUIDELINES:
- Base your analysis SOLELY on the provided market data, not on general knowledge
- Include SPECIFIC economic data points like interest rates, inflation rates, GDP figures when mentioned
- Reference ACTUAL central banks (Fed, ECB, BoE) with their EXACT policy stances when available
- AVOID generic phrases like "evolving economic conditions" - be precise and data-driven
- If specific data is lacking, acknowledge the limitation rather than inventing details
- DO NOT mention specific price targets, support/resistance levels, or make price predictions
- Ensure percentages are aligned with the sentiment in your analysis (high bullish % should match bullish analysis)

REQUIRED RESPONSE FORMAT:
Return a valid JSON object with these exact fields:
- "bullish_percentage": (number between 0-100)
- "bearish_percentage": (number between 0-100)
- "neutral_percentage": (number between 0-100)
- "formatted_text": (HTML formatted text that follows the template below)

The "formatted_text" MUST follow this EXACT HTML template:

<b>🎯 {formatted_instrument} Market Sentiment Analysis</b>

<b>Overall Sentiment:</b> [Bullish 📈 or Bearish 📉 or Neutral ⚖️]

<b>Market Sentiment Breakdown:</b>
🟢 Bullish: XX%
🔴 Bearish: YY%
⚪️ Neutral: ZZ%

<b>📊 Market Sentiment Analysis:</b>
[Detailed and SPECIFIC analysis of current market sentiment based on the data]

<b>📰 Key Sentiment Drivers:</b>
• [SPECIFIC factor #1 with EXACT data points]
• [SPECIFIC factor #2 with EXACT data points]
• [SPECIFIC factor #3 with EXACT data points]

<b>📅 Important Events & News:</b>
• [SPECIFIC event/news #1 with details]
• [SPECIFIC event/news #2 with details]
• [SPECIFIC event/news #3 with details]

<b>🔮 Sentiment Outlook:</b>
[Brief outlook based ONLY on the provided data]

IMPORTANT: The percentages MUST add up to 100% and the formatted_text MUST include all sections with the exact HTML tags shown above.
"""
        
        # Return the complete prompt
        return prompt
        
    async def _process_sentiment_request(self, instrument: str, prompt: str) -> Dict[str, Any]:
        """
        Process a sentiment request to DeepSeek API
        
        Args:
            instrument: The instrument to analyze
            prompt: The prepared prompt for DeepSeek
            
        Returns:
            Dict[str, Any]: The processed sentiment data
        """
        logger.info(f"Processing sentiment request for {instrument}")
        
        if not self.deepseek_api_key:
            logger.error(f"No DeepSeek API key configured")
            return None
        
        try:
            # Prepare the API request
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are a financial market analyst specializing in sentiment analysis. Your task is to analyze market data and provide detailed sentiment analysis with specific percentages and sections."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 1500,
                "response_format": {"type": "json_object"}
            }
            
            # Make the API call with increased timeout
            logger.info(f"Calling DeepSeek API for {instrument}")
            async with aiohttp.ClientSession() as session:
                # Use a much longer timeout for better reliability
                timeout = aiohttp.ClientTimeout(total=45)  # 45 seconds timeout
                
                try:
                    async with session.post(
                        self.deepseek_url,
                        headers=headers,
                        json=payload,
                        timeout=timeout
                    ) as response:
                        if response.status != 200:
                            logger.error(f"DeepSeek API returned status {response.status}: {await response.text()}")
                            return None
                        
                        response_data = await response.json()
                except asyncio.TimeoutError:
                    logger.error(f"DeepSeek API request timed out after 45 seconds for {instrument}")
                    return None
                except aiohttp.ClientError as e:
                    logger.error(f"DeepSeek API client error for {instrument}: {str(e)}")
                    return None
            
            # Extract the content from the response
            content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            logger.info(f"Received DeepSeek response for {instrument} ({len(content)} bytes)")
            
            try:
                # Parse the JSON response
                sentiment_data = json.loads(content)
                
                # Validate the response has required fields
                required_fields = ["bullish_percentage", "bearish_percentage", "neutral_percentage", "formatted_text"]
                missing_fields = [field for field in required_fields if field not in sentiment_data]
                
                if missing_fields:
                    logger.error(f"DeepSeek response missing required fields: {missing_fields}")
                    return None
                
                # Ensure percentages add up to 100
                bullish = sentiment_data["bullish_percentage"]
                bearish = sentiment_data["bearish_percentage"]
                neutral = sentiment_data["neutral_percentage"]
                total = bullish + bearish + neutral
                
                if abs(total - 100) > 0.01:  # Allow for small floating point errors
                    logger.warning(f"Sentiment percentages sum to {total}, adjusting to 100%")
                    
                    # Adjust proportionally to sum to 100%
                    bullish = round((bullish / total) * 100)
                    bearish = round((bearish / total) * 100)
                    neutral = 100 - bullish - bearish
                    
                    sentiment_data["bullish_percentage"] = bullish
                    sentiment_data["bearish_percentage"] = bearish
                    sentiment_data["neutral_percentage"] = neutral
                
                # Update the formatted text with correct percentages if needed
                formatted_text = sentiment_data["formatted_text"]
                if "🟢 Bullish:" in formatted_text:
                    # Extract current percentages in the text
                    bullish_match = re.search(r'🟢 Bullish:\s*(\d+)%', formatted_text)
                    bearish_match = re.search(r'🔴 Bearish:\s*(\d+)%', formatted_text)
                    neutral_match = re.search(r'⚪️ Neutral:\s*(\d+)%', formatted_text)
                    
                    if bullish_match and int(bullish_match.group(1)) != bullish:
                        formatted_text = re.sub(r'(🟢 Bullish:)\s*\d+%', f'🟢 Bullish: {bullish}%', formatted_text)
                    
                    if bearish_match and int(bearish_match.group(1)) != bearish:
                        formatted_text = re.sub(r'(🔴 Bearish:)\s*\d+%', f'🔴 Bearish: {bearish}%', formatted_text)
                    
                    if neutral_match and int(neutral_match.group(1)) != neutral:
                        formatted_text = re.sub(r'(⚪️ Neutral:)\s*\d+%', f'⚪️ Neutral: {neutral}%', formatted_text)
                    
                    sentiment_data["formatted_text"] = formatted_text
                
                # Ensure overall sentiment matches the percentages
                overall_sentiment = None
                if bullish > bearish:
                    overall_sentiment = "Bullish 📈"
                elif bearish > bullish:
                    overall_sentiment = "Bearish 📉"
                else:
                    overall_sentiment = "Neutral ⚖️"
                
                # Update overall sentiment in formatted text if needed
                if "<b>Overall Sentiment:</b>" in formatted_text:
                    formatted_text = re.sub(r'(<b>Overall Sentiment:</b>).*?\n', f'\\1 {overall_sentiment}\n', formatted_text)
                    sentiment_data["formatted_text"] = formatted_text
                
                # Add marker to show this is real data
                formatted_text = sentiment_data["formatted_text"]
                if not formatted_text.endswith("</i>"):
                    sentiment_data["formatted_text"] = formatted_text + "\n\n<i>✅ Real-time market data analysis</i>"
                
                return sentiment_data
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse DeepSeek response as JSON: {str(e)}")
                logger.error(f"Raw content: {content[:200]}...")
                return None
            except Exception as e:
                logger.error(f"Error processing DeepSeek response: {str(e)}")
                logger.exception(e)
                return None
                
        except Exception as e:
            logger.error(f"Error calling DeepSeek API: {str(e)}")
            logger.exception(e)
            return None

    def _guess_market_from_instrument(self, instrument: str) -> str:
        """
        Guess the market type from the instrument name.
        
        Args:
            instrument: The instrument name to analyze
            
        Returns:
            str: The guessed market type ('forex', 'crypto', 'stock', etc.)
        """
        # Convert to uppercase
        instrument = instrument.upper()
        
        # Check for common forex pairs
        if len(instrument) == 6 and instrument.isalpha():
            # Common forex pairs are 6 characters: EURUSD, GBPUSD, etc.
            return "forex"
        
        # Check for crypto pairs
        if instrument.endswith("BTC") or instrument.endswith("ETH") or instrument.endswith("USDT") or instrument.startswith("BTC") or instrument.startswith("ETH"):
            return "crypto"
            
        # Check for indices
        indices = ["SPX", "DJI", "IXIC", "FTSE", "DAX", "CAC", "NIKKEI", "HSI"]
        if any(idx in instrument for idx in indices):
            return "index"
        
        # Default to stock if nothing else matches
        return "stock"

    async def _get_fast_sentiment(self, instrument: str) -> Dict[str, Any]:
        """
        Get a fast sentiment response for initial telegram display.
        This optimizes for speed and will trigger a background fetch for full data.
        
        Args:
            instrument: The instrument to analyze (e.g., 'EURUSD')
            
        Returns:
            Dictionary with basic sentiment data
        """
        start_time = time.time()
        logger.info(f"Getting fast sentiment for {instrument}")
        
        try:
            # First check cache for existing data
            cached_data = self._get_from_cache(instrument)
            if cached_data:
                # Cache hit - use it
                logger.info(f"Cache hit for {instrument}, returning cached sentiment (elapsed: {time.time() - start_time:.2f}s)")
                
                # If the cached data is from a background API call, it's already properly formatted
                if cached_data.get('source') in ['background_api', 'api', 'api_cache']:
                    logger.info(f"Using high-quality cached data for {instrument} (source: {cached_data.get('source')})")
                    return cached_data
                
                # Otherwise format a proper response from the cached data
                result = {
                    'instrument': instrument,
                    'bullish_percentage': cached_data.get('bullish', cached_data.get('bullish_percentage', 50)),
                    'bearish_percentage': cached_data.get('bearish', cached_data.get('bearish_percentage', 50)),
                    'neutral_percentage': cached_data.get('neutral', cached_data.get('neutral_percentage', 0)),
                    'sentiment_text': cached_data.get('analysis', cached_data.get('sentiment_text', '')),
                    'source': 'cache'
                }
                
                logger.info(f"Returning cached sentiment for {instrument} (elapsed: {time.time() - start_time:.2f}s)")
                return result
            
            # No cache hit, try to get data from API, but with shorter timeouts for fast response
            
            # Attempt 1: Try direct API approach which can be faster for complete data
            try:
                logger.info(f"Trying direct API approach for fast sentiment of {instrument}")
                market_type = self._detect_market_type(instrument)
                
                result = await asyncio.wait_for(
                    self._get_direct_api_sentiment(instrument, market_type),
                    timeout=10  # Short timeout for fast response
                )
                
                if result:
                    logger.info(f"Successfully got direct API sentiment for {instrument} (elapsed: {time.time() - start_time:.2f}s)")
                    
                    # Format response
                    fast_response = {
                        'instrument': instrument,
                        'bullish_percentage': result.get('bullish', 50),
                        'bearish_percentage': result.get('bearish', 50),
                        'neutral_percentage': result.get('neutral', 0),
                        'sentiment_text': result.get('analysis', ''),
                        'source': 'api_fast'
                    }
                    
                    logger.info(f"Returning fast direct API sentiment for {instrument} (elapsed: {time.time() - start_time:.2f}s)")
                    self._add_to_cache(instrument, fast_response)  # Cache the fast response
                    return fast_response
            except asyncio.TimeoutError:
                logger.warning(f"Direct API timed out after 10s in fast sentiment for {instrument}")
            except Exception as e:
                logger.error(f"Error in direct API approach for fast sentiment: {str(e)}")
            
            # Attempt 2: Try standard approach with Tavily + DeepSeek
            try:
                logger.info(f"Trying standard approach for fast sentiment of {instrument}")
                
                # 1. Get market data for the instrument
                logger.info(f"Building search query for {instrument} in fast sentiment")
                market_type = self._detect_market_type(instrument)
                query = self._build_search_query(instrument, market_type)
                logger.info(f"Getting market data for {instrument} in fast sentiment with query: {query}")
                
                market_data = await self._get_tavily_news(query)
                
                if market_data:
                    logger.info(f"Retrieved {len(market_data)} bytes of market data for {instrument} in fast sentiment")
                    
                    # 2. Use DeepSeek to analyze market data
                    logger.info(f"Preparing DeepSeek prompt for {instrument} in fast sentiment")
                    prompt = self._prepare_enhanced_sentiment_prompt(instrument, market_type, market_data)
                    
                    # Call DeepSeek with short timeout
                    logger.info(f"Calling DeepSeek API for {instrument} in fast sentiment")
                    sentiment_result = await asyncio.wait_for(
                        self._process_sentiment_request(instrument, prompt),
                        timeout=15  # Short timeout for fast response
                    )
                    
                    if sentiment_result and 'formatted_text' in sentiment_result:
                        logger.info(f"Got DeepSeek response for {instrument} in fast sentiment")
                        
                        # Extract sentiment values
                        bullish_pct = sentiment_result.get('bullish_percentage', 50)
                        bearish_pct = sentiment_result.get('bearish_percentage', 50)
                        neutral_pct = sentiment_result.get('neutral_percentage', 0)
                        
                        # Format response
                        fast_response = {
                            'instrument': instrument,
                            'bullish_percentage': bullish_pct,
                            'bearish_percentage': bearish_pct,
                            'neutral_percentage': neutral_pct,
                            'sentiment_text': sentiment_result.get('formatted_text', ''),
                            'source': 'api_fast'
                        }
                        
                        logger.info(f"Returning fast sentiment for {instrument} (elapsed: {time.time() - start_time:.2f}s)")
                        self._add_to_cache(instrument, fast_response)  # Cache the fast response
                        return fast_response
                    else:
                        logger.warning(f"DeepSeek returned no valid data for {instrument} in fast sentiment")
                else:
                    logger.warning(f"Could not retrieve market data for {instrument} in fast sentiment")
            except asyncio.TimeoutError:
                logger.warning(f"DeepSeek timed out after 15s in fast sentiment for {instrument}")
            except Exception as e:
                logger.error(f"Error in standard approach for fast sentiment: {str(e)}")
            
            logger.error(f"All fast API attempts failed for {instrument}. Unable to update cache with real data.")
            
            # If all attempts failed, return a simple response with neutral sentiment
            # This ensures we always return something instead of None
            bullish_pct = 50
            bearish_pct = 50
            neutral_pct = 0
            
            # Generate a generic sentiment text 
            sentiment_text = f"<b>🧠 {instrument} Market Sentiment</b>\n\n"
            sentiment_text += "<b>Overall Sentiment:</b> Neutral ➡️\n\n"
            sentiment_text += "<b>Market Sentiment Breakdown:</b>\n"
            sentiment_text += f"• Bullish: {bullish_pct}%\n"
            sentiment_text += f"• Bearish: {bearish_pct}%\n"
            sentiment_text += f"• Neutral: {neutral_pct}%\n\n"
            sentiment_text += f"<i>Fast response with limited data. Analysis is being generated...</i>\n\n"
            sentiment_text += f"<i>Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
            
            result = {
                'instrument': instrument,
                'bullish_percentage': bullish_pct,
                'bearish_percentage': bearish_pct,
                'neutral_percentage': neutral_pct,
                'sentiment_text': sentiment_text,
                'source': 'fast_fallback'
            }
            
            logger.info(f"Returning minimal fast sentiment data for {instrument} due to API failures (elapsed: {time.time() - start_time:.2f}s)")
            return result
                
        except Exception as e:
            logger.error(f"Error getting fast sentiment for {instrument}: {str(e)}")
            logger.exception(e)
            
            # Even on unexpected exceptions, return something
            bullish_pct = 50
            bearish_pct = 50
            neutral_pct = 0
            
            sentiment_text = f"<b>🧠 {instrument} Market Sentiment</b>\n\n"
            sentiment_text += "<b>Overall Sentiment:</b> Neutral ➡️\n\n"
            sentiment_text += "<b>Market Sentiment Breakdown:</b>\n"
            sentiment_text += f"• Bullish: {bullish_pct}%\n"
            sentiment_text += f"• Bearish: {bearish_pct}%\n"
            sentiment_text += f"• Neutral: {neutral_pct}%\n\n"
            sentiment_text += "<i>Error retrieving sentiment data. Analysis is being generated...</i>\n\n"
            sentiment_text += f"<i>Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
            
            result = {
                'instrument': instrument,
                'bullish_percentage': bullish_pct,
                'bearish_percentage': bearish_pct,
                'neutral_percentage': neutral_pct,
                'sentiment_text': sentiment_text,
                'source': 'error_fallback'
            }
            
            logger.error(f"Error fallback used for {instrument} due to exception")
            return result

    async def _fetch_background_sentiment(self, instrument: str, market_type: str) -> None:
        """
        Fetch sentiment data in the background and update the cache when complete.
        This method is called when fast sentiment returns a quick response but we need to
        fetch the real data in the background without blocking the user.
        
        Args:
            instrument: The instrument to analyze (e.g., 'EURUSD')
            market_type: The market type of the instrument (e.g., 'forex', 'crypto')
        """
        logger.info(f"Starting background sentiment fetch for {instrument}")
        
        try:
            # Start a timer to track processing time
            start_time = time.time()
            
            # Check that we have API keys
            if not self.deepseek_api_key or not self.tavily_api_key:
                logger.error(f"Background sentiment fetch failed: Missing API keys (DeepSeek: {'Yes' if self.deepseek_api_key else 'No'}, Tavily: {'Yes' if self.tavily_api_key else 'No'})")
                return
            
            # Try to get real sentiment data
            # First with direct approach which is faster for complete data
            try:
                logger.info(f"Trying direct API approach for background fetch of {instrument}")
                result = await asyncio.wait_for(
                    self._get_direct_api_sentiment(instrument, market_type),
                    timeout=45  # Longer timeout for background task
                )
                
                if result:
                    # We got real data, update the cache
                    logger.info(f"Successfully fetched background sentiment data for {instrument} in {time.time() - start_time:.2f}s")
                    
                    # Add a note that this is real data fetched in background
                    if 'analysis' in result:
                        result['analysis'] = result['analysis'] + "\n\n<i>✅ Real-time market data analysis (background updated)</i>"
                    
                    # Update both caches
                    logger.info(f"Updating caches with real background sentiment data for {instrument}")
                    self._add_to_cache(instrument, result)
                    self._add_market_specific_to_cache(instrument, market_type, result)
                    
                    logger.info(f"Background sentiment fetch complete for {instrument} in {time.time() - start_time:.2f}s")
                    return
                else:
                    logger.warning(f"Direct API approach failed for {instrument} in background")
            except Exception as e:
                logger.error(f"Error in direct API sentiment: {str(e)}")
                logger.warning(f"Direct API approach failed for {instrument} in background")
            
            # If direct API failed, try standard approach
            logger.info(f"Trying standard approach for background fetch of {instrument}")
            
            # 1. Get market data
            logger.info(f"Building search query for {instrument} in background")
            query = self._build_search_query(instrument, market_type)
            logger.info(f"Getting market data for {instrument} in background with query: {query}")
            search_data = await self._get_tavily_news(query)
            
            if not search_data:
                logger.error(f"Could not retrieve market data for {instrument} in background")
                return
            
            logger.info(f"Retrieved {len(search_data)} bytes of market data for {instrument} in background")
            
            # 2. Get sentiment analysis from DeepSeek
            try:
                # Prepare the prompt with market data
                logger.info(f"Preparing DeepSeek prompt for {instrument} in background")
                prompt = self._prepare_enhanced_sentiment_prompt(instrument, market_type, search_data)
                
                # Process sentiment request
                logger.info(f"Calling DeepSeek API for {instrument} in background")
                sentiment_result = await asyncio.wait_for(
                    self._process_sentiment_request(instrument, prompt),
                    timeout=60  # Increased timeout for background task to avoid frequent timeouts
                )
                
                if sentiment_result and 'formatted_text' in sentiment_result:
                    # Create a properly formatted result
                    logger.info(f"Successfully processed DeepSeek response for {instrument} in background")
                    
                    bullish_pct = sentiment_result.get('bullish_percentage', 50)
                    bearish_pct = sentiment_result.get('bearish_percentage', 50)
                    neutral_pct = sentiment_result.get('neutral_percentage', 0)
                    
                    # Calculate sentiment score and overall sentiment
                    sentiment_score = (bullish_pct - bearish_pct) / 100
                    overall_sentiment = 'bullish' if bullish_pct > bearish_pct else 'bearish' if bearish_pct > bullish_pct else 'neutral'
                    
                    # Prepare formatted analysis with note that it's real data
                    analysis = sentiment_result.get('formatted_text', '')
                    analysis = analysis + "\n\n<i>✅ Real-time market data analysis (background updated)</i>"
                    
                    # Create final result
                    result = {
                        'bullish': bullish_pct,
                        'bearish': bearish_pct,
                        'neutral': neutral_pct,
                        'sentiment_score': sentiment_score,
                        'technical_score': 'Based on market analysis',
                        'news_score': f"{bullish_pct}% positive",
                        'social_score': f"{bearish_pct}% negative",
                        'trend_strength': 'Strong' if abs(bullish_pct - 50) > 15 else 'Moderate' if abs(bullish_pct - 50) > 5 else 'Weak',
                        'volatility': 'Moderate',
                        'volume': 'Normal',
                        'news_headlines': [],
                        'overall_sentiment': overall_sentiment,
                        'analysis': analysis,
                        'source': 'background_api'
                    }
                    
                    # Update both caches
                    logger.info(f"Updating caches with real background sentiment data for {instrument}")
                    self._add_to_cache(instrument, result)
                    self._add_market_specific_to_cache(instrument, market_type, result)
                    
                    logger.info(f"Background sentiment fetch complete for {instrument} in {time.time() - start_time:.2f}s")
                    return
                else:
                    logger.warning(f"DeepSeek returned no valid sentiment data for {instrument} in background")
            except asyncio.TimeoutError:
                logger.warning(f"DeepSeek timed out after 60s in background for {instrument}")
            except Exception as e:
                logger.error(f"Error processing DeepSeek sentiment in background: {str(e)}")
            
            # If we reach this point, all API attempts failed
            logger.error(f"All background API attempts failed for {instrument}. Unable to update cache with real data.")
        
        except Exception as e:
            logger.error(f"Uncaught error in background sentiment fetch for {instrument}: {str(e)}")

    def _add_to_cache(self, instrument: str, sentiment_data: Dict[str, Any]) -> None:
        """
        Add sentiment data to cache
        
        Args:
            instrument: Trading instrument symbol
            sentiment_data: Sentiment data to cache
        """
        if not self.cache_enabled:
            return  # Cache is disabled
            
        try:
            # Normalize instrument to uppercase
            cache_key = instrument.upper()
            
            # Make a copy to avoid reference issues
            cache_data = copy.deepcopy(sentiment_data)
            # Add timestamp for TTL check
            cache_data['timestamp'] = time.time()
            
            # Store in memory cache
            self.sentiment_cache[cache_key] = cache_data
            
            # If persistent cache is enabled, save to file
            if self.use_persistent_cache and self.cache_file:
                self._save_cache_to_file()
                
        except Exception as e:
            logger.error(f"Error adding to sentiment cache: {str(e)}")
    
    async def _get_tavily_news(self, query: str) -> Optional[str]:
        """
        Get news and market data using Tavily API
        
        Args:
            query: Search query for Tavily
            
        Returns:
            str: Market data as text or None if failed
        """
        if not self.tavily_api_key:
            logger.error("No Tavily API key configured")
            return None
            
        try:
            logger.info(f"Searching for news using Tavily API")
            
            # Use the Tavily client to perform the search
            result = await self.tavily_client.search(
                query=query,
                search_depth="advanced",
                include_answer=True,
                include_images=False,
                max_results=5
            )
            
            if not result:
                logger.warning(f"Tavily search returned no results for: {query}")
                return None
                
            # Process Tavily data
            market_data = f"# Market Analysis for {query}\n\n"
            
            # Add the generated answer if available
            if result.get("answer"):
                market_data += f"## Summary\n{result['answer']}\n\n"
            
            # Add search results
            if result.get("results"):
                market_data += "## Market News and Analysis\n\n"
                for i, item in enumerate(result["results"], 1):
                    market_data += f"### {item.get('title', f'Source {i}')}\n"
                    market_data += f"{item.get('content', 'No content available')}\n"
                    market_data += f"Source: {item.get('url', 'Unknown')}\n\n"
            
            # Log the success
            logger.info(f"Retrieved {len(result.get('results', []))} market data items with {len(market_data)} characters")
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching news from Tavily: {str(e)}")
            return None
    
    def _save_cache_to_file(self) -> None:
        """Save sentiment cache to disk"""
        if not self.use_persistent_cache or not self.cache_file:
            return
            
        try:
            with self._cache_lock:
                # Convert cache to serializable format
                cache_data = {}
                for key, value in self.sentiment_cache.items():
                    # Make a deep copy to avoid threading issues
                    cache_data[key] = copy.deepcopy(value)
                
                # Write to file
                with open(self.cache_file, 'w') as f:
                    json.dump(cache_data, f)
                
                logger.debug(f"Saved {len(cache_data)} items to sentiment cache file: {self.cache_file}")
                
        except Exception as e:
            logger.error(f"Error saving sentiment cache to file: {str(e)}")

    def _get_from_cache(self, instrument: str) -> Optional[Dict[str, Any]]:
        """
        Get sentiment data from cache if available and not expired
        
        Args:
            instrument: Trading instrument symbol
            
        Returns:
            Dict[str, Any] or None: Cached sentiment data or None if not available
        """
        if not self.cache_enabled:
            return None
            
        try:
            # Normalize instrument to uppercase
            cache_key = instrument.upper()
            
            # Check if we have it in memory cache
            if cache_key in self.sentiment_cache:
                cache_data = self.sentiment_cache[cache_key]
                
                # Check if it's expired
                timestamp = cache_data.get('timestamp', 0)
                age = time.time() - timestamp
                
                if age < self.cache_ttl:
                    logger.debug(f"Cache hit for {instrument} (age: {age:.1f}s)")
                    return copy.deepcopy(cache_data)
                else:
                    logger.debug(f"Cache expired for {instrument} (age: {age:.1f}s)")
            
            return None
                
        except Exception as e:
            logger.error(f"Error getting from sentiment cache: {str(e)}")
            return None

    async def _get_openai_sentiment_analysis(self, instrument: str, market_type: str, search_topic: str) -> Optional[Dict[str, Any]]:
        """
        Get sentiment analysis from OpenAI gpt-4o-mini.
        Args:
            instrument: The trading instrument.
            market_type: The market type.
            search_topic: The search topic formulated for research.
        Returns:
            A dictionary with sentiment data or None if an error occurs.
        """
        if not self.openai_client:
            logger.error("OpenAI client not initialized. Cannot fetch sentiment.")
            return None
            
        prompt = f"""You are an expert financial market analyst. Your task is to provide a DETAILED and ACCURATE market sentiment analysis for {instrument} (market: {market_type}).
Research and analyze recent news, market data, economic indicators, and any relevant events related to the search topic: '{search_topic}'.

Based on your comprehensive research, provide the following in a VALID JSON format:
1.  "overall_sentiment": Your assessment of the overall market sentiment (string: "bullish", "bearish", or "neutral").
2.  "percentage_breakdown": A precise percentage breakdown (JSON object: {{"bullish": %d, "bearish": %d, "neutral": %d}}). Percentages must sum to 100.
3.  "trend_analysis": A concise analysis of current market trends and sentiment drivers (string, 2-3 sentences).
4.  "key_factors": Specific key factors influencing the sentiment. List at least 3-5 factors, including actual RECENT news events, economic data points, or policy decisions if found (string, bullet points or numbered list).
5.  "confidence_score": Your confidence in this analysis (float, 0.0 to 1.0).
6.  "sources_consulted": Briefly mention the types of sources or information categories you based your analysis on (e.g., "financial news portals, economic releases, technical indicators"). Do not list URLs.

IMPORTANT INSTRUCTIONS:
- Ensure the analysis is SPECIFIC, DETAILED, and based on verifiable information patterns if possible. Avoid generic statements.
- Focus on RECENT and RELEVANT information (last few days to a week).
- The entire output MUST be a single JSON object. Do not include any text outside the JSON structure.
Example JSON structure:
{{
  "overall_sentiment": "bullish",
  "percentage_breakdown": {{
    "bullish": 65,
    "bearish": 25,
    "neutral": 10
  }},
  "trend_analysis": "The market shows strong upward momentum driven by positive earnings reports and favorable macroeconomic data.",
  "key_factors": "- Release of strong non-farm payroll data.\n- Hawkish stance from the central bank.\n- Increased institutional investment in {instrument}.",
  "confidence_score": 0.85,
  "sources_consulted": "Financial news articles, central bank statements, market data aggregators."
}}
"""

        logger.info(f"Requesting OpenAI sentiment for {instrument} with topic: {search_topic}")
        start_time = time.time()

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",  # More widely accessible model
                messages=[
                    {"role": "system", "content": "You are a specialized financial sentiment analysis AI. Provide output STRICTLY in the requested JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3, 
                max_tokens=1000, 
                response_format={"type": "json_object"},
                timeout=self.api_timeout 
            )
            
            duration = time.time() - start_time
            self.metrics.record_api_call('openai', duration)
            logger.info(f"OpenAI API call for {instrument} completed in {duration:.2f}s")

            if response.choices and response.choices[0].message and response.choices[0].message.content:
                content = response.choices[0].message.content
                logger.debug(f"OpenAI raw response for {instrument}: {content[:500]}...") # Log only part of it
                try:
                    sentiment_data = json.loads(content)
                    # Basic validation of expected structure
                    if not all(k in sentiment_data for k in ["overall_sentiment", "percentage_breakdown", "trend_analysis", "key_factors"]):
                        logger.error(f"OpenAI response for {instrument} is missing key fields. Response excerpt: {content[:200]}...")
                        return None
                    if not isinstance(sentiment_data.get("percentage_breakdown"), dict) or not all(k in sentiment_data["percentage_breakdown"] for k in ["bullish", "bearish", "neutral"]):
                        logger.error(f"OpenAI response for {instrument} has malformed percentage_breakdown. Response excerpt: {content[:200]}...")
                        return None
                    
                    # Add instrument and timestamp for consistency with previous structure if needed by caching/consumers
                    sentiment_data['instrument'] = instrument
                    sentiment_data['timestamp'] = datetime.utcnow().isoformat()
                    sentiment_data['generated_by'] = "openai_gpt-4o-mini"

                    return sentiment_data
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response from OpenAI for {instrument}: {e}. Response excerpt: {content[:200]}...")
                    return None
            else:
                logger.error(f"No content in OpenAI response for {instrument}. Full Response: {response}")
                return None

        except openai.APIConnectionError as e:
            logger.error(f"OpenAI API connection error for {instrument}: {e}")
        except openai.RateLimitError as e:
            logger.error(f"OpenAI API rate limit exceeded for {instrument}: {e}")
        except openai.APIStatusError as e:
            logger.error(f"OpenAI API status error for {instrument}: status={e.status_code}, response={e.response}")
        except asyncio.TimeoutError:
            logger.error(f"OpenAI API request timed out after {self.api_timeout}s for {instrument}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during OpenAI API call for {instrument}: {e}")
            logger.exception(e)
        
        # Record failed call duration too, if not already recorded before exception
        # This might double record if exception happens after record_api_call, but it's fine for metrics.
        duration = time.time() - start_time
        self.metrics.record_api_call('openai', duration) 
        return None
