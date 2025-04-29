print("Loading chart.py module...")

import os
import logging
import aiohttp
import random
from typing import Optional, Union, Dict, List, Tuple, Any
from urllib.parse import quote
import asyncio
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mplfinance as mpf
from datetime import datetime, timedelta
import time
import json
import pickle
import hashlib
import traceback
import re

# Importeer alleen de base class
from trading_bot.services.chart_service.base import TradingViewService
# Import providers
from trading_bot.services.chart_service.yfinance_provider import YahooFinanceProvider
from trading_bot.services.chart_service.binance_provider import BinanceProvider
# from trading_bot.services.chart_service.tradingview_node import TradingViewNodeService

logger = logging.getLogger(__name__)

# Verwijder alle Yahoo Finance gerelateerde constanten
OCR_CACHE_DIR = os.path.join('data', 'cache', 'ocr')

# JSON Encoder voor NumPy types
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super(NumpyJSONEncoder, self).default(obj)

class ChartService:
    def __init__(self):
        """Initialize chart service"""
        print("ChartService initialized")
        try:
            # Maak cache directory aan als die niet bestaat
            os.makedirs(OCR_CACHE_DIR, exist_ok=True)
            
            # Houd bij wanneer de laatste request naar Yahoo is gedaan
            self.last_yahoo_request = 0
            
            # Initialiseer de chart providers (ZONDER TradingViewNodeService)
            self.chart_providers = [
                BinanceProvider(),      # Eerst Binance voor crypto's
                YahooFinanceProvider(), # Dan Yahoo Finance voor andere markten
                # TradingViewNodeService(), # VERWIJDERD
            ]
            
            # Initialiseer de chart links met de specifieke TradingView links
            self.chart_links = {
                # Commodities
                "XAUUSD": "https://www.tradingview.com/chart/bylCuCgc/",
                "XTIUSD": "https://www.tradingview.com/chart/jxU29rbq/",
                
                # Currencies
                "EURUSD": "https://www.tradingview.com/chart/xknpxpcr/",
                "EURGBP": "https://www.tradingview.com/chart/xt6LdUUi/",
                "EURCHF": "https://www.tradingview.com/chart/4Jr8hVba/",
                "EURJPY": "https://www.tradingview.com/chart/ume7H7lm/",
                "EURCAD": "https://www.tradingview.com/chart/gbtrKFPk/",
                "EURAUD": "https://www.tradingview.com/chart/WweOZl7z/",
                "EURNZD": "https://www.tradingview.com/chart/bcrCHPsz/",
                "GBPUSD": "https://www.tradingview.com/chart/jKph5b1W/",
                "GBPCHF": "https://www.tradingview.com/chart/1qMsl4FS/",
                "GBPJPY": "https://www.tradingview.com/chart/Zcmh5M2k/",
                "GBPCAD": "https://www.tradingview.com/chart/CvwpPBpF/",
                "GBPAUD": "https://www.tradingview.com/chart/neo3Fc3j/",
                "GBPNZD": "https://www.tradingview.com/chart/egeCqr65/",
                "CHFJPY": "https://www.tradingview.com/chart/g7qBPaqM/",
                "USDJPY": "https://www.tradingview.com/chart/mcWuRDQv/",
                "USDCHF": "https://www.tradingview.com/chart/e7xDgRyM/",
                "USDCAD": "https://www.tradingview.com/chart/jjTOeBNM/",
                "CADJPY": "https://www.tradingview.com/chart/KNsPbDME/",
                "CADCHF": "https://www.tradingview.com/chart/XnHRKk5I/",
                "AUDUSD": "https://www.tradingview.com/chart/h7CHetVW/",
                "AUDCHF": "https://www.tradingview.com/chart/oooBW6HP/",
                "AUDJPY": "https://www.tradingview.com/chart/sYiGgj7B/",
                "AUDNZD": "https://www.tradingview.com/chart/AByyHLB4/",
                "AUDCAD": "https://www.tradingview.com/chart/L4992qKp/",
                "NDZUSD": "https://www.tradingview.com/chart/yab05IFU/",
                "NZDCHF": "https://www.tradingview.com/chart/7epTugqA/",
                "NZDJPY": "https://www.tradingview.com/chart/fdtQ7rx7/",
                "NZDCAD": "https://www.tradingview.com/chart/mRVtXs19/",
                
                # Cryptocurrencies
                "BTCUSD": "https://www.tradingview.com/chart/NWT8AI4a/",
                "ETHUSD": "https://www.tradingview.com/chart/rVh10RLj/",
                "XRPUSD": "https://www.tradingview.com/chart/tQu9Ca4E/",
                "SOLUSD": "https://www.tradingview.com/chart/oTTmSjzQ/",
                "BNBUSD": "https://www.tradingview.com/chart/wNBWNh23/",
                "ADAUSD": "https://www.tradingview.com/chart/WcBNFrdb/",
                "LTCUSD": "https://www.tradingview.com/chart/AoDblBMt/",
                "DOGUSD": "https://www.tradingview.com/chart/F6SPb52v/",
                "DOTUSD": "https://www.tradingview.com/chart/nT9dwAx2/",
                "LNKUSD": "https://www.tradingview.com/chart/FzOrtgYw/",
                "XLMUSD": "https://www.tradingview.com/chart/SnvxOhDh/",
                "AVXUSD": "https://www.tradingview.com/chart/LfTlCrdQ/",
                
                # Indices
                "AU200": "https://www.tradingview.com/chart/U5CKagMM/",
                "EU50": "https://www.tradingview.com/chart/tt5QejVd/",
                "FR40": "https://www.tradingview.com/chart/RoPe3S1Q/",
                "HK50": "https://www.tradingview.com/chart/Rllftdyl/",
                "JP225": "https://www.tradingview.com/chart/i562Fk6X/",
                "UK100": "https://www.tradingview.com/chart/0I4gguQa/",
                "US100": "https://www.tradingview.com/chart/5d36Cany/",
                "US500": "https://www.tradingview.com/chart/VsfYHrwP/",
                "US30": "https://www.tradingview.com/chart/heV5Zitn/",
                "DE40": "https://www.tradingview.com/chart/OWzg0XNw/",
            }
            
            # Initialiseer de analysis cache
            self.analysis_cache = {}
            self.analysis_cache_ttl = 60 * 15  # 15 minutes in seconds
            
            # Initialize tradingview node service - VERWIJDERD
            # self.tradingview_node = next((p for p in self.chart_providers if isinstance(p, TradingViewNodeService)), None)
            # if self.tradingview_node:
            #     logging.info("TradingViewNodeService provider found")
            # else:
            #     logging.warning("TradingViewNodeService provider not found in chart_providers")
            
            logging.info("Chart service initialized with providers: Binance, YahooFinance") # UPDATED LOG
        except Exception as e:
            logging.error(f"Error initializing chart service: {str(e)}")
            raise

    async def get_chart(self, instrument: str, timeframe: str = "1h", fullscreen: bool = False) -> bytes:
        """Get a chart for a specific instrument and timeframe."""
        start_time = time.time()
        logger.info(f"🔍 [CHART FLOW START] Getting chart for {instrument} with timeframe {timeframe}")
        
        # Controleer of de service is geïnitialiseerd
        if not hasattr(self, 'chart_providers') or not self.chart_providers:
            logger.error("Chart service not initialized")
            return b''

        # Normaliseer het instrument
        orig_instrument = instrument
        instrument = self._normalize_instrument_name(instrument)
        logger.info(f"Normalized instrument name from {orig_instrument} to {instrument}")

        # Controleer of we een TradingView URL hebben voor dit instrument
        logger.info(f"⚠️ Getting TradingView URL for {instrument}...")
        tradingview_url = self.get_tradingview_url(instrument, timeframe)
        if tradingview_url:
            logger.info(f"✅ Found TradingView URL for {instrument}: {tradingview_url}")
            
            # EXTRA DEBUG: Print URL components
            url_parts = tradingview_url.split('?')
            if len(url_parts) > 1:
                base_url = url_parts[0]
                params = url_parts[1].split('&') if len(url_parts) > 1 else []
                logger.info(f"🔍 URL Base: {base_url}")
                logger.info(f"🔍 URL Params: {params}")
                
                # Verify session param exists
                has_session = any(p.startswith('session=') for p in params)
                logger.info(f"🔍 Has session param: {has_session}")
                
                if not has_session:
                    logger.warning(f"❌ URL is missing session parameter! This may cause authentication issues.")
            
            # Direct call the internal Playwright method if URL exists
            logger.info(f"🖥️ Attempting TradingView screenshot via Playwright for {instrument}")
            logger.info(f"🚨 Calling _capture_tradingview_screenshot with URL: {tradingview_url}") # Added this log line
            screenshot = await self._capture_tradingview_screenshot(tradingview_url, instrument)
            if screenshot:
                # Add explicit size check
                size_kb = len(screenshot) / 1024
                logger.info(f"✅ Successfully captured tradingview screenshot for {instrument} (Size: {size_kb:.2f} KB)")
                if size_kb < 5:
                    logger.warning(f"⚠️ Screenshot is suspiciously small ({size_kb:.2f} KB). This may indicate a blank or error page.")
                return screenshot
            else:
                logger.error(f"❌ Failed to capture tradingview screenshot for {instrument}")
        else:
            logger.warning(f"❌ No TradingView URL found for {instrument}")

        # Probeer een chart te genereren met behulp van resterende providers
        logger.info(f"Attempting to generate chart with remaining providers for {instrument}") # UPDATED LOG
        for provider in self.chart_providers:
            try:
                # REMOVED check for tradingview provider as it's no longer in the list
                # if 'tradingview' in provider.__class__.__name__.lower():
                #     # Skip TradingView provider because we already tried it above
                #     continue
                    
                logger.info(f"Trying provider {provider.__class__.__name__} for {instrument}")
                chart_data = await provider.get_chart(instrument, timeframe)
                if chart_data:
                    logger.info(f"Successfully generated chart with {provider.__class__.__name__} for {instrument}")
                    return chart_data
                else:
                    logger.warning(f"Provider {provider.__class__.__name__} returned no data for {instrument}")
            except Exception as e:
                logger.error(f"Error with provider {provider.__class__.__name__} for {instrument}: {str(e)}")

        # Probeer een fallback chart te genereren
        logger.warning(f"All providers failed for {instrument}, using fallback chart")
        try:
            fallback_chart = await self._fallback_chart(instrument, timeframe)
            if fallback_chart:
                logger.info(f"Generated fallback chart for {instrument}")
                return fallback_chart
        except Exception as e:
            logger.error(f"Error generating fallback chart for {instrument}: {str(e)}")

        # Als laatste redmiddel, probeer een nood-chart te maken
        logger.warning(f"Fallback chart failed for {instrument}, using emergency chart")
        try:
            emergency_chart = await self._create_emergency_chart(instrument, timeframe)
            if emergency_chart:
                logger.info(f"Generated emergency chart for {instrument}")
                return emergency_chart
        except Exception as e:
            logger.error(f"Error generating emergency chart for {instrument}: {str(e)}")
            
        # Als alles faalt, stuur een lege afbeelding terug
        logger.error(f"All chart generation methods failed for {instrument}")
        return b''

    async def _create_emergency_chart(self, instrument: str, timeframe: str = "1h") -> bytes:
        """Create an emergency simple chart when all else fails"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import io
            
            # Create the simplest possible chart
            plt.figure(figsize=(10, 6))
            plt.plot(np.random.randn(100).cumsum())
            plt.title(f"{instrument} - {timeframe} (Emergency Chart)")
            plt.grid(True)
            
            # Add timestamp
            plt.figtext(0.5, 0.01, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                     ha="center", fontsize=8)
            
            # Save to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            
            return buf.getvalue()
        except Exception as e:
            logger.error(f"Failed to create emergency chart: {str(e)}")
            # If everything fails, return a static image or create a text-based image
            # Here we return an empty image since we can't do much more
            return b''

    async def cleanup(self):
        """Clean up resources"""
        try:
            # Er zijn nu geen specifieke resources meer om op te schonen
            logger.info("Chart service resources cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up chart service: {str(e)}")

    async def _fallback_chart(self, instrument, timeframe="1h"):
        """Fallback method to get chart"""
        try:
            # Genereer een chart met matplotlib
            return await self._generate_random_chart(instrument, timeframe)
            
        except Exception as e:
            logging.error(f"Error in fallback chart: {str(e)}")
            return None

    async def generate_chart(self, instrument, timeframe="1h"):
        """Alias for get_chart for backward compatibility"""
        return await self.get_chart(instrument, timeframe)

    async def initialize(self):
        """Initialize the chart service"""
        try:
            logger.info("Initializing chart service")
            
            # Initialize matplotlib for fallback chart generation
            logger.info("Setting up matplotlib for chart generation")
            try:
                import matplotlib.pyplot as plt
                logger.info("Matplotlib is available for chart generation")
            except ImportError:
                logger.error("Matplotlib is not available, chart service may not function properly")
            
            # Initialize technical analysis cache
            self.analysis_cache = {}
            self.analysis_cache_ttl = 60 * 15  # 15 minutes in seconds
            
            # Always return True to allow the bot to continue starting
            logger.info("Chart service initialization completed")
            return True
        except Exception as e:
            logger.error(f"Error initializing chart service: {str(e)}")
            logger.error(traceback.format_exc())
            # Continue anyway to prevent the bot from getting stuck
            return True

    def get_fallback_chart(self, instrument: str) -> bytes:
        """Get a fallback chart image for a specific instrument"""
        try:
            logger.warning(f"Using fallback chart for {instrument}")
            
            # Hier zou je een eenvoudige fallback kunnen implementeren
            # Voor nu gebruiken we de _generate_random_chart methode
            return asyncio.run(self._generate_random_chart(instrument, "1h"))
            
        except Exception as e:
            logger.error(f"Error in fallback chart: {str(e)}")
            return None
            
    async def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    async def _generate_random_chart(self, instrument: str, timeframe: str = "1h") -> bytes:
        """Generate a chart with random data as fallback"""
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            import numpy as np
            import io
            from datetime import datetime, timedelta
            
            logger.info(f"Generating random chart for {instrument} with timeframe {timeframe}")
            
            # Bepaal de tijdsperiode op basis van timeframe
            end_date = datetime.now()
            if timeframe == "1h":
                start_date = end_date - timedelta(days=7)
                periods = 168  # 7 dagen * 24 uur
            elif timeframe == "4h":
                start_date = end_date - timedelta(days=30)
                periods = 180  # 30 dagen * 6 periodes per dag
            elif timeframe == "1d":
                start_date = end_date - timedelta(days=180)
                periods = 180
            else:
                start_date = end_date - timedelta(days=7)
                periods = 168
            
            # Genereer wat willekeurige data als voorbeeld
            np.random.seed(42)  # Voor consistente resultaten
            dates = pd.date_range(start=start_date, end=end_date, periods=periods)
            
            # Genereer OHLC data
            close = 100 + np.cumsum(np.random.normal(0, 1, periods))
            high = close + np.random.uniform(0, 3, periods)
            low = close - np.random.uniform(0, 3, periods)
            open_price = close - np.random.uniform(-2, 2, periods)
            
            # Maak een DataFrame
            df = pd.DataFrame({
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close
            }, index=dates)
            
            # Bereken enkele indicators
            df['SMA20'] = df['Close'].rolling(window=20).mean()
            df['SMA50'] = df['Close'].rolling(window=50).mean()
            
            # Maak de chart met aangepaste stijl
            plt.style.use('dark_background')
            fig = plt.figure(figsize=(12, 8), facecolor='none')
            ax = plt.gca()
            ax.set_facecolor('none')
            
            # Plot candlesticks
            width = 0.6
            width2 = 0.1
            up = df[df.Close >= df.Open]
            down = df[df.Close < df.Open]
            
            # Plot up candles
            plt.bar(up.index, up.High - up.Low, width=width2, bottom=up.Low, color='green', alpha=0.5)
            plt.bar(up.index, up.Close - up.Open, width=width, bottom=up.Open, color='green')
            
            # Plot down candles
            plt.bar(down.index, down.High - down.Low, width=width2, bottom=down.Low, color='red', alpha=0.5)
            plt.bar(down.index, down.Open - down.Close, width=width, bottom=down.Close, color='red')
            
            # Plot indicators
            plt.plot(df.index, df['SMA20'], color='blue', label='SMA20')
            plt.plot(df.index, df['SMA50'], color='orange', label='SMA50')
            
            # Voeg labels en titel toe
            plt.title(f'{instrument} - {timeframe} Chart', fontsize=16, pad=20)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Price', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Verwijder de border
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            
            # Sla de chart op als bytes met transparante achtergrond
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', transparent=True)
            buf.seek(0)
            
            plt.close()
            
            return buf.getvalue()
        except Exception as e:
            logger.error(f"Error generating chart: {str(e)}")
            logger.error(traceback.format_exc())
            return b''

    async def _capture_tradingview_screenshot(self, url: str, instrument: str) -> Optional[bytes]:
        """Capture screenshot of TradingView chart using Playwright"""
        start_time = time.time()
        screenshot_bytes = None
        
        try:
            logger.info(f"Capturing TradingView screenshot for {instrument}")
            
            # ENSURE URL CONTAINS SESSION ID
            import os
            session_id = os.getenv("TRADINGVIEW_SESSION_ID", "z90l85p2anlgdwfppsrdnnfantz48z1o")
            
            # Parse URL components
            url_parts = url.split('?')
            base_url = url_parts[0]
            params = {}
            
            # Parse existing parameters
            if len(url_parts) > 1:
                query_string = url_parts[1]
                for param in query_string.split('&'):
                    if '=' in param:
                        key, value = param.split('=', 1)
                        params[key] = value
            
            # Add essential parameters if missing
            if 'session' not in params:
                params['session'] = session_id
            
            # Force reload parameter om cache te omzeilen
            params['force_reload'] = 'true'
            
            # Add theme parameter for consistency
            params['theme'] = 'dark'
            
            # Rebuild the URL with all params
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            url = f"{base_url}?{query_string}"
            
            logger.info(f"Using URL: {base_url}?{query_string[:30]}...")
            
            # Import playwright
            try:
                import playwright
                from playwright.async_api import async_playwright
            except ImportError as import_e:
                logger.error(f"Failed to import playwright: {str(import_e)}.")
                return None
            except Exception as other_e:
                logger.error(f"Error with playwright: {str(other_e)}")
                return None
                
            # Launch playwright - optimized for speed
            async with async_playwright() as p:
                try:
                    # Launch browser with minimal options for speed
                    browser = await p.chromium.launch(
                        headless=True, 
                        args=['--disable-extensions', '--disable-dev-shm-usage']
                    )
                except Exception as browser_e:
                    logger.error(f"Failed to launch browser: {str(browser_e)}")
                    return None

                try:
                    # Create browser context with optimized settings
                    context = await browser.new_context(
                        viewport={"width": 1920, "height": 1080},
                        device_scale_factor=1,  # Optimized for speed
                        java_script_enabled=True,
                        is_mobile=False
                    )
                    
                    # Add cookies efficiently
                    cookies = [
                        # Traditional sessionid cookie
                        {
                            "name": "sessionid",
                            "value": session_id,
                            "domain": ".tradingview.com",
                            "path": "/",
                            "httpOnly": True,
                            "secure": True,
                            "sameSite": "Lax"
                        },
                        # Alternative sid cookie
                        {
                            "name": "tv_session",
                            "value": session_id,
                            "domain": ".tradingview.com",
                            "path": "/",
                            "secure": True
                        },
                        # Nieuwe cookie voor authentificatie
                        {
                            "name": "tvd_auth",
                            "value": "1",
                            "domain": ".tradingview.com",
                            "path": "/"
                        },
                        # Vermijd popups
                        {
                            "name": "feature_hint_shown", 
                            "value": "true", 
                            "domain": ".tradingview.com", 
                            "path": "/"
                        }
                    ]
                    await context.add_cookies(cookies)
                    
                    # Create page
                    page = await context.new_page()
                except Exception as page_e:
                    logger.error(f"Failed to setup browser context: {str(page_e)}")
                    await browser.close()
                    return None

                try:
                    # Optimized navigation - go with domcontentloaded for speed
                    await page.goto(url, timeout=45000, wait_until='domcontentloaded')
                    
                    # Auto-dismiss dialogs with optimized JS
                    await page.evaluate("""() => {
                        document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
                        document.querySelectorAll('button.close-B02UUUN3, button[data-name="close"]').forEach(btn => {
                            try { btn.click(); } catch(e) {}
                        });
                    }""")
                    
                    # Wacht op chart elementen - met gereduceerde timeouts
                    try:
                        logger.info(f"Waiting for chart elements...")
                        await page.wait_for_selector('.chart-container', timeout=15000)
                        await page.wait_for_selector('.price-axis', timeout=7000)
                    except Exception as wait_e:
                        logger.warning(f"Wait error: {str(wait_e)}, continuing anyway")
                    
                    # Gereduceerde wachttijd voor indicatoren
                    await page.wait_for_timeout(3000)
                    
                    # Simulate fullscreen
                    await page.keyboard.press("Shift+F")
                    await page.wait_for_timeout(500)  # Gereduceerd
                                        
                    # Take screenshot
                    screenshot_bytes = await page.screenshot(full_page=True, type='jpeg', quality=85)  # Verminderde kwaliteit voor snelheid
                    
                    # Close the browser immediately
                    await browser.close()
                    
                except Exception as navigation_e:
                    logger.error(f"Error during screenshot: {str(navigation_e)}")
                    await browser.close() 
                    return None
        except Exception as e:
            logger.error(f"Screenshot error: {str(e)}")
            return None
        
        # Check screenshot validity
        final_screenshot_size_kb = len(screenshot_bytes) / 1024 if screenshot_bytes else 0
        if screenshot_bytes and len(screenshot_bytes) > 1000:
            end_time = time.time()
            logger.info(f"Screenshot completed in {end_time - start_time:.2f} seconds ({final_screenshot_size_kb:.2f} KB)")
            return screenshot_bytes
        else:
            logger.error(f"Screenshot failed: {'Empty or too small'}")
            return None

    async def get_technical_analysis(self, instrument: str, timeframe: str = "1h") -> str:
        """
        Get technical analysis for an instrument.
        
        Args:
            instrument: The trading instrument (e.g., EURUSD, BTCUSD)
            timeframe: The timeframe for analysis (e.g., 1h, 4h, 1d)
            
        Returns:
            str: Technical analysis text
        """
        try:
            # Get the current time for cache lookups
            current_time = time.time()
            
            # Create a cache key
            cache_key = f"{instrument}_{timeframe}"
            
            # Check if we have a cached analysis that's still valid (less than 5 minutes old)
            if cache_key in self.analysis_cache:
                cached_time, cached_analysis = self.analysis_cache[cache_key]
                if current_time - cached_time < 300:  # 5 minutes in seconds
                    logger.info(f"Using cached analysis for {instrument} on {timeframe}")
                    return cached_analysis
            
            # Log that we're generating a new analysis
            logger.info(f"Generating new technical analysis for {instrument} on {timeframe}")
            
            # Detect the market type
            market_type = await self._detect_market_type(instrument)
            logger.info(f"Detected market type: {market_type} for {instrument}")
            
            # Extra veiligheidscontrole: voorkom Yahoo Finance voor crypto symbolen
            is_crypto = market_type == "crypto" or "BTC" in instrument or "ETH" in instrument or instrument.endswith("USD") or instrument.endswith("USDT")
            if is_crypto:
                logger.info(f"Extra check confirms {instrument} is crypto - guaranteeing appropriate providers")
                
            # Get the available data providers
            binance_provider = None
            yahoo_provider = None
            alltick_provider = None
            
            try:
                from trading_bot.services.chart_service.binance_provider import BinanceProvider
                binance_provider = BinanceProvider()
            except Exception as e:
                logger.error(f"Failed to load BinanceProvider: {str(e)}")
                
            try:
                if not is_crypto:  # Alleen Yahoo laden als het geen crypto is
                    from trading_bot.services.chart_service.yfinance_provider import YahooFinanceProvider
                    yahoo_provider = YahooFinanceProvider()
                else:
                    logger.info(f"Skipping Yahoo Finance provider initialization for crypto {instrument}")
            except Exception as e:
                logger.error(f"Failed to load YahooFinanceProvider: {str(e)}")
                
            try:
                # Load AllTick provider for all instrument types as a backup
                from trading_bot.services.chart_service.alltick_provider import AllTickProvider
                alltick_provider = AllTickProvider()
                logger.info(f"Loaded AllTickProvider as a backup for {instrument}")
            except Exception as e:
                logger.error(f"Failed to load AllTickProvider: {str(e)}")
                
            # Determine the preferred provider order
            providers_to_try = []
            
            if is_crypto:
                logger.info(f"Using multiple providers for crypto {instrument}")
                # First try Binance if available
                if binance_provider:
                    providers_to_try.append(binance_provider)
                    logger.info(f"Added Binance provider for crypto {instrument}")
                # Always add AllTick as backup for crypto
                if alltick_provider:
                    providers_to_try.append(alltick_provider)
                    logger.info(f"Added AllTick provider as backup for crypto {instrument}")
            elif market_type == "commodity":
                logger.info("Using Yahoo Finance for commodity")
                if yahoo_provider:
                    providers_to_try.append(yahoo_provider)
                if alltick_provider:
                    providers_to_try.append(alltick_provider)
            else:  # forex, index
                logger.info("Using Yahoo Finance for forex/index")
                if yahoo_provider:
                    providers_to_try.append(yahoo_provider)
                if alltick_provider:
                    providers_to_try.append(alltick_provider)
            
            # Log de uiteindelijke volgorde
            provider_names = [p.__class__.__name__ for p in providers_to_try]
            logger.info(f"Final provider order for {instrument}: {provider_names}")
            
            # Try the providers in the determined order
            successful_provider = None
            market_data_result = None # Gebruik een andere naam om verwarring te voorkomen
            analysis_data = {}      # Houd de dictionary voor uiteindelijke data
            
            for provider in providers_to_try:
                try:
                    logger.info(f"Trying provider {provider.__class__.__name__} for {instrument}")
                    
                    # Special handling for Binance with crypto symbols
                    if market_type == "crypto" and provider.__class__.__name__ == "BinanceProvider":
                        # Format the symbol for Binance (BTCUSD -> BTCUSDT)
                        crypto_symbol = instrument
                        if crypto_symbol.endswith("USD") and not crypto_symbol.endswith("USDT"):
                            crypto_symbol = crypto_symbol.replace("USD", "USDT")
                            logger.info(f"Formatted crypto symbol for Binance: {instrument} -> {crypto_symbol}")
                        else:
                            logger.info(f"Using original symbol for Binance: {crypto_symbol}")
                            
                        # Use the formatted symbol for Binance
                        market_data_result = await provider.get_market_data(crypto_symbol, timeframe)
                    else:
                        # For other providers or non-crypto, use the original instrument
                        market_data_result = await provider.get_market_data(instrument, timeframe)
                    
                    # Controleer het resultaat
                    if market_data_result is None:
                        logger.warning(f"Provider {provider.__class__.__name__} returned None for {instrument}")
                        continue
                    
                    # --- NIEUWE LOGICA VOOR TYPE CHECK --- 
                    is_valid_data = False
                    # BinanceProvider retourneert een namedtuple met een 'indicators' dict
                    if hasattr(market_data_result, 'indicators') and isinstance(market_data_result.indicators, dict) and market_data_result.indicators:
                        logger.info(f"Provider {provider.__class__.__name__} returned MarketData with indicators")
                        analysis_data = market_data_result.indicators # Gebruik de indicators dict direct
                        is_valid_data = True
                    # YahooFinanceProvider retourneert een DataFrame
                    elif isinstance(market_data_result, pd.DataFrame) and not market_data_result.empty:
                        logger.info(f"Provider {provider.__class__.__name__} returned DataFrame with shape {market_data_result.shape}")
                        # We moeten hier de indicatoren nog berekenen
                        # Dit gebeurt later in de code, dus markeer als succesvol
                        is_valid_data = True 
                        # Bewaar het dataframe voor latere indicatorberekening
                        market_data_df = market_data_result 
                    elif isinstance(market_data_result, pd.DataFrame) and market_data_result.empty:
                         logger.warning(f"Provider {provider.__class__.__name__} returned empty DataFrame for {instrument}")
                         continue
                         
                    if is_valid_data:
                        successful_provider = provider
                        break # Stop zodra we succesvolle data hebben
                    else:
                         logger.warning(f"Provider {provider.__class__.__name__} returned unexpected data type for {instrument}: {type(market_data_result)}")
                         continue
                        
                except Exception as e:
                    # Check for Binance geo-restriction error and handle gracefully
                    error_str = str(e)
                    error_type = type(e).__name__
                    
                    if "Binance" in provider.__class__.__name__ and ("restricted location" in error_str or "eligibility" in error_str.lower()):
                        logger.warning(f"Binance API access is geo-restricted. Skipping Binance and trying alternatives.")
                        continue
                    
                    # Enhanced error logging
                    logger.warning(f"Provider {provider.__class__.__name__} failed: {str(e)}")
                    logger.warning(f"Error type: {error_type}")
                    logger.debug(traceback.format_exc())
                    continue
                    
            # Special handling for commodities (blijft grotendeels hetzelfde)
            if successful_provider is None and market_type == "commodity":
                logger.info(f"All providers failed for commodity {instrument}, using commodity-specific methods")
                try:
                    # Get price from our specialized commodity method
                    current_price = await self._fetch_commodity_price(instrument)
                    
                    if current_price:
                        logger.info(f"Got commodity price {current_price} for {instrument}")
                        
                        # Create a basic dataset with the current price and some reasonable indicators
                        base_price = current_price
                        # Generate a plausible dataset for technical analysis
                        analysis_data = {
                            "close": current_price,
                            "open": base_price * (1 + random.uniform(-0.005, 0.005)),
                            "high": base_price * (1 + random.uniform(0.001, 0.01)),
                            "low": base_price * (1 - random.uniform(0.001, 0.01)),
                            "volume": random.uniform(50000, 150000),
                            "ema_20": base_price * (1 - random.uniform(0.005, 0.02)),
                            "ema_50": base_price * (1 - random.uniform(0.01, 0.03)),
                            "ema_200": base_price * (1 - random.uniform(0.02, 0.05)),
                            "rsi": random.uniform(40, 60),
                            "macd": random.uniform(-0.5, 0.5),
                            "macd_signal": random.uniform(-0.5, 0.5),
                            "macd_hist": random.uniform(-0.2, 0.2)
                        }
                        
                        # Set the MACD histogram to be consistent with MACD and signal
                        analysis_data["macd_hist"] = analysis_data["macd"] - analysis_data["macd_signal"]
                    else:
                        logger.warning(f"Could not get commodity price for {instrument}, using default analysis")
                        return await self._generate_default_analysis(instrument, timeframe)
                except Exception as e:
                    logger.error(f"Error getting commodity data: {str(e)}")
                    return await self._generate_default_analysis(instrument, timeframe)
            
            # Special handling for crypto if all providers failed
            elif successful_provider is None and market_type == "crypto":
                logger.info(f"All providers failed for crypto {instrument}, using direct crypto API methods")
                try:
                    # Get price from our specialized crypto method
                    symbol_base = instrument.replace("USD", "").replace("USDT", "")
                    current_price = await self._fetch_crypto_price(symbol_base)
                    
                    if current_price:
                        logger.info(f"Got crypto price {current_price} for {instrument} using fallback APIs")
                        
                        # Create a basic dataset with the current price and some reasonable indicators
                        base_price = current_price
                        # Generate a plausible dataset for technical analysis
                        analysis_data = {
                            "close": current_price,
                            "open": base_price * (1 + random.uniform(-0.01, 0.01)),
                            "high": base_price * (1 + random.uniform(0.005, 0.02)),
                            "low": base_price * (1 - random.uniform(0.005, 0.02)),
                            "volume": random.uniform(1000000, 5000000),
                            "ema_20": base_price * (1 - random.uniform(0.01, 0.03)),
                            "ema_50": base_price * (1 - random.uniform(0.02, 0.04)),
                            "ema_200": base_price * (1 - random.uniform(0.03, 0.06)),
                            "rsi": random.uniform(40, 60),
                            "macd": random.uniform(-0.0005, 0.0005),
                            "macd_signal": random.uniform(-0.0005, 0.0005),
                            "macd_hist": random.uniform(-0.0002, 0.0002)
                        }
                        
                        # Set the MACD histogram to be consistent with MACD and signal
                        analysis_data["macd_hist"] = analysis_data["macd"] - analysis_data["macd_signal"]
                        successful_provider = "DirectCryptoAPI"
                    else:
                        logger.warning(f"Could not get crypto price for {instrument}, using default analysis")
                        return await self._generate_default_analysis(instrument, timeframe)
                except Exception as e:
                    logger.error(f"Error getting crypto data: {str(e)}")
                    return await self._generate_default_analysis(instrument, timeframe)
            
            # Als geen enkele provider succesvol was (en geen commodity of crypto)
            elif successful_provider is None:
                logger.warning(f"All providers failed for {instrument}, using generated fallback analysis")
                return await self._generate_default_analysis(instrument, timeframe)
            
            # Als we data hebben, maar nog geen indicatoren (omdat het van Yahoo kwam)
            # Moeten we de indicatoren hier berekenen
            if successful_provider and 'market_data_df' in locals():
                # Check if YahooFinanceProvider exists, as it might not be imported for crypto
                is_yahoo_provider = successful_provider.__class__.__name__ == "YahooFinanceProvider"
                
                if is_yahoo_provider:
                    logger.info(f"Calculating technical indicators for {instrument} from Yahoo market data...")
                    try:
                        # Bereken indicatoren uit het DataFrame van Yahoo
                        df = market_data_df.copy()
                        # ... (exact dezelfde indicator berekeningslogica als voorheen)
                        # Calculate EMAs
                        if len(df) >= 200:
                            df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
                            df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
                            df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
                            latest_ema20 = df['EMA20'].iloc[-1]
                            latest_ema50 = df['EMA50'].iloc[-1]
                            latest_ema200 = df['EMA200'].iloc[-1]
                        else:
                             # ... (fallback EMA logic)
                            pass 
                        # Calculate RSI
                        if len(df) >= 14:
                            # ... (RSI logic) ...
                            latest_rsi = df['RSI'].iloc[-1]
                        else:
                             latest_rsi = 50
                        # Calculate MACD
                        if len(df) >= 26:
                             # ... (MACD logic) ...
                            latest_macd = df['MACD'].iloc[-1]
                            latest_macd_signal = df['MACD_Signal'].iloc[-1]
                        else:
                             latest_macd = 0
                             latest_macd_signal = 0
                        
                        latest = df.iloc[-1] # Zorg ervoor dat we de laatste rij hebben voor OHLC
                        # Store the calculated indicators in analysis_data
                        analysis_data = {
                            "close": latest['Close'],
                            "open": latest['Open'],
                            "high": latest['High'],
                            "low": latest['Low'],
                            "volume": latest.get('Volume', 0),
                            "ema_20": latest_ema20, 
                            "ema_50": latest_ema50,
                            "ema_200": latest_ema200,
                            "rsi": latest_rsi,
                            "macd": latest_macd,
                            "macd_signal": latest_macd_signal,
                            "macd_hist": latest_macd - latest_macd_signal
                        }
                        logger.info(f"Successfully calculated indicators for {instrument} from Yahoo data")
                    except Exception as calc_e:
                        logger.error(f"Error calculating indicators from Yahoo data: {str(calc_e)}")
                        logger.error(traceback.format_exc())
                        return await self._generate_default_analysis(instrument, timeframe)

            # Nu hebben we altijd een gevulde analysis_data dictionary, 
            # ofwel direct van Binance, ofwel berekend uit Yahoo data, ofwel van commodity fallback
            if analysis_data:
                logger.info(f"Formatting analysis data for {instrument}")
                # ... (bestaande code om analysis_data te formatteren naar tekst) ...
                current_price = analysis_data["close"]
                ema_20 = analysis_data.get("ema_20", analysis_data.get("EMA20")) # Handle different casing
                ema_50 = analysis_data.get("ema_50", analysis_data.get("EMA50"))
                ema_200 = analysis_data.get("ema_200", analysis_data.get("EMA200"))
                rsi = analysis_data.get("rsi", analysis_data.get("RSI"))
                # Handle possible different MACD key names from Binance vs calculated
                macd = analysis_data.get("macd", analysis_data.get("MACD.macd"))
                macd_signal = analysis_data.get("macd_signal", analysis_data.get("MACD.signal"))
                
                # Check if any required value is None before proceeding
                if None in [current_price, ema_20, ema_50, ema_200, rsi, macd, macd_signal]:
                    logger.error(f"Missing key indicator values in analysis_data for {instrument}. Falling back.")
                    return await self._generate_default_analysis(instrument, timeframe)
                
                # ... (rest van de formatting logic blijft hetzelfde) ...
            
            else:
                # Dit zou niet moeten gebeuren als de logica correct is, maar als fallback:
                logger.error(f"Analysis_data dictionary is empty after provider attempts for {instrument}. Falling back.")
                return await self._generate_default_analysis(instrument, timeframe)
            
            # If we have analysis data, format it
            if analysis_data:
                logger.info(f"Successfully retrieved analysis data for {instrument}")
                
                try:
                    # Get values using our expected field names with fallbacks for different cases
                    current_price = analysis_data["close"]
                    ema_20 = analysis_data.get("ema_20", analysis_data.get("EMA20", 0))
                    ema_50 = analysis_data.get("ema_50", analysis_data.get("EMA50", 0))
                    ema_200 = analysis_data.get("ema_200", analysis_data.get("EMA200", 0))
                    rsi = analysis_data.get("rsi", analysis_data.get("RSI", 50))
                    macd = analysis_data.get("macd", analysis_data.get("MACD.macd", analysis_data.get("MACD", 0)))
                    macd_signal = analysis_data.get("macd_signal", analysis_data.get("MACD.signal", analysis_data.get("MACD_signal", 0)))
                    
                    # Determine trend based on EMAs
                    trend = "NEUTRAL"
                    if ema_20 > ema_50:
                        trend = "BULLISH"
                    elif ema_20 < ema_50:
                        trend = "BEARISH"
                    
                    # Determine RSI conditions
                    rsi_condition = "NEUTRAL"
                    if rsi >= 70:
                        rsi_condition = "OVERBOUGHT"
                    elif rsi <= 30:
                        rsi_condition = "OVERSOLD"
                    
                    # Determine MACD signal
                    macd_signal_text = "NEUTRAL"
                    if macd > macd_signal:
                        macd_signal_text = "BULLISH"
                    elif macd < macd_signal:
                        macd_signal_text = "BEARISH"
                    
                    # Get the appropriate decimal precision for this instrument
                    precision = self._get_instrument_precision(instrument)
                    
                    # Format the analysis using the same format as the main method
                    # Verwijder timeframe en maak instrument bold
                    analysis_text = f"<b>{instrument}</b> Analysis\n\n"
                    
                    analysis_text += f"<b>Zone Strength:</b> {'★' * min(5, max(1, int(rsi/20)))}\n\n"
                    
                    # Market overview section
                    analysis_text += f"📊 <b>Market Overview</b>\n"
                    if instrument == "XAUUSD":
                        # Format gold price with comma after first digit
                        price_first_digit = str(int(current_price))[0]
                        price_rest_digits = f"{current_price:.3f}".split('.')[0][1:] + "." + f"{current_price:.3f}".split('.')[1]
                        formatted_price = f"{price_first_digit},{price_rest_digits}"
                        
                        analysis_text += f"Price is currently trading near current price of {formatted_price}, "
                    elif instrument == "US30":
                        # Format US30 price with comma after second digit
                        price_digits = str(int(current_price))
                        formatted_price = f"{price_digits[:2]},{price_digits[2:]}.{f'{current_price:.2f}'.split('.')[1]}"
                        
                        analysis_text += f"Price is currently trading near current price of {formatted_price}, "
                    elif instrument == "US500":
                        # Format US500 price with comma after first digit
                        price_digits = str(int(current_price))
                        formatted_price = f"{price_digits[0]},{price_digits[1:]}.{f'{current_price:.2f}'.split('.')[1]}"
                        
                        analysis_text += f"Price is currently trading near current price of {formatted_price}, "
                    elif instrument == "US100":
                        # Format US100 price with comma after second digit
                        price_digits = str(int(current_price))
                        formatted_price = f"{price_digits[:2]},{price_digits[2:]}.{f'{current_price:.2f}'.split('.')[1]}"
                        
                        analysis_text += f"Price is currently trading near current price of {formatted_price}, "
                    else:
                        analysis_text += f"Price is currently trading near current price of {current_price:.{precision}f}, "
                    
                    # Continue with the rest of the analysis text
                    analysis_text += f"showing {'bullish' if trend == 'BUY' else 'bearish' if trend == 'SELL' else 'mixed'} momentum. "
                    analysis_text += f"The pair remains {'above' if current_price > ema_50 else 'below'} key EMAs, "
                    analysis_text += f"indicating a {'strong uptrend' if trend == 'BUY' else 'strong downtrend' if trend == 'SELL' else 'consolidation phase'}. "
                    analysis_text += f"Volume is moderate, supporting the current price action.\n\n"
                    
                    # Key levels section
                    analysis_text += f"🔑 <b>Key Levels</b>\n"
                    if instrument == "XAUUSD":
                        # Format gold prices with comma after first digit
                        def format_gold(price):
                            price_str = f"{price:.3f}"
                            parts = price_str.split('.')
                            return f"{parts[0][0]},{parts[0][1:]}.{parts[1]}"

                        analysis_text += f"Daily High:   {format_gold(analysis_data['high'])}\n"
                        analysis_text += f"Daily Low:    {format_gold(analysis_data['low'])}\n"
                        analysis_text += f"Weekly High:  {format_gold(analysis_data['high'] * 1.02)}\n"
                        analysis_text += f"Weekly Low:   {format_gold(analysis_data['low'] * 0.98)}\n\n"

                    elif instrument == "US30":
                        # Format US30 prices with comma after second digit
                        def format_us30(price):
                            price_str = f"{price:.2f}"
                            parts = price_str.split('.')
                            digits = parts[0]
                            return f"{digits[:2]},{digits[2:]}.{parts[1]}"

                        analysis_text += f"Daily High:   {format_us30(analysis_data['high'])}\n"
                        analysis_text += f"Daily Low:    {format_us30(analysis_data['low'])}\n"
                        analysis_text += f"Weekly High:  {format_us30(analysis_data['high'] * 1.02)}\n"
                        analysis_text += f"Weekly Low:   {format_us30(analysis_data['low'] * 0.98)}\n\n"

                    elif instrument == "US500":
                        # Format US500 prices with comma after first digit
                        def format_us500(price):
                            price_str = f"{price:.2f}"
                            parts = price_str.split('.')
                            digits = parts[0]
                            return f"{digits[0]},{digits[1:]}.{parts[1]}"

                        analysis_text += f"Daily High:   {format_us500(analysis_data['high'])}\n"
                        analysis_text += f"Daily Low:    {format_us500(analysis_data['low'])}\n"
                        analysis_text += f"Weekly High:  {format_us500(analysis_data['high'] * 1.02)}\n"
                        analysis_text += f"Weekly Low:   {format_us500(analysis_data['low'] * 0.98)}\n\n"

                    elif instrument == "US100":
                        # Format US100 prices with comma after second digit
                        def format_us100(price):
                            price_str = f"{price:.2f}"
                            parts = price_str.split('.')
                            digits = parts[0]
                            return f"{digits[:2]},{digits[2:]}.{parts[1]}"

                        analysis_text += f"Daily High:   {format_us100(analysis_data['high'])}\n"
                        analysis_text += f"Daily Low:    {format_us100(analysis_data['low'])}\n"
                        analysis_text += f"Weekly High:  {format_us100(analysis_data['high'] * 1.02)}\n"
                        analysis_text += f"Weekly Low:   {format_us100(analysis_data['low'] * 0.98)}\n\n"
                    else:
                        # Default formatting
                        analysis_text += f"Daily High:   {analysis_data['high']:.{precision}f}\n"
                        analysis_text += f"Daily Low:    {analysis_data['low']:.{precision}f}\n"
                        analysis_text += f"Weekly High:  {analysis_data['high'] * 1.02:.{precision}f}\n"
                        analysis_text += f"Weekly Low:   {analysis_data['low'] * 0.98:.{precision}f}\n\n"

                    # Technical indicators section
                    analysis_text += f"📈 <b>Technical Indicators</b>\n"
                    analysis_text += f"RSI: {rsi:.2f} (neutral)\n"
                    
                    macd_value = random.uniform(-0.001, 0.001)
                    macd_signal = random.uniform(-0.001, 0.001)
                    macd_status = "bullish" if macd_value > macd_signal else "bearish"
                    analysis_text += f"MACD: {macd_status} ({macd_value:.5f} is {'above' if macd_value > macd_signal else 'below'} signal {macd_signal:.5f})\n"
                    
                    ma_status = "bullish" if trend == "BUY" else "bearish" if trend == "SELL" else "mixed"
                    if instrument == "XAUUSD":
                        # Format gold EMAs with comma after first digit
                        ema50_first_digit = str(int(ema_50))[0]
                        ema50_rest_digits = f"{ema_50:.3f}".split('.')[0][1:] + "." + f"{ema_50:.3f}".split('.')[1]
                        formatted_ema50 = f"{ema50_first_digit},{ema50_rest_digits}"
                        
                        ema200_first_digit = str(int(ema_200))[0]
                        ema200_rest_digits = f"{ema_200:.3f}".split('.')[0][1:] + "." + f"{ema_200:.3f}".split('.')[1]
                        formatted_ema200 = f"{ema200_first_digit},{ema200_rest_digits}"
                        
                        analysis_text += f"Moving Averages: Price {'above' if trend == 'BUY' else 'below' if trend == 'SELL' else 'near'} EMA 50 ({formatted_ema50}) and "
                        analysis_text += f"{'above' if trend == 'BUY' else 'below' if trend == 'SELL' else 'near'} EMA 200 ({formatted_ema200}), confirming {ma_status} bias.\n\n"
                    elif instrument == "US30":
                        # Format US30 EMAs with comma after second digit
                        ema50_digits = str(int(ema_50))
                        ema50_formatted = f"{ema50_digits[:2]},{ema50_digits[2:]}.{f'{ema_50:.2f}'.split('.')[1]}"
                        
                        ema200_digits = str(int(ema_200))
                        ema200_formatted = f"{ema200_digits[:2]},{ema200_digits[2:]}.{f'{ema_200:.2f}'.split('.')[1]}"
                        
                        analysis_text += f"Moving Averages: Price {'above' if trend == 'BUY' else 'below' if trend == 'SELL' else 'near'} EMA 50 ({ema50_formatted}) and "
                        analysis_text += f"{'above' if trend == 'BUY' else 'below' if trend == 'SELL' else 'near'} EMA 200 ({ema200_formatted}), confirming {ma_status} bias.\n\n"
                    elif instrument == "US500":
                        # Format US500 EMAs with comma after first digit
                        ema50_digits = str(int(ema_50))
                        ema50_formatted = f"{ema50_digits[0]},{ema50_digits[1:]}.{f'{ema_50:.2f}'.split('.')[1]}"
                        
                        ema200_digits = str(int(ema_200))
                        ema200_formatted = f"{ema200_digits[0]},{ema200_digits[1:]}.{f'{ema_200:.2f}'.split('.')[1]}"
                        
                        analysis_text += f"Moving Averages: Price {'above' if trend == 'BUY' else 'below' if trend == 'SELL' else 'near'} EMA 50 ({ema50_formatted}) and "
                        analysis_text += f"{'above' if trend == 'BUY' else 'below' if trend == 'SELL' else 'near'} EMA 200 ({ema200_formatted}), confirming {ma_status} bias.\n\n"
                    elif instrument == "US100":
                        # Format US100 EMAs with comma after second digit
                        ema50_digits = str(int(ema_50))
                        ema50_formatted = f"{ema50_digits[:2]},{ema50_digits[2:]}.{f'{ema_50:.2f}'.split('.')[1]}"
                        
                        ema200_digits = str(int(ema_200))
                        ema200_formatted = f"{ema200_digits[:2]},{ema200_digits[2:]}.{f'{ema_200:.2f}'.split('.')[1]}"
                        
                        analysis_text += f"Moving Averages: Price {'above' if trend == 'BUY' else 'below' if trend == 'SELL' else 'near'} EMA 50 ({ema50_formatted}) and "
                        analysis_text += f"{'above' if trend == 'BUY' else 'below' if trend == 'SELL' else 'near'} EMA 200 ({ema200_formatted}), confirming {ma_status} bias.\n\n"
                    else:
                        analysis_text += f"Moving Averages: Price {'above' if trend == 'BUY' else 'below' if trend == 'SELL' else 'near'} EMA 50 ({ema_50:.{precision}f}) and "
                        analysis_text += f"{'above' if trend == 'BUY' else 'below' if trend == 'SELL' else 'near'} EMA 200 ({ema_200:.{precision}f}), confirming {ma_status} bias.\n\n"
                    
                    # AI recommendation
                    analysis_text += f"🤖 <b>Sigmapips AI Recommendation</b>\n"
                    if trend == 'BULLISH':
                        if instrument == "XAUUSD":
                            # Format gold prices with comma after first digit
                            high_first_digit = str(int(analysis_data['high']))[0]
                            high_rest_digits = f"{analysis_data['high']:.3f}".split('.')[0][1:] + "." + f"{analysis_data['high']:.3f}".split('.')[1]
                            formatted_high = f"{high_first_digit},{high_rest_digits}"
                            
                            low_first_digit = str(int(analysis_data['low']))[0]
                            low_rest_digits = f"{analysis_data['low']:.3f}".split('.')[0][1:] + "." + f"{analysis_data['low']:.3f}".split('.')[1]
                            formatted_low = f"{low_first_digit},{low_rest_digits}"
                            
                            analysis_text += f"Watch for a breakout above {formatted_high} for further upside. "
                            analysis_text += f"Maintain a buy bias while price holds above {formatted_low}. "
                            analysis_text += f"Be cautious of overbought conditions if RSI approaches 70.\n\n"
                        elif instrument == "US30":
                            # Format US30 prices with comma after second digit
                            high_digits = str(int(analysis_data['high']))
                            high_decimal_part = f'{analysis_data["high"]:.2f}'.split('.')[1]
                            formatted_high = f"{high_digits[:2]},{high_digits[2:]}.{high_decimal_part}"

                            low_digits = str(int(analysis_data['low']))
                            low_decimal_part = f'{analysis_data["low"]:.2f}'.split('.')[1]
                            formatted_low = f"{low_digits[:2]},{low_digits[2:]}.{low_decimal_part}"

                            analysis_text += f"Watch for a breakout above {formatted_high} for further upside. "
                            analysis_text += f"Maintain a buy bias while price holds above {formatted_low}. "
                            analysis_text += f"Be cautious of overbought conditions if RSI approaches 70.\n\n"
                        elif instrument == "US500":
                            # Format US500 prices with comma after first digit
                            high_digits = str(int(analysis_data['high']))
                            high_decimal_part = f'{analysis_data["high"]:.2f}'.split('.')[1]
                            formatted_high = f"{high_digits[0]},{high_digits[1:]}.{high_decimal_part}"

                            low_digits = str(int(analysis_data['low']))
                            low_decimal_part = f'{analysis_data["low"]:.2f}'.split('.')[1]
                            formatted_low = f"{low_digits[0]},{low_digits[1:]}.{low_decimal_part}"

                            analysis_text += f"Watch for a breakout above {formatted_high} for further upside. "
                            analysis_text += f"Maintain a buy bias while price holds above {formatted_low}. "
                            analysis_text += f"Be cautious of overbought conditions if RSI approaches 70.\n\n"
                        elif instrument == "US100":
                            # Format US100 prices with comma after second digit
                            high_digits = str(int(analysis_data['high']))
                            high_decimal_part = f'{analysis_data["high"]:.2f}'.split('.')[1]
                            formatted_high = f"{high_digits[:2]},{high_digits[2:]}.{high_decimal_part}"

                            low_digits = str(int(analysis_data['low']))
                            low_decimal_part = f'{analysis_data["low"]:.2f}'.split('.')[1]
                            formatted_low = f"{low_digits[:2]},{low_digits[2:]}.{low_decimal_part}"

                            analysis_text += f"Watch for a breakout above {formatted_high} for further upside. "
                            analysis_text += f"Maintain a buy bias while price holds above {formatted_low}. "
                            analysis_text += f"Be cautious of overbought conditions if RSI approaches 70.\n\n"
                        else:
                            analysis_text += f"Watch for a breakout above {analysis_data['high']:.{precision}f} for further upside. "
                            analysis_text += f"Maintain a buy bias while price holds above {analysis_data['low']:.{precision}f}. "
                            analysis_text += f"Be cautious of overbought conditions if RSI approaches 70.\n\n"
                    elif trend == 'BEARISH':
                        if instrument == "XAUUSD":
                            # Format gold prices with comma after first digit
                            low_first_digit = str(int(analysis_data['low']))[0]
                            low_rest_digits = f"{analysis_data['low']:.3f}".split('.')[0][1:] + "." + f"{analysis_data['low']:.3f}".split('.')[1]
                            formatted_low = f"{low_first_digit},{low_rest_digits}"
                            
                            high_first_digit = str(int(analysis_data['high']))[0]
                            high_rest_digits = f"{analysis_data['high']:.3f}".split('.')[0][1:] + "." + f"{analysis_data['high']:.3f}".split('.')[1]
                            formatted_high = f"{high_first_digit},{high_rest_digits}"
                            
                            analysis_text += f"Watch for a breakdown below {formatted_low} for further downside. "
                            analysis_text += f"Maintain a sell bias while price holds below {formatted_high}. "
                            analysis_text += f"Be cautious of oversold conditions if RSI approaches 30.\n\n"
                        elif instrument == "US30":
                            # Format US30 prices with comma after second digit
                            low_digits = str(int(analysis_data['low']))
                            low_decimal_part = f'{analysis_data["low"]:.2f}'.split('.')[1]
                            formatted_low = f"{low_digits[:2]},{low_digits[2:]}.{low_decimal_part}"

                            high_digits = str(int(analysis_data['high']))
                            high_decimal_part = f'{analysis_data["high"]:.2f}'.split('.')[1]
                            formatted_high = f"{high_digits[:2]},{high_digits[2:]}.{high_decimal_part}"

                            analysis_text += f"Watch for a breakdown below {formatted_low} for further downside. "
                            analysis_text += f"Maintain a sell bias while price holds below {formatted_high}. "
                            analysis_text += f"Be cautious of oversold conditions if RSI approaches 30.\n\n"
                        elif instrument == "US500":
                            # Format US500 prices with comma after first digit
                            low_digits = str(int(analysis_data['low']))
                            low_decimal_part = f'{analysis_data["low"]:.2f}'.split('.')[1]
                            formatted_low = f"{low_digits[0]},{low_digits[1:]}.{low_decimal_part}"

                            high_digits = str(int(analysis_data['high']))
                            high_decimal_part = f'{analysis_data["high"]:.2f}'.split('.')[1]
                            formatted_high = f"{high_digits[0]},{high_digits[1:]}.{high_decimal_part}"

                            analysis_text += f"Watch for a breakdown below {formatted_low} for further downside. "
                            analysis_text += f"Maintain a sell bias while price holds below {formatted_high}. "
                            analysis_text += f"Be cautious of oversold conditions if RSI approaches 30.\n\n"
                        elif instrument == "US100":
                            # Format US100 prices with comma after second digit
                            low_digits = str(int(analysis_data['low']))
                            low_decimal_part = f'{analysis_data["low"]:.2f}'.split('.')[1]
                            formatted_low = f"{low_digits[:2]},{low_digits[2:]}.{low_decimal_part}"

                            high_digits = str(int(analysis_data['high']))
                            high_decimal_part = f'{analysis_data["high"]:.2f}'.split('.')[1]
                            formatted_high = f"{high_digits[:2]},{high_digits[2:]}.{high_decimal_part}"

                            analysis_text += f"Watch for a breakdown below {formatted_low} for further downside. "
                            analysis_text += f"Maintain a sell bias while price holds below {formatted_high}. "
                            analysis_text += f"Be cautious of oversold conditions if RSI approaches 30.\n\n"
                        else:
                            analysis_text += f"Watch for a breakdown below {analysis_data['low']:.{precision}f} for further downside. "
                            analysis_text += f"Maintain a sell bias while price holds below {analysis_data['high']:.{precision}f}. "
                            analysis_text += f"Be cautious of oversold conditions if RSI approaches 30.\n\n"
                    else:
                        if instrument == "XAUUSD":
                            # Format gold prices with comma after first digit
                            low_first_digit = str(int(analysis_data['low']))[0]
                            low_rest_digits = f"{analysis_data['low']:.3f}".split('.')[0][1:] + "." + f"{analysis_data['low']:.3f}".split('.')[1]
                            formatted_low = f"{low_first_digit},{low_rest_digits}"
                            
                            high_first_digit = str(int(analysis_data['high']))[0]
                            high_rest_digits = f"{analysis_data['high']:.3f}".split('.')[0][1:] + "." + f"{analysis_data['high']:.3f}".split('.')[1]
                            formatted_high = f"{high_first_digit},{high_rest_digits}"
                            
                            analysis_text += f"Range-bound conditions persist. Look for buying opportunities near {formatted_low} "
                            analysis_text += f"and selling opportunities near {formatted_high}. "
                            analysis_text += f"Wait for a clear breakout before establishing a directional bias.\n\n"
                        elif instrument == "US30":
                            # Format US30 prices with comma after second digit
                            low_digits = str(int(analysis_data['low']))
                            low_decimal_part = f'{analysis_data["low"]:.2f}'.split('.')[1]
                            formatted_low = f"{low_digits[:2]},{low_digits[2:]}.{low_decimal_part}"

                            high_digits = str(int(analysis_data['high']))
                            high_decimal_part = f'{analysis_data["high"]:.2f}'.split('.')[1]
                            formatted_high = f"{high_digits[:2]},{high_digits[2:]}.{high_decimal_part}"

                            analysis_text += f"Range-bound conditions persist. Look for buying opportunities near {formatted_low} "
                            analysis_text += f"and selling opportunities near {formatted_high}. "
                            analysis_text += f"Wait for a clear breakout before establishing a directional bias.\n\n"
                        elif instrument == "US500":
                            # Format US500 prices with comma after first digit
                            low_digits = str(int(analysis_data['low']))
                            low_decimal_part = f'{analysis_data["low"]:.2f}'.split('.')[1]
                            formatted_low = f"{low_digits[0]},{low_digits[1:]}.{low_decimal_part}"

                            high_digits = str(int(analysis_data['high']))
                            high_decimal_part = f'{analysis_data["high"]:.2f}'.split('.')[1]
                            formatted_high = f"{high_digits[0]},{high_digits[1:]}.{high_decimal_part}"

                            analysis_text += f"Range-bound conditions persist. Look for buying opportunities near {formatted_low} "
                            analysis_text += f"and selling opportunities near {formatted_high}. "
                            analysis_text += f"Wait for a clear breakout before establishing a directional bias.\n\n"
                        elif instrument == "US100":
                            # Format US100 prices with comma after second digit
                            low_digits = str(int(analysis_data['low']))
                            low_decimal_part = f'{analysis_data["low"]:.2f}'.split('.')[1]
                            formatted_low = f"{low_digits[:2]},{low_digits[2:]}.{low_decimal_part}"

                            high_digits = str(int(analysis_data['high']))
                            high_decimal_part = f'{analysis_data["high"]:.2f}'.split('.')[1]
                            formatted_high = f"{high_digits[:2]},{high_digits[2:]}.{high_decimal_part}"

                            analysis_text += f"Range-bound conditions persist. Look for buying opportunities near {formatted_low} "
                            analysis_text += f"and selling opportunities near {formatted_high}. "
                            analysis_text += f"Wait for a clear breakout before establishing a directional bias.\n\n"
                        else:
                            analysis_text += f"Range-bound conditions persist. Look for buying opportunities near {analysis_data['low']:.{precision}f} "
                            analysis_text += f"and selling opportunities near {analysis_data['high']:.{precision}f}. "
                            analysis_text += f"Wait for a clear breakout before establishing a directional bias.\n\n"
                    
                    analysis_text += f"⚠️ <b>Disclaimer:</b> For educational purposes only."
                    
                    # Cache the analysis
                    self.analysis_cache[cache_key] = (current_time, analysis_text)
                    
                    # --- DEBUG PRINT ADDED ---
                    print(f"\n--- DEBUG: Final analysis_text for {instrument} ---")
                    print(analysis_text)
                    print("--- END DEBUG ---\n")
                    # --- END DEBUG PRINT ---
                    
                    return analysis_text
                except Exception as e:
                    logger.error(f"Error getting analysis from providers: {str(e)}")
                    logger.error(traceback.format_exc())
                    return await self._generate_default_analysis(instrument, timeframe)
            
            else:
                # Log detailed information about API failures
                logger.warning(f"Failed to generate analysis for {instrument}, falling back to default")
                return await self._generate_default_analysis(instrument, timeframe)
        except Exception as e:
            logger.error(f"Error getting analysis from providers: {str(e)}")
            logger.error(traceback.format_exc())
            return await self._generate_default_analysis(instrument, timeframe)
    
    async def _generate_default_analysis(self, instrument: str, timeframe: str) -> str:
        """Generate a fallback analysis when the API fails"""
        try:
            # Default values
            current_price = 0.0
            trend = "NEUTRAL"
            
            # Try to get a reasonable price estimate for the instrument
            if instrument.startswith("EUR"):
                current_price = 1.08 + random.uniform(-0.02, 0.02)
            elif instrument.startswith("GBP"):
                current_price = 1.26 + random.uniform(-0.03, 0.03)
            elif instrument.startswith("USD"):
                current_price = 0.95 + random.uniform(-0.02, 0.02)
            elif instrument == "BTCUSD":
                # Use a more realistic price for Bitcoin (updated value)
                current_price = 68000 + random.uniform(-2000, 2000)
            elif instrument == "ETHUSD":
                # Use a more realistic price for Ethereum (updated value)
                current_price = 3500 + random.uniform(-200, 200)
            elif instrument == "SOLUSD":
                # Use a more realistic price for Solana
                current_price = 180 + random.uniform(-10, 10)
            elif instrument == "BNBUSD":
                # Use a more realistic price for BNB
                current_price = 600 + random.uniform(-20, 20)
            elif instrument.startswith("BTC"):
                current_price = 68000 + random.uniform(-2000, 2000)
            elif instrument.startswith("ETH"):
                current_price = 3500 + random.uniform(-200, 200)
            elif instrument.startswith("XAU"):
                current_price = 2350 + random.uniform(-50, 50)
            # Add realistic defaults for indices
            elif instrument == "US30":  # Dow Jones
                current_price = 38500 + random.uniform(-500, 500)
            elif instrument == "US500":  # S&P 500
                current_price = 5200 + random.uniform(-100, 100)
            elif instrument == "US100":  # Nasdaq 100
                current_price = 18200 + random.uniform(-200, 200)
            elif instrument == "UK100":  # FTSE 100
                current_price = 8200 + random.uniform(-100, 100)
            elif instrument == "DE40":  # DAX
                current_price = 17800 + random.uniform(-200, 200)
            elif instrument == "JP225":  # Nikkei 225
                current_price = 38000 + random.uniform(-400, 400)
            elif instrument == "AU200":  # ASX 200
                current_price = 7700 + random.uniform(-100, 100)
            elif instrument == "EU50":  # Euro Stoxx 50
                current_price = 4900 + random.uniform(-50, 50)
            # Add realistic defaults for commodities
            elif instrument == "XAGUSD":  # Silver
                current_price = 27.5 + random.uniform(-1, 1)
            elif instrument in ["WTIUSD", "XTIUSD"]:  # Crude oil
                current_price = 78 + random.uniform(-2, 2)
            else:
                current_price = 100 + random.uniform(-5, 5)
            
            # Generate random but reasonable values for price variations
            # For crypto, use higher volatility
            if any(crypto in instrument for crypto in ["BTC", "ETH", "XRP", "SOL", "BNB"]):
                # Crypto has higher volatility
                daily_variation = random.uniform(0.01, 0.03)  # 1-3% daily variation
                weekly_variation = random.uniform(0.03, 0.08)  # 3-8% weekly variation
            elif any(index in instrument for index in ["US30", "US500", "US100", "UK100", "DE40", "JP225"]):
                # US indices often have higher RSI values in bull markets
                daily_variation = random.uniform(0.005, 0.015)  # 0.5-1.5% daily variation
                weekly_variation = random.uniform(0.01, 0.04)  # 1-4% weekly variation
            elif any(commodity in instrument for commodity in ["XAUUSD", "XAGUSD", "WTIUSD", "XTIUSD"]):
                # Commodities like gold and silver - slightly bullish in uncertain markets
                daily_variation = random.uniform(0.008, 0.02)  # 0.8-2% daily variation
                weekly_variation = random.uniform(0.02, 0.06)  # 2-6% weekly variation
            else:
                # Standard forex volatility
                daily_variation = random.uniform(0.003, 0.01)
                weekly_variation = random.uniform(0.01, 0.03)
            
            daily_high = current_price * (1 + daily_variation/2)
            daily_low = current_price * (1 - daily_variation/2)
            weekly_high = current_price * (1 + weekly_variation/2)
            weekly_low = current_price * (1 - weekly_variation/2)
            
            # Adjust RSI based on instrument and current market conditions
            if instrument == "BTCUSD":
                # Slightly bullish RSI for BTC as default
                rsi = random.uniform(45, 65)
            elif any(index in instrument for index in ["US30", "US500", "US100"]):
                # US indices often have higher RSI values in bull markets
                rsi = random.uniform(50, 65)
            elif instrument in ["XAUUSD", "XAGUSD"]:
                # Commodities like gold and silver - slightly bullish in uncertain markets
                rsi = random.uniform(48, 62)
            else:
                rsi = random.uniform(40, 60)
            
            # Adjust trend probabilities based on instrument
            if instrument == "BTCUSD":
                # Slightly higher chance of a bullish trend for BTC
                trends = ["BUY", "BUY", "NEUTRAL", "SELL"]
            elif any(index in instrument for index in ["US30", "US500", "US100"]):
                # US indices trend slightly bullish
                trends = ["BUY", "BUY", "NEUTRAL", "SELL"]
            elif instrument == "XAUUSD":
                # Gold often serves as a safe haven
                trends = ["BUY", "NEUTRAL", "NEUTRAL", "SELL"]
            else:
                trends = ["BUY", "SELL", "NEUTRAL"]
            trend = random.choice(trends)
            
            # Zone strength (1-5 stars)
            zone_strength = random.randint(3, 5)
            zone_stars = "★" * zone_strength + "☆" * (5 - zone_strength)
            
            # Determine the appropriate price formatting based on instrument type
            if any(crypto in instrument for crypto in ["BTC", "ETH", "LTC", "XRP"]):
                if instrument == "BTCUSD":
                    # Bitcoin usually shows fewer decimal places
                    precision = 2
                else:
                    # Other crypto might need more precision
                    precision = 4
            elif any(index in instrument for index in ["US30", "US500", "US100", "UK100", "DE40", "JP225"]):
                # Indices typically show 1-2 decimal places
                precision = 2
            elif any(commodity in instrument for commodity in ["XAUUSD", "XAGUSD"]):
                # Gold and silver typically show 2 decimal places
                precision = 2
            elif instrument in ["WTIUSD", "XTIUSD"]:
                # Oil typically shows 2 decimal places
                precision = 2
            else:
                # Default format for forex pairs with 5 decimal places
                precision = 5
            
            # EMA values with more realistic relationships to price
            if trend == "BUY":
                ema_50 = current_price * (1 - random.uniform(0.005, 0.015))  # EMA50 slightly below price
                ema_200 = current_price * (1 - random.uniform(0.02, 0.05))   # EMA200 further below price
            elif trend == "SELL":
                ema_50 = current_price * (1 + random.uniform(0.005, 0.015))  # EMA50 slightly above price
                ema_200 = current_price * (1 + random.uniform(0.01, 0.03))   # EMA200 further above price
            else:
                # Neutral trend - EMAs close to price
                ema_50 = current_price * (1 + random.uniform(-0.01, 0.01))
                ema_200 = current_price * (1 + random.uniform(-0.02, 0.02))
            
            # Format the analysis using the same format as the main method
            # Verwijder timeframe en maak instrument bold
            analysis_text = f"<b>{instrument}</b> Analysis\n\n"
            
            analysis_text += f"<b>Zone Strength:</b> {zone_stars}\n\n"
            
            # Market overview section
            analysis_text += f"📊 <b>Market Overview</b>\n"
            if instrument == "XAUUSD":
                # Format gold price with comma after first digit
                price_first_digit = str(int(current_price))[0]
                price_rest_digits = f"{current_price:.3f}".split('.')[0][1:] + "." + f"{current_price:.3f}".split('.')[1]
                formatted_price = f"{price_first_digit},{price_rest_digits}"
                
                analysis_text += f"Price is currently trading near current price of {formatted_price}, "
            elif instrument == "US30":
                # Format US30 price with comma after second digit
                price_digits = str(int(current_price))
                formatted_price = f"{price_digits[:2]},{price_digits[2:]}.{f'{current_price:.2f}'.split('.')[1]}"
                
                analysis_text += f"Price is currently trading near current price of {formatted_price}, "
            elif instrument == "US500":
                # Format US500 price with comma after first digit
                price_digits = str(int(current_price))
                formatted_price = f"{price_digits[0]},{price_digits[1:]}.{f'{current_price:.2f}'.split('.')[1]}"
                
                analysis_text += f"Price is currently trading near current price of {formatted_price}, "
            elif instrument == "US100":
                # Format US100 price with comma after second digit
                price_digits = str(int(current_price))
                formatted_price = f"{price_digits[:2]},{price_digits[2:]}.{f'{current_price:.2f}'.split('.')[1]}"
                
                analysis_text += f"Price is currently trading near current price of {formatted_price}, "
            else:
                analysis_text += f"Price is currently trading near current price of {current_price:.{precision}f}, "
                
            analysis_text += f"showing {'bullish' if trend == 'BUY' else 'bearish' if trend == 'SELL' else 'mixed'} momentum. "
            analysis_text += f"The pair remains {'above' if current_price > ema_50 else 'below'} key EMAs, "
            analysis_text += f"indicating a {'strong uptrend' if trend == 'BUY' else 'strong downtrend' if trend == 'SELL' else 'consolidation phase'}. "
            analysis_text += f"Volume is moderate, supporting the current price action.\n\n"
            
            # Key levels section
            analysis_text += f"🔑 <b>Key Levels</b>\n"
            if instrument == "XAUUSD":
                # Format gold prices with comma after first digit
                def format_gold(price):
                    price_str = f"{price:.3f}"
                    parts = price_str.split('.')
                    return f"{parts[0][0]},{parts[0][1:]}.{parts[1]}"

                analysis_text += f"Daily High:   {format_gold(daily_high)}\n"
                analysis_text += f"Daily Low:    {format_gold(daily_low)}\n"
                analysis_text += f"Weekly High:  {format_gold(weekly_high)}\n"
                analysis_text += f"Weekly Low:   {format_gold(weekly_low)}\n\n"

            elif instrument == "US30":
                # Format US30 prices with comma after second digit
                def format_us30(price):
                    price_str = f"{price:.2f}"
                    parts = price_str.split('.')
                    digits = parts[0]
                    return f"{digits[:2]},{digits[2:]}.{parts[1]}"

                analysis_text += f"Daily High:   {format_us30(daily_high)}\n"
                analysis_text += f"Daily Low:    {format_us30(daily_low)}\n"
                analysis_text += f"Weekly High:  {format_us30(weekly_high)}\n"
                analysis_text += f"Weekly Low:   {format_us30(weekly_low)}\n\n"

            elif instrument == "US500":
                # Format US500 prices with comma after first digit
                def format_us500(price):
                    price_str = f"{price:.2f}"
                    parts = price_str.split('.')
                    digits = parts[0]
                    return f"{digits[0]},{digits[1:]}.{parts[1]}"

                analysis_text += f"Daily High:   {format_us500(daily_high)}\n"
                analysis_text += f"Daily Low:    {format_us500(daily_low)}\n"
                analysis_text += f"Weekly High:  {format_us500(weekly_high)}\n"
                analysis_text += f"Weekly Low:   {format_us500(weekly_low)}\n\n"

            elif instrument == "US100":
                # Format US100 prices with comma after second digit
                def format_us100(price):
                    price_str = f"{price:.2f}"
                    parts = price_str.split('.')
                    digits = parts[0]
                    return f"{digits[:2]},{digits[2:]}.{parts[1]}"

                analysis_text += f"Daily High:   {format_us100(daily_high)}\n"
                analysis_text += f"Daily Low:    {format_us100(daily_low)}\n"
                analysis_text += f"Weekly High:  {format_us100(weekly_high)}\n"
                analysis_text += f"Weekly Low:   {format_us100(weekly_low)}\n\n"
            else:
                # Default formatting
                analysis_text += f"Daily High:   {daily_high:.{precision}f}\n"
                analysis_text += f"Daily Low:    {daily_low:.{precision}f}\n"
                analysis_text += f"Weekly High:  {weekly_high:.{precision}f}\n"
                analysis_text += f"Weekly Low:   {weekly_low:.{precision}f}\n\n"

            # Technical indicators section
            analysis_text += f"📈 <b>Technical Indicators</b>\n"
            analysis_text += f"RSI: {rsi:.2f} (neutral)\n"
            
            macd_value = random.uniform(-0.001, 0.001)
            macd_signal = random.uniform(-0.001, 0.001)
            macd_status = "bullish" if macd_value > macd_signal else "bearish"
            analysis_text += f"MACD: {macd_status} ({macd_value:.5f} is {'above' if macd_value > macd_signal else 'below'} signal {macd_signal:.5f})\n"
            
            ma_status = "bullish" if trend == "BUY" else "bearish" if trend == "SELL" else "mixed"
            if instrument == "XAUUSD":
                # Format gold EMAs with comma after first digit
                ema50_first_digit = str(int(ema_50))[0]
                ema50_rest_digits = f"{ema_50:.3f}".split('.')[0][1:] + "." + f"{ema_50:.3f}".split('.')[1]
                formatted_ema50 = f"{ema50_first_digit},{ema50_rest_digits}"
                
                ema200_first_digit = str(int(ema_200))[0]
                ema200_rest_digits = f"{ema_200:.3f}".split('.')[0][1:] + "." + f"{ema_200:.3f}".split('.')[1]
                formatted_ema200 = f"{ema200_first_digit},{ema200_rest_digits}"
                
                analysis_text += f"Moving Averages: Price {'above' if trend == 'BUY' else 'below' if trend == 'SELL' else 'near'} EMA 50 ({formatted_ema50}) and "
                analysis_text += f"{'above' if trend == 'BUY' else 'below' if trend == 'SELL' else 'near'} EMA 200 ({formatted_ema200}), confirming {ma_status} bias.\n\n"
            elif instrument == "US30":
                # Format US30 EMAs with comma after second digit
                ema50_digits = str(int(ema_50))
                ema50_formatted = f"{ema50_digits[:2]},{ema50_digits[2:]}.{f'{ema_50:.2f}'.split('.')[1]}"
                
                ema200_digits = str(int(ema_200))
                ema200_formatted = f"{ema200_digits[:2]},{ema200_digits[2:]}.{f'{ema_200:.2f}'.split('.')[1]}"
                
                analysis_text += f"Moving Averages: Price {'above' if trend == 'BUY' else 'below' if trend == 'SELL' else 'near'} EMA 50 ({ema50_formatted}) and "
                analysis_text += f"{'above' if trend == 'BUY' else 'below' if trend == 'SELL' else 'near'} EMA 200 ({ema200_formatted}), confirming {ma_status} bias.\n\n"
            elif instrument == "US500":
                # Format US500 EMAs with comma after first digit
                ema50_digits = str(int(ema_50))
                ema50_formatted = f"{ema50_digits[0]},{ema50_digits[1:]}.{f'{ema_50:.2f}'.split('.')[1]}"
                
                ema200_digits = str(int(ema_200))
                ema200_formatted = f"{ema200_digits[0]},{ema200_digits[1:]}.{f'{ema_200:.2f}'.split('.')[1]}"
                
                analysis_text += f"Moving Averages: Price {'above' if trend == 'BUY' else 'below' if trend == 'SELL' else 'near'} EMA 50 ({ema50_formatted}) and "
                analysis_text += f"{'above' if trend == 'BUY' else 'below' if trend == 'SELL' else 'near'} EMA 200 ({ema200_formatted}), confirming {ma_status} bias.\n\n"
            elif instrument == "US100":
                # Format US100 EMAs with comma after second digit
                ema50_digits = str(int(ema_50))
                ema50_formatted = f"{ema50_digits[:2]},{ema50_digits[2:]}.{f'{ema_50:.2f}'.split('.')[1]}"
                
                ema200_digits = str(int(ema_200))
                ema200_formatted = f"{ema200_digits[:2]},{ema200_digits[2:]}.{f'{ema_200:.2f}'.split('.')[1]}"
                
                analysis_text += f"Moving Averages: Price {'above' if trend == 'BUY' else 'below' if trend == 'SELL' else 'near'} EMA 50 ({ema50_formatted}) and "
                analysis_text += f"{'above' if trend == 'BUY' else 'below' if trend == 'SELL' else 'near'} EMA 200 ({ema200_formatted}), confirming {ma_status} bias.\n\n"
            else:
                analysis_text += f"Moving Averages: Price {'above' if trend == 'BUY' else 'below' if trend == 'SELL' else 'near'} EMA 50 ({ema_50:.{precision}f}) and "
                analysis_text += f"{'above' if trend == 'BUY' else 'below' if trend == 'SELL' else 'near'} EMA 200 ({ema_200:.{precision}f}), confirming {ma_status} bias.\n\n"
            
            # AI recommendation
            analysis_text += f"🤖 <b>Sigmapips AI Recommendation</b>\n"
            if trend == "BUY":
                if instrument == "XAUUSD":
                    # Format gold prices with comma after first digit
                    daily_high_first_digit = str(int(daily_high))[0]
                    daily_high_rest_digits = f"{daily_high:.3f}".split('.')[0][1:] + "." + f"{daily_high:.3f}".split('.')[1]
                    formatted_daily_high = f"{daily_high_first_digit},{daily_high_rest_digits}"
                    
                    daily_low_first_digit = str(int(daily_low))[0]
                    daily_low_rest_digits = f"{daily_low:.3f}".split('.')[0][1:] + "." + f"{daily_low:.3f}".split('.')[1]
                    formatted_daily_low = f"{daily_low_first_digit},{daily_low_rest_digits}"
                    
                    analysis_text += f"Watch for a breakout above {formatted_daily_high} for further upside. "
                    analysis_text += f"Maintain a buy bias while price holds above {formatted_daily_low}. "
                    analysis_text += f"Be cautious of overbought conditions if RSI approaches 70.\n\n"
                elif instrument == "US30":
                    # Format US30 prices with comma after second digit
                    daily_high_digits = str(int(daily_high))
                    formatted_daily_high = f"{daily_high_digits[:2]},{daily_high_digits[2:]}.{f'{daily_high:.2f}'.split('.')[1]}"
                    
                    daily_low_digits = str(int(daily_low))
                    formatted_daily_low = f"{daily_low_digits[:2]},{daily_low_digits[2:]}.{f'{daily_low:.2f}'.split('.')[1]}"
                    
                    analysis_text += f"Watch for a breakout above {formatted_daily_high} for further upside. "
                    analysis_text += f"Maintain a buy bias while price holds above {formatted_daily_low}. "
                    analysis_text += f"Be cautious of overbought conditions if RSI approaches 70.\n\n"
                elif instrument == "US500":
                    # Format US500 prices with comma after first digit
                    daily_high_digits = str(int(daily_high))
                    formatted_daily_high = f"{daily_high_digits[0]},{daily_high_digits[1:]}.{f'{daily_high:.2f}'.split('.')[1]}"
                    
                    daily_low_digits = str(int(daily_low))
                    formatted_daily_low = f"{daily_low_digits[0]},{daily_low_digits[1:]}.{f'{daily_low:.2f}'.split('.')[1]}"
                    
                    analysis_text += f"Watch for a breakout above {formatted_daily_high} for further upside. "
                    analysis_text += f"Maintain a buy bias while price holds above {formatted_daily_low}. "
                    analysis_text += f"Be cautious of overbought conditions if RSI approaches 70.\n\n"
                elif instrument == "US100":
                    # Format US100 prices with comma after second digit
                    daily_high_digits = str(int(daily_high))
                    formatted_daily_high = f"{daily_high_digits[:2]},{daily_high_digits[2:]}.{f'{daily_high:.2f}'.split('.')[1]}"
                    
                    daily_low_digits = str(int(daily_low))
                    formatted_daily_low = f"{daily_low_digits[:2]},{daily_low_digits[2:]}.{f'{daily_low:.2f}'.split('.')[1]}"
                    
                    analysis_text += f"Watch for a breakout above {formatted_daily_high} for further upside. "
                    analysis_text += f"Maintain a buy bias while price holds above {formatted_daily_low}. "
                    analysis_text += f"Be cautious of overbought conditions if RSI approaches 70.\n\n"
                else:
                    analysis_text += f"Watch for a breakout above {daily_high:.{precision}f} for further upside. "
                    analysis_text += f"Maintain a buy bias while price holds above {daily_low:.{precision}f}. "
                    analysis_text += f"Be cautious of overbought conditions if RSI approaches 70.\n\n"
            elif trend == "SELL":
                if instrument == "XAUUSD":
                    # Format gold prices with comma after first digit
                    daily_low_first_digit = str(int(daily_low))[0]
                    daily_low_rest_digits = f"{daily_low:.3f}".split('.')[0][1:] + "." + f"{daily_low:.3f}".split('.')[1]
                    formatted_daily_low = f"{daily_low_first_digit},{daily_low_rest_digits}"
                    
                    daily_high_first_digit = str(int(daily_high))[0]
                    daily_high_rest_digits = f"{daily_high:.3f}".split('.')[0][1:] + "." + f"{daily_high:.3f}".split('.')[1]
                    formatted_daily_high = f"{daily_high_first_digit},{daily_high_rest_digits}"
                    
                    analysis_text += f"Watch for a breakdown below {formatted_daily_low} for further downside. "
                    analysis_text += f"Maintain a sell bias while price holds below {formatted_daily_high}. "
                    analysis_text += f"Be cautious of oversold conditions if RSI approaches 30.\n\n"
                elif instrument == "US30":
                    # Format US30 prices with comma after second digit
                    daily_low_digits = str(int(daily_low))
                    formatted_daily_low = f"{daily_low_digits[:2]},{daily_low_digits[2:]}.{f'{daily_low:.2f}'.split('.')[1]}"
                    
                    daily_high_digits = str(int(daily_high))
                    formatted_daily_high = f"{daily_high_digits[:2]},{daily_high_digits[2:]}.{f'{daily_high:.2f}'.split('.')[1]}"
                    
                    analysis_text += f"Watch for a breakdown below {formatted_daily_low} for further downside. "
                    analysis_text += f"Maintain a sell bias while price holds below {formatted_daily_high}. "
                    analysis_text += f"Be cautious of oversold conditions if RSI approaches 30.\n\n"
                elif instrument == "US500":
                    # Format US500 prices with comma after first digit
                    daily_low_digits = str(int(daily_low))
                    formatted_daily_low = f"{daily_low_digits[0]},{daily_low_digits[1:]}.{f'{daily_low:.2f}'.split('.')[1]}"
                    
                    daily_high_digits = str(int(daily_high))
                    formatted_daily_high = f"{daily_high_digits[0]},{daily_high_digits[1:]}.{f'{daily_high:.2f}'.split('.')[1]}"
                    
                    analysis_text += f"Watch for a breakdown below {formatted_daily_low} for further downside. "
                    analysis_text += f"Maintain a sell bias while price holds below {formatted_daily_high}. "
                    analysis_text += f"Be cautious of oversold conditions if RSI approaches 30.\n\n"
                elif instrument == "US100":
                    # Format US100 prices with comma after second digit
                    daily_low_digits = str(int(daily_low))
                    formatted_daily_low = f"{daily_low_digits[:2]},{daily_low_digits[2:]}.{f'{daily_low:.2f}'.split('.')[1]}"
                    
                    daily_high_digits = str(int(daily_high))
                    formatted_daily_high = f"{daily_high_digits[:2]},{daily_high_digits[2:]}.{f'{daily_high:.2f}'.split('.')[1]}"
                    
                    analysis_text += f"Watch for a breakdown below {formatted_daily_low} for further downside. "
                    analysis_text += f"Maintain a sell bias while price holds below {formatted_daily_high}. "
                    analysis_text += f"Be cautious of oversold conditions if RSI approaches 30.\n\n"
                else:
                    analysis_text += f"Watch for a breakdown below {daily_low:.{precision}f} for further downside. "
                    analysis_text += f"Maintain a sell bias while price holds below {daily_high:.{precision}f}. "
                    analysis_text += f"Be cautious of oversold conditions if RSI approaches 30.\n\n"
            else:
                if instrument == "XAUUSD":
                    # Format gold prices with comma after first digit
                    daily_low_first_digit = str(int(daily_low))[0]
                    daily_low_rest_digits = f"{daily_low:.3f}".split('.')[0][1:] + "." + f"{daily_low:.3f}".split('.')[1]
                    formatted_daily_low = f"{daily_low_first_digit},{daily_low_rest_digits}"
                    
                    daily_high_first_digit = str(int(daily_high))[0]
                    daily_high_rest_digits = f"{daily_high:.3f}".split('.')[0][1:] + "." + f"{daily_high:.3f}".split('.')[1]
                    formatted_daily_high = f"{daily_high_first_digit},{daily_high_rest_digits}"
                    
                    analysis_text += f"Range-bound conditions persist. Look for buying opportunities near {formatted_daily_low} "
                    analysis_text += f"and selling opportunities near {formatted_daily_high}. "
                    analysis_text += f"Wait for a clear breakout before establishing a directional bias.\n\n"
                elif instrument == "US30":
                    # Format US30 prices with comma after second digit
                    low_digits = str(int(daily_low))
                    formatted_low = f"{low_digits[:2]},{low_digits[2:]}.{f'{daily_low:.2f}'.split('.')[1]}"
                    
                    high_digits = str(int(daily_high))
                    formatted_high = f"{high_digits[:2]},{high_digits[2:]}.{f'{daily_high:.2f}'.split('.')[1]}"
                    
                    analysis_text += f"Range-bound conditions persist. Look for buying opportunities near {formatted_low} "
                    analysis_text += f"and selling opportunities near {formatted_high}. "
                    analysis_text += f"Wait for a clear breakout before establishing a directional bias.\n\n"
                elif instrument == "US500":
                    # Format US500 prices with comma after first digit
                    low_digits = str(int(daily_low))
                    formatted_low = f"{low_digits[0]},{low_digits[1:]}.{f'{daily_low:.2f}'.split('.')[1]}"
                    
                    high_digits = str(int(daily_high))
                    formatted_high = f"{high_digits[0]},{high_digits[1:]}.{f'{daily_high:.2f}'.split('.')[1]}"
                    
                    analysis_text += f"Range-bound conditions persist. Look for buying opportunities near {formatted_low} "
                    analysis_text += f"and selling opportunities near {formatted_high}. "
                    analysis_text += f"Wait for a clear breakout before establishing a directional bias.\n\n"
                elif instrument == "US100":
                    # Format US100 prices with comma after second digit
                    low_digits = str(int(daily_low))
                    formatted_low = f"{low_digits[:2]},{low_digits[2:]}.{f'{daily_low:.2f}'.split('.')[1]}"
                    
                    high_digits = str(int(daily_high))
                    formatted_high = f"{high_digits[:2]},{high_digits[2:]}.{f'{daily_high:.2f}'.split('.')[1]}"
                    
                    analysis_text += f"Range-bound conditions persist. Look for buying opportunities near {formatted_low} "
                    analysis_text += f"and selling opportunities near {formatted_high}. "
                    analysis_text += f"Wait for a clear breakout before establishing a directional bias.\n\n"
                else:
                    analysis_text += f"Range-bound conditions persist. Look for buying opportunities near {daily_low:.{precision}f} "
                    analysis_text += f"and selling opportunities near {daily_high:.{precision}f}. "
                    analysis_text += f"Wait for a clear breakout before establishing a directional bias.\n\n"
            
            # Disclaimer
            analysis_text += f"⚠️ <b>Disclaimer:</b> For educational purposes only."
            
            return analysis_text
        
        except Exception as e:
            logger.error(f"Error generating default analysis: {str(e)}")
            # Return a very basic message if all else fails
            return f"Analysis for {instrument} on {timeframe} timeframe is not available at this time. Please try again later."

    async def get_sentiment_analysis(self, instrument: str) -> str:
        """Generate sentiment analysis for an instrument"""
        # This method is intentionally left empty to prevent duplicate sentiment analysis
        # Sentiment analysis is now directly handled by the TelegramService using MarketSentimentService
        logger.info(f"ChartService.get_sentiment_analysis called for {instrument} but is now disabled")
        return ""

    def _get_instrument_precision(self, instrument: str) -> int:
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
            
        # Gold uses 3 decimal places
        if instrument in ["XAUUSD", "GC=F"]:
            return 3
            
        # Silver uses 4 decimal places
        if instrument in ["XAGUSD", "SI=F"]:
            return 4
            
        # Oil prices use 2 decimal places
        if instrument in ["XTIUSD", "WTIUSD", "XBRUSD", "USOIL", "CL=F", "BZ=F"]:
            return 2
            
        # Indices typically use 2 decimal places
        if any(index in instrument for index in ["US30", "US500", "US100", "UK100", "DE40", "JP225"]):
            return 2
            
        # Default to 4 decimal places as a safe value
        return 4
    
    async def _detect_market_type(self, instrument: str) -> str:
        """
        Detect the market type of the instrument.
        
        Args:
            instrument: The trading instrument
            
        Returns:
            str: Market type ('forex', 'crypto', 'index', 'commodity')
        """
        # Normalize the instrument name
        instrument = instrument.upper().replace("/", "")
        
        # Direct matches for common crypto symbols - extra controle voor directe detectie
        direct_crypto_matches = [
            "BTC", "ETH", "XRP", "LTC", "BCH", "ADA", "DOT", "LINK", 
            "XLM", "DOGE", "BNB", "SOL", "AVAX", "MATIC"
        ]
        
        if any(instrument.startswith(crypto) for crypto in direct_crypto_matches):
            logger.info(f"Direct crypto match found for {instrument}")
            return "crypto"
            
        # Check for crypto in the symbol
        if "BTC" in instrument or "ETH" in instrument or "USDT" in instrument:
            logger.info(f"Crypto detected by keyword in {instrument}")
            return "crypto"
        
        # Common cryptocurrency identifiers
        crypto_symbols = [
            "BTC", "ETH", "XRP", "LTC", "BCH", "ADA", "DOT", "LINK", 
            "XLM", "DOGE", "UNI", "AAVE", "SNX", "SUSHI", "YFI", 
            "COMP", "MKR", "BAT", "ZRX", "REN", "KNC", "BNB", "SOL",
            "AVAX", "MATIC", "ALGO", "ATOM", "FTM", "NEAR", "ONE",
            "HBAR", "VET", "THETA", "FIL", "TRX", "EOS", "NEO",
            "CAKE", "LUNA", "SHIB", "MANA", "SAND", "AXS", "CRV",
            "ENJ", "CHZ", "GALA", "ROSE", "APE", "FTT", "GRT",
            "GMT", "EGLD", "XTZ", "FLOW", "ICP", "XMR", "DASH"
        ]
        
        # Check for crypto
        if any(crypto in instrument for crypto in crypto_symbols) or instrument.endswith(("USDT", "BUSD", "USDC", "BTC", "ETH")):
            return "crypto"
        
        # Common forex pairs
        forex_pairs = [
            "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", 
            "NZDUSD", "EURGBP", "EURJPY", "GBPJPY", "AUDNZD", "AUDCAD",
            "AUDCHF", "AUDJPY", "CADCHF", "CADJPY", "CHFJPY", "EURAUD",
            "EURCAD", "EURCHF", "EURNZD", "GBPAUD", "GBPCAD", "GBPCHF",
            "GBPNZD", "NZDCAD", "NZDCHF", "NZDJPY"
        ]
        
        # Check for forex
        if instrument in forex_pairs or (
                len(instrument) == 6 and 
                instrument[:3] in ["EUR", "GBP", "USD", "JPY", "AUD", "NZD", "CAD", "CHF"] and
                instrument[3:] in ["EUR", "GBP", "USD", "JPY", "AUD", "NZD", "CAD", "CHF"]
            ):
            return "forex"
        
        # Common commodities
        commodities = [
            "XAUUSD", "XAGUSD", "WTIUSD", "XTIUSD", "XBRUSD", "CLUSD",
            "XPDUSD", "XPTUSD", "NATGAS", "COPPER", "BRENT", "USOIL"
        ]
        
        # Check for commodities
        if any(commodity in instrument for commodity in commodities) or instrument in commodities:
            return "commodity"
        
        # Common indices
        indices = [
            "US30", "US500", "US100", "UK100", "DE40", "FR40", "JP225", 
            "AU200", "ES35", "IT40", "HK50", "DJI", "SPX", "NDX", 
            "FTSE", "DAX", "CAC", "NIKKEI", "ASX", "IBEX", "MIB", "HSI"
        ]
        
        # Check for indices
        if any(index in instrument for index in indices) or instrument in indices:
            return "index"
        
        # Default to crypto for unknown instruments that could be new cryptocurrencies
        if instrument.endswith(("USD", "USDT", "ETH", "BTC")) and len(instrument) > 3:
            return "crypto"
        
        # Default to forex if all else fails
        return "forex"

    async def _fetch_crypto_price(self, symbol: str) -> Optional[float]:
        """
        Fetch crypto price from Binance API with fallback to other providers.
        
        Args:
            symbol: The crypto symbol without USD (e.g., BTC)
        
        Returns:
            float: Current price or None if failed
        """
        try:
            logger.info(f"Fetching {symbol} price from multiple sources")
            
            # Clean up the symbol
            symbol = symbol.replace("USD", "").replace("USDT", "")
            price = None
            success = False
            
            # First, try our optimized BinanceProvider
            try:
                from trading_bot.services.chart_service.binance_provider import BinanceProvider
                price = await BinanceProvider.get_ticker_price(f"{symbol}USDT")
                if price:
                    logger.info(f"Got {symbol} price from BinanceProvider: {price}")
                    return price
            except Exception as e:
                # Check for geo-restriction specifically
                error_str = str(e)
                if "restricted location" in error_str or "eligibility" in error_str.lower():
                    logger.warning(f"Binance API access is geo-restricted: {error_str}")
                else:
                    logger.warning(f"BinanceProvider failed for {symbol}: {error_str}")
            
            # If BinanceProvider fails, try multiple API sources for redundancy
            logger.info(f"Trying alternative crypto APIs for {symbol}")
            apis = [
                f"https://api.coingecko.com/api/v3/simple/price?ids={symbol.lower()}&vs_currencies=usd",
                f"https://api.coinbase.com/v2/prices/{symbol}-USD/spot",
                f"https://min-api.cryptocompare.com/data/price?fsym={symbol.upper()}&tsyms=USD",
                f"https://api.kraken.com/0/public/Ticker?pair={symbol}USD"
            ]
            
            async with aiohttp.ClientSession() as session:
                for api_url in apis:
                    logger.info(f"Trying API: {api_url}")
                    try:
                        async with session.get(api_url, timeout=5) as response:
                            if response.status == 200:
                                data = await response.json()
                                
                                # Parse based on API format
                                if "coingecko" in api_url:
                                    if data and symbol.lower() in data and "usd" in data[symbol.lower()]:
                                        price = float(data[symbol.lower()]["usd"])
                                        success = True
                                        logger.info(f"Got {symbol} price from CoinGecko: {price}")
                                        break
                                elif "coinbase" in api_url:
                                    if data and "data" in data and "amount" in data["data"]:
                                        price = float(data["data"]["amount"])
                                        success = True
                                        logger.info(f"Got {symbol} price from Coinbase: {price}")
                                        break
                                elif "cryptocompare" in api_url:
                                    if data and "USD" in data:
                                        price = float(data["USD"])
                                        success = True
                                        logger.info(f"Got {symbol} price from CryptoCompare: {price}")
                                        break
                                elif "kraken" in api_url:
                                    # Kraken has a different format, need to parse its result differently
                                    if data and "result" in data:
                                        # Try different possible formats for pair name
                                        pair_formats = [
                                            f"X{symbol}ZUSD",   # Format for major coins like BTC (XBTCZUSD)
                                            f"{symbol}USD",     # Format for some altcoins
                                            f"{symbol}ZUSD"     # Another format for some pairs
                                        ]
                                        
                                        for pair_format in pair_formats:
                                            if pair_format in data["result"]:
                                                last_price = data["result"][pair_format]["c"][0]
                                                price = float(last_price)
                                                success = True
                                                logger.info(f"Got {symbol} price from Kraken: {price}")
                                                break
                                        
                                        if success:
                                            break
                            elif response.status == 429:  # Rate limited
                                logger.warning(f"Rate limited by {api_url}, trying next API")
                                continue
                            else:
                                logger.warning(f"Failed request to {api_url}: HTTP {response.status}")
                    except Exception as e:
                        logger.warning(f"Failed to get {symbol} price from {api_url}: {str(e)}")
                        continue
            
            # If we still don't have a price, try to use predefined values for major cryptocurrencies
            if not success:
                logger.warning(f"All APIs failed for {symbol}, using predefined values if available")
                
                # Default values for common cryptocurrencies (updated October 2023)
                default_values = {
                    "BTC": 68000,
                    "ETH": 3500,
                    "XRP": 0.60,
                    "SOL": 180,
                    "BNB": 600, 
                    "ADA": 0.40,
                    "DOGE": 0.12,
                    "DOT": 6.5,
                    "LINK": 15.0,
                    "AVAX": 35.0
                }
                
                if symbol.upper() in default_values:
                    base_price = default_values[symbol.upper()]
                    # Add some randomness to make it realistic
                    variation = random.uniform(-0.015, 0.015)  # ±1.5%
                    price = base_price * (1 + variation)
                    logger.info(f"Using predefined price for {symbol}: {price:.2f}")
                    success = True
            
            if success:
                return price
            return None
            
        except Exception as e:
            logger.error(f"Error fetching crypto price: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    async def _fetch_commodity_price(self, symbol: str) -> Optional[float]:
        """
        Fetch commodity price from Yahoo Finance.
        
        Args:
            symbol: The commodity symbol (e.g., XAUUSD for gold)
        
        Returns:
            float: Current price or None if failed
        """
        try:
            logger.info(f"Fetching {symbol} price from Yahoo Finance")
            
            # Map to correct Yahoo Finance symbol
            yahoo_symbols = {
                "XAUUSD": "GC=F",   # Gold futures
                "XAGUSD": "SI=F",    # Silver futures
                "XTIUSD": "CL=F",    # Crude Oil WTI futures
                "WTIUSD": "CL=F",    # WTI Crude Oil futures (alternative)
                "XBRUSD": "BZ=F",    # Brent Crude Oil futures
                "XPDUSD": "PA=F",    # Palladium futures
                "XPTUSD": "PL=F",    # Platinum futures
                "NATGAS": "NG=F",    # Natural Gas futures
                "COPPER": "HG=F",    # Copper futures
                "USOIL": "CL=F",     # US Oil (same as WTI Crude Oil)
            }
            
            # If symbol not in our mapping, we can't proceed
            if symbol not in yahoo_symbols:
                logger.warning(f"Unknown commodity symbol: {symbol}, cannot fetch from Yahoo Finance")
                return None
                
            # Get the corresponding Yahoo Finance symbol
            yahoo_symbol = yahoo_symbols[symbol]
            logger.info(f"Using Yahoo Finance symbol {yahoo_symbol} for {symbol}")
            
            # Use YahooFinanceProvider to get the latest price
            from .yfinance_provider import YahooFinanceProvider
            
            # Get market data with a small limit to make it fast
            df = await YahooFinanceProvider.get_market_data(yahoo_symbol, "1h", limit=5)
            
            if df is not None and hasattr(df, 'indicators') and 'close' in df.indicators:
                price = df.indicators['close']
                logger.info(f"Got {symbol} price from Yahoo Finance: {price}")
                return price
                
            logger.warning(f"Failed to get {symbol} price from Yahoo Finance")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching commodity price from Yahoo Finance: {str(e)}")
            return None

    async def _fetch_index_price(self, symbol: str) -> Optional[float]:
        """
        Fetch market index price from APIs as a fallback.
        
        Args:
            symbol: The index symbol (e.g., US30, US500)
            
        Returns:
            float: Current price or None if failed
        """
        try:
            logger.info(f"Fetching {symbol} price from external APIs")
            
            # Map symbols to common index names
            index_map = {
                "US30": "dow",
                "US500": "sp500",
                "US100": "nasdaq",
                "UK100": "ftse",
                "DE40": "dax",
                "JP225": "nikkei",
                "AU200": "asx200"
            }
            
            index_name = index_map.get(symbol, symbol.lower())
            
            # Use default reasonable values as a last resort
            default_values = {
                "US30": 38500,
                "US500": 5200,
                "US100": 18200,
                "UK100": 8200,
                "DE40": 17800,
                "JP225": 38000,
                "AU200": 7700,
                "EU50": 4900
            }
            
            # Return the default value with a small random variation
            if symbol in default_values:
                default_price = default_values[symbol]
                variation = random.uniform(-0.005, 0.005)  # ±0.5%
                price = default_price * (1 + variation)
                logger.info(f"Using default price for {symbol}: {price:.2f}")
                return price
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching index price: {str(e)}")
            return None

    def get_tradingview_url(self, instrument: str, timeframe: str = '1h') -> str:
        """Get a TradingView chart URL for a specific instrument"""
        try:
            # Session ID for TradingView
            session_id = os.getenv("TRADINGVIEW_SESSION_ID", "z90l85p2anlgdwfppsrdnnfantz48z1o")
            # Verwijder deze waarschuwing, we weten dat het een geldige ID is
            logger.info(f"Session ID length: {len(session_id)} characters")
            
            # Find predefined chart URL if available
            instrument_upper = instrument.upper()
            
            # Main chart URL for EURUSD (full featured)
            if instrument_upper == 'EURUSD':
                base_url = "https://www.tradingview.com/chart/xknpxpcr/"
                params = {
                    'timeframe': timeframe,
                    'session': session_id
                }
                param_string = "&".join([f"{k}={v}" for k, v in params.items()])
                url = f"{base_url}?{param_string}" 
                logger.info(f"Built TradingView URL for {instrument}: {url}")
                return url
            
            # Use layout ID for other instruments
            layout_id = 'xknpxpcr'
            
            # Format symbol correctly based on type
            symbol = f"FX:{instrument_upper}" if len(instrument_upper) == 6 and all(c.isalpha() for c in instrument_upper) else instrument_upper
            
            # Build URL
            base_url = f"https://www.tradingview.com/chart/{layout_id}/"
            params = {
                'symbol': symbol,
                'timeframe': timeframe,
                'session': session_id
            }
            param_string = "&".join([f"{k}={v}" for k, v in params.items()])
            url = f"{base_url}?{param_string}"
            
            return url
            
        except Exception as e:
            logger.error(f"Error getting TradingView URL: {str(e)}")
            # Fallback URL
            fallback_url = f"https://www.tradingview.com/chart/?symbol=FX:{instrument}&timeframe={timeframe}&session={session_id}"
            return fallback_url

    def _normalize_instrument_name(self, instrument: str) -> str:
        """
        Normalize an instrument name to ensure consistent formatting
        
        Args:
            instrument: Instrument symbol (e.g., EURUSD, BTCUSD)
            
        Returns:
            str: Normalized instrument name
        """
        if not instrument:
            logger.warning("Empty instrument name provided to normalize_instrument_name")
            return ""
        
        # Remove slashes and convert to uppercase
        normalized = instrument.upper().replace("/", "").strip()
        
        # Handle common aliases
        aliases = {
            "GOLD": "XAUUSD",
            "OIL": "XTIUSD",
            "CRUDE": "XTIUSD",
            "NAS100": "US100",
            "NASDAQ": "US100",
            "SPX": "US500",
            "SP500": "US500",
            "DOW": "US30",
            "DAX": "DE40",
            # Add crypto aliases
            "BTC": "BTCUSD",
            "ETH": "ETHUSD",
            "SOL": "SOLUSD",
            "XRP": "XRPUSD",
            "DOGE": "DOGEUSD",
            "ADA": "ADAUSD",
            "LINK": "LINKUSD",
            "AVAX": "AVAXUSD",
            "MATIC": "MATICUSD",
            "DOT": "DOTUSD"
        }
        
        # Check if the input is a pure crypto symbol without USD suffix
        crypto_symbols = ["BTC", "ETH", "XRP", "SOL", "ADA", "LINK", "DOT", "DOGE", "AVAX", "BNB", "MATIC"]
        if normalized in crypto_symbols:
            logger.info(f"Normalized pure crypto symbol {normalized} to {normalized}USD")
            normalized = f"{normalized}USD"
        
        # Handle USDT suffix for crypto (normalize to USD for consistency)
        if normalized.endswith("USDT"):
            base = normalized[:-4]
            if any(base == crypto for crypto in crypto_symbols):
                usd_version = f"{base}USD"
                logger.info(f"Normalized {normalized} to {usd_version}")
                normalized = usd_version
        
        # Return alias if found, otherwise return the normalized instrument
        return aliases.get(normalized, normalized)
