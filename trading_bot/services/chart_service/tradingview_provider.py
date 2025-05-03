import logging
import asyncio
import pandas as pd
import random
import time
from typing import Optional, Dict, Tuple, List, Any
from datetime import datetime, timedelta
import os

# Ensure this dependency is installed
try:
    from tradingview_ta import TA_Handler, Interval, Exchange
    HAS_TRADINGVIEW_TA = True
except ImportError:
    HAS_TRADINGVIEW_TA = False

# Set up logging
logger = logging.getLogger(__name__)

# Cache for API results
market_data_cache = {}
data_download_cache = {}

class TradingViewProvider:
    """Provider class voor TradingView data als alternatief voor forex en andere marktdata"""
    
    # Mapping van timeframes naar TradingView intervallen
    TIMEFRAME_MAP = {
        "1m": Interval.INTERVAL_1_MINUTE if HAS_TRADINGVIEW_TA else "1m",
        "5m": Interval.INTERVAL_5_MINUTES if HAS_TRADINGVIEW_TA else "5m",
        "15m": Interval.INTERVAL_15_MINUTES if HAS_TRADINGVIEW_TA else "15m",
        "30m": Interval.INTERVAL_30_MINUTES if HAS_TRADINGVIEW_TA else "30m",
        "1h": Interval.INTERVAL_1_HOUR if HAS_TRADINGVIEW_TA else "1h",
        "4h": Interval.INTERVAL_4_HOURS if HAS_TRADINGVIEW_TA else "4h",
        "1d": Interval.INTERVAL_1_DAY if HAS_TRADINGVIEW_TA else "1d",
        "1w": Interval.INTERVAL_1_WEEK if HAS_TRADINGVIEW_TA else "1w",
        "1M": Interval.INTERVAL_1_MONTH if HAS_TRADINGVIEW_TA else "1M",
    }
    
    # Mapping van symbolen naar TradingView format
    SYMBOL_MAP = {
        # Forex paren
        "EURUSD": ("EURUSD", "forex", "FX_IDC"),
        "GBPUSD": ("GBPUSD", "forex", "FX_IDC"),
        "USDJPY": ("USDJPY", "forex", "FX_IDC"),
        "AUDUSD": ("AUDUSD", "forex", "FX_IDC"),
        "USDCAD": ("USDCAD", "forex", "FX_IDC"),
        "USDCHF": ("USDCHF", "forex", "FX_IDC"),
        "NZDUSD": ("NZDUSD", "forex", "FX_IDC"),
        # Commodities
        "XAUUSD": ("GOLD", "cfd", "TVC"),  # Gold op TVC
        "XAGUSD": ("SILVER", "cfd", "TVC"),  # Silver op TVC
        "XTIUSD": ("USOIL", "cfd", "TVC"),  # WTI Olie op TVC
        "XBRUSD": ("UKOIL", "cfd", "TVC"),  # Brent Olie op TVC
        # Aandelen
        "AAPL": ("AAPL", "america", "NASDAQ"),
        "MSFT": ("MSFT", "america", "NASDAQ"),
        "GOOGL": ("GOOGL", "america", "NASDAQ"),
        "AMZN": ("AMZN", "america", "NASDAQ"),
        # Indices
        "US500": ("SPX500", "america", "CBOE"),  # S&P 500 index via CBOE
        "NAS100": ("NDX", "america", "NASDAQ"),  # Nasdaq 100 index
        "US30": ("DJI", "america", "DJ"),        # Dow Jones index
        # Crypto
        "BTCUSD": ("BTCUSD", "crypto", "BINANCE"),
        "ETHUSD": ("ETHUSD", "crypto", "BINANCE"),
    }

    @staticmethod
    def _format_symbol(symbol: str) -> Tuple[str, str, str]:
        """Format een handelssymbool voor gebruik met TradingView API"""
        if symbol in TradingViewProvider.SYMBOL_MAP:
            return TradingViewProvider.SYMBOL_MAP[symbol]
        
        # Fallback: probeer een standaard mapping
        if symbol.endswith("USD") and symbol.startswith("X"):
            # Metaal/commodities in cfd format
            metal_symbol = "GOLD" if "XAU" in symbol else "SILVER" if "XAG" in symbol else symbol
            return (metal_symbol, "cfd", "TVC")
        elif symbol.endswith("USD") and not symbol.startswith("X"):
            # Crypto format aanname
            return (symbol, "crypto", "BINANCE")
        elif len(symbol) <= 5 and not symbol.startswith("X"):
            # Aandeel formaat aanname
            return (symbol, "america", "NASDAQ")
        elif symbol.startswith("US") or symbol in ["SPX", "DJI", "NDX"]:
            # Index format
            return (symbol, "america", "INDEX")
        else:
            # Forex formaat aanname
            return (symbol, "forex", "FX_IDC")

    @staticmethod
    def _map_timeframe(timeframe: str) -> str:
        """Converteer timeframe naar TradingView interval"""
        if timeframe in TradingViewProvider.TIMEFRAME_MAP:
            return TradingViewProvider.TIMEFRAME_MAP[timeframe]
        
        # Fallback to default
        logger.warning(f"[TradingView] Onbekend timeframe '{timeframe}', valt terug op 1h")
        return TradingViewProvider.TIMEFRAME_MAP["1h"]

    @staticmethod
    async def get_technical_analysis(symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
        """Haal technische analyse op van TradingView voor een specifiek symbool"""
        
        if not HAS_TRADINGVIEW_TA:
            logger.error("[TradingView] tradingview-ta pakket niet geïnstalleerd. Installeer met: pip install tradingview-ta")
            return {"error": "tradingview_ta_not_installed"}
            
        # Formatteer symbool voor TradingView
        symbol_formatted, screener, exchange = TradingViewProvider._format_symbol(symbol)
        interval = TradingViewProvider._map_timeframe(timeframe)
        
        # Cache key
        cache_key = f"{symbol_formatted}_{screener}_{exchange}_{interval}"
        
        # Check cache
        if cache_key in market_data_cache:
            cache_entry = market_data_cache[cache_key]
            cache_time = cache_entry.get("time", 0)
            
            # Bepaal maximale cache tijd gebaseerd op timeframe
            max_cache_time = 60  # 60 seconden voor default
            if timeframe == "1m": max_cache_time = 30
            elif timeframe in ["5m", "15m"]: max_cache_time = 60
            elif timeframe == "30m": max_cache_time = 120
            elif timeframe == "1h": max_cache_time = 300
            elif timeframe in ["4h", "1d"]: max_cache_time = 1800
            
            # Als cache nog geldig is
            if time.time() - cache_time < max_cache_time:
                logger.info(f"[TradingView] Cache hit voor {symbol}")
                return cache_entry.get("data", {})
        
        try:
            # Execute in thread pool to not block the event loop
            loop = asyncio.get_running_loop()
            
            def get_analysis():
                try:
                    handler = TA_Handler(
                        symbol=symbol_formatted,
                        screener=screener,
                        exchange=exchange,
                        interval=interval
                    )
                    
                    # Get analysis
                    analysis = handler.get_analysis()
                    
                    return {
                        "summary": analysis.summary,
                        "oscillators": analysis.oscillators,
                        "moving_averages": analysis.moving_averages,
                        "indicators": {
                            # Common indicators
                            "RSI": analysis.indicators.get("RSI", None),
                            "MACD.macd": analysis.indicators.get("MACD.macd", None),
                            "MACD.signal": analysis.indicators.get("MACD.signal", None),
                            "Stoch.K": analysis.indicators.get("Stoch.K", None),
                            "Stoch.D": analysis.indicators.get("Stoch.D", None),
                            "ADX": analysis.indicators.get("ADX", None),
                            "ATR": analysis.indicators.get("ATR", None),
                            "CCI": analysis.indicators.get("CCI", None),
                            "AO": analysis.indicators.get("AO", None),
                            "Mom": analysis.indicators.get("Mom", None),
                            "VWMA": analysis.indicators.get("VWMA", None),
                            # Price data
                            "close": analysis.indicators.get("close", None),
                            "open": analysis.indicators.get("open", None),
                            "high": analysis.indicators.get("high", None),
                            "low": analysis.indicators.get("low", None),
                            # Extra data waar mogelijk
                            "Volatility": analysis.indicators.get("Volatility", None),
                            "Volume": analysis.indicators.get("Volume", None),
                            "Change": analysis.indicators.get("Change", None),
                            "Recommend.All": analysis.indicators.get("Recommend.All", None),
                            "Recommend.MA": analysis.indicators.get("Recommend.MA", None),
                            "Recommend.Other": analysis.indicators.get("Recommend.Other", None),
                        }
                    }
                except Exception as e:
                    logger.error(f"[TradingView] Error in analysis: {str(e)}")
                    return {"error": str(e)}
            
            # Run in thread pool
            result = await loop.run_in_executor(None, get_analysis)
            
            # Update cache
            market_data_cache[cache_key] = {
                "time": time.time(),
                "data": result
            }
            
            return result
            
        except Exception as e:
            logger.error(f"[TradingView] Error getting analysis for {symbol}: {str(e)}")
            return {"error": str(e)}

    @staticmethod
    async def get_market_data(symbol: str, timeframe: str = "1h", limit: int = 100) -> Optional[Tuple[pd.DataFrame, Dict]]:
        """
        Haal marktdata inclusief technische indicatoren op voor het gegeven symbool
        
        Args:
            symbol: Het handelssymbool (bijv. EURUSD, AAPL)
            timeframe: Het tijdsinterval (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)
            limit: Maximaal aantal datapunten (niet relevant voor real-time data)
            
        Returns:
            Tuple[pd.DataFrame, Dict]: (Dataframe met marktdata, dict met indicatoren)
        """
        try:
            logger.info(f"[TradingView] Getting market data for {symbol} ({timeframe}) with limit {limit}")
            
            # Get technical analysis
            analysis = await TradingViewProvider.get_technical_analysis(symbol, timeframe)
            
            if "error" in analysis:
                logger.error(f"[TradingView] Error getting analysis: {analysis['error']}")
                return None
                
            # Extract price data
            indicators = analysis.get("indicators", {})
            
            # Create a dataframe with current price data
            now = datetime.now()
            
            # Build dataframe
            data = {
                'Open': [indicators.get("open", 0)],
                'High': [indicators.get("high", 0)],
                'Low': [indicators.get("low", 0)],
                'Close': [indicators.get("close", 0)],
                'Volume': [indicators.get("Volume", 0)]
            }
            
            df = pd.DataFrame(data, index=[now])
            
            # Format the technical analysis indicators
            analysis_info = {
                "rsi": indicators.get("RSI", 50),
                "macd": indicators.get("MACD.macd", 0),
                "macd_signal": indicators.get("MACD.signal", 0),
                "stochastic_k": indicators.get("Stoch.K", 50),
                "stochastic_d": indicators.get("Stoch.D", 50),
                "adx": indicators.get("ADX", 25),
                "atr": indicators.get("ATR", 0),
                "summary": analysis.get("summary", {}),
                "recommendation": analysis.get("summary", {}).get("RECOMMENDATION", "NEUTRAL"),
                "moving_averages": analysis.get("moving_averages", {}),
                "oscillators": analysis.get("oscillators", {})
            }
            
            logger.info(f"[TradingView] Successfully retrieved data for {symbol}")
            return df, analysis_info
            
        except Exception as e:
            logger.error(f"[TradingView] Error in get_market_data for {symbol}: {str(e)}")
            return None 