import logging
import traceback
import asyncio
import os
from typing import Optional, Dict, Any, Tuple, List, Union
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
import requests_cache
from requests_ratelimiter import LimiterSession

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
    """Provider class voor Yahoo Finance API met verbeterde rate limit afhandeling en caching"""
    
    # Cache configuratie (1 uur cache voor Yahoo Finance data)
    _session = None
    _initialized = False
    
    # Moderne user agents om blokkades te vermijden
    _user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0'
    ]
    
    @classmethod
    def _initialize(cls):
        """Initialize de provider met een gecachte en rate-limited session"""
        if cls._initialized:
            return
            
        try:
            # Maak een cache directory als die nog niet bestaat
            cache_dir = os.path.join('data', 'cache', 'yahoo')
            os.makedirs(cache_dir, exist_ok=True)
            
            # Caching session met een rate limiter
            # Gebruik een SQLite backend voor betere persistentie
            backend = 'sqlite'
            cache_name = os.path.join(cache_dir, 'yfinance_cache')
            expiry = timedelta(hours=1)
            
            # Maak een gecachte session
            session = requests_cache.CachedSession(
                cache_name=cache_name,
                backend=backend,
                expire_after=expiry
            )
            
            # Voeg rate limiting toe (max 2 verzoeken per seconde)
            limiter_session = LimiterSession(
                per_second=2,
                session=session
            )
            
            # Configureer retry logica voor verbeterde betrouwbaarheid
            retries = Retry(
                total=5,
                backoff_factor=0.5,
                status_forcelist=[429, 500, 502, 503, 504],
                respect_retry_after_header=True
            )
            
            adapter = HTTPAdapter(max_retries=retries)
            limiter_session.mount('http://', adapter)
            limiter_session.mount('https://', adapter)
            
            # Stel een willekeurige user agent in
            user_agent = random.choice(cls._user_agents)
            limiter_session.headers.update({'User-Agent': user_agent})
            
            # Sla de session op
            cls._session = limiter_session
            cls._initialized = True
            
            logger.info(f"[Yahoo] Geïnitialiseerd met gecachte session en rate limiting")
            logger.info(f"[Yahoo] Cache locatie: {cache_name}")
            logger.info(f"[Yahoo] Gebruikte User-Agent: {user_agent}")
                
        except Exception as e:
            logger.error(f"[Yahoo] Fout bij initialisatie: {str(e)}")
            # Zorg dat we een basis session hebben, zelfs als caching faalt
            cls._session = requests.Session()
            cls._session.headers.update({'User-Agent': random.choice(cls._user_agents)})
    
    @classmethod
    def get_market_data(cls, symbol: str, timeframe: str = "1h", limit: int = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Haal marktdata op voor het opgegeven symbool en timeframe
        
        Args:
            symbol: Het handelssymbool (bijv. EURUSD, AAPL)
            timeframe: Het tijdsinterval (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)
            limit: Maximaal aantal datapunten
            
        Returns:
            Tuple[pd.DataFrame, Dict]: (Dataframe met marktdata, info dictionary)
        """
        try:
            # Zorg dat de provider is geïnitialiseerd
            if not cls._initialized:
                cls._initialize()
                
            # Converteer timeframe naar Yahoo format
            interval = cls._map_timeframe_to_interval(timeframe)
            
            # Converteer symbool naar Yahoo format
            yahoo_symbol = cls._map_symbol_to_yahoo(symbol)
            
            # Bereken periode op basis van limit en timeframe
            period = cls._calculate_period(timeframe, limit)
            
            logger.info(f"[Yahoo] Ophalen data voor {yahoo_symbol} met interval {interval} en periode {period}")
            
            try:
                # Gebruik yfinance met onze geconfigureerde session
                ticker = yf.Ticker(yahoo_symbol, session=cls._session)
                
                # Download historische data
                df = ticker.history(period=period, interval=interval, auto_adjust=True)
                
                # Als we geen data krijgen, probeer een alternatief symbool voor commodities
                if df.empty and symbol in ["USOIL", "XTIUSD", "WTIUSD"]:
                    alternative_symbols = ["CL=F", "USO", "BNO"]
                    
                    for alt_symbol in alternative_symbols:
                        if alt_symbol == yahoo_symbol:
                            continue
                            
                        logger.info(f"[Yahoo] Probeer alternatief symbool {alt_symbol} voor {symbol}")
                        
                        ticker = yf.Ticker(alt_symbol, session=cls._session)
                        df = ticker.history(period=period, interval=interval, auto_adjust=True)
                        
                        if not df.empty:
                            logger.info(f"[Yahoo] Succesvol data opgehaald met alternatief symbool {alt_symbol}")
                            break
                
                # Als we nog steeds geen data hebben, return een error
                if df.empty:
                    logger.warning(f"[Yahoo] Geen data gevonden voor {yahoo_symbol}")
                    return pd.DataFrame(), {"error": "data_not_available", "message": f"Geen data beschikbaar voor {symbol}"}
                
                # Verwerk dataframe voor consistente format
                df = cls._process_dataframe(df)
                
                # Bereken technische indicatoren
                indicators = cls._calculate_indicators(df)
                
                # Pas limit toe indien nodig
                if limit and len(df) > limit:
                    df = df.tail(limit)
                
                logger.info(f"[Yahoo] Succesvol {len(df)} rijen opgehaald voor {yahoo_symbol}")
                return df, indicators
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Rate limit error afhandeling
                if "rate limit" in error_msg or "too many requests" in error_msg or "429" in error_msg:
                    logger.error(f"[Yahoo] Rate limit bereikt voor {yahoo_symbol}: {str(e)}")
                    
                    # Wacht lang genoeg voor de volgende poging
                    wait_time = random.uniform(20, 30)  # Langere wachttijd bij rate limits
                    logger.info(f"[Yahoo] Wachten {wait_time:.2f} seconden vanwege rate limiting")
                    time.sleep(wait_time)
                    
                    # Roteer de User-Agent
                    if cls._session:
                        new_user_agent = random.choice(cls._user_agents)
                        cls._session.headers.update({'User-Agent': new_user_agent})
                        logger.info(f"[Yahoo] User-Agent gewijzigd naar: {new_user_agent[:30]}...")
                    
                    return pd.DataFrame(), {"error": "rate_limit", "message": f"Rate limit overschreden voor {symbol}. Probeer later opnieuw."}
                
                # Andere fouten
                logger.error(f"[Yahoo] Fout bij ophalen data voor {yahoo_symbol}: {str(e)}")
                return pd.DataFrame(), {"error": "unknown", "message": f"Fout bij ophalen data voor {symbol}: {str(e)}"}
            
        except Exception as e:
            logger.error(f"[Yahoo] Onverwachte fout: {str(e)}")
            logger.error(traceback.format_exc())
        return pd.DataFrame(), {"error": "unknown", "message": str(e)}
    
    @staticmethod
    def _map_timeframe_to_interval(timeframe: str) -> str:
        """Converteer timeframe naar Yahoo Finance interval"""
        mapping = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "1h",  # 4h moet worden geresampled vanuit 1h
            "1d": "1d",
            "1wk": "1wk",
            "1mo": "1mo"
        }
        return mapping.get(timeframe, "1d")  # Default naar daily

    @staticmethod
    def _map_symbol_to_yahoo(symbol: str) -> str:
        """Converteer trading bot symbool naar Yahoo Finance formaat"""
        # Standaard mappings voor bekende symbolen
        symbol_mapping = {
            "XAUUSD": "GC=F",   # Gold futures
            "XAGUSD": "SI=F",    # Silver futures
            "XTIUSD": "CL=F",    # WTI Crude oil futures
            "WTIUSD": "CL=F",    # WTI crude oil (alias)
            "USOIL": "CL=F",     # US Oil (zelfde als WTI)
            "XBRUSD": "BZ=F",    # Brent crude oil
            "US30": "^DJI",      # Dow Jones
            "US500": "^GSPC",    # S&P 500
            "SPX500": "^GSPC",   # S&P 500 (alternatieve naam)
            "US100": "^NDX",     # Nasdaq 100
            "NAS100": "^NDX",    # Nasdaq 100 (alternatieve naam)
            "UK100": "^FTSE",    # FTSE 100
            "DE40": "^GDAXI",    # DAX
            "JP225": "^N225",    # Nikkei 225
            "VIX": "^VIX",       # Volatility Index
        }
        
        # Controleer op directe mapping
        if symbol in symbol_mapping:
            return symbol_mapping[symbol]
        
        # Forex paren (bijv. EURUSD -> EURUSD=X)
        if len(symbol) == 6 and symbol.isalpha():
            base = symbol[:3]
            quote = symbol[3:]
            return f"{base}{quote}=X"
        
        # Aandelen kunnen direct worden doorgegeven
        return symbol

    @staticmethod
    def _calculate_period(timeframe: str, limit: int = None) -> str:
        """Bereken de juiste periode voor Yahoo Finance op basis van timeframe en limit"""
        if not limit:
            # Standaard periodes indien geen limit is opgegeven
            default_periods = {
                "1m": "7d",    # Yahoo heeft max 7 dagen voor 1m data
                "5m": "60d",   # Max 60 dagen voor 5m data
                "15m": "60d",  # Max 60 dagen voor 15m data
                "30m": "60d",  # Max 60 dagen voor 30m data
                "1h": "730d",  # 2 jaar voor 1h data
                "4h": "730d",  # 2 jaar voor 4h data (geresampled)
                "1d": "5y",    # 5 jaar voor dagelijkse data
                "1wk": "10y",  # 10 jaar voor wekelijkse data
                "1mo": "30y"   # 30 jaar voor maandelijkse data
            }
            return default_periods.get(timeframe, "1y")
        
        # Bereken de periode op basis van limit
        if timeframe == "1m":
            days = min(7, limit / 24 / 60)  # Max 7 dagen voor 1m data
            return f"{int(days) + 1}d" if days < 7 else "7d"
        elif timeframe == "5m":
            days = min(60, limit * 5 / 24 / 60)
            return f"{int(days) + 1}d" if days < 60 else "60d"
        elif timeframe == "15m":
            days = min(60, limit * 15 / 24 / 60)
            return f"{int(days) + 1}d" if days < 60 else "60d"
        elif timeframe == "30m":
            days = min(60, limit * 30 / 24 / 60)
            return f"{int(days) + 1}d" if days < 60 else "60d"
        elif timeframe == "1h":
            days = min(730, limit / 24)
            if days <= 60:
                return f"{int(days) + 1}d"
            elif days <= 730:
                return f"{int(days / 30) + 1}mo"
            else:
                return "2y"
        elif timeframe == "4h":
            days = min(730, limit * 4 / 24)
            if days <= 60:
                return f"{int(days) + 1}d"
            elif days <= 730:
                return f"{int(days / 30) + 1}mo"
            else:
                return "2y"
        elif timeframe == "1d":
            if limit <= 60:
                return f"{limit + 5}d"  # Paar extra dagen voor indicatoren
            elif limit <= 365:
                return f"{int(limit / 30) + 2}mo"
            else:
                years = limit / 365
                return f"{int(years) + 1}y"
        elif timeframe == "1wk":
            weeks = limit
            if weeks <= 52:
                return f"{weeks + 2}wk"
            else:
                years = weeks / 52
                return f"{int(years) + 1}y"
        elif timeframe == "1mo":
            months = limit
            if months <= 12:
                return f"{months + 1}mo"
            else:
                years = months / 12
                return f"{int(years) + 1}y"
        
        # Default
        return "1y"

    @staticmethod
    def _process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Verwerk het dataframe voor consistente format"""
        if df.empty:
            return df
        
        # Zorg dat de kolommen de juiste namen hebben
        if isinstance(df.columns, pd.MultiIndex):
            # yfinance geeft soms een MultiIndex terug
            df = df.droplevel(level=0, axis=1)
        
        # Converteer kolommen naar correcte format
        columns = []
        rename_map = {}
        
        for col in df.columns:
            lower_col = col.lower()
            if lower_col in ['open', 'high', 'low', 'close', 'volume']:
                rename_map[col] = lower_col.capitalize()
                columns.append(lower_col.capitalize())
        
        # Hernoem kolommen indien nodig
        if rename_map:
            df = df.rename(columns=rename_map)
        
        # Voeg missende kolommen toe met NaN waarden
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col not in df.columns:
                df[col] = np.nan
        
        # Resample 1h naar 4h indien nodig
        if getattr(df, '_timeframe', None) == '4h':
            try:
                # Aggregatie regels
                agg_dict = {
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }
                
                # Resample
                df = df.resample('4h').agg(agg_dict)
            except Exception as e:
                logger.error(f"[Yahoo] Fout bij resampling naar 4h: {str(e)}")
        
        # Verwerk missende waarden
        df = df.ffill(limit=2)  # Forward fill met limiet
        df = df.dropna()  # Verwijder resterende NaN waarden
        
        # Sorteer op index
        df = df.sort_index()
        
        return df

    @staticmethod
    def _calculate_indicators(df: pd.DataFrame) -> Dict[str, Any]:
        """Bereken technische indicatoren op basis van prijsdata"""
        if df.empty:
            return {"error": "no_data"}
        
        try:
            # Haal de laatste prijs op
            latest = df.iloc[-1]
            
            # Basis indicatoren
            indicators = {
                "close": float(latest["Close"]),
                "open": float(latest["Open"]),
                "high": float(latest["High"]),
                "low": float(latest["Low"]),
                "volume": float(latest["Volume"]) if "Volume" in df else 0,
            }
            
            # EMA berekeningen
            if len(df) >= 20:
                df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
                indicators["ema_20"] = float(df['EMA20'].iloc[-1])
            
            if len(df) >= 50:
                df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
                indicators["ema_50"] = float(df['EMA50'].iloc[-1])
                
            if len(df) >= 200:
                df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
                indicators["ema_200"] = float(df['EMA200'].iloc[-1])
            
            # RSI berekening
            if len(df) >= 14:
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                df['RSI'] = 100 - (100 / (1 + rs))
                indicators["rsi"] = float(df['RSI'].iloc[-1])
            
            # MACD berekening
            if len(df) >= 26:
                df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
                df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = df['EMA12'] - df['EMA26']
                df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                indicators["macd"] = float(df['MACD'].iloc[-1])
                indicators["macd_signal"] = float(df['MACD_signal'].iloc[-1])
                indicators["macd_hist"] = indicators["macd"] - indicators["macd_signal"]
            
            # Bollinger Bands
            if len(df) >= 20:
                df['SMA20'] = df['Close'].rolling(window=20).mean()
                df['SD20'] = df['Close'].rolling(window=20).std()
                df['upper_band'] = df['SMA20'] + (df['SD20'] * 2)
                df['lower_band'] = df['SMA20'] - (df['SD20'] * 2)
                indicators["bb_upper"] = float(df['upper_band'].iloc[-1])
                indicators["bb_middle"] = float(df['SMA20'].iloc[-1])
                indicators["bb_lower"] = float(df['lower_band'].iloc[-1])
            
            # Dagelijkse en wekelijkse range
            indicators["daily_high"] = float(df['High'].iloc[-1])
            indicators["daily_low"] = float(df['Low'].iloc[-1])
            
            # Calculate weekly range if we have enough data
            if len(df) >= 5:
                weekly_data = df.tail(5)  # Laatste 5 datapunten voor dagelijkse data
                indicators["weekly_high"] = float(weekly_data['High'].max())
                indicators["weekly_low"] = float(weekly_data['Low'].min())
            
            return indicators
            
        except Exception as e:
            logger.error(f"[Yahoo] Fout bij berekenen indicatoren: {str(e)}")
            return {"error": "indicator_calculation_failed", "message": str(e)}
