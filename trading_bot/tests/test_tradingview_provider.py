import asyncio
import logging
import os
import sys
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import de provider
from trading_bot.services.chart_service.tradingview_provider import TradingViewProvider, HAS_TRADINGVIEW_TA

# Skip all tests if tradingview-ta is not installed
pytestmark = pytest.mark.skipif(not HAS_TRADINGVIEW_TA, reason="tradingview-ta not installed")

class TestTradingViewProvider:
    """Test suite voor de TradingViewProvider class"""
    
    @pytest.mark.asyncio
    async def test_format_symbol(self):
        """Test de _format_symbol method"""
        # Known mappings
        assert TradingViewProvider._format_symbol("EURUSD") == ("EURUSD", "forex", "FX_IDC")
        assert TradingViewProvider._format_symbol("XAUUSD") == ("GOLD", "cfd", "TVC")
        assert TradingViewProvider._format_symbol("AAPL") == ("AAPL", "america", "NASDAQ")
        assert TradingViewProvider._format_symbol("BTCUSD") == ("BTCUSD", "crypto", "BINANCE")
        
        # Fallback mappings
        assert TradingViewProvider._format_symbol("XTSUSD")[1:] == ("cfd", "TVC")  # Onbekend commodity
        assert TradingViewProvider._format_symbol("ABCUSD")[1:] == ("crypto", "BINANCE")  # Onbekend crypto
        assert TradingViewProvider._format_symbol("ABCD")[1:] == ("america", "NASDAQ")  # Onbekend aandeel
        assert TradingViewProvider._format_symbol("USDCHF")[1:] == ("forex", "FX_IDC")  # Forex pair
    
    @pytest.mark.asyncio
    async def test_map_timeframe(self):
        """Test de _map_timeframe method"""
        # Known mappings
        assert TradingViewProvider._map_timeframe("1m") is not None
        assert TradingViewProvider._map_timeframe("5m") is not None
        assert TradingViewProvider._map_timeframe("1h") is not None
        assert TradingViewProvider._map_timeframe("1d") is not None
        
        # Fallback to default (1h)
        assert TradingViewProvider._map_timeframe("unknown") == TradingViewProvider._map_timeframe("1h")
    
    @pytest.mark.asyncio
    @patch('tradingview_ta.TA_Handler')
    async def test_get_technical_analysis(self, mock_handler):
        """Test de get_technical_analysis method met mock data"""
        # Setup mock
        mock_analysis = MagicMock()
        mock_analysis.summary = {"RECOMMENDATION": "BUY", "BUY": 12, "NEUTRAL": 6, "SELL": 3}
        mock_analysis.oscillators = {"RECOMMENDATION": "NEUTRAL"}
        mock_analysis.moving_averages = {"RECOMMENDATION": "BUY"}
        mock_analysis.indicators = {
            "RSI": 55.5,
            "MACD.macd": 0.5,
            "close": 100.0,
            "open": 99.0,
            "high": 101.0,
            "low": 98.0
        }
        
        mock_instance = MagicMock()
        mock_instance.get_analysis.return_value = mock_analysis
        mock_handler.return_value = mock_instance
        
        # Call method
        result = await TradingViewProvider.get_technical_analysis("EURUSD", "1h")
        
        # Assert results
        assert "summary" in result
        assert result["summary"]["RECOMMENDATION"] == "BUY"
        assert "indicators" in result
        assert result["indicators"]["RSI"] == 55.5
        assert result["indicators"]["close"] == 100.0
    
    @pytest.mark.asyncio
    @patch('tradingview_ta.TA_Handler')
    async def test_get_market_data(self, mock_handler):
        """Test de get_market_data method met mock data"""
        # Setup mock
        mock_analysis = MagicMock()
        mock_analysis.summary = {"RECOMMENDATION": "BUY", "BUY": 12, "NEUTRAL": 6, "SELL": 3}
        mock_analysis.oscillators = {"RECOMMENDATION": "NEUTRAL"}
        mock_analysis.moving_averages = {"RECOMMENDATION": "BUY"}
        mock_analysis.indicators = {
            "RSI": 55.5,
            "MACD.macd": 0.5,
            "MACD.signal": 0.3,
            "Stoch.K": 60.0,
            "Stoch.D": 58.0,
            "ADX": 25.0,
            "ATR": 1.5,
            "close": 100.0,
            "open": 99.0,
            "high": 101.0,
            "low": 98.0,
            "Volume": 1000.0
        }
        
        mock_instance = MagicMock()
        mock_instance.get_analysis.return_value = mock_analysis
        mock_handler.return_value = mock_instance
        
        # Call method
        result = await TradingViewProvider.get_market_data("EURUSD", "1h")
        
        # Assert results
        assert result is not None
        df, analysis_info = result
        
        # Check dataframe
        assert isinstance(df, pd.DataFrame)
        assert "Open" in df.columns
        assert "High" in df.columns
        assert "Low" in df.columns
        assert "Close" in df.columns
        assert "Volume" in df.columns
        
        # Check analysis info
        assert analysis_info["rsi"] == 55.5
        assert analysis_info["macd"] == 0.5
        assert analysis_info["recommendation"] == "BUY"
    
    @pytest.mark.asyncio
    @patch('tradingview_ta.TA_Handler')
    async def test_error_handling(self, mock_handler):
        """Test error handling in the provider"""
        # Setup mock to raise exception
        mock_instance = MagicMock()
        mock_instance.get_analysis.side_effect = Exception("Test error")
        mock_handler.return_value = mock_instance
        
        # Call method
        result = await TradingViewProvider.get_technical_analysis("EURUSD", "1h")
        
        # Assert results
        assert "error" in result
        assert "Test error" in result["error"]
        
        # Test get_market_data error handling
        result = await TradingViewProvider.get_market_data("EURUSD", "1h")
        assert result is None 