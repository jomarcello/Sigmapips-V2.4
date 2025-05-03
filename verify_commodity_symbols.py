import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Import the necessary class
from trading_bot.services.chart_service.direct_yahoo_provider import DirectYahooProvider

def test_format_symbol():
    """Test the _format_symbol method for commodity instruments"""
    commodities = {
        "XAUUSD": "Gold",
        "XTIUSD": "Oil",
        "XAGUSD": "Silver",
        "XBRUSD": "Brent Oil"
    }
    
    print("\nTesting commodity symbol mapping:")
    for symbol, name in commodities.items():
        yahoo_symbol = DirectYahooProvider._format_symbol(symbol)
        print(f"{name} ({symbol}) -> {yahoo_symbol}")
    
    # Also test some other instrument types for comparison
    print("\nTesting other instrument types:")
    
    # Forex
    forex = ["EURUSD", "GBPUSD", "USDJPY"]
    for symbol in forex:
        yahoo_symbol = DirectYahooProvider._format_symbol(symbol)
        print(f"Forex: {symbol} -> {yahoo_symbol}")
    
    # Indices
    indices = ["US30", "US500", "DE40"]
    for symbol in indices:
        yahoo_symbol = DirectYahooProvider._format_symbol(symbol)
        print(f"Index: {symbol} -> {yahoo_symbol}")
    
    # Cryptocurrencies
    cryptos = ["BTCUSD", "ETHUSD"]
    for symbol in cryptos:
        yahoo_symbol = DirectYahooProvider._format_symbol(symbol)
        print(f"Crypto: {symbol} -> {yahoo_symbol}")

if __name__ == "__main__":
    print("Verifying DirectYahooProvider._format_symbol implementation")
    test_format_symbol() 