#!/usr/bin/env python
import asyncio
import logging
import pandas as pd
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the provider
from trading_bot.services.chart_service.yfinance_provider import YahooFinanceProvider

async def test_oil_data():
    """Test fetching oil data from Yahoo Finance provider with our new changes"""
    print("\n=== TESTING OIL DATA FETCHING ===\n")
    
    # Test different oil symbols
    symbols = ["USOIL", "WTIUSD", "XTIUSD"]
    timeframes = ["1h", "4h", "1d"]
    
    for symbol in symbols:
        print(f"\n--- Testing {symbol} ---")
        for timeframe in timeframes:
            print(f"\nFetching {symbol} on {timeframe} timeframe:")
            try:
                # Get data from Yahoo Finance
                result = await YahooFinanceProvider.get_market_data(symbol, timeframe, limit=100)
                
                if result is not None and isinstance(result, tuple) and len(result) == 2:
                    data, indicators = result
                    
                    # Print information about the data
                    print(f"Shape: {data.shape}")
                    print(f"Date range: {data.index[0]} to {data.index[-1]}")
                    
                    # Using safe method to access last close price
                    last_close = data['Close'].iloc[-1]
                    if isinstance(last_close, pd.Series):
                        last_close = last_close.iloc[0]
                    print(f"Last close price: {float(last_close):.2f}")
                    
                    # Print indicators
                    if indicators:
                        print("\nIndicators:")
                        for key, value in indicators.items():
                            if isinstance(value, float):
                                print(f"  {key}: {value:.2f}")
                            else:
                                print(f"  {key}: {value}")
                    
                    # Check for indicator columns to determine if real data
                    if 'EMA20' in data.columns:
                        print("\nIndicator columns found in dataframe")
                    else:
                        print("\nUsing REAL Yahoo Finance data")
                    
                else:
                    print(f"No data returned for {symbol} on {timeframe}")
                    
            except Exception as e:
                print(f"Error fetching {symbol} on {timeframe}: {e}")

if __name__ == "__main__":
    asyncio.run(test_oil_data()) 