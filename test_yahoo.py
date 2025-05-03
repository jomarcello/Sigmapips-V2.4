import asyncio
import os
import sys
import traceback

# Voeg de root directory toe aan het pad, zodat we de tradingbot modules kunnen importeren
sys.path.append('.')

from trading_bot.services.chart_service.direct_yahoo_provider import DirectYahooProvider
from trading_bot.services.chart_service.chart import ChartService

async def test_yahoo_provider():
    try:
        provider = DirectYahooProvider()
        
        # Test verschillende instrumenten en markttypen
        test_symbols = [
            "GBPUSD",       # Forex
            "EURUSD",       # Forex
            "XAUUSD",       # Commodity (Gold)
            "BTCUSD",       # Crypto
            "US500"         # Index
        ]
        
        for symbol in test_symbols:
            print(f"\n===== Testing {symbol} =====")
            # Test yahoo symbol formatting
            yahoo_symbol = DirectYahooProvider._format_symbol(symbol)
            print(f"Original symbol: {symbol} → Yahoo symbol: {yahoo_symbol}")
            
            # Test data retrieval with 1h timeframe
            timeframe = "1h"
            print(f"🔍 Getting market data for {symbol} with timeframe {timeframe}")
            market_data, indicators = await provider.get_market_data(symbol, timeframe=timeframe)
            
            if market_data is not None and not market_data.empty:
                print(f"✅ Successfully got market data!")
                print(f"  Shape: {market_data.shape}")
                print(f"  Columns: {market_data.columns.tolist()}")
                print(f"  Latest prices: Open={market_data['Open'].iloc[-1]:.4f}, Close={market_data['Close'].iloc[-1]:.4f}")
                print(f"  Time range: {market_data.index[0]} to {market_data.index[-1]}")
            else:
                print(f"❌ Failed to get market data")
        
        # Test de ChartService
        print("\n--- Testing ChartService ---")
        chart_service = ChartService()
        await chart_service.initialize()
        
        # Haal technische analyse op voor GBPUSD
        analysis = await chart_service.get_technical_analysis("GBPUSD", timeframe="1h")
        print(f"\nTechnical Analysis for GBPUSD:\n{analysis[:300]}...")  # Print eerste 300 karakters
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        traceback.print_exc()

# Voer test uit
if __name__ == "__main__":
    # Zet PREFER_REAL_MARKET_DATA naar "true" om direct Yahoo data te forceren
    os.environ["PREFER_REAL_MARKET_DATA"] = "true"
    
    asyncio.run(test_yahoo_provider()) 