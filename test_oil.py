import asyncio
import logging
import random
from trading_bot.services.chart_service.chart import ChartService

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_oil():
    print("\n==== Testing Oil Prices ====\n")
    chart_service = ChartService()
    
    # Test different oil symbols
    symbols = ['USOIL', 'XTIUSD', 'WTIUSD']
    
    # First test Yahoo Finance API
    print("\n--- Yahoo Finance API Results ---\n")
    for symbol in symbols:
        print(f"Testing {symbol} price fetch from Yahoo Finance...")
        
        # Store original random function to restore later
        original_random = random.random
        
        # Override random to always return 1.0 so we don't use immediate fallback
        random.random = lambda: 1.0
        
        try:
            price = await chart_service._fetch_commodity_price(symbol)
            print(f"{symbol} price from Yahoo Finance: {price}")
        except Exception as e:
            print(f"Error fetching {symbol} from Yahoo Finance: {str(e)}")
        finally:
            # Restore original random function
            random.random = original_random
    
    # Then test fallback mechanism
    print("\n--- Fallback Mechanism Results ---\n")
    for symbol in symbols:
        print(f"Testing {symbol} price fetch using fallback...")
        price = await chart_service._fetch_oil_price_fallback(symbol)
        print(f"{symbol} fallback price: {price}")
    
    # Test with 50% chance to use fallback
    print("\n--- Testing with 50% Fallback Chance ---\n")
    for i in range(5):
        symbol = random.choice(symbols)
        print(f"Run {i+1}: Testing {symbol} with 50% fallback chance...")
        price = await chart_service._fetch_commodity_price(symbol)
        print(f"{symbol} price: {price}")
    
    print("\n==== Test Complete ====\n")

if __name__ == "__main__":
    asyncio.run(test_oil()) 