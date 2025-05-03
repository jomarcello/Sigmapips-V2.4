import asyncio
import logging
from trading_bot.services.chart_service.chart import ChartService

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_oil_prices():
    print("\n==== Testing Oil Prices Using Yahoo Finance Only ====\n")
    
    # Test through ChartService
    print("Testing oil prices through ChartService (Yahoo Finance only):")
    chart_service = ChartService()
    
    # Test different oil symbols
    symbols = ['USOIL', 'XTIUSD', 'WTIUSD']
    
    for symbol in symbols:
        print(f"\nFetching {symbol} price from Yahoo Finance:")
        price = await chart_service._fetch_commodity_price(symbol)
        if price is not None:
            print(f"{symbol} price from Yahoo Finance: ${price:.2f}")
        else:
            print(f"Failed to get {symbol} price from Yahoo Finance")
        
        # Also test fallback method (which should now use Yahoo Finance as well)
        alt_price = await chart_service._fetch_oil_price_fallback(symbol)
        if alt_price is not None:
            print(f"{symbol} fallback price (also Yahoo Finance): ${alt_price:.2f}")
        else:
            print(f"Failed to get {symbol} fallback price")
    
    print("\n==== Test Complete ====\n")

if __name__ == "__main__":
    asyncio.run(test_oil_prices()) 