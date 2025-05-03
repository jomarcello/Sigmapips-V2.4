import asyncio
import logging
from trading_bot.services.chart_service.chart import ChartService

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_yahoo_only():
    print("\n==== Testing Pure Yahoo Finance Data ====\n")
    
    # Create chart service
    chart_service = ChartService()
    
    # Test different commodities
    symbols = [
        # Oil symbols
        'USOIL', 'XTIUSD', 'WTIUSD', 
        # Metals
        'XAUUSD', 'XAGUSD',
        # Other commodities
        'NATGAS', 'COPPER'
    ]
    
    for symbol in symbols:
        print(f"\nFetching {symbol} price directly from Yahoo Finance:")
        price = await chart_service._fetch_commodity_price(symbol)
        if price is not None:
            print(f"{symbol} price from Yahoo Finance: ${price:.2f}")
        else:
            print(f"Failed to get {symbol} price from Yahoo Finance")
    
    print("\n==== Test Complete ====\n")

if __name__ == "__main__":
    asyncio.run(test_yahoo_only()) 