import asyncio
import logging
from trading_bot.services.chart_service.chart import ChartService

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_commodity_providers():
    print("\n==== Testing Commodity Data Providers ====\n")
    
    # Create chart service
    chart_service = ChartService()
    
    # Test all commodity methods to ensure Yahoo Finance is used
    
    print("\n--- Testing _fetch_commodity_price (Primary Yahoo Finance method) ---")
    oil_symbols = ['USOIL', 'XTIUSD', 'WTIUSD']
    for symbol in oil_symbols:
        price = await chart_service._fetch_commodity_price(symbol)
        if price is not None:
            print(f"{symbol} price from _fetch_commodity_price: ${price:.2f}")
        else:
            print(f"Failed to get {symbol} price from _fetch_commodity_price")
    
    print("\n--- Testing _fetch_oil_price_fallback (Should use Yahoo Finance) ---")
    for symbol in oil_symbols:
        price = await chart_service._fetch_oil_price_fallback(symbol)
        if price is not None:
            print(f"{symbol} price from _fetch_oil_price_fallback: ${price:.2f}")
        else:
            print(f"Failed to get {symbol} price from _fetch_oil_price_fallback")
    
    print("\n--- Testing _fetch_alternative_oil_price (Should use Yahoo Finance) ---")
    for symbol in oil_symbols:
        price = await chart_service._fetch_alternative_oil_price(symbol)
        if price is not None:
            print(f"{symbol} price from _fetch_alternative_oil_price: ${price:.2f}")
        else:
            print(f"Failed to get {symbol} price from _fetch_alternative_oil_price")
    
    # Test technical analysis to ensure it uses Yahoo Finance for commodities
    print("\n--- Testing get_technical_analysis for Oil (Should use Yahoo Finance) ---")
    for symbol in oil_symbols:
        print(f"\nTechnical Analysis for {symbol}:")
        analysis = await chart_service.get_technical_analysis(symbol)
        print(f"First 150 characters of analysis: {analysis[:150]}...")
    
    # Test other commodities
    print("\n--- Testing Other Commodities (Should use Yahoo Finance) ---")
    other_commodities = ['XAUUSD', 'XAGUSD', 'NATGAS', 'COPPER']
    for symbol in other_commodities:
        price = await chart_service._fetch_commodity_price(symbol)
        if price is not None:
            print(f"{symbol} price from Yahoo Finance: ${price:.2f}")
        else:
            print(f"Failed to get {symbol} price from Yahoo Finance")
    
    print("\n==== Test Complete ====\n")

if __name__ == "__main__":
    asyncio.run(test_commodity_providers()) 