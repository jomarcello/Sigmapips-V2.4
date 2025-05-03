#!/usr/bin/env python3
import asyncio
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

async def test_crypto_analysis():
    from trading_bot.services.chart_service.chart import ChartService
    
    print("\n=== Testing Crypto Technical Analysis ===")
    
    # Create chart service
    chart_service = ChartService()
    await chart_service.initialize()
    
    # Test with common crypto symbols
    symbols = ["BTCUSD", "ETHUSD", "BTC", "ETH"]
    
    for symbol in symbols:
        print(f"\n--- Testing {symbol} ---")
        try:
            normalized = chart_service._normalize_instrument_name(symbol)
            print(f"Normalized: {symbol} -> {normalized}")
            
            # Detect market type
            market_type = await chart_service._detect_market_type(normalized)
            print(f"Detected market type: {market_type}")
            
            # Get technical analysis
            analysis = await chart_service.get_technical_analysis(normalized, "1h")
            print(f"Analysis result length: {len(analysis)} characters")
            print(f"First 200 chars: {analysis[:200]}...")
        except Exception as e:
            print(f"Error testing {symbol}: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_crypto_analysis()) 