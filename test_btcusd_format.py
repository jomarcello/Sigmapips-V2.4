#!/usr/bin/env python3
"""
Test script to specifically verify the BTCUSD formatting issue with Binance integration.
"""

import asyncio
import logging
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

async def test_btcusd_format():
    """Test specific BTCUSD format handling for Binance"""
    print("\n=== Testing BTCUSD Format Handling ===")
    
    # Import the providers
    from trading_bot.services.chart_service.binance_provider import BinanceProvider
    from trading_bot.services.chart_service.chart import ChartService
    
    # Test the direct provider formatting
    crypto_symbols = ["BTCUSD", "BTC", "BTCUSDT", "ETHUSD", "ETH"]
    
    print("\nDirect BinanceProvider._format_symbol test:")
    for symbol in crypto_symbols:
        formatted = BinanceProvider._format_symbol(symbol)
        print(f"  {symbol} -> {formatted}")
    
    # Test through chart service
    print("\nChart service provider selection test:")
    chart_service = ChartService()
    
    crypto_symbol = "BTCUSD"
    market_type = await chart_service._detect_market_type(crypto_symbol)
    print(f"  Market type for {crypto_symbol}: {market_type}")
    
    # Import required providers for manual test
    from trading_bot.services.chart_service.binance_provider import BinanceProvider
    from trading_bot.services.chart_service.yfinance_provider import YahooFinanceProvider
    
    binance_provider = BinanceProvider()
    yahoo_provider = YahooFinanceProvider()
    
    # Test the direct market data access with BTCUSD
    print("\nDirect market data test with BTCUSD:")
    print("  Testing Binance provider with BTCUSD...")
    result = await binance_provider.get_market_data("BTCUSD", "1h")
    if result:
        print(f"  ✅ Binance provider successfully returned data for BTCUSD")
        print(f"  Current price: {result.indicators.get('close')}")
    else:
        print(f"  ❌ Binance provider failed for BTCUSD")
    
    # Test with the manual symbol conversion
    print("\nManual symbol conversion test:")
    print("  Testing Binance provider with manually converted BTCUSDT...")
    result = await binance_provider.get_market_data("BTCUSDT", "1h")
    if result:
        print(f"  ✅ Binance provider successfully returned data for BTCUSDT")
        print(f"  Current price: {result.indicators.get('close')}")
    else:
        print(f"  ❌ Binance provider failed for BTCUSDT")
    
    # Now test the full chart service
    print("\nFull chart service test with BTCUSD:")
    await chart_service.initialize()
    analysis = await chart_service.get_technical_analysis("BTCUSD", "1h")
    
    if analysis:
        print(f"  ✅ Chart service successfully returned analysis for BTCUSD")
        print(f"  Analysis contains {len(analysis)} characters")
    else:
        print(f"  ❌ Chart service failed for BTCUSD")
    
    await chart_service.cleanup()

async def main():
    """Run all tests"""
    # Load environment variables
    load_dotenv()
    
    await test_btcusd_format()

if __name__ == "__main__":
    asyncio.run(main()) 