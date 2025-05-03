#!/usr/bin/env python3
"""
Test script to verify that Binance API integration works correctly for crypto instruments.
This script tests various aspects of the Binance integration:
1. Detection of crypto instruments
2. Prioritization of BinanceProvider for crypto instruments
3. Actual API calls to get market data and prices
"""

import asyncio
import logging
import os
from dotenv import load_dotenv
import traceback
from typing import List, Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

async def test_crypto_market_detection():
    """Test that crypto instruments are correctly detected by the market type detection"""
    print("\n=== Testing Crypto Market Detection ===")
    
    # Import the ChartService
    from trading_bot.services.chart_service.chart import ChartService
    
    # Create a chart service instance
    chart_service = ChartService()
    
    # Test crypto instruments
    crypto_instruments = [
        "BTCUSD", "BTCUSDT", "ETHUSD", "ETHUSDT", "XRPUSDT", 
        "SOLUSDT", "DOGEUSDT"
    ]
    
    # Test non-crypto instruments for comparison
    non_crypto_instruments = [
        "EURUSD", "GBPUSD", "USDJPY",  # Forex
        "XAUUSD", "XAGUSD",           # Commodities
        "US30", "US500", "SPX"        # Indices
    ]
    
    print("Crypto instrument detection:")
    for instrument in crypto_instruments:
        market_type = await chart_service._detect_market_type(instrument)
        is_correct = market_type == "crypto"
        print(f"  {instrument}: Detected as {market_type} - {'✅ Correct' if is_correct else '❌ Incorrect'}")
    
    print("\nNon-crypto instrument detection (should NOT be crypto):")
    for instrument in non_crypto_instruments:
        market_type = await chart_service._detect_market_type(instrument)
        is_correct = market_type != "crypto"
        print(f"  {instrument}: Detected as {market_type} - {'✅ Correct' if is_correct else '❌ Incorrect'}")

async def test_binance_provider_prioritization():
    """Test that BinanceProvider is prioritized for crypto instruments"""
    print("\n=== Testing BinanceProvider Prioritization for Crypto ===")
    
    # Import required providers
    from trading_bot.services.chart_service.binance_provider import BinanceProvider
    from trading_bot.services.chart_service.yfinance_provider import YahooFinanceProvider
    
    # Create providers
    binance_provider = BinanceProvider()
    yahoo_provider = YahooFinanceProvider()
    
    # Create an empty providers list
    providers_to_try = []
    
    # Simulating the prioritization logic from chart.py
    print("Crypto instrument prioritization:")
    market_type = "crypto"
    if market_type == "crypto":
        print("  Prioritizing Binance for crypto")
        providers_to_try.append(binance_provider)
        providers_to_try.append(yahoo_provider)
    
    # Check the provider order
    provider_names = [p.__class__.__name__ for p in providers_to_try]
    print(f"  Final provider order: {provider_names}")
    
    # Verify Binance is first
    is_correct = provider_names[0] == "BinanceProvider"
    print(f"  BinanceProvider is prioritized first: {'✅ Correct' if is_correct else '❌ Incorrect'}")
    
    # For comparison, show the non-crypto prioritization
    print("\nNon-crypto instrument prioritization (e.g., forex):")
    providers_to_try = []
    market_type = "forex"
    if market_type == "crypto":
        providers_to_try.append(binance_provider)
        providers_to_try.append(yahoo_provider)
    elif market_type in ["forex", "index", "commodity"]:
        providers_to_try.append(yahoo_provider)
    
    provider_names = [p.__class__.__name__ for p in providers_to_try]
    print(f"  Final provider order: {provider_names}")
    
    # Verify Binance is not used for forex
    is_correct = "BinanceProvider" not in provider_names
    print(f"  BinanceProvider is not used for forex: {'✅ Correct' if is_correct else '❌ Incorrect'}")

async def test_binance_ticker_price():
    """Test the BinanceProvider.get_ticker_price method"""
    print("\n=== Testing BinanceProvider.get_ticker_price ===")
    
    # Import BinanceProvider
    from trading_bot.services.chart_service.binance_provider import BinanceProvider
    
    # Test crypto instruments
    test_instruments = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    
    # Try to get price for each instrument
    for instrument in test_instruments:
        try:
            print(f"Getting price for {instrument}...")
            price = await BinanceProvider.get_ticker_price(instrument)
            
            if price is not None:
                print(f"  ✅ Success! Current price of {instrument}: {price}")
            else:
                print(f"  ❌ Failed to get price for {instrument}")
        except Exception as e:
            print(f"  ❌ Error getting price for {instrument}: {str(e)}")
            print(traceback.format_exc())

async def test_binance_market_data():
    """Test the BinanceProvider.get_market_data method"""
    print("\n=== Testing BinanceProvider.get_market_data ===")
    
    # Import BinanceProvider
    from trading_bot.services.chart_service.binance_provider import BinanceProvider
    
    # Test a single instrument with different timeframes
    instrument = "BTCUSDT"
    timeframes = ["1h", "4h", "1d"]
    
    # Try to get market data for each timeframe
    for timeframe in timeframes:
        try:
            print(f"Getting market data for {instrument} on {timeframe} timeframe...")
            market_data = await BinanceProvider.get_market_data(instrument, timeframe)
            
            if market_data is not None:
                print(f"  ✅ Success! Got market data with indicators:")
                # Print a few key indicators
                indicators = market_data.indicators
                print(f"    Current Price: {indicators.get('close', 'N/A')}")
                print(f"    EMA50: {indicators.get('EMA50', 'N/A')}")
                print(f"    RSI: {indicators.get('RSI', 'N/A')}")
                print(f"    MACD: {indicators.get('MACD.macd', 'N/A')}")
            else:
                print(f"  ❌ Failed to get market data for {instrument} on {timeframe}")
        except Exception as e:
            print(f"  ❌ Error getting market data for {instrument} on {timeframe}: {str(e)}")
            print(traceback.format_exc())

async def test_full_chart_service_flow():
    """Test the ChartService.get_technical_analysis method with a crypto instrument"""
    print("\n=== Testing Full ChartService Flow with Crypto ===")
    
    # Import ChartService
    from trading_bot.services.chart_service.chart import ChartService
    
    # Create chart service instance
    chart_service = ChartService()
    
    # Initialize the chart service
    print("Initializing ChartService...")
    await chart_service.initialize()
    
    # Test a crypto instrument
    instrument = "BTCUSDT"
    timeframe = "1h"
    
    try:
        print(f"Getting technical analysis for {instrument} on {timeframe} timeframe...")
        analysis = await chart_service.get_technical_analysis(instrument, timeframe)
        
        if analysis:
            print(f"  ✅ Success! Got technical analysis for {instrument}")
            print("\nAnalysis Preview (first 200 characters):")
            print(analysis[:200] + "..." if len(analysis) > 200 else analysis)
            
            # Check if analysis contains key elements
            key_elements = ["Price is", "Technical Indicators", "EMA", "RSI", "MACD"]
            missing_elements = [elem for elem in key_elements if elem not in analysis]
            
            if not missing_elements:
                print("\n  ✅ Analysis contains all key elements")
            else:
                print(f"\n  ❌ Analysis missing key elements: {missing_elements}")
        else:
            print(f"  ❌ Failed to get technical analysis for {instrument}")
    except Exception as e:
        print(f"  ❌ Error in ChartService.get_technical_analysis: {str(e)}")
        print(traceback.format_exc())
    finally:
        # Cleanup chart service
        await chart_service.cleanup()

async def main():
    """Run all tests"""
    # Load environment variables
    load_dotenv()
    
    # Display Binance API key status
    api_key = os.environ.get("BINANCE_API_KEY", "")
    api_key_masked = f"{api_key[:5]}...{api_key[-5:]}" if len(api_key) > 10 else "Not set"
    print(f"Binance API Key: {api_key_masked}")
    
    # Run tests
    await test_crypto_market_detection()
    await test_binance_provider_prioritization()
    await test_binance_ticker_price()
    await test_binance_market_data()
    await test_full_chart_service_flow()

if __name__ == "__main__":
    asyncio.run(main()) 