#!/usr/bin/env python3
"""
CLI tool for testing Yahoo Finance integration
Usage: python cli_test_yahoo_finance.py --instrument EURUSD --timeframe 1h
"""

import argparse
import logging
import sys
import os
import asyncio
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Ensure trading_bot package is in the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import YahooFinanceProvider
from trading_bot.services.chart_service.yfinance_provider import YahooFinanceProvider

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test Yahoo Finance integration')
    parser.add_argument('--instrument', '-i', type=str, default='EURUSD', 
                        help='Instrument to test (default: EURUSD)')
    parser.add_argument('--timeframe', '-t', type=str, default='1h',
                        help='Timeframe (default: 1h, options: 1h, 4h, 1d)')
    parser.add_argument('--data', '-d', action='store_true',
                        help='Get market data only (default: chart and data)')
    parser.add_argument('--chart', '-c', action='store_true',
                        help='Get chart only (default: chart and data)')
    return parser.parse_args()

async def test_market_data_async(instrument, timeframe='1h'):
    """Test getting market data from Yahoo Finance asynchronously"""
    logger.info(f"Getting market data for {instrument} ({timeframe})")
    
    try:
        # Check if get_market_data is an async method or not
        if asyncio.iscoroutinefunction(YahooFinanceProvider.get_market_data):
            # Call with await
            logger.info("Calling get_market_data with await (async)")
            data, metadata = await YahooFinanceProvider.get_market_data(instrument)
        else:
            # Call directly
            logger.info("Calling get_market_data directly (sync)")
            data, metadata = YahooFinanceProvider.get_market_data(instrument)
        
        if data is not None and not data.empty and metadata:
            logger.info(f"Successfully retrieved market data: {len(data)} rows")
            logger.info(f"Latest data timestamp: {data.index[-1]}")
            
            # Display latest data point and some metadata
            logger.info("\nLatest price data:")
            for key in ['close', 'open', 'high', 'low']:
                if key in metadata:
                    logger.info(f"  {key.capitalize()}: {metadata[key]}")
            
            # Display some indicators
            logger.info("\nIndicators:")
            for key in ['rsi', 'ema_20', 'ema_50', 'ema_200']:
                if key in metadata:
                    logger.info(f"  {key.upper()}: {metadata[key]:.4f}")
            
            return True
        else:
            logger.error("Failed to retrieve market data")
            return False
    
    except Exception as e:
        logger.error(f"Error retrieving market data: {str(e)}")
        logger.error(f"Traceback: {logging.traceback.format_exc()}")
        return False

def test_chart(instrument, timeframe='1h'):
    """Test generating a chart with Yahoo Finance data"""
    logger.info(f"Generating chart for {instrument} ({timeframe})")
    
    try:
        # Generate chart
        chart_bytes = YahooFinanceProvider.get_chart(instrument, timeframe, fullscreen=True)
        
        if chart_bytes and len(chart_bytes) > 1000:
            logger.info(f"Successfully generated chart: {len(chart_bytes)} bytes")
            
            # Save chart image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"chart_{instrument.lower()}_{timeframe}_{timestamp}.png"
            
            with open(output_file, "wb") as f:
                f.write(chart_bytes)
            logger.info(f"Chart saved to {output_file}")
            
            return True
        else:
            logger.error("Failed to generate chart")
            return False
    
    except Exception as e:
        logger.error(f"Error generating chart: {str(e)}")
        logger.error(f"Traceback: {logging.traceback.format_exc()}")
        return False

async def main_async():
    """Async main function"""
    args = parse_args()
    instrument = args.instrument
    timeframe = args.timeframe
    
    logger.info(f"Testing Yahoo Finance integration for {instrument} ({timeframe})")
    
    # If no specific option is selected, run both tests
    if not args.data and not args.chart:
        data_success = await test_market_data_async(instrument, timeframe)
        chart_success = test_chart(instrument, timeframe)
        
        if data_success and chart_success:
            logger.info("All tests passed!")
        else:
            logger.warning("Some tests failed. Check the logs above.")
    
    # Run data test only
    elif args.data:
        if await test_market_data_async(instrument, timeframe):
            logger.info("Market data test passed!")
        else:
            logger.warning("Market data test failed.")
    
    # Run chart test only
    elif args.chart:
        if test_chart(instrument, timeframe):
            logger.info("Chart test passed!")
        else:
            logger.warning("Chart test failed.")

def main():
    """Main wrapper to run async main"""
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 