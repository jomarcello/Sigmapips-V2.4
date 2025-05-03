#!/usr/bin/env python3
import asyncio
import sys
import os
import logging

# Zorg dat we de modules kunnen importeren
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Logging instellen
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("XRP-Price-Test")

async def test_binance_api_direct():
    """Directe test van de Binance API zonder de ChartService"""
    try:
        from trading_bot.services.chart_service.binance_provider import BinanceProvider

        # Test 1: get_ticker_price (simpele ticker API)
        ticker_price = await BinanceProvider.get_ticker_price("XRPUSDT")
        logger.info(f"Binance Ticker API - XRPUSDT Price: {ticker_price}")

        # Test 2: get_market_data (uitgebreide data met indicators)
        market_data = await BinanceProvider.get_market_data("XRPUSDT", "1h")
        if market_data and hasattr(market_data, 'indicators') and 'close' in market_data.indicators:
            price = market_data.indicators['close']
            logger.info(f"Binance Market Data API - XRPUSDT close price: {price}")
            logger.info(f"All indicators: {market_data.indicators}")
        else:
            logger.error(f"Binance Market Data API - Failed to get data for XRPUSDT")
            logger.info(f"Raw market_data result: {market_data}")

        # Test 3: Probeer XRP zonder USDT suffix
        market_data_xrp = await BinanceProvider.get_market_data("XRP", "1h")
        if market_data_xrp and hasattr(market_data_xrp, 'indicators') and 'close' in market_data_xrp.indicators:
            price = market_data_xrp.indicators['close']
            logger.info(f"Binance Market Data API - XRP (zonder USDT) close price: {price}")
        else:
            logger.error(f"Binance Market Data API - Failed to get data for XRP without suffix")

        return ticker_price
    except Exception as e:
        logger.error(f"Error testing Binance API: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

async def test_chart_service():
    """Test de ChartService _fetch_crypto_price functie"""
    try:
        from trading_bot.services.chart_service.chart import ChartService
        
        chart_service = ChartService()
        # Test voor XRPUSD (zoals in de bot gebruikt)
        xrp_price = await chart_service._fetch_crypto_price("XRP")
        logger.info(f"ChartService._fetch_crypto_price('XRP') result: {xrp_price}")
        
        # Test voor volledige XRPUSD
        xrpusd_price = await chart_service._fetch_crypto_price("XRPUSD")
        logger.info(f"ChartService._fetch_crypto_price('XRPUSD') result: {xrpusd_price}")
        
        return xrp_price
    except Exception as e:
        logger.error(f"Error testing ChartService: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

async def main():
    """Hoofdfunctie die alle tests uitvoert"""
    logger.info("===== STARTING XRP PRICE TESTING =====")
    
    # Directe Binance API test
    binance_price = await test_binance_api_direct()
    
    # ChartService test
    chart_service_price = await test_chart_service()
    
    # Vergelijk resultaten
    logger.info("\n===== TEST RESULTS SUMMARY =====")
    logger.info(f"Binance API direct price: {binance_price}")
    logger.info(f"ChartService price: {chart_service_price}")
    
    if binance_price and chart_service_price:
        diff = abs(binance_price - chart_service_price)
        if diff < 0.05:  # Kleine marge voor fluctuaties
            logger.info("✅ Prices match within acceptable range")
        else:
            logger.warning(f"❌ Significant price difference: {diff}")
    else:
        logger.warning("❌ Could not compare prices because one or both tests failed")
    
    logger.info("===== TEST COMPLETED =====")

if __name__ == "__main__":
    asyncio.run(main()) 