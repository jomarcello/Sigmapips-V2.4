from trading_bot.services.chart_service.chart import ChartService
import asyncio

async def test():
    chart = ChartService()
    await chart.initialize()
    
    instruments = [
        'BTCUSD', 'ETHUSD', 'EURUSD', 'XAUUSD', 
        'US30', 'US500', 'WTIUSD'
    ]
    
    for instrument in instruments:
        market_type = await chart._detect_market_type(instrument)
        print(f"{instrument}: {market_type}")

if __name__ == "__main__":
    asyncio.run(test()) 