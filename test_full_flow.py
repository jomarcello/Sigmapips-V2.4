import logging
import asyncio
from trading_bot.services.chart_service.chart import ChartService

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_technical_analysis():
    print("Initializing ChartService...")
    chart = ChartService()
    await chart.initialize()
    
    # Test alleen crypto instrumenten
    instruments = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'DOGEUSDT', 'ADAUSDT']
    
    for instrument in instruments:
        print(f"\n\n=== Getting technical analysis for {instrument} ===")
        
        analysis = await chart.get_technical_analysis(instrument)
        print("\nTechnical Analysis Result:")
        print(analysis)
        print("="*50)

if __name__ == "__main__":
    asyncio.run(test_technical_analysis()) 