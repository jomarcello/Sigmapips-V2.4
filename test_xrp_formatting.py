#!/usr/bin/env python3
import asyncio
import sys
import os

# Zorg dat we de modules kunnen importeren
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from trading_bot.services.chart_service.chart import ChartService

async def test_xrp_precision():
    # Instantieer de ChartService
    service = ChartService()
    
    # Test de precisie functie voor verschillende instrumenten
    print("Precisie voor verschillende instrumenten:")
    print(f"XRPUSD precisie: {service._get_instrument_precision('XRPUSD')} decimalen")
    print(f"BTCUSD precisie: {service._get_instrument_precision('BTCUSD')} decimalen")
    print(f"ETHUSD precisie: {service._get_instrument_precision('ETHUSD')} decimalen")
    print(f"EURUSD precisie: {service._get_instrument_precision('EURUSD')} decimalen")
    
    # Test de technische analyse formattering voor XRP
    analysis = await service.get_technical_analysis("XRPUSD", "1h")
    print("\nGegenereerde analyse voor XRPUSD:")
    print(analysis)
    
    print("\nTest voltooid.")

# Voer de test uit
if __name__ == "__main__":
    asyncio.run(test_xrp_precision()) 