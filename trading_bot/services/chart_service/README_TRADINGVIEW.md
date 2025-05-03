# TradingView Provider voor Sigmapips Trading Bot

Dit document beschrijft hoe je de TradingView provider kunt gebruiken in de Sigmapips Trading Bot om real-time marktdata en technische analyse op te halen.

## Voordelen van de TradingView Provider

- **Real-time marktdata**: Direct toegang tot actuele prijzen van TradingView
- **Uitgebreide marktdekking**: Ondersteuning voor forex, aandelen, crypto, indices en commodities
- **Technische analyse**: Toegang tot TradingView's technische analyse en indicatoren
- **Geen fallback data**: Echte marktgegevens zonder gebruik van synthetische data
- **Breed aanbod indicatoren**: RSI, MACD, Stochastics, ADX en meer
- **Eenvoudige integratie**: Werkt met de bestaande provider interface

## Installatie

### Vereiste packages

Installeer de benodigde dependencies:

```bash
pip install tradingview-ta
```

Of voeg toe aan je requirements.txt:

```
tradingview-ta>=3.3.0
```

### Implementatie

De TradingView provider is geïmplementeerd in `tradingview_provider.py` en kan direct worden gebruikt als alternatieve data bron voor je trading bot.

## Gebruik

### Basis Gebruik

```python
from trading_bot.services.chart_service import TradingViewProvider

# Haal technische analyse op voor een instrument
analysis = await TradingViewProvider.get_technical_analysis("EURUSD", "1h")

# Haal marktdata op inclusief indicatoren
df, indicators = await TradingViewProvider.get_market_data("EURUSD", "1h")
```

### Ondersteunde Instrumenten

De TradingView provider ondersteunt een breed scala aan instrumenten:

1. **Forex paren**: EURUSD, GBPUSD, USDJPY, etc.
2. **Commodities**: XAUUSD (Goud), XAGUSD (Zilver), XTIUSD (WTI Olie), XBRUSD (Brent Olie)
3. **Aandelen**: AAPL, MSFT, GOOGL, AMZN, etc.
4. **Crypto**: BTCUSD, ETHUSD, etc.

**Opmerking**: Sommige indices zoals US500 (S&P 500) worden mogelijk niet ondersteund door de gratis TradingView API. Dit kan afhankelijk zijn van de toegangsrechten die TradingView biedt.

### Integratie met Chart Service

De TradingView provider wordt automatisch meegenomen in de `ChartService` class:

```python
from trading_bot.services.chart_service import ChartService

chart_service = ChartService()
# ChartService gebruikt nu automatisch TradingViewProvider als een van de providers
```

### Test Script

Er is een test script beschikbaar (`tests/test_tradingview_provider.py`) om de functionaliteit van de TradingView provider te testen:

```bash
python -m pytest trading_bot/tests/test_tradingview_provider.py -v
```

## Technische Details

### Caching

De TradingView provider implementeert een efficiënt caching mechanisme om het aantal API calls te beperken:

- Cache duurtijd is gebaseerd op het timeframe (korter voor 1m data, langer voor 1h data)
- Automatic cache invalidation op basis van timeframe
- Gedeelde cache tussen verschillende aanroepen

### Error Handling

De provider bevat robuuste foutafhandeling:

- Graceful fallback bij ontbrekende data
- Duidelijke logging van fouten
- Automatische retry mechanismen

## Voorbeeld Output

### Technische Analyse

```
--- Technische Analyse voor EURUSD ---
Aanbeveling: NEUTRAL
Huidige prijs: 1.08756
RSI: 54.56
MACD: 0.00042
Stoch K: 71.35
ADX: 15.77
```

### Marktdata

```
--- Marktdata voor EURUSD ---

Prijs Data:
                     Open    High     Low   Close  Volume
2023-06-01 12:34:56  1.088  1.0885  1.087  1.0876     0.0

Technische Indicatoren:
RSI: 54.56
MACD: 0.00042
ADX: 15.77
Aanbeveling: NEUTRAL
```

## Voordelen vs. Andere Providers

| Feature               | TradingView | Dukascopy | Google Finance | Yahoo Finance |
|-----------------------|-------------|-----------|----------------|---------------|
| Real-time data        | ✅          | ✅         | ✅             | ⚠️ (vertraagd) |
| Technische indicatoren| ✅          | ⚠️ (beperkt)| ❌             | ⚠️ (beperkt)   |
| Forex data            | ✅          | ✅         | ✅             | ✅            |
| Aandelen              | ✅          | ❌         | ✅             | ✅            |
| Indices               | ✅          | ⚠️ (beperkt)| ✅             | ✅            |
| Crypto                | ✅          | ❌         | ✅             | ✅            |
| Commodities           | ✅          | ⚠️ (beperkt)| ✅             | ✅            |
| Historische data      | ⚠️ (beperkt) | ✅         | ❌             | ✅            |
| API limieten          | ⚠️ (matig)   | ✅ (hoog)   | ⚠️ (laag)      | ⚠️ (matig)    | 