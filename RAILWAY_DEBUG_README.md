# AI-Powered Railway Debugging System

Dit systeem integreert Railway logs met geavanceerde debugging-functionaliteit en AI-analyse om snel problemen op te lossen.

## Functionaliteiten

1. **Persistente roterende logbestanden** - Automatisch logbestanden aanmaken en beheren
2. **Railway log integratie** - Fetch en analyse van productielogs direct vanuit Railway 
3. **Geautomatiseerde foutanalyse** - Identificeert potentiële problemen en stelt oplossingen voor
4. **Gerichte logging** - Voeg logging toe aan specifieke bestanden of functies
5. **Live debugging sessies** - Monitor lopende applicaties en detecteer problemen in real-time

## Installatie

Zorg dat alle benodigde pakketten zijn geïnstalleerd:

```bash
pip install -r requirements.txt
```

## Gebruik

### 1. Debug een specifiek probleem

```bash
python debug_railway_with_ai.py --service sigmapips-bot debug "Login werkt niet voor bestaande gebruikers"
```

### 2. Start een live debugging sessie

```bash
python debug_railway_with_ai.py --service sigmapips-bot live "Webhook ontvangt geen gegevens"
```

### 3. Analyseer de resultaten van een live sessie

```bash
python debug_railway_with_ai.py analyze railway_logs/railway_stream_20240509_123456.txt "Webhook ontvangt geen gegevens"
```

### 4. Voeg gerichte logging toe

```bash
python debug_railway_with_ai.py log trading_bot/main.py --function process_signal
```

## Workflow voor effectief debuggen

### Stap 1: Creëer een persistente log

Het systeem maakt automatisch roterende logbestanden aan met een beperkte grootte om schijfruimte te besparen. Deze bestanden worden opgeslagen in de `logs/` directory en zijn uitgesloten van Git.

### Stap 2: Specificeer het probleem en log alle relevante I/O

1. Identificeer het probleem zo specifiek mogelijk
2. Voeg gerichte logging toe aan de relevante componenten:

```bash
python debug_railway_with_ai.py log trading_bot/bot/signal_processor.py
```

### Stap 3: Forceer de fout en laat AI de logs analyseren

1. Reproduceer het probleem in de productieomgeving
2. Verzamel de logs:

```bash
python debug_railway_with_ai.py --service sigmapips-bot debug "Signaal verwerking faalt na API-aanroep"
```

3. Bekijk de analyse en aanbevelingen:

```json
{
  "success": true,
  "errors_found": true,
  "num_issues": 3,
  "recommendations": {
    "possible_issues": [
      {
        "error_text": "KeyError: 'signal_data'",
        "related_files": ["trading_bot/bot/signal_processor.py"],
        "related_functions": ["process_signal", "validate_signal"]
      }
    ],
    "suggested_files": ["trading_bot/bot/signal_processor.py"],
    "suggested_functions": ["process_signal", "validate_signal"]
  }
}
```

## Tips voor optimaal gebruik

1. **Wees specifiek** - Hoe specifieker het probleem, hoe beter de analyse
2. **Iteratief proces** - Voeg logica toe, analyseer, verfijn, herhaal
3. **Combineer met Railway-logs** - De kracht zit in de combinatie van lokale en Railway-logs
4. **Behoud logs** - Bewaar belangrijke logs voor toekomstige referentie
5. **Deel logs met AI** - Gebruik logs om betere assistentie van AI te krijgen

## Bestandsoverzicht

- `trading_bot/utils/debug_logger.py` - Core logging functionaliteit
- `utils/railway_log_analyzer.py` - Railway log extractie en analyse
- `utils/debug_integrator.py` - Hulpmiddel om logging toe te voegen aan code
- `debug_railway_with_ai.py` - Hoofdscript voor AI-debug functionaliteit

## Voorbeeldcase: Login probleem oplossen

1. **Probleem identificeren**: "Gebruikers kunnen niet inloggen met bestaande accounts"

2. **Logging toevoegen**:
   ```bash
   python debug_railway_with_ai.py log trading_bot/bot/auth.py
   ```

3. **Deploy bijgewerkte code**

4. **Logs verzamelen wanneer het probleem optreedt**:
   ```bash
   python debug_railway_with_ai.py --service sigmapips-bot debug "Login werkt niet voor bestaande gebruikers"
   ```

5. **Analyse bekijken**:
   - Mogelijke problemen en gerelateerde bestanden/functies
   - Specifieke foutberichten en hun context

6. **Fix implementeren** gebaseerd op AI-analyse 