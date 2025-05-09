#!/usr/bin/env python3
"""
Dit script is speciaal gemaakt om het probleem met circulaire imports
tussen ChartService en MarketService op te lossen.
"""

import os
import sys
import requests
import json
import subprocess
import tempfile
import time
from pathlib import Path

def create_prompt_file(error_message):
    """Maak een prompt bestand voor Cursor met specifieke instructies voor het circulair import probleem"""
    temp_dir = tempfile.mkdtemp(prefix="railway-cursor-")
    
    # Maak het prompt bestand
    prompt_file = os.path.join(temp_dir, "railway_deployment_error.md")
    
    with open(prompt_file, "w") as f:
        f.write(f"""# Railway Deployment Error Analysis

Een Railway deployment is gefaald vanwege circulaire imports. Hier volgt de analyse en instructies om dit op te lossen.

## Deployment Details
- Timestamp: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Gedetecteerde Fouten
```
Circulaire imports tussen ChartService.py en MarketService.py

Het probleem is waarschijnlijk dat deze twee services elkaar importeren, wat leidt tot een
circular import error tijdens de deployment.
```

## Aanbevolen Acties
1. Bekijk de bestanden api/services/ChartService.py en api/services/MarketService.py
2. Zoek waar deze modules elkaar importeren
3. Los het probleem op met een van deze methoden:
   - Verplaats gedeelde functionaliteit naar een aparte module
   - Gebruik lazy imports (importeer binnen een functie)
   - Refactor de code om de circulaire afhankelijkheid te verwijderen

Bijvoorbeeld, als ChartService MarketService importeert en MarketService ChartService importeert:
- Maak een nieuwe module `shared_services.py` voor gedeelde functies
- Of importeer in functies in plaats van op module niveau
- Of gebruik dependency injection

"""
    )
    
    # Maak het AI prompt bestand
    ai_prompt_file = os.path.join(temp_dir, "cursor_ai_prompt.md")
    
    with open(ai_prompt_file, "w") as f:
        f.write("""# DIRECTE ACTIE VEREIST - RAILWAY DEPLOYMENT GEFAALD

Help me de deployment error met circulaire imports op te lossen:

1. Bekijk eerst het andere geopende document 'railway_deployment_error.md' voor de foutdetails
2. Zoek en open de volgende bestanden en analyseer de circulaire import:
   - api/services/ChartService.py
   - api/services/MarketService.py
3. Los de circulaire import op met een van deze methoden:
   - Verplaats gedeelde functionaliteit naar een aparte module
   - Gebruik lazy imports (importeer binnen een functie)
   - Gebruik een basisklasse of interface patroon

Implementeer de oplossing direct, commit en push zodat Railway opnieuw kan deployen.

URGENT: Dit blokkeert de productie deployment!
"""
        )
    
    return prompt_file, ai_prompt_file

def launch_cursor(workspace_path, prompt_file, ai_prompt_file):
    """Start Cursor met de prompt bestanden"""
    cursor_path = "/Applications/Cursor.app/Contents/MacOS/Cursor"
    
    if not os.path.exists(cursor_path):
        print(f"⚠️ Cursor niet gevonden op {cursor_path}")
        return False
    
    command = [
        cursor_path,
        workspace_path,
        ai_prompt_file,
        prompt_file
    ]
    
    try:
        subprocess.Popen(command)
        print("✅ Cursor is succesvol gestart om het circulaire import probleem op te lossen")
        return True
    except Exception as e:
        print(f"❌ Fout bij starten van Cursor: {str(e)}")
        return False

def main():
    # Bepaal het pad naar de workspace
    workspace_path = os.getcwd()
    
    # Maak de prompt bestanden
    prompt_file, ai_prompt_file = create_prompt_file("Circulaire import error")
    
    # Start Cursor
    success = launch_cursor(workspace_path, prompt_file, ai_prompt_file)
    
    if success:
        print("\n" + "="*80)
        print(f"CURSOR IS GESTART om het circulaire import probleem op te lossen")
        print(f"Bekijk Cursor voor verdere instructies")
        print("="*80 + "\n")
    else:
        print("❌ Kon Cursor niet starten.")

if __name__ == "__main__":
    main() 