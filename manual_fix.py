#!/usr/bin/env python3
"""
Dit script laat je de Railway deployment logs handmatig toevoegen
en start Cursor om het probleem op te lossen.
"""

import os
import sys
import subprocess
import tempfile
import time
from pathlib import Path

def get_logs_from_user():
    """Vraag de gebruiker om de logs direct in te voeren of uit een bestand te laden"""
    print("===== Railway Deployment Logs Invoeren =====")
    print("1. Kopieer de logs van de Railway website")
    print("2. Plak de logs hieronder en druk op Enter")
    print("3. Sluit af met CTRL+D (of CMD+D op Mac)")
    print("====================================")
    
    logs = []
    try:
        while True:
            line = input()
            logs.append(line)
    except EOFError:
        pass
    
    return "\n".join(logs)

def create_prompt_file(logs):
    """Maak een prompt bestand voor Cursor met de echte logs"""
    temp_dir = tempfile.mkdtemp(prefix="railway-cursor-")
    
    # Maak het prompt bestand
    prompt_file = os.path.join(temp_dir, "railway_deployment_error.md")
    
    with open(prompt_file, "w") as f:
        f.write(f"""# Railway Deployment Error Analysis

Een Railway deployment is gefaald. Hier volgen de logs en instructies om dit op te lossen.

## Deployment Details
- Timestamp: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Railway Logs
```
{logs}
```

## Aanbevolen Acties
1. Analyseer de logs hierboven om de exacte fout te vinden
2. Identificeer de bestanden die moeten worden aangepast
3. Maak de benodigde wijzigingen om het probleem op te lossen

"""
    )
    
    # Maak het AI prompt bestand
    ai_prompt_file = os.path.join(temp_dir, "cursor_ai_prompt.md")
    
    with open(ai_prompt_file, "w") as f:
        f.write("""# DIRECTE ACTIE VEREIST - RAILWAY DEPLOYMENT GEFAALD

Help me de deployment error op te lossen:

1. Bekijk eerst het andere geopende document 'railway_deployment_error.md' voor de foutdetails en logs
2. Analyseer de logs en identificeer de exacte oorzaak van de fout
3. Zoek en open de relevante bestanden die aangepast moeten worden
4. Implementeer de juiste fix

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
        print("✅ Cursor is succesvol gestart om het deployment probleem op te lossen")
        return True
    except Exception as e:
        print(f"❌ Fout bij starten van Cursor: {str(e)}")
        return False

def main():
    # Bepaal het pad naar de workspace
    workspace_path = os.getcwd()
    
    # Haal de logs op van de gebruiker
    print("Voer de Railway deployment logs in:")
    logs = get_logs_from_user()
    
    if not logs.strip():
        print("Geen logs ingevoerd. Afsluiten.")
        return
    
    # Maak de prompt bestanden
    prompt_file, ai_prompt_file = create_prompt_file(logs)
    
    # Start Cursor
    success = launch_cursor(workspace_path, prompt_file, ai_prompt_file)
    
    if success:
        print("\n" + "="*80)
        print(f"CURSOR IS GESTART om het deployment probleem op te lossen")
        print(f"Bekijk Cursor voor verdere instructies")
        print("="*80 + "\n")
    else:
        print("❌ Kon Cursor niet starten.")

if __name__ == "__main__":
    main() 