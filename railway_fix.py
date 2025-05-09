#!/usr/bin/env python3
"""
Dit script maakt een tekstbestand waar je de Railway logs in kunt plakken,
en gebruikt dan Cursor om het probleem op te lossen.
"""

import os
import sys
import subprocess
import tempfile
import time
from pathlib import Path

def open_logs_file():
    """Maak een tijdelijk bestand voor de logs en open het in de standaard tekstverwerker"""
    log_dir = os.path.join(os.getcwd(), "railway_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"railway_logs_{timestamp}.txt")
    
    # Maak het bestand aan met een sjabloon
    with open(log_file, "w") as f:
        f.write("""# Plak hier de Railway logs van je gefaalde deployment

1. Ga naar railway.app en log in
2. Ga naar je project en klik op de gefaalde deployment
3. Bekijk de logs en kopieer de relevante foutmeldingen
4. Plak de foutmeldingen hieronder
5. Sla dit bestand op (Cmd+S)
6. Sluit dit bestand

--------------------
PLAK LOGS HIERONDER:
--------------------




""")
    
    # Open het bestand in de standaard tekstverwerker
    if sys.platform == "darwin":  # macOS
        subprocess.run(["open", log_file])
    elif sys.platform == "win32":  # Windows
        os.startfile(log_file)
    else:  # Linux/Unix
        subprocess.run(["xdg-open", log_file])
    
    print(f"Een tekstbestand is geopend. Plak de Railway logs, sla op en sluit het bestand.")
    print(f"Het bestand is opgeslagen in: {log_file}")
    
    return log_file

def wait_for_file_edit(file_path, original_mtime=None):
    """Wacht tot het bestand is bewerkt en opgeslagen"""
    if original_mtime is None:
        original_mtime = os.path.getmtime(file_path)
    
    print("Wachten tot je de logs hebt toegevoegd en het bestand hebt opgeslagen...")
    
    while True:
        try:
            current_mtime = os.path.getmtime(file_path)
            if current_mtime > original_mtime:
                # Het bestand is bewerkt
                time.sleep(1)  # Wacht nog een seconde om er zeker van te zijn dat het volledig is opgeslagen
                return True
        except Exception:
            pass
        
        time.sleep(0.5)

def create_prompt_file(logs_file):
    """Maak een prompt bestand voor Cursor op basis van de logs"""
    with open(logs_file, "r") as f:
        logs_content = f.read()
    
    # Haal alleen de logs op (alles na de instructietekst)
    if "PLAK LOGS HIERONDER:" in logs_content:
        _, logs_content = logs_content.split("PLAK LOGS HIERONDER:", 1)
    
    logs_content = logs_content.strip()
    
    if not logs_content:
        print("⚠️ Geen logs gevonden in het bestand. Voeg de logs toe en probeer opnieuw.")
        return None, None
    
    # Maak een tijdelijke directory voor de prompt bestanden
    temp_dir = tempfile.mkdtemp(prefix="railway-cursor-")
    
    # Maak het prompt bestand
    prompt_file = os.path.join(temp_dir, "railway_deployment_error.md")
    
    with open(prompt_file, "w") as f:
        f.write(f"""# Railway Deployment Error Analysis

Een Railway deployment is gefaald. Hier volgt de analyse en instructies om dit op te lossen.

## Deployment Details
- Timestamp: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Railway Logs
```
{logs_content}
```

## Analyse-instructies
1. Analyseer de logs hierboven om de oorzaak van de gefaalde deployment te identificeren
2. Zoek naar foutmeldingen, ontbrekende afhankelijkheid, syntaxfouten, etc.
3. Stel een oplossing voor om het probleem op te lossen

"""
    )
    
    # Maak het AI prompt bestand
    ai_prompt_file = os.path.join(temp_dir, "cursor_ai_prompt.md")
    
    with open(ai_prompt_file, "w") as f:
        f.write("""# DIRECTE ACTIE VEREIST - RAILWAY DEPLOYMENT GEFAALD

Help me de Railway deployment fout op te lossen:

1. Bekijk eerst het andere geopende document 'railway_deployment_error.md' voor de volledige logs
2. Analyseer de logs en identificeer de exacte fout
3. Zoek en open de relevante bestanden in het project
4. Los het probleem op met een concrete fix
5. Test de oplossing lokaal indien mogelijk

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
    
    # Open een tekstbestand om de logs in te plakken
    logs_file = open_logs_file()
    original_mtime = os.path.getmtime(logs_file)
    
    # Wacht tot het bestand is bewerkt
    wait_for_file_edit(logs_file, original_mtime)
    
    # Maak de prompt bestanden
    prompt_file, ai_prompt_file = create_prompt_file(logs_file)
    
    if not prompt_file or not ai_prompt_file:
        return
    
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