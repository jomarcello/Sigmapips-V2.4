#!/bin/bash

# deployment_watcher.sh
# Een eenvoudig script om Railway deployments te bewaken en Cursor te starten als er een gefaalde deployment is

# Activeer de virtuele omgeving
source cursor_bridge_venv/bin/activate

echo "Railway Deployment Watcher Gestart"
echo "Dit script monitort Railway deployments en start Cursor wanneer er een fout optreedt"
echo "Druk Ctrl+C om te stoppen"
echo ""

LOG_FILE="deployment_watcher.log"
echo "$(date) - Railway Deployment Watcher Gestart" > "$LOG_FILE"

while true; do
  echo "Controle van deployments..."
  
  # Uitvoeren in test modus om te zien of Cursor wordt gestart
  python railway-auto-fix/cursor_bridge_enhanced.py --test-logs railway-auto-fix/test_error.log
  
  # Wacht 30 seconden voordat we weer controleren
  echo "Wachten voor volgende controle (30 seconden)..."
  sleep 30
done 