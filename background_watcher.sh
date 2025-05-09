#!/bin/bash

# background_watcher.sh
# Start deployment watcher in de achtergrond en houdt het draaiende

# Pad naar de workspace
WORKSPACE="/Users/jovannitilborg/Downloads/Sigmapips-V2-5.4-main"
cd "$WORKSPACE"

# Start het script met nohup om het in de achtergrond te laten draaien
# Alle output wordt opgeslagen in watcher.log
nohup ./deployment_watcher.sh > watcher.log 2>&1 &

# Haal het proces ID op
PID=$!

# Sla het proces ID op zodat we het later kunnen stoppen
echo $PID > watcher.pid

echo "Deployment watcher is gestart in de achtergrond met proces ID: $PID"
echo "Logs worden opgeslagen in watcher.log"
echo "Om de watcher te stoppen, voer uit: kill $(cat watcher.pid)" 