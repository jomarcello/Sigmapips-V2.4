#!/bin/bash
# Setup script voor Railway-Cursor Bridge

echo "Railway-Cursor Bridge installeren voor automatische deployment fixes..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is niet geïnstalleerd. Installeer dit eerst."
    exit 1
fi

# Pad naar de cursor_bridge_enhanced.py script
SCRIPT_PATH="$(pwd)/railway-auto-fix/cursor_bridge_enhanced.py"
VENV_PATH="$(pwd)/cursor_bridge_venv"
PYTHON_VENV="$VENV_PATH/bin/python"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Kan cursor_bridge_enhanced.py niet vinden op $SCRIPT_PATH"
    exit 1
fi

# Controleer of virtuele omgeving bestaat, anders maak aan
if [ ! -d "$VENV_PATH" ]; then
    echo "Virtuele omgeving aanmaken..."
    python3 -m venv "$VENV_PATH"
fi

# Installeer afhankelijkheden in virtuele omgeving
echo "Benodigde Python packages installeren in virtuele omgeving..."
"$VENV_PATH/bin/pip" install requests python-dotenv plyer

# Ask for Railway Token
read -p "Voer je Railway API Token in: " railway_token
read -p "Voer je GitHub Token in: " github_token
read -p "Voer je Railway Project ID in: " railway_project_id
read -p "Voer de GitHub repository in (bijv. jomarcello/Sigmapips-V2.4): " github_repo

# Update het pad naar Python en de script in de plist file
echo "Python pad: $PYTHON_VENV"
echo "Script pad: $SCRIPT_PATH"

# Maak een aangepaste plist file in de hoofddirectory
cat > com.railway.cursor-bridge.plist << EOL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.railway.cursor-bridge</string>
    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON_VENV}</string>
        <string>${SCRIPT_PATH}</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>RAILWAY_TOKEN</key>
        <string>${railway_token}</string>
        <key>GITHUB_TOKEN</key>
        <string>${github_token}</string>
        <key>GITHUB_REPO</key>
        <string>${github_repo}</string>
        <key>RAILWAY_PROJECT_ID</key>
        <string>${railway_project_id}</string>
        <key>POLL_INTERVAL_SECONDS</key>
        <string>60</string>
        <key>LOCAL_REPO_PATH</key>
        <string>$(pwd)</string>
        <key>CURSOR_PATH</key>
        <string>/Applications/Cursor.app/Contents/MacOS/Cursor</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardErrorPath</key>
    <string>$(pwd)/cursor_bridge_error.log</string>
    <key>StandardOutPath</key>
    <string>$(pwd)/cursor_bridge_output.log</string>
</dict>
</plist>
EOL

# Copy plist to LaunchAgents
echo "LaunchAgent service installeren..."
mkdir -p ~/Library/LaunchAgents
cp com.railway.cursor-bridge.plist ~/Library/LaunchAgents/

# Verwijder eventuele bestaande service
launchctl unload ~/Library/LaunchAgents/com.railway.cursor-bridge.plist 2>/dev/null

# Load the service
echo "Service starten..."
launchctl load ~/Library/LaunchAgents/com.railway.cursor-bridge.plist

echo "Railway-Cursor Bridge is succesvol geinstalleerd!"
echo "Het zal nu automatisch je Railway deployments monitoren en Cursor starten wanneer een deployment faalt."
echo "Logs zijn beschikbaar op:"
echo "  - $(pwd)/cursor_bridge_output.log"
echo "  - $(pwd)/cursor_bridge_error.log"
echo ""
echo "Om de service te stoppen: launchctl unload ~/Library/LaunchAgents/com.railway.cursor-bridge.plist"
echo "Om de service opnieuw te starten: launchctl load ~/Library/LaunchAgents/com.railway.cursor-bridge.plist"
echo ""
echo "Om te testen of het werkt, kun je het script handmatig uitvoeren met:"
echo "$PYTHON_VENV $SCRIPT_PATH --test-logs railway-auto-fix/test_error.log" 