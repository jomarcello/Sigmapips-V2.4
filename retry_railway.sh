#!/bin/bash

echo "Script voor het opnieuw testen van de Railway Cursor Bridge"

# Stop de huidige service als deze actief is
echo "Stoppen van de huidige service..."
launchctl unload ~/Library/LaunchAgents/com.railway.cursor-bridge.plist 2>/dev/null

# Test of de Railway API token werkt
read -p "Voer je Railway API Token in: " railway_token
read -p "Voer je Railway Project ID in: " railway_project_id

echo "Testen van de Railway API verbinding..."
source cursor_bridge_venv/bin/activate

# Maak een tijdelijk Python script om de API te testen
cat > test_railway_api.py << EOF
import os
import requests
import json

token = os.environ.get('RAILWAY_TOKEN')
project_id = os.environ.get('RAILWAY_PROJECT_ID')

query = '''
query {
  project(id: "%s") {
    name
    services {
      edges {
        node {
          name
        }
      }
    }
  }
}
''' % project_id

headers = {
    'Authorization': f'Bearer {token}',
    'Content-Type': 'application/json'
}

try:
    response = requests.post(
        'https://backboard.railway.app/graphql/v2',
        headers=headers,
        json={'query': query}
    )
    
    if response.status_code == 200:
        data = response.json()
        if 'errors' in data:
            print(f'❌ API Fout: {json.dumps(data["errors"])}')
        else:
            print(f'✅ Succesvol verbonden met project: {data["data"]["project"]["name"]}')
            services = data['data']['project']['services']['edges']
            print(f'Services in dit project:')
            for service in services:
                print(f'  - {service["node"]["name"]}')
    else:
        print(f'❌ API Request Gefaald: Status {response.status_code}')
        print(response.text)
except Exception as e:
    print(f'❌ Exception: {str(e)}')
EOF

# Voer het test script uit
RAILWAY_TOKEN="$railway_token" RAILWAY_PROJECT_ID="$railway_project_id" python test_railway_api.py

if [ $? -ne 0 ]; then
    echo "Er was een probleem met je Railway API token of project ID. Probeer opnieuw."
    exit 1
fi

# Nu testen we de cursor bridge met het testbestand
echo "Testen van de cursor bridge met testbestand..."
RAILWAY_TOKEN="$railway_token" RAILWAY_PROJECT_ID="$railway_project_id" python railway-auto-fix/cursor_bridge_enhanced.py --test-logs railway-auto-fix/test_error.log

if [ $? -ne 0 ]; then
    echo "Er was een probleem met de cursor bridge. Controleer de logs."
    exit 1
fi

echo "De test is geslaagd! Nu instellen als service..."

# Maak een aangepaste plist file
cat > com.railway.cursor-bridge.plist << EOL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.railway.cursor-bridge</string>
    <key>ProgramArguments</key>
    <array>
        <string>$(pwd)/cursor_bridge_venv/bin/python</string>
        <string>$(pwd)/railway-auto-fix/cursor_bridge_enhanced.py</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>RAILWAY_TOKEN</key>
        <string>${railway_token}</string>
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