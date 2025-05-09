#!/usr/bin/env python3
"""
Railway Auto Fix Webhook Server

Deze server ontvangt notificaties van Railway deployments
en start het auto-fix proces wanneer een deployment faalt.
"""

import os
import json
import time
import logging
import threading
import traceback
import re
import tempfile
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import requests

# Laad environment variabelen
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("webhook_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialiseer Flask app
app = Flask(__name__)

# Configuratie
RAILWAY_TOKEN = os.environ.get('RAILWAY_TOKEN')
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
GITHUB_REPO = os.environ.get('GITHUB_REPO', 'jomarcello/Sigmapips-V2.4')
LOCAL_REPO_PATH = os.environ.get('LOCAL_REPO_PATH', '/tmp/railway-auto-fix-repo')
AUTH_TOKEN = os.environ.get('WEBHOOK_AUTH_TOKEN')  # Token voor webhook beveiliging
RAILWAY_PROJECT_ID = os.environ.get('RAILWAY_PROJECT_ID')  # Project ID voor Railway API calls

# Functions that were previously imported from cursor_bridge_enhanced
def fetch_deployment_logs(deployment_id):
    """Fetch logs for a specific deployment"""
    # Try direct REST API approach first (often more reliable)
    headers = {
        "Authorization": f"Bearer {RAILWAY_TOKEN}"
    }
    
    try:
        # First try the direct REST API endpoint
        logger.info(f"Attempting to fetch logs for deployment {deployment_id} using REST API")
        rest_response = requests.get(
            f"https://backboard.railway.app/api/deployments/{deployment_id}/logs",
            headers=headers,
            timeout=10
        )
        
        if rest_response.status_code == 200 and rest_response.text:
            logger.info(f"Successfully fetched logs using REST API endpoint: {len(rest_response.text)} characters")
            return rest_response.text
            
        logger.warning(f"REST API endpoint failed with status code: {rest_response.status_code}. Trying GraphQL...")
        
        # Fallback to GraphQL if REST API fails
        graphql_query = """
        query {
          deployment(id: "%s") {
            logs {
              content
            }
          }
        }
        """ % deployment_id
        
        graphql_headers = {
            "Authorization": f"Bearer {RAILWAY_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Try up to 3 times with exponential backoff
        max_retries = 3
        for i in range(max_retries):
            if i > 0:
                backoff_time = 2 ** i
                logger.info(f"Retrying GraphQL query after {backoff_time} seconds (attempt {i+1}/{max_retries})")
                time.sleep(backoff_time)
                
            graphql_response = requests.post(
                "https://backboard.railway.app/graphql/v2",
                headers=graphql_headers,
                json={"query": graphql_query},
                timeout=15
            )
            
            if graphql_response.status_code != 200:
                logger.error(f"Failed to fetch deployment logs (attempt {i+1}/{max_retries}): {graphql_response.text}")
                continue
            
            data = graphql_response.json()
            
            # Check for errors in the response
            if "errors" in data:
                logger.error(f"GraphQL errors (attempt {i+1}/{max_retries}): {json.dumps(data['errors'])}")
                continue
            
            # Extract logs from the response
            logs = data.get("data", {}).get("deployment", {}).get("logs", {}).get("content", "")
            
            if logs:
                logger.info(f"Successfully fetched {len(logs)} characters of logs via GraphQL")
                return logs
            else:
                logger.warning(f"No logs found in GraphQL response (attempt {i+1}/{max_retries})")
        
        # If we get here, both methods failed
        logger.error("All methods to fetch logs failed")
        
        # Try one last approach - fetch project logs instead of deployment logs
        try:
            logger.info("Attempting to fetch project-wide logs instead...")
            project_query = """
            query {
              project(id: "%s") {
                logs(last: 1000) {
                  nodes {
                    message
                    timestamp
                  }
                }
              }
            }
            """ % os.environ.get('RAILWAY_PROJECT_ID', '')
            
            if os.environ.get('RAILWAY_PROJECT_ID'):
                project_response = requests.post(
                    "https://backboard.railway.app/graphql/v2",
                    headers=graphql_headers,
                    json={"query": project_query},
                    timeout=15
                )
                
                if project_response.status_code == 200:
                    project_data = project_response.json()
                    if "data" in project_data and "project" in project_data["data"]:
                        log_nodes = project_data["data"]["project"]["logs"]["nodes"]
                        if log_nodes:
                            # Combine log messages with timestamps
                            combined_logs = "\n".join([
                                f"{node.get('timestamp', '')} - {node.get('message', '')}" 
                                for node in log_nodes
                            ])
                            logger.info(f"Fetched {len(combined_logs)} characters of project logs as fallback")
                            return combined_logs
            
        except Exception as project_e:
            logger.error(f"Failed to fetch project logs: {str(project_e)}")
        
        # Return a placeholder with enough information to be useful
        return f"""
        Could not retrieve detailed logs for deployment {deployment_id}.
        
        Common issues with Railway deployments:
        1. Missing dependencies in requirements.txt
        2. Environment variables not properly set
        3. Circular imports in Python code
        4. Syntax errors in code
        5. Resource limits exceeded
        
        Check your repository for these common issues.
        """
            
    except Exception as e:
        logger.error(f"Error fetching deployment logs: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Error retrieving logs: {str(e)}\n\nCheck your code for common issues like circular imports, missing dependencies, or syntax errors."

def extract_errors_from_logs(logs):
    """Extract error patterns from logs"""
    if not logs:
        return []
    
    error_patterns = [
        # Python import errors
        r"cannot import name ['\"]?(\w+)['\"]? from partially initialized module ['\"]?([\w\.]+)['\"]? \(most likely due to a circular import\)",
        r"ModuleNotFoundError: No module named ['\"]?([\w\._-]+)['\"]?",
        r"ImportError: (.+?)(?:\n|$)",
        
        # General Python errors
        r"TypeError: (.+?)(?:\n|$)",
        r"SyntaxError: (.+?)(?:\n|$)",
        r"NameError: name ['\"]?(\w+)['\"]? is not defined",
        r"AttributeError: ['\"]?(\w+)['\"]? object has no attribute ['\"]?(\w+)['\"]?",
        r"IndentationError: (.+?)(?:\n|$)",
        r"ValueError: (.+?)(?:\n|$)",
        r"KeyError: (.+?)(?:\n|$)",
        r"FileNotFoundError: (.+?)(?:\n|$)",
        r"PermissionError: (.+?)(?:\n|$)",
        
        # Web server errors
        r"Error: (.*?)(?:\n|$)",
        r"Exception: (.*?)(?:\n|$)",
        r"Failed to bind to .+?:(\d+)",
        r"Connection refused",
        
        # Railway specific errors
        r"Health check failed",
        r"Container exited with non-zero exit code",
        r"Resource limit exceeded",
        r"Build failed: (.+?)(?:\n|$)",
        
        # Docker/container errors
        r"standard_init_linux\.go:\d+: exec user process caused ['\"]?(.+?)['\"]?",
    ]
    
    errors = []
    for pattern in error_patterns:
        matches = re.finditer(pattern, logs, re.MULTILINE | re.IGNORECASE)
        errors.extend([match.group(0) for match in matches])
    
    # Special case for common circular imports in Sigmapips
    if "cannot import name 'ChartService'" in logs:
        errors.append("Circular import detected in chart_service module. This is a common issue in the trading_bot/services/chart_service directory.")
    
    # Remove duplicates while preserving order
    unique_errors = []
    seen = set()
    for error in errors:
        error_clean = error.strip()
        if error_clean and error_clean not in seen:
            seen.add(error_clean)
            unique_errors.append(error_clean)
    
    if unique_errors:
        logger.info(f"Found {len(unique_errors)} errors in logs")
        return unique_errors
    
    logger.info("No specific errors detected in logs")
    return []

def create_prompt_file(deployment_id, logs, errors):
    """Create a prompt file for Cursor"""
    temp_dir = tempfile.mkdtemp(prefix="railway-cursor-")
    prompt_file = os.path.join(temp_dir, "railway_deployment_error.md")
    
    with open(prompt_file, "w") as f:
        f.write(f"""# RAILWAY DEPLOYMENT ERROR - DIRECT ACTIE VEREIST

**BELANGRIJKE MELDING: Dit bestand is automatisch gegenereerd door de Railway-Cursor Bridge. Er is een deployment gefaald en directe actie is vereist.**

## Deployment Details
- Deployment ID: {deployment_id}
- Timestamp: {datetime.now().isoformat()}

## Gedetecteerde Fouten
```
{', '.join(errors) if errors else "Geen specifieke fouten gedetecteerd, zie logs voor details"}
```

## Volledige Railway Logs
```
{logs[:5000]}
{'...' if len(logs) > 5000 else ''}
```

## DIRECTE ACTIE VEREIST

Het probleem moet worden onderzocht. Mogelijke oorzaken:
- Ontbrekende dependencies in requirements.txt
- Configuratie problemen
- Code fouten

### Oplossingsrichting:

1. **Lokaliseer** het probleem in de logs
2. **Los het op** via de geschikte methode
3. **Commit en push** je wijzigingen naar GitHub
""")
    
    logger.info(f"Created prompt file at {prompt_file}")
    return prompt_file

def start_cursor_fix(deployment_id=None, deployment_logs=None):
    """
    Start Cursor met het auto fix proces
    """
    try:
        logger.info(f"Starting Cursor fix process for deployment: {deployment_id}")
        
        # 1. Haal deployment logs op als we ze nog niet hebben
        if not deployment_logs and deployment_id:
            deployment_logs = fetch_deployment_logs(deployment_id)
        
        if not deployment_logs:
            logger.error("No deployment logs available, cannot proceed")
            return False
        
        # 2. Extraheer errors uit de logs
        errors = extract_errors_from_logs(deployment_logs)
        
        # 3. Maak een prompt file
        prompt_file = create_prompt_file(deployment_id, deployment_logs, errors)
        
        # 4. Webhook server kan zelf geen Cursor starten, maar we genereren wel het bestand
        # en loggen het pad voor handmatige opvolging
        logger.info(f"Created error analysis at {prompt_file}")
        logger.info(f"Server cannot directly open Cursor. Please manually open this file.")
        
        # 5. Voeg errors toe aan log bestand
        error_log_path = os.path.join(os.path.dirname(prompt_file), "deployment_errors.log")
        with open(error_log_path, "w") as f:
            f.write(f"Deployment ID: {deployment_id}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
            f.write("Detected Errors:\n")
            for error in errors:
                f.write(f"- {error}\n")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in Cursor fix process: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint voor Railway"""
    try:
        # Controleer of we toegang hebben tot de Railway API
        if RAILWAY_TOKEN:
            # Maak een simpele API call om te controleren of de token werkt
            headers = {
                "Authorization": f"Bearer {RAILWAY_TOKEN}",
                "Content-Type": "application/json"
            }
            
            # GraphQL query voor een simpele check
            query = """
            query {
              me {
                id
              }
            }
            """
            
            response = requests.post(
                "https://backboard.railway.app/graphql/v2",
                headers=headers,
                json={"query": query},
                timeout=5
            )
            
            # Als de API call succesvol is, is de service gezond
            if response.status_code == 200:
                app.logger.info("Health check passed - API connection successful")
                return jsonify({
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "message": "Webhook service is running correctly and can connect to Railway API"
                })
        
        # Basic health check - als geen token of API check niet succesvol
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "message": "Webhook service is running"
        })
    except Exception as e:
        app.logger.warning(f"Health check warning: {str(e)}")
        # Nog steeds healthy terugsturen maar met waarschuwing
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "message": f"Webhook service is running with warning: {str(e)}"
        })

@app.route('/webhook', methods=['POST'])
def railway_webhook():
    """
    Webhook endpoint for Railway deployment notifications
    
    Verwacht JSON in het formaat:
    {
        "event": "deployment.failed", // of andere events
        "deployment": {
            "id": "deployment-id",
            "status": "FAILED"
        },
        "service": {
            "id": "service-id",
            "name": "service-name"
        }
    }
    """
    # Valideer auth token als deze is geconfigureerd
    if AUTH_TOKEN:
        auth_header = request.headers.get('Authorization')
        if not auth_header or auth_header != f"Bearer {AUTH_TOKEN}":
            logger.warning("Unauthorized webhook request received")
            return jsonify({"error": "Unauthorized"}), 401
    
    # Valideer request data
    if not request.json:
        logger.warning("Invalid webhook payload - not JSON")
        return jsonify({"error": "Invalid JSON payload"}), 400
    
    logger.info(f"Received webhook: {json.dumps(request.json)[:200]}...")
    
    try:
        data = request.json
        event_type = data.get('event')
        
        # Controleer of dit een deployment failure event is
        if event_type == 'deployment.failed':
            deployment_id = data.get('deployment', {}).get('id')
            service_name = data.get('service', {}).get('name')
            
            logger.info(f"Detected failed deployment: {deployment_id} for service: {service_name}")
            
            # Start het Cursor fix proces in een nieuwe thread
            threading.Thread(
                target=start_cursor_fix,
                args=(deployment_id, None),
                daemon=True
            ).start()
            
            return jsonify({
                "status": "processing",
                "message": f"Processing started for deployment {deployment_id}"
            })
        else:
            logger.info(f"Ignoring non-failure event: {event_type}")
            return jsonify({
                "status": "ignored", 
                "message": "Event is not a deployment failure"
            })
            
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/manual-fix', methods=['POST'])
def manual_fix():
    """
    Handmatig starten van het fix proces voor een specifieke deployment
    
    Verwacht JSON in het formaat:
    {
        "deployment_id": "deployment-id"
    }
    
    Of met logs:
    {
        "logs": "deployment logs content"
    }
    """
    # Valideer auth token
    if AUTH_TOKEN:
        auth_header = request.headers.get('Authorization')
        if not auth_header or auth_header != f"Bearer {AUTH_TOKEN}":
            return jsonify({"error": "Unauthorized"}), 401
    
    # Valideer request data
    if not request.json:
        return jsonify({"error": "Invalid JSON payload"}), 400
    
    deployment_id = request.json.get('deployment_id')
    logs = request.json.get('logs')
    
    if not deployment_id and not logs:
        return jsonify({"error": "Either deployment_id or logs must be provided"}), 400
    
    # Start het Cursor fix proces in een nieuwe thread
    threading.Thread(
        target=start_cursor_fix,
        args=(deployment_id, logs),
        daemon=True
    ).start()
    
    return jsonify({
        "status": "processing",
        "message": f"Processing started for {'deployment ' + deployment_id if deployment_id else 'provided logs'}"
    })

if __name__ == "__main__":
    # Controleer of de vereiste environment variabelen aanwezig zijn
    if not RAILWAY_TOKEN:
        logger.error("Missing required RAILWAY_TOKEN environment variable")
        exit(1)
    
    # Start de webhook server
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting Railway webhook server on port {port}")
    app.run(host='0.0.0.0', port=port) 