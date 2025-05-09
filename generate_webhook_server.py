#!/usr/bin/env python3

content = '''#!/usr/bin/env python3
"""
Railway Auto Fix Webhook Server

Deze server ontvangt notificaties van Railway deployments
en registreert deze voor verdere verwerking.
"""

import os
import json
import time
import logging
import threading
import traceback
from dotenv import load_dotenv
from flask import Flask, request, jsonify

# Laad environment variabelen
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialiseer Flask app
app = Flask(__name__)

# Configuratie
AUTH_TOKEN = os.environ.get('WEBHOOK_AUTH_TOKEN')  # Token voor webhook beveiliging
PORT = int(os.environ.get('PORT', 8080))

# Lijst om deployment notificaties op te slaan
deployment_notifications = []

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": time.time()})

@app.route('/notifications', methods=['GET'])
def get_notifications():
    """Endpoint om opgeslagen notificaties te bekijken"""
    # Valideer auth token als deze is geconfigureerd
    if AUTH_TOKEN:
        auth_header = request.headers.get('Authorization')
        if not auth_header or auth_header != f"Bearer {AUTH_TOKEN}":
            logger.warning("Unauthorized notifications request received")
            return jsonify({"error": "Unauthorized"}), 401
    
    return jsonify({
        "notifications": deployment_notifications,
        "count": len(deployment_notifications)
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
    
    # Log de webhook data
    webhook_data = request.json
    logger.info(f"Received webhook: {json.dumps(webhook_data)[:200]}...")
    
    try:
        event_type = webhook_data.get('event')
        
        # Sla de notificatie op voor later gebruik
        deployment_notifications.append({
            "timestamp": time.time(),
            "data": webhook_data
        })
        
        # Houd de lijst beperkt tot de laatste 50 notificaties
        while len(deployment_notifications) > 50:
            deployment_notifications.pop(0)
        
        # Detecteer gefaalde deployments
        if event_type == 'deployment.failed':
            deployment_id = webhook_data.get('deployment', {}).get('id')
            service_name = webhook_data.get('service', {}).get('name')
            
            logger.info(f"Detected failed deployment: {deployment_id} for service: {service_name}")
            
            return jsonify({
                "status": "received",
                "message": f"Notification received for failed deployment {deployment_id}"
            })
        else:
            logger.info(f"Received non-failure event: {event_type}")
            return jsonify({
                "status": "received", 
                "message": f"Notification received for event {event_type}"
            })
            
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/clear-notifications', methods=['POST'])
def clear_notifications():
    """Endpoint om alle opgeslagen notificaties te wissen"""
    # Valideer auth token als deze is geconfigureerd
    if AUTH_TOKEN:
        auth_header = request.headers.get('Authorization')
        if not auth_header or auth_header != f"Bearer {AUTH_TOKEN}":
            logger.warning("Unauthorized clear request received")
            return jsonify({"error": "Unauthorized"}), 401
    
    global deployment_notifications
    old_count = len(deployment_notifications)
    deployment_notifications = []
    
    return jsonify({
        "status": "success",
        "message": f"Cleared {old_count} notifications"
    })

if __name__ == "__main__":
    # Controleer of de webhook server kan worden gestart
    logger.info(f"Starting Railway Webhook server on port {PORT}")
    app.run(host='0.0.0.0', port=PORT)
'''

with open('webhook_server.py', 'w') as f:
    f.write(content)

print("webhook_server.py bestand is gegenereerd zonder de cursor_bridge_enhanced import!") 