#!/usr/bin/env python3
"""
Zeer eenvoudige webhook server die alle verzoeken accepteert
"""

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

@app.route('/webhook', methods=['POST', 'GET'])
def webhook():
    # Log het verzoek
    print("Webhook request received!")
    print(f"Headers: {request.headers}")
    if request.is_json:
        print(f"JSON data: {request.json}")
    
    # Accepteer elk verzoek
    return jsonify({"status": "success", "message": "Webhook received"})

@app.route('/', methods=['GET', 'POST'])
def root():
    return jsonify({"status": "healthy", "message": "Webhook server is running"})

if __name__ == "__main__":
    print("Starting simple webhook server on port 8080")
    app.run(host='0.0.0.0', port=8080) 