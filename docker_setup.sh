#!/bin/bash
# Docker setup script voor SigmaPips Trading Bot

# Log bericht voor debugging
echo "Starting Docker setup for SigmaPips Trading Bot..."

# Installeer benodigde systeem packages 
echo "Installing system packages..."
apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    libssl-dev \
    libffi-dev \
    libpq-dev

# Installeer yfinance expliciet
echo "Installing yfinance package explicitly..."
pip install yfinance==0.2.57

# Installeer alle dependencies 
echo "Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

# Installeer Playwright voor Python
echo "Installing Playwright for Python..."
pip install playwright
playwright install chromium

# Installeer Playwright voor Node.js
echo "Installing Playwright for Node.js..."
npm install playwright@latest
npx playwright install chromium

echo "Docker setup completed."
echo "You can now start the application." 
