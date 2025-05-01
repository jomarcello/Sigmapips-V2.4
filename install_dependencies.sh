#!/bin/bash
# Script om alle dependencies te installeren, inclusief yfinance

echo "Installing SigmaPips dependencies..."

# Installeer pip als het niet geïnstalleerd is
if ! command -v pip &> /dev/null; then
    echo "pip niet gevonden. pip wordt geïnstalleerd..."
    python -m ensurepip --upgrade
fi

# Installeer yfinance expliciet eerst
echo "Installing yfinance explicitly..."
pip install yfinance==0.2.57

# Installeer cachetools expliciet (nodig voor YahooFinanceProvider)
echo "Installing cachetools explicitly..."
pip install cachetools>=5.5.0

# Installeer overige dependencies
echo "Installing other dependencies..."
pip install -r requirements.txt

# Controleer yfinance versie
echo "Checking yfinance version..."
pip show yfinance

echo "Done installing dependencies!" 
