#!/bin/bash
echo "Starting SigmaPips Trading Bot..."

# Set environment variables from .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    export $(grep -v '^#' .env | xargs)
fi

# Check if TELEGRAM_BOT_TOKEN is set and valid
if [ -z "$TELEGRAM_BOT_TOKEN" ] || [ "$TELEGRAM_BOT_TOKEN" = "YOUR_TELEGRAM_BOT_TOKEN" ] || [ "$TELEGRAM_BOT_TOKEN" = "your_actual_telegram_bot_token_here" ]; then
    echo "ERROR: Valid TELEGRAM_BOT_TOKEN not found in environment."
    echo "Please set a valid Telegram bot token in the .env file or environment variables."
    echo "Example: TELEGRAM_BOT_TOKEN=1234567890:ABCDefGhIJklmnOPQrsTUVwxyZ"
    exit 1
fi

# Set the Python path to include the current directory
export PYTHONPATH=$(pwd)

echo "Python path set to: $PYTHONPATH"
echo "Current directory: $(pwd)"

# Install any missing packages
echo "Installing dependencies..."
pip install --no-cache-dir twelvedata>=1.2.10 -r requirements.txt

# Make sure we're in the right directory
cd "$(dirname "$0")"

# Stop any existing bot instances first
echo "Stopping any existing bot instances..."
python stop_existing_bots.py

# Sleep to allow Telegram API sessions to clear
echo "Waiting for Telegram API sessions to clear..."
sleep 5

echo "Starting the bot..."
# Use the new start_bot.py script which handles module path configuration
python start_bot.py
