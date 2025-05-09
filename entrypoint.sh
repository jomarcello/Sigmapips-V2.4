#!/bin/bash
set -e

echo "Starting SigmaPips Trading Bot..."

# Load environment variables from config if present
if [ -f /app/config/.env ]; then
    echo "Loading environment variables from config/.env"
    export $(cat /app/config/.env | grep -v '^#' | xargs)
fi

# Load environment variables from root .env if present
if [ -f /app/.env ]; then
    echo "Loading environment variables from .env"
    export $(cat /app/.env | grep -v '^#' | xargs)
fi

# Set Python path to include the app directory
# Make sure the current directory is in the Python path
echo "Setting PYTHONPATH to include current directory"
export PYTHONPATH="$PYTHONPATH:$(pwd):/app"
echo "PYTHONPATH=$PYTHONPATH"

# Start the application
echo "Starting the application..."
exec "$@" 
