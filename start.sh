#!/bin/bash
# Robust startup script for the bot with multiple fallbacks

echo "==== STARTING SIGMAPIPS TRADING BOT ===="
echo "Current directory: $(pwd)"
echo "Files in current directory:"
ls -la

# Set Python path more robustly to include both app root and trading_bot
export PYTHONPATH="$(pwd):$(pwd)/trading_bot:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"

# Make sure Python can find the modules
cd $(pwd)

# Add debug info to help diagnose import issues
cat > debug_imports.py << EOF
import sys
import os

print("Python path:")
for p in sys.path:
    print(f"  - {p}")

print("\nChecking critical paths:")
paths_to_check = [
    "trading_bot/__init__.py",
    "trading_bot/services/__init__.py",
    "trading_bot/services/telegram_service/__init__.py",
    "trading_bot/services/telegram_service/states.py"
]
for path in paths_to_check:
    exists = os.path.exists(path)
    print(f"  - {path}: {'EXISTS' if exists else 'MISSING'}")

print("\nTrying imports:")
try:
    import trading_bot
    print("  ✓ import trading_bot")
except ImportError as e:
    print(f"  ✗ import trading_bot: {e}")

try:
    import trading_bot.services
    print("  ✓ import trading_bot.services")
except ImportError as e:
    print(f"  ✗ import trading_bot.services: {e}")

try:
    import trading_bot.services.telegram_service
    print("  ✓ import trading_bot.services.telegram_service")
except ImportError as e:
    print(f"  ✗ import trading_bot.services.telegram_service: {e}")

try:
    from trading_bot.services.telegram_service import states
    print("  ✓ from trading_bot.services.telegram_service import states")
except ImportError as e:
    print(f"  ✗ from trading_bot.services.telegram_service import states: {e}")
EOF

echo "Running import diagnostics..."
python debug_imports.py

# Try multiple startup methods
echo "Attempting to start bot using start_bot.py..."
if [ -f "start_bot.py" ]; then
    python start_bot.py
    RESULT=$?
    if [ $RESULT -eq 0 ]; then
        echo "Bot started successfully with start_bot.py"
        exit 0
    else
        echo "Failed to start with start_bot.py (exit code: $RESULT), trying alternative methods..."
    fi
else
    echo "start_bot.py not found, trying alternative methods..."
fi

echo "Attempting to start bot from trading_bot.main module..."
if [ -f "trading_bot/main.py" ]; then
    python -m trading_bot.main
    RESULT=$?
    if [ $RESULT -eq 0 ]; then
        echo "Bot started successfully with trading_bot.main"
        exit 0
    else
        echo "Failed to start with trading_bot.main (exit code: $RESULT), trying other methods..."
    fi
else
    echo "trading_bot/main.py not found, trying other methods..."
fi

echo "Attempting to start bot with direct Python execution..."
# Set Python path again inside trading_bot directory
export PYTHONPATH="$(pwd):$PYTHONPATH"
cd trading_bot
if [ -f "main.py" ]; then
    python main.py
    RESULT=$?
    if [ $RESULT -eq 0 ]; then
        echo "Bot started successfully with direct execution"
        exit 0
    else
        echo "Failed to start with direct execution (exit code: $RESULT)"
    fi
else
    echo "main.py not found after changing directory to trading_bot/"
fi

echo "==== STARTUP FAILED ===="
echo "All startup methods failed. Please check your installation."
exit 1
