#!/usr/bin/env python3
"""
Direct script to run the SigmaPips Trading Bot
"""

import os
import sys

# Set the Python path to include the project root
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import run_main_bot from the trading_bot module
try:
    from trading_bot.main import run_main_bot
    import asyncio
    
    print(f"✅ Successfully imported run_main_bot from trading_bot.main")
    print(f"Python path: {sys.path}")
    
    # Run the bot
    if __name__ == "__main__":
        print("Starting the bot...")
        asyncio.run(run_main_bot())
except ImportError as e:
    print(f"❌ Error importing from trading_bot.main: {e}")
    print(f"Current Python path: {sys.path}")
    sys.exit(1) 