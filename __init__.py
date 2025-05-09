"""
Root package for the SigmaPips trading bot.
This makes the 'trading_bot' module importable.
"""

import sys
import os

# Add the parent directory to sys.path to make 'trading_bot' importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
