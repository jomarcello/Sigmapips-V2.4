#!/usr/bin/env python3

"""
Script to fix indentation issues in chart.py
"""

import os

def main():
    # Backup original file
    os.system("cp trading_bot/services/chart_service/chart.py trading_bot/services/chart_service/chart.py.bak")
    
    # Read current file content
    with open("trading_bot/services/chart_service/chart.py", "r") as f:
        content = f.read()
    
    # Fix the second try block that needs an except clause
    fixed_content = content.replace(
        "        try:\n            # Get the available data providers\n            binance_provider = None\n            yahoo_provider = None\n            alltick_provider = None",
        """        try:
            # Get the available data providers
            binance_provider = None
            yahoo_provider = None
            alltick_provider = None
        except Exception as e:
            logger.error(f"Error in provider setup for technical analysis: {str(e)}")
            return await self._generate_default_analysis(instrument, timeframe)
            
        try"""
    )
    
    # Write the fixed content to a temporary file
    with open("trading_bot/services/chart_service/chart.py.fixed", "w") as f:
        f.write(fixed_content)
    
    # Replace the original file with the fixed one
    os.system("mv trading_bot/services/chart_service/chart.py.fixed trading_bot/services/chart_service/chart.py")
    
    print("Fixed indentation and except clause issues in chart.py")

if __name__ == "__main__":
    main() 