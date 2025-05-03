#!/usr/bin/env python3

"""
Script to fix indentation of _generate_default_analysis in chart.py
"""

import os
import re

def main():
    # Backup original file if not already done
    if not os.path.exists("trading_bot/services/chart_service/chart.py.bak"):
        os.system("cp trading_bot/services/chart_service/chart.py trading_bot/services/chart_service/chart.py.bak")
    
    # Read current file content
    with open("trading_bot/services/chart_service/chart.py", "r") as f:
        content = f.read()
    
    # Use regex to find and fix the indentation of _generate_default_analysis
    pattern = re.compile(r'(\s+return await self\._generate_default_analysis\(instrument, timeframe\)\s*\n+)(\s{4}async def _generate_default_analysis\()')
    fixed_content = pattern.sub(r'\1\n    async def _generate_default_analysis(', content)
    
    # Write the fixed content to a temporary file
    with open("trading_bot/services/chart_service/chart.py.fixed", "w") as f:
        f.write(fixed_content)
    
    # Replace the original file with the fixed one
    os.system("mv trading_bot/services/chart_service/chart.py.fixed trading_bot/services/chart_service/chart.py")
    
    print("Fixed indentation of _generate_default_analysis in chart.py")

if __name__ == "__main__":
    main() 